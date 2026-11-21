package com.viitorvoice.demo

import android.app.Application
import android.content.Context
import com.viitorvoice.demo.text.Prompt
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Single entry point for everything that touches the NPU.
 *
 * Model layout under filesDir/models/ :
 *
 *   dec_p1.bin        84 MB   codes -> conv  (fp16 graph)
 *   dec_p2.bin        19 MB
 *   dec_p3.bin         6 MB
 *   dec_p4.bin        16 MB
 *   dec_p5.bin         6 MB   time-chunked, halo 48
 *
 * The remaining pipeline graphs (w2v / encoder / llm / head) are not wired in
 * yet; the decoder half is the part that is finished and verified.
 */
object Engine {

    /** The on-device reference encoder has a fixed 8.04 s window (200 frames). */
    const val ENC_MAX_SAMPLES = 192900

    private lateinit var app: Application
    private var inited = false

    fun attach(application: Application) {
        app = application
    }

    val modelsDir: File get() = File(app.filesDir, "models")

    /** Every blob the pipelines open by name — the shared library can serve them all. */
    private val allModelNames: List<String>
        get() = (decoderFiles + narFiles + encoderFiles + "nar_steps").distinct()

    private var storeTried = false
    private var storeBound = false

    /** Human-readable outcome of the shared-library attempt ("" while untried). */
    var storeMsg: String = ""
        private set

    private val decoderFiles = listOf(
        "dec_p1.bin", "dec_p2.bin", "dec_p3.bin", "dec_p4.bin", "dec_p5.bin",
    )

    /** Everything the NAR loop needs on top of the decoder. */
    private val narFiles = listOf(
        "llm.bin", "head.bin", "text_emb.bin", "sem_emb.bin", "aco_emb.bin",
    )

    private var narInited = false

    /** Last nativeNarInit reply ("OK <ms> [llm 2/multi …]"). */
    var narInitMsg: String = ""
        private set
    private var tokenizer: com.viitorvoice.demo.text.VvTokenizer? = null

    fun tokenizer(context: Context): com.viitorvoice.demo.text.VvTokenizer {
        tokenizer?.let { return it }
        val json = context.assets.open("tokenizer.json").use {
            String(it.readBytes(), Charsets.UTF_8)
        }
        return com.viitorvoice.demo.text.VvTokenizer(json).also { tokenizer = it }
    }

    fun missingNarFiles(): List<String> =
        if (storeBound) emptyList()
        else (decoderFiles + narFiles).filter { !File(modelsDir, it).isFile }

    /**
     * Point every model path at the shared library (MLP/1) when it has the set.
     * Runs once per process; on failure the app keeps using filesDir/models.
     */
    fun ensureStore(): Boolean {
        if (storeTried) return storeBound
        storeTried = true
        if (!ModelStore.available(app)) {
            storeMsg = "未安装共享模型库"
            return false
        }
        if (!ModelStore.has(app, ModelStore.NAR_SET, allModelNames)) {
            storeMsg = "共享库缺模型"
            return false
        }
        val err = ModelStore.bind(app, ModelStore.NAR_SET, ModelStore.NAR_ARCH, allModelNames)
        storeBound = err.isEmpty()
        storeMsg = if (storeBound) "共享库 · ${ModelStore.summary(app)}" else err
        return storeBound
    }

    /** True once model paths resolve through the shared library. */
    val usingStore: Boolean get() = storeBound

    /**
     * Producer role: publish this app's private copies into the library so they
     * can be deleted afterwards. Content-addressed, so re-running is free.
     */
    fun publishLocalModels(): String {
        val files = installedModels().filter { it.name in allModelNames }
        return ModelStore.publish(app, ModelStore.NAR_SET, ModelStore.NAR_ARCH, files)
    }

    /** Largest prompt window the installed llm/head .bin carries (256 or 512). */
    fun maxWindow(): Int =
        if (narInited) Native.nativeNarMaxWindow() else Prompt.SEQ_SHORT

    /** Pick the window for this job (256 if it fits, else 512) and apply it. */
    fun useWindow(cond: Int, refFrames: Int, frames: Int): String {
        val sep = if (refFrames > 0) 1 else 0
        val need = cond + refFrames + sep + frames
        val max = maxWindow()
        if (need > max) return "提示 ${cond} + 参考 ${refFrames} + 目标 ${frames} = ${need} token，超过窗口 $max"
        Native.nativeNarSetWindow(if (need <= Prompt.SEQ_SHORT) Prompt.SEQ_SHORT else Prompt.SEQ_LONG)
        return ""
    }

    /** Loads the QNN backend, decoder contexts, llm + head and embed tables. */
    fun ensureNar(): String {
        val err = ensureInit()
        if (err.isNotEmpty()) return err
        if (narInited) return ""
        val missing = if (storeBound) emptyList() else narFiles.filter { !File(modelsDir, it).isFile }
        if (missing.isNotEmpty()) return "缺少模型: ${missing.joinToString()}"
        val r = Native.nativeNarInit(modelsDir.absolutePath)
        narInitMsg = r
        if (!r.startsWith("OK")) return r
        narInited = true
        return ""
    }

    /**
     * Full text -> speech. Runs the 32-step diffusion loop on the NPU and then
     * the decoder chain. [onStep] gets (step, filledCells, remainingCells).
     * Returns "OK <nar_ms>|<dec_ms>" or an error string.
     */
    fun synthesize(
        context: Context,
        text: String,
        out: File,
        onStep: (Int, Int, Int) -> Unit = { _, _, _ -> },
    ): String {
        val err = ensureNar()
        if (err.isNotEmpty()) return err
        val cond = com.viitorvoice.demo.text.Prompt.build(tokenizer(context), text)
        if (cond.isEmpty()) return "请输入要合成的文本"
        val frames = com.viitorvoice.demo.text.Prompt.framesFor(cond.size, 0, maxWindow())
        if (frames < 25) return "文本太长（${cond.size} tokens），请缩短到约 40 字以内"
        useWindow(cond.size, 0, frames).let { if (it.isNotEmpty()) return it }
        var r = Native.nativeNarPrepare(cond, cond.size, frames)
        if (!r.startsWith("OK")) return r
        val t0 = System.currentTimeMillis()
        val steps = Native.nativeNarSteps().let { if (it in 1..STEPS) it else STEPS }
        for (step in 0 until steps) {
            val k = Native.nativeNarStep(step)
            if (k < 0) return Native.nativeNarError()
            onStep(step + 1, k, Native.nativeNarRemaining())
        }
        val narMs = System.currentTimeMillis() - t0
        logT("nar ${narMs} ms | " + Native.nativeNarTiming())
        logT("narInit $narInitMsg")
        r = Native.nativeNarFinish(out.absolutePath)
        if (!r.startsWith("OK")) return r
        val parts = r.removePrefix("OK ").split("|")
        return "OK $narMs|${parts.getOrNull(0) ?: "?"}"
    }

    const val STEPS = 32

    // ------------------------------------------------------------ encoder
    private var encoderInited = false

    /**
     * Encoder assets actually loaded on device: the QNN contexts. The old ORT
     * fallback (w2v_front8s.onnx + dualcodec_encode_core_*.onnx, 2.6 GB) is no
     * longer shipped; vv_encode.cpp only opens those if a QNN context fails.
     */
    private val encoderFiles = listOf(
        "enc_fe.bin", "fe_mel.bin",
        "enc_e1.bin", "enc_e2.bin", "enc_e3.bin", "enc_front.bin",
        "enc_e4b.bin", "enc_e5.bin", "enc_e6.bin",
    )

    fun missingEncoderFiles(): List<String> =
        if (storeBound) emptyList()
        else encoderFiles.filter { !File(modelsDir, it).isFile }

    /** Timing lines go to files/run_log.txt; logcat is unreadable here. */
    private fun logT(s: String) {
        runCatching {
            java.io.File(modelsDir.parentFile, "run_log.txt").appendText("T $s\n")
        }
    }

    fun ensureEncoder(): String {
        if (encoderInited) return ""
        val missing = missingEncoderFiles()
        if (missing.isNotEmpty()) return "缺少模型: ${missing.joinToString()}"
        val te = System.currentTimeMillis()
        val r = Native.nativeEncodeInit(modelsDir.absolutePath)
        logT("encodeInit ${System.currentTimeMillis() - te} ms")
        if (!r.startsWith("OK")) return r
        encoderInited = true
        return ""
    }

    fun releaseEncoder() {
        if (!encoderInited) return
        Native.nativeEncodeRelease()
        encoderInited = false
    }

    /** 24 kHz mono pcm -> (tokens 12*200, realFrames). */
    fun encode(pcm: FloatArray): Pair<IntArray, Int> {
        val err = ensureEncoder()
        if (err.isNotEmpty()) throw IllegalStateException(err)
        val out = IntArray(12 * 200)
        val tw = System.currentTimeMillis()
        val n = Native.nativeEncode(pcm, pcm.size, out)
        logT("encodeCore ${System.currentTimeMillis() - tw} ms")
        if (n <= 0) throw IllegalStateException(Native.nativeEncodeError())
        // nativeEncode fills a 200-frame stride; repack to n so the token matrix
        // is 12 x n (row-major) for the edit API.
        val packed = IntArray(12 * n)
        for (c in 0 until 12) System.arraycopy(out, c * 200, packed, c * n, n)
        return packed to n
    }

    /**
     * Keep the first [keep] frames of a packed 12 x [frames] token matrix.
     * Clone prompts share the 256-token window with the target text, so a long
     * reference has to be cut back before the job is prepared.
     */
    fun trimRef(tokens: IntArray, frames: Int, keep: Int): IntArray {
        if (keep >= frames) return tokens
        val out = IntArray(12 * keep)
        for (c in 0 until 12) System.arraycopy(tokens, c * frames, out, c * keep, keep)
        return out
    }

    /**
     * Local edit: regenerate [maskStart, maskEnd) of [tokens] so that the clip
     * says [newText] (the full new sentence, not just the changed words).
     * Results land in the native edit buffer; call [finishToWav].
     */
    fun editPrepare(
        context: Context,
        tokens: IntArray,
        frames: Int,
        maskStart: Int,
        maskEnd: Int,
        newText: String,
        cfg: Float = 2.0f,
        lang: String = "zh",
        emotionScale: Float = 0f,
        nvvScale: Float = 0f,
    ): String {
        val err = ensureNar()
        if (err.isNotEmpty()) return err
        val tok = tokenizer(context)
        val cond = Prompt.build(tok, newText, lang)
        if (cond.isEmpty()) return "请输入修改后的整句文本"
        useWindow(cond.size, 0, frames).let { if (it.isNotEmpty()) return it }
        val aux = auxBranches(newText, emotionScale, nvvScale) { t ->
            Prompt.build(tok, t, lang)
        }
        val editable = BooleanArray(frames) { it >= maskStart && it < maskEnd }
        val win = tokens.copyOf(12 * frames)
        for (c in 0 until 12) {
            for (i in maskStart until maskEnd) win[c * frames + i] = if (c == 0) 16386 else 1026
        }
        return prepareCfg(cond, aux, null, 0, win, editable, frames, cfg)
    }

    /**
     * Voice cloning: [refTokens] is 12*refFrames of the reference clip,
     * [refText] its transcript (null → no-ref-text branch). Results land in
     * the native cfg buffer; call [finishToWav].
     */
    fun clonePrepare(
        context: Context,
        text: String,
        refText: String?,
        refTokens: IntArray,
        refFrames: Int,
        frames: Int,
        lang: String,
        cfg: Float,
        emotionScale: Float,
        nvvScale: Float,
    ): String {
        val err = ensureNar()
        if (err.isNotEmpty()) return err
        val tok = tokenizer(context)
        val cond = Prompt.buildClone(tok, text, refText, refFrames, lang)
        if (cond.isEmpty()) return "请输入要合成的文本"
        useWindow(cond.size, refFrames, frames).let { if (it.isNotEmpty()) return it }
        val aux = auxBranches(text, emotionScale, nvvScale) { t ->
            Prompt.buildClone(tok, t, refText, refFrames, lang)
        }
        return prepareCfg(cond, aux, refTokens, refFrames, null, null, frames, cfg)
    }

    /** Tag-stripped CFG branches (no_emotion / no_nvv), like generation.py. */
    private fun auxBranches(
        text: String,
        emotionScale: Float,
        nvvScale: Float,
        build: (String) -> IntArray,
    ): List<Pair<IntArray, Float>> {
        val out = ArrayList<Pair<IntArray, Float>>(2)
        if (emotionScale > 0f && com.viitorvoice.demo.text.Tags.hasEmotion(text)) {
            val ids = build(com.viitorvoice.demo.text.Tags.stripEmotion(text))
            if (ids.isNotEmpty()) out += ids to emotionScale
        }
        if (nvvScale > 0f && com.viitorvoice.demo.text.Tags.hasNvv(text)) {
            val ids = build(com.viitorvoice.demo.text.Tags.stripNvv(text))
            if (ids.isNotEmpty()) out += ids to nvvScale
        }
        return out
    }

    fun prepareCfg(
        cond: IntArray,
        aux: List<Pair<IntArray, Float>>,
        refTokens: IntArray?,
        refFrames: Int,
        target: IntArray?,
        editable: BooleanArray?,
        frames: Int,
        cfg: Float,
    ): String {
        val err = ensureNar()
        if (err.isNotEmpty()) return err
        useWindow(cond.size, refFrames, frames).let { if (it.isNotEmpty()) return it }
        val ids = aux.map { it.first }.toTypedArray()
        val scales = FloatArray(aux.size) { aux[it].second }
        return Native.nativeNarPrepareCfg(
            cond, cond.size, ids, scales, refTokens, refFrames,
            target, editable, frames, cfg,
        )
    }

    fun resultTokens(): IntArray? = Native.nativeNarEditTokens()

    /** Debug: dump the step-0 branch logits, or null to stop. */
    fun setDiagPath(path: String?) = Native.nativeNarSetDiagPath(path)

    /** Debug: override the guidance weights (cond, uncond). */
    fun setWeights(wc: Float, wu: Float) = Native.nativeNarSetWeights(wc, wu)

    fun editStep(step: Int): Int = Native.nativeNarEditStep(step)

    /** Debug: where the 32 steps went (gather / llm / head / score / …). */
    fun narTiming(): String = Native.nativeNarTiming()

    fun editRemaining(): Int = Native.nativeNarEditRemaining()

    fun finishToWav(out: File): String = Native.nativeNarFinish(out.absolutePath)

    fun installedModels(): List<File> {
        val dir = modelsDir
        if (!dir.isDirectory) return emptyList()
        return dir.listFiles { f -> f.isFile && f.name.endsWith(".bin") }
            ?.sortedByDescending { it.length() }
            .orEmpty()
    }

    fun missingDecoderFiles(): List<String> =
        if (storeBound) emptyList()
        else decoderFiles.filter { !File(modelsDir, it).isFile }

    fun statusLine(): String {
        if (storeBound) return "共享模型库 · ${ModelStore.summary(app)}"
        val missing = missingNarFiles()
        if (missing.isNotEmpty()) return "缺少 ${missing.size} 个 NAR 模型"
        val enc = missingEncoderFiles()
        val total = installedModels().sumOf { it.length() } / (1024 * 1024)
        return if (enc.isEmpty()) "就绪 · ${installedModels().size} 个模型 · $total MB"
        else "NAR 就绪 · 缺编码器 ${enc.first()}"
    }

    /** Loads the QNN backend and the five decoder contexts. Returns "" on success. */
    fun ensureInit(): String {
        if (inited) return ""
        ensureStore()
        val missing = missingDecoderFiles()
        if (missing.isNotEmpty()) return "缺少模型: ${missing.joinToString()}"
        val ctx = app
        val libDir = ctx.applicationInfo.nativeLibraryDir
        val r = Native.nativeInit(libDir, libDir, modelsDir.absolutePath)
        if (!r.startsWith("OK")) return r
        inited = true
        return ""
    }

    /** Runs the bundled example codes (8 s of speech). Returns timings or error. */
    fun decodeSample(context: Context, outFile: File): String {
        val err = ensureInit()
        if (err.isNotEmpty()) return err
        val sem = readInt32Asset(context, "sample_sem.bin")
        val aco = readInt32Asset(context, "sample_aco.bin")
        if (sem.size != 200 || aco.size != 2200)
            return "示例码本损坏 (${sem.size}/${aco.size})"
        return Native.nativeDecodeToWav(sem, aco, outFile.absolutePath)
    }

    private fun readInt32Asset(context: Context, name: String): IntArray {
        val bytes = context.assets.open(name).use { it.readBytes() }
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        val out = IntArray(bytes.size / 4)
        bb.asIntBuffer().get(out)
        return out
    }
}
