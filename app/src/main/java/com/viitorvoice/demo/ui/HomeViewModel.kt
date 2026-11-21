package com.viitorvoice.demo.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.viitorvoice.demo.Engine
import com.viitorvoice.demo.Native
import com.viitorvoice.demo.audio.AudioIo
import com.viitorvoice.demo.audio.Player
import com.viitorvoice.demo.audio.Recorder
import com.viitorvoice.demo.audio.Render
import com.viitorvoice.demo.audio.WavFile
import com.viitorvoice.demo.data.Store
import com.viitorvoice.demo.data.Voice
import com.viitorvoice.demo.text.Duration
import com.viitorvoice.demo.text.Prompt
import com.viitorvoice.demo.text.Split
import com.viitorvoice.demo.text.Tags
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max
import kotlin.math.min

/** Tab ids: cloning is the primary flow, local edit is the secondary one. */
const val TAB_CLONE = 0
const val TAB_EDIT = 1

/** Which slot a record / import goes into. */
object Slot {
    const val SOURCE = 0
    const val REF = 1
}

/** Reference frames kept per chunk (4 s is plenty for timbre). */
private const val CHUNK_REF_FRAMES = 100

/** Upper bound on one chunk's target frames (~5.6 s at 25 Hz). */
private const val MAX_CHUNK_FRAMES = 140

data class UiState(
    val tab: Int = TAB_CLONE,            // clone is the main entry, edit is secondary
    val text: String = "",
    val baselineText: String = "",
    val refText: String = "",
    val lang: String = "zh",
    val cfg: Float = 2.0f,
    val emotionScale: Float = 1.0f,
    val nvvScale: Float = 0.5f,

    /** Target duration in seconds (0 = auto, the official demo's "auto"). */
    val targetSec: Float = 0f,

    // source clip (edit mode)
    val samples: FloatArray = FloatArray(0),
    val frames: Int = 0,                 // frames the audio provides (<= 200)
    val sampleRate: Int = 24000,
    val selStart: Int = -1,              // frame range to regenerate
    val selEnd: Int = -1,
    val sourcePath: String? = null,
    val durationSec: Float = 0f,

    // reference clip (clone mode)
    val refSamples: FloatArray = FloatArray(0),
    val refFrames: Int = 0,
    val refPath: String? = null,
    val refSec: Float = 0f,

    /** Set when the reference came from the voice library (P0). */
    val activeVoiceId: Long? = null,

    val busy: Boolean = false,
    val stage: String = "",
    val progress: Float = 0f,
    val log: List<String> = emptyList(),
    val resultPath: String? = null,
    val resultSec: Float = 0f,
    val playing: Boolean = false,
    val recording: Boolean = false,
    val recordingSlot: Int = Slot.SOURCE,
    val error: String? = null,
    val modelStatus: String = "",
    val tokensReady: Boolean = false,
)

class HomeViewModel(app: Application) : AndroidViewModel(app) {

    private val _state = MutableStateFlow(UiState(modelStatus = Engine.statusLine()))
    val state: StateFlow<UiState> = _state.asStateFlow()

    private var recorder: Recorder? = null
    private var tokens: IntArray? = null
    private var tokensFrames = 0
    private var cachedSamples: FloatArray? = null

    private var refTokens: IntArray? = null
    private var cachedRefSamples: FloatArray? = null

    /** Debug/calibration override: force the target frames of one clone job. */
    var forceFrames = 0

    // ------------------------------------------------------------ settings
    fun setTab(tab: Int) = _state.update { it.copy(tab = tab, resultPath = null) }
    fun setLang(lang: String) = _state.update { it.copy(lang = lang) }
    fun setCfg(v: Float) = _state.update { it.copy(cfg = v) }
    fun setEmotionScale(v: Float) = _state.update { it.copy(emotionScale = v) }
    fun setNvvScale(v: Float) = _state.update { it.copy(nvvScale = v) }
    fun setRefText(v: String) = _state.update { it.copy(refText = v) }

    /** Empty / 0 means auto (the demo's `target_duration = None`). */
    fun setTargetSec(v: Float) = _state.update { it.copy(targetSec = v.coerceIn(0f, 30f)) }

    // --------------------------------------------------------- voice library
    /** Copy the current reference clip into the voice library. */
    fun saveVoice(name: String) {
        val s = _state.value
        val src = s.refPath?.let(::File) ?: return
        if (!src.isFile) {
            _state.update { it.copy(error = "参考音频不存在") }
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val st = Store.get(getApplication())
                val dst = st.newVoiceFile()
                src.copyTo(dst, overwrite = true)
                val id = st.addVoice(
                    name.trim().ifBlank { "音色" }, dst.absolutePath, s.refText, s.lang, "",
                )
                _state.update { it.copy(activeVoiceId = id) }
                addLog("音色已保存 · ${dst.name}")
            }.onFailure { addLog("保存音色失败: ${it.message}") }
        }
    }

    /** Load a saved voice back into the reference slot. */
    fun loadVoice(v: Voice) {
        if (_state.value.busy) return
        viewModelScope.launch {
            _state.update { it.copy(busy = true, stage = "载入音色", progress = 0.05f, error = null) }
            val res = withContext(Dispatchers.IO) {
                runCatching {
                    val wav = WavFile.read(File(v.wavPath))
                    val pcm = resample(wav.samples, wav.sampleRate, AudioIo.SR)
                    val cut = pcm.copyOf(min(pcm.size, 200 * 960))
                    val f = File(getApplication<Application>().filesDir, "ref.wav")
                    AudioIo.writeWav(f, cut)
                    cut to f
                }.getOrElse { it }
            }
            if (res is Pair<*, *>) {
                @Suppress("UNCHECKED_CAST")
                val pair = res as Pair<FloatArray, File>
                loadSamples(pair.first, AudioIo.SR, pair.second, Slot.REF)
                _state.update {
                    it.copy(
                        refText = v.refText,
                        lang = v.lang.ifBlank { it.lang },
                        activeVoiceId = v.id,
                        tab = TAB_CLONE,
                    )
                }
                addLog("载入音色 ${v.name}")
            } else {
                _state.update {
                    it.copy(
                        busy = false, stage = "",
                        error = "载入失败: ${(res as Throwable).message}",
                    )
                }
            }
        }
    }

    fun onTextChange(value: String) {
        _state.update { it.copy(text = value) }
        if (_state.value.baselineText.isNotEmpty()) reapplyAutoMask()
    }

    fun refreshModels() = _state.update { it.copy(modelStatus = Engine.statusLine()) }
    fun addLog(line: String) {
        _state.update { it.copy(log = (it.log + line).takeLast(200)) }
        runCatching {
            java.io.File(getApplication<android.app.Application>().filesDir, "run_log.txt")
                .appendText(line + "\n")
        }
    }
    fun clearError() = _state.update { it.copy(error = null) }

    /** A drag on the waveform: [start, end) in model frames. */
    fun setSelection(start: Int, end: Int) {
        val s = _state.value
        if (s.frames <= 0) return
        val a = start.coerceIn(0, s.frames - 1)
        val b = end.coerceIn(a + 1, s.frames)
        _state.update { it.copy(selStart = a, selEnd = b) }
    }

    // ------------------------------------------------------------ recording
    fun startRecording(slot: Int) {
        val s = _state.value
        if (s.busy || s.recording) return
        val name = if (slot == Slot.REF) "ref.wav" else "source.wav"
        val f = File(getApplication<Application>().filesDir, name)
        val r = Recorder(f)
        if (!r.start()) {
            _state.update { it.copy(error = "无法启动麦克风") }
            return
        }
        recorder = r
        Player.stop()
        _state.update {
            it.copy(recording = true, recordingSlot = slot, resultPath = null)
        }
    }

    fun stopRecording() {
        val r = recorder ?: return
        val slot = _state.value.recordingSlot
        val secs = r.stop()
        recorder = null
        val name = if (slot == Slot.REF) "ref.wav" else "source.wav"
        val f = File(getApplication<Application>().filesDir, name)
        if (secs < 0.5f) {
            _state.update { it.copy(recording = false, error = "录音太短（至少 0.5 秒）") }
            return
        }
        _state.update { it.copy(recording = false) }
        val wav = WavFile.read(f)
        loadSamples(wav.samples, wav.sampleRate, f, slot)
    }

    // --------------------------------------------------------------- import
    fun importAudio(uri: Uri, slot: Int) {
        val s = _state.value
        if (s.busy) return
        viewModelScope.launch {
            _state.update { it.copy(busy = true, stage = "解码音频", progress = 0.1f, error = null) }
            val res = withContext(Dispatchers.IO) {
                runCatching {
                    val (pcm, rate) = AudioIo.decode(getApplication(), uri)
                    val mono = resample(pcm, rate, AudioIo.SR)
                    val cap = if (slot == Slot.REF) 200 * 960 else 200 * 960
                    val cut = mono.copyOf(min(mono.size, cap))
                    val name = if (slot == Slot.REF) "ref.wav" else "source.wav"
                    val f = File(getApplication<Application>().filesDir, name)
                    AudioIo.writeWav(f, cut)
                    cut to f
                }.getOrElse { it }
            }
            if (res is Pair<*, *>) {
                @Suppress("UNCHECKED_CAST")
                val pair = res as Pair<FloatArray, File>
                loadSamples(pair.first, AudioIo.SR, pair.second, slot)
            } else {
                _state.update {
                    it.copy(busy = false, stage = "", error = "导入失败: ${(res as Throwable).message}")
                }
            }
        }
    }

    /** Loads an existing wav (debug intents / 原音). */
    fun loadFile(file: File, autoEdit: Boolean = false) {
        if (!file.isFile) {
            _state.update { it.copy(error = "找不到 ${file.absolutePath}") }
            return
        }
        viewModelScope.launch {
            _state.update { it.copy(busy = true, stage = "读取音频", progress = 0.05f, error = null) }
            val res = withContext(Dispatchers.IO) {
                runCatching {
                    val wav = WavFile.read(file)
                    val pcm = resample(wav.samples, wav.sampleRate, AudioIo.SR)
                    val cut = pcm.copyOf(min(pcm.size, 200 * 960))
                    val f = File(getApplication<Application>().filesDir, "source.wav")
                    AudioIo.writeWav(f, cut)
                    cut to f
                }.getOrElse { it }
            }
            if (res is Pair<*, *>) {
                @Suppress("UNCHECKED_CAST")
                val pair = res as Pair<FloatArray, File>
                loadSamples(pair.first, AudioIo.SR, pair.second, Slot.SOURCE)
                if (autoEdit) {
                    _state.update { it.copy(tab = TAB_EDIT) }
                    applyEdit()
                }
            } else {
                _state.update {
                    it.copy(busy = false, stage = "", error = "读取失败: ${(res as Throwable).message}")
                }
            }
        }
    }

    private fun resample(pcm: FloatArray, srIn: Int, srOut: Int): FloatArray {
        if (srIn == srOut) return pcm
        val cap = (pcm.size.toLong() * srOut / srIn).toInt() + 8
        val dst = FloatArray(cap)
        val n = Native.nativeResample(pcm, pcm.size, srIn, srOut, dst)
        require(n > 0) { "重采样失败" }
        return dst.copyOf(n)
    }

    /** New audio: cut to whole 40 ms frames and drop the cached tokens. */
    private fun loadSamples(pcm: FloatArray, rate: Int, file: File, slot: Int) {
        val frames = min(pcm.size / 960, 200)
        if (frames <= 0) {
            _state.update { it.copy(busy = false, error = "音频太短") }
            return
        }
        val cut = pcm.copyOf(frames * 960)
        if (slot == Slot.REF) {
            cachedRefSamples = cut
            refTokens = null
            _state.update {
                it.copy(
                    busy = false, stage = "", progress = 0f,
                    refSamples = cut, refFrames = frames, refPath = file.absolutePath,
                    refSec = cut.size / 24000f, resultPath = null, activeVoiceId = null,
                )
            }
            addLog("参考音频 ${"%.2f".format(cut.size / 24000f)} s · $frames 帧")
            return
        }
        cachedSamples = cut
        tokens = null
        tokensFrames = 0
        _state.update {
            it.copy(
                busy = false, stage = "", progress = 0f,
                samples = cut, frames = frames, sampleRate = rate,
                sourcePath = file.absolutePath, resultPath = null,
                durationSec = cut.size / 24000f, tokensReady = false,
                selStart = frames / 3, selEnd = frames * 2 / 3,
                baselineText = if (it.baselineText.isEmpty()) it.text else it.baselineText,
            )
        }
        addLog("音频 ${"%.2f".format(cut.size / 24000f)} s · $frames 帧")
    }

    // ------------------------------------------------------------- the edit
    /** Recompute the selection from the text diff (used once a baseline exists). */
    private fun reapplyAutoMask() {
        val s = _state.value
        if (s.frames <= 0 || s.baselineText.isEmpty()) return
        val (a, b) = diffMask(s.baselineText, s.text, s.frames)
        _state.update { it.copy(selStart = a, selEnd = b) }
    }

    private fun diffMask(baseline: String, edited: String, frames: Int): Pair<Int, Int> {
        if (baseline == edited) return (frames / 3).coerceAtLeast(0) to (frames * 2 / 3).coerceAtLeast(1)
        var p = 0
        while (p < min(baseline.length, edited.length) && baseline[p] == edited[p]) p++
        var q = 0
        while (q < min(baseline.length, edited.length) - p &&
            baseline[baseline.length - 1 - q] == edited[edited.length - 1 - q]
        ) q++
        val i0 = p
        val i1 = edited.length - q
        val len = max(baseline.length, 1).toFloat()
        val f0 = (i0 / len * frames).toInt() - 2
        val f1 = (i1 / len * frames).toInt() + 3
        val a = f0.coerceIn(0, frames - 1)
        val b = f1.coerceIn(a + 1, frames)
        return a to b
    }

    fun applyEdit() {
        val s = _state.value
        if (s.busy || s.frames <= 0) return
        if (s.text.isBlank()) {
            _state.update { it.copy(error = "请输入修改后的整句文本") }
            return
        }
        viewModelScope.launch {
            _state.update { it.copy(busy = true, stage = "准备", progress = 0.02f, error = null) }
            val err = withContext(Dispatchers.IO) {
                runCatching { runEdit(s) }.getOrElse { t ->
                    "ERR ${t.message ?: t.javaClass.simpleName}"
                }
            }
            finishJob(err, "edited.wav")
        }
    }

    private fun runEdit(s: UiState): String {
        var t0 = System.currentTimeMillis()
        fun stage(name: String, p: Float) = _state.update { it.copy(stage = name, progress = p) }

        stage("加载模型", 0.05f)
        Engine.ensureInit().let { if (it.isNotEmpty()) return "ERR $it" }
        Engine.ensureNar().let { if (it.isNotEmpty()) return "ERR $it" }

        val pcm = cachedSamples ?: s.samples
        if (tokens == null || tokensFrames != s.frames) {
            stage("编码音频", 0.10f)
            val e = Engine.ensureEncoder()
            if (e.isNotEmpty()) return "ERR $e"
            val (tk, fr) = Engine.encode(pcm)
            tokens = tk
            tokensFrames = fr
            Engine.releaseEncoder()
            addLog("编码 ${fr} 帧 · ${System.currentTimeMillis() - t0} ms")
            t0 = System.currentTimeMillis()
        }
        val tk = tokens ?: return "ERR 编码失败"

        val a = if (s.selStart >= 0) s.selStart else s.frames / 3
        val b = if (s.selEnd > a) s.selEnd else s.frames * 2 / 3
        addLog("重生成 ${a * 40}–${b * 40} ms（帧 $a–${b - 1}）")

        stage("重生成", 0.15f)
        val prep = Engine.editPrepare(
            getApplication(), tk, s.frames, a, b, s.text,
            cfg = s.cfg, lang = s.lang,
            emotionScale = s.emotionScale, nvvScale = s.nvvScale,
        )
        if (prep != "OK") return "ERR $prep"
        runSteps()
        addLog("重生成 ${System.currentTimeMillis() - t0} ms")
        addLog("NAR分解 " + Engine.narTiming())

        stage("解码", 0.92f)
        val out = File(getApplication<Application>().filesDir, "edited.wav")
        val r = Engine.finishToWav(out)
        if (!r.startsWith("OK")) return "ERR $r"
        val parts = r.removePrefix("OK ").split("|")
        addLog("解码 ${parts.getOrNull(0)} ms · ${out.length() / 1024} KB")
        _state.update {
            it.copy(
                resultPath = out.absolutePath,
                resultSec = (out.length() - 44) / 2f / 24000f,
                baselineText = it.text,
            )
        }
        return "OK"
    }

    // ------------------------------------------------------------ clone
    fun clone() {
        val s = _state.value
        if (s.busy) return
        if (s.refSamples.isEmpty() || s.refFrames <= 0) {
            _state.update { it.copy(error = "先录制或导入一段参考音频") }
            return
        }
        if (s.text.isBlank()) {
            _state.update { it.copy(error = "请输入要合成的文本") }
            return
        }
        viewModelScope.launch {
            _state.update { it.copy(busy = true, stage = "准备", progress = 0.02f, error = null) }
            val err = withContext(Dispatchers.IO) {
                runCatching { runClone(s) }.getOrElse { t ->
                    "ERR ${t.message ?: t.javaClass.simpleName}"
                }
            }
            finishJob(err, "clone.wav")
        }
    }

    /**
     * Headless entry used by the calibration sweep and by `am start` runs:
     * load [refFile] into the reference slot, then clone [text] with [frames]
     * forced (0 = use the rate estimate).
     */
    fun debugClone(refFile: File, text: String, refText: String, frames: Int) {
        if (_state.value.busy) return
        forceFrames = frames
        viewModelScope.launch {
            _state.update { it.copy(busy = true, stage = "准备", progress = 0.02f, error = null) }
            onTextChange(text)
            if (refText.isNotEmpty()) setRefText(refText)
            val err = withContext(Dispatchers.IO) {
                runCatching {
                    val wav = WavFile.read(refFile)
                    val pcm = resample(wav.samples, wav.sampleRate, AudioIo.SR)
                    loadSamples(pcm, AudioIo.SR, refFile, Slot.REF)
                    runClone(_state.value)
                }.getOrElse { "ERR ${it.message ?: it.javaClass.simpleName}" }
            }
            forceFrames = 0
            finishJob(err, "clone.wav")
        }
    }

    private fun runClone(s: UiState): String {
        var t0 = System.currentTimeMillis()
        fun stage(name: String, p: Float) = _state.update { it.copy(stage = name, progress = p) }

        stage("加载模型", 0.05f)
        Engine.ensureInit().let { if (it.isNotEmpty()) return "ERR $it" }
        Engine.ensureNar().let { if (it.isNotEmpty()) return "ERR $it" }

        if (refTokens == null || cachedRefSamples == null) {
            stage("编码参考音频", 0.08f)
            val e = Engine.ensureEncoder()
            if (e.isNotEmpty()) return "ERR $e"
            var pcm = s.refSamples
            if (pcm.size > Engine.ENC_MAX_SAMPLES) {
                addLog("参考音频超过 8 s，只编码前 8 s")
                pcm = pcm.copyOf(Engine.ENC_MAX_SAMPLES)
            }
            val (tk, fr) = Engine.encode(pcm)
            refTokens = tk
            Engine.releaseEncoder()
            addLog("参考编码 $fr 帧 · ${System.currentTimeMillis() - t0} ms")
            t0 = System.currentTimeMillis()
        }
        val rt = refTokens ?: return "ERR 参考编码失败"
        val refFrames = rt.size / 12
        val tok = Engine.tokenizer(getApplication())
        val window = Engine.maxWindow()
        val estAll = if (forceFrames > 0) forceFrames else estimateFrames(s, s.text)

        // One job as long as the whole text fits the 200-frame decoder window.
        if (estAll <= Prompt.DEFAULT_FRAMES) {
            val plan = Prompt.planClone(tok, s.text, s.refText, refFrames, s.lang, estAll, window)
                ?: return "ERR 文本太长：目标约 %.1f s，参考缩到 1 s 也放不进 %.0f s 窗口，请缩短文本"
                    .format(estAll / 25.0, window / 25.0)
            if (plan.refFrames < refFrames) {
                addLog("参考音缩到 %.1f s（%d token 窗口限制）".format(plan.refFrames / 25.0, window))
            }
            val refTok = Engine.trimRef(rt, refFrames, plan.refFrames)
            addLog(
                "克隆 ${s.lang} · 提示 ${plan.cond.size} + 参考 ${plan.refFrames} + 目标 ${plan.frames} 帧"
            )
            stage("克隆生成", 0.15f)
            val prep = Engine.clonePrepare(
                getApplication(), s.text, s.refText.ifBlank { null }, refTok, plan.refFrames,
                plan.frames, s.lang, s.cfg, s.emotionScale, s.nvvScale,
            )
            if (prep != "OK") return "ERR $prep"
            runSteps()
            addLog("克隆 ${System.currentTimeMillis() - t0} ms")
            addLog("NAR分解 " + Engine.narTiming())
            stage("解码", 0.92f)
            val w = File(getApplication<Application>().filesDir, "clone_p0.wav")
            val r = Engine.finishToWav(w)
            if (!r.startsWith("OK")) return "ERR $r"
            addLog("解码 ${r.removePrefix("OK ").split("|").getOrNull(0)} ms")
            return assemble(listOf(w), File(getApplication<Application>().filesDir, "clone.wav"), s.text, t0)
        }

        // Longer than one decoder window: split into sentences and run each.
        // Chunk size is the sentence itself; the loop below only splits a
        // sentence further when its own prompt leaves too little room.
        val refKeep = min(refFrames, CHUNK_REF_FRAMES)
        val cap = if (forceFrames > 0) forceFrames else MAX_CHUNK_FRAMES
        if (min(window, Prompt.SEQ_SHORT) - refKeep - 1 - 20 < 20) {
            return "ERR 窗口 $window 太小（参考 %d 帧），请改短参考音".format(refFrames)
        }
        val todo = ArrayList(Split.chunks(s.text, cap) { estimateFrames(s, it) })
        if (todo.isEmpty()) return "ERR 请输入要合成的文本"
        addLog("文本约 %.1f s，分 %d 段生成".format(estAll / 25.0, todo.size))

        val parts = ArrayList<File>()
        var i = 0
        while (i < todo.size) {
            val piece = todo[i]
            val refN = min(refKeep, refFrames)
            val cond = Prompt.buildClone(tok, piece, s.refText, refN, s.lang)
            val room = window - cond.size - refN - 1
            val want = estimateFrames(s, piece)
            if (want > room || room < 10) {
                // The prompt grew past the reserve: split this piece once more.
                val cap2 = min(room, Prompt.DEFAULT_FRAMES)
                val sub = if (cap2 >= 20) Split.chunks(piece, cap2) { estimateFrames(s, it) }
                else emptyList()
                if (sub.size <= 1) {
                    return "ERR 第 ${i + 1} 段放不进 %d 窗口（提示 %d + 参考 %d）"
                        .format(window, cond.size, refN)
                }
                todo.removeAt(i)
                todo.addAll(i, sub)
                continue
            }
            stage("生成 ${i + 1}/${todo.size}", 0.15f + 0.75f * i / todo.size)
            val refTok = Engine.trimRef(rt, refFrames, refN)
            val prep = Engine.clonePrepare(
                getApplication(), piece, s.refText.ifBlank { null }, refTok, refN,
                want, s.lang, s.cfg, s.emotionScale, s.nvvScale,
            )
            if (prep != "OK") return "ERR 第 ${i + 1} 段: $prep"
            val ms = System.currentTimeMillis()
            runSteps()
            addLog("NAR分解 " + Engine.narTiming())
            val w = File(getApplication<Application>().filesDir, "clone_p${parts.size}.wav")
            val r = Engine.finishToWav(w)
            if (!r.startsWith("OK")) return "ERR $r"
            addLog(
                "段 ${i + 1}/${todo.size} · ${piece.replace(" ", "").length} 字 · 提示 ${cond.size} + 参考 $refN + 目标 $want 帧 · %d ms"
                    .format(System.currentTimeMillis() - ms)
            )
            parts += w
            i++
        }
        stage("解码", 0.92f)
        return assemble(parts, File(getApplication<Application>().filesDir, "clone.wav"), s.text, t0)
    }

    /**
     * Frames for [text] using the official demo's rate model (see [Duration]).
     *
     * An explicit target duration scales every piece by the same factor, the
     * way the demo's `duration` argument replaces the backend's own estimate;
     * per-chunk pieces keep their relative share of the whole text.
     */
    private fun estimateFrames(s: UiState, text: String): Int {
        val est = Duration.frames(text, s.lang)
        if (s.targetSec <= 0f) return est
        val whole = Duration.frames(s.text, s.lang).coerceAtLeast(1)
        return (est * (s.targetSec * Duration.FRAME_RATE / whole)).toInt().coerceAtLeast(1)
    }

    /**
     * Join the per-chunk wavs, cut the silence the overshoot left behind and
     * log the realised speaking rate (the only automatic probe we have for the
     * "does it talk too fast" question).
     */
    private fun assemble(parts: List<File>, out: File, text: String, t0: Long): String {
        // No silence trimming: the model's own sentence-final decay is part of
        // the take, and cutting it per chunk audibly shortened the seams.
        val pcm = parts.map { WavFile.read(it).samples }.filter { it.isNotEmpty() }
        if (pcm.isEmpty()) return "ERR 生成结果为空"
        val all = Render.concat(pcm)
        AudioIo.writeWav(out, all)
        val sec = all.size / 24000f
        val voiced = Render.voicedSec(all)
        val chars = Tags.stripEmotion(text).replace(Regex("\\s"), "").length
        addLog(
            "克隆 %d 段 · %.2f s（有声 %.2f s）· %.1f 字/秒 · 共 %d ms · 峰值 RSS %d MB"
                .format(pcm.size, sec, voiced, if (voiced > 0f) chars / voiced else 0f,
                    System.currentTimeMillis() - t0, peakRssMb())
        )
        _state.update {
            it.copy(resultPath = out.absolutePath, resultSec = sec)
        }
        return "OK"
    }

    /** VmHWM of this process — the number LMK looks at. */
    private fun peakRssMb(): Int = runCatching {
        File("/proc/self/status").readLines()
            .firstOrNull { it.startsWith("VmHWM:") }
            ?.filter { it.isDigit() }?.toInt()?.div(1024) ?: 0
    }.getOrDefault(0)

    private fun runSteps() {
        val nSteps = Native.narSteps()
        for (step in 0 until nSteps) {
            val k = Engine.editStep(step)
            if (k < 0) throw IllegalStateException(Native.nativeNarError())
            _state.update { it.copy(progress = 0.15f + 0.75f * (step + 1) / nSteps) }
        }
    }

    private fun finishJob(err: String, name: String) {
        if (err.startsWith("ERR")) {
            addLog("失败: ${err.removePrefix("ERR ")}")
            _state.update { it.copy(busy = false, stage = "", error = err.removePrefix("ERR ")) }
        } else {
            _state.update { it.copy(busy = false, stage = "", progress = 1f) }
            play()
        }
        refreshModels()
    }

    // ------------------------------------------------------------ playback
    fun play() {
        val path = _state.value.resultPath ?: _state.value.sourcePath ?: return
        startPlayback(File(path))
    }

    private fun startPlayback(file: File) {
        runCatching { Player.play(file) }.onFailure { addLog("播放失败: ${it.message}") }
        _state.update { it.copy(playing = true) }
        viewModelScope.launch {
            // MediaPlayer stops itself on completion; mirror that into the UI.
            while (Player.isPlaying()) kotlinx.coroutines.delay(200)
            _state.update { it.copy(playing = false) }
        }
    }

    fun playSource() {
        val path = _state.value.sourcePath ?: return
        startPlayback(File(path))
    }

    fun playRef() {
        val path = _state.value.refPath ?: return
        startPlayback(File(path))
    }

    fun stop() {
        Player.stop()
        _state.update { it.copy(playing = false) }
    }

    override fun onCleared() {
        recorder?.stop()
        Player.stop()
        Engine.releaseEncoder()
        super.onCleared()
    }
}
