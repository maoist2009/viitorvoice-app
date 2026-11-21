package com.viitorvoice.demo

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.IBinder
import android.util.Log
import com.viitorvoice.demo.audio.WavFile
import com.viitorvoice.demo.text.Duration
import com.viitorvoice.demo.text.Prompt
import com.viitorvoice.demo.text.Tags
import java.io.File

/**
 * Runs the encode -> local-edit -> decode chain out of process lifetime, so a
 * multi-minute NPU run is not killed by the broadcast ANR window.
 *
 * Everything is logged to filesDir/edit.txt (logcat is unreliable here).
 */
class EditService : Service() {

    override fun onBind(intent: Intent?): IBinder? = null

    /** Incremental log: the file is rewritten on every line so a run that is
     *  SIGKILLed (LMK) still leaves a trail up to the moment it died. */
    private var target: File? = null
    private fun StringBuilder.note(s: String) {
        appendLine(s)
        target?.let { f -> runCatching { f.writeText(toString()) } }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        startForeground(1, notification())
        val extras = EditReceiver.copyOf(intent)
        Thread {
            try {
                run(extras)
            } catch (t: Throwable) {
                Log.e(TAG, "edit failed", t)
                File(filesDir, "edit.txt").appendText("\nFAIL ${t.javaClass.simpleName}: ${t.message}\n")
            } finally {
                stopSelf()
            }
        }.start()
        return START_NOT_STICKY
    }

    private fun notification(): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        if (nm.getNotificationChannel(CHANNEL) == null) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL, "ViiTorVoice", NotificationManager.IMPORTANCE_LOW)
            )
        }
        return Notification.Builder(this, CHANNEL)
            .setContentTitle("ViiTorVoice")
            .setContentText("正在处理音频…")
            .setSmallIcon(android.R.drawable.ic_media_play)
            .build()
    }

    /**
     * Headless voice-cloning run:
     *
     *   am start-foreground-service -n com.viitorvoice.demo.tmp/com.viitorvoice.demo.EditService \
     *       --es mode clone --es ref /sdcard/Download/vv/ref.wav \
     *       --es text "要合成的话" [--es reftext "参考音频的原话"] [--es lang zh] \
     *       [--ei frames 0] [--ef cfg 2.0] [--ef emo 1.0] [--ef nvv 0.5]
     */
    private fun runClone(intent: Intent) {
        val log = StringBuilder()
        target = File(filesDir, "clone.txt")
        val t0 = System.currentTimeMillis()
        val out = File(filesDir, "clone_out.wav")
        try {
            val refPath = intent.getStringExtra("ref") ?: (filesDir.absolutePath + "/ref.wav")
            val text = intent.getStringExtra("text") ?: ""
            val refText = intent.getStringExtra("reftext")?.takeIf { it.isNotBlank() }
            val lang = intent.getStringExtra("lang") ?: "zh"
            val cfg = intent.getFloatExtra("cfg", 2.0f)
            val emo = intent.getFloatExtra("emo", 1.0f)
            val nvv = intent.getFloatExtra("nvv", 0.5f)
            val forced = intent.getIntExtra("frames", 0)
            val wc = intent.getFloatExtra("wc", -1f)
            val wu = intent.getFloatExtra("wu", 0f)
            if (wc >= 0f) Engine.setWeights(wc, wu)
            if (intent.getBooleanExtra("diag", false)) {
                Engine.setDiagPath(File(filesDir, "diag.txt").absolutePath)
            }

            log.note("mode clone ref=$refPath lang=$lang cfg=$cfg emo=$emo nvv=$nvv")
            log.note("text=$text")
            log.note("reftext=$refText")
            log.note("init: ${Engine.ensureInit()}")
            log.note("nar: ${Engine.ensureNar().ifEmpty { Engine.narInitMsg }}")
            log.note("encoder: ${Engine.ensureEncoder()}")

            var data = WavFile.read(File(refPath))
            log.note("ref wav ${data.samples.size} @${data.sampleRate} Hz")
            if (data.sampleRate != 24000) {
                val cap = (data.samples.size.toLong() * 24000 / data.sampleRate).toInt() + 8
                val dst = FloatArray(cap)
                val n = Native.nativeResample(data.samples, data.samples.size,
                    data.sampleRate, 24000, dst)
                require(n > 0) { "resample failed" }
                data = WavFile.Data(dst.copyOf(n), 24000)
                log.note("resampled to $n @24000")
            }

            var pcm = data.samples
            if (pcm.size > Engine.ENC_MAX_SAMPLES) {
                log.note("ref ${pcm.size} samples > 8 s window, trimmed")
                pcm = pcm.copyOf(Engine.ENC_MAX_SAMPLES)
            }
            val enc0 = System.currentTimeMillis()
            val (refTokens, refFrames) = Engine.encode(pcm)
            var dg = 2166136261L
            for (t in refTokens) {
                dg = dg xor (t.toLong() and 0xffffffffL)
                dg = (dg * 16777619L) and 0xffffffffL
            }
            log.note("ref encode $refFrames frames in ${System.currentTimeMillis() - enc0} ms" +
                     " digest=${dg.toString(16)}")
            Engine.releaseEncoder()

            val tok = Engine.tokenizer(this)
            val est = Duration.frames(text, lang)      // official demo rate model
            val want = if (forced > 0) forced else est
            val plan = Prompt.planClone(tok, text, refText, refFrames, lang, want, Engine.maxWindow())
                ?: throw IllegalStateException("窗口不够: est $est ref $refFrames")
            val refTok = Engine.trimRef(refTokens, refFrames, plan.refFrames)
            log.note("prompt ${plan.cond.size} tokens: ${plan.cond.joinToString(" ")}")
            log.note("ref $refFrames -> ${plan.refFrames}, target ${plan.frames} (est $est)")

            val r = Engine.clonePrepare(this, text, refText, refTok, plan.refFrames,
                plan.frames, lang, cfg, emo, nvv)
            if (r != "OK") throw IllegalStateException("prepare: $r")

            val nar0 = System.currentTimeMillis()
            val nSteps = Native.nativeNarSteps().let { if (it in 1..Engine.STEPS) it else Engine.STEPS }
            for (step in 0 until nSteps) {
                val k = Engine.editStep(step)
                if (k < 0) throw IllegalStateException("step $step: ${Native.nativeNarError()}")
                if (step % 4 == 3) {
                    log.note("  step ${step + 1}/$nSteps filled=$k left=${Engine.editRemaining()} " +
                        "(${System.currentTimeMillis() - nar0} ms)")
                }
            }
            log.note("nar ${System.currentTimeMillis() - nar0} ms")
            log.note("timing: ${Engine.narTiming()}")
            log.note("finish: ${Engine.finishToWav(out)}")
            log.note("out ${out.length()} B")
            log.note("total ${System.currentTimeMillis() - t0} ms")

            val res = Native.nativeNarEditTokens()
            if (res != null) {
                File(filesDir, "clone_tokens.txt").writeText((0 until 12).joinToString("\n") { c ->
                    (0 until plan.frames).joinToString(" ") { f -> res[c * plan.frames + f].toString() }
                })
            }
            File(filesDir, "clone.txt").writeText(log.toString())
            Log.i(TAG, log.toString())
        } catch (t: Throwable) {
            log.note("FAIL ${t.javaClass.simpleName}: ${t.message}")
            File(filesDir, "clone.txt").writeText(log.toString())
            Log.e(TAG, "clone failed", t)
        }
    }

    /** Plain text -> speech through the legacy (single-branch) diffusion loop. */
    private fun runSynth(intent: Intent) {
        val log = StringBuilder()
        target = File(filesDir, "synth.txt")
        val t0 = System.currentTimeMillis()
        val out = File(filesDir, "synth_out.wav")
        try {
            val text = intent.getStringExtra("text") ?: ""
            log.note("mode synth text=$text")
            log.note("init: ${Engine.ensureInit()}")
            log.note("nar: ${Engine.ensureNar().ifEmpty { Engine.narInitMsg }}")
            val tok = Engine.tokenizer(this)
            val cond = Prompt.build(tok, text)
            val frames = intent.getIntExtra("frames", 0).takeIf { it > 0 }
                ?: Prompt.framesFor(cond.size)
            log.note("prompt ${cond.size} tokens: ${cond.joinToString(" ")}")
            log.note("frames $frames")
            val we = Engine.useWindow(cond.size, 0, frames)
            if (we.isNotEmpty()) throw IllegalStateException(we)
            var r = Native.nativeNarPrepare(cond, cond.size, frames)
            log.note("prepare: $r")
            if (!r.startsWith("OK")) throw IllegalStateException(r)
            val nar0 = System.currentTimeMillis()
            val nSteps = Native.narSteps()
            for (step in 0 until nSteps) {
                val k = Native.nativeNarStep(step)
                if (k < 0) throw IllegalStateException("step $step: ${Native.nativeNarError()}")
                if (step % 4 == 3) {
                    log.note("  step ${step + 1}/$nSteps filled=$k left=${Native.nativeNarRemaining()} " +
                        "(${System.currentTimeMillis() - nar0} ms)")
                }
            }
            log.note("nar ${System.currentTimeMillis() - nar0} ms")
            log.note("timing: ${Engine.narTiming()}")
            log.note("finish: ${Engine.finishToWav(out)}")
            log.note("out ${out.length()} B")
            log.note("total ${System.currentTimeMillis() - t0} ms")
            val res = Native.nativeNarEditTokens()
            if (res != null) {
                File(filesDir, "synth_tokens.txt").writeText((0 until 12).joinToString("\n") { c ->
                    (0 until frames).joinToString(" ") { f -> res[c * frames + f].toString() }
                })
            }
            File(filesDir, "synth.txt").writeText(log.toString())
        } catch (t: Throwable) {
            log.note("FAIL ${t.javaClass.simpleName}: ${t.message}")
            File(filesDir, "synth.txt").writeText(log.toString())
        }
    }

    private fun run(intent: Intent) {
        when (intent.getStringExtra("mode") ?: "edit") {
            "clone" -> { runClone(intent); return }
            "synth" -> { runSynth(intent); return }
        }
        val log = StringBuilder()
        target = File(filesDir, "edit.txt")
        val t0 = System.currentTimeMillis()
        val src = intent.getStringExtra("src") ?: (filesDir.absolutePath + "/edit_in.wav")
        val newText = intent.getStringExtra("text") ?: ""
        val a = intent.getIntExtra("a", -1)
        val b = intent.getIntExtra("b", -1)
        val cfg = intent.getFloatExtra("cfg", 2.0f)
        val wc = intent.getFloatExtra("wc", -1f)
        val wu = intent.getFloatExtra("wu", 0f)
        if (wc >= 0f) Engine.setWeights(wc, wu)
        if (intent.getBooleanExtra("diag", false)) {
            Engine.setDiagPath(File(filesDir, "diag.txt").absolutePath)
        }

        log.note("init: ${Engine.ensureInit()}")
        log.note("nar: ${Engine.ensureNar().ifEmpty { Engine.narInitMsg }}")
        log.note("encoder: ${Engine.ensureEncoder()}")

        var data = WavFile.read(File(src))
        log.note("wav ${data.samples.size} @${data.sampleRate} Hz")
        if (data.sampleRate != 24000) {
            val cap = (data.samples.size.toLong() * 24000 / data.sampleRate).toInt() + 8
            val dst = FloatArray(cap)
            val n = Native.nativeResample(data.samples, data.samples.size,
                data.sampleRate, 24000, dst)
            require(n > 0) { "resample failed" }
            data = WavFile.Data(dst.copyOf(n), 24000)
            log.note("resampled to $n @24000")
        }

        val enc0 = System.currentTimeMillis()
        val (tokens, frames) = Engine.encode(data.samples)
        log.note("encode ${frames} frames in ${System.currentTimeMillis() - enc0} ms")
        // The ORT sessions hold 2.4 GB: drop them before the NPU loop.
        Engine.releaseEncoder()
        log.note("encoder released")

        val m0 = if (a in 0 until frames) a else frames / 4
        val m1 = if (b in (m0 + 1)..frames) b else (frames * 3 / 4)
        log.note("edit mask [$m0,$m1) of $frames, cfg=$cfg")

        val r = Engine.editPrepare(this, tokens, frames, m0, m1, newText, cfg)
        if (r != "OK") throw IllegalStateException("prepare: $r")
        val nar0 = System.currentTimeMillis()
        val nSteps = Native.narSteps()
        for (step in 0 until nSteps) {
            val k = Engine.editStep(step)
            if (k < 0) throw IllegalStateException("step $step: ${Native.nativeNarError()}")
            if (step % 4 == 3) {
                log.note("  step ${step + 1}/$nSteps filled=$k left=${Engine.editRemaining()} " +
                    "(${System.currentTimeMillis() - nar0} ms)")
            }
        }
        log.note("nar ${System.currentTimeMillis() - nar0} ms, left=${Engine.editRemaining()}")
        log.note("timing: ${Engine.narTiming()}")

        val out = File(filesDir, "edit_out.wav")
        log.note("finish: ${Engine.finishToWav(out)}")
        log.note("out ${out.length()} B")
        val edited = Native.nativeNarEditTokens()
        if (edited != null) {
            val same = (0 until frames).count { f ->
                (0 until 12).all { c -> edited[c * frames + f] == tokens[c * frames + f] }
            }
            log.note("frames unchanged: $same/$frames")
            val dump = File(filesDir, "edit_tokens.txt")
            dump.writeText((0 until 12).joinToString("\n") { c ->
                (0 until frames).joinToString(" ") { f -> edited[c * frames + f].toString() }
            })
        }
        log.note("total ${System.currentTimeMillis() - t0} ms")
        File(filesDir, "edit.txt").writeText(log.toString())
        Log.i(TAG, log.toString())
    }

    companion object {
        private const val TAG = "VvEdit"
        private const val CHANNEL = "vv_edit"
    }
}
