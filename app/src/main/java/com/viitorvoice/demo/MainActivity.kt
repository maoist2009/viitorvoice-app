package com.viitorvoice.demo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.viitorvoice.demo.data.Store
import com.viitorvoice.demo.text.Duration
import com.viitorvoice.demo.ui.AppRoot
import com.viitorvoice.demo.ui.theme.ViiTorVoiceTheme
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Headless probe for the shared model library (MLP/1). The UI is a
        // Compose demo that needs a human; this path lets the model wiring be
        // checked from adb:
        //   am start -n com.viitorvoice.demo.tmp/.MainActivity --es selftest 1
        if (intent?.hasExtra("selftest") == true) {
            runSelfTest()
            return
        }
        // Headless studio seed: --es seed 1 builds a project from files/ref.wav.
        if (intent?.hasExtra("seed") == true) {
            runSeed()
            return
        }
        // Headless dubbing import: --es dub <wav path> splits it into slots.
        intent?.getStringExtra("dub")?.let {
            runDubbing(it)
            return
        }
        // Headless word edit: --ei editword <lineId> masks the first two words.
        intent?.getIntExtra("editword", 0)?.takeIf { it > 0 }?.let {
            runEditWord(it)
            return
        }
        enableEdgeToEdge()
        setContent {
            ViiTorVoiceTheme {
                AppRoot()
            }
        }
    }

    private fun runSelfTest() {
        val out = File(filesDir, "selftest.txt")
        out.writeText("")
        Thread {
            fun line(k: String, v: Any?) {
                out.appendText("$k=$v\n")
            }
            fun step(k: String, f: () -> Any?) {
                val t0 = System.currentTimeMillis()
                try {
                    line(k, "${f()} (${System.currentTimeMillis() - t0} ms)")
                } catch (t: Throwable) {
                    line("$k.EX", t.toString())
                }
            }
            step("duration") {
                // Official demo rule: 4.8 chars/s (CJK), NVV = 2-char placeholder,
                // clamp [1, 30] s. Doubles as a regression check for Duration.kt.
                listOf("你好" to "zh", "咳嗽(coughs)一下" to "zh", "Nice to meet you." to "en")
                    .joinToString(" | ") { (t, l) ->
                        "$l %.2fs/%df".format(Duration.seconds(t, l), Duration.frames(t, l))
                    }
            }
            step("tokenizer") {
                // NVV must stay one id (52678) instead of the BPE pieces of the
                // English word "(coughs)"; see VvTokenizer.literal.
                Engine.tokenizer(application).encode("咳嗽(coughs)一下").joinToString(",")
            }
            step("store") { Store.get(application).selfTest() }
            step("host") { ModelStore.library(application).hostPkg }
            step("lib_summary") { ModelStore.summary(application) }
            step("store_ok") { Engine.ensureStore() }
            step("store_msg") { Engine.storeMsg }
            step("bound_fds") { Native.nativeBoundModelFds() }
            step("init") { Engine.ensureInit().ifEmpty { "OK" } }
            step("nar") { Engine.ensureNar().ifEmpty { "OK" } }
            step("nar_msg") { Engine.narInitMsg }
            step("encoder") { Engine.ensureEncoder().ifEmpty { "OK" } }
            step("synth") {
                val wav = File(filesDir, "selftest.wav")
                val r = Engine.synthesize(application, "共享模型库测试。", wav)
                "$r wav=${wav.length()} B"
            }
            step("publish") {
                val probe = File(filesDir, "selftest.wav")
                val r = ModelStore.publish(
                    application, "viitorvoice-test", "any", listOf(probe),
                )
                val e = ModelStore.library(application).resolve("viitorvoice-test", "selftest.wav", "any")
                val un = ModelStore.library(application)
                    .unpublish("viitorvoice-test", "selftest.wav", "any")
                "$r | entry=${e?.sha256?.take(12)}/${e?.size} | unpublish=$un"
            }
            step("status") { Engine.statusLine() }
            runOnUiThread { finish() }
        }.start()
    }

    /** Debug: mask the first two words of a line and run the edit path. */
    private fun runEditWord(lineId: Int) {
        val id = lineId.toLong()
        Thread {
            val out = File(filesDir, "editword.txt")
            try {
                out.writeText("")
                val st = Store.get(application)
                val l = st.line(id) ?: error("no line $id")
                if (l.words.size < 2) error("line has ${l.words.size} words")
                val wav = com.viitorvoice.demo.audio.WavFile.read(File(l.sourceWav))
                val err0 = Engine.ensureInit().ifEmpty { Engine.ensureNar() }
                if (err0.isNotEmpty()) error(err0)
                val (tokens, frames) = Engine.encode(wav.samples)
                val w = l.words.take(2)
                val a = ((w[0].start - l.offsetSec) * 25f).toInt().coerceIn(0, frames)
                val b = ((w[1].end - l.offsetSec) * 25f).toInt().coerceIn(a, frames)
                val drop = w.sumOf { it.text.length }
                val newText = "改" + l.text.drop(drop)
                val t0 = System.currentTimeMillis()
                val prep = Engine.editPrepare(
                    application, tokens, frames, a, b, newText,
                    cfg = l.cfg, lang = l.lang,
                    emotionScale = l.emotion, nvvScale = l.nvv,
                )
                if (prep != "OK") error("prep=$prep")
                val steps = com.viitorvoice.demo.Native.narSteps()
                for (s in 0 until steps) {
                    if (Engine.editStep(s) < 0) error(com.viitorvoice.demo.Native.nativeNarError())
                }
                val f = File(filesDir, "editword.wav")
                val r = Engine.finishToWav(f)
                val sec = (f.length() - 44) / 2f / 24000
                out.writeText(
                    "line=$lineId frames=$frames mask=$a..$b new=$newText prep=$prep " +
                        "finish=$r sec=%.2f wall=%dms\n".format(sec, System.currentTimeMillis() - t0),
                )
            } catch (t: Throwable) {
                out.writeText("ERR $t\n")
            }
            runOnUiThread { finish() }
        }.start()
    }

    /** Debug: build a dubbing project from a local 24 kHz wav. */
    private fun runDubbing(path: String) {
        Thread {
            val out = File(filesDir, "dub.txt")
            try {
                out.writeText("")
                val st = Store.get(application)
                val wav = com.viitorvoice.demo.audio.WavFile.read(File(path))
                val pid = st.addProject(File(path).nameWithoutExtension, null, "", "audio")
                val t0 = System.currentTimeMillis()
                val segs = com.viitorvoice.demo.asr.AsrClient.transcribe(application, File(path))
                val tAsr = System.currentTimeMillis() - t0
                val n = com.viitorvoice.demo.ui.buildDubbingProject(st, pid, wav.samples, segs)
                out.appendText(
                    "project=$pid segments=$n asr=${segs.size} ${tAsr}ms lines=${st.lines(pid).size}\n" +
                        st.lines(pid).joinToString("\n") {
                            "line=${it.id} off=%.2f target=%.2f words=%d text=%s src=%s".format(
                                it.offsetSec, it.targetSec, it.words.size, it.text,
                                File(it.sourceWav).name,
                            )
                        } + "\n",
                )
            } catch (t: Throwable) {
                out.writeText("ERR $t\n")
            }
            runOnUiThread { finish() }
        }.start()
    }

    /** Debug: create one voice + project + line so the studio UI has data. */
    private fun runSeed() {
        Thread {
            val out = File(filesDir, "seed.txt")
            try {
                val st = Store.get(application)
                val ref = File(filesDir, "ref.wav")
                val v = st.addVoice("ref", ref.absolutePath, "", "zh", "")
                val p = st.addProject("种子工程", v)
                st.addLine(p, 0, "今天天气不错，我们出去走走吧。")
                st.addLine(p, 1, "顺便买点水果回来。")
                out.writeText("voice=$v project=$p lines=${st.lines(p).size}\n")
            } catch (t: Throwable) {
                out.writeText("ERR $t\n")
            }
            runOnUiThread { finish() }
        }.start()
    }
}
