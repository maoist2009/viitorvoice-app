package com.viitorvoice.demo.asr

import android.content.Context
import com.viitorvoice.demo.AsrNative
import com.viitorvoice.demo.audio.AudioIo
import com.viitorvoice.demo.audio.WavFile
import java.io.File

/**
 * VAD -> SenseVoice -> forced aligner, one audio file in, timed segments out.
 *
 * The three models are loaded strictly one after another (98 MB VAD, 501 MB
 * SenseVoice, 1.5 GB aligner) and released before the next one starts: the DSP
 * cannot hold them together with the TTS graphs.
 */
object AsrPipeline {

    data class Word(val text: String, val startMs: Long, val endMs: Long)
    data class Seg(
        val startMs: Long,
        val endMs: Long,
        val text: String,
        val words: List<Word>,
    )

    private const val SR = 16000

    fun run(
        ctx: Context,
        wav: File,
        cacheDir: File,
        onLog: (String) -> Unit = {},
        /** Called right after SenseVoice, before the (memory-hungry) aligner. */
        onText: (List<Seg>) -> Unit = {},
    ): List<Seg> {
        val w = WavFile.read(wav)
        val pcm = if (w.sampleRate == SR) w.samples else resample(w.samples, w.sampleRate, SR)
        if (pcm.isEmpty()) return emptyList()

        val dirs = AsrModels.bind(ctx, onLog) ?: return emptyList()
        val libDir = ctx.applicationInfo.nativeLibraryDir
        cacheDir.mkdirs()

        /* ---- 1) VAD ---- */
        val tVad = System.currentTimeMillis()
        val cuts = try {
            sileroVad(pcm, dirs.vad, libDir)
        } catch (e: Throwable) {
            onLog("VAD 失败：${e.message ?: e}")
            return emptyList()
        }
        onLog("VAD ${cuts.size} 段 / ${System.currentTimeMillis() - tVad} ms")
        if (cuts.isEmpty()) return emptyList()

        /* ---- 2) SenseVoice ---- */
        val tSv = System.currentTimeMillis()
        var err = AsrNative.svInit(dirs.models.absolutePath, libDir)
        if (err.isNotEmpty()) {
            onLog("SenseVoice 初始化失败：$err")
            return emptyList()
        }
        val texts = ArrayList<String>(cuts.size)
        try {
            for ((i, r) in cuts.withIndex()) {
                val f = File(cacheDir, "sv_$i.wav")
                AudioIo.writeWav(f, pcm.copyOfRange(r.first, r.last), SR)
                texts.add(AsrNative.svRun(f.absolutePath).trim())
                f.delete()
            }
        } catch (e: Throwable) {
            onLog("SenseVoice 失败：${e.message ?: e}")
            return emptyList()
        } finally {
            AsrNative.svShutdown()
        }
        onLog("SenseVoice ${texts.count { it.isNotBlank() }}/${cuts.size} 段 / ${System.currentTimeMillis() - tSv} ms")
        onText(
            cuts.mapIndexed { i, r ->
                Seg(r.first * 1000L / SR, r.last * 1000L / SR, texts[i], emptyList())
            },
        )

        /* ---- 3) aligner ---- */
        val tAl = System.currentTimeMillis()
        val words = ArrayList<List<Word>>(cuts.size)
        err = AsrNative.alignInit(dirs.aligner.absolutePath, libDir)
        if (err.isNotEmpty()) {
            onLog("对齐器不可用（词级时间戳跳过）：$err")
            repeat(cuts.size) { words.add(emptyList()) }
        } else {
            val tok = AlignerTokenizer(
                File(dirs.aligner, "vocab.json"),
                File(dirs.aligner, "merges.txt"),
            )
            try {
                for ((i, r) in cuts.withIndex()) {
                    val text = texts[i]
                    if (text.isBlank()) {
                        words.add(emptyList())
                        continue
                    }
                    val f = File(cacheDir, "al_$i.wav")
                    AudioIo.writeWav(f, pcm.copyOfRange(r.first, r.last), SR)
                    words.add(alignOne(tok, f, text))
                    f.delete()
                }
            } catch (e: Throwable) {
                onLog("对齐失败：${e.message ?: e}")
                while (words.size < cuts.size) words.add(emptyList())
            } finally {
                AsrNative.alignShutdown()
            }
            onLog("对齐 ${words.count { it.isNotEmpty() }}/${cuts.size} 段 / ${System.currentTimeMillis() - tAl} ms")
        }

        return cuts.mapIndexed { i, r ->
            Seg(
                startMs = r.first * 1000L / SR,
                endMs = r.last * 1000L / SR,
                text = texts[i],
                words = words[i],
            )
        }
    }

    private fun alignOne(tok: AlignerTokenizer, wav: File, text: String): List<Word> {
        val (ws, ids) = tok.buildPrompt(text)
        if (ws.isEmpty()) return emptyList()
        val ms = AsrNative.alignWav(wav.absolutePath, ids)
        val n = minOf(ws.size, ms.size / 2)
        if (n == 0) return emptyList()
        val raw = IntArray(n * 2) { Math.round(ms[it] / 80.0f) }
        val fixed = tok.fixTimestamp(raw)
        return List(n) { k ->
            Word(ws[k], fixed[2 * k].toLong() * 80, fixed[2 * k + 1].toLong() * 80)
        }
    }

    /* ---------------- Silero VAD (same parameters as the ASR app) ---------------- */

    private fun sileroVad(pcm: FloatArray, modelDir: File, libDir: String): List<IntRange> {
        if (pcm.size < 512) return emptyList()
        val err = AsrNative.vadInit(modelDir.absolutePath, libDir)
        if (err.isNotEmpty()) throw IllegalStateException(err)
        val k = AsrNative.vadFrames()
        val block = k * 512
        val nFrames = (pcm.size + 511) / 512
        val probs = try {
            AsrNative.vadReset()
            val nBlocks = (nFrames + k - 1) / k
            val out = FloatArray(nBlocks * k)
            val x = FloatArray(block)
            for (b in 0 until nBlocks) {
                val o = b * block
                val m = minOf(block, pcm.size - o)
                System.arraycopy(pcm, o, x, 0, m)
                if (m < block) java.util.Arrays.fill(x, m, block, 0f)
                System.arraycopy(AsrNative.vadRun(x), 0, out, b * k, k)
            }
            out.copyOf(nFrames)
        } finally {
            AsrNative.vadShutdown()
        }
        return toSegments(pcm.size, probs)
    }

    private fun toSegments(samples: Int, probs: FloatArray): List<IntRange> {
        val hopMs = 512 * 1000 / SR
        val totalMs = samples * 1000 / SR
        val raw = ArrayList<IntArray>()
        var start = -1
        var quiet = 0
        for (i in probs.indices) {
            val t = i * hopMs
            if (start < 0) {
                if (probs[i] >= 0.5f) start = t
            } else if (probs[i] < 0.35f) {
                quiet += hopMs
                if (quiet >= 100) {
                    raw.add(intArrayOf(start, t + hopMs - quiet))
                    start = -1
                    quiet = 0
                }
            } else {
                quiet = 0
            }
        }
        if (start >= 0) raw.add(intArrayOf(start, totalMs))

        val merged = ArrayList<IntArray>()
        for (r in raw) {
            if (r[1] - r[0] < 300) continue
            val a = (r[0] - 30).coerceAtLeast(0)
            val b = (r[1] + 30).coerceAtMost(totalMs)
            val prev = merged.lastOrNull()
            if (prev != null && a <= prev[1]) {
                if (b > prev[1]) prev[1] = b
            } else {
                merged.add(intArrayOf(a, b))
            }
        }

        val out = ArrayList<IntRange>()
        for (r in merged) splitLong(probs, r[0], r[1], hopMs, out)
        return out
    }

    private fun splitLong(probs: FloatArray, a0: Int, b0: Int, hopMs: Int, out: MutableList<IntRange>) {
        var a = a0
        val b = b0
        while (b - a > 7600) {
            val lo = a + 3000
            val hi = a + 7600
            var best = -1
            var bv = Float.MAX_VALUE
            var t = lo / hopMs
            val tEnd = minOf(hi / hopMs, probs.size - 1)
            while (t <= tEnd) {
                if (probs[t] < bv) {
                    bv = probs[t]
                    best = t * hopMs
                }
                t++
            }
            if (best <= a + 500) best = a + 7600
            out.add((a * SR / 1000) until (best * SR / 1000))
            a = best
        }
        if (b - a >= 300) out.add((a * SR / 1000) until (b * SR / 1000))
    }

    private fun resample(x: FloatArray, from: Int, to: Int): FloatArray {
        if (from == to || x.isEmpty()) return x
        val n = (x.size.toLong() * to / from).toInt()
        val out = FloatArray(n)
        for (i in 0 until n) {
            val p = i.toDouble() * from / to
            val a = p.toInt().coerceIn(0, x.size - 1)
            val b = (a + 1).coerceAtMost(x.size - 1)
            val t = (p - a).toFloat()
            out[i] = x[a] * (1f - t) + x[b] * t
        }
        return out
    }
}
