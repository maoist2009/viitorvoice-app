package com.viitorvoice.demo.ui

import com.viitorvoice.demo.asr.AsrPipeline
import com.viitorvoice.demo.audio.AudioIo
import com.viitorvoice.demo.data.Store
import com.viitorvoice.demo.data.Word
import java.io.File
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Turn a decoded source clip into a dubbing project: keep a 24 kHz mono copy,
 * split it on pauses, and create one line per slot with its own source slice,
 * slot start and slot duration. Line text starts empty (no ASR yet).
 */
fun buildDubbingProject(
    store: Store,
    projectId: Long,
    mono: FloatArray,
    asr: List<AsrPipeline.Seg> = emptyList(),
): Int {
    val dir = store.projectDir(projectId)
    val src = File(dir, "source.wav")
    AudioIo.writeWav(src, mono)

    data class Slot(val a: Int, val b: Int, val text: String, val words: List<Word>)
    val slots: List<Slot> = if (asr.isNotEmpty()) {
        asr.map { s ->
            val a = (s.startMs * AudioIo.SR / 1000).toInt().coerceIn(0, mono.size)
            val b = (s.endMs * AudioIo.SR / 1000).toInt().coerceIn(a, mono.size)
            Slot(
                a, b, s.text,
                s.words.map { w ->
                    Word(
                        w.text,
                        (s.startMs + w.startMs) / 1000f,
                        (s.startMs + w.endMs) / 1000f,
                    )
                },
            )
        }
    } else {
        splitBySilence(mono, AudioIo.SR).map { Slot(it.first, it.last, "", emptyList()) }
    }

    slots.forEachIndexed { i, s ->
        val slice = File(dir, "src_$i.wav")
        AudioIo.writeWav(slice, mono.copyOfRange(s.a, s.b))
        val lid = store.addLine(projectId, i, s.text)
        store.line(lid)?.let { l ->
            store.updateLine(
                l.copy(
                    sourceWav = slice.absolutePath,
                    sourceText = s.text,
                    targetSec = (s.b - s.a) / AudioIo.SR.toFloat(),
                    offsetSec = s.a / AudioIo.SR.toFloat(),
                    words = s.words,
                ),
            )
        }
    }
    store.project(projectId)?.let {
        store.updateProject(it.copy(srcAudio = src.absolutePath))
    }
    return slots.size
}

/**
 * Pause-based segmentation: 20 ms RMS, noise floor from the quietest 15 %,
 * gaps over 0.32 s split, 60 ms padding, sub-0.25 s fragments merged back,
 * slots capped at 8 s (the encoder window).
 */
fun splitBySilence(pcm: FloatArray, sr: Int): List<IntRange> {
    val win = (sr * 0.02f).toInt().coerceAtLeast(1)
    val n = pcm.size / win
    if (n < 2) return listOf(0 until pcm.size)
    val rms = FloatArray(n) { i ->
        var acc = 0f
        val a = i * win
        for (k in a until a + win) acc += pcm[k] * pcm[k]
        sqrt(acc / win)
    }
    val sorted = rms.sortedArray()
    val noise = sorted[(n * 0.15f).toInt().coerceIn(0, n - 1)]
    val peak = sorted[n - 1]
    val th = max(0.008f, max(noise * 2.2f, peak * 0.05f))
    val gapWin = (0.32f / 0.02f).toInt()

    val raw = mutableListOf<IntRange>()
    var start = -1
    var last = -1
    for (k in 0 until n) {
        if (rms[k] > th) {
            if (start < 0) start = k
            last = k
        } else if (start >= 0 && k - last > gapWin) {
            raw += start * win until (last + 1) * win
            start = -1
        }
    }
    if (start >= 0) raw += start * win until (last + 1) * win
    if (raw.isEmpty()) return listOf(0 until pcm.size)

    val pad = (sr * 0.06f).toInt()
    val out = mutableListOf<IntRange>()
    for (r in raw) {
        val a = (r.first - pad).coerceAtLeast(0)
        val b = (r.last + pad).coerceAtMost(pcm.size)
        if (b - a < sr * 0.25f) {
            if (out.isNotEmpty() && a - out.last().last < sr * 0.5f) {
                val prev = out.removeAt(out.lastIndex)
                out += prev.first until max(prev.last, b)
            }
            continue
        }
        var s0 = a
        while (b - s0 > 8 * sr) {
            out += s0 until (s0 + 8 * sr)
            s0 += 8 * sr
        }
        out += s0 until b
    }
    return out.ifEmpty { listOf(0 until pcm.size) }
}
