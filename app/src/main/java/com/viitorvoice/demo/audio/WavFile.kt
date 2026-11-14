package com.viitorvoice.demo.audio

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

/** Minimal RIFF/WAVE reader for 16-bit PCM (mono or stereo). */
object WavFile {

    class Data(val samples: FloatArray, val sampleRate: Int)

    fun read(file: File): Data {
        val bytes = file.readBytes()
        val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        require(bytes.size > 44 && bytes[0] == 'R'.code.toByte()) { "not a wav" }
        var pos = 12
        var channels = 1
        var rate = 24000
        var bits = 16
        var dataOff = -1
        var dataLen = 0
        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4, Charsets.US_ASCII)
            val size = bb.getInt(pos + 4)
            if (id == "fmt ") {
                channels = bb.getShort(pos + 10).toInt()
                rate = bb.getInt(pos + 12)
                bits = bb.getShort(pos + 22).toInt()
            } else if (id == "data") {
                dataOff = pos + 8
                dataLen = size.coerceAtMost(bytes.size - dataOff)
                break
            }
            pos += 8 + size + (size and 1)
        }
        require(dataOff > 0) { "no data chunk" }
        require(bits == 16) { "only 16-bit pcm supported" }
        val n = dataLen / 2 / max(1, channels)
        val out = FloatArray(n)
        for (i in 0 until n) {
            var acc = 0
            for (ch in 0 until channels) {
                acc += bb.getShort(dataOff + (i * channels + ch) * 2).toInt()
            }
            out[i] = acc / (32768f * channels)
        }
        return Data(out, rate)
    }
}
