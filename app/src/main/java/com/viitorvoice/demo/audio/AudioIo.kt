package com.viitorvoice.demo.audio

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import androidx.annotation.WorkerThread
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max

/** Recording / decoding / wav writing for the 24 kHz mono pipeline. */
object AudioIo {

    const val SR = 24000

    /** 16-bit PCM wav. */
    fun writeWav(file: File, samples: FloatArray, sampleRate: Int = SR) {
        val n = samples.size
        val bytes = ByteBuffer.allocate(44 + n * 2).order(ByteOrder.LITTLE_ENDIAN)
        bytes.put("RIFF".toByteArray())
        bytes.putInt(36 + n * 2)
        bytes.put("WAVEfmt ".toByteArray())
        bytes.putInt(16)
        bytes.putShort(1)                       // PCM
        bytes.putShort(1)                       // mono
        bytes.putInt(sampleRate)
        bytes.putInt(sampleRate * 2)
        bytes.putShort(2)
        bytes.putShort(16)
        bytes.put("data".toByteArray())
        bytes.putInt(n * 2)
        for (s in samples) {
            bytes.putShort((s.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        }
        file.writeBytes(bytes.array())
    }

    /** Decodes any container the platform understands to mono float pcm. */
    @WorkerThread
    fun decode(ctx: Context, uri: Uri): Pair<FloatArray, Int> {
        val ex = MediaExtractor()
        ctx.contentResolver.openFileDescriptor(uri, "r")!!.use { pfd ->
            ex.setDataSource(pfd.fileDescriptor)
        }
        var track = -1
        var format: MediaFormat? = null
        for (i in 0 until ex.trackCount) {
            val f = ex.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("audio/")) { track = i; format = f; break }
        }
        require(track >= 0 && format != null) { "文件里没有音频轨" }
        ex.selectTrack(track)
        val channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT).coerceAtLeast(1)
        val rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        val pcmEncoding = if (format.containsKey(MediaFormat.KEY_PCM_ENCODING))
            format.getInteger(MediaFormat.KEY_PCM_ENCODING) else AudioFormat.ENCODING_PCM_16BIT

        val codec = MediaCodec.createDecoderByType(format.getString(MediaFormat.KEY_MIME)!!)
        codec.configure(format, null, null, 0)
        codec.start()

        val parts = ArrayList<FloatArray>()
        var total = 0
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        while (!outputDone) {
            if (!inputDone) {
                val idx = codec.dequeueInputBuffer(10_000)
                if (idx >= 0) {
                    val buf = codec.getInputBuffer(idx)!!
                    val size = ex.readSampleData(buf, 0)
                    if (size < 0) {
                        codec.queueInputBuffer(idx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(idx, 0, size, ex.sampleTime, 0)
                        ex.advance()
                    }
                }
            }
            val idx = codec.dequeueOutputBuffer(info, 10_000)
            if (idx >= 0) {
                val buf = codec.getOutputBuffer(idx)!!
                buf.position(info.offset)
                buf.limit(info.offset + info.size)
                val floats: FloatArray = if (pcmEncoding == AudioFormat.ENCODING_PCM_FLOAT) {
                    val fb = buf.order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                    FloatArray(fb.remaining()).also { fb.get(it) }
                } else {
                    val sb = buf.order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                    FloatArray(sb.remaining()) { sb.get(it) / 32768f }
                }
                if (floats.isNotEmpty()) { parts.add(floats); total += floats.size }
                codec.releaseOutputBuffer(idx, false)
                if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
            }
        }
        codec.stop()
        codec.release()
        ex.release()

        require(total > 0) { "解码没有输出" }
        val all = FloatArray(total)
        var off = 0
        for (p in parts) { System.arraycopy(p, 0, all, off, p.size); off += p.size }
        return if (channels == 1) all to rate else downmix(all, channels) to rate
    }

    private fun downmix(interleaved: FloatArray, channels: Int): FloatArray {
        val n = interleaved.size / channels
        val out = FloatArray(n)
        var i = 0
        for (k in 0 until n) {
            var acc = 0f
            for (c in 0 until channels) acc += interleaved[i++]
            out[k] = acc / channels
        }
        return out
    }
}

/** 24 kHz mono recorder writing straight to a 16-bit wav. */
class Recorder(private val file: File) {

    private var record: AudioRecord? = null
    private var thread: Thread? = null
    @Volatile private var running = false
    private var raf: RandomAccessFile? = null
    var sampleRate = AudioIo.SR
        private set

    val isRecording: Boolean get() = running

    fun start(): Boolean {
        val rate = AudioIo.SR
        val min = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT)
        val bufSize = max(min * 2, rate / 2)
        val rec = try {
            AudioRecord(android.media.MediaRecorder.AudioSource.MIC, rate,
                AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, bufSize)
        } catch (t: Throwable) {
            return false
        }
        if (rec.state != AudioRecord.STATE_INITIALIZED) { rec.release(); return false }
        sampleRate = rate
        record = rec
        raf = RandomAccessFile(file, "rw").apply {
            setLength(0)
            write(ByteArray(44))            // header patched on stop
        }
        running = true
        rec.startRecording()
        thread = Thread {
            val buf = ShortArray(1024)
            var samples = 0L
            try {
                while (running) {
                    val n = rec.read(buf, 0, buf.size)
                    if (n <= 0) continue
                    val bytes = ByteArray(n * 2)
                    for (i in 0 until n) {
                        val v = buf[i].toInt()
                        bytes[i * 2] = (v and 0xFF).toByte()
                        bytes[i * 2 + 1] = ((v shr 8) and 0xFF).toByte()
                    }
                    raf?.write(bytes)
                    samples += n
                }
                patchHeader(samples)
            } catch (_: Throwable) {
            }
        }.also { it.start() }
        return true
    }

    /** Stops and returns the recorded duration in seconds. */
    fun stop(): Float {
        running = false
        thread?.join(2000)
        thread = null
        record?.runCatching { stop() }
        record?.release()
        record = null
        raf?.close()
        raf = null
        return (file.length() - 44).coerceAtLeast(0) / 2f / sampleRate
    }

    private fun patchHeader(samples: Long) {
        val n = (samples * 2).toInt()
        val bb = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()); bb.putInt(36 + n)
        bb.put("WAVEfmt ".toByteArray()); bb.putInt(16)
        bb.putShort(1); bb.putShort(1)
        bb.putInt(sampleRate); bb.putInt(sampleRate * 2)
        bb.putShort(2); bb.putShort(16)
        bb.put("data".toByteArray()); bb.putInt(n)
        raf?.seek(0)
        raf?.write(bb.array())
    }
}
