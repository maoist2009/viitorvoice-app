package com.viitorvoice.demo.audio

/**
 * Small helpers on decoded 24 kHz float pcm: silence trimming, chunk joining
 * and the speaking-rate probe used by the clone log.
 *
 * The NAR can only over- or under-shoot the target length (it has no length
 * model), so the safe policy is "give it more frames than needed, then cut the
 * tail". Over-shooting shows up as trailing silence, under-shooting as
 * compressed speech — trimming makes the cheap direction harmless.
 */
object Render {

    /** 10 ms analysis window at 24 kHz. */
    private const val WIN = 240

    private fun thr(db: Float): Float = Math.pow(10.0, db / 20.0).toFloat()

    private fun peak(x: FloatArray, from: Int, to: Int): Float {
        var p = 0f
        for (i in from until to) {
            val a = if (x[i] < 0) -x[i] else x[i]
            if (a > p) p = a
        }
        return p
    }

    /** Index of the first / last 10 ms window above [db]; -1 when all silent. */
    fun voicedBounds(x: FloatArray, db: Float = -45f): IntArray {
        val t = thr(db)
        var first = -1
        var last = -1
        var i = 0
        while (i < x.size) {
            val end = minOf(i + WIN, x.size)
            if (peak(x, i, end) > t) {
                if (first < 0) first = i
                last = end
            }
            i += WIN
        }
        return intArrayOf(first, last)
    }

    /** Cut leading / trailing silence, keeping [leadMs] / [tailMs] of it. */
    fun trimEdges(x: FloatArray, leadMs: Int = 30, tailMs: Int = 60, db: Float = -45f): FloatArray {
        val b = voicedBounds(x, db)
        if (b[0] < 0) return FloatArray(0)
        val head = (b[0] - leadMs * 24).coerceAtLeast(0)
        val tail = (b[1] + tailMs * 24).coerceAtMost(x.size)
        return if (head == 0 && tail == x.size) x else x.copyOfRange(head, tail)
    }

    /** Seconds of non-silent audio. */
    fun voicedSec(x: FloatArray, db: Float = -45f): Float {
        val b = voicedBounds(x, db)
        return if (b[0] < 0) 0f else (b[1] - b[0]) / 24000f
    }

    fun concat(parts: List<FloatArray>): FloatArray {
        var n = 0
        for (p in parts) n += p.size
        val out = FloatArray(n)
        var o = 0
        for (p in parts) {
            System.arraycopy(p, 0, out, o, p.size)
            o += p.size
        }
        return out
    }
}
