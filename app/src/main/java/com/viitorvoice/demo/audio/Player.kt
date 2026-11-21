package com.viitorvoice.demo.audio

import android.media.MediaPlayer
import java.io.File

/** Minimal playback for the generated WAV. */
object Player {
    private var mp: MediaPlayer? = null

    fun play(file: File) {
        stop()
        mp = MediaPlayer().apply {
            setDataSource(file.absolutePath)
            setOnCompletionListener { stop() }
            prepare()
            start()
        }
    }

    fun stop() {
        mp?.runCatching {
            if (isPlaying) stop()
            release()
        }
        mp = null
    }

    fun isPlaying(): Boolean = mp?.isPlaying == true

    fun positionMs(): Int = mp?.runCatching { currentPosition }?.getOrNull() ?: 0

    fun durationMs(): Int = mp?.runCatching { duration }?.getOrNull() ?: 0

    fun seekTo(ms: Int) {
        mp?.runCatching { seekTo(ms.coerceAtLeast(0)) }
    }
}
