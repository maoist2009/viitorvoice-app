package com.viitorvoice.demo

/**
 * JNI surface of libasr_qnn.so (shared with the standalone ASR app).
 *
 * Only the pieces the studio needs are declared here: Silero VAD, SenseVoice
 * CTC and the Qwen3 forced aligner. The Qwen3-ASR / CAM++ entry points are
 * compiled in but unused.
 */
object AsrNative {

    init {
        System.loadLibrary("asr_qnn")
    }

    /* ---- SenseVoice-Small ---- */

    /** Returns "" on success, otherwise the error text. */
    external fun svInit(modelDir: String, libDir: String): String

    /** Runs one wav (16 kHz mono, <=30 s) and returns the text. */
    external fun svRun(wavPath: String): String

    external fun svShutdown()

    /* ---- Silero VAD (QNN) ---- */

    external fun vadInit(modelDir: String, libDir: String): String

    external fun vadReset()

    /** Frames per graph execution. */
    external fun vadFrames(): Int

    /** One block of `vadFrames()*512` samples -> speech probability per frame. */
    external fun vadRun(pcm: FloatArray): FloatArray

    external fun vadShutdown()

    /* ---- Qwen3-ForcedAligner ---- */

    external fun alignInit(modelDir: String, libDir: String): String

    /** ids = full prompt (one <|audio_pad|>); returns ms at each timestamp slot. */
    external fun alignWav(wavPath: String, ids: IntArray): FloatArray

    external fun alignShutdown()
}
