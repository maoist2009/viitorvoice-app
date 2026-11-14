package com.viitorvoice.demo

/** Thin JNI surface. Everything heavy lives in libvv_qnn.so. */
object Native {
    init {
        System.loadLibrary("vv_qnn")
    }

    external fun nativeVersion(): String

    /** Returns "OK <backend_ms>|<decoder_ms>" or an error string. */
    external fun nativeInit(libDir: String, skelDir: String, modelsDir: String): String

    /** Returns "OK <decode_ms>|<wav_ms>" or an error string. */
    external fun nativeDecodeToWav(sem: IntArray, aco: IntArray, wavPath: String): String

    external fun nativeTimings(): String

    // ---- NAR (text -> codes -> wav) -------------------------------------
    /** Loads llm.bin, head.bin and the embedding tables. "OK <ms>" or an error. */
    external fun nativeNarInit(modelsDir: String): String

    /** Prompt window in tokens (256 = 10.2 s, 512 = 20.5 s). 0 on failure. */
    external fun nativeNarSetWindow(seq: Int): Int
    external fun nativeNarWindow(): Int
    external fun nativeNarMaxWindow(): Int

    /** condIds: style + text prompt; frames: audio frames to generate. */
    external fun nativeNarPrepare(condIds: IntArray, condLen: Int, frames: Int): String

    /** Effective step count (32 unless models/nar_steps overrides it). */
    external fun nativeNarSteps(): Int

    /** Same, clamped to the schedule the native side can honour. */
    fun narSteps(): Int = nativeNarSteps().coerceIn(1, 32)

    /** One diffusion step; returns the cells filled, or -1 on error. */
    external fun nativeNarStep(step: Int): Int

    external fun nativeNarRemaining(): Int
    external fun nativeNarError(): String
    /** Debug: host-vs-NPU split of the last 32-step job. */
    external fun nativeNarTiming(): String

    /** Structural filter + the 5-graph decoder + wav. "OK <dec_ms>|<wav_ms>". */
    external fun nativeNarFinish(wavPath: String): String

    external fun nativeNarTokens(frames: Int): IntArray?

    // ---- local edit (CFG) ------------------------------------------------
    external fun nativeNarPrepareEdit(
        condIds: IntArray, condLen: Int, target: IntArray,
        editable: BooleanArray, frames: Int, cfg: Float,
    ): String

    external fun nativeNarEditStep(step: Int): Int
    external fun nativeNarEditRemaining(): Int
    external fun nativeNarEditTokens(): IntArray?

    // ---- unified CFG job (cloning / editing / emotion guidance) ----------
    /**
     * One masked-diffusion job with up to four branches:
     * cond + uncond(cfg) + no_emotion + no_nvv.
     *
     * [refTokens] is 12*refFrames codec tokens of the reference clip (null for
     * editing), [target] the starting window (null = fully masked) and
     * [editable] the frames allowed to update (null = all of them).
     */
    // ---- debug hooks (headless harness) ---------------------------------
    external fun nativeNarSetDiagPath(path: String?)
    external fun nativeNarSetWeights(wc: Float, wu: Float)

    external fun nativeNarPrepareCfg(
        condIds: IntArray, condLen: Int,
        auxIds: Array<IntArray>, auxScales: FloatArray,
        refTokens: IntArray?, refFrames: Int,
        target: IntArray?, editable: BooleanArray?,
        frames: Int, cfg: Float,
    ): String

    // ---- encoder (ORT: w2v frontend + dualcodec encode_core) -------------
    external fun nativeEncodeInit(modelsDir: String): String
    external fun nativeEncodeRelease()

    /** Fills tokensOut (12*200) and returns the real frame count, or -1. */
    external fun nativeEncode(pcm: FloatArray, n: Int, tokensOut: IntArray): Int
    external fun nativeEncodeError(): String

    /** Arbitrary-rate resample into dst; returns samples written or -1. */
    external fun nativeResample(src: FloatArray, nIn: Int, srIn: Int, srOut: Int, dst: FloatArray): Int
}
