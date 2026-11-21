package com.viitorvoice.demo.text

/**
 * Target-length estimate, ported from the **official demo** (`demo_gradio.py`'s
 * `_estimated_duration`), not from the backend's `RuleDurationEstimator`.
 *
 * Why the demo's version: the official product path always passes an explicit
 * `duration` to `/v1/voice-clone`, which makes the backend skip its own
 * estimator (`generation.py`: `if has_duration: target = duration * 25`). The
 * backend fallback (char weights, `low_threshold=50` cubic boost) is what we
 * used to port, and it stretches every sentence shorter than 2 s.
 *
 * The demo's rules, verbatim:
 *   - emotion tags are deleted, `<|nv-...|>` becomes `字字` (CJK) or ` word `
 *   - CJK: seconds = chars / 4.8
 *   - other: seconds = max(words / 2.6, chars / 16)
 *   - clamp to [1, 30] s, rounded to 2 decimals
 *
 * We also accept the literal parenthetical form (`(coughs)`) because the app
 * lets people type the token directly; it means the same thing as the tag.
 */
object Duration {

    private const val CJK_CHARS_PER_SEC = 4.8f
    private const val EN_WORDS_PER_SEC = 2.6f
    private const val EN_CHARS_PER_SEC = 16.0f
    private const val MIN_SEC = 1.0f
    private const val MAX_SEC = 30.0f

    /** Audio frame rate of the NAR (frames per second). */
    const val FRAME_RATE = 25

    private val EMOTION_TAG = Regex("\\s*<\\|emotion-[^|>]+?\\|>\\s*")
    private val NVV_TAG = Regex("\\s*<\\|nv-[^|>]+?\\|>\\s*")
    private val NVV_PAREN = Regex(
        "\\((?:burps|chuckle|clear-throat|coughs|emm|exhale|gasps|groans|hissing|" +
            "humming|inhale|laughs|lip-smacking|pant|sighs|sneezes|sniffs|snorts)\\)"
    )

    private val CJK_LANGS = setOf("zh", "ja", "ko", "yue")

    fun isCjk(lang: String): Boolean = lang.lowercase() in CJK_LANGS

    /** Text the way the demo counts it: tags swapped for a duration placeholder. */
    fun countText(text: String, lang: String): String {
        val placeholder = if (isCjk(lang)) "字字" else " word "
        var t = EMOTION_TAG.replace(text, "")
        t = NVV_TAG.replace(t, placeholder)
        t = NVV_PAREN.replace(t, placeholder)
        return t
    }

    /** Seconds of speech, clamped to [1, 30] like the official demo. */
    fun seconds(text: String, lang: String = "zh"): Float {
        val count = countText(text, lang)
        val stripped = count.filterNot { it.isWhitespace() }
        if (stripped.isEmpty()) return MIN_SEC
        val raw = if (isCjk(lang)) {
            stripped.length / CJK_CHARS_PER_SEC
        } else {
            val words = count.trim().split(Regex("\\s+")).count { it.isNotEmpty() }
            maxOf(words / EN_WORDS_PER_SEC, stripped.length / EN_CHARS_PER_SEC)
        }
        val sec = raw.coerceIn(MIN_SEC, MAX_SEC)
        return Math.round(sec * 100f) / 100f
    }

    /** Frames (25 Hz) for [text]; the backend truncates `duration * 25`. */
    fun frames(text: String, lang: String = "zh"): Int =
        (seconds(text, lang) * FRAME_RATE).toInt().coerceAtLeast(1)
}
