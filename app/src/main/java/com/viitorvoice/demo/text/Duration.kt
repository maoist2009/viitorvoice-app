package com.viitorvoice.demo.text

import java.util.Locale

/**
 * Port of viitorvoice/llm/text_utils.py `RuleDurationEstimator`.
 *
 * The reference estimates how many 25 Hz audio frames the sentence should take
 * by scoring each character (CJK ~3, latin ~1, digits ~3.5, punctuation 0.5)
 * against a reference sentence of known duration, then micro-boosting the
 * result (`low_threshold` 50, `boost_strength` 3):
 *
 *   est = w(target) / (w(ref) / ref_duration)
 *   est = 50 * (est / 50) ** (1 / 3)      if est < 50
 *
 * `refText` "Nice to meet you." at 25 frames is the value the shipping config
 * uses (`no_ref_text_silence_prefix_seconds` = 0, so the text branch wins).
 */
object Duration {

    private const val REF_TEXT = "Nice to meet you."
    private const val REF_FRAMES = 25.0
    private const val LOW_THRESHOLD = 50.0
    private const val BOOST_STRENGTH = 3.0

    private const val CJK = 3.0
    private const val HANGUL = 2.5
    private const val KANA = 2.2
    private const val INDIC = 1.8
    private const val THAI_LAO = 1.5
    private const val KHMER_MYANMAR = 1.8
    private const val ARABIC = 1.5
    private const val HEBREW = 1.5
    private const val LATIN = 1.0
    private const val SPACE = 0.2
    private const val DIGIT = 3.5
    private const val PUNCT = 0.5
    private const val MARK = 0.0
    private const val DEFAULT = 1.0

    /** (upper bound, weight) pairs, checked with the first bound >= code. */
    private val RANGES = intArrayOf(
        0x02AF, 0x03FF, 0x052F, 0x05FF, 0x08FF, 0x0DFF, 0x0EFF, 0x109F,
        0x11FF, 0x309F, 0x30FF, 0x9FFF, 0xD7AF, 0xFAFF, 0xFFEF,
    )
    private val WEIGHTS = doubleArrayOf(
        LATIN, LATIN, LATIN, HEBREW, ARABIC, INDIC, THAI_LAO, KHMER_MYANMAR,
        HANGUL, KANA, KANA, CJK, HANGUL, CJK, LATIN,
    )

    /** Python `unicodedata.category` equivalent, reduced to the classes we use. */
    private fun weightOf(c: Char): Double {
        val code = c.code
        if ((code in 65..90) || (code in 97..122)) return LATIN
        if (code == 32) return SPACE
        when (Character.getType(c).toByte()) {
            Character.NON_SPACING_MARK, Character.ENCLOSING_MARK,
            Character.COMBINING_SPACING_MARK -> return MARK
            Character.DASH_PUNCTUATION, Character.START_PUNCTUATION,
            Character.END_PUNCTUATION, Character.CONNECTOR_PUNCTUATION,
            Character.OTHER_PUNCTUATION, Character.INITIAL_QUOTE_PUNCTUATION,
            Character.FINAL_QUOTE_PUNCTUATION,
            Character.MATH_SYMBOL, Character.CURRENCY_SYMBOL,
            Character.MODIFIER_SYMBOL, Character.OTHER_SYMBOL -> return PUNCT
            Character.SPACE_SEPARATOR, Character.LINE_SEPARATOR,
            Character.PARAGRAPH_SEPARATOR -> return SPACE
            Character.DECIMAL_DIGIT_NUMBER, Character.LETTER_NUMBER,
            Character.OTHER_NUMBER -> return DIGIT
            else -> {}
        }
        var i = 0
        while (i < RANGES.size && RANGES[i] < code) i++
        if (i < RANGES.size) return WEIGHTS[i]
        return if (code > 0x20000) CJK else DEFAULT
    }

    fun totalWeight(text: String): Double {
        var sum = 0.0
        for (c in text) sum += weightOf(c)
        return sum
    }

    /** Frames (25 Hz) for [text]; at least one. */
    fun frames(text: String, refFrames: Double = REF_FRAMES): Int {
        val refWeight = totalWeight(REF_TEXT)
        val weight = totalWeight(text)
        if (refWeight <= 0.0) return 1
        var est = weight / (refWeight / refFrames)
        if (est < LOW_THRESHOLD) {
            est = LOW_THRESHOLD * Math.pow(est / LOW_THRESHOLD, 1.0 / BOOST_STRENGTH)
        }
        return Math.max(1, Math.round(est).toInt())
    }

    /**
     * `_estimate_target_tokens` with a real reference clause: the reference
     * sentence supplies both the speaking rate and the audio length.
     */
    fun frames(text: String, refText: String, refFrames: Int): Int {
        val refWeight = totalWeight(refText)
        if (refWeight <= 0.0 || refFrames <= 0) return frames(text)
        var est = totalWeight(text) / (refWeight / refFrames.toDouble())
        if (est < LOW_THRESHOLD) {
            est = LOW_THRESHOLD * Math.pow(est / LOW_THRESHOLD, 1.0 / BOOST_STRENGTH)
        }
        return Math.max(1, Math.round(est).toInt())
    }

    /** Debug string used by the smoke test. */
    fun describe(text: String): String =
        String.format(Locale.US, "weight %.2f -> %d frames", totalWeight(text), frames(text))
}
