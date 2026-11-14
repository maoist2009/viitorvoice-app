package com.viitorvoice.demo.text

/**
 * Emotion / non-verbal tags, ported from viitorvoice/llm/text_utils.py.
 *
 * NVV tags are *not* special tokens in this checkpoint: `<|nv-Laughter|>` and
 * its friends are rewritten to their English parenthetical form ("(laughs)")
 * before tokenisation. Emotion tags (`<|emotion-angry|>` …) are real added
 * tokens; when the tokenizer lacks one it is dropped, like
 * `_condition_text_for_model`.
 *
 * The two CFG branches in generation.py are built from exactly these helpers:
 * the "no_emotion" prompt is `stripEmotion`, the "no_nvv" prompt is `stripNvv`.
 */
object Tags {

    private val NVV = linkedMapOf(
        "<|nv-Cough|>" to "(coughs)",
        "<|nv-Sigh|>" to "(sighs)",
        "<|nv-Breathing|>" to "(inhale)",
        "<|nv-Surprise-oh|>" to "(gasps)",
        "<|nv-Surprise-ah|>" to "(gasps)",
        "<|nv-Dissatisfaction-hnn|>" to "(groans)",
        "<|nv-Laughter|>" to "(laughs)",
        "<|nv-Uhm|>" to "(emm)",
        "<|nv-Question-en|>" to "(humming)",
        "<|nv-Question-oh|>" to "(humming)",
        "<|nv-Question-ah|>" to "(humming)",
        "<|nv-Question-ei|>" to "(humming)",
        "<|nv-Question-huh|>" to "(emm)",
        "<|nv-Confirmation-en|>" to "(humming)",
        "<|nv-Throat clearing|>" to "(clear-throat)",
        "<|nv-Gasp|>" to "(gasps)",
        "<|nv-Groaning|>" to "(groans)",
        "<|nv-Surprise-wa|>" to "(gasps)",
        "<|nv-Surprise-yo|>" to "(gasps)",
        "<|nv-Question-yi|>" to "(humming)",
    )

    /** Human-facing choices used by the UI chips. */
    val NVV_CHOICES = listOf(
        "laughs" to "<|nv-Laughter|>",
        "sighs" to "<|nv-Sigh|>",
        "gasps" to "<|nv-Surprise-ah|>",
        "coughs" to "<|nv-Cough|>",
        "inhale" to "<|nv-Breathing|>",
        "humming" to "<|nv-Question-en|>",
    )

    val EMOTION_CHOICES = listOf(
        "愤怒" to "<|emotion-angry|>",
        "开心" to "<|emotion-happy|>",
        "悲伤" to "<|emotion-sad|>",
        "恐惧" to "<|emotion-fearful|>",
    )

    private val NVV_RE = Regex(
        NVV.keys.sortedByDescending { it.length }.joinToString("|") { Regex.escape(it) }
    )
    private val PAREN_RE = Regex(
        NVV.values.toSet().sortedByDescending { it.length }.joinToString("|") { Regex.escape(it) },
        RegexOption.IGNORE_CASE,
    )
    private val EMOTION_RE = Regex("^\\s*<\\|emotion-[^|>]+?\\|>\\s*")
    private val PAUSE_ALIAS_RE = Regex("<\\|pause-anchor\\|>")

    fun normalizeNvv(text: String): String =
        PAUSE_ALIAS_RE.replace(NVV_RE.replace(text) { NVV[it.value] ?: it.value },
            "<|pause_anchor|>")

    fun hasEmotion(text: String): Boolean = text.isNotEmpty() && EMOTION_RE.containsMatchIn(text)

    fun emotionTag(text: String): String? = EMOTION_RE.find(text)?.value?.trim()

    fun stripEmotion(text: String): String =
        if (text.isEmpty()) text else tidy(EMOTION_RE.replace(text) { "" })

    fun hasNvv(text: String): Boolean =
        text.isNotEmpty() && (NVV_RE.containsMatchIn(text) || PAREN_RE.containsMatchIn(text))

    fun stripNvv(text: String): String {
        if (text.isEmpty()) return text
        var t = NVV_RE.replace(text, "")
        t = PAREN_RE.replace(t, "")
        return tidy(t)
    }

    private fun tidy(text: String): String {
        var t = Regex("[ \\t]+").replace(text, " ")
        t = Regex("\\s+([,.;:!?。，；：！？、])").replace(t, "$1")
        t = Regex("([（(])\\s+").replace(t, "$1")
        t = Regex("\\s+([）)])").replace(t, "$1")
        return t.trim()
    }
}
