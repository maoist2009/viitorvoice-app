package com.viitorvoice.demo.text

/**
 * Prompt assembly, mirroring viitorvoice/llm/text_utils.py and the two prompt
 * builders in generation.py:
 *
 *   no reference audio (editing, plain synthesis)   `_prepare_inference_inputs`
 *     <|text_start|>{combine_text(text)}<|text_end|>
 *
 *   with reference audio (voice cloning)            same function, ref branch
 *     <|text_start|>
 *       <|prompt_text_start|>{ref text}<|prompt_text_end|>
 *       <|target_text_start|>{target text}<|target_text_end|>
 *     <|text_end|>
 *
 * The audio prompt (reference frames + separator + target window) is appended
 * natively; everything here is the text half.
 */
object Prompt {

    /** Frames the decoder context binaries are built for (8 s at 25 Hz). */
    const val DEFAULT_FRAMES = 200

    /** Prompt windows compiled into the linked llm/head .bin (10.2 s / 20.5 s). */
    const val SEQ_SHORT = 256
    const val SEQ_LONG = 512

    /** Legacy default window (single-graph installs only have this one). */
    const val SEQ = SEQ_SHORT

    private const val PAUSE = "<|pause_anchor|>"
    private const val PAUSE_CHARS = ".!?。！？;；:："
    private const val END_PUNCT = ";:,.!?…)]}\"'；：，。！？、）】"
    private const val TEXT_MASK = "<|text_mask|>"

    private val JA_KANA = Regex("[\\u3040-\\u309f\\u30a0-\\u30ff]")
    private val CJK_SPACE = Regex("(?<=[\\u4e00-\\u9fff])\\s+|\\s+(?=[\\u4e00-\\u9fff])")

    fun styleText(lang: String): String =
        "<|lang_start|>$lang<|lang_end|>" +
            "<|instruct_start|>None<|instruct_end|>"

    /** `combine_text` (no reference clause). */
    fun combine(text: String): String {
        var t = light(text)
        t = t.replace(Regex("[ \t]+"), " ")
        t = CJK_SPACE.replace(t, "")
        return t.trim()
    }

    /** `normalize_nvv_tags` + the newline / full-width paren rule only. */
    fun light(text: String): String {
        var t = Tags.normalizeNvv(text)
        t = t.replace(Regex("[\\r\\n]+"), "")
        t = t.replace('（', '(').replace('）', ')')
        return t
    }

    /**
     * `_condition_text_for_model`: keep the leading emotion tag when the
     * tokenizer knows it, drop it otherwise.
     */
    fun modelText(text: String, tok: VvTokenizer): String {
        val tag = Tags.emotionTag(text) ?: return text.trim()
        return if (tok.has(tag)) text.trim() else Tags.stripEmotion(text)
    }

    /** `insert_text_pause_anchors`. */
    fun pauseAnchors(text: String): String {
        if (text.isEmpty() || text.contains(PAUSE)) return text
        val sb = StringBuilder(text.length + 8)
        for (i in text.indices) {
            sb.append(text[i])
            if (PAUSE_CHARS.indexOf(text[i]) < 0) continue
            var rest = false
            for (j in i + 1 until text.length) {
                if (!text[j].isWhitespace()) { rest = true; break }
            }
            if (rest) sb.append(PAUSE)
        }
        return sb.toString()
    }

    /** `add_punctuation`. */
    fun addPunctuation(text: String): String {
        val t = text.trim()
        if (t.isEmpty()) return t
        if (END_PUNCT.indexOf(t.last()) >= 0) return t
        val zh = t.any { it in '\u4e00'..'\u9fff' }
        return t + if (zh) "。" else "."
    }

    /** `prepare_text_for_tokenizer`: Japanese kanji become `<|ja-漢|>` pieces. */
    fun jaPrepare(text: String, tok: VvTokenizer, lang: String): String {
        if (text.isEmpty()) return text
        if (lang != "ja" && !JA_KANA.containsMatchIn(text)) return text
        val sb = StringBuilder(text.length + 16)
        for (ch in text) {
            if (isJaKanji(ch)) {
                val tag = "<|ja-$ch|>"
                if (tok.has(tag)) { sb.append(tag); continue }
            }
            sb.append(ch)
        }
        return sb.toString()
    }

    private fun isJaKanji(c: Char): Boolean {
        val code = c.code
        return (code in 0x3400..0x4DBF) || (code in 0x4E00..0x9FFF) || (code in 0xF900..0xFAFF)
    }

    /** Prompt for editing / plain synthesis (no reference audio). */
    fun build(tok: VvTokenizer, text: String, lang: String = "zh"): IntArray {
        val body = pauseAnchors(jaPrepare(combine(modelText(text, tok)), tok, lang))
        if (body.isEmpty()) return IntArray(0)
        return tok.encode(styleText(lang)) +
            tok.encode("<|text_start|>$body<|text_end|>")
    }

    /**
     * Prompt for cloning. [refText] null/blank selects the no-ref-text branch:
     * the prompt text is a run of `<|text_mask|>` ids estimated from the
     * reference length (4 tokens per second, generation.py).
     */
    fun buildClone(
        tok: VvTokenizer,
        targetText: String,
        refText: String?,
        refFrames: Int,
        lang: String = "zh",
    ): IntArray {
        val target = pauseAnchors(jaPrepare(combine(modelText(targetText, tok)), tok, lang))
        if (target.isEmpty()) return IntArray(0)
        val prompt = if (refText.isNullOrBlank()) {
            TEXT_MASK.repeat(maskLen(refFrames))
        } else {
            pauseAnchors(jaPrepare(light(addPunctuation(refText)), tok, lang))
        }
        val body = "<|prompt_text_start|>$prompt<|prompt_text_end|>" +
            "<|target_text_start|>$target<|target_text_end|>"
        return tok.encode(styleText(lang)) + tok.encode("<|text_start|>$body<|text_end|>")
    }

    /** `estimate_ref_text_mask_len` with the shipped defaults (4 tokens/s). */
    fun maskLen(refFrames: Int): Int {
        val seconds = refFrames.coerceAtLeast(0) / 25.0
        return Math.round(seconds * 4.0).toInt().coerceIn(1, 256)
    }

    /** Audio frames that still fit next to the prompt and the reference. */
    fun framesFor(condLen: Int, refFrames: Int = 0, seq: Int = SEQ_SHORT): Int {
        val sep = if (refFrames > 0) 1 else 0
        val room = seq - condLen - refFrames - sep
        return if (room > DEFAULT_FRAMES) DEFAULT_FRAMES else room
    }

    /** Smallest window (256 or 512) that holds prompt + reference + target. */
    fun windowFor(condLen: Int, refFrames: Int, targetFrames: Int): Int {
        val sep = if (refFrames > 0) 1 else 0
        return if (condLen + refFrames + sep + targetFrames <= SEQ_SHORT) SEQ_SHORT
        else SEQ_LONG
    }

    /** Shortest reference we are willing to keep (1 s). */
    const val MIN_REF_FRAMES = 25

    /**
     * Overshoot applied to the duration estimate.
     *
     * The NAR has no length model: it spreads the text over exactly the frames
     * it is given, so under-shooting compresses the speech while over-shooting
     * only leaves a trailing silence (which [Render.trimEdges] cuts). Keep this
     * above 1.0 unless the calibration sweep says otherwise.
     */
    const val HEADROOM = 1.0f

    class ClonePlan(val cond: IntArray, val refFrames: Int, val frames: Int)

    /**
     * Fit one cloning job into the 256-token llm window.
     *
     * The window holds `prompt + ref frames + 1 separator + target frames`, so a
     * long reference eats the room the target text needs. Clamping the target
     * instead would make the model read at double speed, so the reference is
     * trimmed (in place, from its tail) until [targetFrames] fit.
     *
     * Returns null when even a 1 s reference leaves fewer than 25 target frames.
     */
    fun planClone(
        tok: VvTokenizer,
        targetText: String,
        refText: String?,
        refFrames: Int,
        lang: String,
        targetFrames: Int,
        seq: Int = SEQ_SHORT,
    ): ClonePlan? {
        val want = targetFrames.coerceAtMost(DEFAULT_FRAMES)
        var r = refFrames.coerceAtMost(DEFAULT_FRAMES)
        while (true) {
            val cond = buildClone(tok, targetText, refText, r, lang)
            if (cond.isEmpty()) return null
            val room = framesFor(cond.size, r, seq)
            if (room >= want) return ClonePlan(cond, r, want)
            if (r <= MIN_REF_FRAMES) return null
            r = maxOf(MIN_REF_FRAMES, r - 5)
        }
    }
}
