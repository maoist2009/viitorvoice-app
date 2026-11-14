package com.viitorvoice.demo.text

/**
 * Sentence-level chunking for the fixed-size NAR window.
 *
 * The NAR has no length model: it spreads the text over exactly the frames it
 * is given, so a text that does not fit one window has to be generated in
 * pieces. Splitting also keeps every piece inside the 200-frame decoder window
 * and makes each piece's rate estimate local (an error in one sentence no
 * longer compresses the whole utterance).
 *
 * Order of the cuts, from coarsest to finest:
 *   sentence end (。！？!?；;…)  ->  clause (，、,：:)  ->  hard cut at [maxFrames].
 */
object Split {

    private const val HARD = "。！？!?；;…\n"
    private const val SOFT = "，、,：:"

    /** Leading emotion tag, or "" when the text carries none. */
    fun lead(text: String): String = Tags.emotionTag(text) ?: ""

    /**
     * Split [text] into pieces whose estimated frame count stays under
     * [maxFrames]. Every piece keeps the leading emotion tag so the CFG
     * branches still see it. Returns an empty list for empty text.
     */
    fun chunks(text: String, maxFrames: Int, est: (String) -> Int): List<String> {
        require(maxFrames >= 8) { "maxFrames too small" }
        val body = Tags.stripEmotion(text).trim()
        if (body.isEmpty()) return emptyList()
        val tag = lead(text)

        val sentences = splitBy(body, HARD)
        val pieces = ArrayList<String>()
        for (s in sentences) {
            if (est(s) <= maxFrames) {
                pieces += s
                continue
            }
            // Too long for one window: fall back to clauses, then to raw cuts.
            for (c in splitBy(s, SOFT)) {
                if (est(c) <= maxFrames) pieces += c
                else pieces += hardCut(c, maxFrames, est)
            }
        }

        // Re-join neighbours that still fit together (short clauses add up).
        val merged = ArrayList<String>()
        for (p in pieces) {
            if (merged.isNotEmpty()) {
                val joined = merged[merged.size - 1] + p
                if (est(joined) <= maxFrames) {
                    merged[merged.size - 1] = joined
                    continue
                }
            }
            merged += p
        }
        return merged.map { if (tag.isEmpty()) it else "$tag $it" }
    }

    /** Cut after every character of [marks], keeping the mark. */
    private fun splitBy(text: String, marks: String): List<String> {
        val out = ArrayList<String>()
        val sb = StringBuilder()
        for (c in text) {
            sb.append(c)
            if (marks.indexOf(c) >= 0) {
                val s = sb.toString().trim()
                if (s.isNotEmpty()) out += s
                sb.setLength(0)
            }
        }
        val last = sb.toString().trim()
        if (last.isNotEmpty()) out += last
        return out
    }

    /**
     * Longest prefix that stays under [maxFrames]. Never cuts inside a
     * `<|tag|>` token, and always returns at least one character.
     */
    fun hardCut(text: String, maxFrames: Int, est: (String) -> Int): List<String> {
        var rest = text
        val out = ArrayList<String>()
        while (rest.isNotEmpty()) {
            var lo = 1
            var hi = rest.length
            while (lo < hi) {
                val mid = (lo + hi + 1) / 2
                if (est(rest.substring(0, mid)) <= maxFrames) lo = mid else hi = mid - 1
            }
            var cut = lo
            // Step back out of a tag that the binary search landed inside.
            val open = rest.lastIndexOf("<|", cut)
            if (open >= 0 && rest.indexOf("|>", open) >= cut) cut = open
            if (cut <= 0) cut = 1
            out += rest.substring(0, cut)
            rest = rest.substring(cut)
        }
        return out
    }
}
