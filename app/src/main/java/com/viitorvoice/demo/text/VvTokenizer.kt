package com.viitorvoice.demo.text

import android.util.Log
import org.json.JSONObject
import java.text.Normalizer

/**
 * The checkpoint's HuggingFace tokenizer, reimplemented for the device.
 *
 * It is a SentencePiece-style BPF fed by a chain of pre-tokenizers, so the
 * pieces are not what a plain word-level BPE would give: the metaspace mark `▁`
 * is emitted as its own token (id 10558) and every Han/Kana/Hangul character and
 * every punctuation/symbol character is isolated before the merges run.
 *
 * Reproducing this exactly matters: the prompt is part of the model input, and a
 * single stray token shifts every position after it.
 *
 * usage: val t = VvTokenizer(assetText); t.encode("你好，世界。")
 */
class VvTokenizer(jsonText: String) {

    private val vocab = HashMap<String, Int>(1 shl 17)
    private val ranks = HashMap<String, Int>(1 shl 16)
    private val added = HashMap<String, Int>(4096)     // special tokens -> id
    private val cache = HashMap<String, IntArray>(4096)

    /**
     * Added tokens that occur as literal text, indexed by first char and sorted
     * longest-first (HF's AddedVocabulary matches greedily, anywhere in the
     * text). This is what carries the 18 NVV tags: `<|nv-Cough|>` is rewritten
     * to `(coughs)` by [Tags.normalizeNvv], and `(coughs)` is *one* token
     * (id 52678). Splitting it as BPE would feed the model the English word
     * "coughs" — it then reads the word out loud instead of coughing.
     */
    private val literal = HashMap<Char, List<Pair<String, Int>>>(64)

    init {
        val root = JSONObject(jsonText)
        val model = root.getJSONObject("model")
        val vocabObj = model.getJSONObject("vocab")
        vocabObj.keys().forEach { vocab[it] = vocabObj.getInt(it) }
        val merges = model.getJSONArray("merges")
        for (i in 0 until merges.length()) {
            val m = merges.get(i)
            val a: String
            val b: String
            if (m is String) {
                val p = m.split(' ')
                a = p[0]; b = p[1]
            } else {
                a = (m as org.json.JSONArray).getString(0)
                b = m.getString(1)
            }
            ranks["$a\u0000$b"] = i
        }
        val at = root.optJSONArray("added_tokens")
        if (at != null) {
            for (i in 0 until at.length()) {
                val o = at.getJSONObject(i)
                val c = o.getString("content")
                added[c] = o.getInt("id")
                if (!vocab.containsKey(c)) vocab[c] = o.getInt("id")
            }
        }
        val byFirst = HashMap<Char, MutableList<Pair<String, Int>>>(64)
        for ((c, id) in added) {
            if (c.isEmpty() || c.startsWith("<|")) continue   // <|...|> goes through specialEnd()
            byFirst.getOrPut(c[0]) { ArrayList() }.add(c to id)
        }
        for ((k, v) in byFirst) literal[k] = v.sortedByDescending { it.first.length }
        Log.i(TAG, "loaded ${vocab.size} pieces, ${ranks.size} merges, " +
            "${added.size} specials, ${literal.values.sumOf { it.size }} literal")
    }

    /** Longest literal added token at [at], or null. */
    private fun literalAt(text: String, at: Int): Pair<String, Int>? {
        val cands = literal[text[at]] ?: return null
        for (c in cands) if (text.startsWith(c.first, at)) return c
        return null
    }

    /** Full text -> token ids, special tokens passed through unchanged. */
    fun encode(text: String): IntArray {
        val out = ArrayList<Int>(text.length + 8)
        var i = 0
        val n = text.length
        while (i < n) {
            val lit = literalAt(text, i)
            if (lit != null) {
                out.add(lit.second)
                i += lit.first.length
                continue
            }
            val end = specialEnd(text, i)
            if (end > i) {
                val tag = text.substring(i, end)
                val id = added[tag] ?: vocab[tag]
                if (id != null) {
                    out.add(id)
                    i = end
                    continue
                }
            }
            var j = i
            while (j < n && specialEnd(text, j) <= j && literalAt(text, j) == null) j++
            out.addAll(tokenizePlain(text.substring(i, j)).toList())
            i = j
        }
        return out.toIntArray()
    }

    /** True when [token] is a single (added) token of this vocabulary. */
    fun has(token: String): Boolean = added.containsKey(token)

    /** Index just past a `<|...|>` tag starting at [at], or [at] if none. */
    private fun specialEnd(text: String, at: Int): Int {
        if (at + 2 > text.length || text[at] != '<' || text[at + 1] != '|') return at
        val close = text.indexOf("|>", at + 2)
        if (close < 0) return at
        return close + 2
    }

    private fun tokenizePlain(segment: String): IntArray {
        if (segment.isEmpty()) return IntArray(0)
        val normalized = Normalizer.normalize(segment, Normalizer.Form.NFKC)
        val text = if (normalized.isEmpty() || normalized[0] == ' ' ||
            !needsMetaspace(normalized[0])) normalized
        else "\u2581$normalized"
        val withMarks = text.replace(' ', '\u2581')
        val out = ArrayList<Int>(withMarks.length + 4)
        val word = StringBuilder()
        fun flush() {
            if (word.isNotEmpty()) {
                out.addAll(bpe(word.toString()).toList())
                word.setLength(0)
            }
        }
        for (ch in withMarks) {
            when {
                ch == '\u2581' -> { flush(); out.addAll(bpe("\u2581").toList()) }
                isIsolated(ch) -> { flush(); out.addAll(bpe(ch.toString()).toList()) }
                else -> word.append(ch)
            }
        }
        flush()
        return out.toIntArray()
    }

    /**
     * The reference tokenizer only prefixes the metaspace mark when a segment
     * starts with a "word" character. CJK text (and 。、) is left bare, exactly
     * like `Tokenizer.encode("你好")` -> [10932, ...] with no 10558 in front.
     */
    private fun needsMetaspace(ch: Char): Boolean {
        if (ch == '。' || ch == '、') return false
        return when (Character.UnicodeScript.of(ch.code)) {
            Character.UnicodeScript.HAN,
            Character.UnicodeScript.HIRAGANA,
            Character.UnicodeScript.KATAKANA -> false
            else -> true
        }
    }

    private fun isIsolated(ch: Char): Boolean {
        if (ch == '。' || ch == '、') return true
        return when (Character.UnicodeScript.of(ch.code)) {
            Character.UnicodeScript.HAN,
            Character.UnicodeScript.HIRAGANA,
            Character.UnicodeScript.KATAKANA,
            Character.UnicodeScript.HANGUL -> true
            else -> {
                when (Character.getType(ch.code).toByte()) {
                    Character.CONNECTOR_PUNCTUATION,
                    Character.DASH_PUNCTUATION,
                    Character.START_PUNCTUATION,
                    Character.END_PUNCTUATION,
                    Character.INITIAL_QUOTE_PUNCTUATION,
                    Character.FINAL_QUOTE_PUNCTUATION,
                    Character.OTHER_PUNCTUATION,
                    Character.MATH_SYMBOL,
                    Character.CURRENCY_SYMBOL,
                    Character.MODIFIER_SYMBOL,
                    Character.OTHER_SYMBOL -> true
                    else -> false
                }
            }
        }
    }

    /** Byte-pair merges over one pre-tokenized piece; cached per piece. */
    private fun bpe(piece: String): IntArray {
        cache[piece]?.let { return it }
        vocab[piece]?.let { id ->
            val one = intArrayOf(id)
            if (piece.length > 1) cache[piece] = one
            return one
        }
        val sym = ArrayList<String>(piece.length)
        for (ch in piece) sym.add(ch.toString())
        while (sym.size > 1) {
            var bestRank = Int.MAX_VALUE
            var bestAt = -1
            for (k in 0 until sym.size - 1) {
                val r = ranks["${sym[k]}\u0000${sym[k + 1]}"] ?: continue
                if (r < bestRank) { bestRank = r; bestAt = k }
            }
            if (bestAt < 0) break
            val merged = sym[bestAt] + sym[bestAt + 1]
            sym[bestAt] = merged
            sym.removeAt(bestAt + 1)
        }
        val ids = IntArray(sym.size)
        for (k in sym.indices) {
            ids[k] = vocab[sym[k]] ?: vocab["<unk>"] ?: 1
        }
        if (piece.length > 1) cache[piece] = ids
        return ids
    }

    companion object {
        private const val TAG = "VvTokenizer"
    }
}
