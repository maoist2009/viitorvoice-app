package com.viitorvoice.demo.asr

import org.json.JSONObject
import java.io.File

/**
 * Qwen2 byte-level BPE + Qwen3-ForcedAligner 的切词/时间戳修复。
 *
 * 端上只需要“一条文本 -> 完整 aligner prompt 的 token id”这一件事，
 * 特殊 token（audio_start / audio_pad / audio_end / timestamp）直接按 id 插入，
 * 因此不必实现 added-token 切分——已经用 Python 侧逐条比对验证过等价性
 * （见 tools/aligner/make_toktests.py 生成的 tok_tests.json）。
 */
class AlignerTokenizer(vocabFile: File, mergesFile: File) {

    private val vocab = HashMap<String, Int>(1 shl 19)
    private val ranks = HashMap<String, Int>(1 shl 18)
    private val cache = HashMap<String, IntArray>(1 shl 12)
    private val byteToChar = CharArray(256)

    init {
        val o = JSONObject(vocabFile.readText())
        val it = o.keys()
        while (it.hasNext()) {
            val k = it.next()
            vocab[k] = o.getInt(k)
        }
        var r = 0
        mergesFile.forEachLine { line ->
            if (line.isNotEmpty() && !line.startsWith("#")) {
                ranks[line] = r++
            }
        }
        // GPT-2 bytes_to_unicode
        val bs = ArrayList<Int>(256)
        for (b in 33..126) bs.add(b)
        for (b in 161..172) bs.add(b)
        for (b in 174..255) bs.add(b)
        val cs = ArrayList<Int>(bs)
        var n = 0
        for (b in 0..255) {
            if (!bs.contains(b)) {
                bs.add(b)
                cs.add(256 + n)
                n++
            }
        }
        for (i in bs.indices) byteToChar[bs[i]] = cs[i].toChar()
    }

    /** 一条文本（不含特殊 token）-> token id。 */
    fun tokenize(text: String): IntArray {
        val out = ArrayList<Int>(text.length + 4)
        for (m in PAT.findAll(text)) {
            val piece = m.value
            val uni = StringBuilder(piece.length)
            for (b in piece.toByteArray(Charsets.UTF_8)) {
                uni.append(byteToChar[b.toInt() and 0xff])
            }
            val key = uni.toString()
            val hit = cache[key]
            if (hit != null) {
                out.addAll(hit.toList())
            } else {
                val ids = bpe(key)
                cache[key] = ids
                out.addAll(ids.toList())
            }
        }
        return out.toIntArray()
    }

    private fun bpe(token: String): IntArray {
        var sym = ArrayList<String>(token.length)
        for (c in token) sym.add(c.toString())
        while (sym.size > 1) {
            var bestRank = Int.MAX_VALUE
            var at = -1
            for (i in 0 until sym.size - 1) {
                val r = ranks[sym[i] + " " + sym[i + 1]] ?: continue
                if (r < bestRank) {
                    bestRank = r
                    at = i
                }
            }
            if (at < 0) break
            val first = sym[at]
            val second = sym[at + 1]
            val merged = ArrayList<String>(sym.size)
            var i = 0
            while (i < sym.size) {
                if (i < sym.size - 1 && sym[i] == first && sym[i + 1] == second) {
                    merged.add(first + second)
                    i += 2
                } else {
                    merged.add(sym[i])
                    i++
                }
            }
            sym = merged
        }
        return IntArray(sym.size) { vocab[sym[it]] ?: 0 }
    }

    // ---- Qwen3ForceAlignProcessor.encode_timestamp 的 Kotlin 版 ----

    fun splitWords(text: String): List<String> {
        val words = ArrayList<String>()
        for (seg in text.split(WHITESPACE)) {
            if (seg.isEmpty()) continue
            val cleaned = seg.filter { isKept(it) }
            if (cleaned.isEmpty()) continue
            val buf = StringBuilder()
            for (ch in cleaned) {
                if (isCjk(ch)) {
                    if (buf.isNotEmpty()) {
                        words.add(buf.toString())
                        buf.setLength(0)
                    }
                    words.add(ch.toString())
                } else {
                    buf.append(ch)
                }
            }
            if (buf.isNotEmpty()) words.add(buf.toString())
        }
        return words
    }

    /** 返回 (words, 完整 prompt ids)，ids 里含 1 个 audio_pad。 */
    fun buildPrompt(text: String): Pair<List<String>, IntArray> {
        val words = splitWords(text)
        val ids = ArrayList<Int>(words.size * 6 + 4)
        ids.add(AUDIO_START)
        ids.add(AUDIO_PAD)
        ids.add(AUDIO_END)
        for (w in words) {
            for (id in tokenize(w)) ids.add(id)
            ids.add(TIMESTAMP)
            ids.add(TIMESTAMP)
        }
        return Pair(words, ids.toIntArray())
    }

    /**
     * 把每对 (start,end) 的原始 argmax 时刻修成单调不减（原实现是 LIS + 插值）。
     * data 是 token 序号，返回同样长度。
     */
    fun fixTimestamp(data: IntArray): IntArray {
        val n = data.size
        if (n == 0) return data
        val dp = IntArray(n) { 1 }
        val parent = IntArray(n) { -1 }
        for (i in 1 until n) {
            for (j in 0 until i) {
                if (data[j] <= data[i] && dp[j] + 1 > dp[i]) {
                    dp[i] = dp[j] + 1
                    parent[i] = j
                }
            }
        }
        var maxIdx = 0
        for (i in 1 until n) if (dp[i] > dp[maxIdx]) maxIdx = i
        val normal = BooleanArray(n)
        var idx = maxIdx
        while (idx != -1) {
            normal[idx] = true
            idx = parent[idx]
        }

        val res = data.copyOf()
        var i = 0
        while (i < n) {
            if (normal[i]) {
                i++
                continue
            }
            var j = i
            while (j < n && !normal[j]) j++
            val count = j - i
            var left: Int? = null
            for (k in i - 1 downTo 0) if (normal[k]) { left = res[k]; break }
            var right: Int? = null
            for (k in j until n) if (normal[k]) { right = res[k]; break }
            if (count <= 2) {
                for (k in i until j) {
                    res[k] = when {
                        left == null && right == null -> res[k]
                        left == null -> right!!
                        right == null -> left!!
                        (k - (i - 1)) <= (j - k) -> left
                        else -> right
                    }
                }
            } else if (left != null && right != null) {
                val step = (right - left).toDouble() / (count + 1)
                for (k in i until j) res[k] = (left + step * (k - i + 1)).toInt()
            } else if (left != null) {
                for (k in i until j) res[k] = left!!
            } else if (right != null) {
                for (k in i until j) res[k] = right!!
            }
            i = j
        }
        return res
    }

    private fun isKept(ch: Char): Boolean {
        if (ch == '\'') return true
        return when (Character.getType(ch).toByte()) {
            Character.UPPERCASE_LETTER, Character.LOWERCASE_LETTER, Character.TITLECASE_LETTER,
            Character.MODIFIER_LETTER, Character.OTHER_LETTER,
            Character.DECIMAL_DIGIT_NUMBER, Character.LETTER_NUMBER, Character.OTHER_NUMBER ->
                true
            else -> false
        }
    }

    private fun isCjk(ch: Char): Boolean {
        val c = ch.code
        return (c in 0x4E00..0x9FFF) || (c in 0x3400..0x4DBF) || (c in 0xF900..0xFAFF)
    }

    companion object {
        const val AUDIO_START = 151669
        const val AUDIO_PAD = 151676
        const val AUDIO_END = 151670
        const val TIMESTAMP = 151705

        private val WHITESPACE = Regex("\\s+")

        /** transformers 4.57 Qwen2Tokenizer.PRETOKENIZE_REGEX（逐字符一致） */
        private val PAT = Regex(
            "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}" +
                "| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
        )
    }
}
