package com.viitorvoice.demo.asr

import org.json.JSONArray
import org.json.JSONObject

/** JSON wire format between the `:asr` process and the editor process. */
object AsrJson {

    fun write(segs: List<AsrPipeline.Seg>): String {
        val a = JSONArray()
        for (s in segs) {
            val o = JSONObject()
                .put("s", s.startMs)
                .put("e", s.endMs)
                .put("t", s.text)
            val w = JSONArray()
            for (x in s.words) {
                w.put(
                    JSONObject()
                        .put("t", x.text)
                        .put("s", x.startMs)
                        .put("e", x.endMs),
                )
            }
            o.put("w", w)
            a.put(o)
        }
        return a.toString()
    }

    fun read(json: String): List<AsrPipeline.Seg> {
        if (json.isBlank()) return emptyList()
        return runCatching {
            val a = JSONArray(json)
            buildList {
                for (i in 0 until a.length()) {
                    val o = a.getJSONObject(i)
                    val w = o.optJSONArray("w")
                    val words = if (w == null) emptyList() else buildList {
                        for (k in 0 until w.length()) {
                            val x = w.getJSONObject(k)
                            add(AsrPipeline.Word(x.getString("t"), x.getLong("s"), x.getLong("e")))
                        }
                    }
                    add(
                        AsrPipeline.Seg(
                            o.getLong("s"),
                            o.getLong("e"),
                            o.optString("t"),
                            words,
                        ),
                    )
                }
            }
        }.getOrDefault(emptyList())
    }
}
