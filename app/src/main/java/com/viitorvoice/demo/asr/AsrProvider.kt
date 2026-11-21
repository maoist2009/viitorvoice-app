package com.viitorvoice.demo.asr

import android.content.ContentProvider
import android.content.ContentValues
import android.database.Cursor
import android.net.Uri
import android.os.Bundle
import android.util.Log
import java.io.File

/**
 * Runs VAD + SenseVoice + aligner in the `:asr` process.
 *
 * The aligner needs ~1.5 GB of graphs and can push the system into lmkd; doing
 * it here means the editor process (and the user's transcript) survives even if
 * this process is killed. The SenseVoice result is written to `partial.json`
 * before the aligner starts, so the editor still gets text + estimated word
 * timings when only the aligner dies.
 */
class AsrProvider : ContentProvider() {

    override fun onCreate(): Boolean = true

    override fun call(method: String, arg: String?, extras: Bundle?): Bundle {
        if (method != "transcribe") return Bundle()
        val ctx = context ?: return Bundle()
        val path = extras?.getString("wav") ?: return Bundle()
        val cache = File(ctx.cacheDir, "asr")
        cache.mkdirs()
        val partial = File(cache, "partial.json")
        partial.delete()

        val segs = AsrPipeline.run(
            ctx = ctx,
            wav = File(path),
            cacheDir = cache,
            onLog = { Log.i("VVASR", it) },
            onText = { draft ->
                runCatching { partial.writeText(AsrJson.write(draft)) }
            },
        )
        return Bundle().apply { putString("json", AsrJson.write(segs)) }
    }

    override fun query(
        uri: Uri,
        projection: Array<out String>?,
        selection: String?,
        selectionArgs: Array<out String>?,
        sortOrder: String?,
    ): Cursor? = null

    override fun getType(uri: Uri): String? = null

    override fun insert(uri: Uri, values: ContentValues?): Uri? = null

    override fun delete(uri: Uri, selection: String?, selectionArgs: Array<out String>?): Int = 0

    override fun update(
        uri: Uri,
        values: ContentValues?,
        selection: String?,
        selectionArgs: Array<out String>?,
    ): Int = 0
}
