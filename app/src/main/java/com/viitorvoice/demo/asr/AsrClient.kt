package com.viitorvoice.demo.asr

import android.content.Context
import android.net.Uri
import android.os.Bundle
import java.io.File

/**
 * Editor-side entry point. Calls the `:asr` provider; if that process is killed
 * while loading the aligner, falls back to the SenseVoice-only result it wrote
 * to `partial.json` beforehand.
 */
object AsrClient {

    fun transcribe(ctx: Context, wav: File): List<AsrPipeline.Seg> {
        val cache = File(ctx.cacheDir, "asr")
        cache.mkdirs()
        val partial = File(cache, "partial.json")
        partial.delete()

        val uri = Uri.parse("content://${ctx.packageName}.asr")
        val args = Bundle().apply { putString("wav", wav.absolutePath) }
        val full = try {
            ctx.contentResolver.call(uri, "transcribe", null, args)?.getString("json")
        } catch (t: Throwable) {
            null
        }
        val segs = AsrJson.read(full ?: "")
        if (segs.isNotEmpty()) return segs
        return AsrJson.read(if (partial.isFile) partial.readText() else "")
    }
}
