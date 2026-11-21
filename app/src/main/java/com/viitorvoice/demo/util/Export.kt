package com.viitorvoice.demo.util

import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Environment
import android.provider.MediaStore
import java.io.File

/**
 * Copy a finished take into the shared Music collection and hand it to the
 * system share sheet. minSdk is 31, so `RELATIVE_PATH` + `IS_PENDING` work
 * without any storage permission.
 */
object Export {

    fun toMusic(ctx: Context, src: File, title: String): Uri? {
        val name = "${slug(title)}_${System.currentTimeMillis()}.wav"
        val values = ContentValues().apply {
            put(MediaStore.Audio.Media.DISPLAY_NAME, name)
            put(MediaStore.Audio.Media.MIME_TYPE, "audio/wav")
            put(MediaStore.Audio.Media.RELATIVE_PATH, Environment.DIRECTORY_MUSIC + "/ViiTorVoice")
            put(MediaStore.Audio.Media.IS_PENDING, 1)
        }
        val res = ctx.contentResolver
        val uri = res.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, values) ?: return null
        return runCatching {
            res.openOutputStream(uri)?.use { out ->
                src.inputStream().use { it.copyTo(out) }
            } ?: error("openOutputStream failed")
            values.clear()
            values.put(MediaStore.Audio.Media.IS_PENDING, 0)
            res.update(uri, values, null, null)
            uri
        }.getOrElse {
            res.delete(uri, null, null)
            null
        }
    }

    fun share(ctx: Context, uri: Uri, title: String) {
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "audio/wav"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, title)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        ctx.startActivity(
            Intent.createChooser(send, "分享").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        )
    }

    private fun slug(s: String): String =
        s.trim().replace(Regex("[^\\p{L}\\p{N}]+"), "_").take(30).ifBlank { "take" }
}
