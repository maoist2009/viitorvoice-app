package com.viitorvoice.modellib

import android.content.Context
import android.net.Uri
import android.os.Bundle
import android.os.ParcelFileDescriptor
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Reference client for the shared model library (MLP/1, see PROTOCOL.md).
 *
 * Drop this file into any app that should read (or contribute models to) the
 * library. It holds no state, no threads and no background work: call it from
 * whatever coroutine already owns the model-loading code.
 *
 * The provider hands out read-only fds, so a consumer can mmap a multi-GB
 * context binary without copying it and without ever learning the host's
 * private path.
 */
class ModelLibrary(private val ctx: Context) {

    data class Entry(
        val set: String,
        val name: String,
        val arch: String,
        val sha256: String,
        val size: Long,
        val tags: List<String>,
    )

    data class Published(val sha256: String, val size: Long, val deduped: Boolean)

    /** Installed host package, or null when nothing on this device serves the library. */
    val hostPkg: String by lazy {
        CANDIDATES.firstOrNull { pkg ->
            runCatching { ctx.packageManager.resolveContentProvider("$pkg.models", 0) }.getOrNull() != null
        }.orEmpty()
    }

    val available: Boolean get() = hostPkg.isNotEmpty()

    private val authority: Uri? get() = hostPkg.takeIf { it.isNotEmpty() }?.let { Uri.parse("content://$it.models") }

    // ------------------------------------------------------------- reading

    /** Whole registry; empty when the host is absent or the call fails. */
    fun catalog(): List<Entry> {
        val auth = authority ?: return emptyList()
        val json = runCatching {
            ctx.contentResolver.call(auth, "catalog", null, null)?.getString("json")
        }.getOrNull() ?: return emptyList()
        val arr = runCatching { JSONObject(json).getJSONArray("entries") }.getOrNull() ?: return emptyList()
        return (0 until arr.length()).mapNotNull { i ->
            runCatching { arr.getJSONObject(i) }.getOrNull()?.let { e ->
                Entry(
                    set = e.getString("set"),
                    name = e.getString("name"),
                    arch = e.optString("arch", "any"),
                    sha256 = e.getString("sha256"),
                    size = e.optLong("size", -1),
                    tags = e.optJSONArray("tags")?.let { a -> (0 until a.length()).map { a.getString(it) } }
                        .orEmpty(),
                )
            }
        }
    }

    /** Exact arch first, then the arch-independent copy (`any`). */
    fun resolve(set: String, name: String, arch: String? = null): Entry? {
        val all = catalog().filter { it.set == set && it.name == name }
        return all.firstOrNull { arch != null && it.arch == arch }
            ?: all.firstOrNull { it.arch == "any" }
            ?: all.firstOrNull()
    }

    fun openFd(entry: Entry): ParcelFileDescriptor? {
        val auth = authority ?: return null
        return runCatching {
            ctx.contentResolver.openFileDescriptor(Uri.parse("$auth/file/${entry.set}/${entry.name}?arch=${entry.arch}"), "r")
        }.getOrNull()
    }

    fun openFd(set: String, name: String, arch: String? = null): ParcelFileDescriptor? =
        resolve(set, name, arch)?.let { openFd(it) }

    fun bytes(set: String, name: String, arch: String? = null): ByteArray? =
        openFd(set, name, arch)?.use { pfd ->
            ParcelFileDescriptor.AutoCloseInputStream(pfd).use { it.readBytes() }
        }

    // ------------------------------------------------------------- publishing

    /**
     * Contribute a local file to the shared library. The host re-hashes it, so
     * [declaredSha] is only a consistency check.
     */
    fun publish(
        file: File,
        set: String,
        name: String,
        arch: String,
        tags: List<String> = emptyList(),
        replace: Boolean = false,
        declaredSha: String? = null,
    ): Published? {
        val auth = authority ?: return null
        val pfd = ParcelFileDescriptor.open(file, ParcelFileDescriptor.MODE_READ_ONLY)
        return pfd.use {
            val extras = Bundle().apply {
                putString("set", set)
                putString("name", name)
                putString("arch", arch)
                putStringArray("tags", tags.toTypedArray())
                putBoolean("replace", replace)
                declaredSha?.let { putString("sha256", it) }
                putParcelable("fd", it)
            }
            val r = runCatching { ctx.contentResolver.call(auth, "publish", null, extras) }.getOrNull()
                ?: return null
            Published(
                sha256 = r.getString("sha256")!!,
                size = r.getLong("size"),
                deduped = r.getBoolean("deduped"),
            )
        }
    }

    fun unpublish(set: String, name: String, arch: String): Boolean {
        val auth = authority ?: return false
        val extras = Bundle().apply {
            putString("set", set); putString("name", name); putString("arch", arch)
        }
        return runCatching {
            ctx.contentResolver.call(auth, "unpublish", null, extras)?.getBoolean("removed") ?: false
        }.getOrDefault(false)
    }

    fun stats(): Triple<Int, Int, Long>? {
        val auth = authority ?: return null
        val r = runCatching { ctx.contentResolver.call(auth, "stats", null, null) }.getOrNull() ?: return null
        return Triple(r.getInt("entries"), r.getInt("blobs"), r.getLong("bytes"))
    }

    companion object {
        /**
         * Debug builds install as `<pkg>.tmp`; both are signed with the same
         * key, so both hold the signature permission.
         */
        val CANDIDATES = listOf(
            "com.viitorvoice.modelhost.tmp",
            "com.viitorvoice.modelhost",
        )

        /** Parse a JSON array of set names — used by tests / tooling. */
        fun sets(json: String): List<String> =
            runCatching {
                val arr = JSONObject(json).getJSONArray("entries") as JSONArray
                (0 until arr.length()).map { arr.getJSONObject(it).getString("set") }.distinct()
            }.getOrDefault(emptyList())
    }
}
