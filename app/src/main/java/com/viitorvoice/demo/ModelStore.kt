package com.viitorvoice.demo

import android.app.Application
import android.os.ParcelFileDescriptor
import com.viitorvoice.modellib.ModelLibrary
import java.io.File

/**
 * Bridge to the shared model library (MLP/1). See
 * ~/projects/viitorvoice/modelhost/PROTOCOL.md.
 *
 * Two roles, both optional at runtime:
 *
 *  - **consumer** — [bind] opens every blob of a set, detaches the fds and hands
 *    them to the native virtual-path table (mstore). `Engine` keeps calling
 *    `qnn_load(modelsDir + "/llm.bin")`; the open resolves to the library, so no
 *    private copy of the 2.7 GB set has to exist.
 *  - **producer** — [publish] contributes a directory of local .bin files back
 *    into the library (content-addressed: re-publishing identical bytes is a
 *    no-op), which is how a private `filesDir/models` gets migrated once.
 *
 * Nothing here runs on its own: the caller decides when to pay the hashing cost.
 */
object ModelStore {

    /** The compiled NAR + encoder + decoder graphs, all in one v79 set. */
    const val NAR_SET = "viitorvoice-nar"
    const val NAR_ARCH = "v79"

    private var cached: ModelLibrary? = null

    fun library(app: Application): ModelLibrary =
        cached ?: ModelLibrary(app).also { cached = it }

    fun available(app: Application): Boolean = library(app).available

    /** True when every one of [names] is registered for the set. */
    fun has(app: Application, set: String, names: List<String>): Boolean {
        val lib = library(app)
        if (!lib.available) return false
        val cat = lib.catalog()
        return names.all { n -> cat.any { it.set == set && it.name == n } }
    }

    /**
     * Bind [names] from [set] into the native virtual-path table. Returns "" on
     * success, or a short error that the caller can show instead of "缺少模型".
     */
    fun bind(app: Application, set: String, arch: String, names: List<String>): String {
        val lib = library(app)
        if (!lib.available) return "未安装共享模型库"
        val suffixes = ArrayList<String>(names.size)
        val fds = ArrayList<Int>(names.size)
        for (name in names) {
            val pfd = lib.openFd(set, name, arch)
            if (pfd == null) {
                closeAll(fds)
                return "共享库缺 $set/$name"
            }
            suffixes += "/$name"
            fds += pfd.detachFd()
        }
        val bound = Native.nativeBindModelFds(suffixes.toTypedArray(), fds.toIntArray())
        return if (bound == names.size) "" else {
            Native.nativeClearModelFds()
            "fd 绑定失败 $bound/${names.size}"
        }
    }

    /**
     * Producer: publish [files] as [set]/[arch]. Content-addressed, so files
     * already in the library are deduped instead of copied twice.
     */
    fun publish(app: Application, set: String, arch: String, files: List<File>): String {
        val lib = library(app)
        if (!lib.available) return "未安装共享模型库"
        if (files.isEmpty()) return "没有要发布的文件"
        var done = 0
        var deduped = 0
        val failed = ArrayList<String>()
        for (f in files) {
            val r = lib.publish(f, set, f.name, arch, replace = true)
            if (r == null) failed += f.name else {
                done++
                if (r.deduped) deduped++
            }
        }
        return if (failed.isEmpty()) {
            "已发布 $done 个（去重 $deduped）"
        } else {
            "发布 $done 个，失败 ${failed.size}: ${failed.joinToString()}"
        }
    }

    /** "12 条 · 2.7 GB"-style one-liner for the status card. */
    fun summary(app: Application): String {
        val lib = library(app)
        if (!lib.available) return "未安装"
        val cat = lib.catalog()
        if (cat.isEmpty()) return "空"
        val mb = cat.sumOf { it.size.coerceAtLeast(0) } / (1024.0 * 1024.0)
        return "${cat.size} 条 · ${"%.1f".format(mb)} MB"
    }

    private fun closeAll(fds: List<Int>) {
        for (fd in fds) runCatching { ParcelFileDescriptor.adoptFd(fd).close() }
    }
}
