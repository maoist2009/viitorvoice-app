package com.viitorvoice.demo.asr

import android.content.Context
import android.os.ParcelFileDescriptor
import android.system.Os
import com.viitorvoice.modellib.ModelLibrary
import java.io.File

/**
 * Binds the ASR model sets from the shared model library into three virtual
 * directories (`filesDir/asr/{models,vad,aligner}`) built from
 * `/proc/self/fd/<n>` symlinks.
 *
 * The native engines only know `fopen(dir + "/sv.bin")`, so this needs no
 * native changes and no 2 GB copy. The fds stay open for the process lifetime;
 * after a process restart [bind] simply runs again.
 */
object AsrModels {

    data class Dirs(val models: File, val vad: File, val aligner: File)

    private data class Spec(val set: String, val name: String, val local: String = name)

    private val SV = listOf(
        Spec("sensevoice-small", "sv.bin"),
        Spec("sensevoice-small", "sv_tokens.bin"),
        Spec("sensevoice-small", "sv_cmvn.bin"),
        Spec("sensevoice-small", "sv_prefix.bin"),
    )

    private val VAD = listOf(
        Spec("silero-vad", "silero_vad_k16.bin", "silero_vad.bin"),
    )

    private val ALIGNER = listOf(
        Spec("qwen3-aligner-0.6b", "enc.bin"),
        Spec("qwen3-aligner-0.6b", "dec.bin"),
        Spec("qwen3-aligner-0.6b", "token_embeds.bin"),
        Spec("qwen3-aligner-0.6b", "vocab.json"),
        Spec("qwen3-aligner-0.6b", "merges.txt"),
    )

    private val keep = ArrayList<ParcelFileDescriptor>()

    fun bind(ctx: Context, onLog: (String) -> Unit = {}): Dirs? {
        val lib = ModelLibrary(ctx)
        if (!lib.available) {
            onLog("没有共享模型库（ASR 不可用）")
            return null
        }
        val root = File(ctx.filesDir, "asr")
        val models = materialize(lib, root, "models", SV, onLog) ?: return null
        val vad = materialize(lib, root, "vad", VAD, onLog) ?: return null
        val aligner = materialize(lib, root, "aligner", ALIGNER, onLog) ?: return null
        return Dirs(models, vad, aligner)
    }

    private fun materialize(
        lib: ModelLibrary,
        root: File,
        group: String,
        specs: List<Spec>,
        onLog: (String) -> Unit,
    ): File? {
        val dir = File(root, group)
        if (!dir.isDirectory && !dir.mkdirs()) {
            onLog("建目录失败 ${dir.absolutePath}")
            return null
        }
        dir.listFiles()?.forEach { it.delete() }
        for (s in specs) {
            val pfd = lib.openFd(s.set, s.name)
            if (pfd == null) {
                onLog("共享库缺 ${s.set}/${s.name}")
                return null
            }
            keep.add(pfd)
            val link = File(dir, s.local)
            try {
                Os.symlink("/proc/self/fd/${pfd.fd}", link.absolutePath)
            } catch (e: Throwable) {
                onLog("链接 ${s.local} 失败：${e.message ?: e}")
                return null
            }
        }
        return dir
    }
}
