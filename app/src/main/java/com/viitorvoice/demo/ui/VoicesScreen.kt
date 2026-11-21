package com.viitorvoice.demo.ui

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.IosShare
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.viitorvoice.demo.Native
import com.viitorvoice.demo.audio.AudioIo
import com.viitorvoice.demo.audio.Player
import com.viitorvoice.demo.data.Store
import com.viitorvoice.demo.data.Voice
import com.viitorvoice.demo.util.Export
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.min

/** Voice library: import a clip, audition it, delete it. */
@Composable
fun VoicesScreen(onBack: () -> Unit) {
    val ctx = LocalContext.current
    val store = remember { Store.get(ctx) }
    val scope = rememberCoroutineScope()
    var items by remember { mutableStateOf<List<Voice>>(emptyList()) }
    var tick by remember { mutableIntStateOf(0) }
    var playingId by remember { mutableLongStateOf(-1L) }
    var note by remember { mutableStateOf<String?>(null) }
    var importing by remember { mutableStateOf(false) }

    LaunchedEffect(tick) {
        items = withContext(Dispatchers.IO) { store.voices() }
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        importing = true
        scope.launch {
            runCatching {
                withContext(Dispatchers.IO) {
                    val (pcm, rate) = AudioIo.decode(ctx, uri)
                    val mono = resample(pcm, rate, AudioIo.SR)
                    val cut = mono.copyOf(min(mono.size, 8 * 24000))
                    val f = store.newVoiceFile()
                    AudioIo.writeWav(f, cut)
                    store.addVoice(displayName(ctx, uri), f.absolutePath, "", "zh", "")
                }
                note = "已导入"
            }.onFailure { note = "导入失败: ${it.message}" }
            importing = false
            tick++
        }
    }

    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        LazyColumn(
            modifier = Modifier.fillMaxSize().statusBarsPadding(),
            contentPadding = PaddingValues(start = 20.dp, end = 20.dp, top = 6.dp, bottom = 20.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            item {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = onBack) {
                        Icon(Icons.Default.ArrowBack, contentDescription = "返回")
                    }
                    Column(Modifier.weight(1f)) {
                        Text("音色", style = MaterialTheme.typography.headlineMedium)
                        Text(
                            if (items.isEmpty()) "导入一段 1–5 秒的清晰人声"
                            else "${items.size} 个音色 · 转写/语言可选",
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    if (importing) {
                        Text("导入中…", style = MaterialTheme.typography.labelSmall)
                    } else {
                        IconButton(onClick = { picker.launch(arrayOf("audio/*")) }) {
                            Icon(Icons.Default.Add, contentDescription = "导入音色")
                        }
                    }
                }
                note?.let {
                    Spacer(Modifier.height(6.dp))
                    Text(it, style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary)
                }
            }

            if (items.isEmpty()) {
                item {
                    Text(
                        "还没有音色。点右上角 + 导入一段参考音频。",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 20.dp),
                    )
                }
            }

            items(items, key = { it.id }) { v ->
                VoiceCard(
                    voice = v,
                    playing = playingId == v.id,
                    onPlay = {
                        if (playingId == v.id) {
                            Player.stop()
                            playingId = -1L
                        } else {
                            runCatching { Player.play(File(v.wavPath)) }
                                .onSuccess { playingId = v.id }
                                .onFailure { note = "播放失败: ${it.message}" }
                        }
                    },
                    onExport = {
                        val uri = Export.toMusic(ctx, File(v.wavPath), v.name)
                        if (uri == null) note = "导出失败"
                        else {
                            note = "已导出到 Music/ViiTorVoice"
                            Export.share(ctx, uri, v.name)
                        }
                    },
                    onDelete = {
                        if (playingId == v.id) {
                            Player.stop()
                            playingId = -1L
                        }
                        store.deleteVoice(v.id)
                        tick++
                    },
                )
            }
        }
    }
}

@Composable
private fun VoiceCard(
    voice: Voice,
    playing: Boolean,
    onPlay: () -> Unit,
    onExport: () -> Unit,
    onDelete: () -> Unit,
) {
    Surface(
        shape = RoundedCornerShape(20.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.72f),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.14f)),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(
            Modifier.padding(14.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(Modifier.weight(1f)) {
                Text(
                    voice.name,
                    style = MaterialTheme.typography.bodyMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    "%.2f s · %s".format(wavSec(voice.wavPath), voice.lang.ifBlank { "语言未定" }),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            IconButton(onClick = onPlay) {
                Icon(
                    if (playing) Icons.Default.Stop else Icons.Default.PlayArrow,
                    contentDescription = "试听",
                    tint = MaterialTheme.colorScheme.primary,
                )
            }
            IconButton(onClick = onExport) {
                Icon(Icons.Default.IosShare, contentDescription = "导出", modifier = Modifier.size(20.dp))
            }
            IconButton(onClick = onDelete) {
                Icon(
                    Icons.Default.Delete, contentDescription = "删除",
                    modifier = Modifier.size(20.dp),
                    tint = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

private fun wavSec(path: String): Float {
    val n = File(path).length() - 44
    return if (n <= 0) 0f else n / 2f / 24000f
}

private fun resample(pcm: FloatArray, srIn: Int, srOut: Int): FloatArray {
    if (srIn == srOut) return pcm
    val cap = (pcm.size.toLong() * srOut / srIn).toInt() + 8
    val dst = FloatArray(cap)
    val n = Native.nativeResample(pcm, pcm.size, srIn, srOut, dst)
    require(n > 0) { "重采样失败" }
    return dst.copyOf(n)
}

private fun displayName(ctx: Context, uri: Uri): String {
    runCatching {
        ctx.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)
            ?.use { c ->
                if (c.moveToFirst()) {
                    val n = c.getString(0) ?: ""
                    if (n.isNotBlank()) return n.substringBeforeLast('.')
                }
            }
    }
    return "音色"
}
