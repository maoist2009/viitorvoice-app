package com.viitorvoice.demo.ui

import android.app.Activity
import android.content.res.Configuration
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.viitorvoice.demo.data.Line
import com.viitorvoice.demo.data.LineStatus
import com.viitorvoice.demo.text.Duration
import kotlinx.coroutines.launch

private data class GenRequest(val ids: List<Long>, val force: Boolean, val cfg: Float? = null)

/** Workspace: docked timeline, compact sentence rows, one transport bar. */
@Composable
fun EditorScreen(vm: StudioViewModel, projectId: Long, onBack: () -> Unit) {
    val state by vm.editor.collectAsStateWithLifecycle()
    val voices by vm.voices.collectAsStateWithLifecycle()
    val playhead by vm.playhead.collectAsStateWithLifecycle()
    val dev by vm.devMode.collectAsStateWithLifecycle()
    val confirm by vm.confirmGen.collectAsStateWithLifecycle()
    val ctx = LocalContext.current
    val landscape = LocalConfiguration.current.orientation == Configuration.ORIENTATION_LANDSCAPE
    var fullTimeline by rememberSaveable {
        mutableStateOf(landscape || (ctx as? Activity)?.intent?.getIntExtra("timeline", 0) == 1)
    }
    var autoGen by rememberSaveable {
        mutableStateOf((ctx as? Activity)?.intent?.getIntExtra("gen", 0) == 1)
    }

    LaunchedEffect(projectId) { vm.openProject(projectId) }

    val st = state
    if (st == null) {
        Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
            CircularProgressIndicator()
        }
        return
    }

    var collapsed by rememberSaveable { mutableStateOf(false) }
    var selectedId by remember { mutableStateOf<Long?>(null) }
    var selLine by remember { mutableStateOf<Long?>(null) }
    var selStart by remember { mutableStateOf(0) }
    var selEnd by remember { mutableStateOf(0) }
    var editOpen by remember { mutableStateOf(false) }
    var inspectId by remember { mutableStateOf<Long?>(null) }
    var showVoicePicker by remember { mutableStateOf(false) }
    var renameOpen by remember { mutableStateOf(false) }
    var pendingGen by remember { mutableStateOf<GenRequest?>(null) }
    val listState = rememberLazyListState()
    val scope = rememberCoroutineScope()
    val snackbar = remember { SnackbarHostState() }

    val segs = remember(st.lines) { buildSegments(st.lines) }
    val totalSec = segs.lastOrNull()?.let { it.start + it.dur } ?: 0f

    LaunchedEffect(st.error) {
        st.error?.let {
            snackbar.showSnackbar(it)
            vm.clearError()
        }
    }

    LaunchedEffect(st.project.id) {
        if (autoGen) {
            autoGen = false
            val ids = vm.pendingIds()
            if (ids.isNotEmpty()) {
                if (confirm) pendingGen = GenRequest(ids, false) else vm.generate(ids)
            }
        }
    }

    fun askGen(ids: List<Long>, force: Boolean) {
        if (ids.isEmpty() || st.busy) return
        if (confirm) pendingGen = GenRequest(ids, force) else vm.generate(ids, force)
    }

    BackHandler {
        when {
            selEnd > selStart -> { selLine = null; selStart = 0; selEnd = 0 }
            fullTimeline -> fullTimeline = false
            else -> {
                vm.closeProject()
                onBack()
            }
        }
    }

    if (fullTimeline) {
        TimelineScreen(st, playhead, vm, onBack = { fullTimeline = false })
        return
    }

    val selectClip: (Long) -> Unit = { id ->
        selectedId = id
        val idx = st.lines.indexOfFirst { it.id == id }
        if (idx >= 0) scope.launch { listState.animateScrollToItem(idx) }
    }

    val dubbing = st.project.mediaKind != "none" || st.lines.any { it.sourceWav.isNotBlank() }
    val voiceName = st.project.voiceId
        ?.let { id -> voices.firstOrNull { it.id == id }?.name }
        ?: if (dubbing) "原音配音" else "选音色"
    val pendingCount = st.lines.count {
        it.text.isNotBlank() &&
            (it.status == LineStatus.DIRTY || it.status == LineStatus.FAILED ||
                it.status == LineStatus.EMPTY)
    }

    Scaffold(
        modifier = Modifier.imePadding(),
        containerColor = MaterialTheme.colorScheme.background,
        snackbarHost = { SnackbarHost(snackbar) },
        topBar = {
            EditorBar(
                title = st.project.title,
                subtitle = "${st.lines.size} 句 · $voiceName",
                voiceMissing = st.project.voiceId == null && !dubbing,
                onBack = { vm.closeProject(); onBack() },
                onRename = { renameOpen = true },
                onVoice = { showVoicePicker = true },
            )
        },
        bottomBar = {
            Column {
                val sl = st.lines.firstOrNull { it.id == selLine }
                if (sl != null && selEnd > selStart) {
                    SelectionBar(
                        text = sl.text.substring(
                            selStart.coerceIn(0, sl.text.length),
                            selEnd.coerceIn(0, sl.text.length),
                        ),
                        onEdit = { editOpen = true },
                        onGen = { askGen(listOf(sl.id), true) },
                        onCancel = { selLine = null; selStart = 0; selEnd = 0 },
                    )
                }
                Transport(
                    playingAll = st.playingAll,
                    hasTakes = st.lines.any { it.takeId != null },
                    busy = st.busy,
                    playhead = playhead,
                    total = totalSec,
                    pending = pendingCount,
                    onAdd = { vm.addLine() },
                    onPlay = { if (st.playingAll) vm.stop() else vm.playAll() },
                    onGen = { askGen(vm.pendingIds(), false) },
                    onCancel = { vm.cancelGeneration() },
                )
            }
        },
    ) { pad ->
        Column(Modifier.fillMaxSize().padding(pad)) {
            TimelineDock(
                lines = st.lines,
                segs = segs,
                project = st.project,
                playhead = playhead,
                selected = selectedId,
                collapsed = collapsed,
                onToggle = { collapsed = !collapsed },
                onExpand = { fullTimeline = true },
                onSeek = { vm.seekTo(it) },
                onPlaySeg = { vm.playAt(it) },
                onSelect = selectClip,
                onBake = { vm.bakeOffsets() },
                onDragClip = { id, o, t -> vm.dragClip(id, o, t) },
                onCommitClip = { id, o, t -> vm.commitClip(id, o, t) },
            )

            val importing by vm.importing.collectAsStateWithLifecycle()
            if (importing != null) {
                Row(
                    Modifier.fillMaxWidth()
                        .background(MaterialTheme.colorScheme.primary.copy(alpha = 0.08f))
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(14.dp),
                        strokeWidth = 2.dp,
                    )
                    Spacer(Modifier.width(10.dp))
                    Text(
                        importing ?: "",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.primary,
                    )
                }
            }

            if (st.busy) QueueRow(st, onCancel = { vm.cancelGeneration() })

            val segById = remember(segs) { segs.associateBy { it.lineId } }
            val activeLine = remember(segs, playhead) {
                segs.firstOrNull { playhead >= it.start && playhead < it.start + it.dur }
            }
            LaunchedEffect(activeLine?.lineId) {
                if (st.playingAll && activeLine != null) {
                    val idx = st.lines.indexOfFirst { it.id == activeLine.lineId }
                    if (idx >= 0) listState.animateScrollToItem(idx)
                }
            }
            LazyColumn(state = listState, modifier = Modifier.weight(1f)) {
                itemsIndexed(st.lines, key = { _, l -> l.id }) { i, line ->
                    val seg = segById[line.id]
                    val words = remember(line.words, line.text, seg?.start, seg?.dur) {
                        resolveWords(line, seg)
                    }
                    TranscriptLine(
                        line = line,
                        index = i,
                        words = words,
                        playhead = playhead,
                        selected = selectedId == line.id,
                        selStart = if (selLine == line.id) selStart else -1,
                        selEnd = if (selLine == line.id) selEnd else -1,
                        onSelect = { a, b ->
                            selLine = line.id
                            selStart = a
                            selEnd = b
                            selectedId = line.id
                        },
                        onSeek = { vm.seekTo(it) },
                        canPlay = line.takeId != null,
                        canSource = line.sourceWav.isNotBlank(),
                        canGen = line.text.isNotBlank(),
                        onPlay = { if (st.playingLine == line.id) vm.stop() else vm.playLine(line.id) },
                        onSource = { vm.playSource(line.id) },
                        onGen = { askGen(listOf(line.id), true) },
                        onInspect = { inspectId = line.id },
                        onDelete = { vm.deleteLine(line.id) },
                    )
                }
                item {
                    TextButton(
                        onClick = { vm.addLine() },
                        enabled = !st.busy,
                        modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp),
                    ) {
                        Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(Modifier.width(6.dp))
                        Text("新增一句")
                    }
                }
                if (dev && st.log.isNotEmpty()) {
                    item {
                        Text(
                            st.log.last(),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp),
                        )
                    }
                }
            }
        }
    }

    val inspecting = inspectId?.let { id -> st.lines.firstOrNull { it.id == id } }
    if (inspecting != null) {
        LineInspectorDialog(
            line = inspecting,
            voices = voices,
            projectVoiceId = st.project.voiceId,
            onDismiss = { inspectId = null },
            onApply = { targetSec, voiceId, cfg, emotion, nvv, lang ->
                vm.setLineParams(inspecting.id, targetSec, voiceId, cfg, emotion, nvv, lang)
                inspectId = null
            },
        )
    }

    if (showVoicePicker) {
        VoicePicker(
            voices = voices,
            current = st.project.voiceId,
            onDismiss = { showVoicePicker = false },
            onPick = {
                vm.setProjectVoice(it)
                showVoicePicker = false
            },
        )
    }

    if (renameOpen) {
        var title by remember { mutableStateOf(st.project.title) }
        AlertDialog(
            onDismissRequest = { renameOpen = false },
            title = { Text("重命名工程") },
            text = {
                OutlinedTextField(
                    value = title,
                    onValueChange = { title = it },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
            },
            confirmButton = {
                TextButton(onClick = {
                    vm.renameProject(st.project.id, title)
                    renameOpen = false
                }) { Text("保存") }
            },
            dismissButton = {
                TextButton(onClick = { renameOpen = false }) { Text("取消") }
            },
        )
    }

    pendingGen?.let { req ->
        val lines = st.lines.filter { it.id in req.ids }
        GenConfirmDialog(
            lines = lines,
            dev = dev,
            onDismiss = { pendingGen = null },
            onStart = { cfg ->
                pendingGen = null
                vm.generate(req.ids, req.force, cfg)
            },
        )
    }

    if (editOpen) {
        val l = st.lines.firstOrNull { it.id == selLine }
        if (l != null && selEnd >= selStart) {
            val a = selStart.coerceIn(0, l.text.length)
            val b = selEnd.coerceIn(a, l.text.length)
            var draft by remember(l.id, a, b) { mutableStateOf(l.text.substring(a, b)) }
            AlertDialog(
                onDismissRequest = { editOpen = false },
                title = { Text("改字") },
                text = {
                    Column {
                        Text(
                            "第 ${l.idx + 1} 句",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Spacer(Modifier.height(8.dp))
                        OutlinedTextField(
                            value = draft,
                            onValueChange = { draft = it },
                            modifier = Modifier.fillMaxWidth(),
                            minLines = 1,
                            maxLines = 4,
                        )
                    }
                },
                confirmButton = {
                    TextButton(onClick = {
                        vm.replaceText(l.id, a, b, draft)
                        selStart = a
                        selEnd = a + draft.length
                        editOpen = false
                    }) { Text("替换") }
                },
                dismissButton = {
                    TextButton(onClick = { editOpen = false }) { Text("取消") }
                },
            )
        }
    }
}

@Composable
private fun EditorBar(
    title: String,
    subtitle: String,
    voiceMissing: Boolean,
    onBack: () -> Unit,
    onRename: () -> Unit,
    onVoice: () -> Unit,
) {
    var menu by remember { mutableStateOf(false) }
    Row(
        Modifier.fillMaxWidth().statusBarsPadding()
            .padding(start = 4.dp, end = 4.dp, top = 4.dp, bottom = 2.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        IconButton(onClick = onBack) {
            Icon(Icons.Default.ArrowBack, contentDescription = "返回")
        }
        Column(
            Modifier.weight(1f).clickable(onClick = onRename).padding(vertical = 4.dp),
        ) {
            Text(
                title,
                style = MaterialTheme.typography.titleMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
            )
            Text(
                subtitle,
                style = MaterialTheme.typography.labelSmall,
                color = if (voiceMissing) MaterialTheme.colorScheme.error
                else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Box {
            IconButton(onClick = { menu = true }) {
                Icon(
                    Icons.Default.MoreVert,
                    contentDescription = "更多",
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                DropdownMenuItem(
                    text = { Text("工程音色") },
                    onClick = { menu = false; onVoice() },
                )
                DropdownMenuItem(
                    text = { Text("重命名") },
                    onClick = { menu = false; onRename() },
                )
            }
        }
    }
}

@Composable
private fun QueueRow(st: EditorState, onCancel: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        LinearProgressIndicator(
            progress = { st.progress },
            modifier = Modifier.weight(1f).height(3.dp),
        )
        Spacer(Modifier.width(10.dp))
        Text(
            "${st.stage} ${(st.progress * 100).toInt()}%",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        TextButton(onClick = onCancel) { Text("停止", style = MaterialTheme.typography.labelMedium) }
    }
}

@Composable
private fun SelectionBar(
    text: String,
    onEdit: () -> Unit,
    onGen: () -> Unit,
    onCancel: () -> Unit,
) {
    Surface(color = MaterialTheme.colorScheme.surfaceVariant) {
        Row(
            Modifier.fillMaxWidth().padding(start = 16.dp, end = 4.dp, top = 2.dp, bottom = 2.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text.replace("\n", " ").take(24),
                style = MaterialTheme.typography.labelMedium,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )
            TextButton(onClick = onEdit) { Text("改字") }
            TextButton(onClick = onGen) { Text("重生成") }
            IconButton(onClick = onCancel) {
                Icon(Icons.Default.Close, contentDescription = "取消选择", modifier = Modifier.size(18.dp))
            }
        }
    }
}

@Composable
private fun Transport(
    playingAll: Boolean,
    hasTakes: Boolean,
    busy: Boolean,
    playhead: Float,
    total: Float,
    pending: Int,
    onAdd: () -> Unit,
    onPlay: () -> Unit,
    onGen: () -> Unit,
    onCancel: () -> Unit,
) {
    Surface(color = MaterialTheme.colorScheme.surface, tonalElevation = 2.dp) {
        Row(
            Modifier.fillMaxWidth().navigationBarsPadding()
                .padding(horizontal = 12.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onAdd, enabled = !busy) {
                Icon(Icons.Default.Add, contentDescription = "新增一句")
            }
            IconButton(onClick = onPlay, enabled = hasTakes) {
                Icon(
                    if (playingAll) Icons.Default.Stop else Icons.Default.PlayArrow,
                    contentDescription = if (playingAll) "停止" else "播放",
                    modifier = Modifier.size(30.dp),
                    tint = MaterialTheme.colorScheme.primary,
                )
            }
            Text(
                "${mmss(playhead)} / ${mmss(total)}",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.weight(1f))
            when {
                busy -> TextButton(onClick = onCancel) { Text("停止") }
                pending > 0 -> Button(onClick = onGen) { Text("生成 $pending") }
                else -> Text(
                    "已全部生成",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f),
                )
            }
        }
    }
}

@Composable
private fun VoicePicker(
    voices: List<com.viitorvoice.demo.data.Voice>,
    current: Long?,
    onDismiss: () -> Unit,
    onPick: (Long) -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("工程音色") },
        text = {
            Column {
                if (voices.isEmpty()) {
                    Text(
                        "还没有音色，先去「音色」页导入一段参考音频。",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
                voices.forEach { v ->
                    Text(
                        (if (v.id == current) "✓ " else "") + v.name,
                        style = MaterialTheme.typography.bodyLarge,
                        modifier = Modifier.fillMaxWidth()
                            .clickable { onPick(v.id) }
                            .padding(vertical = 10.dp),
                    )
                }
            }
        },
        confirmButton = {
            TextButton(onClick = onDismiss) { Text("关闭") }
        },
    )
}

@Composable
private fun GenConfirmDialog(
    lines: List<Line>,
    dev: Boolean,
    onDismiss: () -> Unit,
    onStart: (Float?) -> Unit,
) {
    var override by remember { mutableStateOf(false) }
    var cfg by remember { mutableStateOf(2.0f) }
    val secs = lines.sumOf { l ->
        (l.targetSec.takeIf { it > 0f } ?: Duration.seconds(l.text, l.lang)).toDouble()
    }.toFloat()
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("开始生成？") },
        text = {
            Column(Modifier.heightIn(max = 420.dp).verticalScroll(rememberScrollState())) {
                Text("${lines.size} 句 · 音频约 ${mmss(secs)}")
                Spacer(Modifier.height(8.dp))
                Text(
                    "本地 NPU 推理，手机可能发热变慢；生成中可随时停止。",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                if (dev) {
                    Spacer(Modifier.height(10.dp))
                    Text(
                        "开发者：每句可在「⋮ → 参数」里调 cfg / 情绪 / NVV / 目标时长。",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    Spacer(Modifier.height(6.dp))
                    lines.take(8).forEach { l ->
                        Text(
                            "${l.idx + 1}. ${l.text.take(16)} · ${
                                "%.1fs".format(
                                    l.targetSec.takeIf { it > 0f }
                                        ?: Duration.seconds(l.text, l.lang),
                                )
                            }",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    if (lines.size > 8) {
                        Text(
                            "… 还有 ${lines.size - 8} 句",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Spacer(Modifier.height(10.dp))
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            "本次覆盖 cfg",
                            style = MaterialTheme.typography.labelMedium,
                            modifier = Modifier.weight(1f),
                        )
                        Switch(checked = override, onCheckedChange = { override = it })
                    }
                    if (override) {
                        Text(
                            "cfg %.1f".format(cfg),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                        Slider(
                            value = cfg,
                            onValueChange = { cfg = it },
                            valueRange = 1f..4f,
                        )
                    }
                }
            }
        },
        confirmButton = {
            Button(onClick = { onStart(if (override) cfg else null) }) { Text("开始生成") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("取消") }
        },
    )
}
