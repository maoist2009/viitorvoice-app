package com.viitorvoice.demo.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.horizontalScroll
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
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material.icons.filled.Upload
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.viitorvoice.demo.Engine
import com.viitorvoice.demo.text.Duration
import com.viitorvoice.demo.text.Tags
import kotlin.math.abs
import kotlin.math.min

private val LANGS = listOf(
    "zh" to "中文",
    "en" to "EN",
    "ja" to "日本語",
    "ko" to "한국어",
)

@Composable
fun HomeScreen(vm: HomeViewModel = viewModel()) {
    val state by vm.state.collectAsStateWithLifecycle()
    val snackbar = remember { SnackbarHostState() }
    val ctx = LocalContext.current

    LaunchedEffect(Unit) {
        val act = ctx as? android.app.Activity ?: return@LaunchedEffect
        val ref = act.intent.getStringExtra("ref")
        if (ref != null) {
            // Headless clone: --es ref <wav> --es text "…" [--es reftext "…"] [--ei frames N]
            vm.debugClone(
                java.io.File(ref),
                act.intent.getStringExtra("text") ?: "",
                act.intent.getStringExtra("reftext") ?: "",
                act.intent.getIntExtra("frames", 0),
            )
            return@LaunchedEffect
        }
        val path = act.intent.getStringExtra("load")
        if (path != null) {
            val text = act.intent.getStringExtra("text")
            if (text != null) vm.onTextChange(text)
            vm.loadFile(java.io.File(path), act.intent.getBooleanExtra("auto", false))
        }
    }

    LaunchedEffect(state.error) {
        state.error?.let {
            snackbar.showSnackbar(it)
            vm.clearError()
        }
    }

    var pendingSlot by remember { mutableStateOf(Slot.SOURCE) }

    val micPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) vm.startRecording(pendingSlot) else vm.addLog("没有麦克风权限")
    }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri -> if (uri != null) vm.importAudio(uri, pendingSlot) }

    fun record(slot: Int) {
        pendingSlot = slot
        val granted = ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        if (granted) vm.startRecording(slot) else {
            micPermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    fun pick(slot: Int) {
        pendingSlot = slot
        picker.launch(arrayOf("audio/*"))
    }

    Scaffold(
        containerColor = Color.Transparent,
        snackbarHost = { SnackbarHost(snackbar) },
    ) { pad ->
        Box(Modifier.fillMaxSize()) {
            GradientBackdrop()
            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(
                    start = 20.dp, end = 20.dp,
                    top = pad.calculateTopPadding() + 20.dp,
                    bottom = pad.calculateBottomPadding() + 132.dp,
                ),
                verticalArrangement = Arrangement.spacedBy(14.dp),
            ) {
                item { Header(state, vm::refreshModels) }
                item { Tabs(state.tab) { vm.setTab(it) } }

                if (state.tab == TAB_EDIT) {
                    item {
                        ClipCard(
                            title = "源音频",
                            subtitle = if (state.frames > 0)
                                "%.2f s · %d 帧".format(state.durationSec, state.frames)
                            else "未选择",
                            samples = state.samples,
                            frames = state.frames,
                            selStart = state.selStart,
                            selEnd = state.selEnd,
                            onSelect = vm::setSelection,
                            recording = state.recording && state.recordingSlot == Slot.SOURCE,
                            onRecord = { if (state.recording) vm.stopRecording() else record(Slot.SOURCE) },
                            onImport = { pick(Slot.SOURCE) },
                            onPlay = if (state.playing) vm::stop else vm::playSource,
                            playEnabled = state.sourcePath != null,
                            hint = "录制或导入一段语音，然后在波形上拖选要重写的片段",
                            stageLabel = "源",
                        )
                    }
                    item {
                        EditTextCard(
                            value = state.text,
                            baseline = state.baselineText,
                            enabled = !state.busy && !state.recording,
                            onValueChange = vm::onTextChange,
                            selection = state,
                        )
                    }
                    item { Params(state, vm) }
                    item { ResultCard(state, vm) }
                } else {
                    item {
                        ClipCard(
                            title = "参考音频",
                            subtitle = if (state.refFrames > 0)
                                "%.2f s · %d 帧".format(state.refSec, state.refFrames)
                            else "未选择",
                            samples = state.refSamples,
                            frames = state.refFrames,
                            selStart = -1,
                            selEnd = -1,
                            onSelect = { _, _ -> },
                            recording = state.recording && state.recordingSlot == Slot.REF,
                            onRecord = { if (state.recording) vm.stopRecording() else record(Slot.REF) },
                            onImport = { pick(Slot.REF) },
                            onPlay = if (state.playing) vm::stop else vm::playRef,
                            playEnabled = state.refPath != null,
                            hint = "1–5 秒清晰人声，模型会克隆这个音色",
                            stageLabel = "参考",
                        )
                    }
                    item {
                        RefTextCard(
                            value = state.refText,
                            enabled = !state.busy && !state.recording,
                            onChange = vm::setRefText,
                        )
                    }
                    item { SaveVoiceRow(state, vm) }
                    item { CloneTextCard(state, vm) }
                    item { Params(state, vm) }
                    item { ResultCard(state, vm) }
                }

                if (state.log.isNotEmpty()) item { LogCard(state.log) }
            }

            ActionBar(state, vm)
        }
    }
}

/* ------------------------------------------------------------------ chrome */

@Composable
private fun GradientBackdrop() {
    val scheme = MaterialTheme.colorScheme
    Canvas(
        Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    listOf(scheme.background, scheme.background, scheme.surface)
                )
            )
    ) {
        drawCircle(
            brush = Brush.radialGradient(
                listOf(scheme.primary.copy(alpha = 0.20f), Color.Transparent),
                center = Offset(size.width * 0.15f, size.height * 0.02f),
                radius = size.width * 0.8f,
            ),
            radius = size.width * 0.8f,
            center = Offset(size.width * 0.15f, size.height * 0.02f),
        )
        drawCircle(
            brush = Brush.radialGradient(
                listOf(scheme.tertiary.copy(alpha = 0.14f), Color.Transparent),
                center = Offset(size.width * 0.95f, size.height * 0.28f),
                radius = size.width * 0.7f,
            ),
            radius = size.width * 0.7f,
            center = Offset(size.width * 0.95f, size.height * 0.28f),
        )
    }
}

@Composable
private fun Header(state: UiState, refresh: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(
                "ViiTorVoice",
                style = MaterialTheme.typography.headlineMedium,
                color = MaterialTheme.colorScheme.onBackground,
            )
            Text(
                "端侧非自回归语音编辑 · SM8750 NPU",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier
                        .size(7.dp)
                        .clip(CircleShape)
                        .background(
                            if (state.modelStatus.startsWith("就绪")) Color(0xFF69F0AE)
                            else MaterialTheme.colorScheme.error
                        )
                )
                Spacer(Modifier.width(6.dp))
                Text(
                    state.modelStatus,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
        IconButton(onClick = refresh) {
            Icon(Icons.Default.Refresh, contentDescription = "刷新模型状态")
        }
    }
}

@Composable
private fun Tabs(selected: Int, onSelect: (Int) -> Unit) {
    Surface(
        shape = CircleShape,
        color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.55f),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Row(Modifier.padding(4.dp)) {
            TabButton("音色克隆", selected == TAB_CLONE, Modifier.weight(1f)) { onSelect(TAB_CLONE) }
            TabButton("局部编辑", selected == TAB_EDIT, Modifier.weight(1f)) { onSelect(TAB_EDIT) }
        }
    }
}

@Composable
private fun TabButton(label: String, active: Boolean, modifier: Modifier, onClick: () -> Unit) {
    val bg by animateColorAsState(
        if (active) MaterialTheme.colorScheme.primary else Color.Transparent,
        tween(220), label = "tabBg",
    )
    val fg by animateColorAsState(
        if (active) MaterialTheme.colorScheme.onPrimary else MaterialTheme.colorScheme.onSurfaceVariant,
        tween(220), label = "tabFg",
    )
    Box(
        modifier
            .clip(CircleShape)
            .background(bg)
            .clickable(onClick = onClick)
            .padding(vertical = 11.dp),
        contentAlignment = Alignment.Center,
    ) {
        Text(label, style = MaterialTheme.typography.labelLarge, color = fg)
    }
}

@Composable
private fun CardBox(content: @Composable androidx.compose.foundation.layout.ColumnScope.() -> Unit) {
    Surface(
        shape = RoundedCornerShape(26.dp),
        color = MaterialTheme.colorScheme.surface.copy(alpha = 0.72f),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.14f)),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.padding(16.dp), content = content)
    }
}

/* ------------------------------------------------------------------- cards */

@Composable
private fun ClipCard(
    title: String,
    subtitle: String,
    samples: FloatArray,
    frames: Int,
    selStart: Int,
    selEnd: Int,
    onSelect: (Int, Int) -> Unit,
    recording: Boolean,
    onRecord: () -> Unit,
    onImport: () -> Unit,
    onPlay: () -> Unit,
    playEnabled: Boolean,
    hint: String,
    stageLabel: String,
) {
    CardBox {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                Icons.Default.GraphicEq, contentDescription = null,
                tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text(title, style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.weight(1f))
            Text(
                if (recording) "录音中…" else subtitle,
                style = MaterialTheme.typography.labelMedium,
                color = if (recording) MaterialTheme.colorScheme.error
                else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Spacer(Modifier.height(12.dp))

        if (samples.isEmpty()) {
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(104.dp)
                    .clip(RoundedCornerShape(18.dp))
                    .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    hint,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        } else {
            WaveformView(
                samples = samples,
                frames = frames,
                selStart = selStart,
                selEnd = selEnd,
                selectable = selStart >= 0,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(104.dp)
                    .clip(RoundedCornerShape(18.dp)),
                onSelect = onSelect,
            )
            if (selStart >= 0 && selEnd > selStart) {
                Spacer(Modifier.height(6.dp))
                Text(
                    "重写 %.2f–%.2f s（帧 %d–%d）".format(
                        selStart * 0.04f, selEnd * 0.04f, selStart, selEnd - 1,
                    ),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.primary,
                )
            }
        }

        Spacer(Modifier.height(12.dp))
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            PillButton(
                if (recording) "停止" else "录制",
                if (recording) Icons.Default.Stop else Icons.Default.Mic,
                filled = recording,
                onClick = onRecord,
            )
            PillButton("导入", Icons.Default.Upload, onClick = onImport)
            PillButton("原音", Icons.Default.PlayArrow, enabled = playEnabled, onClick = onPlay)
            Spacer(Modifier.weight(1f))
            Text(
                stageLabel,
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.align(Alignment.CenterVertically),
            )
        }
    }
}

@Composable
private fun EditTextCard(
    value: String,
    baseline: String,
    enabled: Boolean,
    onValueChange: (String) -> Unit,
    selection: UiState,
) {
    CardBox {
        Text("文本", style = MaterialTheme.typography.titleMedium)
        Text(
            if (baseline.isEmpty()) "先输入这段音频说的话"
            else "改掉想变的词，选区会自动跟随",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(Modifier.height(10.dp))
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            modifier = Modifier.fillMaxWidth(),
            placeholder = { Text("例如：今天我们来测试一下语音编辑功能。") },
            minLines = 2,
            maxLines = 5,
            enabled = enabled,
            shape = RoundedCornerShape(16.dp),
        )
        if (selection.frames > 0) {
            Spacer(Modifier.height(6.dp))
            Text(
                "选区也可以在波形上手动拖动",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun RefTextCard(value: String, enabled: Boolean, onChange: (String) -> Unit) {
    var expanded by remember { mutableStateOf(value.isNotEmpty()) }
    CardBox {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("参考文本", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.weight(1f))
            TextButton(onClick = { expanded = !expanded }) {
                Text(if (expanded) "收起" else "填写")
            }
        }
        Text(
            "填上参考音频里说的话会更稳；留空则走 no-ref-text 分支",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        if (expanded) {
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = value,
                onValueChange = onChange,
                modifier = Modifier.fillMaxWidth(),
                placeholder = { Text("参考音频的内容（可留空）") },
                minLines = 1,
                maxLines = 3,
                enabled = enabled,
                shape = RoundedCornerShape(16.dp),
            )
        }
    }
}

@Composable
private fun CloneTextCard(state: UiState, vm: HomeViewModel) {
    var value by remember { mutableStateOf(TextFieldValue(state.text)) }
    LaunchedEffect(state.text) {
        if (state.text != value.text) {
            value = TextFieldValue(state.text, TextRange(state.text.length))
        }
    }

    fun update(text: String, cursor: Int) {
        value = TextFieldValue(text, TextRange(cursor.coerceIn(0, text.length)))
        vm.onTextChange(text)
    }

    fun insertNvv(tag: String) {
        val at = value.selection.start.coerceIn(0, value.text.length)
        val add = if (at > 0 && !value.text[at - 1].isWhitespace()) " $tag" else tag
        update(value.text.substring(0, at) + add + value.text.substring(at), at + add.length)
    }

    fun setEmotion(tag: String?) {
        val body = Tags.stripEmotion(value.text)
        val next = if (tag == null) body else "$tag $body"
        update(next, next.length)
    }

    CardBox {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("要说的内容", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.weight(1f))
            Text(
                "${state.text.length} 字",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        Spacer(Modifier.height(10.dp))
        OutlinedTextField(
            value = value,
            onValueChange = {
                value = it
                vm.onTextChange(it.text)
            },
            modifier = Modifier.fillMaxWidth(),
            placeholder = { Text("输入要合成的话，可用下方标签加情绪或笑声") },
            minLines = 3,
            maxLines = 6,
            enabled = !state.busy && !state.recording,
            shape = RoundedCornerShape(16.dp),
        )

        if (state.refFrames > 0 && state.text.isNotEmpty()) {
            val est = if (state.targetSec > 0f) (state.targetSec * Duration.FRAME_RATE).toInt()
            else Duration.frames(state.text, state.lang)
            val winSec = Engine.maxWindow() / 25f
            val total = (state.refFrames + est) / 25f
            val over = total > winSec
            Text(
                if (over) "参考 %.1f s + 目标约 %.1f s 超过 %.0f s 窗口，参考音会被裁短"
                    .format(state.refSec, est / 25f, winSec)
                else "参考 %.1f s + 目标约 %.1f s / %.0f s".format(state.refSec, est / 25f, winSec),
                style = MaterialTheme.typography.labelSmall,
                color = if (over) MaterialTheme.colorScheme.error
                else MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Spacer(Modifier.height(10.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("目标时长", style = MaterialTheme.typography.labelMedium)
            Spacer(Modifier.width(8.dp))
            var durText by remember {
                mutableStateOf(if (state.targetSec > 0f) "%.2f".format(state.targetSec) else "")
            }
            OutlinedTextField(
                value = durText,
                onValueChange = { v ->
                    durText = v.filter { it.isDigit() || it == '.' }.take(5)
                    vm.setTargetSec(durText.toFloatOrNull() ?: 0f)
                },
                modifier = Modifier.width(110.dp),
                placeholder = { Text("自动") },
                singleLine = true,
                enabled = !state.busy && !state.recording,
                shape = RoundedCornerShape(12.dp),
                textStyle = MaterialTheme.typography.bodySmall,
            )
            Spacer(Modifier.width(8.dp))
            Text(
                "秒，留空=官方 4.8 字/秒估算",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Spacer(Modifier.height(10.dp))
        Text("语言", style = MaterialTheme.typography.labelMedium)
        Row(
            Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            LANGS.forEach { (code, name) ->
                FilterChip(
                    selected = state.lang == code,
                    onClick = { vm.setLang(code) },
                    label = { Text(name) },
                )
            }
        }

        Spacer(Modifier.height(10.dp))
        Text("情绪", style = MaterialTheme.typography.labelMedium)
        Row(
            Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            val current = Tags.emotionTag(state.text)
            FilterChip(
                selected = current == null,
                onClick = { setEmotion(null) },
                label = { Text("默认") },
            )
            Tags.EMOTION_CHOICES.forEach { (name, tag) ->
                FilterChip(
                    selected = current == tag,
                    onClick = { setEmotion(if (current == tag) null else tag) },
                    label = { Text(name) },
                )
            }
        }

        Spacer(Modifier.height(10.dp))
        Text("语气词", style = MaterialTheme.typography.labelMedium)
        Row(
            Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Tags.NVV_CHOICES.forEach { (name, tag) ->
                FilterChip(
                    selected = false,
                    onClick = { insertNvv(tag) },
                    label = { Text(name) },
                )
            }
        }
    }
}

@Composable
private fun Params(state: UiState, vm: HomeViewModel) {
    var open by remember { mutableStateOf(false) }
    CardBox {
        Row(
            Modifier
                .fillMaxWidth()
                .clickable { open = !open },
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                Icons.Default.Tune, contentDescription = null,
                tint = MaterialTheme.colorScheme.primary, modifier = Modifier.size(18.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text("高级参数", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.weight(1f))
            Text(
                "CFG %.1f".format(state.cfg),
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (open) {
            Spacer(Modifier.height(6.dp))
            ParamSlider("强度 CFG", state.cfg, 0f, 4f) { vm.setCfg(it) }
            ParamSlider("情绪引导", state.emotionScale, 0f, 4f) { vm.setEmotionScale(it) }
            ParamSlider("语气词引导", state.nvvScale, 0f, 4f) { vm.setNvvScale(it) }
            Text(
                "情绪/语气词引导只在文本里带对应标签时生效；文本没有标签时自动跳过那一分支。",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun ParamSlider(label: String, value: Float, from: Float, to: Float, onChange: (Float) -> Unit) {
    Column {
        Row {
            Text(label, style = MaterialTheme.typography.labelMedium)
            Spacer(Modifier.weight(1f))
            Text(
                "%.1f".format(value),
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
            )
        }
        Slider(value = value, onValueChange = onChange, valueRange = from..to)
    }
}

@Composable
private fun ResultCard(state: UiState, vm: HomeViewModel) {
    if (state.resultPath == null && !state.busy) return
    CardBox {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Icon(
                Icons.Default.AutoAwesome, contentDescription = null,
                tint = MaterialTheme.colorScheme.tertiary, modifier = Modifier.size(18.dp),
            )
            Spacer(Modifier.width(8.dp))
            Text("结果", style = MaterialTheme.typography.titleMedium)
            Spacer(Modifier.weight(1f))
            if (state.resultPath != null) {
                Text(
                    "%.2f s".format(state.resultSec),
                    style = MaterialTheme.typography.labelMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        Spacer(Modifier.height(10.dp))
        if (state.busy) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(modifier = Modifier.size(18.dp), strokeWidth = 2.dp)
                Spacer(Modifier.width(10.dp))
                Text(
                    "${state.stage} · ${(state.progress * 100).toInt()}%",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
            Spacer(Modifier.height(8.dp))
            ProgressGlow(state.progress)
        } else {
            PillButton(
                if (state.playing) "停止" else "播放结果",
                if (state.playing) Icons.Default.Stop else Icons.Default.PlayArrow,
                filled = true,
                onClick = if (state.playing) vm::stop else vm::play,
            )
        }
    }
}

@Composable
private fun ProgressGlow(progress: Float) {
    val p by animateFloatAsState(progress.coerceIn(0f, 1f), tween(220), label = "p")
    val c0 = MaterialTheme.colorScheme.primary
    val c1 = MaterialTheme.colorScheme.tertiary
    val track = MaterialTheme.colorScheme.surfaceVariant
    Canvas(
        Modifier
            .fillMaxWidth()
            .height(8.dp)
            .clip(CircleShape)
            .background(track)
    ) {
        drawRoundRect(
            brush = Brush.horizontalGradient(listOf(c0, c1)),
            size = Size(size.width * p, size.height),
            cornerRadius = androidx.compose.ui.geometry.CornerRadius(size.height / 2),
        )
    }
}

@Composable
private fun LogCard(log: List<String>) {
    var open by remember { mutableStateOf(false) }
    CardBox {
        Row(
            Modifier
                .fillMaxWidth()
                .clickable { open = !open },
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("日志", style = MaterialTheme.typography.titleSmall)
            Spacer(Modifier.weight(1f))
            Text(
                if (open) "收起" else "${log.size} 行",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
        if (open) {
            Spacer(Modifier.height(6.dp))
            log.takeLast(16).forEach {
                Text(
                    it,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Composable
private fun ActionBar(state: UiState, vm: HomeViewModel) {
    Box(
        Modifier
            .fillMaxSize()
            .padding(20.dp),
        contentAlignment = Alignment.BottomCenter,
    ) {
        val enabled = !state.busy && !state.recording && when (state.tab) {
            TAB_EDIT -> state.frames > 0 && state.text.isNotBlank()
            else -> state.refFrames > 0 && state.text.isNotBlank()
        }
        val shape = RoundedCornerShape(22.dp)
        val brush = if (enabled) Brush.horizontalGradient(
            listOf(MaterialTheme.colorScheme.primary, MaterialTheme.colorScheme.tertiary)
        ) else Brush.horizontalGradient(
            listOf(
                MaterialTheme.colorScheme.surfaceVariant,
                MaterialTheme.colorScheme.surfaceVariant,
            )
        )
        Box(
            Modifier
                .fillMaxWidth()
                .height(58.dp)
                .clip(shape)
                .background(brush)
                .clickable(enabled = enabled) {
                    if (state.tab == TAB_EDIT) vm.applyEdit() else vm.clone()
                },
            contentAlignment = Alignment.Center,
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                if (state.busy) {
                    CircularProgressIndicator(
                        modifier = Modifier.size(18.dp),
                        strokeWidth = 2.dp,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                    Spacer(Modifier.width(10.dp))
                    Text(
                        state.stage,
                        style = MaterialTheme.typography.titleSmall,
                        color = MaterialTheme.colorScheme.onPrimary,
                    )
                } else {
                    Icon(
                        Icons.Default.AutoAwesome,
                        contentDescription = null,
                        tint = if (enabled) MaterialTheme.colorScheme.onPrimary
                        else MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.size(20.dp),
                    )
                    Spacer(Modifier.width(8.dp))
                    Text(
                        if (state.tab == TAB_EDIT) "局部重生成" else "克隆生成",
                        style = MaterialTheme.typography.titleMedium,
                        color = if (enabled) MaterialTheme.colorScheme.onPrimary
                        else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun PillButton(
    label: String,
    icon: androidx.compose.ui.graphics.vector.ImageVector,
    filled: Boolean = false,
    enabled: Boolean = true,
    onClick: () -> Unit,
) {
    val shape = RoundedCornerShape(14.dp)
    val bg = if (filled) MaterialTheme.colorScheme.primary
    else MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.6f)
    val fg = if (filled) MaterialTheme.colorScheme.onPrimary
    else MaterialTheme.colorScheme.onSurface
    Row(
        Modifier
            .clip(shape)
            .background(if (enabled) bg else bg.copy(alpha = 0.4f))
            .clickable(enabled = enabled, onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 9.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(
            icon, contentDescription = null,
            tint = if (enabled) fg else fg.copy(alpha = 0.4f),
            modifier = Modifier.size(16.dp),
        )
        Spacer(Modifier.width(6.dp))
        Text(
            label,
            style = MaterialTheme.typography.labelLarge,
            color = if (enabled) fg else fg.copy(alpha = 0.4f),
        )
    }
}

/* ----------------------------------------------------------------- waveform */

/** Envelope waveform with a draggable selection band (frames of 40 ms). */
@Composable
fun WaveformView(
    samples: FloatArray,
    frames: Int,
    selStart: Int,
    selEnd: Int,
    modifier: Modifier = Modifier,
    selectable: Boolean = true,
    onSelect: (Int, Int) -> Unit,
) {
    var dragFrom by remember { mutableStateOf(-1f) }
    var dragTo by remember { mutableStateOf(-1f) }
    val wave = MaterialTheme.colorScheme.primary
    val band = MaterialTheme.colorScheme.primary.copy(alpha = 0.20f)
    val axis = MaterialTheme.colorScheme.outline

    Canvas(
        modifier = modifier
            .background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.25f))
            .pointerInput(frames, samples, selectable) {
                if (!selectable) return@pointerInput
                detectDragGestures(
                    onDragStart = { off -> dragFrom = off.x; dragTo = off.x },
                    onDragEnd = {
                        val a = dragFrom
                        val b = dragTo
                        dragFrom = -1f; dragTo = -1f
                        if (a >= 0 && b >= 0 && frames > 0 && size.width > 0) {
                            var f0 = (minOf(a, b) / size.width * frames).toInt()
                            var f1 = (maxOf(a, b) / size.width * frames).toInt()
                            if (f1 <= f0) f1 = f0 + 1
                            if (abs(b - a) < 8f) {
                                f0 = (f0 - 2).coerceAtLeast(0)
                                f1 = (f1 + 2).coerceAtMost(frames)
                            }
                            onSelect(f0.coerceIn(0, frames - 1), f1.coerceIn(1, frames))
                        }
                    },
                    onDrag = { change, _ ->
                        change.consume()
                        dragTo = change.position.x
                    },
                )
            },
    ) {
        val w = size.width
        val h = size.height
        val mid = h / 2f
        drawLine(axis.copy(alpha = 0.4f), Offset(0f, mid), Offset(w, mid), strokeWidth = 1f)

        if (frames > 0 && selEnd > selStart && selStart >= 0) {
            val x0 = selStart.toFloat() / frames * w
            val x1 = selEnd.toFloat() / frames * w
            drawRect(band, Offset(x0, 0f), Size(x1 - x0, h))
            drawLine(wave.copy(alpha = 0.8f), Offset(x0, 0f), Offset(x0, h), strokeWidth = 2f)
            drawLine(wave.copy(alpha = 0.8f), Offset(x1, 0f), Offset(x1, h), strokeWidth = 2f)
        }
        if (dragFrom >= 0 && dragTo >= 0) {
            drawRect(
                band.copy(alpha = 0.35f),
                Offset(minOf(dragFrom, dragTo), 0f),
                Size(abs(dragTo - dragFrom), h),
            )
        }

        val cols = w.toInt().coerceAtLeast(1)
        val per = (samples.size / cols).coerceAtLeast(1)
        for (i in 0 until cols) {
            var peak = 0f
            val s0 = i * per
            val s1 = minOf(s0 + per, samples.size)
            var k = s0
            while (k < s1) {
                val v = abs(samples[k])
                if (v > peak) peak = v
                k += 1
            }
            val y = (peak.coerceAtMost(1f) * mid * 0.92f)
            drawLine(
                brush = Brush.verticalGradient(listOf(wave.copy(alpha = 0.5f), wave)),
                start = Offset(i.toFloat(), mid - y),
                end = Offset(i.toFloat(), mid + y),
                strokeWidth = 1.6f,
            )
        }
    }
}
