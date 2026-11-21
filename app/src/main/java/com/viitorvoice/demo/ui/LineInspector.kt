package com.viitorvoice.demo.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.ui.unit.dp
import com.viitorvoice.demo.data.Line
import com.viitorvoice.demo.data.Voice

private val LANGS = listOf("zh" to "中文", "en" to "English", "ja" to "日本語", "ko" to "한국어")

/** Per-line overrides: voice, target duration, and the three engine dials. */
@Composable
fun LineInspectorDialog(
    line: Line,
    voices: List<Voice>,
    projectVoiceId: Long?,
    onDismiss: () -> Unit,
    onApply: (targetSec: Float, voiceId: Long?, cfg: Float, emotion: Float, nvv: Float, lang: String) -> Unit,
) {
    var target by remember { mutableStateOf(if (line.targetSec > 0f) "%.2f".format(line.targetSec) else "") }
    var voiceId by remember { mutableStateOf(line.voiceId) }
    var cfg by remember { mutableStateOf(line.cfg) }
    var emotion by remember { mutableStateOf(line.emotion) }
    var nvv by remember { mutableStateOf(line.nvv) }
    var lang by remember { mutableStateOf(line.lang) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("第 ${line.idx + 1} 句") },
        text = {
            Column(
                Modifier.heightIn(max = 460.dp).verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                OutlinedTextField(
                    value = target,
                    onValueChange = { target = it },
                    label = { Text("目标时长（秒，留空=按字数估算）") },
                    singleLine = true,
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Decimal),
                    modifier = Modifier.fillMaxWidth(),
                )

                Text("音色", style = MaterialTheme.typography.labelMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    FilterChip(
                        selected = voiceId == null,
                        onClick = { voiceId = null },
                        label = {
                            Text(
                                if (projectVoiceId == null) "跟随工程（未选）" else "跟随工程"
                            )
                        },
                    )
                }
                voices.forEach { v ->
                    FilterChip(
                        selected = voiceId == v.id,
                        onClick = { voiceId = v.id },
                        label = { Text(v.name) },
                    )
                }

                Text("语言", style = MaterialTheme.typography.labelMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    LANGS.forEach { (code, name) ->
                        FilterChip(
                            selected = lang == code,
                            onClick = { lang = code },
                            label = { Text(name) },
                        )
                    }
                }

                Dial("引导强度 cfg", cfg, 1f..4f, "%.1f") { cfg = it }
                Dial("情绪", emotion, 0.5f..1.5f, "%.2f") { emotion = it }
                Dial("NVV", nvv, 0f..1f, "%.2f") { nvv = it }

                Text(
                    "改动后需要重新生成这一句。",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        },
        confirmButton = {
            TextButton(onClick = {
                val sec = target.trim().toFloatOrNull()?.coerceIn(0f, 60f) ?: 0f
                onApply(sec, voiceId, cfg, emotion, nvv, lang)
            }) { Text("应用") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("取消") }
        },
    )
}

@Composable
private fun Dial(label: String, value: Float, range: ClosedFloatingPointRange<Float>, fmt: String, onChange: (Float) -> Unit) {
    Column {
        Row(Modifier.fillMaxWidth()) {
            Text(label, style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(1f))
            Text(fmt.format(value), style = MaterialTheme.typography.labelMedium)
        }
        Slider(
            value = value,
            onValueChange = onChange,
            valueRange = range,
        )
    }
}
