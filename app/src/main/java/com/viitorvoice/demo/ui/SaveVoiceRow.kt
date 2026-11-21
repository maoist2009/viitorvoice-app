package com.viitorvoice.demo.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

/**
 * "存为音色" for the current reference clip. The name starts empty on purpose:
 * nothing is pre-filled, the user decides whether this clip is worth keeping.
 */
@Composable
fun SaveVoiceRow(state: UiState, vm: HomeViewModel) {
    if (state.refFrames <= 0) return
    var open by remember { mutableStateOf(false) }
    var name by remember { mutableStateOf("") }

    Row(verticalAlignment = Alignment.CenterVertically) {
        TextButton(onClick = { name = ""; open = true }) { Text("存为音色") }
        if (state.activeVoiceId != null) {
            Text(
                "已关联音色 #${state.activeVoiceId}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.primary,
            )
        }
        Spacer(Modifier.weight(1f))
        Text(
            "转写 / 语言可留空",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }

    if (open) {
        AlertDialog(
            onDismissRequest = { open = false },
            title = { Text("存为音色") },
            text = {
                Column {
                    Text(
                        "参考音频会和当前转写、语言一起保存，以后在「音色」里载入复用。",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(Modifier.height(10.dp))
                    OutlinedTextField(
                        value = name,
                        onValueChange = { name = it },
                        placeholder = { Text("音色名称（可留空）") },
                        singleLine = true,
                    )
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    vm.saveVoice(name)
                    open = false
                }) { Text("保存") }
            },
            dismissButton = {
                TextButton(onClick = { open = false }) { Text("取消") }
            },
        )
    }
}
