package com.viitorvoice.demo.ui

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.GraphicEq
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.viitorvoice.demo.data.Project
import com.viitorvoice.demo.data.ProjectStat
import com.viitorvoice.demo.data.Voice
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Studio home: the project list, not a generation form. */
@Composable
fun ProjectListScreen(
    vm: StudioViewModel,
    onOpen: (Long) -> Unit,
    onVoices: () -> Unit,
    onSettings: () -> Unit,
) {
    val projects by vm.projects.collectAsStateWithLifecycle()
    val stats by vm.stats.collectAsStateWithLifecycle()
    val voices by vm.voices.collectAsStateWithLifecycle()
    var showNew by remember { mutableStateOf(false) }
    var renameId by remember { mutableStateOf<Long?>(null) }
    var pendingDelete by remember { mutableStateOf<Project?>(null) }
    val ctx = LocalContext.current
    var pendingKind by remember { mutableStateOf<String?>(null) }

    val picker = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        val kind = pendingKind ?: return@rememberLauncherForActivityResult
        pendingKind = null
        if (uri == null) return@rememberLauncherForActivityResult
        runCatching {
            ctx.contentResolver.takePersistableUriPermission(
                uri, Intent.FLAG_GRANT_READ_URI_PERMISSION,
            )
        }
        val id = vm.createDubbing(displayName(ctx, uri), kind, uri)
        vm.importDubbing(id, uri)
        onOpen(id)
    }

    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        Column(Modifier.fillMaxSize().statusBarsPadding()) {
            Row(
                Modifier.fillMaxWidth().padding(start = 20.dp, end = 8.dp, top = 14.dp, bottom = 10.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(Modifier.weight(1f)) {
                    Text(
                        "VIITORVOICE STUDIO",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    Text("工程", style = MaterialTheme.typography.headlineMedium)
                }
                IconButton(onClick = onVoices) {
                    Icon(Icons.Default.GraphicEq, contentDescription = "音色库")
                }
                IconButton(onClick = onSettings) {
                    Icon(Icons.Default.Settings, contentDescription = "设置")
                }
            }

            LazyColumn(
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(start = 20.dp, end = 20.dp, bottom = 96.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                if (projects.isEmpty()) {
                    item {
                        Column(Modifier.padding(top = 56.dp)) {
                            Text(
                                "还没有工程",
                                style = MaterialTheme.typography.titleMedium,
                            )
                            Spacer(Modifier.height(6.dp))
                            Text(
                                "新建一个：选音色写台词，或者导入一段音频/视频来配音。",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
                items(projects, key = { it.id }) { p ->
                    ProjectCard(
                        project = p,
                        stat = stats[p.id],
                        onOpen = { onOpen(p.id) },
                        onRename = { renameId = p.id },
                        onDelete = { pendingDelete = p },
                    )
                }
            }
        }

        ExtendedFloatingActionButton(
            onClick = { showNew = true },
            icon = { Icon(Icons.Default.Add, contentDescription = null) },
            text = { Text("新建") },
            containerColor = MaterialTheme.colorScheme.primary,
            contentColor = MaterialTheme.colorScheme.onPrimary,
            modifier = Modifier.align(Alignment.BottomEnd)
                .navigationBarsPadding()
                .padding(end = 20.dp, bottom = 20.dp),
        )
    }

    if (showNew) {
        NewProjectDialog(
            voices = voices,
            onDismiss = { showNew = false },
            onCreate = { title, voiceId ->
                showNew = false
                onOpen(vm.createProject(title, voiceId))
            },
            onImport = { kind ->
                showNew = false
                pendingKind = kind
                picker.launch(arrayOf(if (kind == "video") "video/*" else "audio/*"))
            },
        )
    }

    renameId?.let { id ->
        val p = projects.firstOrNull { it.id == id } ?: return@let
        RenameDialog(
            initial = p.title,
            onDismiss = { renameId = null },
            onRename = {
                vm.renameProject(id, it)
                renameId = null
            },
        )
    }

    pendingDelete?.let { p ->
        AlertDialog(
            onDismissRequest = { pendingDelete = null },
            title = { Text("删除工程") },
            text = { Text("「${p.title}」和它的所有台词、音频都会被删除。") },
            confirmButton = {
                TextButton(onClick = {
                    vm.deleteProject(p.id)
                    pendingDelete = null
                }) { Text("删除") }
            },
            dismissButton = {
                TextButton(onClick = { pendingDelete = null }) { Text("取消") }
            },
        )
    }
}

@Composable
private fun ProjectCard(
    project: Project,
    stat: ProjectStat?,
    onOpen: () -> Unit,
    onRename: () -> Unit,
    onDelete: () -> Unit,
) {
    var menu by remember { mutableStateOf(false) }
    val lines = stat?.lines ?: 0
    val done = stat?.done ?: 0
    val sec = stat?.sec ?: 0f
    val frac = if (lines > 0) (done.toFloat() / lines).coerceIn(0f, 1f) else 0f

    Surface(
        shape = RoundedCornerShape(14.dp),
        color = MaterialTheme.colorScheme.surface,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(Modifier.clickable(onClick = onOpen).padding(start = 16.dp, end = 4.dp, top = 12.dp, bottom = 12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f)) {
                    Text(
                        project.title,
                        style = MaterialTheme.typography.titleMedium,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Spacer(Modifier.height(3.dp))
                    Text(
                        buildString {
                            append("$lines 句")
                            if (sec > 0f) append(" · ${mmss(sec)}")
                            if (lines > 0) append(" · $done/$lines 已生成")
                            when (project.mediaKind) {
                                "video" -> append(" · 视频配音")
                                "audio" -> append(" · 配音")
                            }
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Box {
                    IconButton(onClick = { menu = true }) {
                        Icon(
                            Icons.Default.MoreVert,
                            contentDescription = "更多",
                            modifier = Modifier.size(20.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        DropdownMenuItem(
                            text = { Text("重命名") },
                            onClick = { menu = false; onRename() },
                        )
                        DropdownMenuItem(
                            text = { Text("删除") },
                            onClick = { menu = false; onDelete() },
                        )
                    }
                }
            }
            Spacer(Modifier.height(10.dp))
            Box(
                Modifier.fillMaxWidth().height(3.dp)
                    .clip(RoundedCornerShape(2.dp))
                    .background(MaterialTheme.colorScheme.outlineVariant),
            ) {
                if (frac > 0f) {
                    Box(
                        Modifier.fillMaxWidth(frac).fillMaxHeight()
                            .background(MaterialTheme.colorScheme.primary),
                    )
                }
            }
            Spacer(Modifier.height(6.dp))
            Text(
                SimpleDateFormat("MM-dd HH:mm", Locale.getDefault()).format(Date(project.updated)),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
            )
        }
    }
}

@Composable
private fun RenameDialog(
    initial: String,
    onDismiss: () -> Unit,
    onRename: (String) -> Unit,
) {
    var title by remember { mutableStateOf(initial) }
    AlertDialog(
        onDismissRequest = onDismiss,
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
            TextButton(onClick = { onRename(title) }) { Text("保存") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("取消") }
        },
    )
}

@Composable
private fun NewProjectDialog(
    voices: List<Voice>,
    onDismiss: () -> Unit,
    onCreate: (String, Long?) -> Unit,
    onImport: (kind: String) -> Unit,
) {
    var title by remember { mutableStateOf("") }
    var voiceId by remember { mutableStateOf<Long?>(null) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("新建工程") },
        text = {
            Column(Modifier.heightIn(max = 480.dp).verticalScroll(rememberScrollState())) {
                OutlinedTextField(
                    value = title,
                    onValueChange = { title = it },
                    placeholder = { Text("工程名称（可留空）") },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )

                Spacer(Modifier.height(16.dp))
                Text("从音色创作", style = MaterialTheme.typography.labelMedium)
                Spacer(Modifier.height(6.dp))
                if (voices.isEmpty()) {
                    Text(
                        "还没有音色。先去「音色」导入一段，或建工程后再选。",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        voices.forEach { v ->
                            FilterChip(
                                selected = voiceId == v.id,
                                onClick = { voiceId = v.id },
                                label = { Text(v.name) },
                            )
                        }
                    }
                }

                Spacer(Modifier.height(16.dp))
                HorizontalDivider()
                Spacer(Modifier.height(12.dp))
                Text("配音（导入原音）", style = MaterialTheme.typography.labelMedium)
                Spacer(Modifier.height(4.dp))
                Text(
                    "按停顿切句，每段保留原音，改文本后按原时长重配。",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(onClick = { onImport("audio") }) { Text("导入音频") }
                    OutlinedButton(onClick = { onImport("video") }) { Text("导入视频") }
                }
            }
        },
        confirmButton = {
            TextButton(onClick = { onCreate(title, voiceId) }) { Text("创建") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("取消") }
        },
    )
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
    return "配音工程"
}
