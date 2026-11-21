package com.viitorvoice.demo.ui

import android.app.Activity
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * Studio shell. Home is the project list; voices / settings / the legacy bench
 * are pushed screens; the editor sits on top of everything.
 *
 * Debug hooks:
 *   --ei screen 0|1|2|3   home / voices / settings / legacy bench
 *   --ei editor <id>      open a project directly
 *   --ei timeline 1       open the editor in full-screen timeline mode
 *   --ei gen 1            ask to generate every dirty line after opening
 */
@Composable
fun AppRoot(vm: StudioViewModel = viewModel()) {
    val ctx = LocalContext.current
    val act = ctx as? Activity
    val startPage = act?.intent?.getIntExtra("screen", 0) ?: 0
    val startEditor = act?.intent?.getIntExtra("editor", 0)?.toLong()?.takeIf { it > 0 }

    var page by rememberSaveable { mutableIntStateOf(startPage.coerceIn(0, 3)) }
    var editorId by rememberSaveable { mutableStateOf(startEditor) }

    if (editorId != null) {
        EditorScreen(vm, editorId!!) { editorId = null }
        return
    }

    BackHandler(enabled = page != 0) { page = 0 }

    when (page) {
        1 -> VoicesScreen(onBack = { page = 0 })
        2 -> SettingsScreen(
            vm = vm,
            onBack = { page = 0 },
            onVoices = { page = 1 },
            onBench = { page = 3 },
        )
        3 -> Column(Modifier.fillMaxSize()) {
            ScreenBar("实验台", onBack = { page = 2 })
            Box(Modifier.weight(1f)) { HomeScreen() }
        }
        else -> ProjectListScreen(
            vm = vm,
            onOpen = { editorId = it },
            onVoices = { page = 1 },
            onSettings = { page = 2 },
        )
    }
}
