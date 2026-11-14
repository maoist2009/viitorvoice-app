package com.viitorvoice.demo

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/**
 * Debug entry point for the headless local-edit run.
 *
 *   am broadcast -a com.viitorvoice.demo.RUN_EDIT \
 *       -e text "修改后的整句" -e a 23 -e b 37 --ef cfg 2.0
 *
 * The receiver only forwards to [EditService]: a multi-minute NPU run does not
 * fit in the broadcast timeout.
 */
class EditReceiver : BroadcastReceiver() {
    override fun onReceive(ctx: Context, intent: Intent) {
        if (intent.action != ACTION) return
        ctx.startForegroundService(Intent(ctx, EditService::class.java).putExtras(intent))
    }

    companion object {
        const val ACTION = "com.viitorvoice.demo.RUN_EDIT"

        fun copyOf(intent: Intent?): Intent = Intent().putExtras(intent ?: Intent())
    }
}
