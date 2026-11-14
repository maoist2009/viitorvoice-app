package com.viitorvoice.demo

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import java.io.File

/**
 * Headless entry point for `am broadcast` smoke runs. Debug builds only.
 *
 * Results go to filesDir/smoke.txt as well as logcat, because logcat is not
 * reliable from this Termux setup.
 */
class SmokeReceiver : BroadcastReceiver() {
    override fun onReceive(ctx: Context, intent: Intent) {
        if (intent.action != ACTION) return
        val pending = goAsync()
        Thread {
            val tag = "VvSmoke"
            val log = StringBuilder()
            var result = "?"
            try {
                val out = File(ctx.filesDir, "smoke.wav")
                result = Engine.decodeSample(ctx, out)
                log.appendLine("decode: $result")
                log.appendLine("wav: ${out.absolutePath} ${out.length()} B")
                log.appendLine("timings: ${Native.nativeTimings()}")
                log.appendLine("version: ${Native.nativeVersion()}")
            } catch (t: Throwable) {
                log.appendLine("FAIL: ${t.javaClass.simpleName}: ${t.message}")
                Log.e(tag, "fail", t)
            } finally {
                File(ctx.filesDir, "smoke.txt").writeText(log.toString())
                Log.i(tag, log.toString())
                pending.finish()
            }
        }.start()
    }

    companion object {
        const val ACTION = "com.viitorvoice.demo.RUN_SMOKE"
    }
}
