package com.viitorvoice.demo.data

import android.content.Context

/** Persisted UI preferences (not project data). */
class Prefs(ctx: Context) {
    private val sp = ctx.getSharedPreferences("studio_prefs", Context.MODE_PRIVATE)

    /** Ask before spending a model run. */
    var confirmGen: Boolean
        get() = sp.getBoolean("confirm_gen", true)
        set(v) {
            sp.edit().putBoolean("confirm_gen", v).apply()
        }

    /** Show advanced generation knobs and per-run details. */
    var devMode: Boolean
        get() = sp.getBoolean("dev_mode", false)
        set(v) {
            sp.edit().putBoolean("dev_mode", v).apply()
        }

    companion object {
        @Volatile
        private var inst: Prefs? = null

        fun get(ctx: Context): Prefs = inst ?: synchronized(this) {
            inst ?: Prefs(ctx.applicationContext).also { inst = it }
        }
    }
}
