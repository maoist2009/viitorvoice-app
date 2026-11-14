package com.viitorvoice.demo

import android.app.Application

class VvApp : Application() {
    override fun onCreate() {
        super.onCreate()
        Engine.attach(this)
    }
}
