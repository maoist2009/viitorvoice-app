package com.viitorvoice.demo

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import com.viitorvoice.demo.ui.HomeScreen
import com.viitorvoice.demo.ui.theme.ViiTorVoiceTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ViiTorVoiceTheme {
                HomeScreen()
            }
        }
    }
}
