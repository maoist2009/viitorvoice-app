package com.viitorvoice.demo.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val DarkColors = darkColorScheme(
    primary = Color(0xFF7FE7D2),
    onPrimary = Color(0xFF04201B),
    primaryContainer = Color(0xFF0F3F37),
    onPrimaryContainer = Color(0xFFB9F7EA),
    secondary = Color(0xFFB9C6FF),
    onSecondary = Color(0xFF0C1533),
    secondaryContainer = Color(0xFF22305C),
    onSecondaryContainer = Color(0xFFDCE1FF),
    tertiary = Color(0xFFC6A8FF),
    onTertiary = Color(0xFF1D0B3A),
    tertiaryContainer = Color(0xFF3A2160),
    onTertiaryContainer = Color(0xFFEBDCFF),
    background = Color(0xFF0A0E14),
    onBackground = Color(0xFFE6EAEF),
    surface = Color(0xFF11161E),
    onSurface = Color(0xFFE6EAEF),
    surfaceVariant = Color(0xFF1C2430),
    onSurfaceVariant = Color(0xFFAFBAC8),
    outline = Color(0xFF6C7A8C),
    outlineVariant = Color(0xFF2C3745),
    error = Color(0xFFFFB4AB),
    onError = Color(0xFF3D0503),
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF00695C),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFA7F2E4),
    onPrimaryContainer = Color(0xFF00201B),
    secondary = Color(0xFF445E91),
    onSecondary = Color.White,
    tertiary = Color(0xFF6B4EA8),
    onTertiary = Color.White,
    background = Color(0xFFF6F9FA),
    onBackground = Color(0xFF161C1B),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF161C1B),
    surfaceVariant = Color(0xFFE2EBE8),
    onSurfaceVariant = Color(0xFF3F4946),
    outline = Color(0xFF6F7977),
)

@Composable
fun ViiTorVoiceTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    MaterialTheme(
        colorScheme = if (darkTheme) DarkColors else LightColors,
        content = content,
    )
}
