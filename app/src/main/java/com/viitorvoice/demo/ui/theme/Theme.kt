package com.viitorvoice.demo.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/*
 * Both schemes define the **full** Material 3 role set. Leaving a role out
 * makes Material components fall back to the baseline purple palette, which is
 * very visible in dark mode: AlertDialog uses surfaceContainerHigh, DropdownMenu
 * uses surfaceContainer, Snackbar's action uses inversePrimary, selected chips
 * use secondaryContainer. The values below are the same blue-grey family as the
 * hand-drawn timeline/transcript surfaces.
 */

private val DarkColors = darkColorScheme(
    primary = Color(0xFF7FE7D2),
    onPrimary = Color(0xFF04201B),
    primaryContainer = Color(0xFF0F3F37),
    onPrimaryContainer = Color(0xFFB9F7EA),
    inversePrimary = Color(0xFF00695C),
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
    surfaceTint = Color(0xFF7FE7D2),
    surfaceDim = Color(0xFF0A0E14),
    surfaceBright = Color(0xFF2A3340),
    surfaceContainerLowest = Color(0xFF070A0F),
    surfaceContainerLow = Color(0xFF0E131A),
    surfaceContainer = Color(0xFF141A23),
    surfaceContainerHigh = Color(0xFF1A222C),
    surfaceContainerHighest = Color(0xFF212B37),
    inverseSurface = Color(0xFFE6EAEF),
    inverseOnSurface = Color(0xFF1A222C),
    error = Color(0xFFFFB4AB),
    onError = Color(0xFF3D0503),
    errorContainer = Color(0xFF5C1A16),
    onErrorContainer = Color(0xFFFFDAD6),
    outline = Color(0xFF6C7A8C),
    outlineVariant = Color(0xFF2C3745),
    scrim = Color(0xFF000000),
)

private val LightColors = lightColorScheme(
    primary = Color(0xFF00695C),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFA7F2E4),
    onPrimaryContainer = Color(0xFF00201B),
    inversePrimary = Color(0xFF7FE7D2),
    secondary = Color(0xFF445E91),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFDCE1FF),
    onSecondaryContainer = Color(0xFF0C1533),
    tertiary = Color(0xFF6B4EA8),
    onTertiary = Color.White,
    tertiaryContainer = Color(0xFFEBDCFF),
    onTertiaryContainer = Color(0xFF1D0B3A),
    background = Color(0xFFF6F9FA),
    onBackground = Color(0xFF161C1B),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF161C1B),
    surfaceVariant = Color(0xFFE2EBE8),
    onSurfaceVariant = Color(0xFF3F4946),
    surfaceTint = Color(0xFF00695C),
    surfaceDim = Color(0xFFDDE3E6),
    surfaceBright = Color(0xFFF6F9FA),
    surfaceContainerLowest = Color(0xFFFFFFFF),
    surfaceContainerLow = Color(0xFFF0F4F5),
    surfaceContainer = Color(0xFFE9EFF0),
    surfaceContainerHigh = Color(0xFFE2EBE8),
    surfaceContainerHighest = Color(0xFFDCE5E3),
    inverseSurface = Color(0xFF161C1B),
    inverseOnSurface = Color(0xFFF0F4F5),
    error = Color(0xFFB3261E),
    onError = Color.White,
    errorContainer = Color(0xFFFFDAD6),
    onErrorContainer = Color(0xFF410002),
    outline = Color(0xFF6F7977),
    outlineVariant = Color(0xFFBFC9C6),
    scrim = Color(0xFF000000),
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
