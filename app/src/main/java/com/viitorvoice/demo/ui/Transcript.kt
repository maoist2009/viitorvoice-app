package com.viitorvoice.demo.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.drag
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.viitorvoice.demo.data.Line
import com.viitorvoice.demo.data.Word

/** A word placed inside the line text (char offsets) plus its time span. */
data class RWord(
    val text: String,
    val start: Int,
    val end: Int,
    val from: Float,
    val to: Float,
)

/**
 * Resolve the word list for one line:
 *  1. persisted ASR/aligner words, located in the text in order;
 *  2. otherwise an even estimate over the line's slot (CJK per char, Latin
 *     per word) so the transcript still tracks playback before ASR runs.
 */
fun resolveWords(line: Line, seg: Seg?): List<RWord> {
    val text = line.text
    if (text.isEmpty()) return emptyList()

    if (line.words.isNotEmpty()) {
        var cursor = 0
        val out = ArrayList<RWord>(line.words.size)
        var ok = true
        for (w in line.words) {
            val i = text.indexOf(w.text, cursor)
            if (i < 0) {
                ok = false
                break
            }
            out.add(RWord(w.text, i, i + w.text.length, w.start, w.end))
            cursor = i + w.text.length
        }
        if (ok && out.isNotEmpty()) return out
    }

    val tokens = tokenize(text)
    if (tokens.isEmpty() || seg == null || seg.dur <= 0f) return emptyList()
    val step = seg.dur / tokens.size
    return tokens.mapIndexed { i, t ->
        RWord(t.first, t.second, t.second + t.first.length, seg.start + i * step, seg.start + (i + 1) * step)
    }
}

private fun isCjk(c: Char): Boolean {
    val v = c.code
    return v in 0x4E00..0x9FFF || v in 0x3400..0x4DBF || v in 0x3040..0x30FF ||
        v in 0xAC00..0xD7AF
}

/** CJK per character, Latin/digits per run; whitespace and punctuation carry no time. */
private fun tokenize(text: String): List<Pair<String, Int>> {
    val out = ArrayList<Pair<String, Int>>()
    var i = 0
    while (i < text.length) {
        val c = text[i]
        when {
            c.isWhitespace() -> i++
            isCjk(c) -> {
                out.add(c.toString() to i)
                i++
            }
            c.isLetterOrDigit() -> {
                val s = i
                while (i < text.length && text[i].isLetterOrDigit() && !isCjk(text[i])) i++
                out.add(text.substring(s, i) to s)
            }
            else -> i++
        }
    }
    return out
}

/**
 * One transcript paragraph. Tap a word to seek + select it; drag to extend the
 * selection by character; the word under the playhead is highlighted.
 */
@Composable
fun TranscriptLine(
    line: Line,
    index: Int,
    words: List<RWord>,
    playhead: Float,
    selected: Boolean,
    selStart: Int,
    selEnd: Int,
    onSelect: (Int, Int) -> Unit,
    onSeek: (Float) -> Unit,
    canPlay: Boolean,
    canSource: Boolean,
    canGen: Boolean,
    onPlay: () -> Unit,
    onSource: () -> Unit,
    onGen: () -> Unit,
    onInspect: () -> Unit,
    onDelete: () -> Unit,
) {
    var menu by remember { mutableStateOf(false) }
    val primary = MaterialTheme.colorScheme.primary
    val onSurface = MaterialTheme.colorScheme.onSurface
    val muted = MaterialTheme.colorScheme.onSurfaceVariant
    var layout by remember { mutableStateOf<TextLayoutResult?>(null) }

    val active = words.firstOrNull { playhead >= it.from && playhead < it.to }
    val annotated = remember(line.text, words, active, selStart, selEnd) {
        buildAnnotatedString {
            append(line.text)
            if (active != null) {
                addStyle(
                    SpanStyle(
                        color = primary,
                        background = primary.copy(alpha = 0.16f),
                    ),
                    active.start, active.end,
                )
            }
            if (selEnd > selStart && selStart >= 0 && selEnd <= line.text.length) {
                addStyle(
                    SpanStyle(background = primary.copy(alpha = 0.32f)),
                    selStart, selEnd,
                )
            }
        }
    }

    Column(
        Modifier.fillMaxWidth()
            .background(if (selected) primary.copy(alpha = 0.05f) else Color.Transparent)
            .padding(start = 16.dp, end = 4.dp, top = 8.dp, bottom = 8.dp),
    ) {
        Row(verticalAlignment = Alignment.Top) {
            Text(
                "${index + 1}",
                style = MaterialTheme.typography.labelSmall,
                color = if (selected) primary else muted,
                modifier = Modifier.width(24.dp).padding(top = 4.dp),
            )
            Text(
                annotated,
                style = MaterialTheme.typography.bodyLarge.copy(
                    lineHeight = 26.sp,
                    color = onSurface,
                ),
                maxLines = 6,
                overflow = TextOverflow.Ellipsis,
                onTextLayout = { layout = it },
                modifier = Modifier.weight(1f)
                    .pointerInput(line.id, line.text, words) {
                        awaitEachGesture {
                            val down = awaitFirstDown()
                            val lay = layout ?: return@awaitEachGesture
                            val off = lay.getOffsetForPosition(down.position)
                                .coerceIn(0, line.text.length)
                            val w = words.firstOrNull { off >= it.start && off < it.end }
                            if (w != null) {
                                onSelect(w.start, w.end)
                                onSeek(w.from)
                            } else {
                                onSelect(off, off)
                            }
                            val anchor = w?.start ?: off
                            drag(down.id) { change ->
                                change.consume()
                                val o = lay.getOffsetForPosition(change.position)
                                    .coerceIn(0, line.text.length)
                                onSelect(minOf(anchor, o), maxOf(anchor, o))
                            }
                        }
                    },
            )
            Box {
                IconButton(onClick = { menu = true }) {
                    Icon(
                        Icons.Default.MoreVert,
                        contentDescription = "更多",
                        modifier = Modifier.size(18.dp),
                        tint = muted,
                    )
                }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    if (canPlay) {
                        DropdownMenuItem(
                            text = { Text("试听") },
                            onClick = { menu = false; onPlay() },
                        )
                    }
                    if (canSource) {
                        DropdownMenuItem(
                            text = { Text("原音") },
                            onClick = { menu = false; onSource() },
                        )
                    }
                    if (canGen) {
                        DropdownMenuItem(
                            text = { Text("重新生成") },
                            onClick = { menu = false; onGen() },
                        )
                    }
                    DropdownMenuItem(
                        text = { Text("参数") },
                        onClick = { menu = false; onInspect() },
                    )
                    DropdownMenuItem(
                        text = { Text("删除") },
                        onClick = { menu = false; onDelete() },
                    )
                }
            }
        }
    }
}
