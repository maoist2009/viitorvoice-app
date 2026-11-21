package com.viitorvoice.demo.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.drag
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.TextButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.io.File
import com.viitorvoice.demo.data.Line
import com.viitorvoice.demo.data.LineStatus
import com.viitorvoice.demo.data.Project
import com.viitorvoice.demo.text.Duration

/** One line's slot on the shared time axis. */
data class Seg(
    val lineId: Long,
    val idx: Int,
    val start: Float,
    val dur: Float,
    val done: Boolean,
    val status: String,
)

/**
 * Time axis for the whole project: a generated line uses its take length, a
 * pending one uses the estimator, an empty one takes no time. Playback pads
 * missing takes with silence of the same length, so strip / playhead / mix
 * stay aligned.
 */
fun buildSegments(lines: List<Line>): List<Seg> {
    var cursor = 0f
    return lines.map { l ->
        val d = when {
            // A take that still matches the text is what actually plays;
            // once the line is dirty the new target length wins.
            l.takeId != null && l.takeSec > 0f && l.status == LineStatus.DONE -> l.takeSec
            l.targetSec > 0f -> l.targetSec
            l.text.isBlank() -> 0f
            else -> Duration.seconds(l.text, l.lang)
        }
        val start = if (l.offsetSec >= 0f) l.offsetSec else cursor
        cursor = maxOf(cursor, start + d)
        Seg(l.id, l.idx, start, d, l.takeId != null, l.status)
    }
}

fun mmss(sec: Float): String {
    val s = sec.coerceAtLeast(0f).toInt()
    return "%d:%02d".format(s / 60, s % 60)
}

/** Docked strip: the whole project fits the width, no scroll. */
@Composable
fun TimelineDock(
    lines: List<Line>,
    segs: List<Seg>,
    project: Project,
    playhead: Float,
    selected: Long?,
    collapsed: Boolean,
    onToggle: () -> Unit,
    onExpand: () -> Unit,
    onSeek: (Float) -> Unit,
    onPlaySeg: (Long) -> Unit,
    onSelect: (Long) -> Unit,
    onBake: () -> Unit,
    onDragClip: (Long, Float, Float) -> Unit,
    onCommitClip: (Long, Float, Float) -> Unit,
) {
    val srcSec = remember(project.srcAudio) {
        val f = File(project.srcAudio)
        if (project.srcAudio.isNotBlank() && f.isFile) (f.length() - 44) / 2f / 24000f else 0f
    }
    val total = maxOf(segs.lastOrNull()?.let { it.start + it.dur } ?: 0f, srcSec)
        .coerceAtLeast(1f)

    val primary = MaterialTheme.colorScheme.primary
    val outline = MaterialTheme.colorScheme.outline
    val secondary = MaterialTheme.colorScheme.secondary
    val lane = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.35f)
    val label = MaterialTheme.colorScheme.onSurfaceVariant
    val onSurface = MaterialTheme.colorScheme.onSurface
    val tm = rememberTextMeasurer()

    val curTotal by rememberUpdatedState(total)
    val curSrc by rememberUpdatedState(srcSec)
    val curSegs by rememberUpdatedState(segs)
    val curLines by rememberUpdatedState(lines)
    val curSelected by rememberUpdatedState(selected)
    val curCollapsed by rememberUpdatedState(collapsed)
    val curEditable by rememberUpdatedState(project.mediaKind != "video")
    val curBake by rememberUpdatedState(onBake)
    val curDrag by rememberUpdatedState(onDragClip)
    val curCommit by rememberUpdatedState(onCommitClip)
    val curSelect by rememberUpdatedState(onSelect)
    val curPlay by rememberUpdatedState(onPlaySeg)
    val curSeek by rememberUpdatedState(onSeek)

    Column(Modifier.fillMaxWidth()) {
        Row(
            Modifier.fillMaxWidth().padding(start = 20.dp, end = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                "${mmss(playhead)} / ${mmss(total)}",
                style = MaterialTheme.typography.labelSmall,
                color = label,
            )
            Spacer(Modifier.weight(1f))
            TextButton(onClick = onExpand) {
                Text("全屏", style = MaterialTheme.typography.labelMedium)
            }
            IconButton(onClick = onToggle) {
                Icon(
                    if (collapsed) Icons.Default.ExpandMore else Icons.Default.ExpandLess,
                    contentDescription = if (collapsed) "展开时间线" else "收起时间线",
                    tint = label,
                )
            }
        }

        val h = if (collapsed) 40.dp else 148.dp
        Canvas(
            Modifier.fillMaxWidth().height(h)
                .padding(horizontal = 20.dp)
                .pointerInput(Unit) {
                    awaitEachGesture {
                        val down = awaitFirstDown(requireUnconsumed = false)
                        val px = size.width / curTotal
                        fun seekTo(x: Float) {
                            curSeek((x / px).coerceIn(0f, curTotal))
                        }
                        if (curCollapsed) {
                            down.consume()
                            seekTo(down.position.x)
                            drag(down.id) { c -> c.consume(); seekTo(c.position.x) }
                            return@awaitEachGesture
                        }
                        val hasSrc = curSrc > 0f
                        val dlgTop = (if (hasSrc) 56.dp else 22.dp).toPx()
                        val dlgBot = dlgTop + 84.dp.toPx()
                        val x = down.position.x
                        val y = down.position.y
                        val hit = if (y in dlgTop..dlgBot) {
                            curSegs.firstOrNull {
                                x >= it.start * px && x <= (it.start + it.dur) * px
                            }
                        } else null
                        if (hit == null) {
                            down.consume()
                            seekTo(x)
                            drag(down.id) { c -> c.consume(); seekTo(c.position.x) }
                            return@awaitEachGesture
                        }
                        curSelect(hit.lineId)
                        val line = curLines.firstOrNull { it.id == hit.lineId }
                        if (!curEditable || line == null) {
                            down.consume()
                            curPlay(hit.lineId)
                            return@awaitEachGesture
                        }
                        val right = (hit.start + hit.dur) * px
                        val resize = x >= right - 14.dp.toPx()
                        down.consume()
                        var dx = 0f
                        var moved = false
                        var baked = false
                        drag(down.id) { change ->
                            change.consume()
                            if (!baked) {
                                baked = true
                                curBake()
                            }
                            dx += change.position.x - change.previousPosition.x
                            if (kotlin.math.abs(dx) > 2f) moved = true
                            val d = dx / px
                            if (resize) {
                                curDrag(line.id, hit.start, (hit.dur + d).coerceIn(0.3f, 30f))
                            } else {
                                curDrag(line.id, (hit.start + d).coerceAtLeast(0f), line.targetSec)
                            }
                        }
                        if (moved) {
                            val d = dx / px
                            if (resize) {
                                curCommit(line.id, hit.start, (hit.dur + d).coerceIn(0.3f, 30f))
                            } else {
                                curCommit(line.id, (hit.start + d).coerceAtLeast(0f), line.targetSec)
                            }
                        } else {
                            curPlay(line.id)
                        }
                    }
                },
        ) {
            val px = size.width / total
            val gap = 1.5.dp.toPx()

            if (collapsed) {
                drawRoundRect(
                    lane,
                    topLeft = Offset(0f, 4.dp.toPx()),
                    size = Size(size.width, size.height - 8.dp.toPx()),
                    cornerRadius = CornerRadius(6.dp.toPx()),
                )
                segs.forEach { s ->
                    if (s.dur <= 0f) return@forEach
                    val x0 = s.start * px
                    val x1 = (s.start + s.dur) * px
                    drawRoundRect(
                        color = if (s.done) primary.copy(alpha = 0.6f)
                        else outline.copy(alpha = 0.3f),
                        topLeft = Offset(x0 + gap, 9.dp.toPx()),
                        size = Size(
                            (x1 - x0 - gap * 2).coerceAtLeast(1f),
                            size.height - 18.dp.toPx(),
                        ),
                        cornerRadius = CornerRadius(3.dp.toPx()),
                    )
                }
            } else {
                // ruler
                val step = when {
                    total <= 20f -> 2f
                    total <= 60f -> 5f
                    total <= 180f -> 15f
                    else -> 30f
                }
                var s = 0f
                while (s <= total) {
                    val x = s * px
                    drawLine(
                        label.copy(alpha = 0.4f),
                        Offset(x, 16.dp.toPx()),
                        Offset(x, 19.dp.toPx()),
                        1.dp.toPx(),
                    )
                    val layout = tm.measure(
                        AnnotatedString(mmss(s)),
                        style = TextStyle(fontSize = 9.sp, color = label),
                    )
                    drawText(layout, topLeft = Offset(x + 3.dp.toPx(), 0f))
                    s += step
                }

                val hasSrc = srcSec > 0f
                if (hasSrc) {
                    val y0 = 22.dp.toPx()
                    val hh = 26.dp.toPx()
                    drawRoundRect(
                        color = outline.copy(alpha = 0.16f),
                        topLeft = Offset(0f, y0),
                        size = Size((srcSec * px).coerceAtMost(size.width), hh),
                        cornerRadius = CornerRadius(6.dp.toPx()),
                    )
                    val l = tm.measure(
                        AnnotatedString("原音"),
                        style = TextStyle(fontSize = 10.sp, color = label),
                    )
                    drawText(l, topLeft = Offset(6.dp.toPx(), y0 + 4.dp.toPx()))
                }

                val y0 = (if (hasSrc) 56.dp else 22.dp).toPx()
                val hh = 84.dp.toPx()
                drawRoundRect(
                    lane,
                    topLeft = Offset(0f, y0),
                    size = Size(size.width, hh),
                    cornerRadius = CornerRadius(8.dp.toPx()),
                )
                segs.forEach { s ->
                    if (s.dur <= 0f) return@forEach
                    val x0 = s.start * px + gap
                    val x1 = (s.start + s.dur) * px - gap
                    val w = (x1 - x0).coerceAtLeast(2f)
                    drawRoundRect(
                        color = if (s.done) primary.copy(alpha = 0.55f)
                        else outline.copy(alpha = 0.28f),
                        topLeft = Offset(x0, y0 + 8.dp.toPx()),
                        size = Size(w, hh - 16.dp.toPx()),
                        cornerRadius = CornerRadius(5.dp.toPx()),
                    )
                    if (s.lineId == selected) {
                        drawRoundRect(
                            color = primary,
                            topLeft = Offset(x0, y0 + 8.dp.toPx()),
                            size = Size(w, hh - 16.dp.toPx()),
                            cornerRadius = CornerRadius(5.dp.toPx()),
                            style = androidx.compose.ui.graphics.drawscope.Stroke(2.dp.toPx()),
                        )
                    }
                    if (w > 18.dp.toPx()) {
                        val layout = tm.measure(
                            AnnotatedString("${s.idx + 1}"),
                            style = TextStyle(fontSize = 10.sp, color = onSurface),
                        )
                        drawText(layout, topLeft = Offset(x0 + 4.dp.toPx(), y0 + 12.dp.toPx()))
                    }
                }
            }

            val ph = (playhead / total).coerceIn(0f, 1f) * size.width
            drawLine(secondary, Offset(ph, 0f), Offset(ph, size.height), 2.dp.toPx())
        }
    }
}

/**
 * Zoomable timeline: ruler, optional source lane, dialogue lane. Tap a clip to
 * play from it; drag a clip to move it, drag its right edge to change its
 * target duration. Video projects are time-locked (no dragging).
 */
@Composable
fun TimelineBody(
    lines: List<Line>,
    segs: List<Seg>,
    project: Project,
    playhead: Float,
    onSeek: (Float) -> Unit,
    onPlaySeg: (Long) -> Unit,
    onBake: () -> Unit,
    onDragClip: (Long, Float, Float) -> Unit,
    onCommitClip: (Long, Float, Float) -> Unit,
) {
    val total = segs.lastOrNull()?.let { it.start + it.dur } ?: 0f
    var pxPerSec by rememberSaveable { mutableStateOf(90f) }
    val scroll = rememberScrollState()
    val tm = rememberTextMeasurer()

    val primary = MaterialTheme.colorScheme.primary
    val outline = MaterialTheme.colorScheme.outline
    val secondary = MaterialTheme.colorScheme.secondary
    val lane = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.28f)
    val label = MaterialTheme.colorScheme.onSurfaceVariant
    val onSurface = MaterialTheme.colorScheme.onSurface

    val srcSec = remember(project.srcAudio) {
        val f = File(project.srcAudio)
        if (project.srcAudio.isNotBlank() && f.isFile) (f.length() - 44) / 2f / 24000f else 0f
    }
    val editable = project.mediaKind != "video"

    val curSegs by rememberUpdatedState(segs)
    val curLines by rememberUpdatedState(lines)
    val curPx by rememberUpdatedState(pxPerSec)
    val curEditable by rememberUpdatedState(editable)
    val curBake by rememberUpdatedState(onBake)
    val curDrag by rememberUpdatedState(onDragClip)
    val curCommit by rememberUpdatedState(onCommitClip)

    Column(Modifier.fillMaxSize()) {
        Row(
            Modifier.fillMaxWidth().padding(horizontal = 16.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("时间线 · ${mmss(total)}", style = MaterialTheme.typography.labelMedium)
            if (srcSec > 0f) {
                Spacer(Modifier.width(8.dp))
                Text(
                    "原音 ${mmss(srcSec)}",
                    style = MaterialTheme.typography.labelSmall,
                    color = label,
                )
            }
            Spacer(Modifier.weight(1f))
            Text("缩放", style = MaterialTheme.typography.labelSmall, color = label)
            Slider(
                value = pxPerSec,
                onValueChange = { pxPerSec = it },
                valueRange = 30f..300f,
                modifier = Modifier.width(150.dp),
            )
        }

        Box(Modifier.fillMaxWidth().weight(1f).horizontalScroll(scroll)) {
            val widthDp = (maxOf(total, srcSec) * pxPerSec).dp.coerceAtLeast(240.dp)
            Column(Modifier.width(widthDp)) {
                Canvas(Modifier.fillMaxWidth().height(24.dp)) {
                    val step = when {
                        pxPerSec >= 120f -> 1
                        pxPerSec >= 60f -> 5
                        else -> 10
                    }
                    var s = 0
                    while (s <= maxOf(total, srcSec).toInt() + step) {
                        val x = s * pxPerSec
                        if (x > size.width) break
                        drawLine(
                            label.copy(alpha = 0.5f),
                            Offset(x, size.height - 6.dp.toPx()),
                            Offset(x, size.height),
                            1.dp.toPx(),
                        )
                        if (s % (step * 2) == 0) {
                            val layout = tm.measure(
                                AnnotatedString(mmss(s.toFloat())),
                                style = TextStyle(fontSize = 9.sp, color = label),
                            )
                            drawText(layout, topLeft = Offset(x + 3.dp.toPx(), 1.dp.toPx()))
                        }
                        s += step
                    }
                }

                val hasSrc = srcSec > 0f
                val srcTop = 8.dp
                val srcH = 34.dp
                val dlgTop = if (hasSrc) srcTop + srcH + 8.dp else 8.dp
                val dlgH = 84.dp
                val trackH = dlgTop + dlgH + 8.dp

                Canvas(
                    Modifier.fillMaxWidth().height(trackH)
                        .pointerInput(Unit) {
                            detectTapGestures { off ->
                                val sec = off.x / curPx
                                val seg = curSegs.firstOrNull {
                                    sec >= it.start && sec < it.start + it.dur
                                }
                                if (seg != null) onPlaySeg(seg.lineId) else onSeek(sec)
                            }
                        }
                        .pointerInput(Unit) {
                            val dlgTopPx = dlgTop.toPx()
                            val dlgBotPx = (dlgTop + dlgH).toPx()
                            val edgePx = 16.dp.toPx()
                            awaitEachGesture {
                                val down = awaitFirstDown(requireUnconsumed = false)
                                if (!curEditable) return@awaitEachGesture
                                val x = down.position.x
                                val y = down.position.y
                                if (y < dlgTopPx || y > dlgBotPx) return@awaitEachGesture
                                val px = curPx
                                val hit = curSegs.firstOrNull {
                                    x >= it.start * px && x <= (it.start + it.dur) * px
                                } ?: return@awaitEachGesture
                                val line = curLines.firstOrNull { it.id == hit.lineId }
                                    ?: return@awaitEachGesture
                                val right = (hit.start + hit.dur) * px
                                val resize = x >= right - edgePx
                                down.consume()
                                curBake()
                                var dx = 0f
                                var moved = false
                                drag(down.id) { change ->
                                    change.consume()
                                    dx += change.position.x - change.previousPosition.x
                                    moved = true
                                    val d = dx / px
                                    if (resize) {
                                        curDrag(line.id, hit.start, (hit.dur + d).coerceIn(0.3f, 30f))
                                    } else {
                                        curDrag(line.id, (hit.start + d).coerceAtLeast(0f), line.targetSec)
                                    }
                                }
                                if (moved) {
                                    val d = dx / px
                                    if (resize) {
                                        curCommit(line.id, hit.start, (hit.dur + d).coerceIn(0.3f, 30f))
                                    } else {
                                        curCommit(line.id, (hit.start + d).coerceAtLeast(0f), line.targetSec)
                                    }
                                }
                            }
                        },
                ) {
                    val gap = 2.dp.toPx()

                    if (hasSrc) {
                        val y0 = srcTop.toPx()
                        val h = srcH.toPx()
                        drawRoundRect(
                            color = outline.copy(alpha = 0.18f),
                            topLeft = Offset(0f, y0),
                            size = Size((srcSec * curPx).coerceAtMost(size.width), h),
                            cornerRadius = CornerRadius(8.dp.toPx()),
                        )
                        val l = tm.measure(
                            AnnotatedString("原音"),
                            style = TextStyle(fontSize = 10.sp, color = label),
                        )
                        drawText(l, topLeft = Offset(6.dp.toPx(), y0 + 4.dp.toPx()))
                    }

                    val y0 = dlgTop.toPx()
                    val h = dlgH.toPx()
                    drawRoundRect(
                        lane,
                        topLeft = Offset(0f, y0),
                        size = Size(size.width, h),
                        cornerRadius = CornerRadius(8.dp.toPx()),
                    )
                    segs.forEach { s ->
                        if (s.dur <= 0f) return@forEach
                        val x0 = s.start * curPx + gap
                        val x1 = (s.start + s.dur) * curPx - gap
                        val w = (x1 - x0).coerceAtLeast(2f)
                        drawRoundRect(
                            color = if (s.done) primary.copy(alpha = 0.55f)
                            else outline.copy(alpha = 0.25f),
                            topLeft = Offset(x0, y0 + 8.dp.toPx()),
                            size = Size(w, h - 16.dp.toPx()),
                            cornerRadius = CornerRadius(5.dp.toPx()),
                        )
                        if (w > 18.dp.toPx()) {
                            val layout = tm.measure(
                                AnnotatedString("${s.idx + 1}"),
                                style = TextStyle(fontSize = 10.sp, color = onSurface),
                            )
                            drawText(layout, topLeft = Offset(x0 + 4.dp.toPx(), y0 + 12.dp.toPx()))
                        }
                    }
                    val px = (playhead * curPx).coerceIn(0f, size.width)
                    drawLine(secondary, Offset(px, 0f), Offset(px, size.height), 2.dp.toPx())
                }
            }
        }
    }
}

/** Full-screen timeline: pinch-free zoom slider + horizontal scroll. */
@Composable
fun TimelineScreen(st: EditorState, playhead: Float, vm: StudioViewModel, onBack: () -> Unit) {
    val segs = remember(st.lines) { buildSegments(st.lines) }
    Column(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        ScreenBar("时间线", onBack)
        TimelineBody(
            lines = st.lines,
            segs = segs,
            project = st.project,
            playhead = playhead,
            onSeek = { vm.seekTo(it) },
            onPlaySeg = { vm.playAt(it) },
            onBake = { vm.bakeOffsets() },
            onDragClip = { id, o, t -> vm.dragClip(id, o, t) },
            onCommitClip = { id, o, t -> vm.commitClip(id, o, t) },
        )
    }
}
