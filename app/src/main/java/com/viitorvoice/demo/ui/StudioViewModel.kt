package com.viitorvoice.demo.ui

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.viitorvoice.demo.Engine
import com.viitorvoice.demo.Native
import com.viitorvoice.demo.audio.AudioIo
import com.viitorvoice.demo.audio.Player
import com.viitorvoice.demo.audio.Render
import com.viitorvoice.demo.audio.WavFile
import com.viitorvoice.demo.data.Line
import com.viitorvoice.demo.data.Prefs
import com.viitorvoice.demo.data.LineStatus
import com.viitorvoice.demo.data.Project
import com.viitorvoice.demo.data.ProjectStat
import com.viitorvoice.demo.data.Store
import com.viitorvoice.demo.asr.AsrClient
import com.viitorvoice.demo.data.Voice
import com.viitorvoice.demo.data.Word
import com.viitorvoice.demo.text.Duration
import com.viitorvoice.demo.text.Prompt
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class EditorState(
    val project: Project,
    val lines: List<Line>,
    val busy: Boolean = false,
    val running: Long? = null,
    val stage: String = "",
    val progress: Float = 0f,
    val playingLine: Long? = null,
    val playingAll: Boolean = false,
    val error: String? = null,
    val log: List<String> = emptyList(),
)

/** Studio state: project list, voice library, and the open editor. */
class StudioViewModel(app: Application) : AndroidViewModel(app) {

    private val store = Store.get(app)
    private val prefs = Prefs.get(app)

    private val _confirmGen = MutableStateFlow(prefs.confirmGen)
    val confirmGen: StateFlow<Boolean> = _confirmGen.asStateFlow()

    private val _devMode = MutableStateFlow(prefs.devMode)
    val devMode: StateFlow<Boolean> = _devMode.asStateFlow()

    private val _projects = MutableStateFlow<List<Project>>(emptyList())
    val projects: StateFlow<List<Project>> = _projects.asStateFlow()

    private val _importing = MutableStateFlow<String?>(null)

    /** Non-null while an imported clip is being transcribed (shown in the editor). */
    val importing: StateFlow<String?> = _importing.asStateFlow()

    private val _stats = MutableStateFlow<Map<Long, ProjectStat>>(emptyMap())
    val stats: StateFlow<Map<Long, ProjectStat>> = _stats.asStateFlow()

    private val _voices = MutableStateFlow<List<Voice>>(emptyList())
    val voices: StateFlow<List<Voice>> = _voices.asStateFlow()

    private val _editor = MutableStateFlow<EditorState?>(null)
    val editor: StateFlow<EditorState?> = _editor.asStateFlow()

    /** voice id -> encoded reference tokens (encoder is released after use). */
    private val refTokens = HashMap<Long, IntArray>()

    /** dubbing source path -> (tokens, frames), same encoder lifecycle. */
    private val srcTokens = HashMap<String, Pair<IntArray, Int>>()

    /** Pending word-level edit per line: [startSec, endSec] on the project timeline. */
    private val editSpan = HashMap<Long, Pair<Float, Float>>()

    private var saveJob: Job? = null
    private var genJob: Job? = null

    init { refresh() }

    fun refresh() {
        viewModelScope.launch(Dispatchers.IO) {
            _projects.value = store.projects()
            _stats.value = store.stats()
            _voices.value = store.voices()
        }
    }

    // ------------------------------------------------------------- projects
    fun createProject(title: String, voiceId: Long?): Long {
        val id = store.addProject(title.trim().ifBlank { "未命名" }, voiceId)
        store.addLine(id, 0)
        refresh()
        return id
    }

    fun deleteProject(id: Long) {
        viewModelScope.launch(Dispatchers.IO) {
            store.deleteProject(id)
            refresh()
        }
    }

    fun lineCount(projectId: Long): Int = store.lineCount(projectId)

    fun setConfirmGen(v: Boolean) {
        prefs.confirmGen = v
        _confirmGen.value = v
    }

    fun setDevMode(v: Boolean) {
        prefs.devMode = v
        _devMode.value = v
    }

    fun renameProject(id: Long, title: String) {
        val p = store.project(id) ?: return
        store.updateProject(p.copy(title = title.trim().ifBlank { "未命名" }))
        refresh()
    }

    fun openProject(id: Long) {
        viewModelScope.launch(Dispatchers.IO) {
            val p = store.project(id) ?: return@launch
            _editor.value = EditorState(p, store.lines(id))
        }
    }

    fun closeProject() {
        saveJob?.cancel()
        flushLines()
        _editor.value = null
        refresh()
    }

    fun setProjectVoice(voiceId: Long?) {
        val st = _editor.value ?: return
        val p = st.project.copy(voiceId = voiceId)
        _editor.value = st.copy(project = p)
        viewModelScope.launch(Dispatchers.IO) { store.updateProject(p) }
    }

    // ---------------------------------------------------------------- lines
    fun setLineText(id: Long, text: String) {
        val st = _editor.value ?: return
        val lines = st.lines.map { l ->
            if (l.id != id) l else l.copy(
                text = text,
                words = if (text == l.text) l.words else emptyList(),
                status = when {
                    text.isBlank() -> LineStatus.EMPTY
                    l.takeId != null && l.genText == text -> LineStatus.DONE
                    else -> LineStatus.DIRTY
                },
            )
        }
        _editor.value = st.copy(lines = lines)
        scheduleSave()
    }

    /**
     * Replace [start, end) in a line's text. Words fully covered by the edit
     * collapse into one word that keeps the combined time span, so the
     * transcript stays aligned without re-running the aligner.
     */
    fun replaceText(id: Long, start: Int, end: Int, newText: String) {
        val st = _editor.value ?: return
        val l = st.lines.firstOrNull { it.id == id } ?: return
        val s = start.coerceIn(0, l.text.length)
        val e = end.coerceIn(s, l.text.length)
        val text = l.text.substring(0, s) + newText + l.text.substring(e)
        val (words, span) = spliceWords(l.words, l.text, s, e, newText)
        if (span != null) {
            val prev = editSpan[id]
            editSpan[id] = if (prev == null) span
            else minOf(prev.first, span.first) to maxOf(prev.second, span.second)
        }
        val upd = l.copy(
            text = text,
            words = words,
            status = when {
                text.isBlank() -> LineStatus.EMPTY
                l.takeId != null && l.genText == text -> LineStatus.DONE
                else -> LineStatus.DIRTY
            },
        )
        _editor.value = st.copy(lines = st.lines.map { if (it.id == id) upd else it })
        scheduleSave()
    }

    private fun spliceWords(
        words: List<Word>,
        oldText: String,
        s: Int,
        e: Int,
        newText: String,
    ): Pair<List<Word>, Pair<Float, Float>?> {
        if (words.isEmpty()) return emptyList<Word>() to null
        var cursor = 0
        val ranges = ArrayList<Triple<Int, Int, Word>>(words.size)
        for (w in words) {
            if (w.text.isEmpty()) return emptyList<Word>() to null
            val i = oldText.indexOf(w.text, cursor)
            if (i < 0) return emptyList<Word>() to null
            ranges.add(Triple(i, i + w.text.length, w))
            cursor = i + w.text.length
        }
        val inside = ranges.filter { it.first >= s && it.second <= e }
        if (inside.isEmpty()) return emptyList<Word>() to null
        val out = ArrayList<Word>(words.size)
        var inserted = false
        for (r in ranges) {
            val (a, b, w) = r
            if (a >= s && b <= e) {
                if (!inserted) {
                    inserted = true
                    if (newText.isNotEmpty()) {
                        out.add(Word(newText, inside.first().third.start, inside.last().third.end))
                    }
                }
            } else {
                out.add(w)
            }
        }
        return out to (inside.first().third.start to inside.last().third.end)
    }

    fun setLineParams(
        id: Long,
        targetSec: Float,
        voiceId: Long?,
        cfg: Float,
        emotion: Float,
        nvv: Float,
        lang: String,
    ) {
        val st = _editor.value ?: return
        val l = st.lines.firstOrNull { it.id == id } ?: return
        val upd = l.copy(
            targetSec = targetSec,
            voiceId = voiceId,
            cfg = cfg,
            emotion = emotion,
            nvv = nvv,
            lang = lang,
            status = if (l.text.isBlank()) LineStatus.EMPTY else LineStatus.DIRTY,
        )
        _editor.value = st.copy(lines = st.lines.map { if (it.id == id) upd else it })
        viewModelScope.launch(Dispatchers.IO) { store.updateLine(upd) }
    }

    fun addLine() {
        val st = _editor.value ?: return
        viewModelScope.launch(Dispatchers.IO) {
            store.addLine(st.project.id, st.lines.size)
            val lines = store.lines(st.project.id)
            _editor.update { it?.copy(lines = lines) }
        }
    }

    fun deleteLine(id: Long) {
        val st = _editor.value ?: return
        if (st.project.mediaKind == "video") {
            _editor.update { it?.copy(error = "视频工程不能删句，只能静音") }
            return
        }
        viewModelScope.launch(Dispatchers.IO) {
            store.deleteLine(id)
            val lines = store.lines(st.project.id).mapIndexed { i, l -> l.copy(idx = i) }
            lines.forEach { store.updateLine(it) }
            _editor.update { it?.copy(lines = lines) }
        }
    }

    private fun scheduleSave() {
        saveJob?.cancel()
        saveJob = viewModelScope.launch {
            delay(400)
            flushLines()
        }
    }

    private fun flushLines() {
        val lines = _editor.value?.lines ?: return
        viewModelScope.launch(Dispatchers.IO) { lines.forEach { store.updateLine(it) } }
    }

    // ----------------------------------------------------------- generation
    /** Lines that still need a model run. */
    fun pendingIds(): List<Long> {
        val st = _editor.value ?: return emptyList()
        return st.lines.filter {
            it.text.isNotBlank() &&
                (it.status == LineStatus.DIRTY || it.status == LineStatus.FAILED ||
                    it.status == LineStatus.EMPTY)
        }.map { it.id }
    }

    /** Single entry point for every model run (confirm dialog calls this). */
    fun generate(ids: List<Long>, force: Boolean = false, cfgOverride: Float? = null) {
        if (genJob?.isActive == true) return
        val st = _editor.value ?: return
        val todo = ids.mapNotNull { id -> st.lines.firstOrNull { it.id == id } }
            .filter {
                it.text.isNotBlank() &&
                    (force || it.status == LineStatus.DIRTY ||
                        it.status == LineStatus.FAILED || it.status == LineStatus.EMPTY)
            }
        if (todo.isEmpty()) return
        genJob = viewModelScope.launch(Dispatchers.IO) {
            _editor.update { it?.copy(busy = true) }
            todo.forEach { generateOne(it.id, cfgOverride) }
            _editor.update {
                it?.copy(busy = false, running = null, stage = "", progress = 0f)
            }
        }
    }

    fun generateAll() = generate(pendingIds())

    fun regenerate(id: Long) = generate(listOf(id), force = true)

    fun cancelGeneration() {
        genJob?.cancel()
        genJob = null
        _editor.update { st ->
            if (st == null) return@update null
            st.copy(
                busy = false, running = null, stage = "", progress = 0f,
                lines = st.lines.map {
                    if (it.status == LineStatus.RUNNING || it.status == LineStatus.QUEUED) {
                        it.copy(status = LineStatus.DIRTY)
                    } else it
                },
            )
        }
    }

    private suspend fun generateOne(lineId: Long, cfgOverride: Float? = null) {
        val st = _editor.value ?: return
        val line = st.lines.firstOrNull { it.id == lineId } ?: return
        if (line.text.isBlank()) return
        val useCfg = cfgOverride ?: line.cfg

        val src = line.sourceWav.takeIf { it.isNotBlank() }
            ?.let(::File)?.takeIf { it.isFile }
        val voiceId = line.voiceId ?: st.project.voiceId
        val voice = if (src == null) {
            voiceId?.let { id -> store.voices().firstOrNull { it.id == id } }
        } else null
        if (src == null && voice == null) {
            setStatus(lineId, LineStatus.FAILED)
            addLog("第 ${line.idx + 1} 句：先给工程选一个音色")
            return
        }

        setStatus(lineId, LineStatus.RUNNING)
        _editor.update { it?.copy(running = lineId, stage = "第 ${line.idx + 1} 句", progress = 0f) }
        try {
            Engine.ensureInit().let { if (it.isNotEmpty()) error(it) }
            Engine.ensureNar().let { if (it.isNotEmpty()) error(it) }

            if (src != null) {
                // Dubbing line: the source clip is the voice. Unchanged text
                // reuses it as-is (KEEP); an edit re-reads it in place.
                if (line.text == line.sourceText && line.sourceText.isNotBlank()) {
                    val out = store.newTakeFile(lineId)
                    src.copyTo(out, overwrite = true)
                    saveTake(line, out, useCfg)
                    return
                }
                // Cumulative edits: once a take exists it becomes the base, so
                // the previous word replacements survive the next one.
                val takeFile = line.takeId
                    ?.let { tid -> store.takes(lineId).firstOrNull { it.id == tid } }
                    ?.let { File(it.wavPath) }?.takeIf { it.isFile }
                val base = takeFile ?: src
                val (tokens, frames) = srcTokens[base.absolutePath]
                    ?: encodeFile(base).also { srcTokens[base.absolutePath] = it }
                val span = editSpan.remove(lineId)
                val (maskStart, maskEnd) = if (span != null) {
                    val a = ((span.first - line.offsetSec) * 25f).toInt().coerceIn(0, frames)
                    val b = ((span.second - line.offsetSec) * 25f).toInt().coerceIn(a, frames)
                    if (b > a) a to b else 0 to frames
                } else 0 to frames
                val prep = Engine.editPrepare(
                    getApplication(), tokens, frames, maskStart, maskEnd, line.text,
                    cfg = useCfg, lang = line.lang,
                    emotionScale = line.emotion, nvvScale = line.nvv,
                )
                if (prep != "OK") error(prep)
            } else {
                val ref = refTokens[voice!!.id] ?: encodeVoice(voice)
                val refFrames = ref.size / 12
                val tok = Engine.tokenizer(getApplication())
                val lang = line.lang
                val frames = if (line.targetSec > 0f) (line.targetSec * 25f).toInt()
                else Duration.frames(line.text, lang)
                val plan = Prompt.planClone(
                    tok, line.text, voice.refText.ifBlank { null }, refFrames, lang, frames,
                    Engine.maxWindow(),
                ) ?: error("文本太长，窗口放不下")

                val trimmed = Engine.trimRef(ref, refFrames, plan.refFrames)
                val prep = Engine.clonePrepare(
                    getApplication(), line.text, voice.refText.ifBlank { null }, trimmed,
                    plan.refFrames, plan.frames, lang, useCfg, line.emotion, line.nvv,
                )
                if (prep != "OK") error(prep)
            }

            runSteps(lineId)
            val out = store.newTakeFile(lineId)
            val r = Engine.finishToWav(out)
            if (!r.startsWith("OK")) error(r)
            saveTake(line, out, useCfg)
        } catch (c: kotlinx.coroutines.CancellationException) {
            setStatus(lineId, LineStatus.DIRTY)
            throw c
        } catch (t: Throwable) {
            setStatus(lineId, LineStatus.FAILED)
            addLog("第 ${line.idx + 1} 句失败: ${t.message}")
        }
    }

    private suspend fun runSteps(lineId: Long) {
        val steps = Native.narSteps()
        for (s in 0 until steps) {
            kotlinx.coroutines.currentCoroutineContext().ensureActive()
            if (Engine.editStep(s) < 0) error(Native.nativeNarError())
            _editor.update { it?.copy(progress = (s + 1f) / steps) }
        }
    }

    private fun saveTake(line: Line, out: File, cfg: Float) {
        val sec = (out.length() - 44) / 2f / AudioIo.SR
        val params = "cfg=%.1f emo=%.1f nvv=%.1f".format(cfg, line.emotion, line.nvv)
        val takeId = store.addTake(line.id, out.absolutePath, params, sec)
        val done = line.copy(status = LineStatus.DONE, takeId = takeId, genText = line.text)
        store.updateLine(done)
        updateLine(done)
        addLog("第 ${line.idx + 1} 句完成 · %.2f s".format(sec))
    }

    private fun encodeFile(f: File): Pair<IntArray, Int> {
        Engine.ensureEncoder().let { if (it.isNotEmpty()) error(it) }
        val wav = WavFile.read(f)
        val pcm = resample(wav.samples, wav.sampleRate, AudioIo.SR)
        val cut = if (pcm.size > Engine.ENC_MAX_SAMPLES) pcm.copyOf(Engine.ENC_MAX_SAMPLES) else pcm
        val r = Engine.encode(cut)
        Engine.releaseEncoder()
        return r
    }

    private fun encodeVoice(v: Voice): IntArray {
        val (tokens, _) = encodeFile(File(v.wavPath))
        refTokens[v.id] = tokens
        return tokens
    }

    private fun resample(pcm: FloatArray, srIn: Int, srOut: Int): FloatArray {
        if (srIn == srOut) return pcm
        val cap = (pcm.size.toLong() * srOut / srIn).toInt() + 8
        val dst = FloatArray(cap)
        val n = Native.nativeResample(pcm, pcm.size, srIn, srOut, dst)
        require(n > 0) { "重采样失败" }
        return dst.copyOf(n)
    }

    // --------------------------------------------------------------- clips
    /** Materialise every line's current start so dragging one clip does not
     *  ripple the rest. */
    fun bakeOffsets() {
        val st = _editor.value ?: return
        if (st.lines.all { it.offsetSec >= 0f }) return
        val segs = buildSegments(st.lines)
        val updated = st.lines.mapIndexed { i, l -> l.copy(offsetSec = segs[i].start) }
        _editor.value = st.copy(lines = updated)
        viewModelScope.launch(Dispatchers.IO) { updated.forEach { store.updateLine(it) } }
    }

    /** Live drag feedback; not persisted until [commitClip]. */
    fun dragClip(id: Long, offsetSec: Float, targetSec: Float) {
        _editor.update { st ->
            st?.copy(
                lines = st.lines.map {
                    if (it.id != id) it
                    else {
                        val retime = it.takeId != null && it.text.isNotBlank() &&
                            kotlin.math.abs(targetSec - it.targetSec) > 0.001f
                        it.copy(
                            offsetSec = offsetSec.coerceAtLeast(0f),
                            targetSec = targetSec,
                            status = if (retime) LineStatus.DIRTY else it.status,
                        )
                    }
                },
            )
        }
    }

    fun commitClip(id: Long, offsetSec: Float, targetSec: Float) {
        val st = _editor.value ?: return
        val l = st.lines.firstOrNull { it.id == id } ?: return
        val t = targetSec.coerceIn(0f, 30f)
        val retimed = l.takeId != null && kotlin.math.abs(t - l.takeSec) > 0.05f &&
            l.text.isNotBlank()
        val upd = l.copy(
            offsetSec = offsetSec.coerceAtLeast(0f),
            targetSec = t,
            status = if (retimed) LineStatus.DIRTY else l.status,
        )
        _editor.value = st.copy(lines = st.lines.map { if (it.id == id) upd else it })
        viewModelScope.launch(Dispatchers.IO) { store.updateLine(upd) }
    }

    /** Tap on a clip: source clip if it has no take yet, otherwise the mix. */
    fun playAt(lineId: Long) {
        val l = _editor.value?.lines?.firstOrNull { it.id == lineId } ?: return
        if (l.takeId == null && l.sourceWav.isNotBlank()) playSource(lineId)
        else playAll(fromLineId = lineId)
    }

    fun playSource(id: Long) {
        val line = _editor.value?.lines?.firstOrNull { it.id == id } ?: return
        val f = File(line.sourceWav)
        if (!f.isFile) return
        playJob?.cancel()
        _playhead.value = 0f
        Player.play(f)
        _editor.update { it?.copy(playingLine = id, playingAll = false) }
        viewModelScope.launch {
            while (Player.isPlaying()) delay(150)
            _editor.update { it?.copy(playingLine = null) }
        }
    }

    // ------------------------------------------------------------- dubbing
    fun createDubbing(title: String, kind: String, uri: Uri): Long =
        store.addProject(title.trim().ifBlank { "配音工程" }, null, uri.toString(), kind)

    /**
     * Decode the imported audio/video, keep a 24 kHz mono source, split it on
     * pauses and create one line per slot (source slice + its slot duration).
     * The line text starts empty — transcription is not wired yet.
     */
    fun importDubbing(projectId: Long, uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            _importing.value = "正在转写…"
            try {
                val (pcm, rate) = AudioIo.decode(getApplication(), uri)
                val mono = resample(pcm, rate, AudioIo.SR)
                val dir = store.projectDir(projectId)
                val asrWav = File(dir, "asr16.wav")
                AudioIo.writeWav(asrWav, resample(mono, AudioIo.SR, 16000), 16000)
                addLog("转写中…")
                Engine.releaseEncoder()
                val segs = AsrClient.transcribe(getApplication(), asrWav)
                val n = buildDubbingProject(store, projectId, mono, segs)
                refresh()
                if (_editor.value?.project?.id == projectId) openProject(projectId)
                addLog(
                    if (segs.isEmpty()) "配音导入完成 · $n 段（未转写，按静音切）"
                    else "配音导入完成 · $n 段（已转写）",
                )
            } catch (t: Throwable) {
                store.deleteProject(projectId)
                refresh()
                _editor.update { it?.copy(error = "导入失败: ${t.message}") }
            } finally {
                _importing.value = null
            }
        }
    }

    // ------------------------------------------------------------ playback
    private var playJob: Job? = null

    private val _playhead = MutableStateFlow(0f)

    /**
     * Mix playback position in seconds. Kept out of [EditorState] so the
     * 12 Hz playhead tick does not recompose the whole line list.
     */
    val playhead: StateFlow<Float> = _playhead.asStateFlow()

    fun playLine(id: Long) {
        val line = _editor.value?.lines?.firstOrNull { it.id == id } ?: return
        val take = line.takeId?.let { tid -> store.takes(id).firstOrNull { it.id == tid } }
            ?: return
        playJob?.cancel()
        _playhead.value = 0f
        Player.play(File(take.wavPath))
        _editor.update { it?.copy(playingLine = id, playingAll = false) }
        viewModelScope.launch {
            while (Player.isPlaying()) delay(150)
            _editor.update { it?.copy(playingLine = null) }
        }
    }

    /**
     * Play the whole project. Lines without a take become silence of their
     * estimated length, so the playhead, the overview strip and the mix all
     * share one time axis. [fromLineId] starts at that line's slot.
     */
    fun playAll(fromLineId: Long? = null) {
        val st = _editor.value ?: return
        val segs = buildSegments(st.lines)
        if (segs.isEmpty()) return
        playJob?.cancel()
        viewModelScope.launch(Dispatchers.IO) {
            val pcm = st.lines.mapIndexed { i, l ->
                val take = l.takeId?.let { tid -> store.takes(l.id).firstOrNull { it.id == tid } }
                if (take != null) WavFile.read(File(take.wavPath)).samples
                else FloatArray((segs[i].dur * AudioIo.SR).toInt())
            }
            if (pcm.all { it.isEmpty() }) return@launch
            val out = File(getApplication<Application>().filesDir, "mix.wav")
            AudioIo.writeWav(out, Render.concat(pcm))
            val total = segs.last().let { it.start + it.dur }
            val start = fromLineId?.let { id -> segs.firstOrNull { it.lineId == id }?.start }
                ?: _playhead.value.coerceIn(0f, total)
            withContext(Dispatchers.Main) {
                Player.play(out)
                if (start > 0f) Player.seekTo((start * 1000).toInt())
                _playhead.value = start
                _editor.update { it?.copy(playingAll = true, playingLine = null) }
            }
            playJob = viewModelScope.launch {
                while (Player.isPlaying()) {
                    _playhead.value = Player.positionMs() / 1000f
                    delay(80)
                }
                _editor.update { it?.copy(playingAll = false) }
            }
        }
    }

    fun seekTo(sec: Float) {
        val st = _editor.value ?: return
        val total = buildSegments(st.lines).lastOrNull()?.let { it.start + it.dur } ?: 0f
        val p = sec.coerceIn(0f, total)
        _playhead.value = p
        if (Player.isPlaying()) Player.seekTo((p * 1000).toInt())
    }

    fun stop() {
        playJob?.cancel()
        Player.stop()
        _editor.update { it?.copy(playingLine = null, playingAll = false) }
    }

    // ------------------------------------------------------------- helpers
    private fun updateLine(l: Line) {
        _editor.update { st ->
            st?.copy(lines = st.lines.map { if (it.id == l.id) l else it })
        }
    }

    private fun setStatus(id: Long, status: String) {
        _editor.update { st ->
            st?.copy(lines = st.lines.map { if (it.id == id) it.copy(status = status) else it })
        }
    }

    private fun addLog(msg: String) {
        _editor.update { it?.copy(log = (it?.log ?: emptyList()) + msg) }
        runCatching {
            File(getApplication<Application>().filesDir, "studio_log.txt").appendText(msg + "\n")
        }
    }

    fun clearError() = _editor.update { it?.copy(error = null) }

    override fun onCleared() {
        Player.stop()
        Engine.releaseEncoder()
        super.onCleared()
    }
}
