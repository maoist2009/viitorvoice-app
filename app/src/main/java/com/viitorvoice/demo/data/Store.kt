package com.viitorvoice.demo.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import java.io.File

/** A saved reference clip. Every field except the wav is optional by design. */
data class Voice(
    val id: Long,
    val name: String,
    val wavPath: String,
    val refText: String,
    val lang: String,
    val tags: String,
    val created: Long,
)

data class Project(
    val id: Long,
    val title: String,
    val voiceId: Long?,
    val mediaUri: String,
    val mediaKind: String,      // "none" | "audio" | "video"
    val srcAudio: String,
    val srcText: String,
    val created: Long,
    val updated: Long,
)

/** One word (or CJK character) with its span on the project timeline. */
data class Word(val text: String, val start: Float, val end: Float)

data class Line(
    val id: Long,
    val projectId: Long,
    val idx: Int,
    val text: String,
    val genText: String,        // text the selected take was generated from
    val voiceId: Long?,         // null = inherit the project voice
    val sourceWav: String,
    val sourceText: String,
    val targetSec: Float,       // 0 = auto
    val cfg: Float,
    val emotion: Float,
    val nvv: Float,
    val lang: String,
    val status: String,
    val takeId: Long?,
    val takeSec: Float = 0f,
    val offsetSec: Float = -1f,   // <0 = packed after the previous line
    val words: List<Word> = emptyList(), // ASR/aligner timings, may be empty
)

data class Take(
    val id: Long,
    val lineId: Long,
    val wavPath: String,
    val params: String,
    val sec: Float,
    val created: Long,
)

data class ProjectStat(val lines: Int, val done: Int, val sec: Float)

object LineStatus {
    const val EMPTY = "EMPTY"
    const val DIRTY = "DIRTY"
    const val QUEUED = "QUEUED"
    const val RUNNING = "RUNNING"
    const val DONE = "DONE"
    const val FAILED = "FAILED"
    const val KEEP = "KEEP"
}

/**
 * Studio persistence. v2 replaces the old "one generation = one take" table
 * with `project -> line -> take`; the P0 database had no user data, so the
 * upgrade simply rebuilds.
 */
class Store private constructor(ctx: Context) :
    SQLiteOpenHelper(ctx.applicationContext, "vv.db", null, 4) {

    private val root: File = ctx.applicationContext.filesDir

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """CREATE TABLE voice(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 wav_path TEXT NOT NULL,
                 ref_text TEXT NOT NULL DEFAULT '',
                 lang TEXT NOT NULL DEFAULT '',
                 tags TEXT NOT NULL DEFAULT '',
                 created INTEGER NOT NULL)"""
        )
        db.execSQL(
            """CREATE TABLE project(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 title TEXT NOT NULL,
                 voice_id INTEGER,
                 media_uri TEXT NOT NULL DEFAULT '',
                 media_kind TEXT NOT NULL DEFAULT 'none',
                 src_audio TEXT NOT NULL DEFAULT '',
                 src_text TEXT NOT NULL DEFAULT '',
                 created INTEGER NOT NULL,
                 updated INTEGER NOT NULL)"""
        )
        db.execSQL(
            """CREATE TABLE line(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 project_id INTEGER NOT NULL,
                 idx INTEGER NOT NULL,
                 text TEXT NOT NULL DEFAULT '',
                 gen_text TEXT NOT NULL DEFAULT '',
                 voice_id INTEGER,
                 source_wav TEXT NOT NULL DEFAULT '',
                 source_text TEXT NOT NULL DEFAULT '',
                 target_sec REAL NOT NULL DEFAULT 0,
                 cfg REAL NOT NULL DEFAULT 2.0,
                 emotion REAL NOT NULL DEFAULT 1.0,
                 nvv REAL NOT NULL DEFAULT 0.5,
                 lang TEXT NOT NULL DEFAULT 'zh',
                 status TEXT NOT NULL DEFAULT 'EMPTY',
                 take_id INTEGER,
                 offset_sec REAL NOT NULL DEFAULT -1,
                 words TEXT NOT NULL DEFAULT '')"""
        )
        db.execSQL(
            """CREATE TABLE take(
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 line_id INTEGER NOT NULL,
                 wav_path TEXT NOT NULL,
                 params TEXT NOT NULL DEFAULT '',
                 sec REAL NOT NULL DEFAULT 0,
                 created INTEGER NOT NULL)"""
        )
        db.execSQL("CREATE INDEX line_project ON line(project_id, idx)")
        db.execSQL("CREATE INDEX take_line ON take(line_id, created)")
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) {
            // v1 had no user data (the P0 take list was never shipped), so rebuild.
            db.execSQL("DROP TABLE IF EXISTS take")
            db.execSQL("DROP TABLE IF EXISTS line")
            db.execSQL("DROP TABLE IF EXISTS project")
            db.execSQL("DROP TABLE IF EXISTS voice")
            onCreate(db)
            return
        }
        if (oldVersion < 3) {
            db.execSQL("ALTER TABLE line ADD COLUMN offset_sec REAL NOT NULL DEFAULT -1")
        }
        if (oldVersion < 4) {
            db.execSQL("ALTER TABLE line ADD COLUMN words TEXT NOT NULL DEFAULT ''")
        }
    }

    // ----------------------------------------------------------------- files
    fun newVoiceFile(): File =
        File(root, "voices").apply { mkdirs() }
            .resolve("voice_${System.currentTimeMillis()}.wav")

    fun newTakeFile(lineId: Long): File =
        File(root, "takes").apply { mkdirs() }
            .resolve("line${lineId}_${System.currentTimeMillis()}.wav")

    fun projectDir(projectId: Long): File =
        File(root, "projects").apply { mkdirs() }
            .resolve("p$projectId").apply { mkdirs() }

    // ---------------------------------------------------------------- voices
    fun voices(): List<Voice> = readableDatabase.rawQuery(
        "SELECT id,name,wav_path,ref_text,lang,tags,created FROM voice ORDER BY created DESC",
        null,
    ).use { c ->
        buildList {
            while (c.moveToNext()) add(
                Voice(
                    c.getLong(0), c.getString(1), c.getString(2), c.getString(3),
                    c.getString(4), c.getString(5), c.getLong(6),
                )
            )
        }
    }

    fun addVoice(name: String, wavPath: String, refText: String, lang: String, tags: String): Long {
        val cv = ContentValues().apply {
            put("name", name)
            put("wav_path", wavPath)
            put("ref_text", refText)
            put("lang", lang)
            put("tags", tags)
            put("created", System.currentTimeMillis())
        }
        return writableDatabase.insert("voice", null, cv)
    }

    fun deleteVoice(id: Long) {
        voices().firstOrNull { it.id == id }?.let { runCatching { File(it.wavPath).delete() } }
        writableDatabase.delete("voice", "id=?", arrayOf(id.toString()))
    }

    // -------------------------------------------------------------- projects
    fun projects(): List<Project> = readableDatabase.rawQuery(
        "SELECT id,title,voice_id,media_uri,media_kind,src_audio,src_text,created,updated" +
            " FROM project ORDER BY updated DESC",
        null,
    ).use { c ->
        buildList {
            while (c.moveToNext()) add(
                Project(
                    c.getLong(0), c.getString(1), if (c.isNull(2)) null else c.getLong(2),
                    c.getString(3), c.getString(4), c.getString(5), c.getString(6),
                    c.getLong(7), c.getLong(8),
                )
            )
        }
    }

    fun project(id: Long): Project? = projects().firstOrNull { it.id == id }

    fun addProject(title: String, voiceId: Long?, mediaUri: String = "", mediaKind: String = "none"): Long {
        val now = System.currentTimeMillis()
        val cv = ContentValues().apply {
            put("title", title)
            put("voice_id", voiceId)
            put("media_uri", mediaUri)
            put("media_kind", mediaKind)
            put("created", now)
            put("updated", now)
        }
        return writableDatabase.insert("project", null, cv)
    }

    fun touchProject(id: Long) {
        val cv = ContentValues().apply { put("updated", System.currentTimeMillis()) }
        writableDatabase.update("project", cv, "id=?", arrayOf(id.toString()))
    }

    fun updateProject(p: Project) {
        val cv = ContentValues().apply {
            put("title", p.title)
            put("voice_id", p.voiceId)
            put("media_uri", p.mediaUri)
            put("media_kind", p.mediaKind)
            put("src_audio", p.srcAudio)
            put("src_text", p.srcText)
            put("updated", System.currentTimeMillis())
        }
        writableDatabase.update("project", cv, "id=?", arrayOf(p.id.toString()))
    }

    fun deleteProject(id: Long) {
        lines(id).forEach { deleteLine(it.id) }
        projectDir(id).deleteRecursively()
        writableDatabase.delete("project", "id=?", arrayOf(id.toString()))
    }

    /** Per-project counters for the home cards. */
    fun stats(): Map<Long, ProjectStat> {
        val m = HashMap<Long, ProjectStat>()
        readableDatabase.rawQuery(
            "SELECT project_id, COUNT(*)," +
                " SUM(CASE WHEN take_id IS NOT NULL THEN 1 ELSE 0 END)," +
                " SUM(CASE WHEN take_id IS NOT NULL THEN" +
                " (SELECT sec FROM take t WHERE t.id=line.take_id) ELSE 0 END)" +
                " FROM line GROUP BY project_id",
            null,
        ).use { c ->
            while (c.moveToNext()) {
                m[c.getLong(0)] = ProjectStat(c.getInt(1), c.getInt(2), c.getFloat(3))
            }
        }
        return m
    }

    fun lineCount(projectId: Long): Int = readableDatabase.rawQuery(
        "SELECT COUNT(*) FROM line WHERE project_id=?", arrayOf(projectId.toString()),
    ).use { c -> if (c.moveToFirst()) c.getInt(0) else 0 }

    // ----------------------------------------------------------------- lines
    fun lines(projectId: Long): List<Line> = readableDatabase.rawQuery(
        "SELECT id,project_id,idx,text,gen_text,voice_id,source_wav,source_text,target_sec," +
            "cfg,emotion,nvv,lang,status,take_id," +
            "(SELECT sec FROM take t WHERE t.id=line.take_id),offset_sec,words" +
            " FROM line WHERE project_id=? ORDER BY idx",
        arrayOf(projectId.toString()),
    ).use { c ->
        buildList {
            while (c.moveToNext()) add(
                Line(
                    c.getLong(0), c.getLong(1), c.getInt(2), c.getString(3), c.getString(4),
                    if (c.isNull(5)) null else c.getLong(5), c.getString(6), c.getString(7),
                    c.getFloat(8), c.getFloat(9), c.getFloat(10), c.getFloat(11),
                    c.getString(12), c.getString(13),
                    if (c.isNull(14)) null else c.getLong(14),
                    if (c.isNull(15)) 0f else c.getFloat(15),
                    c.getFloat(16),
                    parseWords(c.getString(17)),
                )
            )
        }
    }

    fun line(id: Long): Line? = readableDatabase.rawQuery(
        "SELECT id,project_id,idx,text,gen_text,voice_id,source_wav,source_text,target_sec," +
            "cfg,emotion,nvv,lang,status,take_id," +
            "(SELECT sec FROM take t WHERE t.id=line.take_id),offset_sec,words FROM line WHERE id=?",
        arrayOf(id.toString()),
    ).use { c -> if (c.moveToNext()) readLine(c) else null }

    private fun readLine(c: android.database.Cursor) = Line(
        c.getLong(0), c.getLong(1), c.getInt(2), c.getString(3), c.getString(4),
        if (c.isNull(5)) null else c.getLong(5), c.getString(6), c.getString(7),
        c.getFloat(8), c.getFloat(9), c.getFloat(10), c.getFloat(11),
        c.getString(12), c.getString(13),
        if (c.isNull(14)) null else c.getLong(14),
        if (c.isNull(15)) 0f else c.getFloat(15),
        c.getFloat(16),
        parseWords(c.getString(17)),
    )

    fun addLine(projectId: Long, idx: Int, text: String = ""): Long {
        val cv = ContentValues().apply {
            put("project_id", projectId)
            put("idx", idx)
            put("text", text)
            put("status", if (text.isBlank()) LineStatus.EMPTY else LineStatus.DIRTY)
        }
        touchProject(projectId)
        return writableDatabase.insert("line", null, cv)
    }

    fun updateLine(l: Line) {
        val cv = ContentValues().apply {
            put("idx", l.idx)
            put("text", l.text)
            put("gen_text", l.genText)
            put("voice_id", l.voiceId)
            put("source_wav", l.sourceWav)
            put("source_text", l.sourceText)
            put("target_sec", l.targetSec)
            put("cfg", l.cfg)
            put("emotion", l.emotion)
            put("nvv", l.nvv)
            put("lang", l.lang)
            put("status", l.status)
            put("take_id", l.takeId)
            put("offset_sec", l.offsetSec)
            put("words", wordsJson(l.words))
        }
        writableDatabase.update("line", cv, "id=?", arrayOf(l.id.toString()))
        touchProject(l.projectId)
    }

    fun deleteLine(id: Long) {
        takes(id).forEach { runCatching { File(it.wavPath).delete() } }
        writableDatabase.delete("take", "line_id=?", arrayOf(id.toString()))
        writableDatabase.delete("line", "id=?", arrayOf(id.toString()))
    }

    // ----------------------------------------------------------------- takes
    fun takes(lineId: Long): List<Take> = readableDatabase.rawQuery(
        "SELECT id,line_id,wav_path,params,sec,created FROM take WHERE line_id=?" +
            " ORDER BY created DESC",
        arrayOf(lineId.toString()),
    ).use { c ->
        buildList {
            while (c.moveToNext()) add(
                Take(c.getLong(0), c.getLong(1), c.getString(2), c.getString(3),
                    c.getFloat(4), c.getLong(5))
            )
        }
    }

    fun addTake(lineId: Long, wavPath: String, params: String, sec: Float): Long {
        val cv = ContentValues().apply {
            put("line_id", lineId)
            put("wav_path", wavPath)
            put("params", params)
            put("sec", sec)
            put("created", System.currentTimeMillis())
        }
        return writableDatabase.insert("take", null, cv)
    }

    /** Round-trip probe used by the headless selftest. */
    fun selfTest(): String {
        val f = File(root, "selftest_voice.wav").apply { writeText("x") }
        val vid = addVoice("selftest", f.absolutePath, "", "zh", "")
        val pid = addProject("selftest", vid)
        val lid = addLine(pid, 0, "你好")
        val tid = addTake(lid, f.absolutePath, "", 0f)
        val ok = project(pid) != null && lines(pid).size == 1 && takes(lid).size == 1
        deleteProject(pid)
        deleteVoice(vid)
        return "ok=$ok wavGone=${!f.exists()}"
    }

    companion object {
        @Volatile
        private var inst: Store? = null

        fun get(ctx: Context): Store = inst ?: synchronized(this) {
            inst ?: Store(ctx.applicationContext).also { inst = it }
        }
    }

    // ----------------------------------------------------------------- words
    private fun parseWords(s: String): List<Word> {
        if (s.isBlank()) return emptyList()
        return runCatching {
            val a = org.json.JSONArray(s)
            buildList {
                for (i in 0 until a.length()) {
                    val o = a.getJSONObject(i)
                    add(Word(o.getString("t"), o.getDouble("s").toFloat(), o.getDouble("e").toFloat()))
                }
            }
        }.getOrDefault(emptyList())
    }

    private fun wordsJson(w: List<Word>): String {
        if (w.isEmpty()) return ""
        val a = org.json.JSONArray()
        w.forEach { a.put(org.json.JSONObject().put("t", it.text).put("s", it.start).put("e", it.end)) }
        return a.toString()
    }
}
