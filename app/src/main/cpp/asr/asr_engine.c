/* asr_engine.c — 见 asr_engine.h。
 *
 * 宿主侧一律用 fp16 缓冲；图边界的实际 dtype 在 init 时用 qnn_describe 查出来，
 * 不一致就现场转（AI Hub 的量化图通常保留浮点 I/O，但这里不赌）。
 */
#include "asr_engine.h"

#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <android/log.h>

#include "asr_mel.h"
#include "qnn_executor.h"

#define CTX       ASR_CTX
#define NL        ASR_NL
#define NKV       ASR_NKV
#define HD        ASR_HD
#define VOCAB     ASR_VOCAB
#define AUDIO_TOK ASR_AUDIO_TOK

/* 几何参数在 init 时从 dec_p0_prefill 的输入形状推出来，1.7B 与 0.6B 共用一套代码：
 *   g_pad < 0 ：sink 方案（1.7B）。sink 行不进图，K/V 由 load_sink_kv 注入，
 *               图内行数 = g_head - 1（118）。
 *   g_pad >= 0：左补零方案（0.6B）。真实 prompt 前面补 g_pad 行零向量，
 *               图内行数 = g_head + g_pad（128）。
 * g_head 是真实 prompt 长度（9 + 104 + 6 = 119），两个模型相同。 */
static int g_tpre, g_head, g_pad, g_hid;

#define KV_ELEMS   ((size_t)CTX * NKV * HD)     /* 单层单侧（key 或 value） */
#define N_MEL      ((size_t)ASR_MEL_N_MELS * ASR_MEL_FRAMES)

static const int LAYERS[3][10] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
    {10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
    {20, 21, 22, 23, 24, 25, 26, 27, -1, -1},
};
static const int NPART[3] = {10, 10, 8};
#define MAXPART 10

static const char *GRAPH_TAGS[8] = {"enc", "dec_p0_prefill", "dec_p0_decode",
                                    "dec_p1_prefill", "dec_p1_decode",
                                    "dec_p2_prefill", "dec_p2_decode", "lm_head"};
#define NGRAPH 8

/* prompt 固定片段（tokenizer 离线取，见 tools/asr/asr_torch.build_prompt_ids） */
static const int IDS_PRE[] = {151644, 8948, 198, 151645, 198, 151644, 872, 198, 151669};
static const int IDS_POST[] = {151670, 151645, 198, 151644, 77091, 198};
#define NPRE  ((int)(sizeof(IDS_PRE) / sizeof(IDS_PRE[0])))
#define NPOST ((int)(sizeof(IDS_POST) / sizeof(IDS_POST[0])))

/* ---------- 状态 ---------- */
static char g_err[512];
static int g_init = 0;
static char g_model_dir[1024];
/* 8 张图全常驻约 4 GB（prefill/decode 各一份），DSP 内存吃不消。
 * 改成按需加载：enc -> 用完卸载 -> 3 张 prefill -> 卸载 -> 3 张 decode + lm_head。
 * 峰值从 ~4 GB 降到 ~2 GB。 */
static int g_loaded[NGRAPH];
/* run_graph 最后一次取回的输出元素数，供 probe 按真实长度落盘（aligner 的
 * enc 输出是 104x1024，不是 ASR 的 104x2048）。 */
static size_t g_last_out_elems;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static uint16_t *g_mask_pre;      /* (g_tpre, CTX) fp16：图里 mask 输入就是 FLOAT_16 */
static uint16_t *g_cos_pre;       /* (g_tpre, HD) */
static uint16_t *g_sin_pre;
static uint16_t *g_cos_tab;       /* (CTX, HD)，按绝对位置 */
static uint16_t *g_sin_tab;

static uint16_t *g_kv;            /* [NL][2][KV_ELEMS] */
static uint16_t *g_kv_out;        /* present 暂存（最大段 NPART*2 个张量） */
static uint16_t *g_embeds;        /* (g_tpre, g_hid) */
static uint16_t *g_hidden_pre;    /* (g_tpre, g_hid) 段间中转 A */
static uint16_t *g_hidden_pre2;   /* (g_tpre, g_hid) 段间中转 B（避免 in/out 同址） */
static uint16_t *g_hidden_dec;    /* (1, g_hid) 段间中转 A */
static uint16_t *g_hidden_dec2;   /* (1, g_hid) 段间中转 B */
static uint16_t *g_mel;           /* (128, 800) */
static float    *g_mel_f32;       /* mel 计算的中间结果 */
static float    *g_pcm;           /* 128000 */

static int    g_tbl_fd = -1;
static void  *g_tbl_map = NULL;
static size_t g_tbl_bytes = 0;
static uint16_t *g_tbl = NULL;    /* (VOCAB, g_hid) mmap */
static const uint16_t *g_final_pre;  /* prefill 最后一段的 hidden */

static asr_mel_t g_mel_ctx;

/* 每张图的 I/O dtype 缓存 */
#define MAXT 40
static struct { char name[64]; int dtype; int rank; uint32_t dims[4]; }
    g_in_desc[NGRAPH][MAXT], g_out_desc[NGRAPH][MAXT];
static int g_in_n[NGRAPH], g_out_n[NGRAPH];

static void set_err(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
}

const char *asr_engine_error(void) { return g_err; }

/* 图在 run 结束时会全部卸载，qnn 的 per-graph 统计随之丢失，所以引擎自己
 * 按 tag 记一次墙钟（含 graphExecute 的全部开销）。 */
#include <time.h>
static int64_t ns_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

#define NSTAT 24
static struct { char tag[32]; int n; int64_t sum, mn, mx; } g_t[NSTAT];

static void t_reset(void) { memset(g_t, 0, sizeof(g_t)); }

static void t_rec(const char *tag, int64_t ns) {
  for (int i = 0; i < NSTAT; i++) {
    if (!g_t[i].tag[0]) {
      snprintf(g_t[i].tag, sizeof(g_t[i].tag), "%s", tag);
    }
    if (!strcmp(g_t[i].tag, tag)) {
      if (g_t[i].n == 0) { g_t[i].mn = ns; g_t[i].mx = ns; }
      g_t[i].n++;
      g_t[i].sum += ns;
      if (ns < g_t[i].mn) g_t[i].mn = ns;
      if (ns > g_t[i].mx) g_t[i].mx = ns;
      return;
    }
  }
}

const char *asr_timings(void) {
  static char buf[4096];
  size_t off = 0;
  for (int i = 0; i < NSTAT; i++) {
    if (!g_t[i].tag[0] || g_t[i].n == 0) continue;
    off += (size_t)snprintf(buf + off, sizeof(buf) - off,
                            "%s{\"tag\":\"%s\",\"n\":%d,\"mean_ms\":%.2f,"
                            "\"min_ms\":%.2f,\"max_ms\":%.2f,\"total_ms\":%.0f}",
                            off ? "," : "", g_t[i].tag, g_t[i].n,
                            (double)g_t[i].sum / g_t[i].n / 1e6,
                            (double)g_t[i].mn / 1e6, (double)g_t[i].mx / 1e6,
                            (double)g_t[i].sum / 1e6);
    if (off > sizeof(buf) - 160) break;
  }
  buf[off] = '\0';
  return buf;
}

/* ---------- fp16 ---------- */
static float h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
  if (e == 0) {
    if (m == 0) f = s << 31;
    else {
      int exp = -1;
      do { m <<= 1; exp++; } while (!(m & 0x400));
      f = (s << 31) | ((uint32_t)(127 - 15 - exp) << 23) | ((m & 0x3ff) << 13);
    }
  } else if (e == 31) {
    f = (s << 31) | 0x7f800000u | (m << 13);
  } else {
    f = (s << 31) | ((e + 112) << 23) | (m << 13);
  }
  float out;
  memcpy(&out, &f, 4);
  return out;
}

static uint16_t f2h(float x) {
  uint32_t f;
  memcpy(&f, &x, 4);
  uint32_t s = (f >> 16) & 0x8000u;
  int32_t e = (int32_t)((f >> 23) & 0xff) - 127 + 15;
  uint32_t m = f & 0x7fffffu;
  if (((f >> 23) & 0xff) == 0xff) return (uint16_t)(s | 0x7c00u | (m ? 0x200u : 0));
  if (e >= 31) return (uint16_t)(s | 0x7c00u);
  if (e <= 0) {
    if (e < -10) return (uint16_t)s;
    m |= 0x800000u;
    int shift = 14 - e;
    uint32_t r = m >> shift;
    uint32_t rem = m & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (r & 1u))) r++;   /* round half to even */
    return (uint16_t)(s | r);
  }
  uint32_t r = ((uint32_t)e << 10) | (m >> 13);
  uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r++;  /* round half to even */
  return (uint16_t)(s | r);
}

/* ---------- 初始化 ---------- */
static void build_rope(void) {
  const double theta = 1e6;
  double inv[HD / 2];
  for (int j = 0; j < HD / 2; j++) inv[j] = 1.0 / pow(theta, (double)(2 * j) / HD);
  for (int t = 0; t < CTX; t++) {
    for (int d = 0; d < HD / 2; d++) {
      double v = t * inv[d];
      uint16_t c = f2h((float)cos(v)), s = f2h((float)sin(v));
      g_cos_tab[(size_t)t * HD + d] = c;
      g_cos_tab[(size_t)t * HD + d + HD / 2] = c;
      g_sin_tab[(size_t)t * HD + d] = s;
      g_sin_tab[(size_t)t * HD + d + HD / 2] = s;
    }
  }
  for (int i = 0; i < g_tpre; i++) {
    /* sink 方案：槽 CTX-g_tpre-1 是 sink（位置 0），图内第 i 行 = 位置 i+1。
     * 左补零方案：前 g_pad 行是零向量，位置一律 0，之后从 0 开始数。 */
    int pos = (g_pad >= 0) ? (i - g_pad) : (i + 1);
    if (pos < 0) pos = 0;
    memcpy(g_cos_pre + (size_t)i * HD, g_cos_tab + (size_t)pos * HD, sizeof(uint16_t) * HD);
    memcpy(g_sin_pre + (size_t)i * HD, g_sin_tab + (size_t)pos * HD, sizeof(uint16_t) * HD);
  }
}

/* prefill 的加性 mask。真实 prompt 右对齐（占最后 g_head 个槽），行内因果。
 *   sink 方案：第 i 行可见 [CTX-g_tpre-1, CTX-g_tpre+i]（首槽是注入的 sink）
 *   左补零方案：第 i 行可见 [CTX-g_tpre+g_pad, CTX-g_tpre+i]，且 i >= g_pad
 *               （补零行全部屏蔽，与 asr_torch.make_mask_prefill 一致） */
static void build_mask_prefill(uint16_t *m) {
  const int lower = (g_pad >= 0) ? (CTX - g_tpre + g_pad) : (CTX - g_tpre - 1);
  const uint16_t m0 = f2h(0.0f), mneg = f2h(-50.0f);
  for (int i = 0; i < g_tpre; i++)
    for (int s = 0; s < CTX; s++) {
      int ok = (s >= lower) && (s <= CTX - g_tpre + i);
      if (g_pad >= 0 && i < g_pad) ok = 0;
      m[(size_t)i * CTX + s] = ok ? m0 : mneg;
    }
}

/* decode 第 step 步可见槽：[CTX-g_tpre-2-step, CTX)
 * sink 在 prefill 后占槽 CTX-g_tpre-1，图内部左移一格后落在 CTX-g_tpre-2-step，
 * 必须一并可见（漏掉它 = 漏掉 attention sink，输出会明显错） */
static void build_mask_decode(uint16_t *m, int step) {
  int start = CTX - g_tpre - 2 - step;
  const uint16_t m0 = f2h(0.0f), mneg = f2h(-50.0f);
  for (int s = 0; s < CTX; s++) m[s] = (s >= start) ? m0 : mneg;
}

static void desc_in(void *user, const char *name, int dtype, uint32_t rank, const uint32_t *dims) {
  int gi = *(int *)user, i = g_in_n[gi];
  if (i < MAXT) {
    snprintf(g_in_desc[gi][i].name, sizeof(g_in_desc[gi][i].name), "%s", name);
    g_in_desc[gi][i].dtype = dtype;
    g_in_desc[gi][i].rank = (int)rank;
    for (uint32_t k = 0; k < rank && k < 4; k++) g_in_desc[gi][i].dims[k] = dims[k];
    g_in_n[gi] = i + 1;
  }
}

static void desc_out(void *user, const char *name, int dtype, uint32_t rank, const uint32_t *dims) {
  int gi = *(int *)user, i = g_out_n[gi];
  if (i < MAXT) {
    snprintf(g_out_desc[gi][i].name, sizeof(g_out_desc[gi][i].name), "%s", name);
    g_out_desc[gi][i].dtype = dtype;
    g_out_desc[gi][i].rank = (int)rank;
    for (uint32_t k = 0; k < rank && k < 4; k++) g_out_desc[gi][i].dims[k] = dims[k];
    g_out_n[gi] = i + 1;
  }
}

static int gi_of(const char *tag) {
  for (int i = 0; i < NGRAPH; i++) if (!strcmp(GRAPH_TAGS[i], tag)) return i;
  return -1;
}

static int g_ensure(int gi);   /* 定义在后面（按需加载 + describe） */

/* 从 dec_p0_prefill 的 inputs_embeds 形状推出几何：tpre 与 hid。
 * 1.7B -> (118, 2048)，0.6B -> (128, 1024)。顺带把这张图载入（第一次识别先用它）。 */
static int derive_geometry(void) {
  int gi = gi_of("dec_p0_prefill");
  if (gi < 0) { set_err("no dec_p0_prefill graph"); return -1; }
  if (g_ensure(gi)) return -1;
  int found = 0;
  for (int i = 0; i < g_in_n[gi]; i++) {
    if (strcmp(g_in_desc[gi][i].name, "inputs_embeds")) continue;
    if (g_in_desc[gi][i].rank != 3) {
      set_err("inputs_embeds rank %d, want 3", g_in_desc[gi][i].rank);
      return -1;
    }
    g_tpre = (int)g_in_desc[gi][i].dims[1];
    g_hid  = (int)g_in_desc[gi][i].dims[2];
    found = 1;
  }
  if (!found) { set_err("dec_p0_prefill has no inputs_embeds"); return -1; }
  __android_log_print(ANDROID_LOG_INFO, "AsrDemo",
                      "GEOM tpre=%d hid=%d in=%d out=%d", g_tpre, g_hid,
                      g_in_n[gi], g_out_n[gi]);
  for (int i = 0; i < g_out_n[gi] && i < 6; i++)
    __android_log_print(ANDROID_LOG_INFO, "AsrDemo", "  out[%d]=%s", i, g_out_desc[gi][i].name);
  g_head = NPRE + (int)AUDIO_TOK + NPOST;
  g_pad  = g_tpre - g_head;     /* 1.7B: -1；0.6B: +9 */
  if (g_pad < -1 || g_pad > 64) {
    set_err("bad prefill geometry: tpre=%d head=%d", g_tpre, g_head);
    return -1;
  }
  return 0;
}

/* 按需加载：第一次用到时 load + describe，之后直接复用。 */
static int g_ensure(int gi) {
  if (g_loaded[gi]) return 0;
  char path[1024];
  snprintf(path, sizeof(path), "%s/%s.bin", g_model_dir, GRAPH_TAGS[gi]);
  int64_t tl0 = ns_now();
  if (qnn_load(GRAPH_TAGS[gi], path)) {
    set_err("load %s: %s", GRAPH_TAGS[gi], qnn_last_error());
    return -1;
  }
  int g2 = gi;
  g_in_n[gi] = 0;
  g_out_n[gi] = 0;
  if (qnn_describe(GRAPH_TAGS[gi], desc_in, desc_out, &g2)) {
    set_err("describe %s: %s", GRAPH_TAGS[gi], qnn_last_error());
    qnn_free(GRAPH_TAGS[gi]);
    return -1;
  }
  {
    char ltag[40];
    snprintf(ltag, sizeof(ltag), "ld:%s", GRAPH_TAGS[gi]);
    t_rec(ltag, ns_now() - tl0);
  }
  g_loaded[gi] = 1;
  return 0;
}

/* 哪些图常驻：enc / 3 张 decode 每次 run 都要用，卸载了下次还得重新
 * createFromBinary（每张 0.3–1.0 s）。prefill 每句只用一次，且三张一起
 * 加载会把峰值推到 DSP 容不下的程度，所以用完即卸。
 * lm_head 虽也常用，但只有 140–170 ms 的加载成本，而它若常驻会把 prefill
 * 阶段的峰值顶到 4.9 GB（实测直接崩进程），故不常驻。
 * 与 GRAPH_TAGS 同序：enc, p0pre, p0dec, p1pre, p1dec, p2pre, p2dec, lm_head */
static const int KEEP[NGRAPH] = {1, 0, 1, 0, 1, 0, 1, 0};

static void g_release(int gi) {
  if (gi < 0 || gi >= NGRAPH) return;
  if (KEEP[gi]) return;
  if (!g_loaded[gi]) return;
  qnn_free(GRAPH_TAGS[gi]);
  g_loaded[gi] = 0;
}

static void g_release_all(void) {
  for (int i = 0; i < NGRAPH; i++) g_release(i);
}

static int dtype_in(int gi, const char *name) {
  for (int i = 0; i < g_in_n[gi]; i++)
    if (!strcmp(g_in_desc[gi][i].name, name)) return g_in_desc[gi][i].dtype;
  return -1;
}

static int open_table(const char *dir) {
  char path[1024];
  snprintf(path, sizeof(path), "%s/token_embeds.bin", dir);
  int fd = open(path, O_RDONLY);
  if (fd < 0) { set_err("open %s failed", path); return -1; }
  struct stat st;
  if (fstat(fd, &st) != 0) { close(fd); set_err("fstat %s failed", path); return -1; }
  size_t want = (size_t)VOCAB * (size_t)g_hid * 2;
  if ((size_t)st.st_size < want) {
    close(fd);
    set_err("%s is %lld bytes, want %zu", path, (long long)st.st_size, want);
    return -1;
  }
  void *p = mmap(NULL, want, PROT_READ, MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) { close(fd); set_err("mmap %s failed", path); return -1; }
  g_tbl_fd = fd;
  g_tbl_map = p;
  g_tbl_bytes = want;
  g_tbl = (uint16_t *)p;
  return 0;
}

int asr_engine_init(const char *model_dir, const char *lib_dir) {
  pthread_mutex_lock(&g_lock);
  if (g_init) { pthread_mutex_unlock(&g_lock); return 0; }
  g_err[0] = '\0';

  if (asr_mel_init(&g_mel_ctx)) { set_err("asr_mel_init failed"); goto fail; }

  for (int i = 0; i < NGRAPH; i++) g_loaded[i] = 0;
  snprintf(g_model_dir, sizeof(g_model_dir), "%s", model_dir);
  if (qnn_init(lib_dir, lib_dir)) { set_err("qnn_init: %s", qnn_last_error()); goto fail; }
  /* 几何（tpre/hid/pad）从图里推，两种模型共用下面的代码 */
  if (derive_geometry()) goto fail;

  g_mask_pre   = (uint16_t *)malloc(sizeof(uint16_t) * g_tpre * CTX);
  g_cos_pre    = (uint16_t *)malloc(sizeof(uint16_t) * g_tpre * HD);
  g_sin_pre    = (uint16_t *)malloc(sizeof(uint16_t) * g_tpre * HD);
  g_cos_tab    = (uint16_t *)malloc(sizeof(uint16_t) * CTX * HD);
  g_sin_tab    = (uint16_t *)malloc(sizeof(uint16_t) * CTX * HD);
  g_kv         = (uint16_t *)malloc(sizeof(uint16_t) * NL * 2 * KV_ELEMS);
  g_kv_out     = (uint16_t *)malloc(sizeof(uint16_t) * 2 * MAXPART * KV_ELEMS);
  g_embeds     = (uint16_t *)malloc(sizeof(uint16_t) * g_tpre * g_hid);
  g_hidden_pre = (uint16_t *)malloc(sizeof(uint16_t) * g_tpre * g_hid);
  g_hidden_pre2= (uint16_t *)malloc(sizeof(uint16_t) * g_tpre * g_hid);
  g_hidden_dec = (uint16_t *)malloc(sizeof(uint16_t) * g_hid);
  g_hidden_dec2= (uint16_t *)malloc(sizeof(uint16_t) * g_hid);
  g_mel        = (uint16_t *)malloc(sizeof(uint16_t) * N_MEL);
  g_mel_f32    = (float *)malloc(sizeof(float) * N_MEL);
  g_pcm        = (float *)malloc(sizeof(float) * ASR_MEL_SAMPLES);
  if (!g_mask_pre || !g_cos_pre || !g_sin_pre || !g_cos_tab || !g_sin_tab ||
      !g_kv || !g_embeds || !g_hidden_pre || !g_hidden_pre2 ||
      !g_hidden_dec || !g_hidden_dec2 || !g_mel ||
      !g_mel_f32 || !g_pcm) {
    set_err("out of memory (%zu MB kv + buffers needed)",
            (size_t)(sizeof(uint16_t) * NL * 2 * KV_ELEMS) >> 20);
    goto fail;
  }
  build_rope();
  build_mask_prefill(g_mask_pre);

  if (open_table(model_dir)) goto fail;

  /* 图按需加载（见 g_ensure）：目录已在前面记好。 */
  g_init = 1;
  pthread_mutex_unlock(&g_lock);
  return 0;

fail:
  pthread_mutex_unlock(&g_lock);
  return -1;
}

void asr_engine_shutdown(void) {
  pthread_mutex_lock(&g_lock);
  if (!g_init) { pthread_mutex_unlock(&g_lock); return; }
  for (int i = 0; i < NGRAPH; i++)
    if (g_loaded[i]) { qnn_free(GRAPH_TAGS[i]); g_loaded[i] = 0; }
  qnn_shutdown_if_empty();
  if (g_tbl_map && g_tbl_map != MAP_FAILED) munmap(g_tbl_map, g_tbl_bytes);
  if (g_tbl_fd >= 0) close(g_tbl_fd);
  g_tbl_map = NULL; g_tbl = NULL; g_tbl_fd = -1;
  free(g_mask_pre); free(g_cos_pre); free(g_sin_pre); free(g_cos_tab);
  free(g_sin_tab); free(g_kv); free(g_kv_out); free(g_embeds);
  free(g_hidden_pre); free(g_hidden_pre2); free(g_hidden_dec); free(g_hidden_dec2);
  free(g_hidden_dec); free(g_mel); free(g_mel_f32); free(g_pcm);
  g_mask_pre = NULL; g_cos_pre = NULL; g_sin_pre = NULL; g_cos_tab = NULL;
  g_sin_tab = NULL; g_kv = NULL; g_kv_out = NULL; g_embeds = NULL;
  g_hidden_pre = NULL; g_hidden_pre2 = NULL; g_hidden_dec = NULL; g_hidden_dec2 = NULL;
  g_hidden_dec = NULL; g_mel = NULL; g_mel_f32 = NULL; g_pcm = NULL;
  asr_mel_free(&g_mel_ctx);
  g_init = 0;
  pthread_mutex_unlock(&g_lock);
}

/* ---------- 图执行 ---------- */
typedef struct {
  const char *name;
  const void *data;
  size_t      elems;
  int         dtype;      /* 源缓冲的 RVC_QNN_* */
} InSpec;

typedef struct {
  const char *name;
  uint16_t   *dst;        /* 目标一律 fp16 */
  size_t      cap;        /* 元素数 */
} OutSpec;

typedef struct {
  QnnIo io[MAXT];
  int   n;
  void *scratch[MAXT];
  int   nscratch;
} Feed;

static void feed_reset(Feed *f) {
  for (int i = 0; i < f->nscratch; i++) free(f->scratch[i]);
  f->n = 0;
  f->nscratch = 0;
}

static int feed_add(Feed *f, const char *name, const void *data, size_t elems,
                    int src_dtype, int tgt_dtype) {
  void *buf = (void *)data;
  size_t nbytes = elems * qnn_elem_size(src_dtype);
  if (tgt_dtype != src_dtype) {
    if (src_dtype == RVC_QNN_FLOAT16 && tgt_dtype == RVC_QNN_FLOAT32) {
      float *p = (float *)malloc(elems * 4);
      if (!p) { set_err("oom converting %s", name); return -1; }
      for (size_t i = 0; i < elems; i++) p[i] = h2f(((const uint16_t *)data)[i]);
      f->scratch[f->nscratch++] = p;
      buf = p;
      nbytes = elems * 4;
    } else if (src_dtype == RVC_QNN_FLOAT32 && tgt_dtype == RVC_QNN_FLOAT16) {
      uint16_t *p = (uint16_t *)malloc(elems * 2);
      if (!p) { set_err("oom converting %s", name); return -1; }
      for (size_t i = 0; i < elems; i++) p[i] = f2h(((const float *)data)[i]);
      f->scratch[f->nscratch++] = p;
      buf = p;
      nbytes = elems * 2;
    } else {
      set_err("%s: unsupported dtype conversion %d -> %d", name, src_dtype, tgt_dtype);
      return -1;
    }
  }
  f->io[f->n].name = name;
  f->io[f->n].dtype = tgt_dtype;
  f->io[f->n].data = buf;
  f->io[f->n].bytes = nbytes;
  f->n++;
  return 0;
}

/* 执行 tag：输入按名字喂，输出按名字取回并转成 fp16 写进 dst。 */
static int run_graph(const char *tag, const InSpec *ins, int n_in,
                     const OutSpec *outs, int n_out) {
  int gi = gi_of(tag);
  if (gi < 0) { set_err("unknown graph %s", tag); return -1; }
  if (g_ensure(gi)) return -1;

  Feed f;
  memset(&f, 0, sizeof(f));
  for (int i = 0; i < n_in; i++) {
    int want = dtype_in(gi, ins[i].name);
    if (want < 0) {
      feed_reset(&f);
      set_err("%s: graph has no input '%s' (check export names)", tag, ins[i].name);
      return -1;
    }
    if (feed_add(&f, ins[i].name, ins[i].data, ins[i].elems, ins[i].dtype, want)) {
      feed_reset(&f);
      return -1;
    }
  }

  QnnIo oo[MAXT];
  memset(oo, 0, sizeof(oo));
  size_t got = 0;
  if (qnn_exec_raw(tag, f.io, (size_t)f.n, oo, MAXT, &got)) {
    feed_reset(&f);
    set_err("%s: %s", tag, qnn_last_error());
    return -1;
  }
  feed_reset(&f);

  int rc = 0;
  for (int k = 0; k < n_out && !rc; k++) {
    int idx = -1;
    for (size_t i = 0; i < got; i++)
      if (oo[i].name && !strcmp(oo[i].name, outs[k].name)) { idx = (int)i; break; }
    if (idx < 0) {
      /* 上下文二进制把名字改成 output_N，而且输出顺序不保证与 ONNX 一致
       * （0.6B enc 的 noclip 版本就把 hid 排到了 output_0）。先按元素数
       * 唯一匹配，匹配不到再退回按位置。 */
      size_t want = outs[k].cap;
      int hit = -1, n_hit = 0;
      for (size_t i = 0; i < got; i++) {
        size_t e = oo[i].bytes / qnn_elem_size(oo[i].dtype);
        if (e == want) { hit = (int)i; n_hit++; }
      }
      if (n_hit == 1) idx = hit;
      else if (k < (int)got) idx = (int)k;
    }
    if (idx < 0) {
      set_err("%s: output '%s' not found", tag, outs[k].name);
      rc = -1;
      break;
    }
    size_t es = qnn_elem_size(oo[idx].dtype);
    size_t elems = oo[idx].bytes / es;
    g_last_out_elems = elems;
    if (elems > outs[k].cap) {
      set_err("%s: output '%s' has %zu elems, buffer holds %zu", tag, outs[k].name,
              elems, outs[k].cap);
      rc = -1;
      break;
    }
    if (oo[idx].dtype == RVC_QNN_FLOAT16) {
      memcpy(outs[k].dst, oo[idx].data, oo[idx].bytes);
    } else if (oo[idx].dtype == RVC_QNN_FLOAT32) {
      const float *s = (const float *)oo[idx].data;
      for (size_t i = 0; i < elems; i++) outs[k].dst[i] = f2h(s[i]);
    } else {
      set_err("%s: output '%s' dtype %d not handled", tag, outs[k].name, oo[idx].dtype);
      rc = -1;
    }
  }
  for (size_t i = 0; i < got; i++) free(oo[i].data);
  return rc;
}

/* ---------- 音频 ---------- */
static int read_wav(const char *path, float **out, size_t *n_out) {
  FILE *f = fopen(path, "rb");
  if (!f) { set_err("open %s failed", path); return -1; }
  unsigned char hdr[12];
  if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
    fclose(f);
    set_err("%s is not a RIFF/WAVE file", path);
    return -1;
  }
  int ch = 1, bits = 16, rate = 16000, fmt = 1;
  long data_off = -1, data_len = 0;
  for (;;) {
    unsigned char ck[8];
    if (fread(ck, 1, 8, f) != 8) break;
    uint32_t sz = (uint32_t)ck[4] | ((uint32_t)ck[5] << 8) | ((uint32_t)ck[6] << 16) |
                  ((uint32_t)ck[7] << 24);
    long here = ftell(f);
    if (!memcmp(ck, "fmt ", 4)) {
      unsigned char b[40];
      memset(b, 0, sizeof(b));
      size_t r = fread(b, 1, sz < 40 ? sz : 40, f);
      if (r >= 16) {
        fmt  = b[0] | (b[1] << 8);
        ch   = b[2] | (b[3] << 8);
        rate = b[4] | (b[5] << 8) | (b[6] << 16) | ((int)b[7] << 24);
        bits = b[14] | (b[15] << 8);
      }
    } else if (!memcmp(ck, "data", 4)) {
      data_off = here;
      data_len = (long)sz;
    }
    if (fseek(f, here + (long)sz + (long)(sz & 1), SEEK_SET) != 0) break;
  }
  if (data_off < 0 || data_len <= 0) {
    fclose(f);
    set_err("no data chunk in %s", path);
    return -1;
  }
  if (fmt != 1 || bits != 16) {
    fclose(f);
    set_err("unsupported wav (fmt=%d bits=%d), need 16-bit PCM", fmt, bits);
    return -1;
  }
  size_t frames = (size_t)data_len / (2 * (size_t)ch);
  short *raw = (short *)malloc(sizeof(short) * frames * (size_t)ch);
  if (!raw) { fclose(f); set_err("oom reading %s", path); return -1; }
  fseek(f, data_off, SEEK_SET);
  size_t got = fread(raw, sizeof(short) * (size_t)ch, frames, f);
  fclose(f);
  frames = got;

  float *mono = (float *)malloc(sizeof(float) * frames);
  if (!mono) { free(raw); set_err("oom"); return -1; }
  for (size_t i = 0; i < frames; i++) {
    int acc = 0;
    for (int c = 0; c < ch; c++) acc += raw[i * (size_t)ch + (size_t)c];
    mono[i] = (float)((double)acc / ch / 32768.0);
  }
  free(raw);

  if (rate == 16000) { *out = mono; *n_out = frames; return 0; }

  /* 重采样到 16 kHz：整数倍用滑动平均（等效低通），否则线性插值 */
  double ratio = 16000.0 / rate;
  size_t n2 = (size_t)((double)frames * ratio);
  float *res = (float *)malloc(sizeof(float) * (n2 ? n2 : 1));
  if (!res) { free(mono); set_err("oom"); return -1; }
  int decim = (rate % 16000 == 0) ? rate / 16000 : 0;
  if (decim > 0) {
    for (size_t i = 0; i < n2; i++) {
      double acc = 0;
      size_t base = i * (size_t)decim;
      for (int k = 0; k < decim && base + (size_t)k < frames; k++)
        acc += mono[base + (size_t)k];
      res[i] = (float)(acc / decim);
    }
  } else {
    for (size_t i = 0; i < n2; i++) {
      double x = (double)i / ratio;
      size_t i0 = (size_t)x;
      double t = x - (double)i0;
      double a = i0 < frames ? mono[i0] : 0.0;
      double b = (i0 + 1) < frames ? mono[i0 + 1] : a;
      res[i] = (float)(a * (1 - t) + b * t);
    }
  }
  free(mono);
  *out = res;
  *n_out = n2;
  return 0;
}

/* ---------- 主流程 ---------- */
/* 音频段在 g_embeds 里的起始行：sink 方案从 NPRE-1 行起（跳过 sink），
 * 左补零方案前面还有 g_pad 行零向量。 */
static size_t audio_row(void) {
  return (size_t)((g_pad >= 0) ? (g_pad + NPRE) : (NPRE - 1));
}

static void build_prompt_embeds(void) {
  uint16_t *p = g_embeds;
  if (g_pad >= 0) {
    /* 0.6B：真实 prompt 前补 g_pad 行。这里不能补零向量——RMSNorm 对零输入是
     * 0/sqrt(0) = NaN，而这几行的 K/V 在 decode 阶段会落到可见槽里（mask 挡不住
     * NaN），整句就退化成重复 token。实测复制首行（与校准脚本一致）可正常工作。 */
    p += (size_t)g_pad * g_hid;
    for (int i = 0; i < NPRE; i++, p += g_hid)
      memcpy(p, g_tbl + (size_t)IDS_PRE[i] * g_hid, sizeof(uint16_t) * g_hid);
    for (int i = 0; i < g_pad; i++)
      memcpy(g_embeds + (size_t)i * g_hid, g_embeds + (size_t)g_pad * g_hid,
             sizeof(uint16_t) * g_hid);
  } else {
    /* 1.7B：IDS_PRE[0] 是 sink 行，它不进图，K/V 由 load_sink_kv 注入 */
    for (int i = 1; i < NPRE; i++, p += g_hid)
      memcpy(p, g_tbl + (size_t)IDS_PRE[i] * g_hid, sizeof(uint16_t) * g_hid);
  }
  p += (size_t)AUDIO_TOK * g_hid;     /* 音频段已由编码器写好 */
  for (int i = 0; i < NPOST; i++, p += g_hid)
    memcpy(p, g_tbl + (size_t)IDS_POST[i] * g_hid, sizeof(uint16_t) * g_hid);
}

/* sink 行的 28 层 K/V 与输入无关，离线算好后随模型一起下发。
 * 文件布局：[layer][k=0/v=1][head][dim]，按槽 CTX-g_tpre-1 拷进 KV cache。 */
static int load_sink_kv(void) {
  char path[1200];
  snprintf(path, sizeof(path), "%s/sink_kv.bin", g_model_dir);
  FILE *f = fopen(path, "rb");
  if (!f) { set_err("sink_kv.bin missing"); return -1; }
  const int slot = CTX - g_tpre - 1;
  for (int lid = 0; lid < NL; lid++)
    for (int side = 0; side < 2; side++) {
      uint16_t *base = g_kv + ((size_t)lid * 2 + side) * KV_ELEMS;
      for (int h = 0; h < NKV; h++) {
        uint16_t *dst = base + ((size_t)h * CTX + slot) * HD;
        if (fread(dst, 2, HD, f) != (size_t)HD) { fclose(f); set_err("sink_kv.bin short"); return -1; }
      }
    }
  fclose(f);
  return 0;
}

/* 临时调试转储（模型目录下 dbg_*.bin），确认与 ORT 参考逐段对齐 */
static void dbg_dump(const char *name, const void *p, size_t bytes) {
  char path[1200];
  snprintf(path, sizeof(path), "%s/dbg_%s.bin", g_model_dir, name);
  FILE *f = fopen(path, "wb");
  if (!f) return;
  fwrite(p, 1, bytes, f);
  fclose(f);
}

/* present_* 先写到 g_kv_out，再拷回 g_kv：QNN 不保证 in/out 同地址时的
 * 读写顺序，直接别名写回可能把还没读的 past 覆盖掉。 */
static void copy_back_presents(int part) {
  size_t bytes = sizeof(uint16_t) * KV_ELEMS;
  for (int j = 0; j < NPART[part]; j++) {
    int lid = LAYERS[part][j];
    memcpy(g_kv + ((size_t)lid * 2 + 0) * KV_ELEMS,
           g_kv_out + (size_t)(2 * j + 0) * KV_ELEMS, bytes);
    memcpy(g_kv + ((size_t)lid * 2 + 1) * KV_ELEMS,
           g_kv_out + (size_t)(2 * j + 1) * KV_ELEMS, bytes);
  }
}

static int run_prefill_part(int part, const uint16_t *embeds, uint16_t *hidden_out) {
  char tag[32], nm[40];
  snprintf(tag, sizeof(tag), "dec_p%d_prefill", part);

  InSpec ins[MAXT];
  OutSpec outs[MAXT];
  int ni = 0, no = 0;

  ins[ni].name = "inputs_embeds"; ins[ni].data = embeds;
  ins[ni].elems = (size_t)g_tpre * g_hid; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;
  ins[ni].name = "attn_mask"; ins[ni].data = g_mask_pre;
  ins[ni].elems = (size_t)g_tpre * CTX; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;
  ins[ni].name = "pos_cos"; ins[ni].data = g_cos_pre;
  ins[ni].elems = (size_t)g_tpre * HD; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;
  ins[ni].name = "pos_sin"; ins[ni].data = g_sin_pre;
  ins[ni].elems = (size_t)g_tpre * HD; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;

  outs[no].name = "hidden"; outs[no].dst = hidden_out;
  outs[no].cap = (size_t)g_tpre * g_hid; no++;

  for (int j = 0; j < NPART[part]; j++) {
    int lid = LAYERS[part][j];
    uint16_t *k = g_kv + ((size_t)lid * 2 + 0) * KV_ELEMS;
    uint16_t *v = g_kv + ((size_t)lid * 2 + 1) * KV_ELEMS;
    snprintf(nm, sizeof(nm), "past_key_%d", lid);
    ins[ni].name = strdup(nm);
    ins[ni].data = k; ins[ni].elems = KV_ELEMS; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;
    snprintf(nm, sizeof(nm), "past_value_%d", lid);
    ins[ni].name = strdup(nm);
    ins[ni].data = v; ins[ni].elems = KV_ELEMS; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;
  }
  /* 输出：hidden, 全部 present_key, 再全部 present_value —— 必须和 ONNX
   * graph.output 的顺序一致，因为 context binary 只会给出 output_0..N。 */
  for (int j = 0; j < NPART[part]; j++) {
    int lid = LAYERS[part][j];
    snprintf(nm, sizeof(nm), "present_key_%d", lid);
    outs[no].name = strdup(nm);
    outs[no].dst = g_kv_out + (size_t)(2 * j + 0) * KV_ELEMS;
    outs[no].cap = KV_ELEMS; no++;
  }
  for (int j = 0; j < NPART[part]; j++) {
    int lid = LAYERS[part][j];
    snprintf(nm, sizeof(nm), "present_value_%d", lid);
    outs[no].name = strdup(nm);
    outs[no].dst = g_kv_out + (size_t)(2 * j + 1) * KV_ELEMS;
    outs[no].cap = KV_ELEMS; no++;
  }

  int64_t t0 = ns_now();
  int rc = run_graph(tag, ins, ni, outs, no);
  t_rec(tag, ns_now() - t0);
  for (int i = 4; i < ni; i++) free((void *)ins[i].name);
  for (int i = 1; i < no; i++) free((void *)outs[i].name);
  if (rc == 0) copy_back_presents(part);
  return rc;
}

static int run_decode_part(int part, int step, const uint16_t *embed,
                           uint16_t *hidden_out) {
  char tag[32], nm[40];
  snprintf(tag, sizeof(tag), "dec_p%d_decode", part);

  uint16_t mask[CTX];
  build_mask_decode(mask, step);
  int pos = g_head + step;
  if (pos >= CTX) pos = CTX - 1;

  InSpec ins[MAXT];
  OutSpec outs[MAXT];
  int ni = 0, no = 0;

  ins[ni].name = "inputs_embeds"; ins[ni].data = embed;
  ins[ni].elems = g_hid; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;
  ins[ni].name = "attn_mask"; ins[ni].data = mask;
  ins[ni].elems = CTX; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;
  ins[ni].name = "pos_cos"; ins[ni].data = g_cos_tab + (size_t)pos * HD;
  ins[ni].elems = HD; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;
  ins[ni].name = "pos_sin"; ins[ni].data = g_sin_tab + (size_t)pos * HD;
  ins[ni].elems = HD; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;

  outs[no].name = "hidden"; outs[no].dst = hidden_out; outs[no].cap = g_hid; no++;

  for (int j = 0; j < NPART[part]; j++) {
    int lid = LAYERS[part][j];
    uint16_t *k = g_kv + ((size_t)lid * 2 + 0) * KV_ELEMS;
    uint16_t *v = g_kv + ((size_t)lid * 2 + 1) * KV_ELEMS;
    snprintf(nm, sizeof(nm), "past_key_%d", lid);
    ins[ni].name = strdup(nm);
    ins[ni].data = k; ins[ni].elems = KV_ELEMS; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;
    snprintf(nm, sizeof(nm), "past_value_%d", lid);
    ins[ni].name = strdup(nm);
    ins[ni].data = v; ins[ni].elems = KV_ELEMS; ins[ni].dtype = RVC_QNN_FLOAT16; ni++;
  }
  /* 输出：hidden, 全部 present_key, 再全部 present_value —— 必须和 ONNX
   * graph.output 的顺序一致，因为 context binary 只会给出 output_0..N。 */
  for (int j = 0; j < NPART[part]; j++) {
    int lid = LAYERS[part][j];
    snprintf(nm, sizeof(nm), "present_key_%d", lid);
    outs[no].name = strdup(nm);
    outs[no].dst = g_kv_out + (size_t)(2 * j + 0) * KV_ELEMS;
    outs[no].cap = KV_ELEMS; no++;
  }
  for (int j = 0; j < NPART[part]; j++) {
    int lid = LAYERS[part][j];
    snprintf(nm, sizeof(nm), "present_value_%d", lid);
    outs[no].name = strdup(nm);
    outs[no].dst = g_kv_out + (size_t)(2 * j + 1) * KV_ELEMS;
    outs[no].cap = KV_ELEMS; no++;
  }

  int64_t t0 = ns_now();
  int rc = run_graph(tag, ins, ni, outs, no);
  t_rec(tag, ns_now() - t0);
  for (int i = 4; i < ni; i++) free((void *)ins[i].name);
  for (int i = 1; i < no; i++) free((void *)outs[i].name);
  if (rc == 0) copy_back_presents(part);
  return rc;
}

static int lm_head_argmax(const uint16_t *hidden) {
  InSpec in;
  in.name = "hidden"; in.data = hidden; in.elems = g_hid; in.dtype = RVC_QNN_FLOAT16;

  /* 这里走的是 qnn_exec_raw（要拿全量 logits），不经过 run_graph，
   * 所以得自己确保图已加载 + 描述符已填。 */
  int gi = gi_of("lm_head");
  if (g_ensure(gi)) { set_err("load lm_head: %s", qnn_last_error()); return -1; }

  Feed f;
  memset(&f, 0, sizeof(f));
  int want = dtype_in(gi, "hidden");
  if (want < 0) { set_err("lm_head: no input 'hidden'"); return -1; }
  if (feed_add(&f, "hidden", hidden, g_hid, RVC_QNN_FLOAT16, want)) { feed_reset(&f); return -1; }

  QnnIo oo[4];
  memset(oo, 0, sizeof(oo));
  size_t got = 0;
  int64_t tl0 = ns_now();
  int lrc = qnn_exec_raw("lm_head", f.io, (size_t)f.n, oo, 4, &got);
  t_rec("lm_head", ns_now() - tl0);
  if (lrc) {
    feed_reset(&f);
    set_err("lm_head: %s", qnn_last_error());
    return -1;
  }
  feed_reset(&f);

  /* context binary 会把输出改名成 output_0..N，所以按名字找不到就取第一个 */
  int li = -1;
  for (size_t i = 0; i < got; i++)
    if (oo[i].name && !strcmp(oo[i].name, "logits")) { li = (int)i; break; }
  if (li < 0 && got > 0) li = 0;

  int best = -1;
  float bv = -1e30f;
  if (li >= 0) {
    size_t es = qnn_elem_size(oo[li].dtype);
    size_t elems = oo[li].bytes / es;
    for (size_t q = 0; q < elems && q < VOCAB; q++) {
      float v = (oo[li].dtype == RVC_QNN_FLOAT16)
                    ? h2f(((const uint16_t *)oo[li].data)[q])
                    : ((const float *)oo[li].data)[q];
      if (v > bv) { bv = v; best = (int)q; }
    }
  }
  {
    static int dumped = 0;
    if (li >= 0 && !dumped) { dbg_dump("log0", oo[li].data, oo[li].bytes); dumped = 1; }
  }
  for (size_t i = 0; i < got; i++) free(oo[i].data);
  if (best < 0) set_err("lm_head: logits missing");
  return best;
}

/* ---------- 单独验证编码器（DLC 到手即可先比对，不必等全部 8 张图） ---------- */
int asr_probe_enc(const char *lib_dir, const char *enc_bin, const char *mel_path,
                  const char *out_path) {
  pthread_mutex_lock(&g_lock);
  g_err[0] = '\0';
  if (qnn_init(lib_dir, lib_dir)) { set_err("qnn_init: %s", qnn_last_error()); goto fail; }
  /* probe 不走 asr_engine_init，得自己把模型目录填好：g_ensure 用 g_model_dir
   * 拼 <dir>/<tag>.bin，留空会拼出 "/enc.bin"。这里从 enc_bin 反推目录。 */
  {
    const char *slash = strrchr(enc_bin, '/');
    size_t n = slash ? (size_t)(slash - enc_bin) : 0;
    if (n >= sizeof(g_model_dir)) n = sizeof(g_model_dir) - 1;
    memcpy(g_model_dir, enc_bin, n);
    g_model_dir[n] = '\0';
  }
  /* onCreate 的预热会把 ASR 的 enc 常驻（g_loaded=1），g_ensure 于是直接返回、
   * 探针实际跑的是 files/asr/enc.bin——aligner 的图永远测不到。这里强制卸掉
   * 已常驻的 enc，保证 probe 用的就是传进来的那个 bin。 */
  {
    int e = gi_of("enc");
    if (e >= 0 && g_loaded[e]) { qnn_free(GRAPH_TAGS[e]); g_loaded[e] = 0; }
  }
  g_last_out_elems = 0;
  if (g_ensure(gi_of("enc"))) { set_err("load enc: %s", qnn_last_error()); goto fail; }
  {
    size_t mel_bytes = N_MEL * 2;
    /* probe 不经过 asr_engine_init，g_hid 可能是 0；用 ASR_HID 当上限，
     * 落盘时再按 g_last_out_elems 收窄（aligner 的 enc 输出只有 104x1024）。 */
    size_t hid_cap = g_hid > 0 ? (size_t)g_hid : (size_t)ASR_HID;
    uint16_t *mel = (uint16_t *)malloc(mel_bytes);
    uint16_t *out = (uint16_t *)malloc((size_t)AUDIO_TOK * hid_cap * 2);
    if (!mel || !out) { free(mel); free(out); set_err("oom"); goto fail; }
    FILE *f = fopen(mel_path, "rb");
    if (!f) { free(mel); free(out); set_err("open %s failed", mel_path); goto fail; }
    size_t got = fread(mel, 1, mel_bytes, f);
    fclose(f);
    if (got < mel_bytes) {
      free(mel); free(out);
      set_err("%s has %zu bytes, want %zu", mel_path, got, mel_bytes);
      goto fail;
    }
    InSpec in;
    in.name = "mel"; in.data = mel; in.elems = N_MEL; in.dtype = RVC_QNN_FLOAT16;
    OutSpec os;
    os.name = "audio_embeds"; os.dst = out; os.cap = (size_t)AUDIO_TOK * hid_cap;
    int rc = run_graph("enc", &in, 1, &os, 1);
    free(mel);
    if (rc) { free(out); goto fail; }
    FILE *o = fopen(out_path, "wb");
    if (!o) { free(out); set_err("create %s failed", out_path); goto fail; }
    size_t nw = g_last_out_elems ? g_last_out_elems : (size_t)AUDIO_TOK * hid_cap;
    if (nw > (size_t)AUDIO_TOK * hid_cap) nw = (size_t)AUDIO_TOK * hid_cap;
    fwrite(out, 2, nw, o);
    fclose(o);
    free(out);
  }
  pthread_mutex_unlock(&g_lock);
  return 0;
fail:
  pthread_mutex_unlock(&g_lock);
  return -1;
}

int asr_engine_preload(void) {
  pthread_mutex_lock(&g_lock);
  if (!g_init) {
    set_err("asr_engine_init not called");
    pthread_mutex_unlock(&g_lock);
    return -1;
  }
  /* 只预载 KEEP[] 里常驻的那几张（enc / decode / lm_head，约 4.6 GB 里的一大半）。
   * prefill 图用完即卸，且三张同时驻留会超出 DSP 内存，不能一起预载。 */
  for (int i = 0; i < NGRAPH; i++) {
    if (!KEEP[i]) continue;
    if (g_ensure(i)) { pthread_mutex_unlock(&g_lock); return -1; }
  }
  t_reset();          /* 预热耗时不要混进第一次 run 的统计 */
  pthread_mutex_unlock(&g_lock);
  return 0;
}

int asr_engine_run_wav(const char *wav_path, int max_new, int *ids, int cap, int *n_out) {
  pthread_mutex_lock(&g_lock);
  if (!g_init) {
    set_err("asr_engine_init not called");
    pthread_mutex_unlock(&g_lock);
    return -1;
  }
  if (max_new > ASR_MAX_NEW) max_new = ASR_MAX_NEW;
  if (max_new < 1) max_new = 1;
  g_release_all();   /* 上次失败可能留下半套图 */
  t_reset();

  float *pcm = NULL;
  size_t npcm = 0;
  if (read_wav(wav_path, &pcm, &npcm)) { pthread_mutex_unlock(&g_lock); return -1; }

  asr_mel_compute(&g_mel_ctx, pcm, npcm, g_mel_f32);
  free(pcm);
  for (size_t i = 0; i < N_MEL; i++) g_mel[i] = f2h(g_mel_f32[i]);

  /* 编码器：mel -> audio_embeds，直接落到 g_embeds 的音频段 */
  {
    InSpec in;
    in.name = "mel"; in.data = g_mel; in.elems = N_MEL; in.dtype = RVC_QNN_FLOAT16;
    OutSpec out;
    out.name = "audio_embeds";
    out.dst = g_embeds + audio_row() * g_hid;
    out.cap = (size_t)AUDIO_TOK * g_hid;
    int64_t t0 = ns_now();
    int rc = run_graph("enc", &in, 1, &out, 1);
    t_rec("enc", ns_now() - t0);
    if (rc) { pthread_mutex_unlock(&g_lock); return -1; }
    dbg_dump("mel", g_mel, N_MEL * 2);
    dbg_dump("audio", g_embeds + audio_row() * g_hid,
             (size_t)AUDIO_TOK * g_hid * 2);
  }
  g_release(gi_of("enc"));            /* 642 MB，后面用不到了 */
  build_prompt_embeds();

  memset(g_kv, 0, sizeof(uint16_t) * NL * 2 * KV_ELEMS);
  if (g_pad < 0 && load_sink_kv()) { pthread_mutex_unlock(&g_lock); return -1; }
  /* 每段吃上一段的 hidden（不是原始 embeds），两块 buffer 交替避免 in/out 同址 */
  {
    const uint16_t *pin = g_embeds;
    uint16_t *pout = g_hidden_pre;
    for (int p = 0; p < 3; p++) {
      if (run_prefill_part(p, pin, pout)) { pthread_mutex_unlock(&g_lock); return -1; }
      /* 用完立刻卸，别让三张 prefill 图同时占着 DSP 内存 */
      {
        char t[32];
        snprintf(t, sizeof(t), "dec_p%d_prefill", p);
        g_release(gi_of(t));
      }
      {
        char dn[32];
        snprintf(dn, sizeof(dn), "hp%d", p);
        dbg_dump(dn, pout, sizeof(uint16_t) * (size_t)g_tpre * g_hid);
        if (p == 0) {
          for (int j = 0; j < 10; j++) {
            char nm[16];
            snprintf(nm, sizeof(nm), "k%d", j);
            dbg_dump(nm, g_kv_out + (size_t)(2 * j) * KV_ELEMS,
                     sizeof(uint16_t) * (size_t)KV_ELEMS);
            snprintf(nm, sizeof(nm), "v%d", j);
            dbg_dump(nm, g_kv_out + (size_t)(2 * j + 1) * KV_ELEMS,
                     sizeof(uint16_t) * (size_t)KV_ELEMS);
          }
        }
      }
      pin = pout;
      pout = (pout == g_hidden_pre) ? g_hidden_pre2 : g_hidden_pre;
    }
    g_final_pre = pin;          /* 最后一段的输出 */
  }

  for (int p = 0; p < 3; p++) {
    char t[32];
    snprintf(t, sizeof(t), "dec_p%d_prefill", p);
    g_release(gi_of(t));
  }

  dbg_dump("embeds", g_embeds, sizeof(uint16_t) * (size_t)g_tpre * g_hid);
  dbg_dump("mask", g_mask_pre, sizeof(uint16_t) * (size_t)g_tpre * CTX);
  dbg_dump("cos", g_cos_pre, sizeof(uint16_t) * (size_t)g_tpre * HD);
  dbg_dump("sin", g_sin_pre, sizeof(uint16_t) * (size_t)g_tpre * HD);
  int nxt = lm_head_argmax(g_final_pre + (size_t)(g_tpre - 1) * g_hid);
  if (nxt < 0) { pthread_mutex_unlock(&g_lock); return -1; }

  int n = 0;
  for (int step = 0; step < max_new && n < cap; step++) {
    if (nxt == 151645) break;                 /* <|im_end|> */
    if (nxt == 151643) break;                 /* <|endoftext|> */
    ids[n++] = nxt;
    const uint16_t *emb = g_tbl + (size_t)nxt * g_hid;
    uint16_t *dout = g_hidden_dec;
    for (int p = 0; p < 3; p++) {
      if (run_decode_part(p, step, emb, dout)) {
        pthread_mutex_unlock(&g_lock);
        return -1;
      }
      emb = dout;                             /* 下一段吃上一段的 hidden */
      dout = (dout == g_hidden_dec) ? g_hidden_dec2 : g_hidden_dec;
    }
    nxt = lm_head_argmax(emb);
    if (nxt < 0) { pthread_mutex_unlock(&g_lock); return -1; }
  }
  *n_out = n;
  g_release_all();
  pthread_mutex_unlock(&g_lock);
  return 0;
}
