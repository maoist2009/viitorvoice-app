/* vad_engine.c — Silero VAD 推理（QNN HTP，一次 K 帧）。
 *
 * 图接口（见 tools/silero/export_qnn.py）：
 *   input[1,K*512] f16  state[2,1,128] f16  ctx[1,64] f16
 *   → probs[1,K]  stateN  ctxN
 *
 * 为什么一次 K 帧：HTP 每次 graphExecute 有 ~1 ms 固定开销，单帧图实测
 * 1.08 ms/帧（AI Hub profile 报的 0.09 ms 只是纯计算）。把 K 帧的 LSTM 状态
 * 串在图内展开后，一次 execute 摊掉这笔开销。
 *
 * 输出张量的顺序在 link 成 context binary 之后不保证和 ONNX 一致，所以按
 * **元素数唯一匹配**来认（K / 256 / 64），不靠名字也不靠下标。
 *
 * LSTM 状态和上下文留在引擎里，Kotlin 侧只管按块喂音频、拿概率。
 */
#include "vad_engine.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qnn_executor.h"

#include <android/log.h>
#define VAD_LOG(...) __android_log_print(ANDROID_LOG_INFO, "AsrDemo", __VA_ARGS__)

#define VAD_TAG "silero_vad"
#define VAD_MAX_IO 4

static char g_err[512];
static int g_inited;
static int g_frames = 1;           /* 一张图展开几帧 */
static int g_n_in, g_n_out;
static char g_in_name[VAD_MAX_IO][64];
static int g_in_dtype[VAD_MAX_IO];
static size_t g_in_elems[VAD_MAX_IO];
static char g_out_name[VAD_MAX_IO][64];
static int g_out_dtype[VAD_MAX_IO];
static size_t g_out_elems[VAD_MAX_IO];

static float g_state[VAD_STATE];
static float g_ctx[VAD_CTX];

static uint16_t *g_h_in;           /* K*512 */
static uint16_t g_h_state[VAD_STATE];
static uint16_t g_h_ctx[VAD_CTX];

static void set_err(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
}

const char *vad_engine_error(void) { return g_err; }
const char *vad_engine_timings(void) { return qnn_timings_json(); }

int vad_engine_frames(void) { return g_frames; }

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
    if (rem > half || (rem == half && (r & 1u))) r++;
    return (uint16_t)(s | r);
  }
  uint32_t r = ((uint32_t)e << 10) | (m >> 13);
  uint32_t rem = m & 0x1fffu;
  if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r++;
  return (uint16_t)(s | r);
}

static float h2f(uint16_t h) {
  uint32_t s = (uint32_t)(h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu;
  uint32_t m = h & 0x3ffu;
  uint32_t f;
  if (e == 0) {
    if (m == 0) f = s;
    else {
      int exp = -1;
      do { m <<= 1; exp++; } while (!(m & 0x400));
      f = s | ((uint32_t)(127 - 15 - exp) << 23) | ((m & 0x3ffu) << 13);
    }
  } else if (e == 31) {
    f = s | 0x7f800000u | (m << 13);
  } else {
    f = s | ((e + 112) << 23) | (m << 13);
  }
  float out;
  memcpy(&out, &f, 4);
  return out;
}

/* ---------- 描述回调 ---------- */
typedef struct {
  int n_in, n_out;
  char names[VAD_MAX_IO][64];
  int dtype[VAD_MAX_IO];
  size_t elems[VAD_MAX_IO];
  char onames[VAD_MAX_IO][64];
  int odtype[VAD_MAX_IO];
  size_t oelems[VAD_MAX_IO];
} desc_t;

static void desc_in_cb(void *user, const char *name, int dtype, uint32_t rank,
                       const uint32_t *dims) {
  desc_t *d = (desc_t *)user;
  if (d->n_in >= VAD_MAX_IO) return;
  size_t n = 1;
  for (uint32_t i = 0; i < rank; i++) n *= dims[i];
  snprintf(d->names[d->n_in], sizeof(d->names[0]), "%s", name);
  d->dtype[d->n_in] = dtype;
  d->elems[d->n_in] = n;
  d->n_in++;
}

static void desc_out_cb(void *user, const char *name, int dtype, uint32_t rank,
                        const uint32_t *dims) {
  desc_t *d = (desc_t *)user;
  if (d->n_out >= VAD_MAX_IO) return;
  size_t n = 1;
  for (uint32_t i = 0; i < rank; i++) n *= dims[i];
  snprintf(d->onames[d->n_out], sizeof(d->onames[0]), "%s", name);
  d->odtype[d->n_out] = dtype;
  d->oelems[d->n_out] = n;
  d->n_out++;
}

/* ---------- 初始化 ---------- */
int vad_engine_init(const char *model_dir, const char *lib_dir) {
  if (g_inited) return 0;
  char p[1024];

  snprintf(p, sizeof(p), "%s/silero_vad.bin", model_dir);
  if (qnn_init(lib_dir, lib_dir)) { set_err("qnn_init: %s", qnn_last_error()); return -1; }
  if (qnn_load(VAD_TAG, p)) { set_err("load silero_vad.bin: %s", qnn_last_error()); return -1; }

  desc_t d;
  memset(&d, 0, sizeof(d));
  if (qnn_describe(VAD_TAG, desc_in_cb, desc_out_cb, &d)) {
    set_err("describe silero_vad: %s", qnn_last_error());
    qnn_free(VAD_TAG);
    return -1;
  }
  g_n_in = d.n_in;
  g_n_out = d.n_out;
  for (int i = 0; i < d.n_in && i < VAD_MAX_IO; i++) {
    snprintf(g_in_name[i], sizeof(g_in_name[0]), "%s", d.names[i]);
    g_in_dtype[i] = d.dtype[i];
    g_in_elems[i] = d.elems[i];
  }
  for (int i = 0; i < d.n_out && i < VAD_MAX_IO; i++) {
    snprintf(g_out_name[i], sizeof(g_out_name[0]), "%s", d.onames[i]);
    g_out_dtype[i] = d.odtype[i];
    g_out_elems[i] = d.oelems[i];
  }
  if (g_n_in != 3 || g_n_out != 3) {
    set_err("silero_vad 图期望 3 进 3 出，实际 %d/%d", g_n_in, g_n_out);
    qnn_free(VAD_TAG);
    return -1;
  }
  /* 输入按元素数认：K*512（音频）/ 256（state）/ 64（ctx） */
  size_t audio_elems = 0;
  for (int i = 0; i < 3; i++)
    if (g_in_elems[i] > audio_elems && g_in_elems[i] % VAD_CHUNK == 0) audio_elems = g_in_elems[i];
  if (!audio_elems) {
    set_err("输入里找不到 K*512 的音频张量");
    qnn_free(VAD_TAG);
    return -1;
  }
  g_frames = (int)(audio_elems / VAD_CHUNK);
  int has_state = 0, has_ctx = 0, has_probs = 0;
  for (int i = 0; i < 3; i++) {
    if (g_in_elems[i] == VAD_STATE) has_state = 1;
    if (g_in_elems[i] == VAD_CTX) has_ctx = 1;
    if (g_out_elems[i] == (size_t)g_frames) has_probs = 1;
  }
  if (!has_state || !has_ctx || !has_probs) {
    set_err("图接口不对（state=%d ctx=%d probs=%d, K=%d）", has_state, has_ctx, has_probs, g_frames);
    qnn_free(VAD_TAG);
    return -1;
  }
  g_h_in = (uint16_t *)malloc(sizeof(uint16_t) * audio_elems);
  if (!g_h_in) {
    set_err("out of memory (%zu)", audio_elems * 2);
    qnn_free(VAD_TAG);
    return -1;
  }

  VAD_LOG("VAD IO: %d in / %d out, K=%d", g_n_in, g_n_out, g_frames);
  for (int i = 0; i < g_n_in; i++)
    VAD_LOG("  in  '%s' dtype=%d elems=%zu", g_in_name[i], g_in_dtype[i], g_in_elems[i]);
  for (int i = 0; i < g_n_out; i++)
    VAD_LOG("  out '%s' dtype=%d elems=%zu", g_out_name[i], g_out_dtype[i], g_out_elems[i]);

  memset(g_state, 0, sizeof(g_state));
  memset(g_ctx, 0, sizeof(g_ctx));
  g_inited = 1;
  return 0;
}

void vad_engine_reset(void) {
  memset(g_state, 0, sizeof(g_state));
  memset(g_ctx, 0, sizeof(g_ctx));
}

void vad_engine_shutdown(void) {
  if (!g_inited) return;
  qnn_free(VAD_TAG);
  qnn_shutdown_if_empty();
  free(g_h_in);
  g_h_in = NULL;
  g_inited = 0;
}

/* 按元素数找输入槽位 */
static int slot_for(size_t elems) {
  for (int i = 0; i < g_n_in; i++) if (g_in_elems[i] == elems) return i;
  return -1;
}

int vad_engine_run(const float *pcm, int n_frames, float *probs) {
  if (!g_inited) { set_err("engine not initialized"); return -1; }
  if (!pcm || !probs) { set_err("bad args"); return -1; }
  if (n_frames != g_frames) {
    set_err("需要 %d 帧，收到 %d", g_frames, n_frames);
    return -1;
  }

  QnnIo in[3];
  memset(in, 0, sizeof(in));
  size_t sizes[3] = {(size_t)g_frames * VAD_CHUNK, VAD_STATE, VAD_CTX};
  const float *srcs[3] = {pcm, g_state, g_ctx};
  for (int k = 0; k < 3; k++) {
    int s = slot_for(sizes[k]);
    if (s < 0) { set_err("输入槽位缺失 (%zu)", sizes[k]); return -1; }
    in[s].name = g_in_name[s];
    in[s].dtype = g_in_dtype[s];
    in[s].bytes = qnn_elem_size(g_in_dtype[s]) * g_in_elems[s];
    if (g_in_dtype[s] == RVC_QNN_FLOAT16) {
      uint16_t *h = sizes[k] == VAD_STATE ? g_h_state
                  : (sizes[k] == VAD_CTX ? g_h_ctx : g_h_in);
      for (size_t i = 0; i < sizes[k]; i++) h[i] = f2h(srcs[k][i]);
      in[s].data = h;
    } else {
      in[s].data = (void *)srcs[k];
    }
  }

  QnnIo out[3];
  memset(out, 0, sizeof(out));
  size_t n_out = 0;
  int rc = qnn_exec_raw(VAD_TAG, in, 3, out, 3, &n_out);
  if (rc) { set_err("exec silero_vad: %s", qnn_last_error()); return -1; }
  if (n_out != 3) {
    for (int i = 0; i < (int)n_out && i < 3; i++) free(out[i].data);
    set_err("输出个数 %zu != 3", n_out);
    return -1;
  }

  /* 按元素数唯一匹配：K → probs，256 → stateN，64 → ctxN */
  int seen_p = 0, seen_s = 0, seen_c = 0;
  for (int i = 0; i < 3; i++) {
    size_t es = qnn_elem_size(out[i].dtype);
    size_t n = es ? out[i].bytes / es : 0;
    if (n == (size_t)g_frames && !seen_p) {
      if (out[i].dtype == RVC_QNN_FLOAT16) {
        const uint16_t *h = (const uint16_t *)out[i].data;
        for (int k = 0; k < g_frames; k++) probs[k] = h2f(h[k]);
      } else {
        memcpy(probs, out[i].data, sizeof(float) * g_frames);
      }
      seen_p = 1;
    } else if (n == VAD_STATE && !seen_s) {
      if (out[i].dtype == RVC_QNN_FLOAT16) {
        const uint16_t *h = (const uint16_t *)out[i].data;
        for (int k = 0; k < VAD_STATE; k++) g_state[k] = h2f(h[k]);
      } else {
        memcpy(g_state, out[i].data, sizeof(float) * VAD_STATE);
      }
      seen_s = 1;
    } else if (n == VAD_CTX && !seen_c) {
      if (out[i].dtype == RVC_QNN_FLOAT16) {
        const uint16_t *h = (const uint16_t *)out[i].data;
        for (int k = 0; k < VAD_CTX; k++) g_ctx[k] = h2f(h[k]);
      } else {
        memcpy(g_ctx, out[i].data, sizeof(float) * VAD_CTX);
      }
      seen_c = 1;
    }
    free(out[i].data);
  }
  if (!seen_p || !seen_s || !seen_c) {
    set_err("输出没匹配全 (probs=%d state=%d ctx=%d)", seen_p, seen_s, seen_c);
    return -1;
  }
  return 0;
}
