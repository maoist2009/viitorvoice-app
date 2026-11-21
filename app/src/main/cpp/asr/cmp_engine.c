/* cmp_engine.c — CAM++ 声纹推理（QNN HTP），见 cmp_engine.h。
 *
 * 和 sv_engine.c 一样直接驱动 qnn_executor 的 raw 接口，但这里只有一个输入
 * 一个输出，没有 tokenizer / CTC，逻辑简单得多。
 */
#include "cmp_engine.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cmp_frontend.h"
#include "qnn_executor.h"

#include <android/log.h>
#define CMP_LOG(...) __android_log_print(ANDROID_LOG_INFO, "AsrDemo", __VA_ARGS__)

#define CMP_TAG "campplus"
#define CMP_MAX_IO 4

static char g_err[512];
static int g_inited;
static cmp_front_t g_front;
static float *g_feats;        /* CMP_T * CMP_N_MELS */
static float *g_emb;          /* CMP_EMB_DIM */

static int g_n_in, g_n_out;
static char g_in_name[CMP_MAX_IO][64];
static int g_in_dtype[CMP_MAX_IO];
static size_t g_in_elems[CMP_MAX_IO];
static char g_out_name[CMP_MAX_IO][64];
static int g_out_dtype[CMP_MAX_IO];
static size_t g_out_elems[CMP_MAX_IO];

static void set_err(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
}

const char *cmp_engine_error(void) { return g_err; }
const char *cmp_timings(void) { return qnn_timings_json(); }

/* ---------- fp16 ---------- */
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

/* ---------- 描述回调（输入/输出分开存，别共用数组） ---------- */
typedef struct {
  int n_in, n_out;
  char names[CMP_MAX_IO][64];
  int dtype[CMP_MAX_IO];
  size_t elems[CMP_MAX_IO];
  char onames[CMP_MAX_IO][64];
  int odtype[CMP_MAX_IO];
  size_t oelems[CMP_MAX_IO];
} desc_t;

static void desc_in_cb(void *user, const char *name, int dtype, uint32_t rank,
                       const uint32_t *dims) {
  desc_t *d = (desc_t *)user;
  if (d->n_in >= CMP_MAX_IO) return;
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
  if (d->n_out >= CMP_MAX_IO) return;
  size_t n = 1;
  for (uint32_t i = 0; i < rank; i++) n *= dims[i];
  snprintf(d->onames[d->n_out], sizeof(d->onames[0]), "%s", name);
  d->odtype[d->n_out] = dtype;
  d->oelems[d->n_out] = n;
  d->n_out++;
}

/* ---------- WAV ---------- */
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

  float *mono = (float *)malloc(sizeof(float) * (frames ? frames : 1));
  if (!mono) { free(raw); set_err("oom"); return -1; }
  for (size_t i = 0; i < frames; i++) {
    int acc = 0;
    for (int c = 0; c < ch; c++) acc += raw[i * (size_t)ch + (size_t)c];
    mono[i] = (float)((double)acc / ch / 32768.0);
  }
  free(raw);

  if (rate == 16000) { *out = mono; *n_out = frames; return 0; }
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

/* ---------- 初始化 ---------- */
int cmp_engine_init(const char *model_dir, const char *lib_dir) {
  if (g_inited) return 0;
  char p[1024];

  if (cmp_front_init(&g_front)) { set_err("frontend init failed"); return -1; }

  g_feats = (float *)calloc((size_t)CMP_T * CMP_N_MELS, sizeof(float));
  g_emb = (float *)calloc(CMP_EMB_DIM, sizeof(float));
  if (!g_feats || !g_emb) { set_err("oom"); return -1; }

  snprintf(p, sizeof(p), "%s/campplus.bin", model_dir);
  if (qnn_init(lib_dir, lib_dir)) { set_err("qnn_init: %s", qnn_last_error()); return -1; }
  if (qnn_load(CMP_TAG, p)) { set_err("load campplus.bin: %s", qnn_last_error()); return -1; }

  desc_t d;
  memset(&d, 0, sizeof(d));
  if (qnn_describe(CMP_TAG, desc_in_cb, desc_out_cb, &d)) {
    set_err("describe campplus: %s", qnn_last_error());
    return -1;
  }
  g_n_in = d.n_in;
  g_n_out = d.n_out;
  for (int i = 0; i < d.n_in && i < CMP_MAX_IO; i++) {
    snprintf(g_in_name[i], sizeof(g_in_name[0]), "%s", d.names[i]);
    g_in_dtype[i] = d.dtype[i];
    g_in_elems[i] = d.elems[i];
  }
  for (int i = 0; i < d.n_out && i < CMP_MAX_IO; i++) {
    snprintf(g_out_name[i], sizeof(g_out_name[0]), "%s", d.onames[i]);
    g_out_dtype[i] = d.odtype[i];
    g_out_elems[i] = d.oelems[i];
  }
  if (g_n_in != 1 || g_n_out != 1) {
    set_err("campplus 图期望 1 进 1 出，实际 %d/%d", g_n_in, g_n_out);
    return -1;
  }
  if (g_in_elems[0] != (size_t)CMP_T * CMP_N_MELS) {
    set_err("campplus 输入元素数 %zu != %d", g_in_elems[0], CMP_T * CMP_N_MELS);
    return -1;
  }
  CMP_LOG("CMP IO: in '%s' dtype=%d elems=%zu | out '%s' dtype=%d elems=%zu",
          g_in_name[0], g_in_dtype[0], g_in_elems[0],
          g_out_name[0], g_out_dtype[0], g_out_elems[0]);
  g_inited = 1;
  return 0;
}

void cmp_engine_shutdown(void) {
  if (!g_inited) return;
  qnn_free(CMP_TAG);
  qnn_shutdown_if_empty();
  free(g_feats); g_feats = NULL;
  free(g_emb); g_emb = NULL;
  g_inited = 0;
}

/* ---------- 推理 ---------- */
int cmp_engine_embed_pcm(const float *pcm, size_t n, float *emb, int cap) {
  if (!g_inited) { set_err("engine not initialized"); return -1; }
  if (!pcm || !emb || cap < CMP_EMB_DIM) { set_err("bad args"); return -1; }

  int n_valid = 0;
  if (cmp_front_compute(&g_front, pcm, n, g_feats, &n_valid)) {
    set_err("frontend failed (n=%zu)", n);
    return -1;
  }

  void *in_buf = g_feats;
  uint16_t *hbuf = NULL;
  if (g_in_dtype[0] == RVC_QNN_FLOAT16) {
    hbuf = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)CMP_T * CMP_N_MELS);
    if (!hbuf) { set_err("oom"); return -1; }
    for (size_t i = 0; i < (size_t)CMP_T * CMP_N_MELS; i++) hbuf[i] = f2h(g_feats[i]);
    in_buf = hbuf;
  }

  QnnIo in;
  in.name = g_in_name[0];
  in.dtype = g_in_dtype[0];
  in.data = in_buf;
  in.bytes = qnn_elem_size(g_in_dtype[0]) * g_in_elems[0];

  QnnIo out;
  out.name = g_out_name[0];
  out.dtype = g_out_dtype[0];
  out.data = NULL;
  out.bytes = 0;

  size_t n_out = 0;
  int rc = qnn_exec_raw(CMP_TAG, &in, 1, &out, 1, &n_out);
  free(hbuf);
  if (rc) { set_err("exec campplus: %s", qnn_last_error()); return -1; }
  if (n_out != 1 || out.bytes < sizeof(float) * CMP_EMB_DIM) {
    free(out.data);
    set_err("campplus 输出异常 (n_out=%zu bytes=%zu)", n_out, out.bytes);
    return -1;
  }

  if (out.dtype == RVC_QNN_FLOAT16) {
    const uint16_t *h = (const uint16_t *)out.data;
    for (int i = 0; i < CMP_EMB_DIM; i++) emb[i] = h2f(h[i]);
  } else {
    memcpy(emb, out.data, sizeof(float) * CMP_EMB_DIM);
  }
  free(out.data);
  return 0;
}

int cmp_engine_embed_wav(const char *wav_path, float *emb, int cap) {
  float *pcm = NULL;
  size_t n = 0;
  if (read_wav(wav_path, &pcm, &n)) return -1;
  int rc = cmp_engine_embed_pcm(pcm, n, emb, cap);
  free(pcm);
  return rc;
}
