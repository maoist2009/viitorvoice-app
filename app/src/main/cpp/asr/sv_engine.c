/* sv_engine.c — SenseVoice-Small 端侧管线，见 sv_engine.h。 */
#include "sv_engine.h"

#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qnn_executor.h"
#include "sv_frontend.h"

#define SV_TAG     "sv"
#define SV_VOCAB   25055
#define SV_RICH    4        /* 图输出开头的 rich 标签帧：语言/情感/事件/ITN */

static char g_err[512];
static int  g_inited;
static sv_front_t g_front;
static float *g_prefix;      /* 4*560 */
static float *g_feats;       /* 500*560 */
static float *g_mask;        /* 500 */
static uint32_t g_ntok;
static uint32_t *g_off;      /* ntok+1 */
static char *g_blob;

/* 图边界（AI Hub 编出来的接口可能和 ONNX 不同：prefix 会被折成常量，
 * 输出缓冲有时会被列为输入 output_0），所以这里按 describe 的结果动态对齐。 */
#define SV_MAX_IO 8
static int g_n_in, g_n_out;
static char g_in_name[SV_MAX_IO][64];
static int g_in_dtype[SV_MAX_IO];
static size_t g_in_elems[SV_MAX_IO];
static const float *g_in_src[SV_MAX_IO];   /* NULL = 补零 */
static int g_out_idx = -1;
static int g_out_dtype = RVC_QNN_FLOAT32;
static size_t g_out_elems = SV_T * SV_VOCAB;

static void set_err(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
}

const char *sv_engine_error(void) { return g_err; }
const char *sv_timings(void) { return qnn_timings_json(); }

/* token id -> UTF-8 文本（越界返回空串）。sv_tokens.bin 里 token 之间没有
 * NUL，靠 off[id..id+1] 切；这里拷进静态缓冲再收尾（引擎单线程，JNI 逐个取用）。 */
const char *sv_engine_token(int id) {
  static char buf[256];
  if (!g_blob || !g_off || id < 0 || (uint32_t)id >= g_ntok) return "";
  uint32_t a = g_off[id], b = g_off[id + 1];
  size_t n = (size_t)(b - a);
  if (n >= sizeof(buf)) n = sizeof(buf) - 1;
  memcpy(buf, g_blob + a, n);
  buf[n] = 0;
  return buf;
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

/* argmax over one CTC frame; blank(0) wins ties, which is what we want. */
static int argmax_row(const float *row) {
  int best = 0;
  float bv = row[0];
  for (int v = 1; v < SV_VOCAB; v++) {
    if (row[v] > bv) { bv = row[v]; best = v; }
  }
  return best;
}

/* ---------- 描述回调 ---------- */
typedef struct {
  int n_in, n_out;
  char names[8][64];
  int dtype[8];
  size_t elems[8];
  char onames[8][64];   /* 输出单独存：和输入共用数组会被覆盖 */
  int odtype[8];
  size_t oelems[8];
} desc_t;

static void desc_in_cb(void *user, const char *name, int dtype, uint32_t rank,
                       const uint32_t *dims) {
  desc_t *d = (desc_t *)user;
  if (d->n_in >= 8) return;
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
  if (d->n_out >= 8) return;
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
int sv_engine_init(const char *model_dir, const char *lib_dir) {
  if (g_inited) return 0;
  char p[1024];

  snprintf(p, sizeof(p), "%s/sv_cmvn.bin", model_dir);
  if (sv_front_init(&g_front, p)) { set_err("load %s failed", p); return -1; }

  snprintf(p, sizeof(p), "%s/sv_prefix.bin", model_dir);
  FILE *fp = fopen(p, "rb");
  if (!fp) { set_err("open %s failed", p); return -1; }
  g_prefix = (float *)malloc(sizeof(float) * 4 * SV_DIM);
  if (!g_prefix || fread(g_prefix, sizeof(float), 4 * SV_DIM, fp) != 4 * SV_DIM) {
    fclose(fp);
    set_err("read %s failed", p);
    return -1;
  }
  fclose(fp);

  snprintf(p, sizeof(p), "%s/sv_tokens.bin", model_dir);
  fp = fopen(p, "rb");
  if (!fp) { set_err("open %s failed", p); return -1; }
  char magic[4];
  uint32_t n = 0;
  if (fread(magic, 1, 4, fp) != 4 || memcmp(magic, "SVTK", 4) || fread(&n, 4, 1, fp) != 1) {
    fclose(fp);
    set_err("%s is not a SVTK table", p);
    return -1;
  }
  g_ntok = n;
  g_off = (uint32_t *)malloc(sizeof(uint32_t) * ((size_t)n + 1));
  if (!g_off || fread(g_off, 4, (size_t)n + 1, fp) != (size_t)n + 1) {
    fclose(fp);
    set_err("read %s offsets failed", p);
    return -1;
  }
  size_t blob = g_off[n];
  g_blob = (char *)malloc(blob ? blob : 1);
  if (!g_blob || fread(g_blob, 1, blob, fp) != blob) {
    fclose(fp);
    set_err("read %s blob failed", p);
    return -1;
  }
  fclose(fp);

  g_feats = (float *)malloc(sizeof(float) * SV_T * SV_DIM);
  g_mask = (float *)malloc(sizeof(float) * SV_T);
  if (!g_feats || !g_mask) { set_err("oom"); return -1; }

  if (qnn_init(lib_dir, lib_dir)) { set_err("qnn_init: %s", qnn_last_error()); return -1; }
  snprintf(p, sizeof(p), "%s/sv.bin", model_dir);
  if (qnn_load(SV_TAG, p)) { set_err("load sv.bin: %s", qnn_last_error()); return -1; }

  desc_t d;
  memset(&d, 0, sizeof(d));
  if (qnn_describe(SV_TAG, desc_in_cb, desc_out_cb, &d)) {
    set_err("describe sv: %s", qnn_last_error());
    return -1;
  }
  g_n_in = d.n_in;
  g_n_out = d.n_out;
  int have_feats = 0, have_mask = 0;
  for (int i = 0; i < d.n_in; i++) {
    snprintf(g_in_name[i], sizeof(g_in_name[0]), "%s", d.names[i]);
    g_in_dtype[i] = d.dtype[i];
    g_in_elems[i] = d.elems[i];
    if (!strcmp(d.names[i], "feats")) { g_in_src[i] = g_feats; have_feats = 1; }
    else if (!strcmp(d.names[i], "mask")) { g_in_src[i] = g_mask; have_mask = 1; }
    else if (!strcmp(d.names[i], "prefix")) { g_in_src[i] = g_prefix; }
    else g_in_src[i] = NULL;          /* output_0 之类：补零即可 */
  }
  if (!have_feats || !have_mask) {
    char list[256] = "";
    for (int i = 0; i < d.n_in; i++) {
      size_t l = strlen(list);
      snprintf(list + l, sizeof(list) - l, "%s%s[%d,%zu]", i ? " " : "",
               d.names[i], d.dtype[i], d.elems[i]);
    }
    set_err("sv 图缺 feats/mask；实际: %s", list);
    return -1;
  }
  g_out_idx = -1;
  for (int i = 0; i < d.n_out; i++) {
    /* 新图 (T+4)×V，旧图 T×V —— 两种都认，解码时按总帧数区分 */
    if (d.oelems[i] == (size_t)(SV_T + SV_RICH) * SV_VOCAB ||
        d.oelems[i] == (size_t)SV_T * SV_VOCAB) {
      g_out_idx = i;
      g_out_dtype = d.odtype[i];
      g_out_elems = d.oelems[i];
    }
  }
  if (g_out_idx < 0) {
    set_err("sv 图找不到 %d/%d x %d 的 logits 输出", SV_T, SV_T + SV_RICH, SV_VOCAB);
    return -1;
  }

  g_inited = 1;
  return 0;
}

void sv_engine_shutdown(void) {
  if (!g_inited) return;
  qnn_free(SV_TAG);
  qnn_shutdown_if_empty();
  free(g_prefix); g_prefix = NULL;
  free(g_feats); g_feats = NULL;
  free(g_mask); g_mask = NULL;
  free(g_off); g_off = NULL;
  free(g_blob); g_blob = NULL;
  g_inited = 0;
}

/* 按图的 dtype 准备一块输入缓冲（f32 直用，f16 现场转）。 */
static void *prep_in(const float *src, size_t elems, int dtype) {
  if (dtype == RVC_QNN_FLOAT32) {
    void *p = malloc(sizeof(float) * elems);
    if (!p) return NULL;
    if (src) memcpy(p, src, sizeof(float) * elems);
    else memset(p, 0, sizeof(float) * elems);
    return p;
  }
  if (dtype == RVC_QNN_FLOAT16) {
    uint16_t *p = (uint16_t *)malloc(sizeof(uint16_t) * elems);
    if (!p) return NULL;
    for (size_t i = 0; i < elems; i++) p[i] = src ? f2h(src[i]) : 0;
    return p;
  }
  return NULL;
}

int sv_engine_run_wav(const char *wav_path, char *text, size_t cap, int *n_tok,
                      int *rich_out) {
  if (!g_inited) { set_err("not initialized"); return -1; }
  if (!text || cap == 0) { set_err("bad text buffer"); return -1; }

  float *pcm = NULL;
  size_t npcm = 0;
  if (read_wav(wav_path, &pcm, &npcm)) return -1;

  int n_valid = 0;
  int rc = sv_front_compute(&g_front, pcm, npcm, g_feats, g_mask, &n_valid);
  free(pcm);
  if (rc) { set_err("frontend failed (audio too short?)"); return -1; }

  QnnIo in[SV_MAX_IO];
  void *bufs[SV_MAX_IO];
  memset(in, 0, sizeof(in));
  memset(bufs, 0, sizeof(bufs));
  for (int i = 0; i < g_n_in; i++) {
    in[i].name = g_in_name[i];
    in[i].dtype = g_in_dtype[i];
    in[i].bytes = qnn_elem_size(g_in_dtype[i]) * g_in_elems[i];
    bufs[i] = prep_in(g_in_src[i], g_in_elems[i], g_in_dtype[i]);
    in[i].data = bufs[i];
    if (!bufs[i]) {
      for (int j = 0; j < i; j++) free(bufs[j]);
      set_err("oom input %d (%s)", i, g_in_name[i]);
      return -1;
    }
  }
  QnnIo out[SV_MAX_IO];
  memset(out, 0, sizeof(out));
  size_t n_out = 0;
  rc = qnn_exec_raw(SV_TAG, in, (size_t)g_n_in, out, (size_t)g_n_out, &n_out);
  for (int i = 0; i < g_n_in; i++) free(bufs[i]);
  if (rc) {
    desc_t d2;
    memset(&d2, 0, sizeof(d2));
    qnn_describe(SV_TAG, desc_in_cb, desc_out_cb, &d2);
    char list[512] = "";
    for (int i = 0; i < d2.n_in; i++) {
      size_t l = strlen(list);
      snprintf(list + l, sizeof(list) - l, "%s'%s'", i ? " " : "", d2.names[i]);
    }
    set_err("exec sv: %s", qnn_last_error());
    return -1;
  }
  if ((int)n_out <= g_out_idx || !out[g_out_idx].data) {
    for (size_t i = 0; i < n_out; i++) free(out[i].data);
    set_err("sv 无 logits 输出");
    return -1;
  }
  float *logits_raw = out[g_out_idx].data;

  /* 输出可能是 f32 或 f16 */
  const float *logits = NULL;
  float *conv = NULL;
  size_t nframes = g_out_elems / SV_VOCAB;
  if (g_out_dtype == RVC_QNN_FLOAT32) {
    logits = (const float *)logits_raw;
  } else if (g_out_dtype == RVC_QNN_FLOAT16) {
    conv = (float *)malloc(sizeof(float) * g_out_elems);
    if (!conv) { free(logits_raw); set_err("oom logits"); return -1; }
    const uint16_t *h = (const uint16_t *)logits_raw;
    for (size_t i = 0; i < g_out_elems; i++) conv[i] = h2f(h[i]);
    logits = conv;
  } else {
    free(logits_raw);
    set_err("sv 输出 dtype 不支持 (%d)", g_out_dtype);
    return -1;
  }

  /* 新版图输出 T+4 帧：前 4 帧是 rich 标签（语种/情感/事件/ITN），正文从第
   * 4 帧起；旧图只有 T 帧，没有标签（rich_out 保持 -1）。判断依据是图的总
   * 帧数是 SV_T+4 还是 SV_T —— 不能拿 n_valid 比，短音频时两者都大于它。 */
  const int rich = (nframes >= (size_t)SV_T + SV_RICH) ? SV_RICH : 0;
  size_t body_n = (size_t)n_valid;
  if (body_n + (size_t)rich > nframes) body_n = nframes - (size_t)rich;

  if (rich_out) {
    for (int k = 0; k < SV_RICH; k++) rich_out[k] = -1;
    for (int k = 0; k < rich; k++) {
      rich_out[k] = argmax_row(logits + (size_t)k * SV_VOCAB);
    }
  }

  /* CTC 贪心：argmax -> 折叠重复 -> 去 blank(0) */
  size_t used = 0;
  int prev = -1, nt = 0;
  for (size_t i = 0; i < body_n; i++) {
    const float *row = logits + ((size_t)rich + i) * SV_VOCAB;
    int best = argmax_row(row);
    if (best != prev && best != 0) {
      uint32_t a = g_off[best], b = g_off[best + 1];
      size_t len = b - a;
      if (used + len + 1 < cap) {
        memcpy(text + used, g_blob + a, len);
        used += len;
      }
      nt++;
    }
    prev = best;
  }
  text[used] = 0;

  /* "▁" (U+2581) -> 空格，再收尾 */
  {
    char *r = text, *w = text;
    while (*r) {
      if ((unsigned char)r[0] == 0xE2 && (unsigned char)r[1] == 0x96 &&
          (unsigned char)r[2] == 0x81) {
        *w++ = ' ';
        r += 3;
      } else {
        *w++ = *r++;
      }
    }
    *w = 0;
    while (w > text && (w[-1] == ' ' || w[-1] == '\n')) *--w = 0;
    char *s = text;
    while (*s == ' ') s++;
    if (s != text) memmove(text, s, strlen(s) + 1);
  }
  if (n_tok) *n_tok = nt;

  free(conv);
  free(logits_raw);
  return 0;
}
