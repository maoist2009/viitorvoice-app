/* aligner_engine.c — Qwen3-ForcedAligner-0.6B 的端侧推理管线（QNN HTP）。
 *
 * 与 ASR 不同：aligner 是**单次前向**，没有 KV cache、没有自回归。
 *   enc : mel(1,128,800)   -> audio_embeds(1,104,1024)
 *   dec : embeds(1,320,1024) -> logits(1,320,5000)
 * 时间戳 = 在 prompt 里 input_id == 151705 的那些位置上 logits 的 argmax × 80 ms。
 *
 * causal mask 和 RoPE 都是常量，已经烘进 dec 图里，运行时不需要喂。
 * 两张图总共约 1.5 GB，启动后一直常驻（不需要像 ASR 那样每句重载）。
 */
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "aligner_engine.h"
#include "asr_mel.h"
#include "qnn_executor.h"

#define T_SEQ        320     /* 静态序列长度，与 dec.onnx 一致 */
#define HID          1024
#define N_CLASS      5000    /* aligner 头 */
#define VOCAB_EMB    152064  /* embedding 表行数 */
#define N_AUDIO      104     /* mel 800 帧 -> 104 个音频 token */
#define AUDIO_PAD_ID 151676
#define TS_ID        151705
#define MS_PER_TOKEN 80
#define MAXT         8

static const char *TAGS[] = {"al_enc", "al_dec"};

static char   g_err[512];
static int    g_init = 0;
static int    g_n_graphs = 0;   /* 已经 load 成功的图数（0/1/2），shutdown 按这个释放 */
static char   g_dir[1024];
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static int      g_tbl_fd = -1;
static void    *g_tbl_map = NULL;
static size_t   g_tbl_bytes = 0;
static uint16_t *g_tbl = NULL;       /* (VOCAB_EMB, HID) fp16 */

static uint16_t *g_mel = NULL;       /* 128*800 */
static float    *g_mel_f32 = NULL;
static float    *g_pcm = NULL;
static uint16_t *g_audio = NULL;     /* 104*HID */
static uint16_t *g_embeds = NULL;    /* T_SEQ*HID */
static uint16_t *g_logits = NULL;    /* T_SEQ*N_CLASS */

static asr_mel_t g_mel_ctx;

/* 计时 */
#define NSTAT 8
static struct { char tag[24]; int n; int64_t sum, mn, mx; } g_t[NSTAT];
static char g_tbuf[512];

static void set_err(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(g_err, sizeof(g_err), fmt, ap);
  va_end(ap);
}

const char *aligner_error(void) { return g_err; }

static int64_t ns_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static void t_reset(void) { memset(g_t, 0, sizeof(g_t)); }

static void t_rec(const char *tag, int64_t ns) {
  int i;
  for (i = 0; i < NSTAT; i++)
    if (g_t[i].n && !strcmp(g_t[i].tag, tag)) break;
  if (i == NSTAT) {
    for (i = 0; i < NSTAT; i++)
      if (!g_t[i].n) { snprintf(g_t[i].tag, sizeof(g_t[i].tag), "%s", tag); break; }
    if (i == NSTAT) return;
  }
  if (!g_t[i].n) { g_t[i].mn = ns; g_t[i].mx = ns; }
  g_t[i].n++;
  g_t[i].sum += ns;
  if (ns < g_t[i].mn) g_t[i].mn = ns;
  if (ns > g_t[i].mx) g_t[i].mx = ns;
}

const char *aligner_timings(void) {
  size_t off = 0;
  off += snprintf(g_tbuf + off, sizeof(g_tbuf) - off, "[");
  for (int i = 0; i < NSTAT && off < sizeof(g_tbuf) - 128; i++) {
    if (!g_t[i].n) continue;
    off += snprintf(g_tbuf + off, sizeof(g_tbuf) - off,
                    "%s{\"tag\":\"%s\",\"n\":%d,\"mean_ms\":%.2f,\"min_ms\":%.2f,"
                    "\"max_ms\":%.2f,\"total_ms\":%.0f}",
                    off > 1 ? "," : "", g_t[i].tag, g_t[i].n,
                    g_t[i].sum / 1e6 / g_t[i].n, g_t[i].mn / 1e6, g_t[i].mx / 1e6,
                    g_t[i].sum / 1e6);
  }
  snprintf(g_tbuf + off, sizeof(g_tbuf) - off, "]");
  return g_tbuf;
}

static float h2f(uint16_t h) {          /* 仅供调试输出 */
  uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  float v;
  if (e == 0) v = m / 1024.0f * powf(2.0f, -14);
  else if (e == 31) v = m ? NAN : INFINITY;
  else v = (1.0f + m / 1024.0f) * powf(2.0f, (int)e - 15);
  return s ? -v : v;
}

static uint16_t f2h(float x) {
  union { float f; uint32_t u; } c = {x};
  uint32_t sign = (c.u >> 16) & 0x8000;
  int32_t  exp  = (int32_t)((c.u >> 23) & 0xff) - 127 + 15;
  uint32_t mant = c.u & 0x7fffff;
  if (((c.u >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (mant ? 0x200 : 0));
  if (exp >= 31) return (uint16_t)(sign | 0x7c00);
  if (exp <= 0) {
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000;
    int sh = 14 - exp;
    uint32_t r = mant >> sh;
    if ((mant >> (sh - 1)) & 1) r++;
    return (uint16_t)(sign | r);
  }
  uint32_t r = ((exp << 10) | (mant >> 13)) & 0x7fff;
  if ((mant & 0x1000) && ((mant & 0x1fff) || (r & 1))) r++;
  return (uint16_t)(sign | r);
}

/* ---------- QNN ---------- */
static int run_graph(const char *tag, const QnnIo *in, size_t n_in,
                     QnnIo *out, size_t n_out_max) {
  size_t got = 0;
  if (qnn_exec_raw(tag, in, n_in, out, n_out_max, &got)) {
    set_err("%s: %s", tag, qnn_last_error());
    return -1;
  }
  return 0;
}

static int load_graph(const char *tag, const char *file) {
  char path[1200];
  snprintf(path, sizeof(path), "%s/%s", g_dir, file);
  int64_t t0 = ns_now();
  if (qnn_load(tag, path)) { set_err("load %s: %s", path, qnn_last_error()); return -1; }
  char tt[32];
  snprintf(tt, sizeof(tt), "ld:%s", tag);
  t_rec(tt, ns_now() - t0);
  return 0;
}

static int open_table(const char *dir) {
  char path[1200];
  snprintf(path, sizeof(path), "%s/token_embeds.bin", dir);
  int fd = open(path, O_RDONLY);
  if (fd < 0) { set_err("open %s failed", path); return -1; }
  struct stat st;
  if (fstat(fd, &st)) { close(fd); set_err("fstat %s failed", path); return -1; }
  size_t want = (size_t)VOCAB_EMB * HID * 2;
  if ((size_t)st.st_size < want) {
    close(fd);
    set_err("%s is %lld bytes, want %zu", path, (long long)st.st_size, want);
    return -1;
  }
  void *p = mmap(NULL, want, PROT_READ, MAP_PRIVATE, fd, 0);
  if (p == MAP_FAILED) { close(fd); set_err("mmap %s failed", path); return -1; }
  g_tbl_fd = fd; g_tbl_map = p; g_tbl_bytes = want; g_tbl = (uint16_t *)p;
  return 0;
}

int aligner_init(const char *model_dir, const char *lib_dir) {
  pthread_mutex_lock(&g_lock);
  if (g_init) { pthread_mutex_unlock(&g_lock); return 0; }
  g_err[0] = '\0';
  snprintf(g_dir, sizeof(g_dir), "%s", model_dir);

  if (asr_mel_init(&g_mel_ctx)) { set_err("asr_mel_init failed"); goto fail; }
  g_mel     = (uint16_t *)malloc(sizeof(uint16_t) * ASR_MEL_N_MELS * ASR_MEL_FRAMES);
  g_mel_f32 = (float *)malloc(sizeof(float) * ASR_MEL_N_MELS * ASR_MEL_FRAMES);
  g_pcm     = (float *)malloc(sizeof(float) * ASR_MEL_SAMPLES);
  g_audio   = (uint16_t *)malloc(sizeof(uint16_t) * N_AUDIO * HID);
  g_embeds  = (uint16_t *)malloc(sizeof(uint16_t) * T_SEQ * HID);
  g_logits  = (uint16_t *)malloc(sizeof(uint16_t) * T_SEQ * N_CLASS);
  if (!g_mel || !g_mel_f32 || !g_pcm || !g_audio || !g_embeds || !g_logits) {
    set_err("out of memory");
    goto fail;
  }
  memset(g_embeds, 0, sizeof(uint16_t) * T_SEQ * HID);

  if (open_table(model_dir)) goto fail;
  if (qnn_init(lib_dir, lib_dir)) { set_err("qnn_init: %s", qnn_last_error()); goto fail; }
  if (load_graph(TAGS[0], "enc.bin")) goto fail;
  g_n_graphs = 1;
  if (load_graph(TAGS[1], "dec.bin")) goto fail;
  g_n_graphs = 2;
  t_reset();
  g_init = 1;
  pthread_mutex_unlock(&g_lock);
  return 0;

fail:
  for (int i = 0; i < g_n_graphs; i++) qnn_free(TAGS[i]);
  g_n_graphs = 0;
  qnn_shutdown_if_empty();
  pthread_mutex_unlock(&g_lock);
  return -1;
}

void aligner_shutdown(void) {
  pthread_mutex_lock(&g_lock);
  if (!g_init) { pthread_mutex_unlock(&g_lock); return; }
  for (int i = 0; i < g_n_graphs; i++) qnn_free(TAGS[i]);
  g_n_graphs = 0;
  qnn_shutdown_if_empty();
  if (g_tbl_map && g_tbl_map != MAP_FAILED) munmap(g_tbl_map, g_tbl_bytes);
  if (g_tbl_fd >= 0) close(g_tbl_fd);
  g_tbl_map = NULL; g_tbl = NULL; g_tbl_fd = -1;
  free(g_mel); free(g_mel_f32); free(g_pcm); free(g_audio);
  free(g_embeds); free(g_logits);
  g_mel = NULL; g_mel_f32 = NULL; g_pcm = NULL; g_audio = NULL;
  g_embeds = NULL; g_logits = NULL;
  asr_mel_free(&g_mel_ctx);
  g_init = 0;
  pthread_mutex_unlock(&g_lock);
}

static int read_wav(const char *path, float **out, size_t *n_out) {
  FILE *f = fopen(path, "rb");
  if (!f) { set_err("open %s failed", path); return -1; }
  unsigned char hdr[12];
  if (fread(hdr, 1, 12, f) != 12 || memcmp(hdr, "RIFF", 4) || memcmp(hdr + 8, "WAVE", 4)) {
    fclose(f);
    set_err("%s is not a RIFF/WAVE file", path);
    return -1;
  }
  uint16_t fmt = 0, ch = 0, bits = 0;
  uint32_t rate = 0;
  size_t n = 0;
  float *pcm = NULL;
  while (1) {
    unsigned char ck[8];
    if (fread(ck, 1, 8, f) != 8) break;
    uint32_t sz = ck[4] | (ck[5] << 8) | (ck[6] << 16) | ((uint32_t)ck[7] << 24);
    if (!memcmp(ck, "fmt ", 4)) {
      unsigned char b[16];
      if (sz < 16 || fread(b, 1, 16, f) != 16) break;
      fmt = b[0]; ch = b[2]; rate = b[4] | (b[5] << 8) | (b[6] << 16) | ((uint32_t)b[7] << 24);
      bits = b[14];
      if (sz > 16) fseek(f, (long)(sz - 16), SEEK_CUR);
    } else if (!memcmp(ck, "data", 4)) {
      int bytes = bits / 8;
      size_t cnt = sz / (bytes * ch);
      unsigned char *raw = (unsigned char *)malloc(sz);
      if (!raw) break;
      size_t rd = fread(raw, 1, sz, f);
      cnt = rd / (bytes * ch);
      pcm = (float *)malloc(sizeof(float) * cnt);
      if (!pcm) { free(raw); break; }
      for (size_t i = 0; i < cnt; i++) {
        float v = 0.0f;
        const unsigned char *p = raw + i * bytes * ch;
        if (fmt == 1 && bits == 16) v = (float)((int16_t)(p[0] | (p[1] << 8))) / 32768.0f;
        else if (fmt == 1 && bits == 32) {
          int32_t s = (int32_t)(p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24));
          v = (float)(s / 2147483648.0);
        } else if (fmt == 3 && bits == 32) {
          float fv;
          memcpy(&fv, p, 4);
          v = fv;
        }
        pcm[i] = v;
      }
      free(raw);
      n = cnt;
      break;
    } else {
      fseek(f, (long)sz + (sz & 1), SEEK_CUR);
    }
  }
  fclose(f);
  if (!pcm) { set_err("%s: no data chunk", path); return -1; }
  *out = pcm;
  *n_out = n;
  (void)rate;
  return 0;
}

int aligner_run_wav(const char *wav_path, const int *ids, int n_ids,
                    float *out_ms, int cap, int *n_out) {
  pthread_mutex_lock(&g_lock);
  if (!g_init) { set_err("aligner_init not called"); pthread_mutex_unlock(&g_lock); return -1; }
  t_reset();
  int rc = -1;
  float *pcm = NULL;
  size_t npcm = 0;

  if (read_wav(wav_path, &pcm, &npcm)) goto done;

  /* 1) mel */
  int64_t t0 = ns_now();
  if (asr_mel_compute(&g_mel_ctx, pcm, npcm, g_mel_f32)) {
    set_err("mel failed"); goto done;
  }
  for (size_t i = 0; i < (size_t)ASR_MEL_N_MELS * ASR_MEL_FRAMES; i++)
    g_mel[i] = f2h(g_mel_f32[i]);
  t_rec("mel", ns_now() - t0);

  /* 2) encoder */
  {
    QnnIo in[1] = {{"mel", RVC_QNN_FLOAT16, g_mel,
                    sizeof(uint16_t) * ASR_MEL_N_MELS * ASR_MEL_FRAMES}};
    QnnIo out[1] = {{"audio_embeds", RVC_QNN_FLOAT16, g_audio,
                     sizeof(uint16_t) * N_AUDIO * HID}};
    t0 = ns_now();
    if (run_graph(TAGS[0], in, 1, out, 1)) goto done;
    t_rec("enc", ns_now() - t0);
  }

  /* 3) 拼 prompt：audio_pad 展开成 104 行，其余查 embedding 表 */
  int row = 0, nts = 0;
  int ts_pos[256];
  for (int i = 0; i < n_ids; i++) {
    if (ids[i] == AUDIO_PAD_ID) {
      if (row + N_AUDIO > T_SEQ) { set_err("prompt too long (%d + audio)", row); goto done; }
      memcpy(g_embeds + (size_t)row * HID, g_audio, sizeof(uint16_t) * N_AUDIO * HID);
      row += N_AUDIO;
    } else {
      if (ids[i] < 0 || ids[i] >= VOCAB_EMB) { set_err("token id %d out of range", ids[i]); goto done; }
      if (row >= T_SEQ) { set_err("prompt too long (%d tokens)", row); goto done; }
      memcpy(g_embeds + (size_t)row * HID, g_tbl + (size_t)ids[i] * HID,
             sizeof(uint16_t) * HID);
      if (ids[i] == TS_ID && nts < 256) ts_pos[nts++] = row;
      row++;
    }
  }
  /* ⚠️ 填充行不能用全 0：RMSNorm 对 0 行算出的 var=0，fp16 下 1/sqrt(var+eps)
   * 会溢出成 inf，0*inf=NaN；attention 里被 mask 掉的位置权重虽≈0，但 0*NaN 仍是
   * NaN，于是 NaN 扩散到**所有行**，argmax 全部退回 0（26 个时间戳只剩 3 个对）。
   * 这里复制最后一行有效 embeds 当填充：causal mask 保证它不影响前面的行。 */
  if (row > 0) {
    for (int r = row; r < T_SEQ; r++)
      memcpy(g_embeds + (size_t)r * HID, g_embeds + (size_t)(row - 1) * HID,
             sizeof(uint16_t) * HID);
  }

  /* 4) decoder */
  {
    QnnIo in[1] = {{"embeds", RVC_QNN_FLOAT16, g_embeds, sizeof(uint16_t) * T_SEQ * HID}};
    QnnIo out[1] = {{"logits", RVC_QNN_FLOAT16, g_logits, sizeof(uint16_t) * T_SEQ * N_CLASS}};
    t0 = ns_now();
    if (run_graph(TAGS[1], in, 1, out, 1)) goto done;
    t_rec("dec", ns_now() - t0);
  }

  /* 5) 每个 timestamp 位置取 argmax */
  t0 = ns_now();
  if (nts > cap) nts = cap;
  for (int k = 0; k < nts; k++) {
    const uint16_t *p = g_logits + (size_t)ts_pos[k] * N_CLASS;
    int best = 0;
    float bv = h2f(p[0]);
    for (int c = 1; c < N_CLASS; c++) {
      float v = h2f(p[c]);
      if (v > bv) { bv = v; best = c; }
    }
    out_ms[k] = (float)best * MS_PER_TOKEN;
  }
  *n_out = nts;
  t_rec("argmax", ns_now() - t0);
  rc = 0;

done:
  free(pcm);
  pthread_mutex_unlock(&g_lock);
  return rc;
}

/* 诊断用：只加载 dec 一张图，读 in_path（T_SEQ*HID 个 fp16）跑一次，写 out_path
 * （T_SEQ*N_CLASS 个 fp16）。用来判断"设备侧 dec 结果和云端不一致"到底是
 * 二进制/执行环境的问题，还是 embeds 拼装的问题。 */
int aligner_dec_dbg(const char *model_dir, const char *lib_dir,
                    const char *in_path, const char *out_path) {
  pthread_mutex_lock(&g_lock);
  g_err[0] = '\0';
  int rc = -1;
  size_t want_in = sizeof(uint16_t) * T_SEQ * HID;
  size_t want_out = sizeof(uint16_t) * T_SEQ * N_CLASS;
  char path[1200];
  if (!g_embeds) g_embeds = (uint16_t *)malloc(want_in);
  if (!g_logits) g_logits = (uint16_t *)malloc(want_out);
  if (!g_embeds || !g_logits) { set_err("out of memory"); goto done; }

  FILE *f = fopen(in_path, "rb");
  if (!f) { set_err("open %s failed", in_path); goto done; }
  size_t rd = fread(g_embeds, 1, want_in, f);
  fclose(f);
  if (rd != want_in) { set_err("%s: read %zu/%zu", in_path, rd, want_in); goto done; }

  if (qnn_init(lib_dir, lib_dir)) { set_err("qnn_init: %s", qnn_last_error()); goto done; }
  snprintf(path, sizeof(path), "%s/dec.bin", model_dir);
  if (qnn_load("al_dec", path)) { set_err("load %s: %s", path, qnn_last_error()); goto done; }

  {
    QnnIo in[1] = {{"embeds", RVC_QNN_FLOAT16, g_embeds, want_in}};
    QnnIo out[1] = {{"logits", RVC_QNN_FLOAT16, g_logits, want_out}};
    if (run_graph("al_dec", in, 1, out, 1)) goto done;
  }

  f = fopen(out_path, "wb");
  if (!f) { set_err("create %s failed", out_path); goto done; }
  size_t wr = fwrite(g_logits, 1, want_out, f);
  fclose(f);
  if (wr != want_out) { set_err("%s: wrote %zu/%zu", out_path, wr, want_out); goto done; }
  rc = 0;
done:
  pthread_mutex_unlock(&g_lock);
  return rc;
}
