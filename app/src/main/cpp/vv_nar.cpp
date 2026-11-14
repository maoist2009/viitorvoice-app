/* vv_nar.cpp — masked-discrete-diffusion loop for ViiTorVoice on the HTP.
 *
 * Mirrors viitorvoice/llm/generation.py (_generate_iterative with the default
 * LLMGenerationConfig: 32 steps, t_shift 0.1, layer penalty 5.0, position
 * temperature 5.0; no CFG) with two compiled graphs standing in for the torch
 * backbone + output heads.
 *
 * Local edit follows _generate_edit_iterative_single: a conditional branch
 * (prompt + target window) and an unconditional branch (window only) are run
 * every step, the log-probabilities are combined with the CFG scale, and only
 * the masked, editable cells are allowed to update.
 *
 * The prompt layout, also from generation.py:
 *
 *   [style tokens][text tokens][target frames]   all 12 codebook rows carry
 *   the same text ids; audio positions use the per-codebook mask ids.
 *
 * Embeddings are gathered on the host from fp16 tables (see
 * extract_embeddings.py): text rows for text positions, sem + sum(aco) for
 * audio positions.
 */
#include "vv_nar.h"

#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <string>

#include <android/log.h>

#include "qnn_executor.h"
#include "vv_pipeline.h"

#define TAG "vv_nar"

/* ---- host-side timing (debug harness only) -------------------------------
 * Everything the HTP does goes through run_graphs(); the rest of a step is
 * host work. These counters answer "how much of the step is actually NPU". */
static double nar_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}
static double T_GATHER, T_LLM, T_HEAD, T_MASK, T_ACC, T_SCORE, T_SELECT;

static const char *nar_timing(void) {
  static char buf[256];
  snprintf(buf, sizeof(buf),
           "gather %.0f | llm %.0f | head %.0f | mask %.0f | acc %.0f | score %.0f | select %.0f ms",
           T_GATHER, T_LLM, T_HEAD, T_MASK, T_ACC, T_SCORE, T_SELECT);
  return buf;
}

#define SEQ_MAX    512
static int SEQ = 256;   /* active window: 256, or 512 when the linked .bin has it */
#define NCB        12
#define DIM        1024
#define SEM_VOCAB  16387
#define ACO_VOCAB  1027
#define N_ACO      11
#define T_SHIFT        0.1f
#define LAYER_PENALTY  5.0f
#define POS_TEMP       5.0f
#define N_STEPS_MAX    32          /* arrays are sized for the official 32 */
static int      N_STEPS = N_STEPS_MAX;   /* runtime: models/nar_steps (1..32) */
#define FRAME_SAMPLES  960          /* 40 ms at 24 kHz */

static const int32_t MASK_IDS[NCB] = {16386, 1026, 1026, 1026, 1026, 1026,
                                      1026, 1026, 1026, 1026, 1026, 1026};
static const int32_t PAUSE_IDS[NCB] = {16384, 1024, 1024, 1024, 1024, 1024,
                                       1024, 1024, 1024, 1024, 1024, 1024};
static const int32_t SEP_IDS[NCB] = {16385, 1025, 1025, 1025, 1025, 1025,
                                     1025, 1025, 1025, 1025, 1025, 1025};

static pthread_mutex_t LOCK = PTHREAD_MUTEX_INITIALIZER;
static int  READY = 0;
static int  LLM_MULTI = 0;   /* llm.bin holds llm_256 + llm_512 */
static int  HEAD_MULTI = 0;  /* head.bin holds head_256 + head_512 */
static char GRAPH_SUMMARY[320] = {0};
static char NAME_LLM[8][64], NAME_HEAD[8][64];
static int  N_LLM = 0, N_HEAD = 0;
/* resolved graph tags per window: [0] = 256, [1] = 512; empty = not present */
static char TAG_LLM[2][64], TAG_HEAD[2][64];
static char ERR[1024];

static uint16_t *TEXT_EMB = nullptr;
static uint16_t *SEM_EMB = nullptr;
static uint16_t *ACO_EMB = nullptr;
static size_t    TEXT_ROWS = 52697;
static const int32_t ACO_OFFS[N_ACO] = {0, 1027, 2054, 3081, 4108, 5135,
                                        6162, 7189, 8216, 9243, 10270};

static int32_t *IDS = nullptr;        /* [NCB][SEQ] */
static int32_t *TOKENS = nullptr;     /* [NCB][400]  synthesis */
static float   *EMB = nullptr;        /* [SEQ][DIM] */
static float   *ATTN = nullptr;       /* [SEQ][SEQ]  conditional */
static int      COND_LEN = 0;
static int      FRAMES = 0;
static int      TOTAL_CELLS = 0;      /* initial number of masked cells */
static int      REMAINING = 0;
static uint64_t RNG = 0x9E3779B97F4A7C15ull;
static float    SCHED_T[N_STEPS_MAX + 1];

/* CFG job: clone / edit / guided synthesis share one loop */
#define MAX_BR 4
static int32_t *CFG_TOK = nullptr;    /* [NCB][400] */
static uint8_t *CFG_MASK = nullptr;   /* [400] 1 = editable */
static int      EDIT_MODE = 0;
static float    CFG = 2.0f;
static float   *ACC_SEM = nullptr;    /* SEQ * SEM_VOCAB */
static float   *ACC_ACO = nullptr;    /* SEQ * N_ACO * ACO_VOCAB */

static int32_t *BR_IDS[MAX_BR];       /* [NCB][SEQ] per branch */
static float   *BR_ATTN[MAX_BR];      /* [SEQ][SEQ] per branch */
static float    BR_W[MAX_BR];         /* signed contribution */
static int      BR_TLEN[MAX_BR];      /* text prompt length */
static int      BR_TSTART[MAX_BR];    /* first target frame */
static int      BR_TOTAL[MAX_BR];     /* last used position */
static int      N_BR = 0;

/* debug hooks (only used by the headless harness) */
static char  DIAG_PATH[512] = {0};
static float W_OVR_C = -1.0f, W_OVR_U = 0.0f;
static float DIAG_LP[MAX_BR][VV_NAR_MAX_FRAMES][2];
static int32_t DIAG_TOK[MAX_BR][VV_NAR_MAX_FRAMES];

static void set_err(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(ERR, sizeof(ERR), fmt, ap);
  va_end(ap);
  __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", ERR);
}

const char *vv_nar_error(void) { return ERR; }
int vv_nar_ready(void) { return READY; }
const char *vv_nar_timing(void) { return nar_timing(); }

/* Pick the prompt window: 256 (10.2 s) or 512 (20.5 s). The 512 graphs only
 * exist in a linked .bin, so the call fails on an old single-window install. */
int vv_nar_set_window(int seq) {
  if (seq != 256 && seq != 512) {
    set_err("bad window %d (256 or 512)", seq);
    return -1;
  }
  if (seq == 512 && !(TAG_LLM[1][0] && TAG_HEAD[1][0])) {
    set_err("512 window not in these models (llm.bin/head.bin are single-graph)");
    return -1;
  }
  if (SEQ != seq) __android_log_print(ANDROID_LOG_INFO, TAG, "window %d -> %d", SEQ, seq);
  SEQ = seq;
  return 0;
}

int vv_nar_window(void) { return SEQ; }
const char *vv_nar_graphs(void) {
  static char buf[512];
  int n = snprintf(buf, sizeof(buf), "%s", GRAPH_SUMMARY);
  for (int i = 0; i < N_LLM && n > 0 && n < (int)sizeof(buf); i++)
    n += snprintf(buf + n, sizeof(buf) - n, " llm:%s", NAME_LLM[i]);
  for (int i = 0; i < N_HEAD && n > 0 && n < (int)sizeof(buf); i++)
    n += snprintf(buf + n, sizeof(buf) - n, " head:%s", NAME_HEAD[i]);
  return buf;
}
int vv_nar_max_window(void) { return (TAG_LLM[1][0] && TAG_HEAD[1][0]) ? 512 : 256; }
const int32_t *vv_nar_tokens(void) { return TOKENS; }
int vv_nar_remaining(void) { return REMAINING; }
const int32_t *vv_nar_edit_tokens(void) { return EDIT_MODE ? CFG_TOK : TOKENS; }
int vv_nar_edit_frames(void) { return FRAMES; }
int vv_nar_edit_remaining(void) { return REMAINING; }
const int32_t *vv_nar_cfg_tokens(void) { return CFG_TOK; }
int vv_nar_cfg_frames(void) { return FRAMES; }
int vv_nar_cfg_remaining(void) { return REMAINING; }
int vv_nar_cfg_prompt_len(void) { return N_BR > 0 ? BR_TSTART[0] : 0; }

/* --------------------------------------------------------------- fp16 tables */
static uint16_t *map_table(const char *dir, const char *name, size_t rows) {
  char path[512];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  int fd = open(path, O_RDONLY);
  if (fd < 0) { set_err("cannot open %s", path); return nullptr; }
  struct stat st;
  if (fstat(fd, &st) != 0) { close(fd); set_err("stat %s", path); return nullptr; }
  const size_t want = rows * DIM * sizeof(uint16_t);
  if ((size_t)st.st_size != want) {
    close(fd);
    set_err("%s is %lld B, expected %zu", path, (long long)st.st_size, want);
    return nullptr;
  }
  void *p = mmap(nullptr, want, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (p == MAP_FAILED) { set_err("mmap %s", path); return nullptr; }
  return (uint16_t *)p;
}

static inline float half2float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t man = h & 0x3FF;
  uint32_t out;
  if (exp == 0) {
    if (man == 0) {
      out = sign;
    } else {
      exp = 127 - 15 + 1;
      while ((man & 0x400) == 0) { man <<= 1; exp--; }
      man &= 0x3FF;
      out = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 31) {
    out = sign | 0x7F800000u | (man << 13);
  } else {
    out = sign | ((exp + 127 - 15) << 23) | (man << 13);
  }
  float f;
  memcpy(&f, &out, 4);
  return f;
}

/* ------------------------------------------------------------------- set-up */
int vv_nar_init(const char *models_dir) {
  pthread_mutex_lock(&LOCK);
  std::string dir(models_dir);
  int rc = -1;

  /* diffusion step count (official 32). Fewer steps cut the host+NPU loop
   * proportionally; models/nar_steps overrides it for A/B runs. */
  N_STEPS = N_STEPS_MAX;
  {
    FILE *f = fopen((dir + "/nar_steps").c_str(), "r");
    if (f) {
      int v = 0;
      if (fscanf(f, "%d", &v) == 1 && v >= 1 && v <= N_STEPS_MAX) N_STEPS = v;
      fclose(f);
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "nar steps %d", N_STEPS);
  }

  /* linked context binaries carry both windows; a single-graph .bin from an
   * older install still loads under the plain "llm" / "head" tags */
  char names[8][64];
  int n_llm = qnn_load_multi((dir + "/llm.bin").c_str(), &names[0][0], 64, 8);
  char multi_err[512];
  snprintf(multi_err, sizeof(multi_err), "%s", qnn_last_error());
  if (n_llm <= 1) {
    /* a single-graph bin is an older install: re-register it as "llm" */
    if (n_llm == 1) qnn_free(names[0]);
    if (qnn_load("llm", (dir + "/llm.bin").c_str()) != 0) {
      set_err("llm.bin: %s [multi: %s]", qnn_last_error(), multi_err);
      pthread_mutex_unlock(&LOCK);
      return rc;
    }
    LLM_MULTI = 0;
  } else {
    LLM_MULTI = 1;
    N_LLM = n_llm;
    memcpy(NAME_LLM, names, sizeof(NAME_LLM));
    for (int i = 0; i < n_llm; i++) {
      int s = qnn_input_dim(names[i], "inputs_embeds", 1);
      if (s == 256) snprintf(TAG_LLM[0], 64, "%s", names[i]);
      else if (s == 512) snprintf(TAG_LLM[1], 64, "%s", names[i]);
    }
    for (int i = 0; i < n_llm; i++)
      __android_log_print(ANDROID_LOG_INFO, TAG, "llm graph %d: %s", i, names[i]);
  }
  int n_head = qnn_load_multi((dir + "/head.bin").c_str(), &names[0][0], 64, 8);
  snprintf(multi_err, sizeof(multi_err), "%s", qnn_last_error());
  if (n_head <= 1) {
    if (n_head == 1) qnn_free(names[0]);
    if (qnn_load("head", (dir + "/head.bin").c_str()) != 0) {
      set_err("head.bin: %s [multi: %s]", qnn_last_error(), multi_err);
      pthread_mutex_unlock(&LOCK);
      return rc;
    }
    HEAD_MULTI = 0;
  } else {
    HEAD_MULTI = 1;
    N_HEAD = n_head;
    memcpy(NAME_HEAD, names, sizeof(NAME_HEAD));
    for (int i = 0; i < n_head; i++) {
      int s = qnn_input_dim(names[i], "hidden", 1);
      if (s == 256) snprintf(TAG_HEAD[0], 64, "%s", names[i]);
      else if (s == 512) snprintf(TAG_HEAD[1], 64, "%s", names[i]);
    }
    for (int i = 0; i < n_head; i++)
      __android_log_print(ANDROID_LOG_INFO, TAG, "head graph %d: %s", i, names[i]);
  }
  snprintf(GRAPH_SUMMARY, sizeof(GRAPH_SUMMARY), "[llm %d/%s] [head %d/%s]", n_llm,
           n_llm > 1 ? "multi" : "single", n_head,
           n_head > 1 ? "multi" : "single");
  {
    TEXT_EMB = map_table(models_dir, "text_emb.bin", TEXT_ROWS);
    SEM_EMB = map_table(models_dir, "sem_emb.bin", SEM_VOCAB);
    ACO_EMB = map_table(models_dir, "aco_emb.bin", (size_t)N_ACO * ACO_VOCAB);
    if (TEXT_EMB && SEM_EMB && ACO_EMB) {
      IDS = (int32_t *)malloc(sizeof(int32_t) * NCB * SEQ_MAX);
      TOKENS = (int32_t *)malloc(sizeof(int32_t) * NCB * VV_NAR_MAX_FRAMES);
      CFG_TOK = (int32_t *)malloc(sizeof(int32_t) * NCB * VV_NAR_MAX_FRAMES);
      CFG_MASK = (uint8_t *)malloc(VV_NAR_MAX_FRAMES);
      EMB = (float *)malloc(sizeof(float) * SEQ_MAX * DIM);
      ATTN = (float *)malloc(sizeof(float) * SEQ_MAX * SEQ_MAX);
      ACC_SEM = (float *)malloc(sizeof(float) * SEQ_MAX * SEM_VOCAB);
      ACC_ACO = (float *)malloc(sizeof(float) * SEQ_MAX * N_ACO * ACO_VOCAB);
      for (int b = 0; b < MAX_BR; b++) {
        BR_IDS[b] = (int32_t *)malloc(sizeof(int32_t) * NCB * SEQ_MAX);
        BR_ATTN[b] = (float *)malloc(sizeof(float) * SEQ_MAX * SEQ_MAX);
      }
      int ok = IDS && TOKENS && CFG_TOK && CFG_MASK && EMB && ATTN && ACC_SEM && ACC_ACO;
      for (int b = 0; b < MAX_BR; b++) ok = ok && BR_IDS[b] && BR_ATTN[b];
      if (ok) {
        READY = 1;
        rc = 0;
      } else {
        set_err("out of memory");
      }
    }
  }
  pthread_mutex_unlock(&LOCK);
  return rc;
}

void vv_nar_release(void) {
  if (TEXT_EMB) { munmap(TEXT_EMB, TEXT_ROWS * DIM * 2); TEXT_EMB = nullptr; }
  if (SEM_EMB) { munmap(SEM_EMB, (size_t)SEM_VOCAB * DIM * 2); SEM_EMB = nullptr; }
  if (ACO_EMB) { munmap(ACO_EMB, (size_t)N_ACO * ACO_VOCAB * DIM * 2); ACO_EMB = nullptr; }
  free(IDS); IDS = nullptr;
  free(TOKENS); TOKENS = nullptr;
  free(CFG_TOK); CFG_TOK = nullptr;
  free(CFG_MASK); CFG_MASK = nullptr;
  free(EMB); EMB = nullptr;
  free(ATTN); ATTN = nullptr;
  free(ACC_SEM); ACC_SEM = nullptr;
  free(ACC_ACO); ACC_ACO = nullptr;
  for (int b = 0; b < MAX_BR; b++) {
    free(BR_IDS[b]); BR_IDS[b] = nullptr;
    free(BR_ATTN[b]); BR_ATTN[b] = nullptr;
  }
  N_BR = 0;
  READY = 0;
}

/* ------------------------------------------------------------------ prepare */
static void fill_attention(float *attn, int total) {
  memset(attn, 0, sizeof(float) * SEQ * SEQ);
  for (int i = 0; i < total; i++)
    for (int j = 0; j < total; j++) attn[(size_t)i * SEQ + j] = 1.0f;
  for (int i = total; i < SEQ; i++) attn[(size_t)i * SEQ + i] = 1.0f;
}

int vv_nar_prepare(const int32_t *cond_ids, int cond_len, int frames) {
  if (!READY) { set_err("nar not initialised"); return -1; }
  if (frames <= 0 || frames > VV_NAR_MAX_FRAMES) { set_err("bad frame count %d", frames); return -1; }
  if (cond_len <= 0 || cond_len + frames > SEQ) {
    set_err("prompt %d + %d frames exceeds %d tokens", cond_len, frames, SEQ);
    return -1;
  }
  COND_LEN = cond_len;
  FRAMES = frames;
  EDIT_MODE = 0;
  const int total = cond_len + frames;

  for (int c = 0; c < NCB; c++) {
    int32_t *row = IDS + (size_t)c * SEQ;
    for (int i = 0; i < SEQ; i++) row[i] = MASK_IDS[c];
    for (int i = 0; i < cond_len; i++) row[i] = cond_ids[i];
    int32_t *t = TOKENS + (size_t)c * frames;
    for (int i = 0; i < frames; i++) t[i] = MASK_IDS[c];
  }

  fill_attention(ATTN, total);
  TOTAL_CELLS = NCB * frames;
  REMAINING = NCB * frames;
  __android_log_print(ANDROID_LOG_INFO, TAG, "prepare cond %d + %d frames", cond_len, frames);
  return 0;
}

/* ------------------------------------------------------------ CFG branches */
/* Builds one branch: text ids at the head, then the reference audio prompt,
 * the separator frame, and the (partly masked) target window. */
static void build_branch(int b, const int32_t *ids, int text_len,
                         const int32_t *ref, int ref_frames,
                         const int32_t *target, int frames, int uncond) {
  const int tstart = uncond ? 0 : text_len + ref_frames + (ref_frames > 0 ? 1 : 0);
  const int total = tstart + frames;
  for (int c = 0; c < NCB; c++) {
    int32_t *row = BR_IDS[b] + (size_t)c * SEQ;
    for (int i = 0; i < SEQ; i++) row[i] = MASK_IDS[c];
    if (!uncond && text_len > 0) memcpy(row, ids, sizeof(int32_t) * text_len);
    for (int i = 0; i < ref_frames; i++)
      row[text_len + i] = ref[(size_t)c * ref_frames + i];
    if (ref_frames > 0) row[text_len + ref_frames] = SEP_IDS[c];
    for (int i = 0; i < frames; i++)
      row[tstart + i] = target ? target[(size_t)c * frames + i] : MASK_IDS[c];
  }
  BR_TLEN[b] = uncond ? 0 : text_len;
  BR_TSTART[b] = tstart;
  BR_TOTAL[b] = total;

  float *attn = BR_ATTN[b];
  memset(attn, 0, sizeof(float) * SEQ * SEQ);
  for (int i = 0; i < total; i++)
    for (int j = 0; j < total; j++) attn[(size_t)i * SEQ + j] = 1.0f;
  for (int i = total; i < SEQ; i++) attn[(size_t)i * SEQ + i] = 1.0f;
}

int vv_nar_prepare_cfg(const int32_t *cond_ids, int cond_len,
                       const int32_t *const *aux_ids, const int *aux_lens,
                       const float *aux_scales, int n_aux,
                       const int32_t *ref_tokens, int ref_frames,
                       const int32_t *target, const uint8_t *editable,
                       int frames, float cfg) {
  if (!READY) { set_err("nar not initialised"); return -1; }
  if (frames <= 0 || frames > VV_NAR_MAX_FRAMES) { set_err("bad frame count %d", frames); return -1; }
  if (cond_len <= 0) { set_err("empty prompt"); return -1; }
  if (n_aux < 0 || n_aux > 2) { set_err("bad aux branch count %d", n_aux); return -1; }
  if (ref_frames < 0 || ref_frames > VV_NAR_MAX_FRAMES) { set_err("bad ref frames %d", ref_frames); return -1; }
  if (ref_frames > 0 && !ref_tokens) { set_err("missing reference tokens"); return -1; }

  const int sep = ref_frames > 0 ? 1 : 0;
  if (cond_len + ref_frames + sep + frames > SEQ) {
    set_err("prompt %d + ref %d + %d frames exceeds %d tokens",
            cond_len, ref_frames, frames, SEQ);
    return -1;
  }
  float aux_sum = 0.0f;
  for (int a = 0; a < n_aux; a++) {
    if (!aux_ids || !aux_ids[a] || aux_lens[a] <= 0) { set_err("bad aux prompt"); return -1; }
    if (aux_lens[a] + ref_frames + sep + frames > SEQ) {
      set_err("aux prompt %d + ref %d + %d frames exceeds %d",
              aux_lens[a], ref_frames, frames, SEQ);
      return -1;
    }
    aux_sum += aux_scales[a];
  }

  COND_LEN = cond_len;
  FRAMES = frames;
  EDIT_MODE = 1;
  CFG = cfg;

  int cells = 0;
  for (int i = 0; i < frames; i++) {
    CFG_MASK[i] = (!editable || editable[i]) ? 1 : 0;
    if (!CFG_MASK[i]) continue;
    for (int c = 0; c < NCB; c++)
      if (!target || target[(size_t)c * frames + i] == MASK_IDS[c]) cells++;
  }
  if (cells == 0) { set_err("cfg: no masked cell to generate"); return -1; }
  if (target) memcpy(CFG_TOK, target, sizeof(int32_t) * NCB * frames);
  else {
    for (int c = 0; c < NCB; c++)
      for (int i = 0; i < frames; i++) CFG_TOK[(size_t)c * frames + i] = MASK_IDS[c];
  }
  if (ref_frames >= frames) { /* keep ref separate; nothing to do */ }

  N_BR = 0;
  BR_W[0] = 1.0f + cfg + aux_sum;
  build_branch(0, cond_ids, cond_len, ref_tokens, ref_frames, target, frames, 0);
  N_BR = 1;
  if (cfg != 0.0f) {
    build_branch(1, nullptr, 0, nullptr, 0, target, frames, 1);
    BR_W[1] = -cfg;
    N_BR = 2;
  }
  for (int a = 0; a < n_aux; a++) {
    if (aux_scales[a] == 0.0f) continue;
    build_branch(N_BR, aux_ids[a], aux_lens[a], ref_tokens, ref_frames, target, frames, 0);
    BR_W[N_BR] = -aux_scales[a];
    N_BR++;
  }

  if (W_OVR_C >= 0.0f) {
    BR_W[0] = W_OVR_C;
    if (N_BR > 1) BR_W[1] = W_OVR_U;
  }

  TOTAL_CELLS = cells;
  REMAINING = cells;
  __android_log_print(ANDROID_LOG_INFO, TAG,
                      "cfg prompt %d + ref %d + %d frames, %d cells, %d branches, cfg %.2f",
                      cond_len, ref_frames, frames, cells, N_BR, CFG);
  return 0;
}

int vv_nar_prepare_edit(const int32_t *cond_ids, int cond_len,
                        const int32_t *target, const uint8_t *editable,
                        int frames, float guidance) {
  return vv_nar_prepare_cfg(cond_ids, cond_len, nullptr, nullptr, nullptr, 0,
                            nullptr, 0, target, editable, frames, guidance);
}

/* -------------------------------------------------------------- embeddings */
static void gather_embeddings(int audio_off, int nframes) {
  const int total = audio_off + nframes;
  for (int i = 0; i < SEQ; i++) {
    float *dst = EMB + (size_t)i * DIM;
    const int audio = (i >= audio_off && i < total);
    if (!audio) {
      int32_t id = (i < audio_off) ? IDS[i] : 0;
      if (id < 0 || (size_t)id >= TEXT_ROWS) id = 0;
      const uint16_t *src = TEXT_EMB + (size_t)id * DIM;
      for (int d = 0; d < DIM; d++) dst[d] = half2float(src[d]);
      continue;
    }
    int32_t sid = IDS[i];
    if (sid < 0 || sid >= SEM_VOCAB) sid = 0;
    const uint16_t *s = SEM_EMB + (size_t)sid * DIM;
    for (int d = 0; d < DIM; d++) dst[d] = half2float(s[d]);
    for (int c = 0; c < N_ACO; c++) {
      int32_t aid = IDS[(size_t)(c + 1) * SEQ + i];
      if (aid < 0 || aid >= ACO_VOCAB) aid = 0;
      const uint16_t *a = ACO_EMB + (size_t)(ACO_OFFS[c] + aid) * DIM;
      for (int d = 0; d < DIM; d++) dst[d] += half2float(a[d]);
    }
  }
}

/* runs llm + head once; caller frees both buffers */
static int run_graphs(const float *attn, float **sem_out, float **aco_out) {
  const int widx = (SEQ == 512) ? 1 : 0;
  const char *ltag = TAG_LLM[widx][0] ? TAG_LLM[widx] : "llm";
  const char *htag = TAG_HEAD[widx][0] ? TAG_HEAD[widx] : "head";
  const char *llm_in[2] = {"inputs_embeds", "attention_mask"};
  const int llm_dt[2] = {RVC_QNN_FLOAT32, RVC_QNN_FLOAT32};
  const void *llm_data[2] = {EMB, attn};
  const size_t llm_el[2] = {(size_t)SEQ * DIM, (size_t)SEQ * SEQ};
  float *out[2] = {nullptr, nullptr};
  size_t out_el[2] = {0, 0};
  size_t n_out = 0;

  const double t_llm0 = nar_now_ms();
  const int llm_rc = qnn_exec(ltag, llm_in, llm_dt, llm_data, llm_el, 2, out, out_el, 2, &n_out);
  T_LLM += nar_now_ms() - t_llm0;
  if (llm_rc != 0) {
    set_err("llm exec: %s", qnn_last_error());
    return -1;
  }
  if (n_out < 1 || out_el[0] < (size_t)SEQ * DIM) {
    set_err("llm output %zu tensors (%zu elems)", n_out, n_out ? out_el[0] : 0);
    free(out[0]);
    return -1;
  }

  const char *hd_in[1] = {"hidden"};
  const int hd_dt[1] = {RVC_QNN_FLOAT32};
  const void *hd_data[1] = {out[0]};
  const size_t hd_el[1] = {(size_t)SEQ * DIM};
  float *hout[2] = {nullptr, nullptr};
  size_t hout_el[2] = {0, 0};
  size_t n_hout = 0;

  const double t_hd0 = nar_now_ms();
  const int hd_rc = qnn_exec(htag, hd_in, hd_dt, hd_data, hd_el, 1, hout, hout_el, 2, &n_hout);
  T_HEAD += nar_now_ms() - t_hd0;
  if (hd_rc != 0) {
    set_err("head exec: %s", qnn_last_error());
    free(out[0]);
    return -1;
  }
  free(out[0]);
  if (n_hout < 2 || hout_el[0] < (size_t)SEQ * SEM_VOCAB ||
      hout_el[1] < (size_t)SEQ * N_ACO * ACO_VOCAB) {
    set_err("head outputs %zu (%zu, %zu)", n_hout,
            n_hout > 0 ? hout_el[0] : 0, n_hout > 1 ? hout_el[1] : 0);
    if (n_hout > 0) free(hout[0]);
    if (n_hout > 1) free(hout[1]);
    return -1;
  }
  *sem_out = hout[0];
  *aco_out = hout[1];
  return 0;
}

/* ---------------------------------------------------------------- sampling */
static inline double rnd_uniform(void) {
  RNG ^= RNG << 13; RNG ^= RNG >> 7; RNG ^= RNG << 17;
  return ((double)(RNG >> 11) + 0.5) * (1.0 / 9007199254740992.0);
}

static inline float gumbel_noise(void) {
  const double u = rnd_uniform();
  return (float)(-log(-log(u + 1e-10) + 1e-10));
}

static void build_schedule(void) {
  for (int i = 0; i <= N_STEPS; i++) {
    const double u = (double)i / N_STEPS;
    SCHED_T[i] = (float)(T_SHIFT * u / (1.0 + (T_SHIFT - 1.0) * u));
  }
}

static int step_k(int step_index) {
  if (step_index == N_STEPS - 1) return REMAINING;
  int k = (int)ceil((double)TOTAL_CELLS *
                    (double)(SCHED_T[step_index + 1] - SCHED_T[step_index]));
  if (k > REMAINING) k = REMAINING;
  if (k < 0) k = 0;
  return k;
}

static char SEM_MASK[SEM_VOCAB];
static char ACO_MASK[ACO_VOCAB];
static int  MASKED_READY = 0;

static void build_masks(void) {
  if (MASKED_READY) return;
  memset(SEM_MASK, 0, sizeof(SEM_MASK));
  memset(ACO_MASK, 0, sizeof(ACO_MASK));
  SEM_MASK[MASK_IDS[0]] = 1;
  SEM_MASK[SEP_IDS[0]] = 1;
  ACO_MASK[MASK_IDS[1]] = 1;
  ACO_MASK[SEP_IDS[1]] = 1;
  MASKED_READY = 1;
}

/* The scoring pass is host-bound and was ~25 % of a 512-token step: 12 800
 * rows/step x 27 684 logits, and the exp() in the normaliser was scalar double.
 * The arm64 path below is 4-wide float with a polynomial exp (rel. error
 * ~1e-7, i.e. far below the fp16 noise already in the logits). */
#if defined(__aarch64__)
#include <arm_neon.h>

/* exp(x) for x <= 0, four lanes. Horner degree 6 on r = x - n*ln2, |r| <= 0.347
 * => truncation error ~1.2e-7; the clamp keeps 2^n inside the exponent range. */
static inline float32x4_t exp4_neg(float32x4_t x) {
  const float32x4_t ln2 = vdupq_n_f32(0.6931471805599453f);
  const float32x4_t inv_ln2 = vdupq_n_f32(1.4426950408889634f);
  x = vmaxq_f32(x, vdupq_n_f32(-87.0f));
  const float32x4_t n = vrndnq_f32(vmulq_f32(x, inv_ln2));
  const float32x4_t r = vfmsq_f32(x, n, ln2);          /* x - n*ln2 */
  float32x4_t p = vdupq_n_f32(1.0f / 720.0f);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 120.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 24.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f / 6.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.5f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  int32x4_t e = vcvtq_s32_f32(n);
  e = vshlq_n_s32(vaddq_s32(e, vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(e));
}
#endif

/* log-sum-exp over the whole vocabulary (generation.py keeps the mask/sep ids
 * out of argmax only, so the normaliser spans every id) */
static float lse_full(const float *x, int V) {
#if defined(__aarch64__)
  int v = 0;
  float32x4_t vm = vdupq_n_f32(-INFINITY);
  for (; v + 8 <= V; v += 8) {
    vm = vmaxq_f32(vm, vld1q_f32(x + v));
    vm = vmaxq_f32(vm, vld1q_f32(x + v + 4));
  }
  for (; v + 4 <= V; v += 4) vm = vmaxq_f32(vm, vld1q_f32(x + v));
  float mx = vmaxvq_f32(vm);
  for (; v < V; v++) if (x[v] > mx) mx = x[v];
  if (!isfinite(mx)) return 0.0f;
  const float32x4_t vmx = vdupq_n_f32(mx);
  float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
  v = 0;
  for (; v + 8 <= V; v += 8) {
    a0 = vaddq_f32(a0, exp4_neg(vsubq_f32(vld1q_f32(x + v), vmx)));
    a1 = vaddq_f32(a1, exp4_neg(vsubq_f32(vld1q_f32(x + v + 4), vmx)));
  }
  for (; v + 4 <= V; v += 4)
    a0 = vaddq_f32(a0, exp4_neg(vsubq_f32(vld1q_f32(x + v), vmx)));
  double s = (double)vaddvq_f32(vaddq_f32(a0, a1));
  for (; v < V; v++) {
    const float d = x[v] - mx;
    if (d > -30.0f) s += exp((double)d);
  }
  return mx + (float)log(s);
#else
  float mx = -INFINITY;
  for (int v = 0; v < V; v++) if (x[v] > mx) mx = x[v];
  if (!isfinite(mx)) return 0.0f;
  double s = 0.0;
  for (int v = 0; v < V; v++) {
    const float d = x[v] - mx;
    if (d > -30.0f) s += exp((double)d);
  }
  return mx + (float)log(s);
#endif
}

/* argmax with the structural ids skipped, keeping the earliest index on ties
 * (same order as the scalar scan it replaces) */
static void argmax_unmasked(const float *x, int V, const char *is_masked,
                            int32_t *arg_out, float *mx_out) {
#if defined(__aarch64__)
  const float32x4_t lanes = {0.0f, 1.0f, 2.0f, 3.0f};
  float32x4_t vbest = vdupq_n_f32(-INFINITY);
  float32x4_t vidx = vdupq_n_f32(0.0f);
  int v = 0;
  for (; v + 4 <= V; v += 4) {
    const float32x4_t xv = vld1q_f32(x + v);
    const float32x4_t iv = vaddq_f32(vdupq_n_f32((float)v), lanes);
    const uint32x4_t gt = vcgtq_f32(xv, vbest);
    vbest = vbslq_f32(gt, xv, vbest);
    vidx = vbslq_f32(gt, iv, vidx);
  }
  float bv[4], iv[4];
  vst1q_f32(bv, vbest);
  vst1q_f32(iv, vidx);
  float best = bv[0];
  int besti = (int)iv[0];
  for (int l = 1; l < 4; l++) {
    if (bv[l] > best || (bv[l] == best && (int)iv[l] < besti)) {
      best = bv[l];
      besti = (int)iv[l];
    }
  }
  for (; v < V; v++) if (x[v] > best) { best = x[v]; besti = v; }
  if (besti >= 0 && besti < V && is_masked[besti]) {
    best = -INFINITY;
    besti = 0;
    for (int i = 0; i < V; i++) {
      if (is_masked[i]) continue;
      if (x[i] > best) { best = x[i]; besti = i; }
    }
  }
  *arg_out = besti;
  *mx_out = best;
#else
  float mx = -INFINITY;
  int arg = 0;
  for (int v = 0; v < V; v++) {
    if (is_masked[v]) continue;
    if (x[v] > mx) { mx = x[v]; arg = v; }
  }
  *arg_out = arg;
  *mx_out = mx;
#endif
}

/* plain (synthesis) row: argmax + log-prob with the structural ids skipped */
static void score_row(const float *x, int V, const char *is_masked,
                      int32_t *pred, float *score) {
  const float lse = lse_full(x, V);
  float mx = -INFINITY;
  int32_t arg = 0;
  argmax_unmasked(x, V, is_masked, &arg, &mx);
  if (!isfinite(mx)) { *pred = 0; *score = -INFINITY; return; }
  *pred = arg;
  *score = mx - lse;
}

struct Cell { float score; int32_t idx; };

static int cmp_cell(const void *a, const void *b) {
  const Cell *x = (const Cell *)a, *y = (const Cell *)b;
  if (x->score < y->score) return 1;
  if (x->score > y->score) return -1;
  return 0;
}

/* shared tail of both loops: rank the still-masked cells, fill the top k */
static int apply_topk(int32_t *tokens, const int32_t *pred, const float *scores,
                      int k, int frames) {
  if (k <= 0) return 0;
  Cell *cells = (Cell *)malloc(sizeof(Cell) * NCB * frames);
  if (!cells) { set_err("out of memory in topk"); return -1; }
  int n = 0;
  for (int c = 0; c < NCB; c++) {
    for (int pos = 0; pos < frames; pos++) {
      const int32_t idx = c * frames + pos;
      if (tokens[idx] != MASK_IDS[c]) continue;
      cells[n].score = scores[idx];
      cells[n].idx = idx;
      n++;
    }
  }
  if (n > k) {
    qsort(cells, (size_t)n, sizeof(Cell), cmp_cell);
    n = k;
  }
  for (int i = 0; i < n; i++) {
    const int32_t idx = cells[i].idx;
    tokens[idx] = pred[idx];
  }
  /* a frame whose semantic cell was filled with the pause id collapses all
   * twelve codebooks to that frame's pause ids (generation.py:1490) */
  for (int i = 0; i < n; i++) {
    const int32_t idx = cells[i].idx;
    if (idx >= frames) continue;                 /* only codebook 0 */
    if (pred[idx] != PAUSE_IDS[0]) continue;
    const int pos = idx;                         /* idx < frames => pos == idx */
    for (int c = 0; c < NCB; c++) tokens[(size_t)c * frames + pos] = PAUSE_IDS[c];
  }
  free(cells);
  return n;
}

static int count_remaining(const int32_t *tokens, int frames) {
  int left = 0;
  for (int c = 0; c < NCB; c++)
    for (int pos = 0; pos < frames; pos++)
      if (tokens[(size_t)c * frames + pos] == MASK_IDS[c]) left++;
  return left;
}

/* -------------------------------------------------------------------- step */
int vv_nar_steps(void) { return N_STEPS; }

int vv_nar_step(int step_index) {
  if (!READY) { set_err("nar not initialised"); return -1; }
  if (EDIT_MODE) { set_err("nar in edit mode"); return -1; }
  if (step_index < 0 || step_index >= N_STEPS) { set_err("bad step %d", step_index); return -1; }
  build_schedule();
  build_masks();

  if (step_index == 0) {
    T_GATHER = T_LLM = T_HEAD = T_MASK = T_ACC = T_SCORE = T_SELECT = 0.0;
  }
  const double t_g0 = nar_now_ms();

  for (int c = 0; c < NCB; c++)
    memcpy(IDS + (size_t)c * SEQ + COND_LEN, TOKENS + (size_t)c * FRAMES,
           sizeof(int32_t) * FRAMES);
  gather_embeddings(COND_LEN, FRAMES);
  T_GATHER += nar_now_ms() - t_g0;

  float *sem = nullptr, *aco = nullptr;
  if (run_graphs(ATTN, &sem, &aco) != 0) return -1;

  int32_t *pred = (int32_t *)malloc(sizeof(int32_t) * NCB * FRAMES);
  float *scores = (float *)malloc(sizeof(float) * NCB * FRAMES);
  if (!pred || !scores) {
    set_err("out of memory in step");
    free(pred); free(scores); free(sem); free(aco);
    return -1;
  }

  const double t_score0 = nar_now_ms();
  for (int pos = 0; pos < FRAMES; pos++) {
    const size_t row = (size_t)(COND_LEN + pos);
    int32_t p0;
    float s0;
    score_row(sem + row * SEM_VOCAB, SEM_VOCAB, SEM_MASK, &p0, &s0);
    pred[0 * FRAMES + pos] = p0;
    scores[0 * FRAMES + pos] = s0;
    for (int c = 0; c < N_ACO; c++) {
      int32_t pc;
      float sc;
      score_row(aco + row * N_ACO * ACO_VOCAB + (size_t)c * ACO_VOCAB, ACO_VOCAB,
                ACO_MASK, &pc, &sc);
      pred[(c + 1) * FRAMES + pos] = pc;
      scores[(c + 1) * FRAMES + pos] = sc;
    }
  }
  free(sem);
  free(aco);
  T_SCORE += nar_now_ms() - t_score0;

  const double t_sel0 = nar_now_ms();
  for (int c = 0; c < NCB; c++) {
    for (int pos = 0; pos < FRAMES; pos++) {
      const int32_t idx = c * FRAMES + pos;
      float s = scores[idx] - (float)c * LAYER_PENALTY;
      s = s / POS_TEMP + gumbel_noise();
      if (TOKENS[idx] != MASK_IDS[c]) s = -INFINITY;
      scores[idx] = s;
    }
  }

  int filled = apply_topk(TOKENS, pred, scores, step_k(step_index), FRAMES);
  T_SELECT += nar_now_ms() - t_sel0;
  free(pred);
  free(scores);
  if (filled < 0) return -1;

  REMAINING = count_remaining(TOKENS, FRAMES);
  return filled;
}

/* --------------------------------------------------------------- edit step */
/* Gathers the embeddings of one branch: text rows for the prompt, sem + sum of
 * aco for the reference frames, the separator and the target window. */
static void gather_branch(int b) {
  const int32_t *ids = BR_IDS[b];
  const int tl = BR_TLEN[b];
  const int total = BR_TOTAL[b];
  for (int i = 0; i < SEQ; i++) {
    float *dst = EMB + (size_t)i * DIM;
    if (i >= tl && i < total) {
      int32_t sid = ids[i];
      if (sid < 0 || sid >= SEM_VOCAB) sid = 0;
      const uint16_t *s = SEM_EMB + (size_t)sid * DIM;
      for (int d = 0; d < DIM; d++) dst[d] = half2float(s[d]);
      for (int c = 0; c < N_ACO; c++) {
        int32_t aid = ids[(size_t)(c + 1) * SEQ + i];
        if (aid < 0 || aid >= ACO_VOCAB) aid = 0;
        const uint16_t *a = ACO_EMB + (size_t)(ACO_OFFS[c] + aid) * DIM;
        for (int d = 0; d < DIM; d++) dst[d] += half2float(a[d]);
      }
      continue;
    }
    int32_t id = (i < tl) ? ids[i] : MASK_IDS[0];
    if (id < 0 || (size_t)id >= TEXT_ROWS) id = 0;
    const uint16_t *src = TEXT_EMB + (size_t)id * DIM;
    for (int d = 0; d < DIM; d++) dst[d] = half2float(src[d]);
  }
}

/* copy the tokens generated so far back into every branch's target window */
static void sync_branch_targets(void) {
  for (int b = 0; b < N_BR; b++) {
    const int t0 = BR_TSTART[b];
    for (int c = 0; c < NCB; c++)
      memcpy(BR_IDS[b] + (size_t)c * SEQ + t0, CFG_TOK + (size_t)c * FRAMES,
             sizeof(int32_t) * FRAMES);
  }
}

/* --------------------------------------------------------------- cfg step */
int vv_nar_cfg_step(int step_index) {
  if (!READY) { set_err("nar not initialised"); return -1; }
  if (!EDIT_MODE) { set_err("nar not in cfg mode"); return -1; }
  if (step_index < 0 || step_index >= N_STEPS) { set_err("bad step %d", step_index); return -1; }
  build_schedule();
  build_masks();
  sync_branch_targets();

  if (step_index == 0) {
    T_GATHER = T_LLM = T_HEAD = T_MASK = T_ACC = T_SCORE = T_SELECT = 0.0;
  }

  const double t_mask0 = nar_now_ms();
  memset(ACC_SEM, 0, sizeof(float) * (size_t)FRAMES * SEM_VOCAB);
  memset(ACC_ACO, 0, sizeof(float) * (size_t)FRAMES * N_ACO * ACO_VOCAB);
  T_MASK += nar_now_ms() - t_mask0;

  for (int b = 0; b < N_BR; b++) {
    const double t_g0 = nar_now_ms();
    gather_branch(b);
    T_GATHER += nar_now_ms() - t_g0;
    float *sem = nullptr, *aco = nullptr;
    if (run_graphs(BR_ATTN[b], &sem, &aco) != 0) return -1;
    const double t_acc0 = nar_now_ms();
    const float w = BR_W[b];
    const int base = BR_TSTART[b];
    if (step_index == 0) {
      for (int pos = 0; pos < FRAMES; pos++) {
        const float *cs = sem + (size_t)(base + pos) * SEM_VOCAB;
        float mx1 = -INFINITY, mx2 = -INFINITY;
        int32_t a1 = 0;
        for (int v = 0; v < SEM_VOCAB; v++) {
          if (SEM_MASK[v]) continue;
          if (cs[v] > mx1) { mx2 = mx1; mx1 = cs[v]; a1 = v; }
          else if (cs[v] > mx2) { mx2 = cs[v]; }
        }
        DIAG_TOK[b][pos] = a1;
        DIAG_LP[b][pos][0] = mx1;
        DIAG_LP[b][pos][1] = mx2;
      }
    }
    for (int pos = 0; pos < FRAMES; pos++) {
      const size_t row = (size_t)(base + pos);
      const float *cs = sem + row * SEM_VOCAB;
      const float *as = aco + row * N_ACO * ACO_VOCAB;
      float *ds = ACC_SEM + (size_t)pos * SEM_VOCAB;
      float *da = ACC_ACO + (size_t)pos * N_ACO * ACO_VOCAB;
      const float lse_s = lse_full(cs, SEM_VOCAB);
      for (int v = 0; v < SEM_VOCAB; v++) ds[v] += w * (cs[v] - lse_s);
      for (int c = 0; c < N_ACO; c++) {
        const float *ac = as + (size_t)c * ACO_VOCAB;
        float *dc = da + (size_t)c * ACO_VOCAB;
        const float lse_a = lse_full(ac, ACO_VOCAB);
        for (int v = 0; v < ACO_VOCAB; v++) dc[v] += w * (ac[v] - lse_a);
      }
    }
    free(sem);
    free(aco);
    T_ACC += nar_now_ms() - t_acc0;
  }

  int32_t *pred = (int32_t *)malloc(sizeof(int32_t) * NCB * FRAMES);
  float *scores = (float *)malloc(sizeof(float) * NCB * FRAMES);
  if (!pred || !scores) {
    set_err("out of memory in cfg step");
    free(pred); free(scores);
    return -1;
  }

  const double t_score0 = nar_now_ms();
  for (int pos = 0; pos < FRAMES; pos++) {
    int32_t p0;
    float s0;
    score_row(ACC_SEM + (size_t)pos * SEM_VOCAB, SEM_VOCAB, SEM_MASK, &p0, &s0);
    pred[0 * FRAMES + pos] = p0;
    scores[0 * FRAMES + pos] = s0;
    for (int c = 0; c < N_ACO; c++) {
      int32_t pc;
      float sc;
      score_row(ACC_ACO + (size_t)pos * N_ACO * ACO_VOCAB + (size_t)c * ACO_VOCAB,
                ACO_VOCAB, ACO_MASK, &pc, &sc);
      pred[(c + 1) * FRAMES + pos] = pc;
      scores[(c + 1) * FRAMES + pos] = sc;
    }
  }
  T_SCORE += nar_now_ms() - t_score0;

  const double t_sel0 = nar_now_ms();
  for (int c = 0; c < NCB; c++) {
    for (int pos = 0; pos < FRAMES; pos++) {
      const int32_t idx = c * FRAMES + pos;
      float s = scores[idx] - (float)c * LAYER_PENALTY;
      s = s / POS_TEMP + gumbel_noise();
      if (!CFG_MASK[pos] || CFG_TOK[idx] != MASK_IDS[c]) s = -INFINITY;
      scores[idx] = s;
    }
  }

  int filled = apply_topk(CFG_TOK, pred, scores, step_k(step_index), FRAMES);
  T_SELECT += nar_now_ms() - t_sel0;
  free(pred);
  free(scores);
  if (step_index == 0 && DIAG_PATH[0]) {
    FILE *f = fopen(DIAG_PATH, "w");
    if (f) {
      for (int b = 0; b < N_BR; b++) {
        fprintf(f, "branch %d w=%.2f tstart=%d\n", b, BR_W[b], BR_TSTART[b]);
        for (int pos = 0; pos < FRAMES; pos++)
          fprintf(f, "%d %d %.4f %.4f %.4f\n", pos, DIAG_TOK[b][pos],
                  DIAG_LP[b][pos][0], DIAG_LP[b][pos][1],
                  DIAG_LP[b][pos][0] - DIAG_LP[b][pos][1]);
      }
      fclose(f);
    }
  }
  if (filled < 0) return -1;

  REMAINING = count_remaining(CFG_TOK, FRAMES);
  return filled;
}

int vv_nar_edit_step(int step_index) { return vv_nar_cfg_step(step_index); }

/* ------------------------------------------------------------------ decode */
int vv_nar_decode(const int32_t *tokens, int frames, float *out, int out_len) {
  if (!vv_decoder_ready()) { set_err("decoder not loaded"); return -1; }
  if (frames <= 0 || frames > VV_NAR_MAX_FRAMES) { set_err("bad frame count %d", frames); return -1; }

  static int32_t sem[200], aco[2200];
  static int keep[VV_NAR_MAX_FRAMES];
  int n = 0;
  for (int pos = 0; pos < frames; pos++) {
    int all_pause = 1, all_sep = 1;
    for (int c = 0; c < NCB; c++) {
      const int32_t v = tokens[(size_t)c * frames + pos];
      if (v != PAUSE_IDS[c]) all_pause = 0;
      if (v != SEP_IDS[c]) all_sep = 0;
    }
    if (all_pause || all_sep) continue;
    keep[n++] = pos;
    if (n >= VV_NAR_MAX_FRAMES) break;
  }
  if (n == 0) { set_err("no speech frames produced"); return -1; }

  for (int i = 0; i < 200; i++) {
    const int src = keep[i < n ? i : n - 1];
    int32_t s = tokens[src];
    if (s < 0 || s >= 16384) s = 0;             /* structural ids cannot decode */
    sem[i] = s;
    for (int c = 0; c < N_ACO; c++) {
      int32_t a = tokens[(size_t)(c + 1) * frames + src];
      if (a < 0 || a >= 1024) a = 1023;          /* decoder table is 1024 rows */
      aco[c * 200 + i] = a;
    }
  }
  int rc = vv_decoder_decode(sem, aco, out, out_len);
  if (rc != 0) return -1;
  int samples = n * FRAME_SAMPLES;
  if (samples > out_len) samples = out_len;
  __android_log_print(ANDROID_LOG_INFO, TAG, "decoding %d/%d frames -> %d samples",
                      n, frames, samples);
  return samples;
}

/* ------------------------------------------------------------------ finish */
int vv_nar_finish(float *out_audio, int out_len) {
  if (!READY) { set_err("nar not initialised"); return -1; }
  if (EDIT_MODE) return vv_nar_decode(CFG_TOK, FRAMES, out_audio, out_len);
  return vv_nar_decode(TOKENS, FRAMES, out_audio, out_len);
}

/* ------------------------------------------------------- debug entry points */
void vv_nar_set_diag_path(const char *path) {
  if (!path) { DIAG_PATH[0] = 0; return; }
  snprintf(DIAG_PATH, sizeof(DIAG_PATH), "%s", path);
}

void vv_nar_set_weights(float wc, float wu) { W_OVR_C = wc; W_OVR_U = wu; }
