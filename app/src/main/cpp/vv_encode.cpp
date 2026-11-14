/* vv_encode.cpp — DualCodec encoder half on the CPU (ONNX Runtime).
 *
 *   pcm 24 kHz ──resample──> 16 kHz ──> w2v_front8s.onnx ──> semantic_repr
 *        └──pad to 192900──> encode_core ──(+ semantic_repr)──> 12 codebooks
 *
 * Both graphs are fixed/dynamic exactly as exported and verified against the
 * python reference (tokens identical 12/12 x 200 frames on the demo clip).
 */
#include "vv_encode.h"

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

#include <string>
#include <vector>

#include <android/log.h>
#include <onnxruntime_cxx_api.h>

#include "qnn_executor.h"
#include "vv_mel.h"

#define TAG "vv_encode"

/* fixed shapes of the exported semantic frontend */
#define FE_SAMPLES  128640        /* 8.04 s @ 16 kHz */
#define FE_FRAMES   200
#define CORE_SAMPLES 192000       /* 8.0 s @ 24 kHz -> 200 codec frames */
#define CORE_PERIOD  960          /* 40 ms @ 24 kHz */

/* The encode core ships as six chained QNN graphs: one 197 MB context cannot
 * be uploaded in one piece, and the full-rate front has to be time-sliced
 * anyway because HTP insists on holding the whole 64-channel activation in
 * VTCM (16.4 MB) while the device only has 8 MB. */
#define SEG_T   (768 * 200)       /* convnext encoder/decoder feature map */
#define SEG_Z   (1024 * 200)      /* 1024-d latent, 25 Hz */
#define SEG_B3  (512 * 1600)      /* block.3 output, 400 Hz */
/* The sliced front is compiled as a *single* 24960-sample window: eight clones
 * of block.0-2 in one graph come back with ~1/3 of every tile mis-computed by
 * HTP, so the window is executed one at a time and concatenated here. */
#define FRONT_TILE   6000         /* samples per window after the halo */
#define FRONT_HALO   480          /* receptive field is 439 */
#define FRONT_WIN    (FRONT_TILE + 2 * FRONT_HALO)
#define FRONT_TILES  (CORE_SAMPLES / FRONT_TILE)
#define FRONT_CH     256
#define FRONT_FRAMES (FRONT_TILE / 20)   /* 1200 Hz, usable part */
#define FRONT_HALO_F 24                  /* FRONT_HALO / 20 */
#define FRONT_OUTF   (FRONT_FRAMES + 2 * FRONT_HALO_F)
#define FRONT_ALL    (FRONT_TILES * FRONT_FRAMES)

#define SEG_N   7

static const char *SEG_TAG[SEG_N] = {"enc_e1", "enc_e2", "enc_e3",
                                     "enc_front", "enc_e4b", "enc_e5",
                                     "enc_e6"};
static const char *SEG_FILE[SEG_N] = {"enc_e1.bin", "enc_e2.bin", "enc_e3.bin",
                                      "enc_front.bin", "enc_e4b.bin",
                                      "enc_e5.bin", "enc_e6.bin"};

static pthread_mutex_t LOCK = PTHREAD_MUTEX_INITIALIZER;
static char ERR[1024];
static int READY = 0;

static Ort::Env *ENV = nullptr;
static Ort::Session *FE = nullptr;
static Ort::Session *CORE = nullptr;
static Ort::MemoryInfo *MEM = nullptr;
static int CORE_QNN = 0;                  /* encode_core runs on the NPU */
static int SEG_QNN = 0;                   /* six chained segment contexts */
static std::vector<float> FEAT_DBG;       /* cached semantic_repr (debug only) */
static char CODES_PATH[512];
static char DBG_DIR[512];
static int E4_SWAP = 0;                   /* e4s emits two same-shaped tensors */
static int FE_QNN = 0;                    /* w2v semantic frontend runs on the NPU */
static std::vector<float> LOGMEL;         /* VV_MEL_FRAMES * VV_MEL_BINS */
static std::vector<float> FEAT_BUF;       /* NPU semantic_repr, host copy */

static std::vector<float>   A16;          /* FE_SAMPLES */
static std::vector<float>   A24;          /* CORE_SAMPLES */
static std::vector<int64_t> SEM;          /* 1*1*FE_FRAMES */
static std::vector<int64_t> ACO;          /* 1*11*FE_FRAMES */

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static void set_err(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(ERR, sizeof(ERR), fmt, ap);
  va_end(ap);
  __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", ERR);
}

const char *vv_encode_error(void) { return ERR; }
int vv_encode_ready(void) { return READY; }

/* Debug trace: models_dir/../encode_trace.txt (the app's files dir).  logcat
 * is not reliably readable on this device, so the encoder keeps its own log. */
static char TRACE_PATH[512];
static void trace(const char *fmt, ...);

static int DUMP_ON = 0;                   /* models/../DUMP or SELFTEST */
static void dump_f32(const char *name, const float *v, size_t n) {
  if (!DUMP_ON) return;
  char path[512];
  if (!DBG_DIR[0]) return;
  snprintf(path, sizeof(path), "%s/%s", DBG_DIR, name);
  if (FILE *f = fopen(path, "wb")) {
    fwrite(v, sizeof(float), n, f);
    fclose(f);
  }
}

static void trace_ck(const char *tag, const float *v, size_t n) {
  double s = 0, a = 0;
  for (size_t i = 0; i < n; i++) {
    s += v[i];
    a += fabs(v[i]);
  }
  trace("  ck %s n=%zu sum=%.4f abs=%.4f v0=%.5f v1=%.5f",
        tag, n, s, a, n > 0 ? v[0] : 0.0, n > 1 ? v[1] : 0.0);
}

static void trace_tensor(const char *kind, const char *name, int dt,
                         uint32_t rank, const uint32_t *dims) {
  char buf[256];
  int o = snprintf(buf, sizeof(buf), "  %s %s dt%d [", kind, name, dt);
  for (uint32_t k = 0; k < rank && o > 0 && o < (int)sizeof(buf); k++)
    o += snprintf(buf + o, sizeof(buf) - o, "%u%s", dims[k],
                  k + 1 < rank ? "," : "");
  if (o > 0 && o < (int)sizeof(buf)) snprintf(buf + o, sizeof(buf) - o, "]");
  trace("%s", buf);
}

static void trace_in(void *u, const char *n, int dt, uint32_t r, const uint32_t *d) {
  (void)u; trace_tensor("in ", n, dt, r, d);
}

static void trace_out(void *u, const char *n, int dt, uint32_t r, const uint32_t *d) {
  (void)u; trace_tensor("out", n, dt, r, d);
}

static void trace(const char *fmt, ...) {
  if (!TRACE_PATH[0]) return;
  FILE *fp = fopen(TRACE_PATH, "a");
  if (!fp) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(fp, fmt, ap);
  va_end(ap);
  fputc('\n', fp);
  fclose(fp);
}

/* ------------------------------------------------------------- resampler */
#define RS_TAPS 32
#define RS_HALF 16
#define RS_PHASES 1024

static float RSTAB[RS_PHASES][RS_TAPS];
static int   RSTAB_SR_IN = 0, RSTAB_SR_OUT = 0;

static double bessel_i0(double x) {
  double s = 1.0, t = 1.0;
  for (int k = 1; k < 30; k++) {
    t *= (x / (2.0 * k)) * (x / (2.0 * k));
    s += t;
    if (t < 1e-12 * s) break;
  }
  return s;
}

static void build_table(int sr_in, int sr_out) {
  if (RSTAB_SR_IN == sr_in && RSTAB_SR_OUT == sr_out) return;
  const double fc = 0.5 * (sr_out < sr_in ? (double)sr_out / sr_in : 1.0);
  const double beta = 7.0;
  for (int p = 0; p < RS_PHASES; p++) {
    const double frac = (double)p / RS_PHASES;
    double sum = 0.0;
    for (int t = 0; t < RS_TAPS; t++) {
      const double x = (double)(t - RS_HALF + 1) - frac;   /* in input samples */
      const double s = (x == 0.0) ? 2.0 * fc : sin(2.0 * M_PI * fc * x) / (M_PI * x);
      const double r = 2.0 * x / RS_TAPS;
      const double w = bessel_i0(beta * sqrt(fmax(0.0, 1.0 - r * r))) / bessel_i0(beta);
      RSTAB[p][t] = (float)(s * w);
      sum += RSTAB[p][t];
    }
    if (fabs(sum) > 1e-9) {
      for (int t = 0; t < RS_TAPS; t++) RSTAB[p][t] = (float)(RSTAB[p][t] / sum);
    }
  }
  RSTAB_SR_IN = sr_in;
  RSTAB_SR_OUT = sr_out;
}

int vv_resample(const float *in, int n_in, int sr_in, int sr_out,
                float *out, int out_cap, int *n_out) {
  if (!in || !out || n_in <= 0 || sr_in <= 0 || sr_out <= 0) {
    set_err("resample: bad args");
    return -1;
  }
  if (sr_in == sr_out) {
    const int n = n_in < out_cap ? n_in : out_cap;
    memcpy(out, in, sizeof(float) * n);
    *n_out = n;
    return 0;
  }
  if (sr_in != RSTAB_SR_IN || sr_out != RSTAB_SR_OUT) build_table(sr_in, sr_out);
  const int need = (int)((double)n_in * sr_out / sr_in) + 1;
  const int n = need < out_cap ? need : out_cap;
  const double step = (double)sr_in / sr_out;
  for (int m = 0; m < n; m++) {
    const double pos = m * step;
    const int base = (int)floor(pos) - RS_HALF + 1;
    int p = (int)lrint((pos - floor(pos)) * RS_PHASES);
    if (p >= RS_PHASES) p = RS_PHASES - 1;
    const float *h = RSTAB[p];
    float acc = 0.0f;
    for (int t = 0; t < RS_TAPS; t++) {
      const int i = base + t;
      if (i < 0 || i >= n_in) continue;
      acc += h[t] * in[i];
    }
    out[m] = acc;
  }
  *n_out = n;
  return 0;
}

/* ---------------------------------------------------------------- set-up */
static int session_open(const char *path, Ort::Session **out) {
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(2);
  /* w2v_front8s carries 2.25 GB of external fp32 weights; ORT_ENABLE_ALL folds
   * constants into huge temporaries and the 8 GB phone kills the app. */
  so.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
  so.DisableCpuMemArena();
  so.DisableMemPattern();
  try {
    *out = new Ort::Session(*ENV, path, so);
  } catch (const Ort::Exception &e) {
    set_err("%s: %s", path, e.what());
    return -1;
  }
  __android_log_print(ANDROID_LOG_INFO, TAG, "loaded %s", path);
  return 0;
}

int vv_encode_init(const char *models_dir) {
  pthread_mutex_lock(&LOCK);
  if (READY) { pthread_mutex_unlock(&LOCK); return 0; }
  std::string dir(models_dir);
  snprintf(TRACE_PATH, sizeof(TRACE_PATH), "%s/../encode_trace.txt", models_dir);
  int rc = -1;
  try {
    ENV = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "vv_enc");
    MEM = new Ort::MemoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    const std::string fe_path = dir + "/w2v_front8s.onnx";
    const std::string core_path = dir + "/dualcodec_encode_core_192000.onnx";
    const std::string core_bin = dir + "/encode_core_s200.bin";
    for (int i = 0; i < SEG_N && !SEG_QNN; i++) {
      const std::string p = dir + "/" + SEG_FILE[i];
      if (qnn_load(SEG_TAG[i], p.c_str()) != 0) break;
      if (i == SEG_N - 1) SEG_QNN = 1;
    }
    if (SEG_QNN) {
      for (int i = 0; i < SEG_N; i++) {
        trace("== %s", SEG_TAG[i]);
        qnn_describe_raw(SEG_TAG[i], trace_in, trace_out, nullptr);
      }
      const char *sw = getenv("VV_E4_SWAP");
      E4_SWAP = sw && sw[0] == '1';
      __android_log_print(ANDROID_LOG_INFO, TAG,
                          "encode_core -> NPU (%d segments, e4_swap=%d)",
                          SEG_N, E4_SWAP);
    } else if (qnn_load("enc_core", core_bin.c_str()) == 0) {
      CORE_QNN = 1;
      __android_log_print(ANDROID_LOG_INFO, TAG, "encode_core -> NPU (%s)",
                          core_bin.c_str());
    }
    /* w2v semantic frontend on the NPU: log-mel is computed here in fp32 (the
     * graph's FFT magnitudes reach ~1e5, far outside fp16) and only the 560 M
     * parameter backbone runs on HTP. */
    const std::string fe_bin = dir + "/enc_fe.bin";
    const std::string mel_bin = dir + "/fe_mel.bin";
    trace("dump_on=%d fe_bin=%s", DUMP_ON, fe_bin.c_str());
    if (qnn_load("enc_fe", fe_bin.c_str()) == 0) {
      if (vv_mel_init(mel_bin.c_str()) == 0) {
        FE_QNN = 1;
        LOGMEL.assign((size_t)VV_MEL_FRAMES * VV_MEL_BINS, 0.0f);
        FEAT_BUF.assign((size_t)1024 * FE_FRAMES, 0.0f);
        trace("== enc_fe");
        qnn_describe_raw("enc_fe", trace_in, trace_out, nullptr);
        __android_log_print(ANDROID_LOG_INFO, TAG, "semantic frontend -> NPU");
      } else {
        qnn_free("enc_fe");
        trace("fe_mel.bin missing: frontend stays on ORT");
      }
    } else {
      trace("enc_fe load FAILED: %s", qnn_last_error());
    }
    int core_ok = SEG_QNN || CORE_QNN;
    if (!core_ok) core_ok = (session_open(core_path.c_str(), &CORE) == 0);
    snprintf(CODES_PATH, sizeof(CODES_PATH), "%s/../npu_codes.txt", models_dir);
    snprintf(DBG_DIR, sizeof(DBG_DIR), "%s/..", models_dir);
    {
      const std::string dpath = std::string(DBG_DIR) + "/DUMP";
      DUMP_ON = access(dpath.c_str(), F_OK) == 0 ||
                access((std::string(DBG_DIR) + "/SELFTEST").c_str(), F_OK) == 0;
      trace("dump marker %s -> %d (errno %d)", dpath.c_str(), DUMP_ON, errno);
    }
    /* Debug only: a cached semantic_repr freezes the semantic branch, so any
     * reference audio would produce the same cb0. Opt-in via use_feat_cache. */
    const std::string fdbg = dir + "/feat_debug.bin";
    const bool use_cache =
        access((dir + "/use_feat_cache").c_str(), F_OK) == 0;
    if (FILE *ff = use_cache ? fopen(fdbg.c_str(), "rb") : nullptr) {
      std::vector<float> tmp((size_t)1024 * FE_FRAMES);
      if (fread(tmp.data(), sizeof(float), tmp.size(), ff) == tmp.size())
        FEAT_DBG.swap(tmp);
      fclose(ff);
      trace("cached feat: %zu floats", FEAT_DBG.size());
    }
    const int fe_ok = (FE_QNN || !FEAT_DBG.empty())
                          ? 1
                          : (session_open(fe_path.c_str(), &FE) == 0);
    trace("core_ok=%d fe_ok=%d", core_ok, fe_ok);
    if (!fe_ok) {
      trace("fe open FAILED: %s", ERR);
    } else if (!core_ok) {
      /* nothing */
    } else {
      A16.assign(FE_SAMPLES, 0.0f);
      A24.assign(CORE_SAMPLES, 0.0f);
      SEM.assign(FE_FRAMES, 0);
      ACO.assign(11 * FE_FRAMES, 0);
      READY = 1;
      trace("init ok seg=%d core_qnn=%d", SEG_QNN, CORE_QNN);
      rc = 0;
    }
  } catch (const Ort::Exception &e) {
    set_err("ort init: %s", e.what());
  } catch (const std::exception &e) {
    set_err("init: %s", e.what());
  }
  pthread_mutex_unlock(&LOCK);
  return rc;
}

void vv_encode_release(void) {
  pthread_mutex_lock(&LOCK);
  delete FE; FE = nullptr;
  delete CORE; CORE = nullptr;
  if (FE_QNN) { qnn_free("enc_fe"); FE_QNN = 0; }
  if (CORE_QNN) { qnn_free("enc_core"); CORE_QNN = 0; }
  if (SEG_QNN) {
    for (int i = 0; i < SEG_N; i++) qnn_free(SEG_TAG[i]);
    SEG_QNN = 0;
  }
  delete MEM; MEM = nullptr;
  delete ENV; ENV = nullptr;
  READY = 0;
  pthread_mutex_unlock(&LOCK);
}

/* ------------------------------------------------------------------ run */
int vv_encode(const float *pcm24k, int n_samples, int32_t *tokens, int *frames) {
  if (!READY) { set_err("encoder not initialised"); return -1; }
  if (!pcm24k || n_samples <= 0 || !tokens || !frames) { set_err("encode: bad args"); return -1; }

  trace("encode enter n=%d", n_samples);
  const int n24 = n_samples < CORE_SAMPLES ? n_samples : CORE_SAMPLES;
  int n16 = 0;
  /* 24 kHz -> 16 kHz */
  static std::vector<float> tmp;
  tmp.resize(n24 + 1);
  if (vv_resample(pcm24k, n24, VV_ENC_SR24, VV_ENC_SR16, tmp.data(), (int)tmp.size(), &n16) != 0)
    return -1;
  if (n16 > FE_SAMPLES) n16 = FE_SAMPLES;

  memset(A16.data(), 0, sizeof(float) * FE_SAMPLES);
  memcpy(A16.data(), tmp.data(), sizeof(float) * n16);
  memset(A24.data(), 0, sizeof(float) * CORE_SAMPLES);
  memcpy(A24.data(), pcm24k, sizeof(float) * n24);
  dump_f32("npu_a24.f32", A24.data(), CORE_SAMPLES);

  const float valid = (float)n16;
  const int64_t fe_shape[2] = {1, FE_SAMPLES};
  const int64_t vl_shape[1] = {1};

  std::vector<Ort::Value> fe_in;
  std::vector<Ort::Value> fe_out;
  const float *feat = nullptr;
  size_t feat_n = 0;
  try {
    if (FE_QNN) {
      const double m0 = now_ms();
      dump_f32("fe_a16.f32", A16.data(), FE_SAMPLES);
      vv_mel_logmel(A16.data(), LOGMEL.data());
      dump_f32("fe_logmel.f32", LOGMEL.data(), LOGMEL.size());
      const int vf = n16 > 400 ? (n16 - 400) / 160 + 1 : 1;
      float vff = (float)vf;
      const char *fn[2] = {"log", "valid_frames"};
      const int fd[2] = {RVC_QNN_FLOAT32, RVC_QNN_FLOAT32};
      const void *fp[2] = {LOGMEL.data(), &vff};
      const size_t fel[2] = {(size_t)VV_MEL_FRAMES * VV_MEL_BINS, 1};
      float *oo[2] = {nullptr, nullptr};
      size_t oe[2] = {0, 0}, nout = 0;
      const double t0 = now_ms();
      const int r = qnn_exec("enc_fe", fn, fd, fp, fel, 2, oo, oe, 2, &nout);
      trace("mel %.1f ms, enc_fe %.1f ms rc=%d outs=%zu vf=%d",
            t0 - m0, now_ms() - t0, r, nout, vf);
      if (r != 0 || nout < 1 || oe[0] != FEAT_BUF.size()) {
        set_err("enc_fe rc=%d outs=%zu elems=%zu: %s", r, nout,
                nout ? oe[0] : 0, qnn_last_error());
        for (size_t k = 0; k < nout; k++) free(oo[k]);
        return -1;
      }
      memcpy(FEAT_BUF.data(), oo[0], sizeof(float) * oe[0]);
      for (size_t k = 0; k < nout; k++) free(oo[k]);
      feat = FEAT_BUF.data();
      feat_n = FEAT_BUF.size();
      trace_ck("fe", feat, feat_n);
      dump_f32("fe_sem.f32", feat, feat_n);
    } else if (!FEAT_DBG.empty()) {
      feat = FEAT_DBG.data();
      feat_n = FEAT_DBG.size();
      trace("fe cached (%zu floats)", feat_n);
    } else {
      fe_in.push_back(Ort::Value::CreateTensor<float>(
          *MEM, A16.data(), FE_SAMPLES, fe_shape, 2));
      fe_in.push_back(Ort::Value::CreateTensor<float>(
          *MEM, const_cast<float *>(&valid), 1, vl_shape, 1));
      const char *names_in[2] = {"audio16k", "valid_len"};
      const char *names_out[1] = {"semantic_repr"};
      fe_out = FE->Run(Ort::RunOptions{nullptr}, names_in, fe_in.data(), 2,
                       names_out, 1);
      trace("fe done");
      feat = fe_out[0].GetTensorData<float>();
      feat_n = fe_out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    }
    if (feat_n < (size_t)1024 * FE_FRAMES) {
      set_err("frontend returned %zu floats", feat_n);
      return -1;
    }

    /* encode_core: audio (1,1,N) + semantic_repr (1,1024,200) */
    if (SEG_QNN) {
      const int dts[3] = {RVC_QNN_FLOAT32, RVC_QNN_FLOAT32, RVC_QNN_FLOAT32};
      auto run = [&](const char *tag, const char *const *nm,
                     const void *const *pt, const size_t *el, size_t ni) {
        std::vector<std::vector<float>> res;
        float *oo[4] = {nullptr, nullptr, nullptr, nullptr};
        size_t oe[4] = {0, 0, 0, 0}, n = 0;
        const double t0 = now_ms();
        const int rc = qnn_exec(tag, nm, dts, pt, el, ni, oo, oe, 4, &n);
        trace("%s %.1f ms rc=%d outs=%zu", tag, now_ms() - t0, rc, n);
        if (rc != 0) {
          set_err("%s: %s", tag, qnn_last_error());
          for (size_t k = 0; k < n; k++) free(oo[k]);
          return res;
        }
        for (size_t k = 0; k < n; k++) {
          res.emplace_back(oo[k], oo[k] + oe[k]);
          free(oo[k]);
        }
        return res;
      };
      const double q0 = now_ms();
      /* e1: w2v features -> convnext encoder */
      const char *e1n[1] = {"semantic_repr"};
      const void *e1p[1] = {feat};
      const size_t e1e[1] = {(size_t)1024 * FE_FRAMES};
      /* qai-hub widens every graph output that ends in a transpose: index 0 is
       * the pre-permute [1,200,768] copy, index 1 the real [1,768,200]. */
      auto s1 = run("enc_e1", e1n, e1p, e1e, 1);
      if (s1.size() != 2 || s1[1].size() != (size_t)SEG_T) {
        set_err("enc_e1 outs %zu", s1.size());
        return -1;
      }
      /* e2: semantic vq + decoder.0-1 -> [768x200, semantic codes] */
      trace_ck("e1", s1[1].data(), s1[1].size());
      const char *e2n[1] = {"/convnext_encoder/convnext_encoder.3/Add_output_0"};
      const void *e2p[1] = {s1[1].data()};
      const size_t e2e[1] = {(size_t)SEG_T};
      auto s2 = run("enc_e2", e2n, e2p, e2e, 1);
      /* out0 [1,200,768], out1 semantic codes, out2 [1,768,200] */
      if (s2.size() != 3 || s2[2].size() != (size_t)SEG_T ||
          s2[1].size() < (size_t)FE_FRAMES) {
        set_err("enc_e2 outs %zu", s2.size());
        return -1;
      }
      const std::vector<float> *t2 = &s2[2], *semc = &s2[1];
      trace_ck("e2.t2", s2[2].data(), s2[2].size());
      trace_ck("e2.codes", s2[1].data(), s2[1].size());
      /* e3: decoder.2-4 -> 1024-d semantic latent */
      const char *e3n[1] = {"/convnext_decoder/convnext_decoder.1/Add_output_0"};
      const void *e3p[1] = {t2->data()};
      const size_t e3e[1] = {(size_t)SEG_T};
      auto s3 = run("enc_e3", e3n, e3p, e3e, 1);
      if (s3.size() != 1 || s3[0].size() != (size_t)SEG_Z) {
        set_err("enc_e3 out %zu", s3.empty() ? 0 : s3[0].size());
        return -1;
      }
      trace_ck("e3", s3[0].data(), s3[0].size());
      /* front: block.0-2, one 24960-sample window per run */
      static std::vector<float> PAD;
      PAD.assign(CORE_SAMPLES + 2 * FRONT_HALO, 0.0f);
      memcpy(PAD.data() + FRONT_HALO, A24.data(),
             sizeof(float) * CORE_SAMPLES);
      static std::vector<float> FRONTV;
      FRONTV.resize((size_t)FRONT_CH * FRONT_ALL);
      const double f0 = now_ms();
      /* Self test: models_dir/SELFTEST present -> run each audio segment on
       * data taken from files, so a host reference can be compared against
       * this device section by section (logcat is not readable here). */
      if (DBG_DIR[0] &&
          access((std::string(DBG_DIR) + "/SELFTEST").c_str(), F_OK) == 0) {
        char p[512];
        static std::vector<float> ZEROW(FRONT_WIN, 0.0f);
        const void *sp[3] = {ZEROW.data(), PAD.data(),
                             PAD.data() + (size_t)20 * FRONT_TILE};
        const char *nm[3] = {"st_c0.f32", "st_c1.f32", "st_c2.f32"};
        for (int k = 0; k < 3; k++) {
          const char *fn[1] = {"audio"};
          const void *fp[1] = {sp[k]};
          const size_t fe[1] = {(size_t)FRONT_WIN};
          auto fo = run("enc_front", fn, fp, fe, 1);
          if (fo.size() != 1) {
            set_err("selftest front outs %zu", fo.size());
            return -1;
          }
          dump_f32(nm[k], fo[0].data(), fo[0].size());
          trace_ck(nm[k], fo[0].data(), fo[0].size());
        }
        auto rd = [&](const char *nm, size_t n) {
          std::vector<float> v(n, 0.0f);
          snprintf(p, sizeof(p), "%s/%s", DBG_DIR, nm);
          if (FILE *f = fopen(p, "rb")) {
            if (fread(v.data(), 4, n, f) != n) trace("st short %s", nm);
            fclose(f);
          } else {
            trace("st missing %s", nm);
          }
          return v;
        };
        {
          auto fr = rd("st_front.f32", (size_t)FRONT_CH * FRONT_ALL);
          const char *e4n[1] = {"/encoder/block/block.2/block/block.4/Conv_output_0"};
          const void *e4p[1] = {fr.data()};
          const size_t e4e[1] = {(size_t)FRONT_CH * FRONT_ALL};
          auto s4 = run("enc_e4b", e4n, e4p, e4e, 1);
          if (s4.size() == 2 && s4[0].size() == (size_t)SEG_B3) {
            dump_f32("st_e4a.f32", s4[0].data(), s4[0].size());
            dump_f32("st_e4b.f32", s4[1].data(), s4[1].size());
            trace_ck("st.e4a", s4[0].data(), s4[0].size());
            trace_ck("st.e4b", s4[1].data(), s4[1].size());
          } else {
            trace("st e4b outs %zu", s4.size());
          }
        }
        {
          auto b3 = rd("st_b3.f32", (size_t)SEG_B3);
          auto i164 = rd("st_i164.f32", (size_t)SEG_B3);
          const char *e5n[2] = {"/encoder/block/block.3/block/block.4/Conv_output_0",
                                "input.164"};
          const void *e5p[2] = {b3.data(), i164.data()};
          const size_t e5e[2] = {(size_t)SEG_B3, (size_t)SEG_B3};
          auto s5 = run("enc_e5", e5n, e5p, e5e, 2);
          if (s5.size() == 1 && s5[0].size() == (size_t)SEG_Z) {
            dump_f32("st_e5.f32", s5[0].data(), s5[0].size());
            trace_ck("st.e5", s5[0].data(), s5[0].size());
          } else {
            trace("st e5 outs %zu", s5.size());
          }
        }
        set_err("selftest done");
        return -1;
      }
      for (int i = 0; i < FRONT_TILES; i++) {
        const char *fn[1] = {"audio"};
        const void *fp[1] = {PAD.data() + (size_t)i * FRONT_TILE};
        const size_t fe[1] = {(size_t)FRONT_WIN};
        auto fo = run("enc_front", fn, fp, fe, 1);
        if (fo.size() != 1 || fo[0].size() != (size_t)FRONT_CH * FRONT_OUTF) {
          set_err("enc_front outs %zu size %zu", fo.size(),
                  fo.empty() ? 0 : fo[0].size());
          return -1;
        }
        if (i == 0) {
          dump_f32("npu_fo0.f32", fo[0].data(), fo[0].size());
        }
        /* the graph returns the halo as well, trim it here */
        for (int c = 0; c < FRONT_CH; c++)
          memcpy(FRONTV.data() + (size_t)c * FRONT_ALL +
                     (size_t)i * FRONT_FRAMES,
                 fo[0].data() + (size_t)c * FRONT_OUTF + FRONT_HALO_F,
                 sizeof(float) * FRONT_FRAMES);
      }
      trace("front x%d total %.1f ms", FRONT_TILES, now_ms() - f0);
      dump_f32("npu_front.f32", FRONTV.data(), FRONTV.size());
      /* e4b: block.3 -> two 512x1600 tensors */
      const char *e4n[1] = {"/encoder/block/block.2/block/block.4/Conv_output_0"};
      const void *e4p[1] = {FRONTV.data()};
      const size_t e4e[1] = {(size_t)FRONT_CH * FRONT_ALL};
      auto s4 = run("enc_e4b", e4n, e4p, e4e, 1);
      if (s4.size() != 2 || s4[0].size() != (size_t)SEG_B3 ||
          s4[1].size() != (size_t)SEG_B3) {
        set_err("enc_e4b outs %zu", s4.size());
        return -1;
      }
      trace_ck("e4.a", s4[0].data(), s4[0].size());
      dump_f32("npu_e4a.f32", s4[0].data(), s4[0].size());
      dump_f32("npu_e4b.f32", s4[1].data(), s4[1].size());
      trace_ck("e4.b", s4[1].data(), s4[1].size());
      const int sw = E4_SWAP ? 1 : 0;
      const std::vector<float> &b3a = s4[sw ? 1 : 0], &b3b = s4[sw ? 0 : 1];
      /* e5: block.4 */
      const char *e5n[2] = {"/encoder/block/block.3/block/block.4/Conv_output_0",
                            "input.164"};
      const void *e5p[2] = {b3a.data(), b3b.data()};
      const size_t e5e[2] = {(size_t)SEG_B3, (size_t)SEG_B3};
      auto s5 = run("enc_e5", e5n, e5p, e5e, 2);
      if (s5.size() != 1 || s5[0].size() != (size_t)SEG_Z) {
        set_err("enc_e5 out %zu", s5.empty() ? 0 : s5[0].size());
        return -1;
      }
      trace_ck("e5", s5[0].data(), s5[0].size());
      dump_f32("npu_e5.f32", s5[0].data(), s5[0].size());
      dump_f32("npu_e3.f32", s3[0].data(), s3[0].size());
      /* e6: block.6 + the eleven-layer RVQ */
      const char *e6n[2] = {"/convnext_decoder/convnext_decoder.4/Conv_output_0",
                            "input.192"};
      const void *e6p[2] = {s3[0].data(), s5[0].data()};
      const size_t e6e[2] = {(size_t)SEG_Z, (size_t)SEG_Z};
      auto s6 = run("enc_e6", e6n, e6p, e6e, 2);
      if (s6.size() != 1 || s6[0].size() < (size_t)11 * FE_FRAMES) {
        set_err("enc_e6 out %zu", s6.empty() ? 0 : s6[0].size());
        return -1;
      }
      for (int i = 0; i < FE_FRAMES; i++) {
        tokens[i] = (int32_t)(*semc)[i];
        for (int c = 0; c < 11; c++)
          tokens[(c + 1) * FE_FRAMES + i] = (int32_t)s6[0][c * FE_FRAMES + i];
      }
      __android_log_print(ANDROID_LOG_INFO, TAG, "encode_core npu %.0f ms",
                          now_ms() - q0);
    } else if (CORE_QNN) {
      const char *co_names_in[2] = {"audio", "semantic_repr"};
      const int co_dt[2] = {RVC_QNN_FLOAT32, RVC_QNN_FLOAT32};
      const void *co_ptr[2] = {A24.data(), feat};
      const size_t co_el[2] = {(size_t)CORE_SAMPLES, (size_t)1024 * FE_FRAMES};
      float *co_out[2] = {nullptr, nullptr};
      size_t co_out_n[2] = {0, 0}, n_out = 0;
      double q0 = now_ms();
      if (qnn_exec("enc_core", co_names_in, co_dt, co_ptr, co_el, 2,
                   co_out, co_out_n, 2, &n_out) != 0) {
        set_err("encode_core qnn: %s", qnn_last_error());
        return -1;
      }
      __android_log_print(ANDROID_LOG_INFO, TAG, "encode_core npu %.0f ms",
                          now_ms() - q0);
      if (n_out != 2 || co_out_n[0] < (size_t)FE_FRAMES ||
          co_out_n[1] < (size_t)11 * FE_FRAMES) {
        set_err("encode_core qnn outs %zu (%zu/%zu)", n_out,
                n_out > 0 ? co_out_n[0] : 0, n_out > 1 ? co_out_n[1] : 0);
        for (size_t k = 0; k < n_out && k < 2; k++) free(co_out[k]);
        return -1;
      }
      for (int i = 0; i < FE_FRAMES; i++) {
        tokens[i] = (int32_t)co_out[0][i];
        for (int c = 0; c < 11; c++)
          tokens[(c + 1) * FE_FRAMES + i] = (int32_t)co_out[1][c * FE_FRAMES + i];
      }
      free(co_out[0]);
      free(co_out[1]);
    } else {
    const int64_t a_shape[3] = {1, 1, CORE_SAMPLES};
    const int64_t s_shape[3] = {1, 1024, FE_FRAMES};
    std::vector<Ort::Value> co_in;
    co_in.push_back(Ort::Value::CreateTensor<float>(*MEM, A24.data(), CORE_SAMPLES, a_shape, 3));
    co_in.push_back(Ort::Value::CreateTensor<float>(
        *MEM, const_cast<float *>(feat), 1024 * FE_FRAMES, s_shape, 3));
    const char *co_names_in[2] = {"audio", "semantic_repr"};
    const char *co_names_out[2] = {"semantic_codes", "acoustic_codes"};
    auto co_out = CORE->Run(Ort::RunOptions{nullptr}, co_names_in, co_in.data(), 2,
                            co_names_out, 2);
    const int64_t *sem = co_out[0].GetTensorData<int64_t>();
    const int64_t *aco = co_out[1].GetTensorData<int64_t>();
    const size_t sem_n = co_out[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const size_t aco_n = co_out[1].GetTensorTypeAndShapeInfo().GetElementCount();
    if (sem_n < (size_t)FE_FRAMES || aco_n < (size_t)11 * FE_FRAMES) {
      set_err("encode_core returned %zu/%zu", sem_n, aco_n);
      return -1;
    }
      for (int i = 0; i < FE_FRAMES; i++) {
        tokens[i] = (int32_t)sem[i];
        for (int c = 0; c < 11; c++)
          tokens[(c + 1) * FE_FRAMES + i] = (int32_t)aco[c * FE_FRAMES + i];
      }
    }
    uint32_t digest = 2166136261u;
    for (int i = 0; i < 12 * FE_FRAMES; i++) {
      digest ^= (uint32_t)tokens[i];
      digest *= 16777619u;
    }
    __android_log_print(ANDROID_LOG_INFO, TAG,
                        "tokens digest %08x sem0=%d aco0=%d",
                        digest, tokens[0], tokens[FE_FRAMES]);
    trace("digest %08x sem0=%d aco0=%d frames=%d",
          digest, tokens[0], tokens[FE_FRAMES], FE_FRAMES);
    if (FILE *cf = fopen(CODES_PATH, "w")) {
      for (int i = 0; i < FE_FRAMES; i++) fprintf(cf, "%lld ", (long long)tokens[i]);
      fputc('\n', cf);
      for (int c = 0; c < 11; c++) {
        for (int i = 0; i < FE_FRAMES; i++)
          fprintf(cf, "%lld ", (long long)tokens[(c + 1) * FE_FRAMES + i]);
        fputc('\n', cf);
      }
      fclose(cf);
    }
  } catch (const Ort::Exception &e) {
    set_err("ort run: %s", e.what());
    return -1;
  }

  int f = n24 / CORE_PERIOD;
  if (f > FE_FRAMES) f = FE_FRAMES;
  if (f < 1) f = 1;
  *frames = f;
  __android_log_print(ANDROID_LOG_INFO, TAG, "encode %d samples -> %d frames", n_samples, f);
  return 0;
}
