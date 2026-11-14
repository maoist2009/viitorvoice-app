/* vv_pipeline.cpp — the decoder half of ViiTorVoice on the HTP.
 *
 * The dualcodec decoder does not link as one graph (the preparer OOMs), so it
 * ships as five context binaries chained by this file:
 *
 *   p1  int32 codes          -> convolution    [1,768,1600]   (fp16 graph)
 *   p2  convolution          -> convolution_1  [1,384,9600]
 *   p3  convolution_1        -> convolution_2  [1,192,47999]
 *   p4  convolution_2        -> convolution_3  [1,96,191996]
 *   p5  convolution_3 window -> audio          [1,1,48095]     x4, halo 48
 *
 * 200 frames = 191996 samples = 8.0 s at 24 kHz. p5 is time-chunked because its
 * input window needs more VTCM than the 8 MB the HTP has; the block's measured
 * receptive radius is 42 samples, so a 48-sample halo on each side makes the
 * chunked result identical to the single-shot run (rms 2.4e-4, float tail only).
 */
#include "vv_pipeline.h"

#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <android/log.h>

#include "qnn_executor.h"

#define TAG "vv_pipeline"

#define DEC_FRAMES   200
#define DEC_SEM_LEN  200          /* 1 codebook  x 200 frames */
#define DEC_ACO_LEN  2200         /* 11 codebooks x 200 frames */
#define DEC_CHUNK    47999        /* samples per p5 chunk */
#define DEC_TOTAL    191996       /* 4 * DEC_CHUNK */
#define DEC_CH       96           /* p4 output channels */
#define P5_HALO      48
#define P5_IN_LEN    (DEC_CHUNK + 2 * P5_HALO)   /* 48095 */

static pthread_mutex_t LOCK = PTHREAD_MUTEX_INITIALIZER;
static int  READY = 0;
static char ERR[1024];

static int  P1_OUT = 768 * 1600;
static int  P2_OUT = 384 * 9600;
static int  P3_OUT = 192 * 47999;
static int  P4_OUT = 96 * 191996;
static int  P5_OUT = 1 * 1 * P5_IN_LEN;

extern const char *qnn_last_error(void);

static void set_error(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(ERR, sizeof(ERR), fmt, ap);
  va_end(ap);
  __android_log_print(ANDROID_LOG_ERROR, TAG, "%s", ERR);
}

const char *vv_last_error(void) { return ERR; }

int vv_decoder_ready(void) { return READY; }

void vv_decoder_release(void) {
  pthread_mutex_lock(&LOCK);
  if (READY) {
    qnn_free("p1");
    qnn_free("p2");
    qnn_free("p3");
    qnn_free("p4");
    qnn_free("p5");
    READY = 0;
  }
  pthread_mutex_unlock(&LOCK);
}

int vv_decoder_init(const char *models_dir) {
  pthread_mutex_lock(&LOCK);
  if (READY) {
    pthread_mutex_unlock(&LOCK);
    return 0;
  }
  struct {
    const char *tag;
    const char *file;
  } bins[] = {
      {"p1", "dec_p1.bin"},
      {"p2", "dec_p2.bin"},
      {"p3", "dec_p3.bin"},
      {"p4", "dec_p4.bin"},
      {"p5", "dec_p5.bin"},
  };
  char path[1024];
  for (size_t i = 0; i < sizeof(bins) / sizeof(bins[0]); i++) {
    snprintf(path, sizeof(path), "%s/%s", models_dir, bins[i].file);
    if (qnn_load(bins[i].tag, path) != 0) {
      set_error("%s: %s", bins[i].file, qnn_last_error());
      pthread_mutex_unlock(&LOCK);
      return -1;
    }
    __android_log_print(ANDROID_LOG_INFO, TAG, "loaded %s", bins[i].file);
  }
  READY = 1;
  pthread_mutex_unlock(&LOCK);
  return 0;
}

/* One graphExecute with a single output. Returns a malloc'd float buffer. */
static float *run1(const char *tag,
                   const char *const *names, const int *dtypes,
                   const void *const *data, const size_t *elems, size_t n_in,
                   size_t expected, size_t *got) {
  float *out = NULL;
  size_t out_elems = 0, n_out = 0;
  if (qnn_exec(tag, names, dtypes, data, elems, n_in,
               &out, &out_elems, 1, &n_out) != 0) {
    set_error("%s: %s", tag, qnn_last_error());
    return NULL;
  }
  if (n_out != 1 || out_elems != expected) {
    set_error("%s: unexpected output (%zu tensors, %zu elems, want %zu)",
              tag, n_out, out_elems, expected);
    free(out);
    return NULL;
  }
  *got = out_elems;
  return out;
}

int vv_decoder_decode(const int32_t *sem, const int32_t *aco,
                      float *out, int out_len) {
  if (!READY) {
    set_error("decoder not loaded");
    return -1;
  }
  if (!sem || !aco || !out || out_len != DEC_TOTAL) {
    set_error("bad arguments (out_len=%d, want %d)", out_len, DEC_TOTAL);
    return -1;
  }
  pthread_mutex_lock(&LOCK);
  int rc = -1;
  float *o1 = NULL, *o2 = NULL, *o3 = NULL, *o4 = NULL;
  float *halo = NULL;
  size_t got = 0;

  const char *p1_names[] = {"semantic_codes", "acoustic_codes"};
  int p1_types[] = {RVC_QNN_INT32, RVC_QNN_INT32};
  const void *p1_data[] = {sem, aco};
  size_t p1_elems[] = {DEC_SEM_LEN, DEC_ACO_LEN};
  o1 = run1("p1", p1_names, p1_types, p1_data, p1_elems, 2, P1_OUT, &got);
  if (!o1) goto done;

  {
    const char *n[] = {"convolution"};
    int t[] = {RVC_QNN_FLOAT32};
    const void *d[] = {o1};
    size_t e[] = {(size_t)P1_OUT};
    o2 = run1("p2", n, t, d, e, 1, P2_OUT, &got);
    if (!o2) goto done;
    free(o1); o1 = NULL;

    const void *d2[] = {o2};
    size_t e2[] = {(size_t)P2_OUT};
    const char *n2[] = {"convolution_1"};
    o3 = run1("p3", n2, t, d2, e2, 1, P3_OUT, &got);
    if (!o3) goto done;
    free(o2); o2 = NULL;

    const void *d3[] = {o3};
    size_t e3[] = {(size_t)P3_OUT};
    const char *n3[] = {"convolution_2"};
    o4 = run1("p4", n3, t, d3, e3, 1, P4_OUT, &got);
    if (!o4) goto done;
    free(o3); o3 = NULL;
  }

  halo = (float *)calloc((size_t)DEC_CH * P5_IN_LEN, sizeof(float));
  if (!halo) {
    set_error("calloc halo failed");
    goto done;
  }

  for (int chunk = 0; chunk < 4; chunk++) {
    const int start = chunk * DEC_CHUNK - P5_HALO;   /* may be negative */
    int s0 = start < 0 ? 0 : start;
    int e0 = start + P5_IN_LEN;
    if (e0 > DEC_TOTAL) e0 = DEC_TOTAL;
    const int width = e0 - s0;
    if (width > 0) {
      const int offset = s0 - start;
      for (int c = 0; c < DEC_CH; c++) {
        memcpy(halo + (size_t)c * P5_IN_LEN + offset,
               o4 + (size_t)c * DEC_TOTAL + s0,
               (size_t)width * sizeof(float));
      }
    }

    const char *n[] = {"convolution_3"};
    int t[] = {RVC_QNN_FLOAT32};
    const void *d[] = {halo};
    size_t e[] = {(size_t)DEC_CH * P5_IN_LEN};
    float *o5 = run1("p5", n, t, d, e, 1, P5_OUT, &got);
    if (!o5) goto done;
    memcpy(out + (size_t)chunk * DEC_CHUNK,
           o5 + P5_HALO, (size_t)DEC_CHUNK * sizeof(float));
    free(o5);

    /* Clear the halo for the next chunk (memset whole buffer is 18 MB and this
     * runs 4 times; only the edges differ, so wipe the two halo strips). */
    if (chunk + 1 < 4) {
      for (int c = 0; c < DEC_CH; c++) {
        memset(halo + (size_t)c * P5_IN_LEN, 0, P5_HALO * sizeof(float));
        memset(halo + (size_t)c * P5_IN_LEN + P5_IN_LEN - P5_HALO, 0,
               P5_HALO * sizeof(float));
      }
    }
  }
  rc = 0;

done:
  free(o1); free(o2); free(o3); free(o4); free(halo);
  pthread_mutex_unlock(&LOCK);
  return rc;
}
