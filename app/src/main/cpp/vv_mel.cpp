#include "vv_mel.h"

#include "mstore.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

constexpr int N_FFT = 512;
constexpr int WIN = 400;
constexpr int HOP = 160;
constexpr int BINS = N_FFT / 2 + 1;   /* 257 */
constexpr int MEL = VV_MEL_BINS;
constexpr int FRAMES = VV_MEL_FRAMES;
constexpr float PREEMPH = 0.97f;
constexpr float CLAMP_MIN = 1.1920929e-7f;
constexpr float SCALE_IN = 32768.0f;

float g_win[WIN];
float g_fb[BINS * MEL];
float g_tw[9][256][2];   /* per-stage twiddles, contiguous per stage */
int g_rev[N_FFT];

/* contiguous nonzero span of each mel channel: cost drops from 257 to ~6 */
struct Span {
  int start;
  int len;
  const float *w;
};
Span g_span[MEL];
/* filterbank transposed to [mel][bin] so the dot product above is contiguous
 * and vectorizes; the blob itself is [bin][mel] */
float g_fbT[MEL * BINS];
bool g_ready = false;

void dsp_init(void) {
  for (int i = 0; i < N_FFT; i++) {
    int r = 0;
    for (int b = 0; b < 9; b++)
      if ((i >> b) & 1) r |= 1 << (8 - b);
    g_rev[i] = r;
  }
  /* Per-stage twiddle tables. A single 256-entry table read with stride
   * N_FFT/len would touch a different cache line per butterfly on the early
   * stages; copying the twiddles stage by stage keeps each pass sequential. */
  for (int s = 0; s < 9; s++) {
    const int len = 2 << s, half = 1 << s, step = N_FFT / len;
    for (int k = 0; k < half; k++) {
      const double t = -2.0 * M_PI * (double)(k * step) / (double)N_FFT;
      g_tw[s][k][0] = (float)std::cos(t);
      g_tw[s][k][1] = (float)std::sin(t);
    }
  }
}

/* in-place radix-2 complex FFT, 512 points */
void fft512(float *__restrict re, float *__restrict im) {
  for (int i = 0; i < N_FFT; i++) {
    const int j = g_rev[i];
    if (j > i) {
      float t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int s = 0; s < 9; s++) {
    const int len = 2 << s, half = 1 << s;
    const float (*tw)[2] = g_tw[s];
    for (int i = 0; i < N_FFT; i += len) {
      for (int k = 0; k < half; k++) {
        const float wr = tw[k][0];
        const float wi = tw[k][1];
        const int a = i + k, b = a + half;
        const float xr = re[b] * wr - im[b] * wi;
        const float xi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - xr;
        im[b] = im[a] - xi;
        re[a] += xr;
        im[a] += xi;
      }
    }
  }
}

}  // namespace

int vv_mel_init(const char *blob_path) {
  FILE *f = mstore_fopen(blob_path);
  if (!f) return -1;
  const size_t got_w = fread(g_win, sizeof(float), WIN, f);
  const size_t got_f = fread(g_fb, sizeof(float), (size_t)BINS * MEL, f);
  fclose(f);
  if (got_w != WIN || got_f != (size_t)BINS * MEL) return -2;
  for (int k = 0; k < BINS; k++)
    for (int m = 0; m < MEL; m++) g_fbT[m * BINS + k] = g_fb[k * MEL + m];
  for (int m = 0; m < MEL; m++) {
    int s = -1, e = -1;
    for (int k = 0; k < BINS; k++)
      if (g_fbT[m * BINS + k] != 0.0f) {
        if (s < 0) s = k;
        e = k;
      }
    g_span[m].start = s < 0 ? 0 : s;
    g_span[m].len = s < 0 ? 0 : e - s + 1;
    g_span[m].w = g_fbT + m * BINS + (s < 0 ? 0 : s);
  }
  dsp_init();
  g_ready = true;
  return 0;
}

void vv_mel_logmel(const float *a16, float *out) {
  if (!g_ready) return;
  float re[N_FFT], im[N_FFT], sub[WIN], pw[BINS];

  for (int fr = 0; fr < FRAMES; fr++) {
    const float *src = a16 + (size_t)fr * HOP;

    float sum = 0.0f;
    for (int i = 0; i < WIN; i++) {
      sub[i] = src[i] * SCALE_IN;
      sum += sub[i];
    }
    const float mean = sum / (float)WIN;

    for (int i = 0; i < WIN; i++) sub[i] -= mean;

    /* pre-emphasis, first sample edge-padded: x0 - PREEMPH*x0 */
    float prev = sub[0];
    re[0] = ((1.0f - PREEMPH) * sub[0]) * g_win[0];
    for (int i = 1; i < WIN; i++) {
      re[i] = (sub[i] - PREEMPH * prev) * g_win[i];
      prev = sub[i];
    }
    for (int i = WIN; i < N_FFT; i++) re[i] = 0.0f;
    memset(im, 0, sizeof(im));

    fft512(re, im);

    for (int k = 0; k < BINS; k++) {
      pw[k] = re[k] * re[k] + im[k] * im[k];
      if (pw[k] < 0.0f) pw[k] = 0.0f;
    }

    float *o = out + (size_t)fr * MEL;
    for (int m = 0; m < MEL; m++) {
      const Span &s = g_span[m];
      const float *w = s.w;
      const float *p = pw + s.start;
      float acc = 0.0f;
      for (int k = 0; k < s.len; k++) acc += p[k] * w[k];
      o[m] = std::log(acc < CLAMP_MIN ? CLAMP_MIN : acc);
    }
  }
}
