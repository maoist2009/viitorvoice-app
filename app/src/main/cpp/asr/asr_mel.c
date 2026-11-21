/* asr_mel.c — Whisper 风格 log-mel（见 asr_mel.h）。
 *
 * 数值路径逐项对齐 transformers.WhisperFeatureExtractor：
 *   reflect pad 200 / periodic hann(400) / |rfft_400|^2 / 丢最后一帧 /
 *   slaney mel(128,201) + slaney 面积归一化 / log10(clamp 1e-10) /
 *   max(x, max-8) / (x+4)/4
 *
 * 400 点 DFT 直接用定义式（每帧 201 个正频率 bin）：单帧 ~8 万次乘加，
 * 8 秒窗口 ~6400 万次，手机单核几十毫秒量级，先保证正确；v73 起 HTP 有
 * DFT op，后续可把整段挪到 NPU。
 */
#include "asr_mel.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define N_BINS (ASR_MEL_N_FFT / 2 + 1)          /* 201 */
#define PAD    (ASR_MEL_N_FFT / 2)              /* 200 */
#define TWO_PI 6.283185307179586476925286766559

/* ---- slaney mel ---- */
static double hz_to_mel(double hz) {
  const double f_sp = 200.0 / 3.0;
  const double min_log_hz = 1000.0;
  const double min_log_mel = min_log_hz / f_sp;          /* 15.0 */
  const double logstep = log(6.4) / 27.0;
  if (hz >= min_log_hz) return min_log_mel + log(hz / min_log_hz) / logstep;
  return hz / f_sp;
}

static double mel_to_hz(double mel) {
  const double f_sp = 200.0 / 3.0;
  const double min_log_hz = 1000.0;
  const double min_log_mel = min_log_hz / f_sp;
  const double logstep = log(6.4) / 27.0;
  if (mel >= min_log_mel) return min_log_hz * exp(logstep * (mel - min_log_mel));
  return mel * f_sp;
}

int asr_mel_init(asr_mel_t *m) {
  memset(m, 0, sizeof(*m));
  m->fb = (double *)calloc((size_t)ASR_MEL_N_MELS * N_BINS, sizeof(double));
  m->win = (double *)calloc(ASR_MEL_N_FFT, sizeof(double));
  m->ct = (double *)calloc(ASR_MEL_N_FFT, sizeof(double));
  m->st = (double *)calloc(ASR_MEL_N_FFT, sizeof(double));
  if (!m->fb || !m->win || !m->ct || !m->st) {
    asr_mel_free(m);
    return -1;
  }

  /* torch.hann_window(400) 默认 periodic=True */
  for (int n = 0; n < ASR_MEL_N_FFT; n++)
    m->win[n] = 0.5 - 0.5 * cos(TWO_PI * n / ASR_MEL_N_FFT);

  for (int i = 0; i < ASR_MEL_N_FFT; i++) {
    m->ct[i] = cos(TWO_PI * i / ASR_MEL_N_FFT);
    m->st[i] = sin(TWO_PI * i / ASR_MEL_N_FFT);
  }

  /* mel 三角滤波器：mel_min..mel_max 等距 130 点 -> Hz */
  double mel_min = hz_to_mel(0.0);
  double mel_max = hz_to_mel(8000.0);
  double hz[ASR_MEL_N_MELS + 2];
  for (int i = 0; i < ASR_MEL_N_MELS + 2; i++) {
    double mel = mel_min + (mel_max - mel_min) * i / (ASR_MEL_N_MELS + 1);
    hz[i] = mel_to_hz(mel);
  }
  /* fft_freqs = linspace(0, sr//2, 201) */
  for (int k = 0; k < N_BINS; k++) {
    double f = 8000.0 * k / (N_BINS - 1);
    for (int mi = 0; mi < ASR_MEL_N_MELS; mi++) {
      double lower = hz[mi], center = hz[mi + 1], upper = hz[mi + 2];
      double down = (f - lower) / (center - lower);
      double up = (upper - f) / (upper - center);
      double w = down < up ? down : up;
      if (w < 0.0) w = 0.0;
      /* slaney 面积归一化：enorm = 2 / (hz[mi+2] - hz[mi]) */
      m->fb[(size_t)mi * N_BINS + k] = w * (2.0 / (upper - lower));
    }
  }
  return 0;
}

void asr_mel_free(asr_mel_t *m) {
  free(m->fb);
  free(m->win);
  free(m->ct);
  free(m->st);
  memset(m, 0, sizeof(*m));
}

/* 左边界 numpy 'reflect'（不重复端点）；右边界一律 0 —— HF 先把波形 zero-pad
 * 到 chunk_length(30 s)，所以 8 s 窗口的右边界之外是 0 而不是镜像。 */
static double sample_left_reflect(const float *x, size_t n, long j) {
  if (j < 0) j = -j;
  if (j >= (long)n) j = (long)n - 1;
  return (double)x[j];
}

int asr_mel_compute(const asr_mel_t *m, const float *pcm, size_t n, float *out) {
  static const size_t NS = (size_t)ASR_MEL_SAMPLES;
  if (n > NS) n = NS;

  double frame[ASR_MEL_N_FFT];
  double power[N_BINS];

  for (int f = 0; f < ASR_MEL_FRAMES; f++) {
    long start = (long)f * ASR_MEL_HOP - PAD;
    for (int i = 0; i < ASR_MEL_N_FFT; i++) {
      long j = start + i;
      double v;
      if (j < 0) v = sample_left_reflect(pcm, n, j);
      else if (j < (long)n) v = (double)pcm[j];
      else v = 0.0;
      frame[i] = v * m->win[i];
    }
    for (int k = 0; k < N_BINS; k++) {
      double re = 0.0, im = 0.0;
      int idx = 0;
      for (int i = 0; i < ASR_MEL_N_FFT; i++) {
        re += frame[i] * m->ct[idx];
        im -= frame[i] * m->st[idx];
        idx += k;
        if (idx >= ASR_MEL_N_FFT) idx -= ASR_MEL_N_FFT;
      }
      power[k] = re * re + im * im;
    }
    for (int mi = 0; mi < ASR_MEL_N_MELS; mi++) {
      const double *w = m->fb + (size_t)mi * N_BINS;
      double acc = 0.0;
      for (int k = 0; k < N_BINS; k++) acc += w[k] * power[k];
      out[(size_t)mi * ASR_MEL_FRAMES + f] = (float)acc;
    }
  }

  /* log10(clamp 1e-10)，全局 max-8 截断，再 (x+4)/4 */
  const double LOG_FLOOR = -10.0;   /* log10(1e-10) */
  double gmax = LOG_FLOOR;
  size_t total = (size_t)ASR_MEL_N_MELS * ASR_MEL_FRAMES;
  for (size_t i = 0; i < total; i++) {
    double v = out[i];
    double l = v > 1e-10 ? log10(v) : LOG_FLOOR;
    out[i] = (float)l;
    if (l > gmax) gmax = l;
  }
  double hi = gmax - 8.0;
  for (size_t i = 0; i < total; i++) {
    double l = out[i];
    if (l < hi) l = hi;
    out[i] = (float)((l + 4.0) / 4.0);
  }
  return 0;
}
