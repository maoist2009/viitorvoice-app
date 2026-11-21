/* sv_frontend.c — 见 sv_frontend.h 的数值说明。 */
#include "sv_frontend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double hz2mel(double f) { return 1127.0 * log(1.0 + f / 700.0); }

int sv_front_init(sv_front_t *f, const char *cmvn_path) {
  if (!f) return -1;
  memset(f, 0, sizeof(*f));

  /* --- kaldi 三角 mel 滤波器组 (80, 257) --- */
  const double sr = SV_SR;
  const double high = sr / 2.0;          /* high_freq=0 -> nyquist */
  const double binw = sr / (double)SV_NFFT;
  const double mel_low = hz2mel(20.0);
  const double mel_high = hz2mel(high);
  const double delta = (mel_high - mel_low) / (SV_N_MELS + 1);
  for (int i = 0; i < SV_N_MELS; i++) {
    const double left = mel_low + i * delta;
    const double center = mel_low + (i + 1) * delta;
    const double right = mel_low + (i + 2) * delta;
    for (int k = 0; k < SV_NBINS; k++) {
      const double fm = hz2mel((double)k * binw);
      double w = (fm <= center) ? (fm - left) / (center - left)
                                : (right - fm) / (right - center);
      if (fm <= left || fm >= right) w = 0.0;
      f->banks[(size_t)i * SV_NBINS + (size_t)k] = w;
    }
  }

  /* --- hamming 窗 --- */
  for (int i = 0; i < SV_FRAME; i++)
    f->win[i] = 0.54 - 0.46 * cos(2.0 * M_PI * (double)i / (double)(SV_FRAME - 1));

  /* --- radix-2 512 点旋转因子 --- */
  for (int i = 0; i < SV_NFFT / 2; i++) {
    f->twr[i] = cos(-2.0 * M_PI * (double)i / (double)SV_NFFT);
    f->twi[i] = sin(-2.0 * M_PI * (double)i / (double)SV_NFFT);
  }

  /* --- CMVN --- */
  FILE *fp = fopen(cmvn_path, "rb");
  if (!fp) return -1;
  size_t got = fread(f->cmvn, sizeof(float), 2 * SV_DIM, fp);
  fclose(fp);
  return got == (size_t)(2 * SV_DIM) ? 0 : -1;
}

/* 就地 512 点复数 FFT（输入实部 re，虚部 im 置 0）。 */
static void fft512(const sv_front_t *f, double *re, double *im) {
  for (int i = 1, j = 0; i < SV_NFFT; i++) {
    int bit = SV_NFFT >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      double t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= SV_NFFT; len <<= 1) {
    const int half = len >> 1;
    const int step = SV_NFFT / len;
    for (int base = 0; base < SV_NFFT; base += len) {
      for (int k = 0; k < half; k++) {
        const double wr = f->twr[k * step], wi = f->twi[k * step];
        const double ur = re[base + k], ui = im[base + k];
        const double xr = re[base + k + half], xi = im[base + k + half];
        const double vr = xr * wr - xi * wi;
        const double vi = xr * wi + xi * wr;
        re[base + k] = ur + vr;
        im[base + k] = ui + vi;
        re[base + k + half] = ur - vr;
        im[base + k + half] = ui - vi;
      }
    }
  }
}

int sv_front_compute(const sv_front_t *f, const float *pcm, size_t n,
                     float *feats, float *mask, int *n_valid) {
  if (!f || !pcm || !feats || !mask) return -1;
  if (n < SV_FRAME) return -1;

  const int nf = 1 + (int)((n - SV_FRAME) / SV_HOP);
  double *fb = (double *)malloc(sizeof(double) * (size_t)nf * SV_N_MELS);
  if (!fb) return -1;

  double re[SV_NFFT], im[SV_NFFT], pw[SV_NBINS];
  for (int t = 0; t < nf; t++) {
    const float *src = pcm + (size_t)t * SV_HOP;
    double mean = 0.0;
    for (int j = 0; j < SV_FRAME; j++) mean += (double)src[j] * 32768.0;
    mean /= (double)SV_FRAME;
    for (int j = 0; j < SV_FRAME; j++) re[j] = (double)src[j] * 32768.0 - mean;
    /* preemph 用原始值：从后往前减 */
    for (int j = SV_FRAME - 1; j >= 1; j--) re[j] -= 0.97 * re[j - 1];
    for (int j = 0; j < SV_FRAME; j++) re[j] *= f->win[j];
    for (int j = SV_FRAME; j < SV_NFFT; j++) re[j] = 0.0;
    memset(im, 0, sizeof(im));
    fft512(f, re, im);
    for (int k = 0; k < SV_NBINS; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
    double *row = fb + (size_t)t * SV_N_MELS;
    for (int m = 0; m < SV_N_MELS; m++) {
      const double *b = f->banks + (size_t)m * SV_NBINS;
      double acc = 0.0;
      for (int k = 0; k < SV_NBINS; k++) acc += pw[k] * b[k];
      row[m] = log(acc > SV_EPS ? acc : SV_EPS);
    }
  }

  /* --- LFR(7,6)：左补 3 帧、尾部按 funasr 公式补齐（补的都是最后一帧） --- */
  const int left = (SV_LFR_M - 1) / 2;            /* 3 */
  const int t_lfr = (nf + SV_LFR_N - 1) / SV_LFR_N;
  const int rows = left + nf;                     /* 实际存在的行数 */
  int jmax = t_lfr < SV_T ? t_lfr : SV_T;

  for (int j = 0; j < jmax; j++) {
    float *dst = feats + (size_t)j * SV_DIM;
    for (int m = 0; m < SV_LFR_M; m++) {
      int idx = j * SV_LFR_N + m;
      const double *src;
      if (idx < left) src = fb;                   /* 左补：第 0 帧 */
      else if (idx < rows) src = fb + (size_t)(idx - left) * SV_N_MELS;
      else src = fb + (size_t)(nf - 1) * SV_N_MELS;  /* 尾补：最后一帧 */
      for (int k = 0; k < SV_N_MELS; k++) {
        const int d = m * SV_N_MELS + k;
        dst[d] = (float)(((double)src[k] + f->cmvn[d]) * f->cmvn[SV_DIM + d]);
      }
    }
  }
  /* 补零到 SV_T */
  if (jmax < SV_T)
    memset(feats + (size_t)jmax * SV_DIM, 0, sizeof(float) * (size_t)(SV_T - jmax) * SV_DIM);

  for (int j = 0; j < SV_T; j++) mask[j] = j < jmax ? 1.0f : 0.0f;
  if (n_valid) *n_valid = jmax;

  free(fb);
  return 0;
}
