/* cmp_frontend.c — CAM++ 的 Kaldi FBank 前端，见 cmp_frontend.h。
 *
 * 与 sv_frontend.c 的差别（别混用）：
 *   窗函数 povey 而不是 hamming；
 *   没有 LFR、没有 CMVN 文件，改成逐句 CMN（减时间轴均值）；
 *   输出固定 CMP_T=150 帧，不足补零、超出截断（CMN 只用有效帧统计）。
 */
#include "cmp_frontend.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CMP_EPS 1.1921e-07

static double hz2mel(double f) { return 1127.0 * log(1.0 + f / 700.0); }

int cmp_front_init(cmp_front_t *f) {
  if (!f) return -1;
  memset(f, 0, sizeof(*f));

  /* --- kaldi 三角 mel 滤波器组 (80, 257)，20 Hz..nyquist --- */
  const double sr = CMP_SR;
  const double high = sr / 2.0;
  const double binw = sr / (double)CMP_NFFT;
  const double mel_low = hz2mel(20.0);
  const double mel_high = hz2mel(high);
  const double delta = (mel_high - mel_low) / (CMP_N_MELS + 1);
  for (int i = 0; i < CMP_N_MELS; i++) {
    const double left = mel_low + i * delta;
    const double center = mel_low + (i + 1) * delta;
    const double right = mel_low + (i + 2) * delta;
    for (int k = 0; k < CMP_NBINS; k++) {
      const double fm = hz2mel((double)k * binw);
      double w = 0.0;
      if (fm > left && fm < right) {
        w = (fm <= center) ? (fm - left) / (center - left)
                           : (right - fm) / (right - center);
      }
      f->banks[(size_t)i * CMP_NBINS + k] = w;
    }
  }

  /* --- povey 窗：(0.5 - 0.5cos)^0.85 --- */
  for (int i = 0; i < CMP_FRAME; i++) {
    const double c = 0.5 - 0.5 * cos(2.0 * M_PI * (double)i / (double)(CMP_FRAME - 1));
    f->win[i] = pow(c, 0.85);
  }

  /* --- radix-2 512 点旋转因子 --- */
  for (int i = 0; i < CMP_NFFT / 2; i++) {
    f->twr[i] = cos(-2.0 * M_PI * (double)i / (double)CMP_NFFT);
    f->twi[i] = sin(-2.0 * M_PI * (double)i / (double)CMP_NFFT);
  }
  return 0;
}

/* 就地 512 点复数 FFT（输入实部 re，虚部 im 置 0）。 */
static void fft512(const cmp_front_t *f, double *re, double *im) {
  for (int i = 1, j = 0; i < CMP_NFFT; i++) {
    int bit = CMP_NFFT >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      double t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (int len = 2; len <= CMP_NFFT; len <<= 1) {
    const int half = len >> 1;
    const int step = CMP_NFFT / len;
    for (int base = 0; base < CMP_NFFT; base += len) {
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

int cmp_front_compute(const cmp_front_t *f, const float *pcm, size_t n,
                      float *feats, int *n_valid) {
  if (!f || !pcm || !feats) return -1;
  if (n < CMP_FRAME) return -1;

  const int nf = 1 + (int)((n - CMP_FRAME) / CMP_HOP);
  double *fb = (double *)malloc(sizeof(double) * (size_t)nf * CMP_N_MELS);
  if (!fb) return -1;

  double re[CMP_NFFT], im[CMP_NFFT], pw[CMP_NBINS];
  for (int t = 0; t < nf; t++) {
    const float *src = pcm + (size_t)t * CMP_HOP;
    double mean = 0.0;
    for (int j = 0; j < CMP_FRAME; j++) mean += (double)src[j] * 32768.0;
    mean /= (double)CMP_FRAME;
    for (int j = 0; j < CMP_FRAME; j++) re[j] = (double)src[j] * 32768.0 - mean;
    /* preemph 用去直流后的值：从后往前减，避免覆盖 */
    for (int j = CMP_FRAME - 1; j >= 1; j--) re[j] -= 0.97 * re[j - 1];
    for (int j = 0; j < CMP_FRAME; j++) re[j] *= f->win[j];
    for (int j = CMP_FRAME; j < CMP_NFFT; j++) re[j] = 0.0;
    memset(im, 0, sizeof(im));
    fft512(f, re, im);
    for (int k = 0; k < CMP_NBINS; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
    double *row = fb + (size_t)t * CMP_N_MELS;
    for (int m = 0; m < CMP_N_MELS; m++) {
      const double *b = f->banks + (size_t)m * CMP_NBINS;
      double acc = 0.0;
      for (int k = 0; k < CMP_NBINS; k++) acc += pw[k] * b[k];
      row[m] = log(acc > CMP_EPS ? acc : CMP_EPS);
    }
  }

  /* --- 逐句 CMN：减时间轴均值（只用有效帧） --- */
  double mu[CMP_N_MELS];
  for (int m = 0; m < CMP_N_MELS; m++) mu[m] = 0.0;
  for (int t = 0; t < nf; t++) {
    const double *row = fb + (size_t)t * CMP_N_MELS;
    for (int m = 0; m < CMP_N_MELS; m++) mu[m] += row[m];
  }
  for (int m = 0; m < CMP_N_MELS; m++) mu[m] /= (double)nf;

  /* --- 固定 150 帧：截断或补零 --- */
  const int tmax = nf < CMP_T ? nf : CMP_T;
  for (int t = 0; t < tmax; t++) {
    const double *row = fb + (size_t)t * CMP_N_MELS;
    float *dst = feats + (size_t)t * CMP_N_MELS;
    for (int m = 0; m < CMP_N_MELS; m++) dst[m] = (float)(row[m] - mu[m]);
  }
  if (tmax < CMP_T)
    memset(feats + (size_t)tmax * CMP_N_MELS, 0,
           sizeof(float) * (size_t)(CMP_T - tmax) * CMP_N_MELS);

  if (n_valid) *n_valid = tmax;
  free(fb);
  return 0;
}
