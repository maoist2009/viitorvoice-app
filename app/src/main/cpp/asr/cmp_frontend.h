/* cmp_frontend.h — CAM++ 的 Kaldi FBank 前端（CPU）。
 *
 * 与 torchaudio.compliance.kaldi.fbank 对齐（CAM++ 官方前端）：
 *   16 kHz、25/10 ms、×32768、remove_dc_offset、preemph 0.97、
 *   povey 窗 (0.5-0.5cos)^0.85、rfft 512、80 mel（20 Hz..nyquist）、
 *   log(max(·, 1.1921e-07))，然后逐句 CMN（减去时间轴均值）。
 *
 * 固定 T=150 帧（1.5 s）：不足补零、超出截断。
 */
#ifndef CMP_FRONTEND_H
#define CMP_FRONTEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CMP_SR      16000
#define CMP_FRAME   400
#define CMP_HOP     160
#define CMP_NFFT    512
#define CMP_NBINS   (CMP_NFFT / 2 + 1)
#define CMP_N_MELS  80
#define CMP_T       150
#define CMP_DIM     (CMP_N_MELS)

typedef struct {
  double banks[CMP_N_MELS * CMP_NBINS];
  double win[CMP_FRAME];
  double twr[CMP_NFFT / 2];
  double twi[CMP_NFFT / 2];
} cmp_front_t;

int cmp_front_init(cmp_front_t *f);

/* pcm [-1,1] → feats (CMP_T * CMP_N_MELS) 行优先；*n_valid 返回有效帧数。 */
int cmp_front_compute(const cmp_front_t *f, const float *pcm, size_t n,
                      float *feats, int *n_valid);

#ifdef __cplusplus
}
#endif

#endif /* CMP_FRONTEND_H */
