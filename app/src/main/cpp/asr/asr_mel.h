/* asr_mel.h — Qwen3-ASR 的 Whisper 风格 log-mel 前端（CPU）。
 *
 * 忠实复刻 transformers.WhisperFeatureExtractor 的数值路径：
 *   reflect pad 200 -> hann(400) -> |rfft|^2（丢掉最后一帧）
 *   -> slaney mel(128,201) -> log10(clamp 1e-10)
 *   -> max(x, max-8) -> (x+4)/4
 * 之所以不复用 dsp.c 的 RMVPE mel：那是 HTK mel / n_fft=1024 / 不同的
 * 归一化，喂给编码器会静默掉点。
 */
#ifndef ASR_MEL_H
#define ASR_MEL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ASR_MEL_SR      16000
#define ASR_MEL_N_FFT   400
#define ASR_MEL_HOP     160
#define ASR_MEL_N_MELS  128
#define ASR_MEL_FRAMES  800        /* 8 s @16k：1+(128000/160) 帧再丢 1 帧 */
#define ASR_MEL_SAMPLES (ASR_MEL_FRAMES * ASR_MEL_HOP)   /* 128000 */

typedef struct {
  double *fb;          /* (128, 201) slaney 归一化滤波器组，行主序 */
  double *win;         /* (400) periodic hann */
  double *ct, *st;     /* (400) cos/sin 表，按 (k*n) % 400 取 */
} asr_mel_t;

int  asr_mel_init(asr_mel_t *m);
void asr_mel_free(asr_mel_t *m);

/* pcm: n 个 16 kHz 单声道样本（float，[-1,1]）。n 少于 128000 时右侧补零；
 * 多于时只取前 128000。out 必须是 128*800 个 float，行主序（mel 优先），
 * 与 ONNX 输入 (1,128,800) 同布局。返回 0。 */
int asr_mel_compute(const asr_mel_t *m, const float *pcm, size_t n, float *out);

#ifdef __cplusplus
}
#endif

#endif /* ASR_MEL_H */
