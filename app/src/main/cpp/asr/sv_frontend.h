/* sv_frontend.h — SenseVoice-Small 的 kaldi fbank + LFR + CMVN 前端（CPU）。
 *
 * 忠实复刻 funasr 1.4.24 / tools/sensevoice/export_sensevoice.py 的数值路径：
 *   x = pcm*32768 -> 400 点帧(25ms/10ms) -> 去直流 -> preemph 0.97
 *   -> hamming(0.54-0.46cos) -> rfft(512) -> 功率谱 -> kaldi 三角 mel(80)
 *   -> log(max(·, 1.1921e-07))
 *   -> LFR(7,6)（左补 3 帧、尾部按 funasr 公式补齐）
 *   -> CMVN: (x + shift) * rescale
 *   -> 补零到 500 帧 + mask
 */
#ifndef SV_FRONTEND_H
#define SV_FRONTEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SV_SR        16000
#define SV_FRAME     400          /* 25 ms */
#define SV_HOP       160          /* 10 ms */
#define SV_NFFT      512
#define SV_NBINS     257          /* 512/2+1 */
#define SV_N_MELS    80
#define SV_LFR_M     7
#define SV_LFR_N     6
#define SV_DIM       560          /* 80 * 7 */
#define SV_T         500          /* 固定 30 s */
#define SV_EPS       1.1921e-07

typedef struct {
  double banks[SV_N_MELS * SV_NBINS];   /* 行主序 (80,257) */
  double win[SV_FRAME];
  double twr[SV_NFFT / 2], twi[SV_NFFT / 2];
  float  cmvn[2 * SV_DIM];              /* [0..559] shift, [560..1119] rescale */
} sv_front_t;

/* cmvn_path = sv_cmvn.bin（2x560 f32）。返回 0 成功。 */
int sv_front_init(sv_front_t *f, const char *cmvn_path);

/* pcm: 16 kHz 单声道 float [-1,1]，n 个样本。
 * feats 必须是 SV_T*SV_DIM 个 float；mask 是 SV_T 个 float。
 * *n_valid 返回有效 LFR 帧数（= mask 里 1 的个数）。返回 0 成功。 */
int sv_front_compute(const sv_front_t *f, const float *pcm, size_t n,
                     float *feats, float *mask, int *n_valid);

#ifdef __cplusplus
}
#endif

#endif /* SV_FRONTEND_H */
