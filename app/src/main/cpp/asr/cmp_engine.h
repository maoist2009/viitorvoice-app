/* cmp_engine.h — CAM++ 声纹（QNN HTP）。
 *
 * 图：campplus.bin
 *   feats (1,150,80) f32   Kaldi FBank + 逐句 CMN（1.5 s，不足补零）
 *   -> emb (1,192,1) f32   192 维说话人 embedding（未归一化）
 *
 * 前端（FBank/CMN）在 CPU，见 cmp_frontend.h；图里只有网络本体。
 */
#ifndef CMP_ENGINE_H
#define CMP_ENGINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CMP_EMB_DIM 192

int  cmp_engine_init(const char *model_dir, const char *lib_dir);
void cmp_engine_shutdown(void);
const char *cmp_engine_error(void);
const char *cmp_timings(void);

/* wav（任意采样率，内部重采样到 16k）→ embedding。cap 是 emb 的 float 容量。 */
int cmp_engine_embed_wav(const char *wav_path, float *emb, int cap);

/* 直接喂 PCM [-1,1] @16 kHz → embedding（切片场景用这个）。 */
int cmp_engine_embed_pcm(const float *pcm, size_t n, float *emb, int cap);

#ifdef __cplusplus
}
#endif

#endif /* CMP_ENGINE_H */
