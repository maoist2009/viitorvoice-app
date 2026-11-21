/* vad_engine.h — Silero VAD（QNN HTP，v73/v79 context binary）。 */
#ifndef ASR_VAD_ENGINE_H
#define ASR_VAD_ENGINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VAD_CHUNK 512       /* 16 kHz / 32 ms */
#define VAD_STATE 256       /* 2 × 1 × 128（h, c） */
#define VAD_CTX   64        /* 上一帧尾 64 点 */

/* 加载 model_dir/silero_vad.bin，失败返回 -1（错误见 vad_engine_error）。 */
int vad_engine_init(const char *model_dir, const char *lib_dir);

/* 清 LSTM 状态与上下文（每段新音频前调用）。 */
void vad_engine_reset(void);

/* 一张图展开几帧（init 之后才有意义）。 */
int vad_engine_frames(void);

/* 跑一块（frames*512 点）；probs 写每帧语音概率。 */
int vad_engine_run(const float *pcm, int n_frames, float *probs);

void vad_engine_shutdown(void);

const char *vad_engine_error(void);
const char *vad_engine_timings(void);

#ifdef __cplusplus
}
#endif

#endif /* ASR_VAD_ENGINE_H */
