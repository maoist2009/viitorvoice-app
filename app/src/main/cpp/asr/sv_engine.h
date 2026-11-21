/* sv_engine.h — SenseVoice-Small 端侧管线（QNN HTP）。
 *
 * 图：sv.bin
 *   feats  (1,500,560) f32   已做 fbank+LFR+CMVN 的特征（30 s，不足补零）
 *   mask   (1,500)     f32   1=有效帧
 *   prefix (1,4,560)   f32   embed.weight[[0,1,2,15]]（auto/event/emo/woitn）
 *   -> logits (1,504,25055) f32  CTC 后验；前 4 帧是 rich 标签
 *                                 （语种/情感/事件/ITN），第 5 帧起是正文
 *
 * 旧图（2026-11-08 编的）输出只有 500 帧、没有 rich 标签；本引擎两种都能跑，
 * 按图的总帧数判断，旧图上 rich_out 全填 -1。
 *
 * 模型目录里除 sv.bin 外还需要：
 *   sv_cmvn.bin    2x560 f32
 *   sv_prefix.bin  4x560 f32
 *   sv_tokens.bin  'SVTK' | u32 n | u32 off[n+1] | bytes
 */
#ifndef SV_ENGINE_H
#define SV_ENGINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int  sv_engine_init(const char *model_dir, const char *lib_dir);
void sv_engine_shutdown(void);
const char *sv_engine_error(void);

/* 跑一次识别。text 写入 UTF-8（NUL 结尾），cap 是字节容量。
 * *n_tok 返回 CTC 输出 token 数（含被折叠前的有效帧数不适用，这里给去重后的数量）。
 * rich_out（可为 NULL）收 4 个 rich 标签的 token id（语言/情感/事件/ITN）；
 * 旧图上全填 -1。标签本身不进 text。 */
int sv_engine_run_wav(const char *wav_path, char *text, size_t cap, int *n_tok,
                      int *rich_out);

/* 图加载/执行的 wall-time 统计（JSON）。 */
const char *sv_timings(void);

/* token id -> UTF-8 文本（越界返回空串）；JNI 用它把 rich id 翻成标签串。 */
const char *sv_engine_token(int id);

#ifdef __cplusplus
}
#endif

#endif /* SV_ENGINE_H */
