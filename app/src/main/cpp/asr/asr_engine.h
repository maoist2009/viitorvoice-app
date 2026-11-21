/* asr_engine.h — Qwen3-ASR 1.7B 的端侧推理管线（QNN HTP）。
 *
 * 图的划分（与 tools/asr/export.py 一致）：
 *   enc                : mel(1,128,800)        -> audio_embeds(1,104,2048)
 *   dec_p{0,1,2}_prefill: embeds(1,128,2048) + mask + rope + 各层 KV
 *                          -> hidden(1,128,2048) + 新的 KV
 *   dec_p{0,1,2}_decode : 同上去掉时间维（T=1），KV 右移一格
 *   lm_head             : hidden(1,1,2048)      -> logits(1,1,151936)
 *
 * KV cache 是右对齐静态布局（ctx=512，prefill 占最后 128 格），decode 每步
 * 左移一格再追加；可见性靠外部传入的加性 mask（-50 / 0）。
 */
#ifndef ASR_ENGINE_H
#define ASR_ENGINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 上下文几何（与导出脚本绑定，改动必须两边同步） */
#define ASR_CTX        512
#define ASR_TPRE       118     /* 图内行数 = 119 个 prompt token 去掉 sink 行 */
#define ASR_HEAD       119     /* 真实 prompt 长度：9 + 104 + 6（第一个 token 是 sink） */
#define ASR_PAD        0       /* 不再补零行：sink 的 K/V 由外部注入 KV cache */
#define ASR_NL         28
#define ASR_NKV        8
#define ASR_HD         128
#define ASR_HID        2048
#define ASR_VOCAB      151936
#define ASR_AUDIO_TOK  104
#define ASR_MAX_NEW    96      /* 512 - 119 - 9 格余量的保守上限 */

int  asr_engine_init(const char *model_dir, const char *lib_dir);
void asr_engine_shutdown(void);
const char *asr_engine_error(void);

/* 只加载 enc 图跑一个 mel 窗口，用于 DLC 到手后的快速比对。
 * mel_path / out_path 都是裸 fp16（128*800 / 104*2048）。 */
int  asr_probe_enc(const char *lib_dir, const char *enc_bin, const char *mel_path,
                   const char *out_path);

/* 跑一次识别：wav（16 kHz 单声道；其他采样率线性重采样）前 8 秒。
 * 生成的 token id 写进 ids（不含 prompt），返回 0 表示成功。 */
int asr_engine_run_wav(const char *wav_path, int max_new, int *ids, int cap,
                       int *n_out);

/* 预加载常驻图（enc / 3 张 decode / lm_head）。启动时调一次，第一次识别的
 * wall 时间就从 ~9 s 掉到 ~4 s。返回 0 表示成功。 */
int asr_engine_preload(void);

/* Per-graph graphExecute wall-time stats (JSON), for headless benchmarking. */
const char *asr_timings(void);

#ifdef __cplusplus
}
#endif

#endif /* ASR_ENGINE_H */
