/* aligner_engine.h — Qwen3-ForcedAligner-0.6B 的端侧管线（QNN HTP）。 */
#ifndef ALIGNER_ENGINE_H
#define ALIGNER_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

/* model_dir 需含 enc.bin / dec.bin / token_embeds.bin。返回 0 成功。 */
int  aligner_init(const char *model_dir, const char *lib_dir);
void aligner_shutdown(void);
const char *aligner_error(void);

/* ids 是完整的 aligner prompt token 序列（含 1 个 <|audio_pad|>=151676，
 * 运行时展开成 104 个音频 token）。out_ms 收到每个 <timestamp> 位置的时间戳，
 * 单位毫秒。返回 0 成功。 */
int aligner_run_wav(const char *wav_path, const int *ids, int n_ids,
                    float *out_ms, int cap, int *n_out);

const char *aligner_timings(void);

/* 诊断：只加载 dec 图，跑一次固定输入（fp16 raw），把 logits 写到 out_path。 */
int  aligner_dec_dbg(const char *model_dir, const char *lib_dir,
                     const char *in_path, const char *out_path);

#ifdef __cplusplus
}
#endif

#endif /* ALIGNER_ENGINE_H */
