/* vv_encode.h — audio -> 12 codebooks (the DualCodec encoder half).
 *
 * The semantic frontend (mel + W2V-BERT 2.0 + stats + pool) runs as one ONNX
 * graph exported from the python reference (see export_w2v_front8s.py), the
 * codec core (DualCodec encode_core) is the official ONNX export.  Both run on
 * the CPU through ONNX Runtime; the NPU is left for the llm / decoder.
 */
#ifndef VV_ENCODE_H
#define VV_ENCODE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VV_ENC_FRAMES   200      /* 8 s at 25 Hz, the compiled llm window */
#define VV_ENC_SR24     24000
#define VV_ENC_SR16     16000

/* Loads w2v_front8s.onnx (+ .onnx.data) and dualcodec_encode_core_30s.onnx. */
int  vv_encode_init(const char *models_dir);
int  vv_encode_ready(void);
void vv_encode_release(void);
const char *vv_encode_error(void);

/* pcm24k: mono float samples in [-1,1] at 24 kHz.
 * tokens: 12*VV_ENC_FRAMES int32, row 0 semantic, rows 1..11 acoustic.
 * *frames: number of real frames (<= VV_ENC_FRAMES) for the clip. */
int  vv_encode(const float *pcm24k, int n_samples, int32_t *tokens, int *frames);

/* Generic windowed-sinc resampler (used by the recorder / file import). */
int  vv_resample(const float *in, int n_in, int sr_in, int sr_out,
                 float *out, int out_cap, int *n_out);

#ifdef __cplusplus
}
#endif

#endif /* VV_ENCODE_H */
