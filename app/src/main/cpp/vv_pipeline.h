/* vv_pipeline.h — decoder chain (codes -> 24 kHz waveform). */
#ifndef VV_PIPELINE_H
#define VV_PIPELINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Loads dec_p1..dec_p5.bin from models_dir onto already-initialised QNN.
 * Returns 0 on success; see vv_last_error() otherwise. */
int  vv_decoder_init(const char *models_dir);
int  vv_decoder_ready(void);
void vv_decoder_release(void);

/* sem: 200 int32 (semantic codebook), aco: 11*200 int32.
 * out: 191996 float samples (8.0 s at 24 kHz). */
int  vv_decoder_decode(const int32_t *sem, const int32_t *aco,
                       float *out, int out_len);

const char *vv_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* VV_PIPELINE_H */
