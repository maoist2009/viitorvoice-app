#ifndef VV_MEL_H
#define VV_MEL_H

/*
 * Log-mel frontend of the dualcodec w2v semantic encoder, in fp32.
 *
 * The original ONNX graph computes this in fp32 with intermediates up to
 * ~1e10 (FFT magnitude^2), so it cannot live in the fp16 NPU graph. It is
 * reproduced here bit-for-bit, following the exported nodes:
 *
 *   audio[-1..1] * 32768 -> per-frame mean removal
 *   -> pre-emphasis 0.97 with the first sample edge-padded
 *   -> hann(400) -> zero pad to 512 -> rFFT -> |X|^2
 *   -> mel filterbank (257 x 80) -> clamp(1.1920929e-7) -> log
 */

#define VV_MEL_FRAMES 802
#define VV_MEL_BINS 80

#ifdef __cplusplus
extern "C" {
#endif

/* blob: 400 float32 window values followed by 257*80 float32 mel weights */
int vv_mel_init(const char *blob_path);

/* a16: 16 kHz samples in [-1,1]; out: 802*80 row-major log-mel */
void vv_mel_logmel(const float *a16, float *out);

#ifdef __cplusplus
}
#endif

#endif /* VV_MEL_H */
