/* vv_nar.h — the NAR half of ViiTorVoice (text -> 12 codebooks).
 *
 * Masked-discrete-diffusion loop, mirroring viitorvoice/llm/generation.py with
 * the graphs replaced by compiled context binaries:
 *
 *   llm.bin   inputs_embeds[1,256,1024] + attention_mask[1,1,256,256] -> hidden
 *   head.bin  hidden -> semantic_logits[1,256,16387] + acoustic[1,256,11297]
 *
 * The host does the embedding gather, the log-softmax scoring, the top-k cell
 * selection and the pause collapse; everything else is on the HTP.
 *
 * Two modes share the graphs:
 *   synthesis  vv_nar_prepare() + vv_nar_step()   (no CFG)
 *   local edit vv_nar_prepare_edit() + vv_nar_edit_step()  (CFG, masked span)
 */
#ifndef VV_NAR_H
#define VV_NAR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Loads llm.bin, head.bin and the three embedding tables from models_dir.
 * The decoder (vv_decoder_init) must be initialised separately. */
int  vv_nar_init(const char *models_dir);
int  vv_nar_ready(void);
void vv_nar_release(void);

/* Prompt window in tokens: 256 (10.2 s) or 512 (20.5 s). 512 requires the
 * linked llm.bin/head.bin that carry both shapes. Must be set before a job. */
int  vv_nar_set_window(int seq);
int  vv_nar_window(void);
int  vv_nar_max_window(void);
const char *vv_nar_graphs(void);

/* cond_ids: the prompt token ids (style + text), already tokenised.
 * frames:   number of audio frames to generate (200 = 8 s at 25 Hz).
 * Returns 0 on success. cond_len + frames must be <= 256. */
int  vv_nar_prepare(const int32_t *cond_ids, int cond_len, int frames);

/* One diffusion step. Returns the number of cells filled, or -1 on error. */
int  vv_nar_step(int step_index);

/* Effective diffusion step count (models/nar_steps, default 32). */
int  vv_nar_steps(void);

/* Structural filter + the 5-graph decoder + wav. "OK <dec_ms>|<wav_ms>". */
int  vv_nar_finish(float *out_audio, int out_len);

const int32_t *vv_nar_tokens(void);
int  vv_nar_remaining(void);
const char *vv_nar_error(void);

/* Host/NPU split of the last job, accumulated over its 32 steps:
 * "gather .. | llm .. | head .. | mask .. | acc .. | score .. | select .. ms".
 * Timings are cumulative and reset at step 0. DEBUG harness only. */
const char *vv_nar_timing(void);

/* ------------------------------------------------------------ local edit */
#define VV_NAR_MAX_FRAMES 480

/* cond_ids: prompt ids for the *new* full text (style + text).
 * target:   12*T existing tokens (row-major, row 0 semantic).
 * editable: T bytes, 1 = regenerate this frame.
 * guidance: CFG scale (2.0 matches the reference demo). */
int  vv_nar_prepare_edit(const int32_t *cond_ids, int cond_len,
                         const int32_t *target, const uint8_t *editable,
                         int frames, float guidance);

/* -------------------------------------------------- unified CFG job (v2)
 *
 * Mirrors generation.py `_generate_iterative` / `_generate_edit_iterative_single`
 * including the optional reference audio prompt and the emotion / NVV CFG
 * branches:
 *
 *   guided = (1 + cfg + sum(aux)) * logp_c - cfg * logp_u
 *            - sum_i aux_i * logp_aux_i        (then log_softmax)
 *
 * Prompt layout of every branch (cloning and editing share it):
 *
 *   [text ids][ref audio frames][separator][target frames]
 *
 * The conditional branch carries the style + text prompt; the unconditional
 * branch is the target window alone (audio starts at position 0) and is only
 * built when cfg != 0. Auxiliary branches are full prompts with the emotion /
 * NVV tags stripped, each with its own training-free guidance scale.
 *
 * ref_tokens: 12*ref_frames existing tokens, or NULL for no reference audio.
 * target:     12*frames starting tokens, or NULL to mask every frame.
 * editable:   frames bytes, or NULL to allow every frame to update.
 */
int  vv_nar_prepare_cfg(const int32_t *cond_ids, int cond_len,
                        const int32_t *const *aux_ids, const int *aux_lens,
                        const float *aux_scales, int n_aux,
                        const int32_t *ref_tokens, int ref_frames,
                        const int32_t *target, const uint8_t *editable,
                        int frames, float cfg);

int  vv_nar_cfg_step(int step_index);
const int32_t *vv_nar_cfg_tokens(void);
int  vv_nar_cfg_frames(void);
int  vv_nar_cfg_remaining(void);
int  vv_nar_cfg_prompt_len(void);   /* text + ref + separator */

/* debug hooks used by the headless harness only */
void vv_nar_set_diag_path(const char *path);
void vv_nar_set_weights(float wc, float wu);

/* One CFG step (two llm + two head runs). Returns cells filled or -1. */
int  vv_nar_edit_step(int step_index);

const int32_t *vv_nar_edit_tokens(void);
int  vv_nar_edit_frames(void);
int  vv_nar_edit_remaining(void);

/* Decodes any 12*frames token buffer: structural frames are dropped and the
 * rest is padded to the compiled 200-frame window.  Returns the number of
 * samples written (frames_kept * 960) or -1 on error. */
int  vv_nar_decode(const int32_t *tokens, int frames, float *out, int out_len);

#ifdef __cplusplus
}
#endif

#endif /* VV_NAR_H */
