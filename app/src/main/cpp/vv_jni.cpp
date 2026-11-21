/* vv_jni.cpp — JNI surface for the ViiTorVoice demo engine. */
#include <android/log.h>
#include <jni.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#include <string>

#include "mstore.h"
#include "qnn_executor.h"
#include "vv_encode.h"
#include "vv_nar.h"
#include "vv_pipeline.h"

#define TAG "vv_jni"
#define DEC_TOTAL 191996
#define DEC_SR    24000

static std::string jstr(JNIEnv *env, jstring s) {
  if (!s) return std::string();
  const char *c = env->GetStringUTFChars(s, nullptr);
  std::string out(c ? c : "");
  if (c) env->ReleaseStringUTFChars(s, c);
  return out;
}

static double now_ms() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* 16-bit PCM WAV, mono. */
static int write_wav(const char *path, const float *samples, int n, int sr) {
  FILE *f = fopen(path, "wb");
  if (!f) return -1;
  const uint32_t data_bytes = (uint32_t)n * 2;
  uint8_t hdr[44];
  memcpy(hdr, "RIFF", 4);
  uint32_t riff = 36 + data_bytes;
  memcpy(hdr + 4, &riff, 4);
  memcpy(hdr + 8, "WAVEfmt ", 8);
  uint32_t fmt_len = 16;
  memcpy(hdr + 16, &fmt_len, 4);
  uint16_t pcm = 1, ch = 1, bits = 16, align = 2;
  uint32_t rate = (uint32_t)sr, byte_rate = (uint32_t)sr * 2;
  memcpy(hdr + 20, &pcm, 2);
  memcpy(hdr + 22, &ch, 2);
  memcpy(hdr + 24, &rate, 4);
  memcpy(hdr + 28, &byte_rate, 4);
  memcpy(hdr + 32, &align, 2);
  memcpy(hdr + 34, &bits, 2);
  memcpy(hdr + 36, "data", 4);
  memcpy(hdr + 40, &data_bytes, 4);
  fwrite(hdr, 1, 44, f);

  const int kBlock = 4096;
  int16_t *buf = (int16_t *)malloc(sizeof(int16_t) * kBlock);
  if (!buf) { fclose(f); return -1; }
  for (int i = 0; i < n; i += kBlock) {
    int m = n - i < kBlock ? n - i : kBlock;
    for (int j = 0; j < m; j++) {
      float v = samples[i + j];
      if (v > 1.0f) v = 1.0f;
      if (v < -1.0f) v = -1.0f;
      buf[j] = (int16_t)lrintf(v * 32767.0f);
    }
    fwrite(buf, sizeof(int16_t), (size_t)m, f);
  }
  free(buf);
  fclose(f);
  return 0;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeVersion(JNIEnv *env, jclass) {
  return env->NewStringUTF("vv_qnn 0.1 · QNN 2.51.40");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeInit(JNIEnv *env, jclass, jstring libDir,
                                            jstring skelDir, jstring modelsDir) {
  std::string lib = jstr(env, libDir);
  std::string skel = jstr(env, skelDir);
  std::string models = jstr(env, modelsDir);

  double t0 = now_ms();
  if (qnn_init(lib.c_str(), skel.c_str()) != 0)
    return env->NewStringUTF(qnn_last_error());
  double t1 = now_ms();
  if (vv_decoder_init(models.c_str()) != 0)
    return env->NewStringUTF(vv_last_error());
  double t2 = now_ms();
  __android_log_print(ANDROID_LOG_INFO, TAG, "init backend %.0f ms, decoder %.0f ms",
                      t1 - t0, t2 - t1);
  char msg[256];
  snprintf(msg, sizeof(msg), "OK %.0f|%.0f", t1 - t0, t2 - t1);
  return env->NewStringUTF(msg);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeDecodeToWav(JNIEnv *env, jclass,
                                                   jintArray sem, jintArray aco,
                                                   jstring wavPath) {
  if (!vv_decoder_ready()) return env->NewStringUTF("decoder not loaded");
  jsize ns = env->GetArrayLength(sem);
  jsize na = env->GetArrayLength(aco);
  if (ns != 200 || na != 2200) return env->NewStringUTF("bad code shapes");

  jint *s = env->GetIntArrayElements(sem, nullptr);
  jint *a = env->GetIntArrayElements(aco, nullptr);
  float *audio = (float *)malloc(sizeof(float) * DEC_TOTAL);
  if (!audio || !s || !a) {
    free(audio);
    if (s) env->ReleaseIntArrayElements(sem, s, JNI_ABORT);
    if (a) env->ReleaseIntArrayElements(aco, a, JNI_ABORT);
    return env->NewStringUTF("out of memory");
  }

  double t0 = now_ms();
  int rc = vv_decoder_decode((const int32_t *)s, (const int32_t *)a, audio, DEC_TOTAL);
  double t1 = now_ms();
  env->ReleaseIntArrayElements(sem, s, JNI_ABORT);
  env->ReleaseIntArrayElements(aco, a, JNI_ABORT);

  if (rc != 0) {
    free(audio);
    return env->NewStringUTF(vv_last_error());
  }

  std::string path = jstr(env, wavPath);
  int wr = write_wav(path.c_str(), audio, DEC_TOTAL, DEC_SR);
  double t2 = now_ms();
  free(audio);
  if (wr != 0) return env->NewStringUTF("wav write failed");

  char msg[256];
  snprintf(msg, sizeof(msg), "OK %.0f|%.0f", t1 - t0, t2 - t1);
  return env->NewStringUTF(msg);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeTimings(JNIEnv *env, jclass) {
  return env->NewStringUTF(qnn_timings_json());
}

/* ------------------------------------------------- shared model library --
 * MLP/1 delivers model blobs as read-only fds from a ContentProvider, so the
 * host's private paths never reach this process. Each fd is bound to a path
 * suffix; every later open of "<anything>/<name>" resolves to it. The engine
 * code above keeps calling qnn_load(dir + "/llm.bin") and friends unchanged.
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeBindModelFds(JNIEnv *env, jclass,
                                                    jobjectArray suffixes,
                                                    jintArray fds) {
  const jsize n = suffixes ? env->GetArrayLength(suffixes) : 0;
  const jsize m = fds ? env->GetArrayLength(fds) : 0;
  if (n <= 0 || n != m) return -1;
  jint *raw = env->GetIntArrayElements(fds, nullptr);
  int bound = 0;
  for (jsize i = 0; i < n; i++) {
    auto js = (jstring)env->GetObjectArrayElement(suffixes, i);
    const std::string suffix = jstr(env, js);
    if (js) env->DeleteLocalRef(js);
    if (mstore_bind(suffix.c_str(), raw[i]) == 0) {
      bound++;  /* ownership moved to the table */
    } else {
      close(raw[i]);
    }
  }
  env->ReleaseIntArrayElements(fds, raw, JNI_ABORT);
  __android_log_print(ANDROID_LOG_INFO, TAG, "bound %d/%d model fds", bound, (int)n);
  return bound;
}

extern "C" JNIEXPORT void JNICALL
Java_com_viitorvoice_demo_Native_nativeClearModelFds(JNIEnv *, jclass) {
  mstore_clear();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeBoundModelFds(JNIEnv *, jclass) {
  return mstore_bound();
}

/* ------------------------------------------------------------ NAR pipeline */

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeNarInit(JNIEnv *env, jclass, jstring modelsDir) {
  std::string dir = jstr(env, modelsDir);
  double t0 = now_ms();
  if (vv_nar_init(dir.c_str()) != 0) return env->NewStringUTF(vv_nar_error());
  char msg[512];
  snprintf(msg, sizeof(msg), "OK %.0f %s", now_ms() - t0, vv_nar_graphs());
  return env->NewStringUTF(msg);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeNarPrepare(JNIEnv *env, jclass, jintArray ids,
                                                  jint condLen, jint frames) {
  jsize n = env->GetArrayLength(ids);
  if (n != condLen) return env->NewStringUTF("prompt length mismatch");
  jint *v = env->GetIntArrayElements(ids, nullptr);
  int rc = vv_nar_prepare((const int32_t *)v, condLen, frames);
  env->ReleaseIntArrayElements(ids, v, JNI_ABORT);
  if (rc != 0) return env->NewStringUTF(vv_nar_error());
  return env->NewStringUTF("OK");
}

extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeNarSetWindow(JNIEnv *, jclass, jint seq) {
  return vv_nar_set_window(seq);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeNarWindow(JNIEnv *, jclass) {
  return vv_nar_window();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeNarMaxWindow(JNIEnv *, jclass) {
  return vv_nar_max_window();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeNarSteps(JNIEnv *, jclass) {
  return vv_nar_steps();
}

extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeNarStep(JNIEnv *, jclass, jint step) {
  int k = vv_nar_step(step);
  return k;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeNarRemaining(JNIEnv *, jclass) {
  return vv_nar_remaining();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeNarError(JNIEnv *env, jclass) {
  return env->NewStringUTF(vv_nar_error());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeNarTiming(JNIEnv *env, jclass) {
  return env->NewStringUTF(vv_nar_timing());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeNarFinish(JNIEnv *env, jclass, jstring wavPath) {
  float *audio = (float *)malloc(sizeof(float) * DEC_TOTAL);
  if (!audio) return env->NewStringUTF("out of memory");
  double t0 = now_ms();
  int n = vv_nar_finish(audio, DEC_TOTAL);
  double t1 = now_ms();
  if (n <= 0) {
    free(audio);
    return env->NewStringUTF(vv_nar_error());
  }
  std::string path = jstr(env, wavPath);
  int wr = write_wav(path.c_str(), audio, n, DEC_SR);
  double t2 = now_ms();
  free(audio);
  if (wr != 0) return env->NewStringUTF("wav write failed");
  char msg[128];
  snprintf(msg, sizeof(msg), "OK %.0f|%.0f|%d", t1 - t0, t2 - t1, n);
  return env->NewStringUTF(msg);
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_viitorvoice_demo_Native_nativeNarTokens(JNIEnv *env, jclass, jint frames) {
  const int32_t *t = vv_nar_tokens();
  if (!t) return nullptr;
  jintArray out = env->NewIntArray(12 * frames);
  if (!out) return nullptr;
  env->SetIntArrayRegion(out, 0, 12 * frames, (const jint *)t);
  return out;
}

/* ------------------------------------------------------- local edit (CFG) */
extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeNarPrepareEdit(JNIEnv *env, jclass, jintArray ids,
                                                      jint condLen, jintArray target,
                                                      jbooleanArray editable, jint frames,
                                                      jfloat cfg) {
  if (env->GetArrayLength(ids) != condLen) return env->NewStringUTF("prompt length mismatch");
  if (env->GetArrayLength(target) != 12 * frames) return env->NewStringUTF("token length mismatch");
  if (env->GetArrayLength(editable) != frames) return env->NewStringUTF("mask length mismatch");
  jint *c = env->GetIntArrayElements(ids, nullptr);
  jint *t = env->GetIntArrayElements(target, nullptr);
  jboolean *m = env->GetBooleanArrayElements(editable, nullptr);
  uint8_t *mask = (uint8_t *)malloc((size_t)frames);
  for (int i = 0; mask && i < frames; i++) mask[i] = m[i] ? 1 : 0;
  int rc = mask ? vv_nar_prepare_edit((const int32_t *)c, condLen, (const int32_t *)t,
                                      mask, frames, cfg) : -1;
  free(mask);
  env->ReleaseIntArrayElements(ids, c, JNI_ABORT);
  env->ReleaseIntArrayElements(target, t, JNI_ABORT);
  env->ReleaseBooleanArrayElements(editable, m, JNI_ABORT);
  if (rc != 0) return env->NewStringUTF(vv_nar_error());
  return env->NewStringUTF("OK");
}

/* --------------------------------------------------- unified CFG job (v2) */
extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeNarPrepareCfg(
    JNIEnv *env, jclass, jintArray condIds, jint condLen,
    jobjectArray auxIds, jfloatArray auxScales,
    jintArray refTokens, jint refFrames,
    jintArray target, jbooleanArray editable, jint frames, jfloat cfg) {
  if (!condIds || env->GetArrayLength(condIds) != condLen) {
    return env->NewStringUTF("prompt length mismatch");
  }
  if (target && env->GetArrayLength(target) != 12 * frames) {
    return env->NewStringUTF("token length mismatch");
  }
  if (editable && env->GetArrayLength(editable) != frames) {
    return env->NewStringUTF("mask length mismatch");
  }
  if (refTokens && refFrames > 0 && env->GetArrayLength(refTokens) != 12 * refFrames) {
    return env->NewStringUTF("reference token length mismatch");
  }
  const int n_aux = auxIds ? env->GetArrayLength(auxIds) : 0;
  if (n_aux > 2) return env->NewStringUTF("too many aux branches");
  if (n_aux > 0 && (!auxScales || env->GetArrayLength(auxScales) < n_aux)) {
    return env->NewStringUTF("aux scales missing");
  }

  jint *cond = env->GetIntArrayElements(condIds, nullptr);
  jint *ref = refTokens ? env->GetIntArrayElements(refTokens, nullptr) : nullptr;
  jint *tgt = target ? env->GetIntArrayElements(target, nullptr) : nullptr;
  jboolean *ed = editable ? env->GetBooleanArrayElements(editable, nullptr) : nullptr;
  float *scales = n_aux > 0 ? (float *)malloc(sizeof(float) * n_aux) : nullptr;
  if (n_aux > 0 && !scales) {
    env->ReleaseIntArrayElements(condIds, cond, JNI_ABORT);
    if (ref) env->ReleaseIntArrayElements(refTokens, ref, JNI_ABORT);
    if (tgt) env->ReleaseIntArrayElements(target, tgt, JNI_ABORT);
    if (ed) env->ReleaseBooleanArrayElements(editable, ed, JNI_ABORT);
    return env->NewStringUTF("out of memory");
  }
  jint *aux_buf[2] = {nullptr, nullptr};
  jintArray aux_arr[2] = {nullptr, nullptr};
  const int32_t *aux_ptr[2] = {nullptr, nullptr};
  int aux_len[2] = {0, 0};
  if (auxScales && scales) env->GetFloatArrayRegion(auxScales, 0, n_aux, scales);
  for (int i = 0; i < n_aux; i++) {
    jintArray a = (jintArray)env->GetObjectArrayElement(auxIds, i);
    aux_arr[i] = a;
    aux_len[i] = a ? env->GetArrayLength(a) : 0;
    aux_buf[i] = a ? env->GetIntArrayElements(a, nullptr) : nullptr;
    aux_ptr[i] = (const int32_t *)aux_buf[i];
  }
  uint8_t *mask = nullptr;
  if (ed) {
    mask = (uint8_t *)malloc((size_t)frames);
    if (mask) {
      for (int i = 0; i < frames; i++) mask[i] = ed[i] ? 1 : 0;
    }
  }

  int rc = -1;
  if (ed && !mask) {
    __android_log_print(ANDROID_LOG_ERROR, TAG, "cfg: out of memory for mask");
  } else {
    rc = vv_nar_prepare_cfg((const int32_t *)cond, condLen, aux_ptr, aux_len,
                            scales, n_aux, (const int32_t *)ref, refFrames,
                            (const int32_t *)tgt, mask, frames, cfg);
  }

  free(scales);
  free(mask);
  for (int i = 0; i < n_aux; i++) {
    if (aux_buf[i]) env->ReleaseIntArrayElements(aux_arr[i], aux_buf[i], JNI_ABORT);
    if (aux_arr[i]) env->DeleteLocalRef(aux_arr[i]);
  }
  env->ReleaseIntArrayElements(condIds, cond, JNI_ABORT);
  if (ref) env->ReleaseIntArrayElements(refTokens, ref, JNI_ABORT);
  if (tgt) env->ReleaseIntArrayElements(target, tgt, JNI_ABORT);
  if (ed) env->ReleaseBooleanArrayElements(editable, ed, JNI_ABORT);

  if (rc != 0) return env->NewStringUTF(vv_nar_error());
  return env->NewStringUTF("OK");
}

extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeNarEditStep(JNIEnv *, jclass, jint step) {
  return vv_nar_edit_step(step);
}

extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeNarEditRemaining(JNIEnv *, jclass) {
  return vv_nar_edit_remaining();
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_viitorvoice_demo_Native_nativeNarEditTokens(JNIEnv *env, jclass) {
  const int32_t *t = vv_nar_edit_tokens();
  const int frames = vv_nar_edit_frames();
  if (!t || frames <= 0) return nullptr;
  jintArray out = env->NewIntArray(12 * frames);
  if (!out) return nullptr;
  env->SetIntArrayRegion(out, 0, 12 * frames, (const jint *)t);
  return out;
}

/* ------------------------------------------------------------- encoder */
extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeEncodeInit(JNIEnv *env, jclass, jstring modelsDir) {
  std::string dir = jstr(env, modelsDir);
  double t0 = now_ms();
  if (vv_encode_init(dir.c_str()) != 0) return env->NewStringUTF(vv_encode_error());
  char msg[512];
  snprintf(msg, sizeof(msg), "OK %.0f %s", now_ms() - t0, vv_nar_graphs());
  return env->NewStringUTF(msg);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeEncodeRelease(JNIEnv *, jclass) {
  vv_encode_release();
  return nullptr;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeEncode(JNIEnv *env, jclass, jfloatArray pcm,
                                              jint n, jintArray tokensOut) {
  if (env->GetArrayLength(pcm) < n) return -1;
  if (env->GetArrayLength(tokensOut) < 12 * VV_ENC_FRAMES) return -1;
  jfloat *a = env->GetFloatArrayElements(pcm, nullptr);
  float *pcm_mut = a;
  int32_t *tok = (int32_t *)malloc(sizeof(int32_t) * 12 * VV_ENC_FRAMES);
  int frames = 0;
  int rc = tok ? vv_encode((const float *)pcm_mut, n, tok, &frames) : -1;
  if (rc == 0) env->SetIntArrayRegion(tokensOut, 0, 12 * VV_ENC_FRAMES, (const jint *)tok);
  free(tok);
  env->ReleaseFloatArrayElements(pcm, a, JNI_ABORT);
  if (rc != 0) __android_log_print(ANDROID_LOG_ERROR, TAG, "encode: %s", vv_encode_error());
  return rc == 0 ? frames : -1;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_Native_nativeEncodeError(JNIEnv *env, jclass) {
  return env->NewStringUTF(vv_encode_error());
}

/* Arbitrary resample (import path): returns the number of output samples. */
extern "C" JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_Native_nativeResample(JNIEnv *env, jclass, jfloatArray in,
                                                jint nIn, jint srIn, jint srOut,
                                                jfloatArray out) {
  if (env->GetArrayLength(in) < nIn) return -1;
  const int cap = env->GetArrayLength(out);
  jfloat *src = env->GetFloatArrayElements(in, nullptr);
  float *dst = (float *)malloc(sizeof(float) * cap);
  int nOut = 0;
  int rc = dst ? vv_resample((const float *)src, nIn, srIn, srOut, dst, cap, &nOut) : -1;
  if (rc == 0) env->SetFloatArrayRegion(out, 0, nOut, (const jfloat *)dst);
  free(dst);
  env->ReleaseFloatArrayElements(in, src, JNI_ABORT);
  return rc == 0 ? nOut : -1;
}

extern "C" JNIEXPORT void JNICALL
Java_com_viitorvoice_demo_Native_nativeNarSetDiagPath(JNIEnv *env, jclass, jstring path) {
  if (!path) { vv_nar_set_diag_path(nullptr); return; }
  const char *p = env->GetStringUTFChars(path, nullptr);
  vv_nar_set_diag_path(p);
  env->ReleaseStringUTFChars(path, p);
}

extern "C" JNIEXPORT void JNICALL
Java_com_viitorvoice_demo_Native_nativeNarSetWeights(JNIEnv *, jclass, jfloat wc, jfloat wu) {
  vv_nar_set_weights(wc, wu);
}
