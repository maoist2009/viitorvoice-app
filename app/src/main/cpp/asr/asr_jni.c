/* asr_jni.c — Kotlin <-> asr_engine 的桥。
 *
 * init() 返回空串表示成功，否则是错误文本（不抛异常，方便 UI 直接显示）；
 * runWav() 出错抛 IllegalStateException，文本同 asr_engine_error()。
 */
#include <jni.h>
#include <stdlib.h>
#include <string.h>

#include "asr_engine.h"
#include "cmp_engine.h"
#include "sv_engine.h"

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_init(JNIEnv *env, jclass clazz, jstring model_dir,
                                      jstring lib_dir) {
  (void)clazz;
  const char *md = (*env)->GetStringUTFChars(env, model_dir, NULL);
  const char *ld = (*env)->GetStringUTFChars(env, lib_dir, NULL);
  int rc = asr_engine_init(md, ld);
  (*env)->ReleaseStringUTFChars(env, model_dir, md);
  (*env)->ReleaseStringUTFChars(env, lib_dir, ld);
  return (*env)->NewStringUTF(env, rc ? asr_engine_error() : "");
}

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_preload(JNIEnv *env, jclass clazz) {
  (void)clazz;
  int rc = asr_engine_preload();
  return (*env)->NewStringUTF(env, rc ? asr_engine_error() : "");
}

JNIEXPORT jintArray JNICALL
Java_com_viitorvoice_demo_AsrNative_runWav(JNIEnv *env, jclass clazz, jstring wav_path,
                                        jint max_new) {
  (void)clazz;
  const char *path = (*env)->GetStringUTFChars(env, wav_path, NULL);
  int cap = max_new > 0 ? max_new : ASR_MAX_NEW;
  int *ids = (int *)malloc(sizeof(int) * (size_t)cap);
  if (!ids) {
    (*env)->ReleaseStringUTFChars(env, wav_path, path);
    jclass c = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
    if (c) (*env)->ThrowNew(env, c, "token buffer");
    return NULL;
  }
  int n = 0;
  int rc = asr_engine_run_wav(path, max_new, ids, cap, &n);
  (*env)->ReleaseStringUTFChars(env, wav_path, path);
  if (rc) {
    free(ids);
    jclass c = (*env)->FindClass(env, "java/lang/IllegalStateException");
    if (c) (*env)->ThrowNew(env, c, asr_engine_error());
    return NULL;
  }
  jintArray out = (*env)->NewIntArray(env, n);
  if (out) (*env)->SetIntArrayRegion(env, out, 0, n, (const jint *)ids);
  free(ids);
  return out;
}

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_probeEnc(JNIEnv *env, jclass clazz, jstring lib_dir,
                                          jstring enc_bin, jstring mel_path,
                                          jstring out_path) {
  (void)clazz;
  const char *ld = (*env)->GetStringUTFChars(env, lib_dir, NULL);
  const char *eb = (*env)->GetStringUTFChars(env, enc_bin, NULL);
  const char *mp = (*env)->GetStringUTFChars(env, mel_path, NULL);
  const char *op = (*env)->GetStringUTFChars(env, out_path, NULL);
  int rc = asr_probe_enc(ld, eb, mp, op);
  (*env)->ReleaseStringUTFChars(env, lib_dir, ld);
  (*env)->ReleaseStringUTFChars(env, enc_bin, eb);
  (*env)->ReleaseStringUTFChars(env, mel_path, mp);
  (*env)->ReleaseStringUTFChars(env, out_path, op);
  return (*env)->NewStringUTF(env, rc ? asr_engine_error() : "");
}

JNIEXPORT void JNICALL
Java_com_viitorvoice_demo_AsrNative_shutdown(JNIEnv *env, jclass clazz) {
  (void)env;
  (void)clazz;
  asr_engine_shutdown();
}

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_lastError(JNIEnv *env, jclass clazz) {
  (void)clazz;
  return (*env)->NewStringUTF(env, asr_engine_error());
}

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_timings(JNIEnv *env, jclass clazz) {
  return (*env)->NewStringUTF(env, asr_timings());
}

/* ---------------- SenseVoice-Small ---------------- */

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_svInit(JNIEnv *env, jclass clazz, jstring model_dir,
                                        jstring lib_dir) {
  (void)clazz;
  const char *md = (*env)->GetStringUTFChars(env, model_dir, NULL);
  const char *ld = (*env)->GetStringUTFChars(env, lib_dir, NULL);
  int rc = sv_engine_init(md, ld);
  (*env)->ReleaseStringUTFChars(env, model_dir, md);
  (*env)->ReleaseStringUTFChars(env, lib_dir, ld);
  return (*env)->NewStringUTF(env, rc ? sv_engine_error() : "");
}

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_svRun(JNIEnv *env, jclass clazz, jstring wav_path) {
  (void)clazz;
  const char *path = (*env)->GetStringUTFChars(env, wav_path, NULL);
  char text[8192];
  int nt = 0;
  int rc = sv_engine_run_wav(path, text, sizeof(text), &nt, NULL);
  (*env)->ReleaseStringUTFChars(env, wav_path, path);
  if (rc) {
    jclass c = (*env)->FindClass(env, "java/lang/IllegalStateException");
    if (c) (*env)->ThrowNew(env, c, sv_engine_error());
    return NULL;
  }
  return (*env)->NewStringUTF(env, text);
}

/* 同一次前向，额外把 4 个 rich 标签（语言/情感/事件/ITN）的**字符串**写进
 * tags[0..3]。旧图（2026-11-08 那版，没有 rich 帧）四个都是 null。 */
JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_svRunRich(JNIEnv *env, jclass clazz, jstring wav_path,
                                           jobjectArray tags) {
  (void)clazz;
  const char *path = (*env)->GetStringUTFChars(env, wav_path, NULL);
  char text[8192];
  int nt = 0;
  int ids[4] = {-1, -1, -1, -1};
  int rc = sv_engine_run_wav(path, text, sizeof(text), &nt, ids);
  (*env)->ReleaseStringUTFChars(env, wav_path, path);
  if (rc) {
    jclass c = (*env)->FindClass(env, "java/lang/IllegalStateException");
    if (c) (*env)->ThrowNew(env, c, sv_engine_error());
    return NULL;
  }
  if (tags && (*env)->GetArrayLength(env, tags) >= 4) {
    for (int k = 0; k < 4; k++) {
      if (ids[k] < 0) continue;
      (*env)->SetObjectArrayElement(env, tags, k,
                                    (*env)->NewStringUTF(env, sv_engine_token(ids[k])));
    }
  }
  return (*env)->NewStringUTF(env, text);
}

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_svTimings(JNIEnv *env, jclass clazz) {
  return (*env)->NewStringUTF(env, sv_timings());
}

/* 释放 SenseVoice 常驻图（流水线串行换模型用，和 cmpShutdown 同理）。 */
JNIEXPORT void JNICALL
Java_com_viitorvoice_demo_AsrNative_svShutdown(JNIEnv *env, jclass clazz) {
  (void)env; (void)clazz;
  sv_engine_shutdown();
}

/* ---------------- CAM++ 声纹 ---------------- */

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_cmpInit(JNIEnv *env, jclass clazz, jstring model_dir,
                                         jstring lib_dir) {
  (void)clazz;
  const char *md = (*env)->GetStringUTFChars(env, model_dir, NULL);
  const char *ld = (*env)->GetStringUTFChars(env, lib_dir, NULL);
  int rc = cmp_engine_init(md, ld);
  (*env)->ReleaseStringUTFChars(env, model_dir, md);
  (*env)->ReleaseStringUTFChars(env, lib_dir, ld);
  return (*env)->NewStringUTF(env, rc ? cmp_engine_error() : "");
}

JNIEXPORT jfloatArray JNICALL
Java_com_viitorvoice_demo_AsrNative_cmpRun(JNIEnv *env, jclass clazz, jstring wav_path) {
  (void)clazz;
  const char *path = (*env)->GetStringUTFChars(env, wav_path, NULL);
  float emb[CMP_EMB_DIM];
  int rc = cmp_engine_embed_wav(path, emb, CMP_EMB_DIM);
  (*env)->ReleaseStringUTFChars(env, wav_path, path);
  if (rc) {
    jclass c = (*env)->FindClass(env, "java/lang/IllegalStateException");
    if (c) (*env)->ThrowNew(env, c, cmp_engine_error());
    return NULL;
  }
  jfloatArray out = (*env)->NewFloatArray(env, CMP_EMB_DIM);
  if (out) (*env)->SetFloatArrayRegion(env, out, 0, CMP_EMB_DIM, emb);
  return out;
}

JNIEXPORT jfloatArray JNICALL
Java_com_viitorvoice_demo_AsrNative_cmpRunPcm(JNIEnv *env, jclass clazz, jfloatArray pcm) {
  (void)clazz;
  jsize n = (*env)->GetArrayLength(env, pcm);
  jfloat *buf = (*env)->GetFloatArrayElements(env, pcm, NULL);
  float emb[CMP_EMB_DIM];
  int rc = cmp_engine_embed_pcm((const float *)buf, (size_t)n, emb, CMP_EMB_DIM);
  (*env)->ReleaseFloatArrayElements(env, pcm, buf, JNI_ABORT);
  if (rc) {
    jclass c = (*env)->FindClass(env, "java/lang/IllegalStateException");
    if (c) (*env)->ThrowNew(env, c, cmp_engine_error());
    return NULL;
  }
  jfloatArray out = (*env)->NewFloatArray(env, CMP_EMB_DIM);
  if (out) (*env)->SetFloatArrayRegion(env, out, 0, CMP_EMB_DIM, emb);
  return out;
}

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_cmpTimings(JNIEnv *env, jclass clazz) {
  return (*env)->NewStringUTF(env, cmp_timings());
}

/* 释放 CAM++ / aligner 常驻图：会议流水线要串行换模型，避免两套图抢 DSP 内存。 */
JNIEXPORT void JNICALL
Java_com_viitorvoice_demo_AsrNative_cmpShutdown(JNIEnv *env, jclass clazz) {
  (void)env; (void)clazz;
  cmp_engine_shutdown();
}
