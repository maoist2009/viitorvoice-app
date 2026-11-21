/* vad_jni.c — Kotlin <-> vad_engine 的桥（与 ASR 共用 libasr_qnn.so）。 */
#include <jni.h>
#include <stdlib.h>

#include "vad_engine.h"

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_vadInit(JNIEnv *env, jclass clazz, jstring model_dir,
                                         jstring lib_dir) {
  (void)clazz;
  const char *md = (*env)->GetStringUTFChars(env, model_dir, NULL);
  const char *ld = (*env)->GetStringUTFChars(env, lib_dir, NULL);
  int rc = vad_engine_init(md, ld);
  (*env)->ReleaseStringUTFChars(env, model_dir, md);
  (*env)->ReleaseStringUTFChars(env, lib_dir, ld);
  return (*env)->NewStringUTF(env, rc ? vad_engine_error() : "");
}

JNIEXPORT void JNICALL
Java_com_viitorvoice_demo_AsrNative_vadReset(JNIEnv *env, jclass clazz) {
  (void)env;
  (void)clazz;
  vad_engine_reset();
}

JNIEXPORT jint JNICALL
Java_com_viitorvoice_demo_AsrNative_vadFrames(JNIEnv *env, jclass clazz) {
  (void)env;
  (void)clazz;
  return (jint)vad_engine_frames();
}

/** 跑一块（frames*512 点），返回每帧概率；出错抛 IllegalStateException。 */
JNIEXPORT jfloatArray JNICALL
Java_com_viitorvoice_demo_AsrNative_vadRun(JNIEnv *env, jclass clazz, jfloatArray pcm) {
  (void)clazz;
  int k = vad_engine_frames();
  jsize n = (*env)->GetArrayLength(env, pcm);
  if (k <= 0 || n != (jsize)k * VAD_CHUNK) {
    jclass c = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
    (*env)->ThrowNew(env, c, "vadRun 点数必须等于 K*512");
    return NULL;
  }
  jfloat *buf = (*env)->GetFloatArrayElements(env, pcm, NULL);
  float *probs = (float *)malloc(sizeof(float) * k);
  jfloatArray res = (*env)->NewFloatArray(env, k);
  int rc = probs && res ? vad_engine_run((const float *)buf, k, probs) : -1;
  (*env)->ReleaseFloatArrayElements(env, pcm, buf, JNI_ABORT);
  if (rc) {
    free(probs);
    jclass c = (*env)->FindClass(env, "java/lang/IllegalStateException");
    (*env)->ThrowNew(env, c, vad_engine_error());
    return NULL;
  }
  (*env)->SetFloatArrayRegion(env, res, 0, k, probs);
  free(probs);
  return res;
}

JNIEXPORT void JNICALL
Java_com_viitorvoice_demo_AsrNative_vadShutdown(JNIEnv *env, jclass clazz) {
  (void)env;
  (void)clazz;
  vad_engine_shutdown();
}

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_vadTimings(JNIEnv *env, jclass clazz) {
  (void)clazz;
  return (*env)->NewStringUTF(env, vad_engine_timings());
}
