/* aligner_jni.c — Kotlin <-> aligner_engine 的桥（与 ASR 共用 libasr_qnn.so）。 */
#include <jni.h>
#include <stdlib.h>
#include <string.h>

#include "aligner_engine.h"

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_alignInit(JNIEnv *env, jclass clazz, jstring model_dir,
                                           jstring lib_dir) {
  (void)clazz;
  const char *md = (*env)->GetStringUTFChars(env, model_dir, NULL);
  const char *ld = (*env)->GetStringUTFChars(env, lib_dir, NULL);
  int rc = aligner_init(md, ld);
  (*env)->ReleaseStringUTFChars(env, model_dir, md);
  (*env)->ReleaseStringUTFChars(env, lib_dir, ld);
  return (*env)->NewStringUTF(env, rc ? aligner_error() : "");
}

JNIEXPORT jfloatArray JNICALL
Java_com_viitorvoice_demo_AsrNative_alignWav(JNIEnv *env, jclass clazz, jstring wav_path,
                                          jintArray ids) {
  (void)clazz;
  const char *path = (*env)->GetStringUTFChars(env, wav_path, NULL);
  jsize n = (*env)->GetArrayLength(env, ids);
  jint *buf = (*env)->GetIntArrayElements(env, ids, NULL);
  float out[512];
  int n_out = 0;
  int rc = aligner_run_wav(path, (const int *)buf, (int)n, out, 512, &n_out);
  (*env)->ReleaseIntArrayElements(env, ids, buf, JNI_ABORT);
  (*env)->ReleaseStringUTFChars(env, wav_path, path);
  if (rc) {
    jclass c = (*env)->FindClass(env, "java/lang/IllegalStateException");
    (*env)->ThrowNew(env, c, aligner_error());
    return NULL;
  }
  jfloatArray r = (*env)->NewFloatArray(env, n_out);
  (*env)->SetFloatArrayRegion(env, r, 0, n_out, out);
  return r;
}

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_alignTimings(JNIEnv *env, jclass clazz) {
  (void)clazz;
  return (*env)->NewStringUTF(env, aligner_timings());
}

JNIEXPORT jstring JNICALL
Java_com_viitorvoice_demo_AsrNative_alignDecDbg(JNIEnv *env, jclass clazz,
                                             jstring model_dir, jstring lib_dir,
                                             jstring in_path, jstring out_path) {
  (void)clazz;
  const char *md = (*env)->GetStringUTFChars(env, model_dir, NULL);
  const char *ld = (*env)->GetStringUTFChars(env, lib_dir, NULL);
  const char *ip = (*env)->GetStringUTFChars(env, in_path, NULL);
  const char *op = (*env)->GetStringUTFChars(env, out_path, NULL);
  int rc = aligner_dec_dbg(md, ld, ip, op);
  (*env)->ReleaseStringUTFChars(env, model_dir, md);
  (*env)->ReleaseStringUTFChars(env, lib_dir, ld);
  (*env)->ReleaseStringUTFChars(env, in_path, ip);
  (*env)->ReleaseStringUTFChars(env, out_path, op);
  return (*env)->NewStringUTF(env, rc ? aligner_error() : "");
}

JNIEXPORT void JNICALL
Java_com_viitorvoice_demo_AsrNative_alignShutdown(JNIEnv *env, jclass clazz) {
  (void)env; (void)clazz;
  aligner_shutdown();
}
