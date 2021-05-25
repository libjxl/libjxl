// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jni.h>

#include "tools/jni/org/jpeg/jpegxl/wrapper/decoder_jni.h"

#ifdef __cplusplus
extern "C" {
#endif

static const JNINativeMethod kDecoderMethods[] = {
    {"nativeGetBasicInfo", "([ILjava/nio/Buffer;)V",
     reinterpret_cast<void*>(
         Java_org_jpeg_jpegxl_wrapper_DecoderJni_nativeGetBasicInfo)},
    {"nativeGetPixels",
     "([ILjava/nio/Buffer;Ljava/nio/Buffer;Ljava/nio/Buffer;)V",
     reinterpret_cast<void*>(
         Java_org_jpeg_jpegxl_wrapper_DecoderJni_nativeGetPixels)}};

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return -1;
  }

  jclass clazz = env->FindClass("org/jpeg/jpegxl/wrapper/DecoderJni");
  if (clazz == nullptr) {
    return -1;
  }

  if (env->RegisterNatives(
          clazz, kDecoderMethods,
          sizeof(kDecoderMethods) / sizeof(kDecoderMethods[0])) < 0) {
    return -1;
  }

  return JNI_VERSION_1_6;
}

#ifdef __cplusplus
}
#endif
