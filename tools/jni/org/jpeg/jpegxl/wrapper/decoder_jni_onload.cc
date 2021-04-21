// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
