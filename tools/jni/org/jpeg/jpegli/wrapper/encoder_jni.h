// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_JNI_ORG_JPEG_JPEGLI_WRAPPER_ENCODER_JNI
#define TOOLS_JNI_ORG_JPEG_JPEGLI_WRAPPER_ENCODER_JNI

#include <jni.h>

namespace org_jpeg_jpegli_wrapper {
jint JniRegister(JavaVM* vm);
}  // namespace org_jpeg_jpegli_wrapper

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Cache references to java world; should be invoked once, on library
 * initialization.
 */
JNIEXPORT jint JNICALL
Java_org_jpeg_jpegli_wrapper_Encoder_nativeInit(JNIEnv* env, jobject /*jobj*/);

/**
 * Encode image with jpegli.
 */
JNIEXPORT jint JNICALL Java_org_jpeg_jpegli_wrapper_Encoder_nativeEncode(
    JNIEnv* env, jobject /*jobj*/, jint width, jint height, jintArray config,
    jintArray input, jobject output);

#ifdef __cplusplus
}
#endif

#endif  // TOOLS_JNI_ORG_JPEG_JPEGLI_WRAPPER_ENCODER_JNI
