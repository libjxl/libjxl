// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jni.h>

#include "tools/jni/org/jpeg/jpegli/wrapper/encoder_jni.h"

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  return org_jpeg_jpegli_wrapper::JniRegister(vm);
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved) {
  // TODO(eustas): actually we have cached jclass ref, but let it be so for now.
}

#ifdef __cplusplus
}
#endif
