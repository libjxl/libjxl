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

#ifndef TOOLS_JNI_ORG_JPEG_JPEGXL_WRAPPER_DECODER_JNI
#define TOOLS_JNI_ORG_JPEG_JPEGXL_WRAPPER_DECODER_JNI

#include <jni.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get basic image information (size, etc.)
 *
 * @param ctx {out_status, out_width, out_height, pixels_size, icc_size} tuple
 * @param data [in] Buffer with encoded JXL stream
 */
JNIEXPORT void JNICALL
Java_org_jpeg_jpegxl_wrapper_DecoderJni_nativeGetBasicInfo(JNIEnv* env,
                                                           jobject /*jobj*/,
                                                           jintArray ctx,
                                                           jobject data_buffer);

/**
 * Get image pixel data.
 *
 * @param ctx {out_status} tuple
 * @param data [in] Buffer with encoded JXL stream
 * @param pixels [out] Buffer to place pixels to
 */
JNIEXPORT void JNICALL Java_org_jpeg_jpegxl_wrapper_DecoderJni_nativeGetPixels(
    JNIEnv* env, jobject /*jobj*/, jintArray ctx, jobject data_buffer,
    jobject pixels_buffer, jobject icc_buffer);

#ifdef __cplusplus
}
#endif

#endif  // TOOLS_JNI_ORG_JPEG_JPEGXL_WRAPPER_DECODER_JNI