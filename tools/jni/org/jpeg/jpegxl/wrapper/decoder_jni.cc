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

#include "tools/jni/org/jpeg/jpegxl/wrapper/decoder_jni.h"

#include <jni.h>

#include "jxl/decode.h"
#include "jxl/thread_parallel_runner.h"
#include "lib/jxl/base/status.h"

namespace {

template <typename From, typename To>
bool StaticCast(const From& from, To* to) {
  To tmp = static_cast<To>(from);
  // Check sign is preserved.
  if ((from < 0 && tmp > 0) || (from > 0 && tmp < 0)) return false;
  // Check value is preserved.
  if (from != static_cast<From>(tmp)) return false;
  *to = tmp;
  return true;
}

bool BufferToSpan(JNIEnv* env, jobject buffer, uint8_t** data, size_t* size) {
  if (buffer == nullptr) return true;

  *data = reinterpret_cast<uint8_t*>(env->GetDirectBufferAddress(buffer));
  if (*data == nullptr) return false;
  return StaticCast(env->GetDirectBufferCapacity(buffer), size);
}

jxl::Status DoDecode(JNIEnv* env, jobject data_buffer, size_t* info_pixels_size,
                     size_t* info_icc_size, JxlBasicInfo* info,
                     jobject pixels_buffer, jobject icc_buffer) {
  if (data_buffer == nullptr) return JXL_FAILURE("No data buffer");

  uint8_t* data = nullptr;
  size_t data_size = 0;
  if (!BufferToSpan(env, data_buffer, &data, &data_size)) {
    return JXL_FAILURE("Failed to access data buffer");
  }

  uint8_t* pixels = nullptr;
  size_t pixels_size = 0;
  if (!BufferToSpan(env, pixels_buffer, &pixels, &pixels_size)) {
    return JXL_FAILURE("Failed to access pixels buffer");
  }

  uint8_t* icc = nullptr;
  size_t icc_size = 0;
  if (!BufferToSpan(env, icc_buffer, &icc, &icc_size)) {
    return JXL_FAILURE("Failed to access ICC buffer");
  }

  JxlDecoder* dec = JxlDecoderCreate(NULL);

  constexpr size_t kNumThreads = 0;  // Do everyting in this thread.
  void* runner = JxlThreadParallelRunnerCreate(NULL, kNumThreads);

  struct Defer {
    JxlDecoder* dec;
    void* runner;
    ~Defer() {
      JxlThreadParallelRunnerDestroy(runner);
      JxlDecoderDestroy(dec);
    }
  } defer{dec, runner};

  auto status =
      JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner);
  if (status != JXL_DEC_SUCCESS) {
    return JXL_FAILURE("Failed to set parallel runner");
  }
  status = JxlDecoderSubscribeEvents(
      dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE | JXL_DEC_COLOR_ENCODING);
  if (status != JXL_DEC_SUCCESS) {
    return JXL_FAILURE("Failed to subscribe for events");
  }
  status = JxlDecoderSetInput(dec, data, data_size);
  if (status != JXL_DEC_SUCCESS) {
    return JXL_FAILURE("Failed to set input");
  }
  status = JxlDecoderProcessInput(dec);
  if (status != JXL_DEC_BASIC_INFO) {
    return JXL_FAILURE("Unexpected notification (want: basic info)");
  }
  // RGBA (4-bytes per pixel), no scanline padding.
  JxlPixelFormat format = {4, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};
  if (info_pixels_size) {
    status = JxlDecoderImageOutBufferSize(dec, &format, info_pixels_size);
    if (status != JXL_DEC_SUCCESS) {
      return JXL_FAILURE("Failed to get pixels size");
    }
  }
  if (info) {
    status = JxlDecoderGetBasicInfo(dec, info);
    if (status != JXL_DEC_SUCCESS) {
      return JXL_FAILURE("Failed to get basic info");
    }
  }
  status = JxlDecoderProcessInput(dec);
  if (status != JXL_DEC_COLOR_ENCODING) {
    return JXL_FAILURE("Unexpected notification (want: color encoding)");
  }
  if (info_icc_size) {
    status = JxlDecoderGetICCProfileSize(
        dec, &format, JXL_COLOR_PROFILE_TARGET_DATA, info_icc_size);
    if (status != JXL_DEC_SUCCESS) *info_icc_size = 0;
  }
  if (icc && icc_size > 0) {
    status = JxlDecoderGetColorAsICCProfile(
        dec, &format, JXL_COLOR_PROFILE_TARGET_DATA, icc, icc_size);
    if (status != JXL_DEC_SUCCESS) {
      return JXL_FAILURE("Failed to get ICC");
    }
  }
  if (pixels) {
    status = JxlDecoderProcessInput(dec);
    if (status != JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      return JXL_FAILURE("Unexpected notification (want: need out buffer)");
    }
    status = JxlDecoderSetImageOutBuffer(dec, &format, pixels, pixels_size);
    if (status != JXL_DEC_SUCCESS) {
      return JXL_FAILURE("Failed to set out buffer");
    }
    status = JxlDecoderProcessInput(dec);
    if (status != JXL_DEC_FULL_IMAGE) {
      return JXL_FAILURE("Unexpected notification (want: full image)");
    }
    status = JxlDecoderProcessInput(dec);
    if (status != JXL_DEC_SUCCESS) {
      return JXL_FAILURE("Unexpected notification (want: success)");
    }
  }

  return true;
}

#undef FAILURE

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_org_jpeg_jpegxl_wrapper_DecoderJni_nativeGetBasicInfo(
    JNIEnv* env, jobject /*jobj*/, jintArray ctx, jobject data_buffer) {
  jint context[5] = {0};

  JxlBasicInfo info;
  size_t pixels_size = 0;
  size_t icc_size = 0;

  bool ok = true;

  if (ok) {
    ok = DoDecode(env, data_buffer, &pixels_size, &icc_size, &info,
                  /* pixels_buffer= */ nullptr, /* icc_buffer= */ nullptr);
  }

  if (ok) {
    ok &= StaticCast(info.xsize, context + 1);
    ok &= StaticCast(info.ysize, context + 2);
    ok &= StaticCast(pixels_size, context + 3);
    ok &= StaticCast(icc_size, context + 4);
  }

  if (!ok) context[0] = -1;

  env->SetIntArrayRegion(ctx, 0, 5, context);
}

/**
 * Get image pixel data.
 *
 * @param ctx {out_status} tuple
 * @param data [in] Buffer with encoded JXL stream
 * @param pixels [out] Buffer to place pixels to
 */
JNIEXPORT void JNICALL Java_org_jpeg_jpegxl_wrapper_DecoderJni_nativeGetPixels(
    JNIEnv* env, jobject /*jobj*/, jintArray ctx, jobject data_buffer,
    jobject pixels_buffer, jobject icc_buffer) {
  jint context[1] = {0};

  bool ok = true;

  if (ok) {
    ok = DoDecode(env, data_buffer, /* info_pixels_size= */ nullptr,
                  /* info_icc_size= */ nullptr, /* info= */ nullptr,
                  pixels_buffer, icc_buffer);
  }

  if (!ok) context[0] = -1;

  env->SetIntArrayRegion(ctx, 0, 1, context);
}

#ifdef __cplusplus
}
#endif
