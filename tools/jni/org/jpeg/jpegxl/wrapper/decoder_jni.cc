// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/jni/org/jpeg/jpegxl/wrapper/decoder_jni.h"

#include <jni.h>

#include <cstdlib>

#include "jxl/decode.h"
#include "jxl/thread_parallel_runner.h"
#include "lib/jxl/base/status.h"

#if JPEGXL_ENABLE_SKCMS
#include "skcms.h"
#endif  // JPEGXL_ENABLE_SKCMS

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

struct Span {
  uint8_t* data = nullptr;
  size_t size = 0;
};

bool BufferToSpan(JNIEnv* env, jobject buffer, Span* span) {
  if (buffer == nullptr) return true;

  span->data = reinterpret_cast<uint8_t*>(env->GetDirectBufferAddress(buffer));
  if (span->data == nullptr) return false;
  return StaticCast(env->GetDirectBufferCapacity(buffer), &span->size);
}

int ToStatusCode(const jxl::Status& status) {
  if (status) return 0;
  if (status.IsFatalError()) return -1;
  return 1;  // Non-fatal -> not enough input.
}

constexpr const size_t kLastPixelFormat = 3;
constexpr const size_t kNoPixelFormat = static_cast<size_t>(-1);

constexpr const size_t kLastColorSpace = 1;

JxlPixelFormat ToPixelFormat(size_t pixel_format) {
  if (pixel_format == 0) {
    // RGBA, 4 x byte per pixel, no scanline padding.
    return {/*num_channels=*/4, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, /*align=*/0};
  } else if (pixel_format == 1) {
    // RGBA, 4 x float16 per pixel, no scanline padding.
    return {/*num_channels=*/4, JXL_TYPE_FLOAT16, JXL_LITTLE_ENDIAN,
            /*align=*/0};
  } else if (pixel_format == 2) {
    // RGB, 4 x byte per pixel, no scanline padding.
    return {/*num_channels=*/3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, /*align=*/0};
  } else if (pixel_format == 3) {
    // RGB, 4 x float16 per pixel, no scanline padding.
    return {/*num_channels=*/3, JXL_TYPE_FLOAT16, JXL_LITTLE_ENDIAN,
            /*align=*/0};
  } else {
    abort();
    return {0, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};
  }
}

jxl::Status DoDecode(JNIEnv* env, jobject data_buffer, size_t* info_pixels_size,
                     size_t* info_icc_size, JxlBasicInfo* info,
                     size_t pixel_format, Span pixels, Span icc) {
  if (data_buffer == nullptr) return JXL_FAILURE("No data buffer");

  Span data = {};
  if (!BufferToSpan(env, data_buffer, &data)) {
    return JXL_FAILURE("Failed to access data buffer");
  }

  JxlDecoder* dec = JxlDecoderCreate(NULL);

  constexpr size_t kNumThreads = 0;  // Do everything in this thread.
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
  status = JxlDecoderSetInput(dec, data.data, data.size);
  if (status != JXL_DEC_SUCCESS) {
    return JXL_FAILURE("Failed to set input");
  }
  status = JxlDecoderProcessInput(dec);
  if (status == JXL_DEC_NEED_MORE_INPUT) {
    return JXL_STATUS(jxl::StatusCode::kNotEnoughBytes, "Not enough input");
  }
  if (status != JXL_DEC_BASIC_INFO) {
    return JXL_FAILURE("Unexpected notification (want: basic info)");
  }
  if (info_pixels_size) {
    JxlPixelFormat format = ToPixelFormat(pixel_format);
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
    JxlPixelFormat format = ToPixelFormat(pixel_format);
    status = JxlDecoderGetICCProfileSize(
        dec, &format, JXL_COLOR_PROFILE_TARGET_DATA, info_icc_size);
    if (status != JXL_DEC_SUCCESS) *info_icc_size = 0;
  }
  if (icc.data && icc.size > 0) {
    JxlPixelFormat format = ToPixelFormat(pixel_format);
    status = JxlDecoderGetColorAsICCProfile(
        dec, &format, JXL_COLOR_PROFILE_TARGET_DATA, icc.data, icc.size);
    if (status != JXL_DEC_SUCCESS) {
      return JXL_FAILURE("Failed to get ICC");
    }
  }
  if (pixels.data) {
    JxlPixelFormat format = ToPixelFormat(pixel_format);
    status = JxlDecoderProcessInput(dec);
    if (status != JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      return JXL_FAILURE("Unexpected notification (want: need out buffer)");
    }
    status =
        JxlDecoderSetImageOutBuffer(dec, &format, pixels.data, pixels.size);
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

jxl::Status ConvertPixels(const Span pixels, const Span icc,
                          size_t output_pixel_format,
                          size_t output_colorspace) {
#if JPEGXL_ENABLE_SKCMS
  skcms_ICCProfile profile;
  if (!skcms_Parse(icc.data, icc.size, &profile)) {
    return JXL_FAILURE("Failed to parse ICC profile");
  }
  size_t bytes_per_pixel = (output_pixel_format == 1) ? 8 : 6;

  skcms_PixelFormat pixel_format = (output_pixel_format == 1)
                                       ? skcms_PixelFormat_RGBA_hhhh
                                       : skcms_PixelFormat_RGB_hhh;
  skcms_AlphaFormat alpha_format = (output_pixel_format == 1)
                                       ? skcms_AlphaFormat_Unpremul
                                       : skcms_AlphaFormat_Opaque;
  const skcms_ICCProfile* output_profile =
      (output_colorspace == 0) ? skcms_sRGB_profile() : skcms_XYZD50_profile();
  if (!skcms_Transform(pixels.data, pixel_format, alpha_format, &profile,
                       pixels.data, pixel_format, alpha_format, output_profile,
                       pixels.size / bytes_per_pixel)) {
    return JXL_FAILURE("Color conversion failed");
  }
  return true;
#else   // JPEGXL_ENABLE_SKCMS
  return JXL_FAILURE("Color conversion not supported");
#endif  // JPEGXL_ENABLE_SKCMS
}

}  // namespace

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT void JNICALL
Java_org_jpeg_jpegxl_wrapper_DecoderJni_nativeGetBasicInfo(
    JNIEnv* env, jobject /*jobj*/, jintArray ctx, jobject data_buffer) {
  jint context[6] = {0};
  env->GetIntArrayRegion(ctx, 0, 1, context);

  JxlBasicInfo info;
  size_t pixels_size = 0;
  size_t icc_size = 0;
  size_t pixel_format = 0;

  jxl::Status status = true;

  if (status) {
    pixel_format = context[0];
    if (pixel_format == kNoPixelFormat) {
      // OK
    } else if (pixel_format > kLastPixelFormat) {
      status = JXL_FAILURE("Unrecognized pixel format");
    }
  }

  if (status) {
    bool want_output_size = (pixel_format != kNoPixelFormat);
    if (want_output_size) {
      status = DoDecode(env, data_buffer, &pixels_size, &icc_size, &info,
                        pixel_format,
                        /* pixels= */ Span(), /* icc= */ Span());
    } else {
      status = DoDecode(env, data_buffer, /* info_pixels_size= */ nullptr,
                        /* info_icc_size= */ nullptr, &info, pixel_format,
                        /* pixels= */ Span(), /* icc= */ Span());
    }
  }

  if (status) {
    bool ok = true;
    ok &= StaticCast(info.xsize, context + 1);
    ok &= StaticCast(info.ysize, context + 2);
    ok &= StaticCast(pixels_size, context + 3);
    ok &= StaticCast(icc_size, context + 4);
    ok &= StaticCast(info.alpha_bits, context + 5);
    if (!ok) status = JXL_FAILURE("Invalid value");
  }

  context[0] = ToStatusCode(status);

  env->SetIntArrayRegion(ctx, 0, 6, context);
}

/**
 * Get image pixel data.
 *
 * @param ctx {out_status} tuple
 * @param data [in] Buffer with encoded JXL stream
 * @param pixels [out] Buffer to place pixels to
 */
JNIEXPORT void JNICALL Java_org_jpeg_jpegxl_wrapper_DecoderJni_nativeGetPixels(
    JNIEnv* env, jobject /* jobj */, jintArray ctx, jobject data_buffer,
    jobject pixels_buffer, jobject icc_buffer) {
  jint context[2] = {0};
  env->GetIntArrayRegion(ctx, 0, 2, context);

  jxl::Status status = true;

  bool want_color_transform = false;
  size_t output_colorspace = 0;
  Span pixels = {};
  Span icc = {};
  size_t pixel_format = 0;
  int error_code = 0;

  if (status) {
    // Unlike getBasicInfo, "no-pixel-format" is not supported.
    pixel_format = context[0];
    if (pixel_format > kLastPixelFormat) {
      status = JXL_FAILURE("Unrecognized pixel format");
    }
  }

  if (status) {
    if (context[1] >= 0) {
      want_color_transform = true;
      output_colorspace = context[1];
      if (output_colorspace > kLastColorSpace) {
        status = JXL_FAILURE("Unrecognized color space");
      } else if ((pixel_format & 1) == 0) {
        error_code = -2;
        status = JXL_FAILURE("Only FP16 color transform is supported");
      }
#if !JPEGXL_ENABLE_SKCMS
      error_code = -2;
      status = JXL_FAILURE("Color transform is not supported");
#endif  // JPEGXL_ENABLE_SKCMS
    }
  }

  if (status && !BufferToSpan(env, pixels_buffer, &pixels)) {
    status = JXL_FAILURE("Failed to access pixels buffer");
  }

  if (status && !BufferToSpan(env, icc_buffer, &icc)) {
    status = JXL_FAILURE("Failed to access icc buffer");
  }

  if (status) {
    status = DoDecode(env, data_buffer, /* info_pixels_size= */ nullptr,
                      /* info_icc_size= */ nullptr, /* info= */ nullptr,
                      pixel_format, pixels, icc);
  }

  if (status && want_color_transform) {
    status = ConvertPixels(pixels, icc, pixel_format, output_colorspace);
  }

  context[0] = error_code ? error_code : ToStatusCode(status);
  env->SetIntArrayRegion(ctx, 0, 1, context);
}

#ifdef __cplusplus
}
#endif
