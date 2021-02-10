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

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <vector>

#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "lib/jxl/base/status.h"

namespace jxl {

// Externally visible value to ensure pixels are used in the fuzzer.
int external_code = 0;

int DecodeJpegXlOneShot(const uint8_t* jxl, size_t size, size_t max_pixels,
                        bool use_float, bool get_alpha, bool get_grayscale,
                        std::vector<uint8_t>* pixels, size_t* xsize,
                        size_t* ysize, std::vector<uint8_t>* icc_profile) {
  // Multi-threaded parallel runner. Limit to max 2 threads since the fuzzer
  // itself is already multithreded.
  size_t num_threads =
      std::min<size_t>(2, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  auto runner = JxlThreadParallelRunnerMake(nullptr, num_threads);

  auto dec = JxlDecoderMake(nullptr);
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO |
                                               JXL_DEC_COLOR_ENCODING |
                                               JXL_DEC_FULL_IMAGE)) {
    return false;
  }

  if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(dec.get(),
                                                     JxlThreadParallelRunner,
                                                     runner.get())) {
    return false;
  }

  JxlBasicInfo info;
  uint32_t channels = (get_grayscale ? 1 : 3) + (get_alpha ? 0 : 1);
  JxlPixelFormat format = {channels,
                           use_float ? JXL_TYPE_FLOAT : JXL_TYPE_UINT8,
                           JXL_NATIVE_ENDIAN, 0};

  size_t bytes_per_pixel = channels * (use_float ? 4 : 1);

  JxlDecoderSetInput(dec.get(), jxl, size);

  bool seen_basic_info = false;
  bool seen_color_encoding = false;
  bool seen_need_image_out = false;
  bool seen_full_image = false;
  bool seen_success = false;

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

    if (status == JXL_DEC_ERROR) {
      return false;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      return false;
    } else if (status == JXL_DEC_BASIC_INFO) {
      if (seen_basic_info) JXL_ABORT("already seen basic info");
      seen_basic_info = true;

      if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info)) {
        return false;
      }
      *xsize = info.xsize;
      *ysize = info.ysize;
      size_t num_pixels = *xsize * *ysize;
      // num_pixels overflow
      if (*xsize != 0 && num_pixels / *xsize != *ysize) return false;
      // limit max memory of this fuzzer test
      if (num_pixels > max_pixels) return false;
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      if (!seen_basic_info) JXL_ABORT("expected basic info first");
      if (seen_color_encoding) JXL_ABORT("already seen color encoding");
      seen_color_encoding = true;

      // Get the ICC color profile of the pixel data
      size_t icc_size;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderGetICCProfileSize(
              dec.get(), &format, JXL_COLOR_PROFILE_TARGET_DATA, &icc_size)) {
        return false;
      }
      icc_profile->resize(icc_size);
      if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(
                                 dec.get(), &format,
                                 JXL_COLOR_PROFILE_TARGET_DATA,
                                 icc_profile->data(), icc_profile->size())) {
        return false;
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      if (!seen_color_encoding) JXL_ABORT("expected color encoding first");
      if (seen_need_image_out) JXL_ABORT("already seen need image out");
      seen_need_image_out = true;

      size_t buffer_size;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size)) {
        return false;
      }
      if (buffer_size != *xsize * *ysize * bytes_per_pixel) {
        return false;
      }
      pixels->resize(*xsize * *ysize * bytes_per_pixel);
      void* pixels_buffer = (void*)pixels->data();
      size_t pixels_buffer_size = pixels->size() * sizeof(float);
      if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(dec.get(), &format,
                                                         pixels_buffer,
                                                         pixels_buffer_size)) {
        return false;
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      if (!seen_need_image_out) JXL_ABORT("expected need image out first");
      if (seen_full_image) JXL_ABORT("already seen full image");
      seen_full_image = true;
      // "Use" all the pixels
      for (size_t i = 0; i < pixels->size(); i++) {
        external_code ^= (*pixels)[i];
      }

      // Nothing to do. Do not yet return. If the image is an animation, more
      // full frames may be decoded. This example only keeps the last one.
    } else if (status == JXL_DEC_SUCCESS) {
      if (!seen_full_image) JXL_ABORT("expected full image before finishing");
      if (seen_success) JXL_ABORT("already seen success");
      seen_success = true;

      // All decoding successfully finished.
      // It's not required to call JxlDecoderReleaseInput(dec.get()) here since
      // the decoder will be destroyed.
      return true;
    } else {
      return false;
    }
  }
}

int TestOneInput(const uint8_t* data, size_t size) {
  if (size == 0) return 0;
  uint8_t flags = data[size - 1];
  size--;

  bool use_float = !!(flags & 1);
  bool get_alpha = !!(flags & 2);
  bool get_grayscale = !!(flags & 4);

  std::vector<uint8_t> pixels;
  std::vector<uint8_t> icc;
  size_t xsize, ysize;
  size_t max_pixels = 1 << 21;

  DecodeJpegXlOneShot(data, size, max_pixels, use_float, get_alpha,
                      get_grayscale, &pixels, &xsize, &ysize, &icc);

  return 0;
}

}  // namespace jxl

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return jxl::TestOneInput(data, size);
}
