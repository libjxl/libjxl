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
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <random>
#include <vector>

#include "hwy/targets.h"
#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/thread_parallel_runner.h"
#include "jxl/thread_parallel_runner_cxx.h"

// Unpublised API.
void SetDecoderMemoryLimitBase_(size_t memory_limit_base);

namespace {

// Externally visible value to ensure pixels are used in the fuzzer.
int external_code = 0;

constexpr const size_t kStreamingTargetNumberOfChunks = 128;

// Options for the fuzzing
struct FuzzSpec {
  bool use_float;
  bool get_alpha;
  bool get_grayscale;
  bool use_streaming;
  uint32_t streaming_seed;
};

// use_streaming: if true, decodes the data in small chunks, if false, decodes
// it in one shot.
bool DecodeJpegXl(const uint8_t* jxl, size_t size, size_t max_pixels,
                  const FuzzSpec& spec, std::vector<uint8_t>* pixels,
                  size_t* xsize, size_t* ysize,
                  std::vector<uint8_t>* icc_profile) {
  SetDecoderMemoryLimitBase_(max_pixels);
  // Multi-threaded parallel runner. Limit to max 2 threads since the fuzzer
  // itself is already multithreded.
  size_t num_threads =
      std::min<size_t>(2, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  auto runner = JxlThreadParallelRunnerMake(nullptr, num_threads);

  std::mt19937 mt(spec.streaming_seed);
  std::exponential_distribution<> dis(kStreamingTargetNumberOfChunks);

  auto dec = JxlDecoderMake(nullptr);
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(dec.get(),
                                JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING |
                                    JXL_DEC_FRAME | JXL_DEC_FULL_IMAGE)) {
    return false;
  }
  if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(dec.get(),
                                                     JxlThreadParallelRunner,
                                                     runner.get())) {
    return false;
  }

  JxlBasicInfo info;
  uint32_t channels = (spec.get_grayscale ? 1 : 3) + (spec.get_alpha ? 1 : 0);
  JxlPixelFormat format = {channels,
                           spec.use_float ? JXL_TYPE_FLOAT : JXL_TYPE_UINT8,
                           JXL_NATIVE_ENDIAN, 0};

  size_t bytes_per_pixel = channels * (spec.use_float ? 4 : 1);

  if (!spec.use_streaming) {
    // Set all input at once
    JxlDecoderSetInput(dec.get(), jxl, size);
  }

  bool seen_basic_info = false;
  bool seen_color_encoding = false;
  bool seen_need_image_out = false;
  bool seen_full_image = false;
  bool seen_success = false;
  bool seen_frame = false;
  // If streaming and seen around half the input, test flushing
  bool tested_flush = false;

  // Size made available for the streaming input, emulating a subset of the
  // full input size.
  size_t streaming_size = 0;
  size_t leftover = size;

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

    if (status == JXL_DEC_ERROR) {
      return false;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      if (spec.use_streaming) {
        size_t remaining = JxlDecoderReleaseInput(dec.get());
        // move any remaining bytes to the front if necessary
        size_t used = streaming_size - remaining;
        jxl += used;
        leftover -= used;
        streaming_size -= used;
        size_t chunk_size =
            std::max<size_t>(1, size * std::min<double>(1.0, dis(mt)));
        size_t add_size =
            std::min<size_t>(chunk_size, leftover - streaming_size);
        if (add_size == 0) {
          // End of the streaming data reached
          return false;
        }
        streaming_size += add_size;
        JxlDecoderSetInput(dec.get(), jxl, streaming_size);

        if (!tested_flush && seen_frame) {
          // Test flush max once to avoid too slow fuzzer run
          tested_flush = true;
          JxlDecoderFlushImage(dec.get());
        }
      } else {
        return false;
      }
    } else if (status == JXL_DEC_BASIC_INFO) {
      if (seen_basic_info) abort();  // already seen basic info
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
      if (!seen_basic_info) abort();     // expected basic info first
      if (seen_color_encoding) abort();  // already seen color encoding
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
    } else if (status == JXL_DEC_FRAME ||
               status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      if (!seen_color_encoding) abort();  // expected color encoding first
      if (status == JXL_DEC_FRAME) {
        if (seen_frame) abort();  // already seen JXL_DEC_FRAME
        seen_frame = true;
        // When not testing streaming, test that JXL_DEC_NEED_IMAGE_OUT_BUFFER
        // occurs instead, so do not set buffer now.
        if (!spec.use_streaming) continue;
      }
      if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
        // expected JXL_DEC_FRAME instead
        if (!seen_frame) abort();
        // already should have set buffer if streaming
        if (spec.use_streaming) abort();
        // already seen need image out
        if (seen_need_image_out) abort();
        seen_need_image_out = true;
      }

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
      // expected need image out or frame first
      if (!seen_need_image_out && !seen_frame) abort();

      seen_full_image = true;  // there may be multiple if animated

      // There may be a next animation frame so expect those again:
      seen_need_image_out = false;
      seen_frame = false;

      // "Use" all the pixels
      for (size_t i = 0; i < pixels->size(); i++) {
        external_code ^= (*pixels)[i];
      }

      // Nothing to do. Do not yet return. If the image is an animation, more
      // full frames may be decoded. This example only keeps the last one.
    } else if (status == JXL_DEC_SUCCESS) {
      if (!seen_full_image) abort();  // expected full image before finishing
      if (seen_success) abort();      // already seen success
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

  FuzzSpec spec;
  spec.use_float = !!(flags & 1);
  spec.get_alpha = !!(flags & 2);
  spec.get_grayscale = !!(flags & 4);
  spec.use_streaming = !!(flags & 8);
  // Allos some different possible variations in the chunk sizes of the
  // streaming case
  spec.streaming_seed = flags ^ size;

  std::vector<uint8_t> pixels;
  std::vector<uint8_t> icc;
  size_t xsize, ysize;
  size_t max_pixels = 1 << 21;

  const auto targets = hwy::SupportedAndGeneratedTargets();
  hwy::SetSupportedTargetsForTest(spec.streaming_seed % targets.size());
  DecodeJpegXl(data, size, max_pixels, spec, &pixels, &xsize, &ysize, &icc);
  hwy::SetSupportedTargetsForTest(0);

  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return TestOneInput(data, size);
}
