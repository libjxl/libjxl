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
#include <map>
#include <mutex>
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
  bool jpeg_to_pixels;  // decode to pixels even if it is JPEG-reconstructible
  // Whether to use the callback mechanism for the output image or not.
  bool use_callback;
  uint32_t streaming_seed;
};

// use_streaming: if true, decodes the data in small chunks, if false, decodes
// it in one shot.
bool DecodeJpegXl(const uint8_t* jxl, size_t size, size_t max_pixels,
                  const FuzzSpec& spec, std::vector<uint8_t>* pixels,
                  std::vector<uint8_t>* jpeg, size_t* xsize, size_t* ysize,
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
      JxlDecoderSubscribeEvents(
          dec.get(), JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING |
                         JXL_DEC_FRAME | JXL_DEC_JPEG_RECONSTRUCTION |
                         JXL_DEC_FULL_IMAGE)) {
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
  uint32_t num_frames = 0;
  bool seen_jpeg_reconstruction = false;
  bool seen_jpeg_need_more_output = false;
  // If streaming and seen around half the input, test flushing
  bool tested_flush = false;

  // Size made available for the streaming input, emulating a subset of the
  // full input size.
  size_t streaming_size = 0;
  size_t leftover = size;

  // Callback function used when decoding with use_callback.
  struct DecodeCallbackData {
    JxlBasicInfo info;
    std::mutex called_rows_mutex;
    // For each row stores the segments of the row being called. For each row
    // the sum of all the int values in the map up to [i] (inclusive) tell how
    // many times a callback included the pixel i of that row.
    std::vector<std::map<uint32_t, int>> called_rows;

    // Use the pixel values.
    uint32_t value = 0;
  };
  DecodeCallbackData decode_callback_data;
  auto decode_callback = +[](void* opaque, size_t x, size_t y,
                             size_t num_pixels, const void* pixels) {
    DecodeCallbackData* data = static_cast<DecodeCallbackData*>(opaque);
    if (num_pixels > data->info.xsize) abort();
    if (x + num_pixels > data->info.xsize) abort();
    if (y >= data->info.ysize) abort();
    if (num_pixels && !pixels) abort();
    // Keep track of the segments being called by the callback.
    {
      const std::lock_guard<std::mutex> lock(data->called_rows_mutex);
      data->called_rows[y][x]++;
      data->called_rows[y][x + num_pixels]--;
      data->value += *static_cast<const uint8_t*>(pixels);
    }
  };

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
    } else if (status == JXL_DEC_JPEG_NEED_MORE_OUTPUT) {
      if (spec.jpeg_to_pixels) abort();
      if (!seen_jpeg_reconstruction) abort();
      seen_jpeg_need_more_output = true;
      size_t used_jpeg_output =
          jpeg->size() - JxlDecoderReleaseJPEGBuffer(dec.get());
      jpeg->resize(std::max<size_t>(4096, jpeg->size() * 2));
      uint8_t* jpeg_buffer = jpeg->data() + used_jpeg_output;
      size_t jpeg_buffer_size = jpeg->size() - used_jpeg_output;

      if (JXL_DEC_SUCCESS !=
          JxlDecoderSetJPEGBuffer(dec.get(), jpeg_buffer, jpeg_buffer_size)) {
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
      decode_callback_data.info = info;
      decode_callback_data.called_rows.resize(info.ysize);
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

      if (spec.use_callback) {
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSetImageOutCallback(dec.get(), &format, decode_callback,
                                          &decode_callback_data)) {
          return false;
        }
      } else {
        // Use the pixels output buffer.
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
        size_t pixels_buffer_size = pixels->size();
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSetImageOutBuffer(dec.get(), &format, pixels_buffer,
                                        pixels_buffer_size)) {
          return false;
        }
      }
    } else if (status == JXL_DEC_JPEG_RECONSTRUCTION) {
      if (seen_jpeg_reconstruction) abort();
      seen_jpeg_reconstruction = true;
      if (!spec.jpeg_to_pixels) {
        // Make sure buffer is allocated, but current size is too small to
        // contain valid JPEG.
        jpeg->resize(1);
        uint8_t* jpeg_buffer = jpeg->data();
        size_t jpeg_buffer_size = jpeg->size();
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSetJPEGBuffer(dec.get(), jpeg_buffer, jpeg_buffer_size)) {
          return false;
        }
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      if (!spec.jpeg_to_pixels && seen_jpeg_reconstruction) {
        if (!seen_jpeg_need_more_output) abort();
        jpeg->resize(jpeg->size() - JxlDecoderReleaseJPEGBuffer(dec.get()));
      } else {
        // expected need image out or frame first
        if (!seen_need_image_out && !seen_frame) abort();
      }

      seen_full_image = true;  // there may be multiple if animated

      // There may be a next animation frame so expect those again:
      seen_need_image_out = false;
      seen_frame = false;
      num_frames++;

      const auto consume = [&](uint8_t b) {
        if (b == 0) {
          external_code ^= ~0;
        } else {
          external_code ^= b;
        }
      };

      // "Use" all the pixels; MSAN needs a conditional to count as usage.
      for (size_t i = 0; i < pixels->size(); i++) consume(pixels->at(i));
      for (size_t i = 0; i < jpeg->size(); i++) consume(jpeg->at(i));

      // Nothing to do. Do not yet return. If the image is an animation, more
      // full frames may be decoded. This example only keeps the last one.
    } else if (status == JXL_DEC_SUCCESS) {
      if (!seen_full_image) abort();  // expected full image before finishing
      if (seen_success) abort();      // already seen success
      seen_success = true;

      // When decoding we may not get seen_need_image_out unless we were
      // decoding the image to pixels.
      if (seen_need_image_out && spec.use_callback) {
        // Check that the callback sent all the pixels
        for (uint32_t y = 0; y < info.ysize; y++) {
          // Check that each row was at least called once.
          if (decode_callback_data.called_rows[y].empty()) abort();
          uint32_t last_idx = 0;
          int calls = 0;
          for (auto it : decode_callback_data.called_rows[y]) {
            if (it.first > last_idx) {
              if (static_cast<uint32_t>(calls) != num_frames) abort();
            }
            calls += it.second;
            last_idx = it.first;
          }
        }
      }

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
  spec.jpeg_to_pixels = !!(flags & 16);
  spec.use_callback = !!(flags & 32);
  // Allows some different possible variations in the chunk sizes of the
  // streaming case
  spec.streaming_seed = flags ^ size;

  std::vector<uint8_t> pixels;
  std::vector<uint8_t> jpeg;
  std::vector<uint8_t> icc;
  size_t xsize, ysize;
  size_t max_pixels = 1 << 21;

  const auto targets = hwy::SupportedAndGeneratedTargets();
  hwy::SetSupportedTargetsForTest(spec.streaming_seed % targets.size());
  DecodeJpegXl(data, size, max_pixels, spec, &pixels, &jpeg, &xsize, &ysize,
               &icc);
  hwy::SetSupportedTargetsForTest(0);

  return 0;
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  return TestOneInput(data, size);
}
