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

#include "c_interop.h"

#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"

extern "C" uint8_t *JxlMemoryToPixels(const uint8_t *data, size_t size,
                                      size_t *stride, size_t *xsize,
                                      size_t *ysize, int *has_alpha) {
  jxl::PaddedBytes bytes(size);
  memcpy(bytes.data(), data, size);
  jxl::DecompressParams params;
  jxl::ThreadPoolInternal pool(4);
  jxl::CodecInOut io;

  if (jxl::DecodeFile(params, bytes, &io, nullptr, &pool) == false)
    return nullptr;

  jxl::Image3B converted;

  if (!io.Main().CopyToSRGB(jxl::Rect(io), &converted, &pool)) {
    return nullptr;
  }

  size_t io_stride = io.Main().color()->PixelsPerRow();

  if (io.Main().HasAlpha()) {
    uint8_t *image = new uint8_t[4 * io_stride * io.ysize()];

    if (image == nullptr) {
      return nullptr;
    }

    *stride = 4 * io_stride;
    *xsize = io.xsize();
    *ysize = io.ysize();
    *has_alpha = 1;
    const int alpha_right_shift_amount =
        static_cast<int>(io.metadata.m.GetAlphaBits()) - 8;
    for (size_t y = 0; y < *ysize; ++y) {
      uint8_t *JXL_RESTRICT const row = image + y * *stride;
      const uint16_t *const alpha_row = io.Main().alpha()->ConstRow(y);
      const uint8_t *JXL_RESTRICT const red_row = converted.ConstPlaneRow(0, y);
      const uint8_t *JXL_RESTRICT const green_row =
          converted.ConstPlaneRow(1, y);
      const uint8_t *JXL_RESTRICT const blue_row =
          converted.ConstPlaneRow(2, y);
      for (size_t x = 0; x < *xsize; ++x) {
        row[4 * x] = red_row[x];
        row[4 * x + 1] = green_row[x];
        row[4 * x + 2] = blue_row[x];
        row[4 * x + 3] = alpha_row[x] >> alpha_right_shift_amount;
      }
    }
    return image;
  } else {
    uint8_t *image = new uint8_t[3 * io_stride * io.ysize()];

    if (image == nullptr) {
      return nullptr;
    }

    *stride = 3 * io_stride;
    *xsize = io.xsize();
    *ysize = io.ysize();
    *has_alpha = 0;
    for (size_t y = 0; y < *ysize; ++y) {
      uint8_t *JXL_RESTRICT const row = image + y * *stride;
      const uint8_t *JXL_RESTRICT const red_row = converted.ConstPlaneRow(0, y);
      const uint8_t *JXL_RESTRICT const green_row =
          converted.ConstPlaneRow(1, y);
      const uint8_t *JXL_RESTRICT const blue_row =
          converted.ConstPlaneRow(2, y);
      for (size_t x = 0; x < *xsize; ++x) {
        row[3 * x] = red_row[x];
        row[3 * x + 1] = green_row[x];
        row[3 * x + 2] = blue_row[x];
      }
    }
    return image;
  }
}

extern "C" void JxlFreePixels(uint8_t *pixels) { delete[] pixels; }
