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

#include <stdio.h>

#include <utility>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {
namespace {

void PrintXybRange() {
  Image3F linear(1u << 16, 257);
  for (int b = 0; b < 256; ++b) {
    float* JXL_RESTRICT row0 = linear.PlaneRow(0, b + 1);
    float* JXL_RESTRICT row1 = linear.PlaneRow(1, b + 1);
    float* JXL_RESTRICT row2 = linear.PlaneRow(2, b + 1);
    for (int r = 0; r < 256; ++r) {
      for (int g = 0; g < 256; ++g) {
        const int x = (r << 8) + g;
        row0[x] = r;
        row1[x] = g;
        row2[x] = b;
      }
    }
  }
  CodecInOut io;
  io.metadata.m.SetUintSamples(8);
  io.metadata.m.color_encoding = ColorEncoding::LinearSRGB();
  io.SetFromImage(std::move(linear), io.metadata.m.color_encoding);
  const ImageBundle& ib = io.Main();
  ThreadPool* null_pool = nullptr;
  Image3F opsin(ib.xsize(), ib.ysize());
  (void)ToXYB(ib, null_pool, &opsin);
  for (size_t c = 0; c < 3; ++c) {
    float minval = 1e10f;
    float maxval = -1e10f;
    int rgb_min = 0;
    int rgb_max = 0;
    for (int b = 0; b < 256; ++b) {
      const float* JXL_RESTRICT row = opsin.PlaneRow(c, b);
      for (int r = 0; r < 256; ++r) {
        for (int g = 0; g < 256; ++g) {
          float val = row[(r << 8) + g];
          if (val < minval) {
            minval = val;
            rgb_min = (r << 16) + (g << 8) + b;
          }
          if (val > maxval) {
            maxval = val;
            rgb_max = (r << 16) + (g << 8) + b;
          }
        }
      }
    }
    printf(
        "Opsin image plane %zu range: [%8.4f, %8.4f] "
        "center: %.12f, range: %.12f (RGBmin=%06x, RGBmax=%06x)\n",
        c, minval, maxval, 0.5 * (minval + maxval), 0.5 * (maxval - minval),
        rgb_min, rgb_max);
    // Ensure our constants are at least as wide as those obtained from sRGB.
  }
}

}  // namespace
}  // namespace jxl

int main() { jxl::PrintXybRange(); }
