// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/memory_manager.h>

#include <cstdio>
#include <cstdlib>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/image.h"
#include "tools/cmdline.h"
#include "tools/no_memory_manager.h"

namespace jpegxl {
namespace tools {
namespace {

using ::jxl::ColorEncoding;
using ::jxl::Image3F;
using ::jxl::Status;

Status PrintXybRange() {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  JXL_ASSIGN_OR_RETURN(Image3F linear,
                       Image3F::Create(memory_manager, 1u << 16, 257));
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
  JXL_RETURN_IF_ERROR(ToXYB(ColorEncoding::LinearSRGB(),
                            jxl::kDefaultIntensityTarget, nullptr, nullptr,
                            &linear, *JxlGetDefaultCms(), nullptr));
  Image3F& opsin = linear;
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
    printf("Opsin image plane %" PRIuS
           " range: [%8.4f, %8.4f] "
           "center: %.12f, range: %.12f (RGBmin=%06x, RGBmax=%06x)\n",
           c, minval, maxval, 0.5 * (minval + maxval), 0.5 * (maxval - minval),
           rgb_min, rgb_max);
    // Ensure our constants are at least as wide as those obtained from sRGB.
  }
  return true;
}

}  // namespace
}  // namespace tools
}  // namespace jpegxl

// NOLINTBEGIN
/* clang-format off */
/*
 * Expected output:
 *
 * Opsin image plane 0 range: [ -0.0979,   0.1799] center: 0.040977656841, range: 0.138920247555 (RGBmin=00ff01, RGBmax=ff0001)
 * Opsin image plane 1 range: [  0.0000,   6.1848] center: 3.092378616333, range: 3.092378616333 (RGBmin=000000, RGBmax=ffffff)
 * Opsin image plane 2 range: [  0.0000,   6.1808] center: 3.090413093567, range: 3.090413093567 (RGBmin=000000, RGBmax=ffffff)
 */
/* clang-format on */
// NOLINTEND
int main() {
  JPEGXL_TOOLS_CHECK(jpegxl::tools::PrintXybRange());
  return EXIT_SUCCESS;
}
