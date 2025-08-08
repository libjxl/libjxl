// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/memory_manager.h>

#include <cstddef>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

TEST(OpsinInverseTest, LinearInverseInverts) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  constexpr size_t kDim = 128;
  JXL_TEST_ASSIGN_OR_DIE(Image3F linear,
                         Image3F::Create(memory_manager, kDim, kDim));
  RandomFillImage(&linear, 0.0f, 1.0f);
  JXL_TEST_ASSIGN_OR_DIE(Image3F opsin,
                         Image3F::Create(memory_manager, kDim, kDim));
  ASSERT_TRUE(CopyImageTo(linear, &opsin));
  ASSERT_TRUE(ToXYB(ColorEncoding::LinearSRGB(), kDefaultIntensityTarget,
                    nullptr, nullptr, &opsin, *JxlGetDefaultCms(), nullptr));

  JXL_TEST_ASSIGN_OR_DIE(Image3F relinear,
                         Image3F::Create(memory_manager, kDim, kDim));

  OpsinParams opsin_params;
  opsin_params.Init(/*intensity_target=*/255.0f);
  ASSERT_TRUE(OpsinToLinear(opsin, Rect(opsin), /*pool=*/nullptr, &relinear,
                            opsin_params));

  JXL_TEST_ASSERT_OK(VerifyRelativeError(linear, relinear, 3E-3, 2E-4, _));
}

// Transform YCbCr to RGB.
// Bt.601 to match JPEG/JFIF. Inputs are _signed_ YCbCr values suitable for DCT,
// see F.1.1.3 of T.81 (because our data type is float, there is no need to add
// a bias to make the values unsigned).
// Could be performed in-place (i.e. Y, Cb and Cr could alias R, B and B).
void YcbcrToRgb(const Image3F& ycbcr, Image3F* rgb, const Rect& rect) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();
  if ((xsize == 0) || (ysize == 0)) return;

  // Full-range BT.601 as defined by JFIF Clause 7:
  // https://www.itu.int/rec/T-REC-T.871-201105-I/en
  const float c128 = 128.0f / 255;
  const float crcr = 1.402f;
  const float cgcb = -0.114f * 1.772f / 0.587f;
  const float cgcr = -0.299f * 1.402f / 0.587f;
  const float cbcb = 1.772f;

  for (size_t y = 0; y < ysize; y++) {
    const float* y_row = rect.ConstPlaneRow(ycbcr, 1, y);
    const float* cb_row = rect.ConstPlaneRow(ycbcr, 0, y);
    const float* cr_row = rect.ConstPlaneRow(ycbcr, 2, y);
    float* r_row = rect.PlaneRow(rgb, 0, y);
    float* g_row = rect.PlaneRow(rgb, 1, y);
    float* b_row = rect.PlaneRow(rgb, 2, y);
    for (size_t x = 0; x < xsize; x++) {
      const float y_vec = y_row[x] + c128;
      const float cb_vec = cb_row[x];
      const float cr_vec = cr_row[x];
      const float r_vec = crcr * cr_vec + y_vec;
      const float g_vec = cgcr * cr_vec + cgcb * cb_vec + y_vec;
      const float b_vec = cbcb * cb_vec + y_vec;
      r_row[x] = r_vec;
      g_row[x] = g_vec;
      b_row[x] = b_vec;
    }
  }
}

TEST(OpsinInverseTest, YcbCrInverts) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  JXL_TEST_ASSIGN_OR_DIE(Image3F rgb,
                         Image3F::Create(memory_manager, 128, 128));
  RandomFillImage(&rgb, 0.0f, 1.0f);

  ThreadPool* null_pool = nullptr;
  JXL_TEST_ASSIGN_OR_DIE(
      Image3F ycbcr, Image3F::Create(memory_manager, rgb.xsize(), rgb.ysize()));
  EXPECT_TRUE(RgbToYcbcr(rgb.Plane(0), rgb.Plane(1), rgb.Plane(2),
                         &ycbcr.Plane(1), &ycbcr.Plane(0), &ycbcr.Plane(2),
                         null_pool));

  JXL_TEST_ASSIGN_OR_DIE(
      Image3F rgb2, Image3F::Create(memory_manager, rgb.xsize(), rgb.ysize()));
  YcbcrToRgb(ycbcr, &rgb2, Rect(rgb));

  JXL_TEST_ASSERT_OK(VerifyRelativeError(rgb, rgb2, 4E-5, 4E-7, _));
}

}  // namespace
}  // namespace jxl
