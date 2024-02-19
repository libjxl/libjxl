// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>

#include <utility>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

TEST(OpsinInverseTest, LinearInverseInverts) {
  JXL_ASSIGN_OR_DIE(Image3F linear, Image3F::Create(128, 128));
  RandomFillImage(&linear, 0.0f, 1.0f);

  CodecInOut io;
  io.metadata.m.SetFloat32Samples();
  io.metadata.m.color_encoding = ColorEncoding::LinearSRGB();
  JXL_ASSIGN_OR_DIE(Image3F linear2, Image3F::Create(128, 128));
  CopyImageTo(linear, &linear2);
  io.SetFromImage(std::move(linear2), io.metadata.m.color_encoding);
  ThreadPool* null_pool = nullptr;
  JXL_ASSIGN_OR_DIE(Image3F opsin, Image3F::Create(io.xsize(), io.ysize()));
  (void)ToXYB(io.Main(), null_pool, &opsin, *JxlGetDefaultCms());

  OpsinParams opsin_params;
  opsin_params.Init(/*intensity_target=*/255.0f);
  OpsinToLinearInplace(&opsin, /*pool=*/nullptr, opsin_params);

  JXL_ASSERT_OK(VerifyRelativeError(linear, opsin, 3E-3, 2E-4, _));
}

TEST(OpsinInverseTest, YcbCrInverts) {
  JXL_ASSIGN_OR_DIE(Image3F rgb, Image3F::Create(128, 128));
  RandomFillImage(&rgb, 0.0f, 1.0f);

  ThreadPool* null_pool = nullptr;
  JXL_ASSIGN_OR_DIE(Image3F ycbcr, Image3F::Create(rgb.xsize(), rgb.ysize()));
  EXPECT_TRUE(RgbToYcbcr(rgb.Plane(0), rgb.Plane(1), rgb.Plane(2),
                         &ycbcr.Plane(1), &ycbcr.Plane(0), &ycbcr.Plane(2),
                         null_pool));

  JXL_ASSIGN_OR_DIE(Image3F rgb2, Image3F::Create(rgb.xsize(), rgb.ysize()));
  YcbcrToRgb(ycbcr, &rgb2, Rect(rgb));

  JXL_ASSERT_OK(VerifyRelativeError(rgb, rgb2, 4E-5, 4E-7, _));
}

}  // namespace
}  // namespace jxl
