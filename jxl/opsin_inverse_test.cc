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

#include "gtest/gtest.h"
#include "jxl/base/data_parallel.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/dec_xyb.h"
#include "jxl/enc_xyb.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_test_utils.h"

namespace jxl {
namespace {

TEST(OpsinInverseTest, LinearInverseInverts) {
  const uint32_t targets_bits = hwy::SupportedTargets();

  Image3F linear(128, 128);
  RandomFillImage(&linear, 255.0f);

  CodecInOut io;
  io.metadata.bits_per_sample = 32;
  io.metadata.floating_point_sample = true;
  io.metadata.color_encoding = ColorEncoding::LinearSRGB();
  io.SetFromImage(CopyImage(linear), io.metadata.color_encoding);
  ThreadPool* null_pool = nullptr;
  Image3F opsin(io.xsize(), io.ysize());
  ImageBundle unused_linear;
  (void)(ChooseToXYB(targets_bits)(io.Main(), null_pool, &opsin,
                                   &unused_linear));

  OpsinParams opsin_params;
  opsin_params.Init();
  ChooseOpsinToLinearInplace(targets_bits)(&opsin, /*pool=*/nullptr,
                                           opsin_params);

  VerifyRelativeError(linear, opsin, 3E-3, 2E-4);
}

}  // namespace
}  // namespace jxl
