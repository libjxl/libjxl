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

#include "jxl/luminance.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "jxl/image_test_utils.h"

namespace jxl {
namespace {

TEST(LuminanceTest, Scale) {
  constexpr float kFactor = 2;
  constexpr float kGamma = 2.2f;

  Image3F source_image(3, 1), expected_image(3, 1);
  for (size_t c = 0; c < 3; ++c) {
    float* const JXL_RESTRICT source_row = source_image.PlaneRow(c, 0);
    source_row[0] = 0.f;
    source_row[1] = 118.f;
    source_row[2] = 255.f;

    float* const JXL_RESTRICT expected_row = expected_image.PlaneRow(c, 0);
    for (size_t x = 0; x < source_image.xsize(); ++x) {
      expected_row[x] =
          std::pow(kFactor * std::pow(source_row[x], kGamma), 1.f / kGamma);
    }
  }
  ColorEncoding gamma_encoding = ColorEncoding::SRGB();
  ASSERT_TRUE(gamma_encoding.tf.SetGamma(1.f / kGamma));
  ASSERT_TRUE(gamma_encoding.CreateICC());

  CodecInOut io;
  io.SetFromImage(CopyImage(source_image), gamma_encoding);
  io.target_nits = kFactor * kDefaultIntensityTarget;

  ASSERT_TRUE(Map255ToTargetNits(&io, /*pool=*/nullptr));

  VerifyRelativeError(expected_image, io.Main().color(), 1e-1f, 1e-3f);

  ASSERT_TRUE(MapTargetNitsTo255(&io, /*pool=*/nullptr));

  VerifyRelativeError(source_image, io.Main().color(), 1e-1f, 1e-3f);
}

}  // namespace
}  // namespace jxl
