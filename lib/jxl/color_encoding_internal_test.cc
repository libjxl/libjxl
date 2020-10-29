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

#include "lib/jxl/color_encoding_internal.h"

#include <stdio.h>

#include "gtest/gtest.h"
#include "lib/jxl/test_utils.h"

namespace jxl {
namespace {

TEST(ColorEncodingTest, RoundTripAll) {
  for (const test::ColorEncodingDescriptor& cdesc : test::AllEncodings()) {
    const ColorEncoding c_original = test::ColorEncodingFromDescriptor(cdesc);
    // Verify Set(Get) yields the same white point/primaries/gamma.
    {
      ColorEncoding c;
      EXPECT_TRUE(c.SetWhitePoint(c_original.GetWhitePoint()));
      EXPECT_EQ(c_original.white_point, c.white_point);
    }
    {
      ColorEncoding c;
      EXPECT_TRUE(c.SetPrimaries(c_original.GetPrimaries()));
      EXPECT_EQ(c_original.primaries, c.primaries);
    }
    if (c_original.tf.IsGamma()) {
      ColorEncoding c;
      EXPECT_TRUE(c.tf.SetGamma(c_original.tf.GetGamma()));
      EXPECT_TRUE(c_original.tf.IsSame(c.tf));
    }

    // Verify ParseDescription(Description) yields the same ColorEncoding
    {
      const std::string description = Description(c_original);
      printf("%s\n", description.c_str());
      ColorEncoding c;
      EXPECT_TRUE(ParseDescription(description, &c));
      EXPECT_TRUE(c_original.SameColorEncoding(c));
    }
  }
}

// Verify Set(Get) for specific custom values

TEST(ColorEncodingTest, NanGamma) {
  const std::string description = "Gra_2_Per_gnan";
  ColorEncoding c;
  EXPECT_FALSE(ParseDescription(description, &c));
}

TEST(ColorEncodingTest, CustomWhitePoint) {
  ColorEncoding c;
  // Nonsensical values
  CIExy xy_in;
  xy_in.x = 0.8;
  xy_in.y = 0.01;
  EXPECT_TRUE(c.SetWhitePoint(xy_in));
  const CIExy xy = c.GetWhitePoint();

  ColorEncoding c2;
  EXPECT_TRUE(c2.SetWhitePoint(xy));
  EXPECT_TRUE(c.SameColorSpace(c2));
}

TEST(ColorEncodingTest, CustomPrimaries) {
  ColorEncoding c;
  PrimariesCIExy xy_in;
  // Nonsensical values
  xy_in.r.x = -0.01;
  xy_in.r.y = 0.2;
  xy_in.g.x = 0.4;
  xy_in.g.y = 0.401;
  xy_in.b.x = 1.1;
  xy_in.b.y = -1.2;
  EXPECT_TRUE(c.SetPrimaries(xy_in));
  const PrimariesCIExy xy = c.GetPrimaries();

  ColorEncoding c2;
  EXPECT_TRUE(c2.SetPrimaries(xy));
  EXPECT_TRUE(c.SameColorSpace(c2));
}

TEST(ColorEncodingTest, CustomGamma) {
  ColorEncoding c;
#ifndef JXL_CRASH_ON_ERROR
  EXPECT_FALSE(c.tf.SetGamma(0.0));
  EXPECT_FALSE(c.tf.SetGamma(-1E-6));
  EXPECT_FALSE(c.tf.SetGamma(1.001));
#endif
  EXPECT_TRUE(c.tf.SetGamma(1.0));
  EXPECT_FALSE(c.tf.IsGamma());
  EXPECT_TRUE(c.tf.IsLinear());

  EXPECT_TRUE(c.tf.SetGamma(0.123));
  EXPECT_TRUE(c.tf.IsGamma());
  const double gamma = c.tf.GetGamma();

  ColorEncoding c2;
  EXPECT_TRUE(c2.tf.SetGamma(gamma));
  EXPECT_TRUE(c.SameColorEncoding(c2));
  EXPECT_TRUE(c2.tf.IsGamma());
}

}  // namespace
}  // namespace jxl
