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

#include "jxl/ac_strategy.h"

#include <string.h>

#include <cmath>
#include <utility>

#include "gtest/gtest.h"
#include "jxl/common.h"
#include "jxl/dct.h"

namespace jxl {
namespace {

// Test that DCT -> IDCT is a noop.
void TestAcStrategyRoundtrip(AcStrategy::Type type) {
  AcStrategy acs = AcStrategy::FromRawStrategy(type);
  HWY_ALIGN float coeffs[AcStrategy::kMaxCoeffArea] = {};
  HWY_ALIGN float idct[AcStrategy::kMaxCoeffArea];
  for (size_t i = 0; i < 64 << acs.log2_covered_blocks(); i++) {
    HWY_ALIGN float input[AcStrategy::kMaxCoeffArea] = {};
    input[i] = 0.2;
    acs.TransformFromPixels(input, acs.covered_blocks_x() * 8, coeffs);
    ASSERT_NEAR(coeffs[0], 0.2 / (64 << acs.log2_covered_blocks()), 1e-6)
        << " i = " << i;
    acs.TransformToPixels(coeffs, idct, acs.covered_blocks_x() * 8);
    for (size_t j = 0; j < 64 << acs.log2_covered_blocks(); j++) {
      EXPECT_NEAR(idct[j], input[j], 1e-6) << "j = " << j << " i = " << i;
    }
  }
  // Test DC.
  std::fill(std::begin(idct), std::end(idct), 0);
  for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
    for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
      HWY_ALIGN float dc[AcStrategy::kMaxCoeffArea] = {};
      dc[y * acs.covered_blocks_x() * 8 + x] = 0.2;
      acs.LowestFrequenciesFromDC(dc, acs.covered_blocks_x() * 8, coeffs);
      acs.DCFromLowestFrequencies(coeffs, idct, acs.covered_blocks_x() * 8);
      for (size_t j = 0; j < 64 << acs.log2_covered_blocks(); j++) {
        EXPECT_NEAR(idct[j], dc[j], 1e-6)
            << "j = " << j << " x = " << x << " y = " << y;
      }
    }
  }
}

// Test that DC(2x2) -> DCT coefficients -> IDCT -> downsampled IDCT is a noop.
void TestAcStrategyRoundtripDownsample(AcStrategy::Type type) {
  AcStrategy acs = AcStrategy::FromRawStrategy(type);
  HWY_ALIGN float coeffs[AcStrategy::kMaxCoeffArea] = {};
  HWY_ALIGN float idct[AcStrategy::kMaxCoeffArea];
  for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
    for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
      HWY_ALIGN float dc[AcStrategy::kMaxCoeffArea] = {};
      dc[y * acs.covered_blocks_x() * 8 + x] = 0.2;
      acs.LowestFrequenciesFromDC(dc, acs.covered_blocks_x() * 8, coeffs);
      acs.TransformToPixels(coeffs, idct, acs.covered_blocks_x() * 8);
      // Downsample
      for (size_t dy = 0; dy < acs.covered_blocks_y(); dy++) {
        for (size_t dx = 0; dx < acs.covered_blocks_x(); dx++) {
          float sum = 0;
          for (size_t iy = 0; iy < 8; iy++) {
            for (size_t ix = 0; ix < 8; ix++) {
              sum += idct[(dy * 8 + iy) * 8 * acs.covered_blocks_x() + dx * 8 +
                          ix];
            }
          }
          sum /= 64.0f;
          EXPECT_NEAR(sum, dc[dy * 8 * acs.covered_blocks_x() + dx], 1e-6);
        }
      }
    }
  }
}

// Test that IDCT(block with zeros in the non-topleft corner) -> downsampled
// IDCT is the same as IDCT -> DC(2x2) of the same block.
void TestAcStrategyDownsample(AcStrategy::Type type) {
  AcStrategy acs = AcStrategy::FromRawStrategy(type);
  HWY_ALIGN float idct[AcStrategy::kMaxCoeffArea];
  HWY_ALIGN float idct_acs_downsampled[AcStrategy::kMaxCoeffArea] = {};
  size_t cx = acs.covered_blocks_y();
  size_t cy = acs.covered_blocks_x();
  CoefficientLayout(&cy, &cx);
  for (size_t y = 0; y < cy; y++) {
    for (size_t x = 0; x < cx; x++) {
      HWY_ALIGN float coeffs[AcStrategy::kMaxCoeffArea] = {};
      coeffs[y * cx * 8 + x] = 0.2;
      acs.TransformToPixels(coeffs, idct, acs.covered_blocks_x() * 8);
      acs.DCFromLowestFrequencies(coeffs, idct_acs_downsampled,
                                  acs.covered_blocks_x() * 8);
      // Downsample
      for (size_t dy = 0; dy < acs.covered_blocks_y(); dy++) {
        for (size_t dx = 0; dx < acs.covered_blocks_x(); dx++) {
          float sum = 0;
          for (size_t iy = 0; iy < 8; iy++) {
            for (size_t ix = 0; ix < 8; ix++) {
              sum += idct[(dy * 8 + iy) * 8 * acs.covered_blocks_x() + dx * 8 +
                          ix];
            }
          }
          sum /= 64;
          EXPECT_NEAR(
              sum, idct_acs_downsampled[dy * 8 * acs.covered_blocks_x() + dx],
              1e-6);
        }
      }
    }
  }
}

float I8(int N, int u) {
  float eps = u == 0 ? std::sqrt(0.5) : 1.0;
  return std::sqrt(2.0 / N) * eps * cos(u * kPi / (2.0 * N));
}

float D8(int N, int u) { return 1.0f / (N * I8(N, u)); }

float I(int N, int u) {
  if (N == 8)
    return I8(N, u);
  else
    return D8(N, u);
}

float D(int N, int u) {
  if (N == 8)
    return D8(N, u);
  else
    return I8(N, u);
}

float C(int N, int n, int x) {
  if (n > N) return 1.0 / C(n, N, x);
  if (n == N)
    return 1.0;
  else
    return cos(x * kPi / (2 * N)) * C(N / 2, n, x);
}

float ScaleF(int N, int n, int x) {
  return std::sqrt(n * N) * D(N, x) * I(n, x) * C(N, n, x);
}

TEST(AcStrategyTest, TestConstant32) {
  for (size_t i = 0; i < 32; i++) {
    EXPECT_NEAR(D(32, i), DCTScales<32>()[i], 1e-4);
    EXPECT_NEAR(I(32, i), IDCTScales<32>()[i], 1e-4);
  }
  for (int i = 0; i < 8; i++) {
    float e1 = C(32, 8, i) - DCTResampleScales<32, 8>::kScales[i];
    float e2 = C(8, 32, i) - DCTResampleScales<8, 32>::kScales[i];
    EXPECT_LT(std::fabs(e1), 1e-4);
    EXPECT_LT(std::fabs(e2), 1e-4);
  }
  EXPECT_NEAR(std::sqrt(8), square_root<8>::value, 1e-4);
  EXPECT_NEAR(std::sqrt(16), square_root<16>::value, 1e-4);
  for (int i = 0; i < 8; i++) {
    float e1 = ScaleF(32, 8, i) - DCTTotalResampleScale<32, 8>(i);
    EXPECT_LT(std::fabs(e1), 1e-4);
  }
  for (int i = 0; i < 4; i++) {
    float e1 = ScaleF(32, 4, i) - DCTTotalResampleScale<32, 4>(i);
    EXPECT_LT(std::fabs(e1), 1e-4);
  }
}

TEST(AcStrategyTest, TestConstant16) {
  for (size_t i = 0; i < 16; i++) {
    EXPECT_NEAR(D(16, i), DCTScales<16>()[i], 1e-4);
    EXPECT_NEAR(I(16, i), IDCTScales<16>()[i], 1e-4);
  }
  for (int i = 0; i < 4; i++) {
    float e1 = ScaleF(16, 4, i) - DCTTotalResampleScale<16, 4>(i);
    EXPECT_LT(std::fabs(e1), 1e-4);
  }
  for (int i = 0; i < 2; i++) {
    float e1 = ScaleF(16, 2, i) - DCTTotalResampleScale<16, 2>(i);
    EXPECT_LT(std::fabs(e1), 1e-4);
  }
}

TEST(AcStrategyTest, TestConstant8) {
  for (size_t i = 0; i < 8; i++) {
    EXPECT_NEAR(D(8, i), DCTScales<8>()[i], 1e-4);
    EXPECT_NEAR(I(8, i), IDCTScales<8>()[i], 1e-4);
  }
  for (int i = 0; i < 2; i++) {
    float e1 = ScaleF(8, 2, i) - DCTTotalResampleScale<8, 2>(i);
    EXPECT_LT(std::fabs(e1), 1e-4);
  }
  for (int i = 0; i < 1; i++) {
    float e1 = ScaleF(8, 1, i) - DCTTotalResampleScale<8, 1>(i);
    EXPECT_LT(std::fabs(e1), 1e-4);
  }
}

TEST(AcStrategyTest, DownsampleDCT) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT);
}
TEST(AcStrategyTest, DownsampleIDENTITY) {
  TestAcStrategyDownsample(AcStrategy::Type::IDENTITY);
}
TEST(AcStrategyTest, DownsampleDCT2X2) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT2X2);
}
TEST(AcStrategyTest, DownsampleDCT4X4) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT4X4);
}
TEST(AcStrategyTest, DownsampleDCT16X16) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT16X16);
}
TEST(AcStrategyTest, DownsampleDCT32X32) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT32X32);
}
TEST(AcStrategyTest, DownsampleDCT16X8) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT16X8);
}
TEST(AcStrategyTest, DownsampleDCT8X16) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT8X16);
}
TEST(AcStrategyTest, DownsampleDCT32X8) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT32X8);
}
TEST(AcStrategyTest, DownsampleDCT8X32) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT8X32);
}
TEST(AcStrategyTest, DownsampleDCT32X16) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT32X16);
}
TEST(AcStrategyTest, DownsampleDCT16X32) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT16X32);
}
TEST(AcStrategyTest, DownsampleDCT4X8) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT4X8);
}
TEST(AcStrategyTest, DownsampleDCT8X4) {
  TestAcStrategyDownsample(AcStrategy::Type::DCT8X4);
}
TEST(AcStrategyTest, DownsampleAFV0) {
  TestAcStrategyDownsample(AcStrategy::Type::AFV0);
}
TEST(AcStrategyTest, DownsampleAFV1) {
  TestAcStrategyDownsample(AcStrategy::Type::AFV1);
}

TEST(AcStrategyTest, RoundtripDownsampleDCT) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT);
}
TEST(AcStrategyTest, RoundtripDownsampleIDENTITY) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::IDENTITY);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT2X2) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT2X2);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT4X4) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT4X4);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT16X16) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT16X16);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT32X32) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT32X32);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT16X8) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT16X8);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT8X16) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT8X16);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT32X8) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT32X8);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT8X32) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT8X32);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT32X16) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT32X16);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT16X32) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT16X32);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT4X8) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT4X8);
}
TEST(AcStrategyTest, RoundtripDownsampleDCT8X4) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::DCT8X4);
}
TEST(AcStrategyTest, RoundtripDownsampleAFV0) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::AFV0);
}
TEST(AcStrategyTest, RoundtripDownsampleAFV1) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::AFV1);
}
TEST(AcStrategyTest, RoundtripDownsampleAFV2) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::AFV2);
}
TEST(AcStrategyTest, RoundtripDownsampleAFV3) {
  TestAcStrategyRoundtripDownsample(AcStrategy::Type::AFV3);
}

TEST(AcStrategyTest, RoundtripDCT) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT);
}
TEST(AcStrategyTest, RoundtripIDENTITY) {
  TestAcStrategyRoundtrip(AcStrategy::Type::IDENTITY);
}
TEST(AcStrategyTest, RoundtripDCT2X2) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT2X2);
}
TEST(AcStrategyTest, RoundtripDCT4X4) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT4X4);
}
TEST(AcStrategyTest, RoundtripDCT16X16) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT16X16);
}
TEST(AcStrategyTest, RoundtripDCT32X32) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT32X32);
}
TEST(AcStrategyTest, RoundtripDCT16X8) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT16X8);
}
TEST(AcStrategyTest, RoundtripDCT8X16) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT8X16);
}
TEST(AcStrategyTest, RoundtripDCT32X8) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT32X8);
}
TEST(AcStrategyTest, RoundtripDCT8X32) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT8X32);
}
TEST(AcStrategyTest, RoundtripDCT32X16) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT32X16);
}
TEST(AcStrategyTest, RoundtripDCT16X32) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT16X32);
}
TEST(AcStrategyTest, RoundtripDCT4X8) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT4X8);
}
TEST(AcStrategyTest, RoundtripDCT8X4) {
  TestAcStrategyRoundtrip(AcStrategy::Type::DCT8X4);
}
TEST(AcStrategyTest, RoundtripAFV0) {
  TestAcStrategyRoundtrip(AcStrategy::Type::AFV0);
}
TEST(AcStrategyTest, RoundtripAFV1) {
  TestAcStrategyRoundtrip(AcStrategy::Type::AFV1);
}
TEST(AcStrategyTest, RoundtripAFV2) {
  TestAcStrategyRoundtrip(AcStrategy::Type::AFV2);
}
TEST(AcStrategyTest, RoundtripAFV3) {
  TestAcStrategyRoundtrip(AcStrategy::Type::AFV3);
}

TEST(AcStrategyTest, RoundtripAFVDCT) {
  HWY_ALIGN float idct[16];
  for (size_t i = 0; i < 16; i++) {
    HWY_ALIGN float pixels[16] = {};
    pixels[i] = 1;
    HWY_ALIGN float coeffs[16] = {};

    AFVDCT4x4(pixels, coeffs);
    AFVIDCT4x4(coeffs, idct);
    for (size_t j = 0; j < 16; j++) {
      EXPECT_NEAR(idct[j], pixels[j], 1e-6);
    }
  }
}

TEST(AcStrategyTest, BenchmarkAFV) {
  AcStrategy acs = AcStrategy::FromRawStrategy(AcStrategy::Type::AFV0);
  HWY_ALIGN float pixels[64] = {1};
  HWY_ALIGN float coeffs[64] = {};
  for (size_t i = 0; i < 1 << 22; i++) {
    acs.TransformToPixels(coeffs, pixels, 8);
    acs.TransformFromPixels(pixels, 8, coeffs);
  }
}

TEST(AcStrategyTest, BenchmarkAFVDCT) {
  HWY_ALIGN float pixels[64] = {1};
  HWY_ALIGN float coeffs[64] = {};
  for (size_t i = 0; i < 1 << 22; i++) {
    AFVDCT4x4(pixels, coeffs);
    AFVIDCT4x4(coeffs, pixels);
  }
}
}  // namespace
}  // namespace jxl
