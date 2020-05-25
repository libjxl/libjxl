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

#include <hwy/base.h>  // HWY_ALIGN_MAX
#include <hwy/tests/test_util-inl.h>  // RunTest

#include "jxl/common.h"
#include "jxl/dct_scales.h"
#include "jxl/dec_transforms.h"
#include "jxl/enc_transforms.h"

namespace jxl {
namespace {

// Test that DCT -> IDCT is a noop.
class AcStrategyRoundtrip : public testing::TestWithParam<int> {
 protected:
  void Run() {
    const AcStrategy::Type type = static_cast<AcStrategy::Type>(GetParam());
    const auto from_pixels = ChooseTransformFromPixels();
    const auto to_pixels = ChooseTransformToPixels();
    const auto from_dc = ChooseLowestFrequenciesFromDC();
    const auto to_dc = ChooseDCFromLowestFrequencies();
    const AcStrategy acs = AcStrategy::FromRawStrategy(type);

    HWY_ALIGN_MAX float coeffs[AcStrategy::kMaxCoeffArea] = {};
    HWY_ALIGN_MAX float idct[AcStrategy::kMaxCoeffArea];

    for (size_t i = 0; i < 64 << acs.log2_covered_blocks(); i++) {
      HWY_ALIGN_MAX float input[AcStrategy::kMaxCoeffArea] = {};
      input[i] = 0.2f;
      from_pixels(type, input, acs.covered_blocks_x() * 8, coeffs);
      ASSERT_NEAR(coeffs[0], 0.2 / (64 << acs.log2_covered_blocks()), 1e-6)
          << " i = " << i;
      to_pixels(type, coeffs, idct, acs.covered_blocks_x() * 8);
      for (size_t j = 0; j < 64 << acs.log2_covered_blocks(); j++) {
        ASSERT_NEAR(idct[j], input[j], 1e-6)
            << "j = " << j << " i = " << i << " acs " << type;
      }
    }
    // Test DC.
    std::fill(std::begin(idct), std::end(idct), 0);
    for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
      for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
        HWY_ALIGN_MAX float dc[AcStrategy::kMaxCoeffArea] = {};
        dc[y * acs.covered_blocks_x() * 8 + x] = 0.2;
        from_dc(type, dc, acs.covered_blocks_x() * 8, coeffs);
        to_dc(type, coeffs, idct, acs.covered_blocks_x() * 8);
        for (size_t j = 0; j < 64 << acs.log2_covered_blocks(); j++) {
          ASSERT_NEAR(idct[j], dc[j], 1e-6)
              << "j = " << j << " x = " << x << " y = " << y << " acs " << type;
        }
      }
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    AcStrategyRoundtripInstantiation, AcStrategyRoundtrip,
    ::testing::Range(0, int(AcStrategy::Type::kNumValidStrategies)));

TEST_P(AcStrategyRoundtrip, Test) {
  hwy::RunTest([this]() { Run(); });
}

// Test that DC(2x2) -> DCT coefficients -> IDCT -> downsampled IDCT is a noop.
class AcStrategyRoundtripDownsample : public testing::TestWithParam<int> {
 protected:
  void Run() {
    const AcStrategy::Type type = static_cast<AcStrategy::Type>(GetParam());
    const auto from_dc = ChooseLowestFrequenciesFromDC();
    const auto to_pixels = ChooseTransformToPixels();
    const AcStrategy acs = AcStrategy::FromRawStrategy(type);

    HWY_ALIGN_MAX float coeffs[AcStrategy::kMaxCoeffArea] = {};
    HWY_ALIGN_MAX float idct[AcStrategy::kMaxCoeffArea];

    for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
      for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
        HWY_ALIGN_MAX float dc[AcStrategy::kMaxCoeffArea] = {};
        dc[y * acs.covered_blocks_x() * 8 + x] = 0.2f;
        from_dc(type, dc, acs.covered_blocks_x() * 8, coeffs);
        to_pixels(type, coeffs, idct, acs.covered_blocks_x() * 8);
        // Downsample
        for (size_t dy = 0; dy < acs.covered_blocks_y(); dy++) {
          for (size_t dx = 0; dx < acs.covered_blocks_x(); dx++) {
            float sum = 0;
            for (size_t iy = 0; iy < 8; iy++) {
              for (size_t ix = 0; ix < 8; ix++) {
                sum += idct[(dy * 8 + iy) * 8 * acs.covered_blocks_x() +
                            dx * 8 + ix];
              }
            }
            sum /= 64.0f;
            ASSERT_NEAR(sum, dc[dy * 8 * acs.covered_blocks_x() + dx], 1e-6)
                << "acs " << type;
          }
        }
      }
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    AcStrategyRoundtripDownsampleInstantiation, AcStrategyRoundtripDownsample,
    ::testing::Range(0, int(AcStrategy::Type::kNumValidStrategies)));

TEST_P(AcStrategyRoundtripDownsample, Test) {
  hwy::RunTest([this]() { Run(); });
}

// Test that IDCT(block with zeros in the non-topleft corner) -> downsampled
// IDCT is the same as IDCT -> DC(2x2) of the same block.
class AcStrategyDownsample : public testing::TestWithParam<int> {
 protected:
  void Run() {
    const AcStrategy::Type type = static_cast<AcStrategy::Type>(GetParam());
    const auto to_pixels = ChooseTransformToPixels();
    const auto to_dc = ChooseDCFromLowestFrequencies();
    const AcStrategy acs = AcStrategy::FromRawStrategy(type);
    size_t cx = acs.covered_blocks_y();
    size_t cy = acs.covered_blocks_x();
    CoefficientLayout(&cy, &cx);

    HWY_ALIGN_MAX float idct[AcStrategy::kMaxCoeffArea];
    HWY_ALIGN_MAX float idct_acs_downsampled[AcStrategy::kMaxCoeffArea] = {};

    for (size_t y = 0; y < cy; y++) {
      for (size_t x = 0; x < cx; x++) {
        HWY_ALIGN_MAX float coeffs[AcStrategy::kMaxCoeffArea] = {};
        coeffs[y * cx * 8 + x] = 0.2f;
        to_pixels(type, coeffs, idct, acs.covered_blocks_x() * 8);
        to_dc(type, coeffs, idct_acs_downsampled, acs.covered_blocks_x() * 8);
        // Downsample
        for (size_t dy = 0; dy < acs.covered_blocks_y(); dy++) {
          for (size_t dx = 0; dx < acs.covered_blocks_x(); dx++) {
            float sum = 0;
            for (size_t iy = 0; iy < 8; iy++) {
              for (size_t ix = 0; ix < 8; ix++) {
                sum += idct[(dy * 8 + iy) * 8 * acs.covered_blocks_x() +
                            dx * 8 + ix];
              }
            }
            sum /= 64;
            ASSERT_NEAR(
                sum, idct_acs_downsampled[dy * 8 * acs.covered_blocks_x() + dx],
                1e-6)
                << " acs " << type;
          }
        }
      }
    }
  }
};

INSTANTIATE_TEST_SUITE_P(
    AcStrategyDownsampleInstantiation, AcStrategyDownsample,
    ::testing::Range(0, int(AcStrategy::Type::kNumValidStrategies)));

TEST_P(AcStrategyDownsample, Test) {
  hwy::RunTest([this]() { Run(); });
}

float I8(int N, int u) {
  float eps = u == 0 ? std::sqrt(0.5) : 1.0;
  return std::sqrt(2.0 / N) * eps * cos(u * kPi / (2.0 * N));
}

float D8(int N, int u) { return 1.0f / (N * I8(N, u)); }

float I(int N, int u) {
  if (N == 8) {
    return I8(N, u);
  } else {
    return D8(N, u);
  }
}

float D(int N, int u) {
  if (N == 8) {
    return D8(N, u);
  } else {
    return I8(N, u);
  }
}

float C(int N, int n, int x) {
  if (n > N) return 1.0 / C(n, N, x);
  if (n == N) {
    return 1.0;
  } else {
    return cos(x * kPi / (2 * N)) * C(N / 2, n, x);
  }
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

TEST(AcStrategyTest, RoundtripAFVDCT) {
  hwy::RunTest([]() {
    const auto do_dct = ChooseAFVDCT4x4();
    const auto do_idct = ChooseAFVIDCT4x4();
    HWY_ALIGN_MAX float idct[16];
    for (size_t i = 0; i < 16; i++) {
      HWY_ALIGN_MAX float pixels[16] = {};
      pixels[i] = 1;
      HWY_ALIGN_MAX float coeffs[16] = {};

      do_dct(pixels, coeffs);
      do_idct(coeffs, idct);
      for (size_t j = 0; j < 16; j++) {
        EXPECT_NEAR(idct[j], pixels[j], 1e-6);
      }
    }
  });
}

TEST(AcStrategyTest, BenchmarkAFV) {
  hwy::RunTest([]() {
    const auto from_pixels = ChooseTransformFromPixels();
    const auto to_pixels = ChooseTransformToPixels();

    const AcStrategy::Type type = AcStrategy::Type::AFV0;
    HWY_ALIGN_MAX float pixels[64] = {1};
    HWY_ALIGN_MAX float coeffs[64] = {};
    for (size_t i = 0; i < 1 << 14; i++) {
      to_pixels(type, coeffs, pixels, 8);
      from_pixels(type, pixels, 8, coeffs);
    }
    EXPECT_NEAR(pixels[0], 0.0, 1E-6);
  });
}

TEST(AcStrategyTest, BenchmarkAFVDCT) {
  hwy::RunTest([]() {
    const auto do_dct = ChooseAFVDCT4x4();
    const auto do_idct = ChooseAFVIDCT4x4();

    HWY_ALIGN_MAX float pixels[64] = {1};
    HWY_ALIGN_MAX float coeffs[64] = {};
    for (size_t i = 0; i < 1 << 14; i++) {
      do_dct(pixels, coeffs);
      do_idct(coeffs, pixels);
    }
    EXPECT_NEAR(pixels[0], 1.0, 1E-6);
  });
}

}  // namespace
}  // namespace jxl
