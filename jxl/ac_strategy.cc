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

#include <algorithm>
#include <hwy/static_targets.h>
#include <numeric>  // iota
#include <type_traits>
#include <utility>

#include "jxl/base/profiler.h"
#include "jxl/block.h"
#include "jxl/common.h"
#include "jxl/dct.h"
#include "jxl/image_ops.h"

namespace jxl {
namespace {

template <size_t ROWS, size_t COLS>
struct DoDCT {
  template <typename From, typename To>
  HWY_ATTR void operator()(const From& from, const To& to) {
    ComputeScaledDCT<ROWS, COLS>()(from, to);
  }
};

template <size_t N>
struct DoDCT<N, N> {
  template <typename From, typename To>
  HWY_ATTR void operator()(const From& from, const To& to) {
    ComputeTransposedScaledDCT<N>()(from, to);
  }
};

template <size_t ROWS, size_t COLS>
struct DoIDCT {
  template <typename From, typename To>
  HWY_ATTR void operator()(const From& from, const To& to) {
    ComputeScaledIDCT<ROWS, COLS>()(from, to);
  }
};

template <size_t N>
struct DoIDCT<N, N> {
  template <typename From, typename To>
  HWY_ATTR void operator()(const From& from, const To& to) const {
    ComputeTransposedScaledIDCT<N>()(from, to);
  }
};

// Computes the lowest-frequency LF_ROWSxLF_COLS-sized square in output, which
// is a DCT_ROWS*DCT_COLS-sized DCT block, by doing a ROWS*COLS DCT on the
// input block.
template <size_t DCT_ROWS, size_t DCT_COLS, size_t LF_ROWS, size_t LF_COLS,
          size_t ROWS, size_t COLS>
HWY_ATTR JXL_INLINE void ReinterpretingDCT(const float* input,
                                           const size_t input_stride,
                                           float* output,
                                           const size_t output_stride) {
  static_assert(LF_ROWS == ROWS,
                "ReinterpretingDCT should only be called with LF == N");
  static_assert(LF_COLS == COLS,
                "ReinterpretingDCT should only be called with LF == N");
  HWY_ALIGN float block[ROWS * COLS] = {};
  for (size_t y = 0; y < ROWS; y++) {
    for (size_t x = 0; x < COLS; x++) {
      block[y * COLS + x] = input[y * input_stride + x];
    }
  }

  constexpr size_t OUT_ROWS = CoefficientRows(ROWS, COLS);
  constexpr size_t OUT_COLS = CoefficientColumns(ROWS, COLS);
  DoDCT<ROWS, COLS>()(FromBlock<ROWS, COLS>(block),
                      ScaleToBlock<OUT_ROWS, OUT_COLS>(block));
  if (ROWS < COLS) {
    for (size_t y = 0; y < LF_ROWS; y++) {
      for (size_t x = 0; x < LF_COLS; x++) {
        output[y * output_stride + x] =
            block[y * COLS + x] * DCTTotalResampleScale<ROWS, DCT_ROWS>(y) *
            DCTTotalResampleScale<COLS, DCT_COLS>(x);
      }
    }
  } else {
    for (size_t y = 0; y < LF_COLS; y++) {
      for (size_t x = 0; x < LF_ROWS; x++) {
        output[y * output_stride + x] =
            block[y * ROWS + x] * DCTTotalResampleScale<COLS, DCT_COLS>(y) *
            DCTTotalResampleScale<ROWS, DCT_ROWS>(x);
      }
    }
  }
}

// Inverse of ReinterpretingDCT.
template <size_t DCT_ROWS, size_t DCT_COLS, size_t LF_ROWS, size_t LF_COLS,
          size_t ROWS, size_t COLS>
HWY_ATTR JXL_INLINE void ReinterpretingIDCT(const float* input,
                                            const size_t input_stride,
                                            float* output,
                                            const size_t output_stride) {
  HWY_ALIGN float block[ROWS * COLS] = {};
  if (ROWS < COLS) {
    for (size_t y = 0; y < LF_ROWS; y++) {
      for (size_t x = 0; x < LF_COLS; x++) {
        block[y * COLS + x] = input[y * input_stride + x] *
                              DCTTotalResampleScale<DCT_ROWS, ROWS>(y) *
                              DCTTotalResampleScale<DCT_COLS, COLS>(x);
      }
    }
  } else {
    for (size_t y = 0; y < LF_COLS; y++) {
      for (size_t x = 0; x < LF_ROWS; x++) {
        block[y * ROWS + x] = input[y * input_stride + x] *
                              DCTTotalResampleScale<DCT_COLS, COLS>(y) *
                              DCTTotalResampleScale<DCT_ROWS, ROWS>(x);
      }
    }
  }

  constexpr size_t IN_ROWS = CoefficientRows(ROWS, COLS);
  constexpr size_t IN_COLS = CoefficientColumns(ROWS, COLS);
  DoIDCT<ROWS, COLS>()(FromBlock<IN_ROWS, IN_COLS>(block),
                       ToBlock<ROWS, COLS>(block));

  for (size_t y = 0; y < ROWS; y++) {
    for (size_t x = 0; x < COLS; x++) {
      output[y * output_stride + x] = block[y * COLS + x];
    }
  }
}

template <size_t S>
void DCT2TopBlock(const float* block, size_t stride, float* out) {
  static_assert(kBlockDim % S == 0, "S should be a divisor of kBlockDim");
  static_assert(S % 2 == 0, "S should be even");
  float temp[kDCTBlockSize];
  constexpr size_t num_2x2 = S / 2;
  for (size_t y = 0; y < num_2x2; y++) {
    for (size_t x = 0; x < num_2x2; x++) {
      float c00 = block[y * 2 * stride + x * 2];
      float c01 = block[y * 2 * stride + x * 2 + 1];
      float c10 = block[(y * 2 + 1) * stride + x * 2];
      float c11 = block[(y * 2 + 1) * stride + x * 2 + 1];
      float r00 = c00 + c01 + c10 + c11;
      float r01 = c00 + c01 - c10 - c11;
      float r10 = c00 - c01 + c10 - c11;
      float r11 = c00 - c01 - c10 + c11;
      r00 *= 0.25f;
      r01 *= 0.25f;
      r10 *= 0.25f;
      r11 *= 0.25f;
      temp[y * kBlockDim + x] = r00;
      temp[y * kBlockDim + num_2x2 + x] = r01;
      temp[(y + num_2x2) * kBlockDim + x] = r10;
      temp[(y + num_2x2) * kBlockDim + num_2x2 + x] = r11;
    }
  }
  for (size_t y = 0; y < S; y++) {
    for (size_t x = 0; x < S; x++) {
      out[y * kBlockDim + x] = temp[y * kBlockDim + x];
    }
  }
}

template <size_t S>
void IDCT2TopBlock(const float* block, size_t stride_out, float* out) {
  static_assert(kBlockDim % S == 0, "S should be a divisor of kBlockDim");
  static_assert(S % 2 == 0, "S should be even");
  float temp[kDCTBlockSize];
  constexpr size_t num_2x2 = S / 2;
  for (size_t y = 0; y < num_2x2; y++) {
    for (size_t x = 0; x < num_2x2; x++) {
      float c00 = block[y * kBlockDim + x];
      float c01 = block[y * kBlockDim + num_2x2 + x];
      float c10 = block[(y + num_2x2) * kBlockDim + x];
      float c11 = block[(y + num_2x2) * kBlockDim + num_2x2 + x];
      float r00 = c00 + c01 + c10 + c11;
      float r01 = c00 + c01 - c10 - c11;
      float r10 = c00 - c01 + c10 - c11;
      float r11 = c00 - c01 - c10 + c11;
      temp[y * 2 * kBlockDim + x * 2] = r00;
      temp[y * 2 * kBlockDim + x * 2 + 1] = r01;
      temp[(y * 2 + 1) * kBlockDim + x * 2] = r10;
      temp[(y * 2 + 1) * kBlockDim + x * 2 + 1] = r11;
    }
  }
  for (size_t y = 0; y < S; y++) {
    for (size_t x = 0; x < S; x++) {
      out[y * stride_out + x] = temp[y * kBlockDim + x];
    }
  }
}

// Tries to generalize zig-zag order to non-square blocks. Surprisingly, in
// square block frequency along the (i + j == const) diagonals is roughly the
// same. For historical reasons, consecutive diagonals are traversed
// in alternating directions - so called "zig-zag" (or "snake") order.
static AcStrategy::CoeffOrderAndLut ComputeNaturalCoeffOrder() {
  AcStrategy::CoeffOrderAndLut coeff;
  for (size_t s = 0; s < AcStrategy::kNumValidStrategies; s++) {
    const AcStrategy acs = AcStrategy::FromRawStrategy(s);
    size_t cx = acs.covered_blocks_x();
    size_t cy = acs.covered_blocks_y();
    CoefficientLayout(&cy, &cx);
    const size_t num_coeffs = kDCTBlockSize * cx * cy;
    coeff_order_t* JXL_RESTRICT order_start =
        coeff.order + s * AcStrategy::kMaxCoeffArea;
    coeff_order_t* JXL_RESTRICT lut_start =
        coeff.lut + s * AcStrategy::kMaxCoeffArea;
    std::iota(order_start, order_start + num_coeffs, 0);

    auto compute_key = [cx, cy](int32_t pos) {
      JXL_DASSERT(cx != 0 && cy != 0);
      size_t y = pos / (cx * kBlockDim);
      size_t x = pos % (cx * kBlockDim);
      // Ensure that LLFs are first in the order.
      if (x < cx && y < cy) {
        return std::make_pair(-1, static_cast<int>(y * cx + x));
      }
      int max_dim = std::max(cx, cy);
      int scaled_y = y * max_dim / cy;
      int scaled_x = x * max_dim / cx;
      return std::make_pair(scaled_x + scaled_y, (scaled_x + scaled_y) % 2 == 0
                                                     ? scaled_x - scaled_y
                                                     : scaled_y - scaled_x);
    };

    std::sort(order_start, order_start + num_coeffs,
              [compute_key](int32_t pos, int32_t other_pos) {
                return compute_key(pos) < compute_key(other_pos);
              });

    for (size_t i = 0; i < num_coeffs; i++) {
      lut_start[order_start[i]] = i;
    }
  }
  return coeff;
};

HWY_ALIGN constexpr float k4x4AFVBasisTranspose[16][16] = {
    {
        0.2500000000000000,
        0.8769029297991420,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        -0.4105377591765233,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
    },
    {
        0.2500000000000000,
        0.2206518106944235,
        0.0000000000000000,
        0.0000000000000000,
        -0.7071067811865474,
        0.6235485373547691,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
    },
    {
        0.2500000000000000,
        -0.1014005039375376,
        0.4067007583026075,
        -0.2125574805828875,
        0.0000000000000000,
        -0.0643507165794627,
        -0.4517556589999482,
        -0.3046847507248690,
        0.3017929516615495,
        0.4082482904638627,
        0.1747866975480809,
        -0.2110560104933578,
        -0.1426608480880726,
        -0.1381354035075859,
        -0.1743760259965107,
        0.1135498731499434,
    },
    {
        0.2500000000000000,
        -0.1014005039375375,
        0.4444481661973445,
        0.3085497062849767,
        0.0000000000000000,
        -0.0643507165794627,
        0.1585450355184006,
        0.5112616136591823,
        0.2579236279634118,
        0.0000000000000000,
        0.0812611176717539,
        0.1856718091610980,
        -0.3416446842253372,
        0.3302282550303788,
        0.0702790691196284,
        -0.0741750459581035,
    },
    {
        0.2500000000000000,
        0.2206518106944236,
        0.0000000000000000,
        0.0000000000000000,
        0.7071067811865476,
        0.6235485373547694,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
    },
    {
        0.2500000000000000,
        -0.1014005039375378,
        0.0000000000000000,
        0.4706702258572536,
        0.0000000000000000,
        -0.0643507165794628,
        -0.0403851516082220,
        0.0000000000000000,
        0.1627234014286620,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.7367497537172237,
        0.0875511500058708,
        -0.2921026642334881,
        0.1940289303259434,
    },
    {
        0.2500000000000000,
        -0.1014005039375377,
        0.1957439937204294,
        -0.1621205195722993,
        0.0000000000000000,
        -0.0643507165794628,
        0.0074182263792424,
        -0.2904801297289980,
        0.0952002265347504,
        0.0000000000000000,
        -0.3675398009862027,
        0.4921585901373873,
        0.2462710772207515,
        -0.0794670660590957,
        0.3623817333531167,
        -0.4351904965232280,
    },
    {
        0.2500000000000000,
        -0.1014005039375376,
        0.2929100136981264,
        0.0000000000000000,
        0.0000000000000000,
        -0.0643507165794627,
        0.3935103426921017,
        -0.0657870154914280,
        0.0000000000000000,
        -0.4082482904638628,
        -0.3078822139579090,
        -0.3852501370925192,
        -0.0857401903551931,
        -0.4613374887461511,
        0.0000000000000000,
        0.2191868483885747,
    },
    {
        0.2500000000000000,
        -0.1014005039375376,
        -0.4067007583026072,
        -0.2125574805828705,
        0.0000000000000000,
        -0.0643507165794627,
        -0.4517556589999464,
        0.3046847507248840,
        0.3017929516615503,
        -0.4082482904638635,
        -0.1747866975480813,
        0.2110560104933581,
        -0.1426608480880734,
        -0.1381354035075829,
        -0.1743760259965108,
        0.1135498731499426,
    },
    {
        0.2500000000000000,
        -0.1014005039375377,
        -0.1957439937204287,
        -0.1621205195722833,
        0.0000000000000000,
        -0.0643507165794628,
        0.0074182263792444,
        0.2904801297290076,
        0.0952002265347505,
        0.0000000000000000,
        0.3675398009862011,
        -0.4921585901373891,
        0.2462710772207514,
        -0.0794670660591026,
        0.3623817333531165,
        -0.4351904965232251,
    },
    {
        0.2500000000000000,
        -0.1014005039375375,
        0.0000000000000000,
        -0.4706702258572528,
        0.0000000000000000,
        -0.0643507165794627,
        0.1107416575309343,
        0.0000000000000000,
        -0.1627234014286617,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        0.1488339922711357,
        0.4972464710953509,
        0.2921026642334879,
        0.5550443808910661,
    },
    {
        0.2500000000000000,
        -0.1014005039375377,
        0.1137907446044809,
        -0.1464291867126764,
        0.0000000000000000,
        -0.0643507165794628,
        0.0829816309488205,
        -0.2388977352334460,
        -0.3531238544981630,
        -0.4082482904638630,
        0.4826689115059883,
        0.1741941265991622,
        -0.0476868035022925,
        0.1253805944856366,
        -0.4326608024727445,
        -0.2546827712406646,
    },
    {
        0.2500000000000000,
        -0.1014005039375377,
        -0.4444481661973438,
        0.3085497062849487,
        0.0000000000000000,
        -0.0643507165794628,
        0.1585450355183970,
        -0.5112616136592012,
        0.2579236279634129,
        0.0000000000000000,
        -0.0812611176717504,
        -0.1856718091610990,
        -0.3416446842253373,
        0.3302282550303805,
        0.0702790691196282,
        -0.0741750459581023,
    },
    {
        0.2500000000000000,
        -0.1014005039375376,
        -0.2929100136981264,
        0.0000000000000000,
        0.0000000000000000,
        -0.0643507165794627,
        0.3935103426921022,
        0.0657870154914254,
        0.0000000000000000,
        0.4082482904638634,
        0.3078822139579031,
        0.3852501370925211,
        -0.0857401903551927,
        -0.4613374887461554,
        0.0000000000000000,
        0.2191868483885728,
    },
    {
        0.2500000000000000,
        -0.1014005039375376,
        -0.1137907446044814,
        -0.1464291867126654,
        0.0000000000000000,
        -0.0643507165794627,
        0.0829816309488214,
        0.2388977352334547,
        -0.3531238544981624,
        0.4082482904638630,
        -0.4826689115059858,
        -0.1741941265991621,
        -0.0476868035022928,
        0.1253805944856431,
        -0.4326608024727457,
        -0.2546827712406641,
    },
    {
        0.2500000000000000,
        -0.1014005039375374,
        0.0000000000000000,
        0.4251149611657548,
        0.0000000000000000,
        -0.0643507165794626,
        -0.4517556589999480,
        0.0000000000000000,
        -0.6035859033230976,
        0.0000000000000000,
        0.0000000000000000,
        0.0000000000000000,
        -0.1426608480880724,
        -0.1381354035075845,
        0.3487520519930227,
        0.1135498731499429,
    },
};

HWY_ALIGN constexpr float k4x4AFVBasis[16][16] = {
    {
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
        0.25,
    },
    {
        0.876902929799142,
        0.2206518106944235,
        -0.10140050393753763,
        -0.1014005039375375,
        0.2206518106944236,
        -0.10140050393753777,
        -0.10140050393753772,
        -0.10140050393753763,
        -0.10140050393753758,
        -0.10140050393753769,
        -0.1014005039375375,
        -0.10140050393753768,
        -0.10140050393753768,
        -0.10140050393753759,
        -0.10140050393753763,
        -0.10140050393753741,
    },
    {
        0.0,
        0.0,
        0.40670075830260755,
        0.44444816619734445,
        0.0,
        0.0,
        0.19574399372042936,
        0.2929100136981264,
        -0.40670075830260716,
        -0.19574399372042872,
        0.0,
        0.11379074460448091,
        -0.44444816619734384,
        -0.29291001369812636,
        -0.1137907446044814,
        0.0,
    },
    {
        0.0,
        0.0,
        -0.21255748058288748,
        0.3085497062849767,
        0.0,
        0.4706702258572536,
        -0.1621205195722993,
        0.0,
        -0.21255748058287047,
        -0.16212051957228327,
        -0.47067022585725277,
        -0.1464291867126764,
        0.3085497062849487,
        0.0,
        -0.14642918671266536,
        0.4251149611657548,
    },
    {
        0.0,
        -0.7071067811865474,
        0.0,
        0.0,
        0.7071067811865476,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    },
    {
        -0.4105377591765233,
        0.6235485373547691,
        -0.06435071657946274,
        -0.06435071657946266,
        0.6235485373547694,
        -0.06435071657946284,
        -0.0643507165794628,
        -0.06435071657946274,
        -0.06435071657946272,
        -0.06435071657946279,
        -0.06435071657946266,
        -0.06435071657946277,
        -0.06435071657946277,
        -0.06435071657946273,
        -0.06435071657946274,
        -0.0643507165794626,
    },
    {
        0.0,
        0.0,
        -0.4517556589999482,
        0.15854503551840063,
        0.0,
        -0.04038515160822202,
        0.0074182263792423875,
        0.39351034269210167,
        -0.45175565899994635,
        0.007418226379244351,
        0.1107416575309343,
        0.08298163094882051,
        0.15854503551839705,
        0.3935103426921022,
        0.0829816309488214,
        -0.45175565899994796,
    },
    {
        0.0,
        0.0,
        -0.304684750724869,
        0.5112616136591823,
        0.0,
        0.0,
        -0.290480129728998,
        -0.06578701549142804,
        0.304684750724884,
        0.2904801297290076,
        0.0,
        -0.23889773523344604,
        -0.5112616136592012,
        0.06578701549142545,
        0.23889773523345467,
        0.0,
    },
    {
        0.0,
        0.0,
        0.3017929516615495,
        0.25792362796341184,
        0.0,
        0.16272340142866204,
        0.09520022653475037,
        0.0,
        0.3017929516615503,
        0.09520022653475055,
        -0.16272340142866173,
        -0.35312385449816297,
        0.25792362796341295,
        0.0,
        -0.3531238544981624,
        -0.6035859033230976,
    },
    {
        0.0,
        0.0,
        0.40824829046386274,
        0.0,
        0.0,
        0.0,
        0.0,
        -0.4082482904638628,
        -0.4082482904638635,
        0.0,
        0.0,
        -0.40824829046386296,
        0.0,
        0.4082482904638634,
        0.408248290463863,
        0.0,
    },
    {
        0.0,
        0.0,
        0.1747866975480809,
        0.0812611176717539,
        0.0,
        0.0,
        -0.3675398009862027,
        -0.307882213957909,
        -0.17478669754808135,
        0.3675398009862011,
        0.0,
        0.4826689115059883,
        -0.08126111767175039,
        0.30788221395790305,
        -0.48266891150598584,
        0.0,
    },
    {
        0.0,
        0.0,
        -0.21105601049335784,
        0.18567180916109802,
        0.0,
        0.0,
        0.49215859013738733,
        -0.38525013709251915,
        0.21105601049335806,
        -0.49215859013738905,
        0.0,
        0.17419412659916217,
        -0.18567180916109904,
        0.3852501370925211,
        -0.1741941265991621,
        0.0,
    },
    {
        0.0,
        0.0,
        -0.14266084808807264,
        -0.3416446842253372,
        0.0,
        0.7367497537172237,
        0.24627107722075148,
        -0.08574019035519306,
        -0.14266084808807344,
        0.24627107722075137,
        0.14883399227113567,
        -0.04768680350229251,
        -0.3416446842253373,
        -0.08574019035519267,
        -0.047686803502292804,
        -0.14266084808807242,
    },
    {
        0.0,
        0.0,
        -0.13813540350758585,
        0.3302282550303788,
        0.0,
        0.08755115000587084,
        -0.07946706605909573,
        -0.4613374887461511,
        -0.13813540350758294,
        -0.07946706605910261,
        0.49724647109535086,
        0.12538059448563663,
        0.3302282550303805,
        -0.4613374887461554,
        0.12538059448564315,
        -0.13813540350758452,
    },
    {
        0.0,
        0.0,
        -0.17437602599651067,
        0.0702790691196284,
        0.0,
        -0.2921026642334881,
        0.3623817333531167,
        0.0,
        -0.1743760259965108,
        0.36238173335311646,
        0.29210266423348785,
        -0.4326608024727445,
        0.07027906911962818,
        0.0,
        -0.4326608024727457,
        0.34875205199302267,
    },
    {
        0.0,
        0.0,
        0.11354987314994337,
        -0.07417504595810355,
        0.0,
        0.19402893032594343,
        -0.435190496523228,
        0.21918684838857466,
        0.11354987314994257,
        -0.4351904965232251,
        0.5550443808910661,
        -0.25468277124066463,
        -0.07417504595810233,
        0.2191868483885728,
        -0.25468277124066413,
        0.1135498731499429,
    },
};

using D = HWY_CAPPED(float, 16);

// Coefficient layout:
//  - (even, even) positions hold AFV coefficients
//  - (odd, even) positions hold DCT4x4 coefficients
//  - (any, odd) positions hold DCT4x8 coefficients
template <size_t afv_kind>
HWY_ATTR void AFVTransformFromPixels(const float* JXL_RESTRICT pixels,
                                     size_t pixels_stride,
                                     float* JXL_RESTRICT coefficients) {
  size_t afv_x = afv_kind & 1;
  size_t afv_y = afv_kind / 2;
  HWY_ALIGN float block[4 * 8];
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 4; ix++) {
      block[(afv_y == 1 ? 3 - iy : iy) * 4 + (afv_x == 1 ? 3 - ix : ix)] =
          pixels[(iy + 4 * afv_y) * pixels_stride + ix + 4 * afv_x];
    }
  }
  // AFV coefficients in (even, even) positions.
  HWY_ALIGN float coeff[4 * 4];
  AFVDCT4x4(block, coeff);
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 4; ix++) {
      coefficients[iy * 2 * 8 + ix * 2] = coeff[iy * 4 + ix];
    }
  }
  // 4x4 DCT of the block with same y and different x.
  ComputeTransposedScaledDCT<4>()(
      FromLines<4>(pixels + afv_y * 4 * pixels_stride + (afv_x == 1 ? 0 : 4),
                   pixels_stride),
      ScaleToBlock<4, 4>(block));
  // ... in (odd, even) positions.
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 8; ix++) {
      coefficients[iy * 2 * 8 + ix * 2 + 1] = block[iy * 4 + ix];
    }
  }
  // 4x8 DCT of the other half of the block.
  ComputeScaledDCT<4, 8>()(
      FromLines<8>(pixels + (afv_y == 1 ? 0 : 4) * pixels_stride,
                   pixels_stride),
      ScaleToBlock<4, 8>(block));
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 8; ix++) {
      coefficients[(1 + iy * 2) * 8 + ix] = block[iy * 8 + ix];
    }
  }
  float block00 = coefficients[0] * 0.25f;
  float block01 = coefficients[1];
  float block10 = coefficients[8];
  coefficients[0] = (block00 + block01 + 2 * block10) * 0.25f;
  coefficients[1] = (block00 - block01) * 0.5f;
  coefficients[8] = (block00 + block01 - 2 * block10) * 0.25f;
}

template <size_t afv_kind>
HWY_ATTR void AFVTransformToPixels(const float* JXL_RESTRICT coefficients,
                                   float* JXL_RESTRICT pixels,
                                   size_t pixels_stride) {
  size_t afv_x = afv_kind & 1;
  size_t afv_y = afv_kind / 2;
  float dcs[3] = {};
  float block00 = coefficients[0];
  float block01 = coefficients[1];
  float block10 = coefficients[8];
  dcs[0] = (block00 + block10 + block01) * 4.0f;
  dcs[1] = (block00 + block10 - block01);
  dcs[2] = block00 - block10;
  // IAFV: (even, even) positions.
  HWY_ALIGN float coeff[4 * 4];
  coeff[0] = dcs[0];
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 4; ix++) {
      if (ix == 0 && iy == 0) continue;
      coeff[iy * 4 + ix] = coefficients[iy * 2 * 8 + ix * 2];
    }
  }
  HWY_ALIGN float block[4 * 8];
  AFVIDCT4x4(coeff, block);
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 4; ix++) {
      pixels[(iy + afv_y * 4) * pixels_stride + afv_x * 4 + ix] =
          block[(afv_y == 1 ? 3 - iy : iy) * 4 + (afv_x == 1 ? 3 - ix : ix)];
    }
  }
  // IDCT4x4 in (odd, even) positions.
  block[0] = dcs[1];
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 4; ix++) {
      if (ix == 0 && iy == 0) continue;
      block[iy * 4 + ix] = coefficients[iy * 2 * 8 + ix * 2 + 1];
    }
  }
  ComputeTransposedScaledIDCT<4>()(
      FromBlock<4, 4>(block),
      ToLines<4>(pixels + afv_y * 4 * pixels_stride + (afv_x == 1 ? 0 : 4),
                 pixels_stride));
  // IDCT4x8.
  block[0] = dcs[2];
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 8; ix++) {
      if (ix == 0 && iy == 0) continue;
      block[iy * 8 + ix] = coefficients[(1 + iy * 2) * 8 + ix];
    }
  }
  ComputeScaledIDCT<4, 8>()(
      FromBlock<4, 8>(block),
      ToLines<8>(pixels + (afv_y == 1 ? 0 : 4) * pixels_stride, pixels_stride));
}

}  // namespace

HWY_ATTR void AFVDCT4x4(const float* JXL_RESTRICT pixels,
                        float* JXL_RESTRICT coeffs) {
  for (size_t i = 0; i < 16; i += D::N) {
    auto scalar = Zero(D());
    for (size_t j = 0; j < 16; j++) {
      auto px = Set(D(), pixels[j]);
      auto basis = Load(D(), k4x4AFVBasisTranspose[j] + i);
      scalar = MulAdd(px, basis, scalar);
    }
    Store(scalar, D(), coeffs + i);
  }
}

HWY_ATTR void AFVIDCT4x4(const float* JXL_RESTRICT coeffs,
                         float* JXL_RESTRICT pixels) {
  for (size_t i = 0; i < 16; i += D::N) {
    auto pixel = Zero(D());
    for (size_t j = 0; j < 16; j++) {
      auto cf = Set(D(), coeffs[j]);
      auto basis = Load(D(), k4x4AFVBasis[j] + i);
      pixel = MulAdd(cf, basis, pixel);
    }
    Store(pixel, D(), pixels + i);
  }
}

const AcStrategy::CoeffOrderAndLut* AcStrategy::CoeffOrder() {
  static AcStrategy::CoeffOrderAndLut order = ComputeNaturalCoeffOrder();
  return &order;
}

// These definitions are needed before C++17.
constexpr size_t AcStrategy::kMaxCoeffBlocks;
constexpr size_t AcStrategy::kMaxBlockDim;
constexpr size_t AcStrategy::kMaxCoeffArea;

HWY_ATTR void AcStrategy::TransformFromPixels(
    const float* JXL_RESTRICT pixels, size_t pixels_stride,
    float* JXL_RESTRICT coefficients) const {
  switch (strategy_) {
    case Type::IDENTITY: {
      PROFILER_ZONE("DCT Identity");
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          float block_dc = 0;
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              block_dc += pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix];
            }
          }
          block_dc *= 1.0f / 16;
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              if (ix == 1 && iy == 1) continue;
              coefficients[(y + iy * 2) * 8 + x + ix * 2] =
                  pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix] -
                  pixels[(y * 4 + 1) * pixels_stride + x * 4 + 1];
            }
          }
          coefficients[(y + 2) * 8 + x + 2] = coefficients[y * 8 + x];
          coefficients[y * 8 + x] = block_dc;
        }
      }
      float block00 = coefficients[0];
      float block01 = coefficients[1];
      float block10 = coefficients[8];
      float block11 = coefficients[9];
      coefficients[0] = (block00 + block01 + block10 + block11) * 0.25f;
      coefficients[1] = (block00 + block01 - block10 - block11) * 0.25f;
      coefficients[8] = (block00 - block01 + block10 - block11) * 0.25f;
      coefficients[9] = (block00 - block01 - block10 + block11) * 0.25f;
      break;
    }
    case Type::DCT8X4: {
      PROFILER_ZONE("DCT 8x4");
      for (size_t x = 0; x < 2; x++) {
        HWY_ALIGN float block[4 * 8];
        ComputeScaledDCT<8, 4>()(FromLines<4>(pixels + x * 4, pixels_stride),
                                 ScaleToBlock<4, 8>(block));
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 8; ix++) {
            // Store transposed.
            coefficients[(x + iy * 2) * 8 + ix] = block[iy * 8 + ix];
          }
        }
      }
      float block0 = coefficients[0];
      float block1 = coefficients[8];
      coefficients[0] = (block0 + block1) * 0.5f;
      coefficients[8] = (block0 - block1) * 0.5f;
      break;
    }
    case Type::DCT4X8: {
      PROFILER_ZONE("DCT 4x8");
      for (size_t y = 0; y < 2; y++) {
        HWY_ALIGN float block[4 * 8];
        ComputeScaledDCT<4, 8>()(
            FromLines<8>(pixels + y * 4 * pixels_stride, pixels_stride),
            ScaleToBlock<4, 8>(block));
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 8; ix++) {
            coefficients[(y + iy * 2) * 8 + ix] = block[iy * 8 + ix];
          }
        }
      }
      float block0 = coefficients[0];
      float block1 = coefficients[8];
      coefficients[0] = (block0 + block1) * 0.5f;
      coefficients[8] = (block0 - block1) * 0.5f;
      break;
    }
    case Type::DCT4X4: {
      PROFILER_ZONE("DCT 4");
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          HWY_ALIGN float block[4 * 4];
          ComputeTransposedScaledDCT<4>()(
              FromLines<4>(pixels + y * 4 * pixels_stride + x * 4,
                           pixels_stride),
              ScaleToBlock<4>(block));
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              coefficients[(y + iy * 2) * 8 + x + ix * 2] = block[iy * 4 + ix];
            }
          }
        }
      }
      float block00 = coefficients[0];
      float block01 = coefficients[1];
      float block10 = coefficients[8];
      float block11 = coefficients[9];
      coefficients[0] = (block00 + block01 + block10 + block11) * 0.25f;
      coefficients[1] = (block00 + block01 - block10 - block11) * 0.25f;
      coefficients[8] = (block00 - block01 + block10 - block11) * 0.25f;
      coefficients[9] = (block00 - block01 - block10 + block11) * 0.25f;
      break;
    }
    case Type::DCT2X2: {
      PROFILER_ZONE("DCT 2");
      DCT2TopBlock<8>(pixels, pixels_stride, coefficients);
      DCT2TopBlock<4>(coefficients, kBlockDim, coefficients);
      DCT2TopBlock<2>(coefficients, kBlockDim, coefficients);
      break;
    }
    case Type::DCT16X16: {
      PROFILER_ZONE("DCT 16");
      ComputeTransposedScaledDCT<2 * kBlockDim>()(
          FromLines<2 * kBlockDim>(pixels, pixels_stride),
          ScaleToBlock<2 * kBlockDim>(coefficients));
      break;
    }
    case Type::DCT16X8: {
      PROFILER_ZONE("DCT 16x8");
      ComputeScaledDCT<16, 8>()(FromLines<8>(pixels, pixels_stride),
                                ScaleToBlock<8, 16>(coefficients));
      break;
    }
    case Type::DCT8X16: {
      PROFILER_ZONE("DCT 8x16");
      ComputeScaledDCT<8, 16>()(FromLines<16>(pixels, pixels_stride),
                                ScaleToBlock<8, 16>(coefficients));
      break;
    }
    case Type::DCT32X8: {
      PROFILER_ZONE("DCT 32x8");
      ComputeScaledDCT<32, 8>()(FromLines<8>(pixels, pixels_stride),
                                ScaleToBlock<8, 32>(coefficients));
      break;
    }
    case Type::DCT8X32: {
      PROFILER_ZONE("DCT 8x32");
      ComputeScaledDCT<8, 32>()(FromLines<32>(pixels, pixels_stride),
                                ScaleToBlock<8, 32>(coefficients));
      break;
    }
    case Type::DCT32X16: {
      PROFILER_ZONE("DCT 32x16");
      ComputeScaledDCT<32, 16>()(FromLines<16>(pixels, pixels_stride),
                                 ScaleToBlock<16, 32>(coefficients));
      break;
    }
    case Type::DCT16X32: {
      PROFILER_ZONE("DCT 16x32");
      ComputeScaledDCT<16, 32>()(FromLines<32>(pixels, pixels_stride),
                                 ScaleToBlock<16, 32>(coefficients));
      break;
    }
    case Type::DCT32X32: {
      PROFILER_ZONE("DCT 32");
      ComputeTransposedScaledDCT<4 * kBlockDim>()(
          FromLines<4 * kBlockDim>(pixels, pixels_stride),
          ScaleToBlock<4 * kBlockDim>(coefficients));
      break;
    }
    case Type::DCT: {
      PROFILER_ZONE("DCT 8");
      ComputeTransposedScaledDCT<kBlockDim>()(
          FromLines<kBlockDim>(pixels, pixels_stride),
          ScaleToBlock<kBlockDim>(coefficients));
      break;
    }
    case Type::AFV0: {
      PROFILER_ZONE("AFV0");
      AFVTransformFromPixels<0>(pixels, pixels_stride, coefficients);
      break;
    }
    case Type::AFV1: {
      PROFILER_ZONE("AFV1");
      AFVTransformFromPixels<1>(pixels, pixels_stride, coefficients);
      break;
    }
    case Type::AFV2: {
      PROFILER_ZONE("AFV2");
      AFVTransformFromPixels<2>(pixels, pixels_stride, coefficients);
      break;
    }
    case Type::AFV3: {
      PROFILER_ZONE("AFV3");
      AFVTransformFromPixels<3>(pixels, pixels_stride, coefficients);
      break;
    }
    case Type::kNumValidStrategies:
      JXL_ABORT("Invalid strategy");
  }
}

HWY_ATTR void AcStrategy::TransformToPixels(
    const float* JXL_RESTRICT coefficients, float* JXL_RESTRICT pixels,
    size_t pixels_stride) const {
  switch (strategy_) {
    case Type::IDENTITY: {
      PROFILER_ZONE("IDCT Identity");
      float dcs[4] = {};
      float block00 = coefficients[0];
      float block01 = coefficients[1];
      float block10 = coefficients[8];
      float block11 = coefficients[9];
      dcs[0] = block00 + block01 + block10 + block11;
      dcs[1] = block00 + block01 - block10 - block11;
      dcs[2] = block00 - block01 + block10 - block11;
      dcs[3] = block00 - block01 - block10 + block11;
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          float block_dc = dcs[y * 2 + x];
          float residual_sum = 0;
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              if (ix == 0 && iy == 0) continue;
              residual_sum += coefficients[(y + iy * 2) * 8 + x + ix * 2];
            }
          }
          pixels[(4 * y + 1) * pixels_stride + 4 * x + 1] =
              block_dc - residual_sum * (1.0f / 16);
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              if (ix == 1 && iy == 1) continue;
              pixels[(y * 4 + iy) * pixels_stride + x * 4 + ix] =
                  coefficients[(y + iy * 2) * 8 + x + ix * 2] +
                  pixels[(4 * y + 1) * pixels_stride + 4 * x + 1];
            }
          }
          pixels[y * 4 * pixels_stride + x * 4] =
              coefficients[(y + 2) * 8 + x + 2] +
              pixels[(4 * y + 1) * pixels_stride + 4 * x + 1];
        }
      }
      break;
    }
    case Type::DCT8X4: {
      PROFILER_ZONE("IDCT 8x4");
      float dcs[2] = {};
      float block0 = coefficients[0];
      float block1 = coefficients[8];
      dcs[0] = block0 + block1;
      dcs[1] = block0 - block1;
      for (size_t x = 0; x < 2; x++) {
        HWY_ALIGN float block[4 * 8];
        block[0] = dcs[x];
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 8; ix++) {
            if (ix == 0 && iy == 0) continue;
            block[iy * 8 + ix] = coefficients[(x + iy * 2) * 8 + ix];
          }
        }
        ComputeScaledIDCT<8, 4>()(FromBlock<4, 8>(block),
                                  ToLines<4>(pixels + x * 4, pixels_stride));
      }
      break;
    }
    case Type::DCT4X8: {
      PROFILER_ZONE("IDCT 4x8");
      float dcs[2] = {};
      float block0 = coefficients[0];
      float block1 = coefficients[8];
      dcs[0] = block0 + block1;
      dcs[1] = block0 - block1;
      for (size_t y = 0; y < 2; y++) {
        HWY_ALIGN float block[4 * 8];
        block[0] = dcs[y];
        for (size_t iy = 0; iy < 4; iy++) {
          for (size_t ix = 0; ix < 8; ix++) {
            if (ix == 0 && iy == 0) continue;
            block[iy * 8 + ix] = coefficients[(y + iy * 2) * 8 + ix];
          }
        }
        ComputeScaledIDCT<4, 8>()(
            FromBlock<4, 8>(block),
            ToLines<8>(pixels + y * 4 * pixels_stride, pixels_stride));
      }
      break;
    }
    case Type::DCT4X4: {
      PROFILER_ZONE("IDCT 4");
      float dcs[4] = {};
      float block00 = coefficients[0];
      float block01 = coefficients[1];
      float block10 = coefficients[8];
      float block11 = coefficients[9];
      dcs[0] = block00 + block01 + block10 + block11;
      dcs[1] = block00 + block01 - block10 - block11;
      dcs[2] = block00 - block01 + block10 - block11;
      dcs[3] = block00 - block01 - block10 + block11;
      for (size_t y = 0; y < 2; y++) {
        for (size_t x = 0; x < 2; x++) {
          HWY_ALIGN float block[4 * 4];
          block[0] = dcs[y * 2 + x];
          for (size_t iy = 0; iy < 4; iy++) {
            for (size_t ix = 0; ix < 4; ix++) {
              if (ix == 0 && iy == 0) continue;
              block[iy * 4 + ix] = coefficients[(y + iy * 2) * 8 + x + ix * 2];
            }
          }
          ComputeTransposedScaledIDCT<4>()(
              FromBlock<4>(block),
              ToLines<4>(pixels + y * 4 * pixels_stride + x * 4,
                         pixels_stride));
        }
      }
      break;
    }
    case Type::DCT2X2: {
      PROFILER_ZONE("IDCT 2");
      HWY_ALIGN float coeffs[kDCTBlockSize];
      memcpy(coeffs, coefficients, sizeof(float) * kDCTBlockSize);
      IDCT2TopBlock<2>(coeffs, kBlockDim, coeffs);
      IDCT2TopBlock<4>(coeffs, kBlockDim, coeffs);
      IDCT2TopBlock<8>(coeffs, kBlockDim, coeffs);
      for (size_t y = 0; y < kBlockDim; y++) {
        for (size_t x = 0; x < kBlockDim; x++) {
          pixels[y * pixels_stride + x] = coeffs[y * kBlockDim + x];
        }
      }
      break;
    }
    case Type::DCT16X16: {
      PROFILER_ZONE("IDCT 16");
      ComputeTransposedScaledIDCT<2 * kBlockDim>()(
          FromBlock<2 * kBlockDim>(coefficients),
          ToLines<2 * kBlockDim>(pixels, pixels_stride));
      break;
    }
    case Type::DCT16X8: {
      PROFILER_ZONE("IDCT 16x8");
      ComputeScaledIDCT<16, 8>()(FromBlock<8, 16>(coefficients),
                                 ToLines<8>(pixels, pixels_stride));
      break;
    }
    case Type::DCT8X16: {
      PROFILER_ZONE("IDCT 8x16");
      ComputeScaledIDCT<8, 16>()(FromBlock<8, 16>(coefficients),
                                 ToLines<16>(pixels, pixels_stride));
      break;
    }
    case Type::DCT32X8: {
      PROFILER_ZONE("IDCT 32x8");
      ComputeScaledIDCT<32, 8>()(FromBlock<8, 32>(coefficients),
                                 ToLines<8>(pixels, pixels_stride));
      break;
    }
    case Type::DCT8X32: {
      PROFILER_ZONE("IDCT 8x32");
      ComputeScaledIDCT<8, 32>()(FromBlock<8, 32>(coefficients),
                                 ToLines<32>(pixels, pixels_stride));
      break;
    }
    case Type::DCT32X16: {
      PROFILER_ZONE("IDCT 32x16");
      ComputeScaledIDCT<32, 16>()(FromBlock<16, 32>(coefficients),
                                  ToLines<16>(pixels, pixels_stride));
      break;
    }
    case Type::DCT16X32: {
      PROFILER_ZONE("IDCT 16x32");
      ComputeScaledIDCT<16, 32>()(FromBlock<16, 32>(coefficients),
                                  ToLines<32>(pixels, pixels_stride));
      break;
    }
    case Type::DCT32X32: {
      PROFILER_ZONE("IDCT 32");
      ComputeTransposedScaledIDCT<4 * kBlockDim>()(
          FromBlock<4 * kBlockDim>(coefficients),
          ToLines<4 * kBlockDim>(pixels, pixels_stride));
      break;
    }
    case Type::DCT: {
      PROFILER_ZONE("IDCT 8");
      ComputeTransposedScaledIDCT<kBlockDim>()(
          FromBlock<kBlockDim>(coefficients),
          ToLines<kBlockDim>(pixels, pixels_stride));
      break;
    }
    case Type::AFV0: {
      PROFILER_ZONE("IAFV0");
      AFVTransformToPixels<0>(coefficients, pixels, pixels_stride);
      break;
    }
    case Type::AFV1: {
      PROFILER_ZONE("IAFV1");
      AFVTransformToPixels<1>(coefficients, pixels, pixels_stride);
      break;
    }
    case Type::AFV2: {
      PROFILER_ZONE("IAFV2");
      AFVTransformToPixels<2>(coefficients, pixels, pixels_stride);
      break;
    }
    case Type::AFV3: {
      PROFILER_ZONE("IAFV3");
      AFVTransformToPixels<3>(coefficients, pixels, pixels_stride);
      break;
    }
    case Type::kNumValidStrategies:
      JXL_ABORT("Invalid strategy");
  }
}

HWY_ATTR void AcStrategy::LowestFrequenciesFromDC(const float* dc,
                                                  size_t dc_stride,
                                                  float* llf) const {
  switch (strategy_) {
    case Type::DCT16X8: {
      ReinterpretingDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/kBlockDim,
                        /*LF_ROWS=*/2, /*LF_COLS=*/1, /*ROWS=*/2, /*COLS=*/1>(
          dc, dc_stride, llf, 2 * kBlockDim);
      break;
    }
    case Type::DCT8X16: {
      ReinterpretingDCT</*DCT_ROWS=*/kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                        /*LF_ROWS=*/1, /*LF_COLS=*/2, /*ROWS=*/1, /*COLS=*/2>(
          dc, dc_stride, llf, 2 * kBlockDim);
      break;
    }
    case Type::DCT16X16: {
      ReinterpretingDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                        /*LF_ROWS=*/2, /*LF_COLS=*/2, /*ROWS=*/2, /*COLS=*/2>(
          dc, dc_stride, llf, 2 * kBlockDim);
      break;
    }
    case Type::DCT32X8: {
      ReinterpretingDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/kBlockDim,
                        /*LF_ROWS=*/4, /*LF_COLS=*/1, /*ROWS=*/4, /*COLS=*/1>(
          dc, dc_stride, llf, 4 * kBlockDim);
      break;
    }
    case Type::DCT8X32: {
      ReinterpretingDCT</*DCT_ROWS=*/kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                        /*LF_ROWS=*/1, /*LF_COLS=*/4, /*ROWS=*/1, /*COLS=*/4>(
          dc, dc_stride, llf, 4 * kBlockDim);
      break;
    }
    case Type::DCT32X16: {
      ReinterpretingDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                        /*LF_ROWS=*/4, /*LF_COLS=*/2, /*ROWS=*/4, /*COLS=*/2>(
          dc, dc_stride, llf, 4 * kBlockDim);
      break;
    }
    case Type::DCT16X32: {
      ReinterpretingDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                        /*LF_ROWS=*/2, /*LF_COLS=*/4, /*ROWS=*/2, /*COLS=*/4>(
          dc, dc_stride, llf, 4 * kBlockDim);
      break;
    }
    case Type::DCT32X32: {
      ReinterpretingDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                        /*LF_ROWS=*/4, /*LF_COLS=*/4, /*ROWS=*/4, /*COLS=*/4>(
          dc, dc_stride, llf, 4 * kBlockDim);
      break;
    }
    case Type::DCT:
    case Type::DCT2X2:
    case Type::DCT4X4:
    case Type::DCT4X8:
    case Type::DCT8X4:
    case Type::AFV0:
    case Type::AFV1:
    case Type::AFV2:
    case Type::AFV3:
    case Type::IDENTITY:
      llf[0] = dc[0];
      break;
    case Type::kNumValidStrategies:
      JXL_ABORT("Invalid strategy");
  };
}

HWY_ATTR void AcStrategy::DCFromLowestFrequencies(const float* block, float* dc,
                                                  size_t dc_stride) const {
  switch (strategy_) {
    case Type::DCT16X8: {
      ReinterpretingIDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/kBlockDim,
                         /*LF_ROWS=*/2, /*LF_COLS=*/1, /*ROWS=*/2, /*COLS=*/1>(
          block, 2 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT8X16: {
      ReinterpretingIDCT</*DCT_ROWS=*/kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                         /*LF_ROWS=*/1, /*LF_COLS=*/2, /*ROWS=*/1, /*COLS=*/2>(
          block, 2 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT16X16: {
      ReinterpretingIDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                         /*LF_ROWS=*/2, /*LF_COLS=*/2, /*ROWS=*/2, /*COLS=*/2>(
          block, 2 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT32X8: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/1, /*ROWS=*/4, /*COLS=*/1>(
          block, 4 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT8X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/1, /*LF_COLS=*/4, /*ROWS=*/1, /*COLS=*/4>(
          block, 4 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT32X16: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/2 * kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/2, /*ROWS=*/4, /*COLS=*/2>(
          block, 4 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT16X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/2 * kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/2, /*LF_COLS=*/4, /*ROWS=*/2, /*COLS=*/4>(
          block, 4 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT32X32: {
      ReinterpretingIDCT</*DCT_ROWS=*/4 * kBlockDim, /*DCT_COLS=*/4 * kBlockDim,
                         /*LF_ROWS=*/4, /*LF_COLS=*/4, /*ROWS=*/4, /*COLS=*/4>(
          block, 4 * kBlockDim, dc, dc_stride);
      break;
    }
    case Type::DCT:
    case Type::DCT2X2:
    case Type::DCT4X4:
    case Type::DCT4X8:
    case Type::DCT8X4:
    case Type::AFV0:
    case Type::AFV1:
    case Type::AFV2:
    case Type::AFV3:
    case Type::IDENTITY:
      dc[0] = block[0];
      break;
    case Type::kNumValidStrategies:
      JXL_ABORT("Invalid strategy");
  }
}

AcStrategyImage::AcStrategyImage(size_t xsize, size_t ysize)
    : layers_(xsize, ysize) {
  row_ = layers_.Row(0);
  stride_ = layers_.PixelsPerRow();
}

size_t AcStrategyImage::CountBlocks(AcStrategy::Type type) const {
  size_t ret = 0;
  for (size_t y = 0; y < layers_.ysize(); y++) {
    const uint8_t* JXL_RESTRICT row = layers_.ConstRow(y);
    for (size_t x = 0; x < layers_.xsize(); x++) {
      if (row[x] == ((static_cast<uint8_t>(type) << 1) | 1)) ret++;
    }
  }
  return ret;
}

}  // namespace jxl
