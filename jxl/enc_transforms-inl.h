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

#if defined(JXL_ENC_TRANSFORMS_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef JXL_ENC_TRANSFORMS_INL_H_
#undef JXL_ENC_TRANSFORMS_INL_H_
#else
#define JXL_ENC_TRANSFORMS_INL_H_
#endif

#include <stddef.h>

#include <hwy/highway.h>

#include "jxl/ac_strategy.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/dct-inl.h"
#include "jxl/dct_scales.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

template <size_t ROWS, size_t COLS>
struct DoDCT {
  template <typename From, typename To>
  void operator()(const From& from, const To& to) {
    ComputeScaledDCT<ROWS, COLS>()(from, to);
  }
};

template <size_t N>
struct DoDCT<N, N> {
  template <typename From, typename To>
  void operator()(const From& from, const To& to) {
    ComputeTransposedScaledDCT<N>()(from, to);
  }
};

// Computes the lowest-frequency LF_ROWSxLF_COLS-sized square in output, which
// is a DCT_ROWS*DCT_COLS-sized DCT block, by doing a ROWS*COLS DCT on the
// input block.
template <size_t DCT_ROWS, size_t DCT_COLS, size_t LF_ROWS, size_t LF_COLS,
          size_t ROWS, size_t COLS>
JXL_INLINE void ReinterpretingDCT(const float* input, const size_t input_stride,
                                  float* output, const size_t output_stride) {
  static_assert(LF_ROWS == ROWS,
                "ReinterpretingDCT should only be called with LF == N");
  static_assert(LF_COLS == COLS,
                "ReinterpretingDCT should only be called with LF == N");
  // TODO(veluca): avoid copy using special FromBlock adapters.
  HWY_ALIGN float block[ROWS * COLS] = {};
  for (size_t y = 0; y < ROWS; y++) {
    for (size_t x = 0; x < COLS; x++) {
      block[y * COLS + x] = input[y * input_stride + x];
    }
  }

  constexpr size_t OUT_ROWS = CoefficientRows(ROWS, COLS);
  constexpr size_t OUT_COLS = CoefficientColumns(ROWS, COLS);
  DoDCT<ROWS, COLS>()(FromBlock(ROWS, COLS, block),
                      ScaleToBlock(OUT_ROWS, OUT_COLS, block));
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

void AFVDCT4x4(const float* JXL_RESTRICT pixels, float* JXL_RESTRICT coeffs) {
  HWY_ALIGN static constexpr float k4x4AFVBasisTranspose[16][16] = {
      {
          0.2500000000000000,
          0.8769029297991420f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          -0.4105377591765233f,
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
          0.2206518106944235f,
          0.0000000000000000,
          0.0000000000000000,
          -0.7071067811865474f,
          0.6235485373547691f,
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
          -0.1014005039375376f,
          0.4067007583026075f,
          -0.2125574805828875f,
          0.0000000000000000,
          -0.0643507165794627f,
          -0.4517556589999482f,
          -0.3046847507248690f,
          0.3017929516615495f,
          0.4082482904638627f,
          0.1747866975480809f,
          -0.2110560104933578f,
          -0.1426608480880726f,
          -0.1381354035075859f,
          -0.1743760259965107f,
          0.1135498731499434f,
      },
      {
          0.2500000000000000,
          -0.1014005039375375f,
          0.4444481661973445f,
          0.3085497062849767f,
          0.0000000000000000f,
          -0.0643507165794627f,
          0.1585450355184006f,
          0.5112616136591823f,
          0.2579236279634118f,
          0.0000000000000000,
          0.0812611176717539f,
          0.1856718091610980f,
          -0.3416446842253372f,
          0.3302282550303788f,
          0.0702790691196284f,
          -0.0741750459581035f,
      },
      {
          0.2500000000000000,
          0.2206518106944236f,
          0.0000000000000000,
          0.0000000000000000,
          0.7071067811865476f,
          0.6235485373547694f,
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
          -0.1014005039375378f,
          0.0000000000000000,
          0.4706702258572536f,
          0.0000000000000000,
          -0.0643507165794628f,
          -0.0403851516082220f,
          0.0000000000000000,
          0.1627234014286620f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.7367497537172237f,
          0.0875511500058708f,
          -0.2921026642334881f,
          0.1940289303259434f,
      },
      {
          0.2500000000000000,
          -0.1014005039375377f,
          0.1957439937204294f,
          -0.1621205195722993f,
          0.0000000000000000,
          -0.0643507165794628f,
          0.0074182263792424f,
          -0.2904801297289980f,
          0.0952002265347504f,
          0.0000000000000000,
          -0.3675398009862027f,
          0.4921585901373873f,
          0.2462710772207515f,
          -0.0794670660590957f,
          0.3623817333531167f,
          -0.4351904965232280f,
      },
      {
          0.2500000000000000,
          -0.1014005039375376f,
          0.2929100136981264f,
          0.0000000000000000,
          0.0000000000000000,
          -0.0643507165794627f,
          0.3935103426921017f,
          -0.0657870154914280f,
          0.0000000000000000,
          -0.4082482904638628f,
          -0.3078822139579090f,
          -0.3852501370925192f,
          -0.0857401903551931f,
          -0.4613374887461511f,
          0.0000000000000000,
          0.2191868483885747f,
      },
      {
          0.2500000000000000,
          -0.1014005039375376f,
          -0.4067007583026072f,
          -0.2125574805828705f,
          0.0000000000000000,
          -0.0643507165794627f,
          -0.4517556589999464f,
          0.3046847507248840f,
          0.3017929516615503f,
          -0.4082482904638635f,
          -0.1747866975480813f,
          0.2110560104933581f,
          -0.1426608480880734f,
          -0.1381354035075829f,
          -0.1743760259965108f,
          0.1135498731499426f,
      },
      {
          0.2500000000000000,
          -0.1014005039375377f,
          -0.1957439937204287f,
          -0.1621205195722833f,
          0.0000000000000000,
          -0.0643507165794628f,
          0.0074182263792444f,
          0.2904801297290076f,
          0.0952002265347505f,
          0.0000000000000000,
          0.3675398009862011f,
          -0.4921585901373891f,
          0.2462710772207514f,
          -0.0794670660591026f,
          0.3623817333531165f,
          -0.4351904965232251f,
      },
      {
          0.2500000000000000,
          -0.1014005039375375f,
          0.0000000000000000,
          -0.4706702258572528f,
          0.0000000000000000,
          -0.0643507165794627f,
          0.1107416575309343f,
          0.0000000000000000,
          -0.1627234014286617f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          0.1488339922711357f,
          0.4972464710953509f,
          0.2921026642334879f,
          0.5550443808910661f,
      },
      {
          0.2500000000000000,
          -0.1014005039375377f,
          0.1137907446044809f,
          -0.1464291867126764f,
          0.0000000000000000,
          -0.0643507165794628f,
          0.0829816309488205f,
          -0.2388977352334460f,
          -0.3531238544981630f,
          -0.4082482904638630f,
          0.4826689115059883f,
          0.1741941265991622f,
          -0.0476868035022925f,
          0.1253805944856366f,
          -0.4326608024727445f,
          -0.2546827712406646f,
      },
      {
          0.2500000000000000,
          -0.1014005039375377f,
          -0.4444481661973438f,
          0.3085497062849487f,
          0.0000000000000000,
          -0.0643507165794628f,
          0.1585450355183970f,
          -0.5112616136592012f,
          0.2579236279634129f,
          0.0000000000000000,
          -0.0812611176717504f,
          -0.1856718091610990f,
          -0.3416446842253373f,
          0.3302282550303805f,
          0.0702790691196282f,
          -0.0741750459581023f,
      },
      {
          0.2500000000000000,
          -0.1014005039375376f,
          -0.2929100136981264f,
          0.0000000000000000,
          0.0000000000000000,
          -0.0643507165794627f,
          0.3935103426921022f,
          0.0657870154914254f,
          0.0000000000000000,
          0.4082482904638634f,
          0.3078822139579031f,
          0.3852501370925211f,
          -0.0857401903551927f,
          -0.4613374887461554f,
          0.0000000000000000,
          0.2191868483885728f,
      },
      {
          0.2500000000000000,
          -0.1014005039375376f,
          -0.1137907446044814f,
          -0.1464291867126654f,
          0.0000000000000000,
          -0.0643507165794627f,
          0.0829816309488214f,
          0.2388977352334547f,
          -0.3531238544981624f,
          0.4082482904638630f,
          -0.4826689115059858f,
          -0.1741941265991621f,
          -0.0476868035022928f,
          0.1253805944856431f,
          -0.4326608024727457f,
          -0.2546827712406641f,
      },
      {
          0.2500000000000000,
          -0.1014005039375374f,
          0.0000000000000000,
          0.4251149611657548f,
          0.0000000000000000,
          -0.0643507165794626f,
          -0.4517556589999480f,
          0.0000000000000000,
          -0.6035859033230976f,
          0.0000000000000000,
          0.0000000000000000,
          0.0000000000000000,
          -0.1426608480880724f,
          -0.1381354035075845f,
          0.3487520519930227f,
          0.1135498731499429f,
      },
  };

  const HWY_CAPPED(float, 16) d;
  for (size_t i = 0; i < 16; i += Lanes(d)) {
    auto scalar = Zero(d);
    for (size_t j = 0; j < 16; j++) {
      auto px = Set(d, pixels[j]);
      auto basis = Load(d, k4x4AFVBasisTranspose[j] + i);
      scalar = MulAdd(px, basis, scalar);
    }
    Store(scalar, d, coeffs + i);
  }
}

// Coefficient layout:
//  - (even, even) positions hold AFV coefficients
//  - (odd, even) positions hold DCT4x4 coefficients
//  - (any, odd) positions hold DCT4x8 coefficients
template <size_t afv_kind>
void AFVTransformFromPixels(const float* JXL_RESTRICT pixels,
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
      FromLines(pixels + afv_y * 4 * pixels_stride + (afv_x == 1 ? 0 : 4),
                pixels_stride),
      ScaleToBlock(4, 4, block));
  // ... in (odd, even) positions.
  for (size_t iy = 0; iy < 4; iy++) {
    for (size_t ix = 0; ix < 8; ix++) {
      coefficients[iy * 2 * 8 + ix * 2 + 1] = block[iy * 4 + ix];
    }
  }
  // 4x8 DCT of the other half of the block.
  ComputeScaledDCT<4, 8>()(
      FromLines(pixels + (afv_y == 1 ? 0 : 4) * pixels_stride, pixels_stride),
      ScaleToBlock(4, 8, block));
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

HWY_MAYBE_UNUSED void TransformFromPixels(const AcStrategy::Type strategy,
                                          const float* JXL_RESTRICT pixels,
                                          size_t pixels_stride,
                                          float* JXL_RESTRICT coefficients) {
  using Type = AcStrategy::Type;
  switch (strategy) {
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
        ComputeScaledDCT<8, 4>()(FromLines(pixels + x * 4, pixels_stride),
                                 ScaleToBlock(4, 8, block));
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
            FromLines(pixels + y * 4 * pixels_stride, pixels_stride),
            ScaleToBlock(4, 8, block));
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
              FromLines(pixels + y * 4 * pixels_stride + x * 4, pixels_stride),
              ScaleToBlock(4, 4, block));
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
          FromLines(pixels, pixels_stride),
          ScaleToBlock(2 * kBlockDim, 2 * kBlockDim, coefficients));
      break;
    }
    case Type::DCT16X8: {
      PROFILER_ZONE("DCT 16x8");
      ComputeScaledDCT<16, 8>()(FromLines(pixels, pixels_stride),
                                ScaleToBlock(8, 16, coefficients));
      break;
    }
    case Type::DCT8X16: {
      PROFILER_ZONE("DCT 8x16");
      ComputeScaledDCT<8, 16>()(FromLines(pixels, pixels_stride),
                                ScaleToBlock(8, 16, coefficients));
      break;
    }
    case Type::DCT32X8: {
      PROFILER_ZONE("DCT 32x8");
      ComputeScaledDCT<32, 8>()(FromLines(pixels, pixels_stride),
                                ScaleToBlock(8, 32, coefficients));
      break;
    }
    case Type::DCT8X32: {
      PROFILER_ZONE("DCT 8x32");
      ComputeScaledDCT<8, 32>()(FromLines(pixels, pixels_stride),
                                ScaleToBlock(8, 32, coefficients));
      break;
    }
    case Type::DCT32X16: {
      PROFILER_ZONE("DCT 32x16");
      ComputeScaledDCT<32, 16>()(FromLines(pixels, pixels_stride),
                                 ScaleToBlock(16, 32, coefficients));
      break;
    }
    case Type::DCT16X32: {
      PROFILER_ZONE("DCT 16x32");
      ComputeScaledDCT<16, 32>()(FromLines(pixels, pixels_stride),
                                 ScaleToBlock(16, 32, coefficients));
      break;
    }
    case Type::DCT32X32: {
      PROFILER_ZONE("DCT 32");
      ComputeTransposedScaledDCT<4 * kBlockDim>()(
          FromLines(pixels, pixels_stride),
          ScaleToBlock(4 * kBlockDim, 4 * kBlockDim, coefficients));
      break;
    }
    case Type::DCT: {
      PROFILER_ZONE("DCT 8");
      ComputeTransposedScaledDCT<kBlockDim>()(
          FromLines(pixels, pixels_stride),
          ScaleToBlock(kBlockDim, kBlockDim, coefficients));
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

HWY_MAYBE_UNUSED void LowestFrequenciesFromDC(const AcStrategy::Type strategy,
                                              const float* dc, size_t dc_stride,
                                              float* llf) {
  using Type = AcStrategy::Type;
  switch (strategy) {
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

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // include guard
