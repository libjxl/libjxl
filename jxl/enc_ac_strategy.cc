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

#include "jxl/enc_ac_strategy.h"

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/enc_ac_strategy.cc"
#include <hwy/foreach_target.h>
// ^ must come before highway.h and any *-inl.h.

#include <hwy/highway.h>

#include "jxl/ac_strategy.h"
#include "jxl/ans_params.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/convolve.h"
#include "jxl/dct_scales.h"
#include "jxl/enc_params.h"
#include "jxl/enc_transforms-inl.h"
#include "jxl/entropy_coder.h"

// This must come before the begin/end_target, but HWY_ONCE is only true
// after that, so use an "include guard".
#ifndef JXL_ENC_AC_STRATEGY_
#define JXL_ENC_AC_STRATEGY_
// Parameters of the heuristic are marked with a OPTIMIZE comment.
namespace jxl {
namespace {

// Debugging utilities.

// Returns a linear sRGB color (as bytes) for each AC strategy.
const uint8_t* TypeColor(const uint8_t& raw_strategy) {
  JXL_ASSERT(AcStrategy::IsRawStrategyValid(raw_strategy));
  static_assert(AcStrategy::kNumValidStrategies == 21, "Change colors");
  static constexpr uint8_t kColors[][3] = {
      {0x00, 0xBB, 0xBB},  // DCT8
      {0x00, 0xFF, 0xFF},  // IDENTITY
      {0x00, 0xF8, 0xF8},  // DCT2x2
      {0x00, 0xF0, 0xF0},  // DCT4x4
      {0x00, 0x77, 0x77},  // DCT16x16
      {0x00, 0x33, 0x33},  // DCT32x32
      {0x00, 0x99, 0x99},  // DCT16x8
      {0x00, 0x99, 0x99},  // DCT8x16
      {0x00, 0x55, 0x55},  // DCT32x8
      {0x00, 0x55, 0x55},  // DCT8x32
      {0x00, 0x44, 0x44},  // DCT32x16
      {0x00, 0x44, 0x44},  // DCT16x32
      {0x00, 0xE8, 0xE8},  // DCT4x8
      {0x00, 0xE8, 0xE8},  // DCT8x4
      {0x00, 0xFF, 0xFF},  // AFV0
      {0x00, 0xFF, 0xFF},  // AFV1
      {0x00, 0xFF, 0xFF},  // AFV2
      {0x00, 0xFF, 0xFF},  // AFV3
      {0x00, 0x00, 0x00},  // DCT64x64
      {0x00, 0x19, 0x19},  // DCT64x32
      {0x00, 0x19, 0x19},  // DCT32x64
  };
  return kColors[raw_strategy];
}

const uint8_t* TypeMask(const uint8_t& raw_strategy) {
  JXL_ASSERT(AcStrategy::IsRawStrategyValid(raw_strategy));
  static_assert(AcStrategy::kNumValidStrategies == 21, "Add masks");
  static constexpr uint8_t kMask[][64] = {
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT8
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // IDENTITY
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 0, 0, 0, 1, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 1, 0, 0, 0, 0,  //
          0, 0, 1, 1, 1, 1, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // 2x2
      {
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
      },                           // 4x4
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 1, 1, 1, 1, 0, 0,  //
          0, 0, 1, 1, 1, 1, 0, 0,  //
          0, 0, 1, 1, 1, 1, 0, 0,  //
          0, 0, 1, 1, 1, 1, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT16x16
      {
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
      },                           // DCT32x32
      {
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
      },                           // DCT16x8
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT8x16
      {
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
      },                           // DCT32x8
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT8x32
      {
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
      },                           // DCT32x16
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT16x32
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT4x8
      {
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
      },                           // DCT8x4
      {
          1, 1, 0, 0, 0, 0, 0, 0,  //
          1, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // AFV0
      {
          0, 0, 0, 0, 0, 0, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // AFV1
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 0, 0, 0, 0, 0, 0,  //
      },                           // AFV2
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 1,  //
          0, 0, 0, 0, 0, 0, 1, 1,  //
      },                           // AFV3
      {
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
      },                           // DCT64x64
      {
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
          0, 0, 0, 1, 1, 0, 0, 0,  //
      },                           // DCT64x32
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT32x64
  };
  return kMask[raw_strategy];
}

void DumpAcStrategy(const AcStrategyImage& ac_strategy, size_t xsize,
                    size_t ysize, const char* tag, AuxOut* aux_out) {
  Image3F color_acs(xsize, ysize);
  for (size_t y = 0; y < ysize; y++) {
    float* JXL_RESTRICT rows[3] = {
        color_acs.PlaneRow(0, y),
        color_acs.PlaneRow(1, y),
        color_acs.PlaneRow(2, y),
    };
    const AcStrategyRow acs_row = ac_strategy.ConstRow(y / kBlockDim);
    for (size_t x = 0; x < xsize; x++) {
      AcStrategy acs = acs_row[x / kBlockDim];
      const uint8_t* JXL_RESTRICT color = TypeColor(acs.RawStrategy());
      for (size_t c = 0; c < 3; c++) {
        rows[c][x] = color[c];
      }
    }
  }
  size_t stride = color_acs.PixelsPerRow();
  const uint8_t highlight_color[3] = {0xFF, 0xFF, 0x00};
  for (size_t c = 0; c < 3; c++) {
    for (size_t by = 0; by < DivCeil(ysize, kBlockDim); by++) {
      float* JXL_RESTRICT row = color_acs.PlaneRow(c, by * kBlockDim);
      const AcStrategyRow acs_row = ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < DivCeil(xsize, kBlockDim); bx++) {
        AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock()) continue;
        const uint8_t* JXL_RESTRICT mask = TypeMask(acs.RawStrategy());
        size_t xstart = (acs.covered_blocks_x() - 1) * kBlockDim / 2;
        size_t ystart = (acs.covered_blocks_y() - 1) * kBlockDim / 2;
        for (size_t iy = 0;
             iy < kBlockDim && by * kBlockDim + iy + ystart < ysize; iy++) {
          for (size_t ix = 0;
               ix < kBlockDim && bx * kBlockDim + ix + xstart < xsize; ix++) {
            if (mask[iy * kBlockDim + ix]) {
              row[(iy + ystart) * stride + bx * kBlockDim + ix + xstart] =
                  highlight_color[c];
            }
          }
        }
      }
    }
  }
  aux_out->DumpImage(tag, color_acs);
}

// AC strategy selection: utility struct and entropy estimation.

// Highest observed token > 64.
constexpr size_t kNumTokens = ANS_MAX_ALPHABET_SIZE;

struct ACSConfig {
  const DequantMatrices* JXL_RESTRICT dequant;
  float token_bits[kNumTokens];
  float info_loss_multiplier;
  float* JXL_RESTRICT quant_field_row;
  size_t quant_field_stride;
  const float* JXL_RESTRICT src_rows[3];
  size_t src_stride;
  const float& Pixel(size_t c, size_t x, size_t y) const {
    return src_rows[c][y * src_stride + x];
  }
  float Quant(size_t bx, size_t by) const {
    JXL_DASSERT(quant_field_row[by * quant_field_stride + bx] > 0);
    return quant_field_row[by * quant_field_stride + bx];
  }
  void SetQuant(size_t bx, size_t by, float value) const {
    JXL_DASSERT(value > 0);
    quant_field_row[by * quant_field_stride + bx] = value;
  }
};

void ComputeTokenBits(float butteraugli_target, float* token_bits) {
  const double kSmallValueBase = 7.2618801707528009;
  const double kSmallValueMul = 61.512220067759564;
  const double kLargeValueFactor = 0.74418618655898428;

  const double kMaxCost = ANS_LOG_TAB_SIZE;

  const double kLargeParam = std::max(
      0.05f, 0.01f * std::pow(butteraugli_target, 0.1f) - 0.015f);  // OPTIMIZE
  const double kSmallParam = 8.25f * kLargeParam - 0.08913395766;

  for (size_t i = 0; i < 16; i++) {
    token_bits[i] =
        kSmallValueBase + kSmallValueMul *
                              (pow((i + 1) / 2, kSmallParam) +
                               pow(i ? (i - 1) / 2 : 0, kSmallParam)) *
                              0.5f;
  }
  for (size_t i = 16; i < kNumTokens; i++) {
    token_bits[i] = std::min(
        kMaxCost, std::exp(kLargeParam * i * (i * kLargeValueFactor + 1)));
  }
}

// AC strategy selection: recursive block splitting.

template <size_t N>
size_t ACSCandidates(const AcStrategy::Type (&in)[N],
                     AcStrategy::Type* JXL_RESTRICT out) {
  memcpy(out, in, N * sizeof(AcStrategy::Type));
  return N;
}

// Order in which transforms are tested for max delta: the first
// acceptable one is chosen as initial guess.
constexpr AcStrategy::Type kACSOrder[] = {
    AcStrategy::Type::DCT64X64,
    AcStrategy::Type::DCT64X32,
    AcStrategy::Type::DCT32X64,
    AcStrategy::Type::DCT32X32,
    AcStrategy::Type::DCT32X16,
    AcStrategy::Type::DCT16X32,
    AcStrategy::Type::DCT16X16,
    AcStrategy::Type::DCT8X32,
    AcStrategy::Type::DCT32X8,
    AcStrategy::Type::DCT16X8,
    AcStrategy::Type::DCT8X16,
    // DCT8x8 is the "fallback" option if no bigger transform can be used.
    AcStrategy::Type::DCT,
};

size_t ACSPossibleReplacements(AcStrategy::Type current,
                               AcStrategy::Type* JXL_RESTRICT out) {
  // TODO(veluca): is this decision tree optimal?
  if (current == AcStrategy::Type::DCT64X64) {
    return ACSCandidates(
        {AcStrategy::Type::DCT64X32, AcStrategy::Type::DCT32X64, AcStrategy::Type::DCT32X32}, out);
  }
  if (current == AcStrategy::Type::DCT64X32 ||
      current == AcStrategy::Type::DCT32X64) {
    return ACSCandidates({AcStrategy::Type::DCT32X32}, out);
  }
  if (current == AcStrategy::Type::DCT32X32) {
    return ACSCandidates(
        {AcStrategy::Type::DCT32X16, AcStrategy::Type::DCT16X32}, out);
  }
  if (current == AcStrategy::Type::DCT32X16) {
    return ACSCandidates(
        {AcStrategy::Type::DCT32X8, AcStrategy::Type::DCT16X16}, out);
  }
  if (current == AcStrategy::Type::DCT16X32) {
    return ACSCandidates(
        {AcStrategy::Type::DCT8X32, AcStrategy::Type::DCT16X16}, out);
  }
  if (current == AcStrategy::Type::DCT32X8) {
    return ACSCandidates({AcStrategy::Type::DCT16X8, AcStrategy::Type::DCT},
                         out);
  }
  if (current == AcStrategy::Type::DCT8X32) {
    return ACSCandidates({AcStrategy::Type::DCT8X16, AcStrategy::Type::DCT},
                         out);
  }
  if (current == AcStrategy::Type::DCT16X16) {
    return ACSCandidates({AcStrategy::Type::DCT8X16, AcStrategy::Type::DCT16X8},
                         out);
  }
  if (current == AcStrategy::Type::DCT16X8 ||
      current == AcStrategy::Type::DCT8X16) {
    return ACSCandidates({AcStrategy::Type::DCT}, out);
  }
  if (current == AcStrategy::Type::DCT) {
    return ACSCandidates({AcStrategy::Type::DCT4X8, AcStrategy::Type::DCT8X4,
                          AcStrategy::Type::DCT4X4, AcStrategy::Type::DCT2X2,
                          AcStrategy::Type::IDENTITY, AcStrategy::Type::AFV0,
                          AcStrategy::Type::AFV1, AcStrategy::Type::AFV2,
                          AcStrategy::Type::AFV3},
                         out);
  }
  // Other 8x8 have no replacements - they already were chosen as the best
  // between all the 8x8s.
  return 0;
}

}  // namespace
}  // namespace jxl
#endif  // JXL_ENC_AC_STRATEGY_

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

float EstimateEntropy(const AcStrategy& acs, size_t x, size_t y,
                      const ACSConfig& config,
                      const float* JXL_RESTRICT cmap_factors, float* block,
                      float* scratch_space, int* quantized) {
  const size_t size = (1 << acs.log2_covered_blocks()) * kDCTBlockSize;

  // Apply transform.
  for (size_t c = 0; c < 3; c++) {
    float* JXL_RESTRICT block_c = block + size * c;
    std::fill(block_c, block_c + size, 0.0f);
    TransformFromPixels(acs.Strategy(), &config.Pixel(c, x, y),
                        config.src_stride, block_c, scratch_space);
  }

  float quant = 0;
  // Load QF value
  for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
    for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
      quant = std::max(quant, config.Quant(x / 8 + ix, y / 8 + iy));
    }
  }

  // Compute entropy.
  // Keep separate accumulators for integer and float parts to avoid unnecessary
  // conversions.
  float entropy_f = 35.0;
  int entropy_i = 0;
  HWY_FULL(float) df;
  HWY_FULL(int) di;
  auto info_loss = Zero(df);
  const coeff_order_t* JXL_RESTRICT order = acs.NaturalCoeffOrder();
  for (size_t c = 0; c < 3; c++) {
    int extra_nbits = 0;
    float extra_tbits = 0.0;
    size_t num_nzeros = 0;
    size_t num_blocks = acs.covered_blocks_x() * acs.covered_blocks_y();
    const float* inv_matrix = config.dequant->InvMatrix(acs.RawStrategy(), c);
    const auto cmap_factor = Set(df, cmap_factors[c]);
    const auto q = Set(df, quant);
    for (size_t i = 0; i < num_blocks * kDCTBlockSize; i += Lanes(df)) {
      const auto in = Load(df, block + c * size + i);
      const auto in_y = Load(df, block + size + i) * cmap_factor;
      const auto im = Load(df, inv_matrix + i);
      const auto val = (in - in_y) * im * q;
      const auto rval = Round(val);
      info_loss += AbsDiff(val, rval);
      Store(ConvertTo(di, rval), di, quantized + i);
    }
    for (size_t i = num_blocks; i < num_blocks * kDCTBlockSize; i++) {
      size_t k = order[i];
      uint32_t token, nbits, bits;
      HybridUintConfig().Encode(PackSigned(quantized[k]), &token, &nbits,
                                &bits);
      // nbits + bits for token. Skip trailing zeros in natural coeff order.
      extra_nbits += nbits;
      extra_tbits += config.token_bits[token];
      if (quantized[k] != 0) {
        num_nzeros++;
        entropy_i += extra_nbits;
        entropy_f += extra_tbits;
        extra_tbits = 0;
        extra_nbits = 0;
      }
    }

    // Add #bit of num_nonzeros, as an estimate of the cost for encoding the
    // number of non-zeros of the block.
    size_t nbits = CeilLog2Nonzero(num_nzeros + 1) + 1;
    // Also add #bit of #bit of num_nonzeros, to estimate the ANS cost, with a
    // bias.
    entropy_i += CeilLog2Nonzero(nbits + 17) + nbits;
  }
  float ret = entropy_i + entropy_f +
              config.info_loss_multiplier * GetLane(SumOfLanes(info_loss));
  return ret;
}

void MaybeReplaceACS(size_t bx, size_t by, const ACSConfig& config,
                     const float* JXL_RESTRICT cmap_factors,
                     AcStrategyImage* JXL_RESTRICT ac_strategy,
                     float* JXL_RESTRICT entropy_estimate, float* block,
                     float* scratch_space, int* quantized) {
  AcStrategy::Type current =
      AcStrategy::Type(ac_strategy->ConstRow(by)[bx].RawStrategy());
  AcStrategy::Type candidates[AcStrategy::kNumValidStrategies];
  size_t num_candidates = ACSPossibleReplacements(current, candidates);
  if (num_candidates == 0) return;
  size_t best = num_candidates;
  size_t best_ee = entropy_estimate[0];
  // For each candidate replacement strategy, keep track of its entropy
  // estimate.
  float ee_val[AcStrategy::kNumValidStrategies][AcStrategy::kMaxCoeffBlocks];
  AcStrategy current_acs = AcStrategy::FromRawStrategy(current);
  if (current == AcStrategy::Type::DCT64X64) {
    best_ee *= 0.69;
  }
  for (size_t cand = 0; cand < num_candidates; cand++) {
    AcStrategy acs = AcStrategy::FromRawStrategy(candidates[cand]);
    size_t idx = 0;
    float total_entropy = 0;
    for (size_t iy = 0; iy < current_acs.covered_blocks_y();
         iy += acs.covered_blocks_y()) {
      for (size_t ix = 0; ix < current_acs.covered_blocks_x();
           ix += acs.covered_blocks_x()) {
        float entropy =
            EstimateEntropy(acs, (bx + ix) * 8, (by + iy) * 8, config,
                            cmap_factors, block, scratch_space, quantized);
        if (acs.RawStrategy() == AcStrategy::Type::DCT) {
          entropy *= 0.96;
        }
        if (acs.RawStrategy() == AcStrategy::Type::DCT4X4) {
          entropy += 40.0;
          entropy *= 0.88;
        }
        if (acs.RawStrategy() == AcStrategy::Type::DCT2X2) {
          entropy += 40.0;
          entropy *= 1.028;
        }
        if (acs.RawStrategy() == AcStrategy::Type::DCT16X16) {
          entropy *= 0.99;
        }
        if (acs.RawStrategy() == AcStrategy::Type::DCT64X32 ||
            acs.RawStrategy() == AcStrategy::Type::DCT32X64) {
          entropy *= 0.73;
        }
        if (acs.RawStrategy() == AcStrategy::Type::DCT32X32) {
          entropy *= 0.8;
        }
        if (acs.RawStrategy() == AcStrategy::Type::DCT16X32 ||
            acs.RawStrategy() == AcStrategy::Type::DCT32X16) {
          entropy *= 0.992;
        }
        if (acs.RawStrategy() == AcStrategy::Type::DCT32X8 ||
            acs.RawStrategy() == AcStrategy::Type::DCT8X32) {
          entropy *= 0.96;
        }
        if (acs.RawStrategy() == AcStrategy::Type::DCT16X8 ||
            acs.RawStrategy() == AcStrategy::Type::DCT8X16) {
          entropy *= 0.95;
        }
        if (acs.RawStrategy() == AcStrategy::Type::DCT4X8 ||
            acs.RawStrategy() == AcStrategy::Type::DCT8X4) {
          entropy += 30.0;
          entropy *= 1.045;
        }
        if (acs.RawStrategy() == AcStrategy::Type::IDENTITY) {
          entropy += 80.0;
          entropy *= 1.33;
        }
        if (acs.RawStrategy() == AcStrategy::Type::AFV0 ||
            acs.RawStrategy() == AcStrategy::Type::AFV1 ||
            acs.RawStrategy() == AcStrategy::Type::AFV2 ||
            acs.RawStrategy() == AcStrategy::Type::AFV3) {
          entropy += 30;
          entropy *= 0.995;
        }
        ee_val[cand][idx] = entropy;
        total_entropy += entropy;
        idx++;
      }
    }
    if (total_entropy < best_ee) {
      best_ee = total_entropy;
      best = cand;
    }
  }
  // Nothing changed.
  if (best == num_candidates) return;
  AcStrategy acs = AcStrategy::FromRawStrategy(candidates[best]);
  size_t idx = 0;
  for (size_t y = 0; y < current_acs.covered_blocks_y();
       y += acs.covered_blocks_y()) {
    for (size_t x = 0; x < current_acs.covered_blocks_x();
         x += acs.covered_blocks_x()) {
      ac_strategy->Set(bx + x, by + y, candidates[best]);
      for (size_t iy = y; iy < y + acs.covered_blocks_y(); iy++) {
        for (size_t ix = x; ix < x + acs.covered_blocks_x(); ix++) {
          entropy_estimate[iy * 8 + ix] = ee_val[best][idx];
        }
      }
      idx++;
    }
  }
}

void FindBestAcStrategy(const Image3F& src,
                        PassesEncoderState* JXL_RESTRICT enc_state,
                        ThreadPool* pool, AuxOut* aux_out) {
  PROFILER_FUNC;
  const CompressParams& cparams = enc_state->cparams;
  const float butteraugli_target = cparams.butteraugli_distance;
  AcStrategyImage* ac_strategy = &enc_state->shared.ac_strategy;

  const size_t xsize_blocks = enc_state->shared.frame_dim.xsize_blocks;
  const size_t ysize_blocks = enc_state->shared.frame_dim.ysize_blocks;
  // In Falcon mode, use DCT8 everywhere and uniform quantization.
  if (cparams.speed_tier == SpeedTier::kFalcon) {
    ac_strategy->FillDCT8();
    return;
  }

  constexpr bool kOnly64 = false;
  if (kOnly64) {
    ac_strategy->FillDCT8();
    for (size_t y = 0; y + 7 < ysize_blocks; y += 8) {
      for (size_t x = 0; x + 7 < xsize_blocks; x += 8) {
        if ((x + y) % 16 == 0) {
          ac_strategy->Set(x, y, AcStrategy::DCT64X64);
        } else if ((x + y) % 32 == 8) {
          ac_strategy->Set(x, y, AcStrategy::DCT64X32);
          ac_strategy->Set(x + 4, y, AcStrategy::DCT64X32);
        } else if ((x + y) % 32 == 24) {
          ac_strategy->Set(x, y, AcStrategy::DCT32X64);
          ac_strategy->Set(x, y + 4, AcStrategy::DCT32X64);
        }
      }
    }
    return;
  }

  // Maximum delta that every strategy type is allowed to have in the area
  // it covers. Ignored for 8x8 transforms.
  const float kMaxDelta = 0.10f * sqrt(butteraugli_target);  // OPTIMIZE
  const float kFlat = 5.0f * sqrt(butteraugli_target);       // OPTIMIZE

  // Scale of channels when computing delta.
  const float kDeltaScale[3] = {
      9.4174165405614652,
      1.0,
      0.2,
  };

  ACSConfig config;
  config.dequant = &enc_state->shared.matrices;

  // Entropy estimate is composed of two factors:
  //  - estimate of the number of bits that will be used by the block
  //  - information loss due to quantization
  // The following constant controls the relative weights of these components.
  config.info_loss_multiplier = 234;

  ComputeTokenBits(butteraugli_target, config.token_bits);

  // Image row pointers and strides.
  config.quant_field_row = enc_state->initial_quant_field.Row(0);
  config.quant_field_stride = enc_state->initial_quant_field.PixelsPerRow();

  config.src_rows[0] = src.ConstPlaneRow(0, 0);
  config.src_rows[1] = src.ConstPlaneRow(1, 0);
  config.src_rows[2] = src.ConstPlaneRow(2, 0);
  config.src_stride = src.PixelsPerRow();

  size_t xsize64 = DivCeil(xsize_blocks, 8);
  size_t ysize64 = DivCeil(ysize_blocks, 8);
  const auto compute_initial_acs_guess = [&](int block64, int _) {
    auto mem = hwy::AllocateAligned<float>(5 * AcStrategy::kMaxCoeffArea);
    auto qmem = hwy::AllocateAligned<int>(AcStrategy::kMaxCoeffArea);
    int* JXL_RESTRICT quantized = qmem.get();
    float* JXL_RESTRICT block = mem.get();
    float* JXL_RESTRICT scratch_space =
        mem.get() + 3 * AcStrategy::kMaxCoeffArea;
    size_t bx = block64 % xsize64;
    size_t by = block64 / xsize64;
    size_t tx = bx * 8 / kColorTileDimInBlocks;
    size_t ty = by * 8 / kColorTileDimInBlocks;
    const float cmap_factors[3] = {
        enc_state->shared.cmap.YtoXRatio(
            enc_state->shared.cmap.ytox_map.ConstRow(ty)[tx]),
        0.0f,
        enc_state->shared.cmap.YtoBRatio(
            enc_state->shared.cmap.ytob_map.ConstRow(ty)[tx]),
    };
    HWY_CAPPED(float, kBlockDim) d;
    const size_t N = Lanes(d);
    // Pre-compute maximum delta in each 8x8 block.
    // Find a minimum delta of three options:
    // 1) all, 2) not accounting vertical, 3) not accounting horizontal
    HWY_ALIGN float pixels[3][64];
    float max_delta[3][64] = {};
    float flat[3][64] = {};
    float entropy_estimate[64] = {};
    for (size_t c = 0; c < 3; c++) {
      for (size_t iy = 0; iy < 8; iy++) {
        size_t dy = by * 8 + iy;
        if (dy >= ysize_blocks) continue;
        for (size_t ix = 0; ix < 8; ix++) {
          size_t dx = bx * 8 + ix;
          if (dx >= xsize_blocks) continue;
          for (size_t c = 0; c < 3; c++) {
            for (size_t y = 0; y < 8; y++) {
              for (size_t x = 0; x < 8; x += N) {
                const auto v =
                    Load(d, &config.Pixel(c, dx * 8 + x, dy * 8 + y));
                Store(v, d, &pixels[c][y * 8 + x]);
              }
            }
          }
          for (auto& pixel : pixels) {
            // Sums of rows
            float side[8];
            for (size_t y = 0; y < 8; y++) {
              auto sum = Load(d, &pixel[y * 8]);
              for (size_t x = N; x < 8; x += N) {
                sum += Load(d, &pixel[y * 8 + x]);
              }
              side[y] = GetLane(SumOfLanes(sum));
            }

            // Sum of columns (one per lane).
            HWY_ALIGN float top[8];
            for (size_t x = 0; x < 8; x += N) {
              auto sums_of_columns = Load(d, &pixel[x]);
              for (size_t y = 1; y < 8; y++) {
                sums_of_columns += Load(d, &pixel[y * 8 + x]);
              }
              Store(sums_of_columns, d, top + x);
            }

            // Subtract fraction of row+col sums from each pixel
            const auto mul = Set(d, 1.0f / 8);
            for (size_t y = 0; y < 8; y++) {
              const auto side_y = Set(d, side[y]) * mul;
              for (size_t x = 0; x < 8; x += N) {
                const auto top_x = Load(d, &top[x]);
                auto v = Load(d, &pixel[y * 8 + x]);
                v -= MulAdd(mul, top_x, side_y);
                Store(v, d, &pixel[y * 8 + x]);
              }
            }
          }
          auto delta = Zero(d);
          for (size_t x = 0; x < 8; x += N) {
            for (size_t y = 1; y < 7; y++) {
              float* pix = &pixels[c][y * 8 + x];
              const auto p = Load(d, pix);
              const auto n = Load(d, pix + 8);
              const auto s = Load(d, pix - 8);
              const auto w = LoadU(d, pix - 1);
              const auto e = LoadU(d, pix + 1);
              // Compute amount of per-pixel variation.
              const auto m1 = Max(AbsDiff(n, p), AbsDiff(s, p));
              const auto m2 = Max(AbsDiff(w, p), AbsDiff(e, p));
              const auto m3 = Max(AbsDiff(e, w), AbsDiff(s, n));
              const auto m4 = Max(m1, m2);
              const auto m5 = Max(m3, m4);
              delta = Max(delta, m5);
            }
            HWY_ALIGN float lmax[MaxLanes(d)];
            Store(delta, d, lmax);
            float mdelta = 0;
            for (size_t i = 0; i < N; i++) {
              int ioff = (i + x) & 7;
              if (ioff != 0 && ioff != 7) {
                mdelta = std::max(mdelta, lmax[i]);
              }
            }
            max_delta[c][iy * 8 + ix] =
                std::max(max_delta[c][iy * 8 + ix], mdelta);
          }
          // How 'flat' is this area, i.e., how observable would ringing
          // artefacts be here?
          for (size_t c = 0; c < 3; c++) {
            HWY_ALIGN float s_vals[64];
            for (size_t y = 0; y < 8; y++) {
              for (size_t x = 0; x < 8; x++) {
                float v = pixels[c][y * 8 + x];
                float s = 0;
                if (y >= 2) {
                  s += std::fabs(v - pixels[c][y * 8 + x - 16]);
                }
                if (y < 6) {
                  s += std::fabs(v - pixels[c][y * 8 + x + 16]);
                }
                if (x >= 2) {
                  s += std::fabs(v - pixels[c][y * 8 + x - 2]);
                }
                if (x < 6) {
                  s += std::fabs(v - pixels[c][y * 8 + x + 2]);
                }
                s_vals[y * 8 + x] = s;
              }
            }
            const auto smul = Set(d, 0.25f * kFlat * kDeltaScale[c]);
            const auto one = Set(d, 1.0f);
            const auto numerator = Set(d, 1.0f / 48.0f);
            auto accum = Zero(d);
            for (size_t i = 0; i < 64; i += Lanes(d)) {
              const auto sv = Load(d, s_vals + i) * smul;
              accum += numerator / MulAdd(sv, sv, one);
            }
            flat[c][iy * 8 + ix] = GetLane(SumOfLanes(accum));
          }
        }
      }
    }
    for (size_t k = 0; k < 64; ++k) {
      //      flat[0][k] = std::max<float>(flat[1][k], flat[0][k]);
      flat[2][k] = flat[1][k];
    }
    for (size_t i = 0; i < 3; ++i) {
      for (size_t k = 0; k < 64; ++k) {
        max_delta[i][k] *= kDeltaScale[i];
      }
    }
    // Choose the first transform that can be used to cover each block.
    uint8_t chosen_mask[64] = { 0 };
    for (size_t iy = 0; iy < 8 && by * 8 + iy < ysize_blocks; iy++) {
      for (size_t ix = 0; ix < 8 && bx * 8 + ix < xsize_blocks; ix++) {
        if (chosen_mask[iy * 8 + ix]) continue;
        for (auto i : kACSOrder) {
          AcStrategy acs = AcStrategy::FromRawStrategy(i);
          size_t cx = acs.covered_blocks_x();
          size_t cy = acs.covered_blocks_y();
          float max_delta_v[3] = {max_delta[0][iy * 8 + ix],
                                  max_delta[1][iy * 8 + ix],
                                  max_delta[2][iy * 8 + ix]};
          float max2_delta_v[3] = {0, 0, 0};
          float max_flatness[3] = {0, 0, 0};
          float max_delta_acs = std::max(
              std::max(max_delta_v[0], max_delta_v[1]), max_delta_v[2]);
          float min_delta_v[3] = {1e30, 1e30, 1e30};
          float ave_delta_v[3] = {};
          // Check if strategy is usable
          if (cx != 1 || cy != 1) {
            // Alignment
            if ((iy & (cy - 1)) != 0) continue;
            if ((ix & (cx - 1)) != 0) continue;
            // Out of block64 bounds
            if (iy + cy > 8) continue;
            if (ix + cx > 8) continue;
            // Out of image bounds
            if (by * 8 + iy + cy > ysize_blocks) continue;
            if (bx * 8 + ix + cx > xsize_blocks) continue;
            // Block would overwrite an already-chosen block
            bool overwrites_covered = false;
            for (size_t y = 0; y < cy; y++) {
              for (size_t x = 0; x < cx; x++) {
                if (chosen_mask[(y + iy) * 8 + x + ix])
                  overwrites_covered = true;
              }
            }
            if (overwrites_covered) continue;
            for (size_t c = 0; c < 3; ++c) {
              max_delta_v[c] = 0;
              max2_delta_v[c] = 0;
              min_delta_v[c] = 1e30f;
              ave_delta_v[c] = 0;
              max_flatness[c] = 0;
              // Max delta in covered area
              for (size_t y = 0; y < cy; y++) {
                for (size_t x = 0; x < cx; x++) {
                  int pix = (iy + y) * 8 + ix + x;
                  max_flatness[c] =
                      std::max<float>(max_flatness[c], flat[c][pix]);
                  if (max_delta_v[c] < max_delta[c][pix]) {
                    max2_delta_v[c] = max_delta_v[c];
                    max_delta_v[c] = max_delta[c][pix];
                  } else if (max2_delta_v[c] < max_delta[c][pix]) {
                    max2_delta_v[c] = max_delta[c][pix];
                  }
                  min_delta_v[c] = std::min(min_delta_v[c], max_delta[c][pix]);
                  ave_delta_v[c] += max_delta[c][pix];
                }
              }
              ave_delta_v[c] -= max_delta_v[c];
              if (cy * cx >= 5) {
                ave_delta_v[c] -= max2_delta_v[c];
                ave_delta_v[c] /= (cy * cx - 2);
              } else {
                ave_delta_v[c] /= (cy * cx - 1);
              }
              max_delta_v[c] -= 0.03 * max2_delta_v[c];
              max_delta_v[c] -= 0.25 * min_delta_v[c];
              max_delta_v[c] -= 0.25 * ave_delta_v[c];
              max_delta_v[c] *= max_flatness[c];
            }
            max_delta_acs = max_delta_v[0] + max_delta_v[1] + max_delta_v[2];
            max_delta_acs *= pow(1.044, cx * cy);
            if (max_delta_acs > kMaxDelta) continue;
          }
          // Estimate entropy and qf value
          float entropy = 0.0f;
          // In modes faster than Wombat mode, AC strategy replacement is not
          // attempted: no need to estimate entropy.
          if (cparams.speed_tier <= SpeedTier::kWombat) {
            entropy =
                EstimateEntropy(acs, bx * 64 + ix * 8, by * 64 + iy * 8, config,
                                cmap_factors, block, scratch_space, quantized);
          }
          // In modes faster than Hare mode, we don't use InitialQuantField -
          // hence, we need to come up with quant field values.
          if (cparams.speed_tier > SpeedTier::kHare) {
            // OPTIMIZE
            float quant = 1.1f / (1.0f + max_delta_acs) / butteraugli_target;
            for (size_t y = 0; y < cy; y++) {
              for (size_t x = 0; x < cx; x++) {
                config.SetQuant(bx * 8 + ix + x, by * 8 + iy + y, quant);
              }
            }
          }
          // Mark blocks as chosen and write to acs image.
          ac_strategy->Set(bx * 8 + ix, by * 8 + iy, i);
          for (size_t y = 0; y < cy; y++) {
            for (size_t x = 0; x < cx; x++) {
              chosen_mask[(y + iy) * 8 + x + ix] = 1;
              entropy_estimate[(iy + y) * 8 + ix + x] = entropy;
            }
          }
          break;
        }
      }
    }
    // Do not try to replace ACS in modes faster than wombat mode.
    if (cparams.speed_tier > SpeedTier::kWombat) return;
    // Iterate through the 32-block attempting to replace the current strategy.
    // If replaced, repeat for the top-left new block and let the other ones be
    // taken care of by future iterations.
    uint8_t computed_mask[64] = {};
    for (size_t iy = 0; iy < 8; iy++) {
      if (by * 8 + iy >= ysize_blocks) continue;
      for (size_t ix = 0; ix < 8; ix++) {
        if (bx * 8 + ix >= xsize_blocks) continue;
        if (computed_mask[iy * 8 + ix]) continue;
        uint8_t prev = AcStrategy::kNumValidStrategies;
        while (prev !=
               ac_strategy->ConstRow(by * 8 + iy)[bx * 8 + ix].RawStrategy()) {
          prev = ac_strategy->ConstRow(by * 8 + iy)[bx * 8 + ix].RawStrategy();
          MaybeReplaceACS(bx * 8 + ix, by * 8 + iy, config, cmap_factors,
                          ac_strategy, entropy_estimate + (iy * 8 + ix), block,
                          scratch_space, quantized);
        }
        AcStrategy acs = ac_strategy->ConstRow(by * 8 + iy)[bx * 8 + ix];
        for (size_t y = 0; y < acs.covered_blocks_y(); y++) {
          for (size_t x = 0; x < acs.covered_blocks_x(); x++) {
            computed_mask[(iy + y) * 8 + ix + x] = 1;
          }
        }
      }
    }
  };
  RunOnPool(pool, 0, xsize64 * ysize64, ThreadPool::SkipInit(),
            compute_initial_acs_guess, "ChooseACS");

  // Accounting and debug output.
  if (aux_out != nullptr) {
    aux_out->num_dct2_blocks =
        32 * (ac_strategy->CountBlocks(AcStrategy::Type::DCT32X64) +
              ac_strategy->CountBlocks(AcStrategy::Type::DCT64X32));
    aux_out->num_dct4_blocks =
        64 * ac_strategy->CountBlocks(AcStrategy::Type::DCT64X64);
    aux_out->num_dct4x8_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT4X8) +
        ac_strategy->CountBlocks(AcStrategy::Type::DCT8X4);
    aux_out->num_afv_blocks = ac_strategy->CountBlocks(AcStrategy::Type::AFV0) +
                              ac_strategy->CountBlocks(AcStrategy::Type::AFV1) +
                              ac_strategy->CountBlocks(AcStrategy::Type::AFV2) +
                              ac_strategy->CountBlocks(AcStrategy::Type::AFV3);
    aux_out->num_dct8_blocks = ac_strategy->CountBlocks(AcStrategy::Type::DCT);
    aux_out->num_dct8x16_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT8X16) +
        ac_strategy->CountBlocks(AcStrategy::Type::DCT16X8);
    aux_out->num_dct8x32_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT8X32) +
        ac_strategy->CountBlocks(AcStrategy::Type::DCT32X8);
    aux_out->num_dct16_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT16X16);
    aux_out->num_dct16x32_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT16X32) +
        ac_strategy->CountBlocks(AcStrategy::Type::DCT32X16);
    aux_out->num_dct32_blocks =
        ac_strategy->CountBlocks(AcStrategy::Type::DCT32X32);
  }

  if (WantDebugOutput(aux_out)) {
    DumpAcStrategy(*ac_strategy, enc_state->shared.frame_dim.xsize,
                   enc_state->shared.frame_dim.ysize, "ac_strategy", aux_out);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(FindBestAcStrategy);
void FindBestAcStrategy(const Image3F& src,
                        PassesEncoderState* JXL_RESTRICT enc_state,
                        ThreadPool* pool, AuxOut* aux_out) {
  return HWY_DYNAMIC_DISPATCH(FindBestAcStrategy)(src, enc_state, pool,
                                                  aux_out);
}

}  // namespace jxl
#endif  // HWY_ONCE
