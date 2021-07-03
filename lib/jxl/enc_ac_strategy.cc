// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_ac_strategy.h"

#include <stdint.h>
#include <string.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_ac_strategy.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/dct_scales.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_transforms-inl.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/fast_math-inl.h"

// Some of the floating point constants in this file and in other
// files in the libjxl project have been obtained using the
// tools/optimizer/simplex_fork.py tool. It is a variation of
// Nelder-Mead optimization, and we generally try to minimize
// BPP * pnorm aggregate as reported by the benchmark_xl tool,
// but occasionally the values are optimized by using additional
// constraints such as maintaining a certain density, or ratio of
// popularity of integral transforms. Jyrki visually reviews all
// such changes and often makes manual changes to maintain good
// visual quality to changes where butteraugli was not sufficiently
// sensitive to some kind of degradation. Unfortunately image quality
// is still more of an art than science.

// This must come before the begin/end_target, but HWY_ONCE is only true
// after that, so use an "include guard".
#ifndef LIB_JXL_ENC_AC_STRATEGY_
#define LIB_JXL_ENC_AC_STRATEGY_
// Parameters of the heuristic are marked with a OPTIMIZE comment.
namespace jxl {

// Debugging utilities.

// Returns a linear sRGB color (as bytes) for each AC strategy.
const uint8_t* TypeColor(const uint8_t& raw_strategy) {
  JXL_ASSERT(AcStrategy::IsRawStrategyValid(raw_strategy));
  static_assert(AcStrategy::kNumValidStrategies == 27, "Change colors");
  static constexpr uint8_t kColors[][3] = {
      {0xFF, 0xFF, 0x00},  // DCT8
      {0xFF, 0x80, 0x80},  // HORNUSS
      {0xFF, 0x80, 0x80},  // DCT2x2
      {0xFF, 0x80, 0x80},  // DCT4x4
      {0x80, 0xFF, 0x00},  // DCT16x16
      {0x00, 0xC0, 0x00},  // DCT32x32
      {0xC0, 0xFF, 0x00},  // DCT16x8
      {0xC0, 0xFF, 0x00},  // DCT8x16
      {0x00, 0xFF, 0x00},  // DCT32x8
      {0x00, 0xFF, 0x00},  // DCT8x32
      {0x00, 0xFF, 0x00},  // DCT32x16
      {0x00, 0xFF, 0x00},  // DCT16x32
      {0xFF, 0x80, 0x00},  // DCT4x8
      {0xFF, 0x80, 0x00},  // DCT8x4
      {0xFF, 0xFF, 0x80},  // AFV0
      {0xFF, 0xFF, 0x80},  // AFV1
      {0xFF, 0xFF, 0x80},  // AFV2
      {0xFF, 0xFF, 0x80},  // AFV3
      {0x00, 0xC0, 0xFF},  // DCT64x64
      {0x00, 0xFF, 0xFF},  // DCT64x32
      {0x00, 0xFF, 0xFF},  // DCT32x64
      {0x00, 0x40, 0xFF},  // DCT128x128
      {0x00, 0x80, 0xFF},  // DCT128x64
      {0x00, 0x80, 0xFF},  // DCT64x128
      {0x00, 0x00, 0xC0},  // DCT256x256
      {0x00, 0x00, 0xFF},  // DCT256x128
      {0x00, 0x00, 0xFF},  // DCT128x256
  };
  return kColors[raw_strategy];
}

const uint8_t* TypeMask(const uint8_t& raw_strategy) {
  JXL_ASSERT(AcStrategy::IsRawStrategyValid(raw_strategy));
  static_assert(AcStrategy::kNumValidStrategies == 27, "Add masks");
  // implicitly, first row and column is made dark
  static constexpr uint8_t kMask[][64] = {
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT8
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 1, 1, 1, 1, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 1, 0, 0, 1, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // HORNUSS
      {
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          1, 0, 1, 0, 1, 0, 1, 0,  //
      },                           // 2x2
      {
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
      },                           // 4x4
      {},                          // DCT16x16 (unused)
      {},                          // DCT32x32 (unused)
      {},                          // DCT16x8 (unused)
      {},                          // DCT8x16 (unused)
      {},                          // DCT32x8 (unused)
      {},                          // DCT8x32 (unused)
      {},                          // DCT32x16 (unused)
      {},                          // DCT16x32 (unused)
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // DCT4x8
      {
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
          0, 0, 0, 0, 1, 0, 0, 0,  //
      },                           // DCT8x4
      {
          1, 1, 1, 1, 1, 0, 0, 0,  //
          1, 1, 1, 1, 0, 0, 0, 0,  //
          1, 1, 1, 0, 0, 0, 0, 0,  //
          1, 1, 0, 0, 0, 0, 0, 0,  //
          1, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
      },                           // AFV0
      {
          0, 0, 0, 0, 1, 1, 1, 1,  //
          0, 0, 0, 0, 0, 1, 1, 1,  //
          0, 0, 0, 0, 0, 0, 1, 1,  //
          0, 0, 0, 0, 0, 0, 0, 1,  //
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
          1, 0, 0, 0, 0, 0, 0, 0,  //
          1, 1, 0, 0, 0, 0, 0, 0,  //
          1, 1, 1, 0, 0, 0, 0, 0,  //
          1, 1, 1, 1, 0, 0, 0, 0,  //
      },                           // AFV2
      {
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 0,  //
          0, 0, 0, 0, 0, 0, 0, 1,  //
          0, 0, 0, 0, 0, 0, 1, 1,  //
          0, 0, 0, 0, 0, 1, 1, 1,  //
      },                           // AFV3
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
        rows[c][x] = color[c] / 255.f;
      }
    }
  }
  size_t stride = color_acs.PixelsPerRow();
  for (size_t c = 0; c < 3; c++) {
    for (size_t by = 0; by < DivCeil(ysize, kBlockDim); by++) {
      float* JXL_RESTRICT row = color_acs.PlaneRow(c, by * kBlockDim);
      const AcStrategyRow acs_row = ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < DivCeil(xsize, kBlockDim); bx++) {
        AcStrategy acs = acs_row[bx];
        if (!acs.IsFirstBlock()) continue;
        const uint8_t* JXL_RESTRICT color = TypeColor(acs.RawStrategy());
        const uint8_t* JXL_RESTRICT mask = TypeMask(acs.RawStrategy());
        if (acs.covered_blocks_x() == 1 && acs.covered_blocks_y() == 1) {
          for (size_t iy = 0; iy < kBlockDim && by * kBlockDim + iy < ysize;
               iy++) {
            for (size_t ix = 0; ix < kBlockDim && bx * kBlockDim + ix < xsize;
                 ix++) {
              if (mask[iy * kBlockDim + ix]) {
                row[iy * stride + bx * kBlockDim + ix] = color[c] / 800.f;
              }
            }
          }
        }
        // draw block edges
        for (size_t ix = 0; ix < kBlockDim * acs.covered_blocks_x() &&
                            bx * kBlockDim + ix < xsize;
             ix++) {
          row[0 * stride + bx * kBlockDim + ix] = color[c] / 350.f;
        }
        for (size_t iy = 0; iy < kBlockDim * acs.covered_blocks_y() &&
                            by * kBlockDim + iy < ysize;
             iy++) {
          row[iy * stride + bx * kBlockDim + 0] = color[c] / 350.f;
        }
      }
    }
  }
  aux_out->DumpImage(tag, color_acs);
}

}  // namespace jxl
#endif  // LIB_JXL_ENC_AC_STRATEGY_

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

using hwy::HWY_NAMESPACE::ShiftLeft;
using hwy::HWY_NAMESPACE::ShiftRight;

float EstimateEntropy(const AcStrategy& acs, size_t x, size_t y,
                      const ACSConfig& config,
                      const float* JXL_RESTRICT cmap_factors, float* block,
                      float* scratch_space, uint32_t* quantized) {
  const size_t size = (1 << acs.log2_covered_blocks()) * kDCTBlockSize;

  // Apply transform.
  for (size_t c = 0; c < 3; c++) {
    float* JXL_RESTRICT block_c = block + size * c;
    TransformFromPixels(acs.Strategy(), &config.Pixel(c, x, y),
                        config.src_stride, block_c, scratch_space);
  }

  HWY_FULL(float) df;

  const size_t num_blocks = acs.covered_blocks_x() * acs.covered_blocks_y();
  float quant_norm8 = 0;
  float masking = 0;
  if (num_blocks == 1) {
    // When it is only one 8x8, we don't need aggregation of values.
    quant_norm8 = config.Quant(x / 8, y / 8);
    masking = 2.0f * config.Masking(x / 8, y / 8);
  } else if (num_blocks == 2) {
    // Taking max instead of 8th norm seems to work
    // better for smallest blocks up to 16x8. Jyrki couldn't get
    // improvements in trying the same for 16x16 blocks.
    if (acs.covered_blocks_y() == 2) {
      quant_norm8 =
          std::max(config.Quant(x / 8, y / 8), config.Quant(x / 8, y / 8 + 1));
      masking = 2.0f * std::max(config.Masking(x / 8, y / 8),
                                config.Masking(x / 8, y / 8 + 1));
    } else {
      quant_norm8 =
          std::max(config.Quant(x / 8, y / 8), config.Quant(x / 8 + 1, y / 8));
      masking = 2.0f * std::max(config.Masking(x / 8, y / 8),
                                config.Masking(x / 8 + 1, y / 8));
    }
  } else {
    float masking_norm2 = 0;
    float masking_max = 0;
    // Load QF value, calculate empirical heuristic on masking field
    // for weighting the information loss. Information loss manifests
    // itself as ringing, and masking could hide it.
    for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
      for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
        float qval = config.Quant(x / 8 + ix, y / 8 + iy);
        qval *= qval;
        qval *= qval;
        quant_norm8 += qval * qval;
        float maskval = config.Masking(x / 8 + ix, y / 8 + iy);
        masking_max = std::max<float>(masking_max, maskval);
        masking_norm2 += maskval * maskval;
      }
    }
    quant_norm8 /= num_blocks;
    quant_norm8 = FastPowf(quant_norm8, 1.0f / 8.0f);
    masking_norm2 = sqrt(masking_norm2 / num_blocks);
    // This is a highly empirical formula.
    masking = (masking_norm2 + masking_max);
  }
  const auto q = Set(df, quant_norm8);

  // Compute entropy.
  float entropy = config.base_entropy;
  auto info_loss = Zero(df);
  auto info_loss2 = Zero(df);

  for (size_t c = 0; c < 3; c++) {
    const float* inv_matrix = config.dequant->InvMatrix(acs.RawStrategy(), c);
    const auto cmap_factor = Set(df, cmap_factors[c]);

    auto entropy_v = Zero(df);
    auto nzeros_v = Zero(df);
    auto cost1 = Set(df, config.cost1);
    auto cost2 = Set(df, config.cost2);
    auto cost_delta = Set(df, config.cost_delta);
    for (size_t i = 0; i < num_blocks * kDCTBlockSize; i += Lanes(df)) {
      const auto in = Load(df, block + c * size + i);
      const auto in_y = Load(df, block + size + i) * cmap_factor;
      const auto im = Load(df, inv_matrix + i);
      const auto val = (in - in_y) * im * q;
      const auto rval = Round(val);
      const auto diff = AbsDiff(val, rval);
      info_loss += diff;
      info_loss2 += diff * diff;
      const auto q = Abs(rval);
      const auto q_is_zero = q == Zero(df);
      entropy_v += IfThenElseZero(q >= Set(df, 1.5f), cost2);
      // We used to have q * C here, but that cost model seems to
      // be punishing large values more than necessary. Sqrt tries
      // to avoid large values less aggressively. Having high accuracy
      // around zero is most important at low qualities, and there
      // we have directly specified costs for 0, 1, and 2.
      entropy_v += Sqrt(q) * cost_delta;
      nzeros_v += IfThenZeroElse(q_is_zero, Set(df, 1.0f));
    }
    entropy_v += nzeros_v * cost1;

    entropy += GetLane(SumOfLanes(entropy_v));
    size_t num_nzeros = GetLane(SumOfLanes(nzeros_v));
    // Add #bit of num_nonzeros, as an estimate of the cost for encoding the
    // number of non-zeros of the block.
    size_t nbits = CeilLog2Nonzero(num_nzeros + 1) + 1;
    // Also add #bit of #bit of num_nonzeros, to estimate the ANS cost, with a
    // bias.
    entropy += config.zeros_mul * (CeilLog2Nonzero(nbits + 17) + nbits);
  }
  float ret =
      entropy +
      masking *
          ((config.info_loss_multiplier * GetLane(SumOfLanes(info_loss))) +
           (config.info_loss_multiplier2 *
            sqrt(num_blocks * GetLane(SumOfLanes(info_loss2)))));
  return ret;
}

uint8_t FindBest8x8Transform(size_t x, size_t y, int encoding_speed_tier,
                             const ACSConfig& config,
                             const float* JXL_RESTRICT cmap_factors,
                             AcStrategyImage* JXL_RESTRICT ac_strategy,
                             float* block, float* scratch_space,
                             uint32_t* quantized, float* entropy_out) {
  struct TransformTry8x8 {
    AcStrategy::Type type;
    int encoding_speed_tier_max_limit;
    float entropy_add;
    float entropy_mul;
  };
  static const TransformTry8x8 kTransforms8x8[] = {
      {
          AcStrategy::Type::DCT,
          9,
          3.0f,
          0.745f,
      },
      {
          AcStrategy::Type::DCT4X4,
          5,
          4.0f,
          1.0179946967008329f,
      },
      {
          AcStrategy::Type::DCT2X2,
          4,
          4.0f,
          0.76721119707580943f,
      },
      {
          AcStrategy::Type::DCT4X8,
          5,
          0.0f,
          0.700754622182473063f,
      },
      {
          AcStrategy::Type::DCT8X4,
          5,
          0.0f,
          0.700754622182473063f,
      },
      {
          AcStrategy::Type::IDENTITY,
          5,
          8.0f,
          0.81217614513585534f,
      },
      {
          AcStrategy::Type::AFV0,
          4,
          3.0f,
          0.70086131125719425f,
      },
      {
          AcStrategy::Type::AFV1,
          4,
          3.0f,
          0.70086131125719425f,
      },
      {
          AcStrategy::Type::AFV2,
          4,
          3.0f,
          0.70086131125719425f,
      },
      {
          AcStrategy::Type::AFV3,
          4,
          3.0f,
          0.70086131125719425f,
      },
  };
  double best = 1e30;
  uint8_t best_tx = kTransforms8x8[0].type;
  for (auto tx : kTransforms8x8) {
    if (tx.encoding_speed_tier_max_limit < encoding_speed_tier) {
      continue;
    }
    AcStrategy acs = AcStrategy::FromRawStrategy(tx.type);
    float entropy = EstimateEntropy(acs, x, y, config, cmap_factors, block,
                                    scratch_space, quantized);
    entropy = tx.entropy_add + tx.entropy_mul * entropy;
    if (entropy < best) {
      best_tx = tx.type;
      best = entropy;
    }
  }
  *entropy_out = best;
  return best_tx;
}

// bx, by addresses the 64x64 block at 8x8 subresolution
// cx, cy addresses the left, upper 8x8 block position of the candidate
// transform.
void TryMergeAcs(AcStrategy::Type acs_raw, size_t bx, size_t by, size_t cx,
                 size_t cy, const ACSConfig& config,
                 const float* JXL_RESTRICT cmap_factors,
                 AcStrategyImage* JXL_RESTRICT ac_strategy,
                 const float entropy_mul, const uint8_t candidate_priority,
                 uint8_t* priority, float* JXL_RESTRICT entropy_estimate,
                 float* block, float* scratch_space, uint32_t* quantized) {
  AcStrategy acs = AcStrategy::FromRawStrategy(acs_raw);
  float entropy_current = 0;
  for (size_t iy = 0; iy < acs.covered_blocks_y(); ++iy) {
    for (size_t ix = 0; ix < acs.covered_blocks_x(); ++ix) {
      if (priority[(cy + iy) * 8 + (cx + ix)] >= candidate_priority) {
        // Transform would reuse already allocated blocks and
        // lead to invalid overlaps, for example DCT64X32 vs.
        // DCT32X64.
        return;
      }
      entropy_current += entropy_estimate[(cy + iy) * 8 + (cx + ix)];
    }
  }
  float entropy_candidate =
      entropy_mul * EstimateEntropy(acs, (bx + cx) * 8, (by + cy) * 8, config,
                                    cmap_factors, block, scratch_space,
                                    quantized);
  if (entropy_candidate >= entropy_current) return;
  // Accept the candidate.
  for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
    for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
      entropy_estimate[(cy + iy) * 8 + cx + ix] = 0;
      priority[(cy + iy) * 8 + cx + ix] = candidate_priority;
    }
  }
  ac_strategy->Set(bx + cx, by + cy, acs_raw);
  entropy_estimate[cy * 8 + cx] = entropy_candidate;
}

// The following function tries to merge 8x8 transforms into
// 16X8 and 8X16 DCTs fairly, by trying them and their combinations
// with best 8x8 at the same time.
//
// TODO(jyrki):
// This idea could be generalized to larger transforms.
void FindBest16X16(size_t bx, size_t by, size_t cx, size_t cy,
                   const ACSConfig& config,
                   const float* JXL_RESTRICT cmap_factors,
                   AcStrategyImage* JXL_RESTRICT ac_strategy,
                   const float entropy_mul, const float entropy_mul_16X16,
                   float* JXL_RESTRICT entropy_estimate, float* block,
                   float* scratch_space, uint32_t* quantized) {
  constexpr AcStrategy::Type acs_raw16X8 =
      AcStrategy::Type::DCT16X8;  // y=16, x=8
  constexpr AcStrategy::Type acs_raw8X16 =
      AcStrategy::Type::DCT8X16;  // y=8, x=16
  constexpr AcStrategy::Type acs_raw16X16 = AcStrategy::Type::DCT16X16;
  const AcStrategy acs16X8 = AcStrategy::FromRawStrategy(acs_raw16X8);
  const AcStrategy acs8X16 = AcStrategy::FromRawStrategy(acs_raw8X16);
  const AcStrategy acs16X16 = AcStrategy::FromRawStrategy(acs_raw16X16);
  AcStrategyRow row0 = ac_strategy->ConstRow(by + cy + 0);
  AcStrategyRow row1 = ac_strategy->ConstRow(by + cy + 1);
  const bool is8X8[2][2] = {
      {ac_strategy->IsValid(bx + cx + 0, by + cy + 0) &&
           !row0[bx + cx + 0].IsMultiblock(),
       ac_strategy->IsValid(bx + cx + 1, by + cy + 0) &&
           !row0[bx + cx + 1].IsMultiblock()},
      {ac_strategy->IsValid(bx + cx + 0, by + cy + 1) &&
           !row1[bx + cx + 0].IsMultiblock(),
       ac_strategy->IsValid(bx + cx + 1, by + cy + 1) &&
           !row1[bx + cx + 1].IsMultiblock()},
  };
  bool has16X8 = row0[bx + cx + 0].RawStrategy() == acs_raw16X8 ||
                 row0[bx + cx + 1].RawStrategy() == acs_raw16X8;
  bool has8X16 = row0[bx + cx + 0].RawStrategy() == acs_raw8X16 ||
                 row1[bx + cx + 0].RawStrategy() == acs_raw8X16;
  if (has16X8) {
    bool ok0 = (row0[bx + cx + 0].IsFirstBlock() &&
                row0[bx + cx + 0].RawStrategy() == acs_raw16X8) ||
               (is8X8[0][0] && is8X8[1][0]);
    bool ok1 = (row0[bx + cx + 1].IsFirstBlock() &&
                row0[bx + cx + 1].RawStrategy() == acs_raw16X8) ||
               (is8X8[0][1] && is8X8[1][1]);
    if (!ok0 || !ok1) {
      return;
    }
  } else {
    bool ok0 = (row0[bx + cx + 0].IsFirstBlock() &&
                row0[bx + cx + 0].RawStrategy() == acs_raw8X16) ||
               (is8X8[0][0] && is8X8[0][1]);
    bool ok1 = (row1[bx + cx + 0].IsFirstBlock() &&
                row1[bx + cx + 0].RawStrategy() == acs_raw8X16) ||
               (is8X8[1][0] && is8X8[1][1]);
    if (!ok0 || !ok1) {
      return;
    }
  }
  {
    bool has16X16 = row0[bx + cx + 0].RawStrategy() == acs_raw16X16;
    if (has16X16) {
      return;
    }
  }
  // Current entropies from the best 8x8 transforms in this 2x2 area:
  const float entropy00 = entropy_estimate[(cy + 0) * 8 + (cx + 0)];
  const float entropy01 = entropy_estimate[(cy + 0) * 8 + (cx + 1)];
  const float entropy10 = entropy_estimate[(cy + 1) * 8 + (cx + 0)];
  const float entropy11 = entropy_estimate[(cy + 1) * 8 + (cx + 1)];
  float try16X8_0 = std::numeric_limits<float>::max();
  float try16X8_1 = std::numeric_limits<float>::max();
  float try8X16_0 = std::numeric_limits<float>::max();
  float try8X16_1 = std::numeric_limits<float>::max();
  const bool allow_16X8 = !has8X16;
  const bool allow_8X16 = !has16X8;
  if (allow_16X8) {
    if (row0[bx + cx + 0].RawStrategy() != acs_raw16X8) {
      try16X8_0 =
          entropy_mul * EstimateEntropy(acs16X8, (bx + cx + 0) * 8,
                                        (by + cy + 0) * 8, config, cmap_factors,
                                        block, scratch_space, quantized);
    }
    if (row0[bx + cx + 1].RawStrategy() != acs_raw16X8) {
      try16X8_1 =
          entropy_mul * EstimateEntropy(acs16X8, (bx + cx + 1) * 8,
                                        (by + cy + 0) * 8, config, cmap_factors,
                                        block, scratch_space, quantized);
    }
  }
  if (allow_8X16) {
    if (row0[bx + cx].RawStrategy() != acs_raw8X16) {
      try8X16_0 =
          entropy_mul * EstimateEntropy(acs8X16, (bx + cx + 0) * 8,
                                        (by + cy + 0) * 8, config, cmap_factors,
                                        block, scratch_space, quantized);
    }
    if (row1[bx + cx].RawStrategy() != acs_raw8X16) {
      try8X16_1 =
          entropy_mul * EstimateEntropy(acs8X16, (bx + cx + 0) * 8,
                                        (by + cy + 1) * 8, config, cmap_factors,
                                        block, scratch_space, quantized);
    }
  }
  float try16X16 =
      entropy_mul_16X16 *
      EstimateEntropy(acs16X16, (bx + cx + 0) * 8, (by + cy + 0) * 8, config,
                      cmap_factors, block, scratch_space, quantized);

  // Test if this block should have 16X8 or 8X16 transforms,
  // because it can have only one or the other.
  float cost16x8 = std::min(try16X8_0, entropy00 + entropy10) +
                   std::min(try16X8_1, entropy01 + entropy11);
  float cost8x16 = std::min(try8X16_0, entropy00 + entropy01) +
                   std::min(try8X16_1, entropy10 + entropy11);
  if (try16X16 < cost16x8 && try16X16 < cost8x16) {
    ac_strategy->Set(bx + cx, by + cy, acs_raw16X16);
    entropy_estimate[(cy + 0) * 8 + cx + 0] = try16X16;
    entropy_estimate[(cy + 0) * 8 + cx + 1] = 0;
    entropy_estimate[(cy + 1) * 8 + cx + 0] = 0;
    entropy_estimate[(cy + 1) * 8 + cx + 1] = 0;
  } else if (cost16x8 < cost8x16) {
    if (try16X8_0 < entropy00 + entropy10) {
      ac_strategy->Set(bx + cx, by + cy, acs_raw16X8);
      entropy_estimate[(cy + 0) * 8 + cx + 0] = try16X8_0;
      entropy_estimate[(cy + 1) * 8 + cx + 0] = 0;
    }
    if (try16X8_1 < entropy01 + entropy11) {
      ac_strategy->Set(bx + cx + 1, by + cy, acs_raw16X8);
      entropy_estimate[(cy + 0) * 8 + cx + 1] = try16X8_1;
      entropy_estimate[(cy + 1) * 8 + cx + 1] = 0;
    }
  } else {
    if (try8X16_0 < entropy00 + entropy01) {
      ac_strategy->Set(bx + cx, by + cy, acs_raw8X16);
      entropy_estimate[(cy + 0) * 8 + cx + 0] = try8X16_0;
      entropy_estimate[(cy + 0) * 8 + cx + 1] = 0;
    }
    if (try8X16_1 < entropy10 + entropy11) {
      ac_strategy->Set(bx + cx, by + cy + 1, acs_raw8X16);
      entropy_estimate[(cy + 1) * 8 + cx + 0] = try8X16_1;
      entropy_estimate[(cy + 1) * 8 + cx + 1] = 0;
    }
  }
}

void ProcessRectACS(PassesEncoderState* JXL_RESTRICT enc_state,
                    const ACSConfig& config, const Rect& rect) {
  // Main philosophy here:
  // 1. First find best 8x8 transform for each area.
  // 2. Merging them into larger transforms where possibly, but
  // starting from the smallest transforms (16x8 and 8x16).
  // Additional complication: 16x8 and 8x16 are considered
  // simultanouesly and fairly against each other.
  // We are looking at 64x64 squares since the YtoX and YtoB
  // maps happen to be at that resolution, and having
  // integral transforms cross these boundaries leads to
  // additional complications.
  const CompressParams& cparams = enc_state->cparams;
  const float butteraugli_target = cparams.butteraugli_distance;
  AcStrategyImage* ac_strategy = &enc_state->shared.ac_strategy;
  // TODO(veluca): reuse allocations
  auto mem = hwy::AllocateAligned<float>(5 * AcStrategy::kMaxCoeffArea);
  auto qmem = hwy::AllocateAligned<uint32_t>(AcStrategy::kMaxCoeffArea);
  uint32_t* JXL_RESTRICT quantized = qmem.get();
  float* JXL_RESTRICT block = mem.get();
  float* JXL_RESTRICT scratch_space = mem.get() + 3 * AcStrategy::kMaxCoeffArea;
  size_t bx = rect.x0();
  size_t by = rect.y0();
  JXL_ASSERT(rect.xsize() <= 8);
  JXL_ASSERT(rect.ysize() <= 8);
  size_t tx = bx / kColorTileDimInBlocks;
  size_t ty = by / kColorTileDimInBlocks;
  const float cmap_factors[3] = {
      enc_state->shared.cmap.YtoXRatio(
          enc_state->shared.cmap.ytox_map.ConstRow(ty)[tx]),
      0.0f,
      enc_state->shared.cmap.YtoBRatio(
          enc_state->shared.cmap.ytob_map.ConstRow(ty)[tx]),
  };
  if (cparams.speed_tier > SpeedTier::kHare) return;
  // First compute the best 8x8 transform for each square. Later, we do not
  // experiment with different combinations, but only use the best of the 8x8s
  // when DCT8X8 is specified in the tree search.
  // 8x8 transforms have 10 variants, but every larger transform is just a DCT.
  float entropy_estimate[64] = {};
  // Favor all 8x8 transforms (against 16x8 and larger transforms)) at
  // low butteraugli_target distances.
  static const float k8x8mul1 = -0.38173536034815592f;
  static const float k8x8mul2 = 1.0205692427138704f;
  static const float k8x8base = 1.5789348369698299f;
  const float mul8x8 = k8x8mul2 + k8x8mul1 / (butteraugli_target + k8x8base);
  for (size_t iy = 0; iy < rect.ysize(); iy++) {
    for (size_t ix = 0; ix < rect.xsize(); ix++) {
      float entropy = 0.0;
      const uint8_t best_of_8x8s = FindBest8x8Transform(
          8 * (bx + ix), 8 * (by + iy), static_cast<int>(cparams.speed_tier),
          config, cmap_factors, ac_strategy, block, scratch_space, quantized,
          &entropy);
      ac_strategy->Set(bx + ix, by + iy,
                       static_cast<AcStrategy::Type>(best_of_8x8s));
      entropy_estimate[iy * 8 + ix] = entropy * mul8x8;
    }
  }
  // Merge when a larger transform is better than the previously
  // searched best combination of 8x8 transforms.
  struct MergeTry {
    AcStrategy::Type type;
    uint8_t priority;
    uint8_t decoding_speed_tier_max_limit;
    uint8_t encoding_speed_tier_max_limit;
    float entropy_mul;
  };
  static const float k8X16mul1 = -0.51923137374961237;
  static const float k8X16mul2 = 0.9145;
  static const float k8X16base = 1.6637730066379945f;
  const float entropy_mul16X8 =
      k8X16mul2 + k8X16mul1 / (butteraugli_target + k8X16base);
  //  const float entropy_mul16X8 = mul8X16 * 0.91195782912371126f;

  static const float k16X16mul1 = -0.3255063063403677;
  static const float k16X16mul2 = 0.8424362630789904748;
  static const float k16X16base = 2.19008132121404f;
  const float entropy_mul16X16 =
      k16X16mul2 + k16X16mul1 / (butteraugli_target + k16X16base);
  //  const float entropy_mul16X16 = mul16X16 * 0.83183417727960129f;

  // TODO(jyrki): Consider this feedback in further changes:
  // Also effectively when the multipliers for smaller blocks are
  // below 1, this raises the bar for the bigger blocks even higher
  // in that sense these constants are not independent (e.g. changing
  // the constant for DCT16x32 by -5% (making it more likely) also
  // means that DCT32x32 becomes harder to do when starting from
  // two DCT16x32s). It might be better to make them more independent,
  // e.g. by not applying the multiplier when storing the new entropy
  // estimates in TryMergeToACSCandidate().
  const MergeTry kTransformsForMerge[9] = {
      {AcStrategy::Type::DCT16X8, 2, 4, 5, entropy_mul16X8},
      {AcStrategy::Type::DCT8X16, 2, 4, 5, entropy_mul16X8},
      // FindBest16X16 looks for DCT16X16 and its subdivisions.
      // {AcStrategy::Type::DCT16X16, 3, entropy_mul16X16},
      {AcStrategy::Type::DCT16X32, 4, 4, 4, 0.90254513227338527f},
      {AcStrategy::Type::DCT32X16, 4, 4, 4, 0.90254513227338527f},
      {AcStrategy::Type::DCT32X32, 5, 1, 5, 0.9822994906548809f},
      // TODO(jyrki): re-enable 64x32 and 64x64 if/when possible.
      {AcStrategy::Type::DCT64X32, 6, 0, 3, 2.0858810264509633f},
      {AcStrategy::Type::DCT32X64, 6, 0, 3, 2.0858810264509633f},
      {AcStrategy::Type::DCT64X64, 8, 0, 3, 2.0846542128012948f},
  };
  /*
  These sizes not yet included in merge heuristic:
  set(AcStrategy::Type::DCT32X8, 0.0f, 2.261390410971102f);
  set(AcStrategy::Type::DCT8X32, 0.0f, 2.261390410971102f);
  set(AcStrategy::Type::DCT128X128, 0.0f, 1.0f);
  set(AcStrategy::Type::DCT128X64, 0.0f, 0.73f);
  set(AcStrategy::Type::DCT64X128, 0.0f, 0.73f);
  set(AcStrategy::Type::DCT256X256, 0.0f, 1.0f);
  set(AcStrategy::Type::DCT256X128, 0.0f, 0.73f);
  set(AcStrategy::Type::DCT128X256, 0.0f, 0.73f);
  */

  // Priority is a tricky kludge to avoid collisions so that transforms
  // don't overlap.
  uint8_t priority[64] = {};
  for (auto tx : kTransformsForMerge) {
    if (tx.decoding_speed_tier_max_limit < cparams.decoding_speed_tier) {
      continue;
    }
    AcStrategy acs = AcStrategy::FromRawStrategy(tx.type);
    for (size_t cy = 0; cy + acs.covered_blocks_y() - 1 < rect.ysize();
         cy += acs.covered_blocks_y()) {
      for (size_t cx = 0; cx + acs.covered_blocks_x() - 1 < rect.xsize();
           cx += acs.covered_blocks_x()) {
        if (cy + 1 < rect.ysize() && cx + 1 < rect.xsize()) {
          if (tx.type == AcStrategy::Type::DCT8X16) {
            // We handle both DCT8X16 and DCT16X8 at the same time.
            if ((cy | cx) % 2 == 0) {
              FindBest16X16(bx, by, cx, cy, config, cmap_factors, ac_strategy,
                            tx.entropy_mul, entropy_mul16X16, entropy_estimate,
                            block, scratch_space, quantized);
            }
            continue;
          } else if (tx.type == AcStrategy::Type::DCT16X8) {
            // We handled both DCT8X16 and DCT16X8 at the same time,
            // and that is above. The last column and last row,
            // when the last column or last row is odd numbered,
            // are still handled by TryMergeAcs.
            continue;
          }
        }
        if ((tx.type == AcStrategy::Type::DCT8X16 && cy % 2 == 1) ||
            (tx.type == AcStrategy::Type::DCT16X8 && cx % 2 == 1)) {
          // already covered by the 2x2 approach above.
          continue;
        }
        // All other merge sizes are handled here.
        // Some of the DCT16X8s and DCT8X16s will still leak through here
        // when there is an odd number of 8x8 blocks, then the last row
        // and column will get their DCT16X8s and DCT8X16s through the
        // normal integral transform merging process.
        TryMergeAcs(tx.type, bx, by, cx, cy, config, cmap_factors, ac_strategy,
                    tx.entropy_mul, tx.priority, &priority[0], entropy_estimate,
                    block, scratch_space, quantized);
      }
    }
  }
  // Here we still try to do some non-aligned matching, find a few more
  // 16X8, 8X16 and 16X16s between the non-2-aligned blocks.
  if (cparams.speed_tier >= SpeedTier::kHare) {
    return;
  }
  for (int ii = 0; ii < 3; ++ii) {
    for (size_t cy = 1 - (ii == 1); cy + 1 < rect.ysize(); cy += 2) {
      for (size_t cx = 1 - (ii == 2); cx + 1 < rect.xsize(); cx += 2) {
        FindBest16X16(bx, by, cx, cy, config, cmap_factors, ac_strategy,
                      entropy_mul16X8, entropy_mul16X16, entropy_estimate,
                      block, scratch_space, quantized);
      }
    }
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(ProcessRectACS);

void AcStrategyHeuristics::Init(const Image3F& src,
                                PassesEncoderState* enc_state) {
  this->enc_state = enc_state;
  config.dequant = &enc_state->shared.matrices;
  const CompressParams& cparams = enc_state->cparams;
  const float butteraugli_target = cparams.butteraugli_distance;

  // Image row pointers and strides.
  config.quant_field_row = enc_state->initial_quant_field.Row(0);
  config.quant_field_stride = enc_state->initial_quant_field.PixelsPerRow();
  auto& mask = enc_state->initial_quant_masking;
  if (mask.xsize() > 0 && mask.ysize() > 0) {
    config.masking_field_row = mask.Row(0);
    config.masking_field_stride = mask.PixelsPerRow();
  }

  config.src_rows[0] = src.ConstPlaneRow(0, 0);
  config.src_rows[1] = src.ConstPlaneRow(1, 0);
  config.src_rows[2] = src.ConstPlaneRow(2, 0);
  config.src_stride = src.PixelsPerRow();

  // Entropy estimate is composed of two factors:
  //  - estimate of the number of bits that will be used by the block
  //  - information loss due to quantization
  // The following constant controls the relative weights of these components.
  config.info_loss_multiplier = 138.0f;
  config.info_loss_multiplier2 = 50.46839691767866;
  config.base_entropy = 56.030596115736621f;
  config.zeros_mul = 7.444405659772416f;
  // Lots of +1 and -1 coefficients at high quality, it is
  // beneficial to favor them. At low qualities zeros matter more
  // and +1 / -1 coefficients are already quite harmful.
  float slope = std::min<float>(1.0f, butteraugli_target * (1.0f / 3));
  config.cost1 = 1 + slope * 8.8703248061477744f;
  config.cost2 = 4.4417860847109791f;
  config.cost_delta = 6.1101822565620658f;
  JXL_ASSERT(enc_state->shared.ac_strategy.xsize() ==
             enc_state->shared.frame_dim.xsize_blocks);
  JXL_ASSERT(enc_state->shared.ac_strategy.ysize() ==
             enc_state->shared.frame_dim.ysize_blocks);
}

void AcStrategyHeuristics::ProcessRect(const Rect& rect) {
  PROFILER_FUNC;
  const CompressParams& cparams = enc_state->cparams;
  // In Falcon mode, use DCT8 everywhere and uniform quantization.
  if (cparams.speed_tier >= SpeedTier::kCheetah) {
    enc_state->shared.ac_strategy.FillDCT8(rect);
    return;
  }
  HWY_DYNAMIC_DISPATCH(ProcessRectACS)
  (enc_state, config, rect);
}

void AcStrategyHeuristics::Finalize(AuxOut* aux_out) {
  const auto& ac_strategy = enc_state->shared.ac_strategy;
  // Accounting and debug output.
  if (aux_out != nullptr) {
    aux_out->num_dct2_blocks =
        32 * (ac_strategy.CountBlocks(AcStrategy::Type::DCT32X64) +
              ac_strategy.CountBlocks(AcStrategy::Type::DCT64X32));
    aux_out->num_dct4_blocks =
        64 * ac_strategy.CountBlocks(AcStrategy::Type::DCT64X64);
    aux_out->num_dct4x8_blocks =
        ac_strategy.CountBlocks(AcStrategy::Type::DCT4X8) +
        ac_strategy.CountBlocks(AcStrategy::Type::DCT8X4);
    aux_out->num_afv_blocks = ac_strategy.CountBlocks(AcStrategy::Type::AFV0) +
                              ac_strategy.CountBlocks(AcStrategy::Type::AFV1) +
                              ac_strategy.CountBlocks(AcStrategy::Type::AFV2) +
                              ac_strategy.CountBlocks(AcStrategy::Type::AFV3);
    aux_out->num_dct8_blocks = ac_strategy.CountBlocks(AcStrategy::Type::DCT);
    aux_out->num_dct8x16_blocks =
        ac_strategy.CountBlocks(AcStrategy::Type::DCT8X16) +
        ac_strategy.CountBlocks(AcStrategy::Type::DCT16X8);
    aux_out->num_dct8x32_blocks =
        ac_strategy.CountBlocks(AcStrategy::Type::DCT8X32) +
        ac_strategy.CountBlocks(AcStrategy::Type::DCT32X8);
    aux_out->num_dct16_blocks =
        ac_strategy.CountBlocks(AcStrategy::Type::DCT16X16);
    aux_out->num_dct16x32_blocks =
        ac_strategy.CountBlocks(AcStrategy::Type::DCT16X32) +
        ac_strategy.CountBlocks(AcStrategy::Type::DCT32X16);
    aux_out->num_dct32_blocks =
        ac_strategy.CountBlocks(AcStrategy::Type::DCT32X32);
  }

  if (WantDebugOutput(aux_out)) {
    DumpAcStrategy(ac_strategy, enc_state->shared.frame_dim.xsize,
                   enc_state->shared.frame_dim.ysize, "ac_strategy", aux_out);
  }
}

}  // namespace jxl
#endif  // HWY_ONCE
