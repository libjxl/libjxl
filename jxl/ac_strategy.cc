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
#include <numeric>  // iota
#include <type_traits>
#include <utility>

#include "jxl/base/profiler.h"
#include "jxl/common.h"
#include "jxl/image_ops.h"

namespace jxl {
namespace {

// Tries to generalize zig-zag order to non-square blocks. Surprisingly, in
// square block frequency along the (i + j == const) diagonals is roughly the
// same. For historical reasons, consecutive diagonals are traversed
// in alternating directions - so called "zig-zag" (or "snake") order.
AcStrategy::CoeffOrderAndLut ComputeNaturalCoeffOrder() {
  AcStrategy::CoeffOrderAndLut coeff;
  for (size_t s = 0; s < AcStrategy::kNumValidStrategies; s++) {
    const AcStrategy acs = AcStrategy::FromRawStrategy(s);
    size_t cx = acs.covered_blocks_x();
    size_t cy = acs.covered_blocks_y();
    CoefficientLayout(&cy, &cx);
    const size_t num_coeffs = kDCTBlockSize * cx * cy;
    JXL_ASSERT((AcStrategy::CoeffOrderAndLut::kOffset[s + 1] -
                AcStrategy::CoeffOrderAndLut::kOffset[s]) *
                   kDCTBlockSize ==
               num_coeffs);
    coeff_order_t* JXL_RESTRICT order_start =
        coeff.order + AcStrategy::CoeffOrderAndLut::kOffset[s] * kDCTBlockSize;
    coeff_order_t* JXL_RESTRICT lut_start =
        coeff.lut + AcStrategy::CoeffOrderAndLut::kOffset[s] * kDCTBlockSize;
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
}

}  // namespace

const AcStrategy::CoeffOrderAndLut* AcStrategy::CoeffOrder() {
  static AcStrategy::CoeffOrderAndLut order = ComputeNaturalCoeffOrder();
  return &order;
}

// These definitions are needed before C++17.
constexpr size_t AcStrategy::kMaxCoeffBlocks;
constexpr size_t AcStrategy::kMaxBlockDim;
constexpr size_t AcStrategy::kMaxCoeffArea;
constexpr size_t AcStrategy::CoeffOrderAndLut::kOffset[];

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
