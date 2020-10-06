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

#ifndef JXL_COEFF_ORDER_H_
#define JXL_COEFF_ORDER_H_

#include <stddef.h>
#include <stdint.h>

#include "jxl/ac_strategy.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/status.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/common.h"
#include "jxl/dct_util.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/enc_params.h"

namespace jxl {

// Those offsets get multiplied by kDCTBlockSize.
static constexpr size_t kCoeffOrderOffset[3 * kNumOrders + 1] = {
    0,  1,  2,  3,  4,  5,  6,   10,  14,  18,  34,  50,  66,  68,
    70, 72, 76, 80, 84, 92, 100, 108, 172, 236, 300, 332, 364, 396,
};

static constexpr size_t CoeffOrderOffset(size_t order, size_t c) {
  return kCoeffOrderOffset[3 * order + c] * kDCTBlockSize;
}

static constexpr size_t kCoeffOrderSize =
    kCoeffOrderOffset[3 * kNumOrders] * kDCTBlockSize;

// Mapping from AC strategy to order bucket. Strategies with different natural
// orders must have different buckets.
constexpr uint8_t kStrategyOrder[AcStrategy::kNumValidStrategies] = {
    0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 6, 6, 1, 1, 1, 1, 1, 1, 7, 8, 8};

static_assert(AcStrategy::kNumValidStrategies == 21,
              "Update this array when adding or removing AC strategies.");

// Orders that are actually used in part of image. `rect` is in block units.
uint32_t ComputeUsedOrders(SpeedTier speed, const AcStrategyImage& ac_strategy,
                           const Rect& rect);

// Modify zig-zag order, so that DCT bands with more zeros go later.
// Order of DCT bands with same number of zeros is untouched, so
// permutation will be cheaper to encode.
void ComputeCoeffOrder(SpeedTier speed, const ACImage3& acs,
                       const AcStrategyImage& ac_strategy,
                       const FrameDimensions& frame_dim, uint32_t used_orders,
                       coeff_order_t* JXL_RESTRICT order);

void EncodeCoeffOrders(uint16_t used_orders,
                       const coeff_order_t* JXL_RESTRICT order,
                       BitWriter* writer, size_t layer,
                       AuxOut* JXL_RESTRICT aux_out);

Status DecodeCoeffOrders(uint16_t used_orders, coeff_order_t* order,
                         BitReader* br);

// Encoding/decoding of a single permutation. `size`: number of elements in the
// permutation. `skip`: number of elements to skip from the *beginning* of the
// permutation.
void EncodePermutation(const coeff_order_t* JXL_RESTRICT order, size_t skip,
                       size_t size, BitWriter* writer, int layer,
                       AuxOut* aux_out);

Status DecodePermutation(size_t skip, size_t size, coeff_order_t* order,
                         BitReader* br);

}  // namespace jxl

#endif  // JXL_COEFF_ORDER_H_
