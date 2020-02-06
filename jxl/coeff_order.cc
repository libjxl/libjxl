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

#include "jxl/coeff_order.h"

#include <stdint.h>

#include <algorithm>
#include <vector>

#include "jxl/ans_params.h"
#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/base/span.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_ans.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/entropy_coder.h"
#include "jxl/lehmer_code.h"
#include "jxl/modular/encoding/encoding.h"
#include "jxl/modular/image/image.h"

namespace jxl {
namespace {
void SetDefaultOrder(AcStrategy acs, coeff_order_t* JXL_RESTRICT order) {
  PROFILER_FUNC;
  const size_t size =
      kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();
  const coeff_order_t* natural_coeff_order = acs.NaturalCoeffOrder();
  for (size_t k = 0; k < size; ++k) {
    order[k] = natural_coeff_order[k];
  }
}
}  // namespace

uint32_t ComputeUsedOrders(const SpeedTier speed,
                           const AcStrategyImage& ac_strategy,
                           const Rect& rect) {
  // Use default orders for small images.
  if (ac_strategy.xsize() < 5 && ac_strategy.ysize() < 5) return 0;

  // Only uses DCT8 = 0, so bitfield = 1.
  if (speed == SpeedTier::kFalcon) return 1;

  uint32_t ret = 0;
  size_t xsize_blocks = rect.xsize();
  size_t ysize_blocks = rect.ysize();
  // TODO(janwas): parallelize
  for (size_t by = 0; by < ysize_blocks; ++by) {
    AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      int ord = kStrategyOrder[acs_row[bx].RawStrategy()];
      ret |= 1u << ord;
    }
  }
  return ret;
}

void ComputeCoeffOrder(const ACImage3& acs, const AcStrategyImage& ac_strategy,
                       const FrameDimensions& frame_dim, uint32_t used_orders,
                       coeff_order_t* JXL_RESTRICT order) {
  int32_t num_zeros[3 * AcStrategy::kNumValidStrategies *
                    AcStrategy::kMaxCoeffArea] = {};
  // No need to compute number of zero coefficients if all orders are the
  // default.
  if (used_orders != 0) {
    // Count number of zero coefficients, separately for each DCT band.
    for (size_t group_index = 0; group_index < frame_dim.num_groups;
         group_index++) {
      const size_t gx = group_index % frame_dim.xsize_groups;
      const size_t gy = group_index / frame_dim.xsize_groups;
      const Rect rect(gx * kGroupDimInBlocks, gy * kGroupDimInBlocks,
                      kGroupDimInBlocks, kGroupDimInBlocks,
                      frame_dim.xsize_blocks, frame_dim.ysize_blocks);
      const ac_qcoeff_t* JXL_RESTRICT rows[3] = {
          acs.ConstPlaneRow(0, group_index),
          acs.ConstPlaneRow(1, group_index),
          acs.ConstPlaneRow(2, group_index),
      };
      size_t ac_offset = 0;

      for (size_t by = 0; by < rect.ysize(); ++by) {
        AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
        for (size_t bx = 0; bx < rect.xsize(); ++bx) {
          AcStrategy acs = acs_row[bx];
          if (!acs.IsFirstBlock()) continue;
          size_t size = kDCTBlockSize << acs.log2_covered_blocks();
          for (size_t c = 0; c < 3; ++c) {
            const size_t order_offset =
                (kStrategyOrder[acs.RawStrategy()] * 3 + c) *
                AcStrategy::kMaxCoeffArea;
            for (size_t k = 0; k < size; k++) {
              if (rows[c][ac_offset + k] == 0) {
                num_zeros[order_offset + k]++;
              }
            }
            // Ensure LLFs are first in the order.
            size_t cx = acs.covered_blocks_x();
            size_t cy = acs.covered_blocks_y();
            CoefficientLayout(&cy, &cx);
            for (size_t iy = 0; iy < cy; iy++) {
              for (size_t ix = 0; ix < cx; ix++) {
                num_zeros[order_offset + iy * kBlockDim * cx + ix] = -1;
              }
            }
          }
          ac_offset += size;
        }
      }
    }
  }

  uint16_t computed = 0;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    // Ensure natural coefficient order is not permuted if the order is
    // not transmitted.
    if ((1 << ord) & ~used_orders) {
      for (size_t c = 0; c < 3; c++) {
        SetDefaultOrder(AcStrategy::FromRawStrategy(o),
                        &order[(3 * ord + c) * AcStrategy::kMaxCoeffArea]);
      }
      continue;
    }
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    size_t sz = kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();
    const coeff_order_t* natural_coeff_order = acs.NaturalCoeffOrder();

    for (uint8_t c = 0; c < 3; c++) {
      struct PosAndCount {
        uint32_t pos;
        uint32_t count;
      };

      // Apply zig-zag order.
      PosAndCount pos_and_val[AcStrategy::kMaxCoeffArea];
      size_t offset = (ord * 3 + c) * AcStrategy::kMaxCoeffArea;
      for (size_t i = 0; i < sz; ++i) {
        size_t pos = natural_coeff_order[i];
        pos_and_val[i].pos = pos;
        // We don't care for the exact number -> quantize number of zeros,
        // to get less permuted order.
        pos_and_val[i].count = num_zeros[offset + pos] / std::sqrt(sz) + 0.1;
      }

      // Stable-sort -> elements with same number of zeros will preserve their
      // order.
      auto comparator = [](const PosAndCount& a, const PosAndCount& b) -> bool {
        return a.count < b.count;
      };
      std::stable_sort(pos_and_val, pos_and_val + sz, comparator);

      // Grab indices.
      for (size_t i = 0; i < sz; ++i) {
        order[(ord * 3 + c) * AcStrategy::kMaxCoeffArea + i] =
            pos_and_val[i].pos;
      }
    }
  }
}
namespace {
constexpr uint32_t kPermutationContexts = 8;
uint32_t Context(uint32_t val) {
  uint32_t nbits, bits;
  EncodeVarLenUint(val, &nbits, &bits);
  return std::min(nbits, kPermutationContexts - 1);
}

void TokenizePermutation(const coeff_order_t* JXL_RESTRICT order, size_t skip,
                         size_t size, std::vector<Token>* tokens) {
  std::vector<LehmerT> lehmer(size);
  std::vector<uint32_t> temp(size + 1);
  ComputeLehmerCode(order, temp.data(), size, lehmer.data());
  size_t end = size;
  while (end > skip && lehmer[end - 1] == 0) {
    --end;
  }
  uint32_t nbits, bits;
  EncodeVarLenUint(end - skip, &nbits, &bits);
  tokens->emplace_back(Context(size), nbits, nbits, bits);
  uint32_t last = 0;
  for (size_t i = skip; i < end; ++i) {
    EncodeVarLenUint(lehmer[i], &nbits, &bits);
    tokens->emplace_back(Context(last), nbits, nbits, bits);
    last = lehmer[i];
  }
}

HWY_ATTR Status ReadPermutation(size_t skip, size_t size, coeff_order_t* order,
                                BitReader* br, ANSSymbolReader* reader,
                                const std::vector<uint8_t>& context_map) {
  std::vector<LehmerT> lehmer(size);
  // temp space needs to be as large as the next power of 2, so doubling the
  // allocated size is enough.
  std::vector<uint32_t> temp(size * 2);
  uint32_t nbits = reader->ReadSymbol(context_map[Context(size)], br);
  uint32_t bits = br->ReadBits(nbits);
  if (nbits > 31) {
    return JXL_FAILURE("Too many bits");
  }
  uint32_t end = DecodeVarLenUint(nbits, bits) + skip;
  if (end > size) {
    return JXL_FAILURE("Invalid permutation size");
  }
  uint32_t last = 0;
  for (size_t i = skip; i < end; ++i) {
    br->Refill();  // covers ReadSymbolWithoutRefill + ReadBits
    uint32_t nbits =
        reader->ReadSymbolWithoutRefill(context_map[Context(last)], br);
    uint32_t bits = br->ReadBits(nbits);
    if (nbits > 31) {
      return JXL_FAILURE("Too many bits");
    }
    lehmer[i] = DecodeVarLenUint(nbits, bits);
    if (lehmer[i] + i >= size) {
      return JXL_FAILURE("Invalid lehmer code");
    }
    last = lehmer[i];
  }
  DecodeLehmerCode(lehmer.data(), temp.data(), size, order);
  return true;
}

}  // namespace

Status DecodePermutation(size_t skip, size_t size, coeff_order_t* order,
                         BitReader* br) {
  std::vector<uint8_t> context_map;
  ANSCode code;
  JXL_RETURN_IF_ERROR(DecodeHistograms(
      br, kPermutationContexts, ANS_MAX_ALPHA_SIZE, &code, &context_map));
  ANSSymbolReader reader(&code, br);
  JXL_RETURN_IF_ERROR(
      ReadPermutation(skip, size, order, br, &reader, context_map));
  if (!reader.CheckANSFinalState()) {
    return JXL_FAILURE("Invalid ANS stream");
  }
  return true;
}

void EncodePermutation(const coeff_order_t* JXL_RESTRICT order, size_t skip,
                       size_t size, BitWriter* writer, int layer,
                       AuxOut* aux_out) {
  std::vector<Token> tokens;
  TokenizePermutation(order, skip, size, &tokens);
  std::vector<uint8_t> context_map;
  EntropyEncodingData codes;
  BuildAndEncodeHistograms(HistogramParams(), kPermutationContexts, {tokens},
                           &codes, &context_map, writer, layer, aux_out);
  WriteTokens(tokens, codes, context_map, writer, layer, aux_out);
}

namespace {
void EncodeCoeffOrder(const coeff_order_t* JXL_RESTRICT order, AcStrategy acs,
                      std::vector<Token>* tokens) {
  const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
  const size_t size = kDCTBlockSize * llf;
  coeff_order_t order_zigzag[AcStrategy::kMaxCoeffArea];
  const coeff_order_t* natural_coeff_order_lut = acs.NaturalCoeffOrderLut();
  for (size_t i = 0; i < size; ++i) {
    order_zigzag[i] = natural_coeff_order_lut[order[i]];
  }
  TokenizePermutation(order_zigzag, llf, size, tokens);
}
}  // namespace

void EncodeCoeffOrders(uint16_t used_orders, const coeff_order_t* order,
                       BitWriter* writer, size_t layer, AuxOut* aux_out) {
  uint16_t computed = 0;
  std::vector<Token> tokens;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    if ((used_orders & (1 << ord)) == 0) continue;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    for (size_t c = 0; c < 3; c++) {
      EncodeCoeffOrder(&order[(3 * ord + c) * AcStrategy::kMaxCoeffArea], acs,
                       &tokens);
    }
  }
  // Do not write anything if no order is used.
  if (used_orders != 0) {
    std::vector<uint8_t> context_map;
    EntropyEncodingData codes;
    BuildAndEncodeHistograms(HistogramParams(), kPermutationContexts, {tokens},
                             &codes, &context_map, writer, layer, aux_out);
    WriteTokens(tokens, codes, context_map, writer, layer, aux_out);
  }
}

namespace {

Status DecodeCoeffOrder(AcStrategy acs, coeff_order_t* order, BitReader* br,
                        ANSSymbolReader* reader,
                        const std::vector<uint8_t>& context_map) {
  PROFILER_FUNC;
  const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
  const size_t size = kDCTBlockSize * llf;

  JXL_RETURN_IF_ERROR(
      ReadPermutation(llf, size, order, br, reader, context_map));
  const coeff_order_t* natural_coeff_order = acs.NaturalCoeffOrder();
  for (size_t k = 0; k < size; ++k) {
    order[k] = natural_coeff_order[order[k]];
  }
  return true;
}

}  // namespace

Status DecodeCoeffOrders(uint16_t used_orders, coeff_order_t* order,
                         BitReader* br) {
  uint16_t computed = 0;
  std::vector<uint8_t> context_map;
  ANSCode code;
  std::unique_ptr<ANSSymbolReader> reader;
  // Bitstream does not have histograms if no coefficient order is used.
  if (used_orders != 0) {
    JXL_RETURN_IF_ERROR(DecodeHistograms(
        br, kPermutationContexts, ANS_MAX_ALPHA_SIZE, &code, &context_map));
    reader = make_unique<ANSSymbolReader>(&code, br);
  }
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    if ((used_orders & (1 << ord)) == 0) {
      for (size_t c = 0; c < 3; c++) {
        SetDefaultOrder(acs, &order[(3 * ord + c) * AcStrategy::kMaxCoeffArea]);
      }
    } else {
      for (size_t c = 0; c < 3; c++) {
        JXL_RETURN_IF_ERROR(DecodeCoeffOrder(
            acs, &order[(3 * ord + c) * AcStrategy::kMaxCoeffArea], br,
            reader.get(), context_map));
      }
    }
  }
  if (used_orders && !reader->CheckANSFinalState()) {
    return JXL_FAILURE("Invalid ANS stream");
  }
  return true;
}

}  // namespace jxl
