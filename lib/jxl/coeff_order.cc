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

#include "lib/jxl/coeff_order.h"

#include <stdint.h>

#include <algorithm>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/lehmer_code.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/modular_image.h"

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
  // TODO(veluca): precompute when doing DCT.
  for (size_t by = 0; by < ysize_blocks; ++by) {
    AcStrategyRow acs_row = ac_strategy.ConstRow(rect, by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      int ord = kStrategyOrder[acs_row[bx].RawStrategy()];
      // Do not customize coefficient orders for blocks bigger than 32x32.
      if (ord > 6) {
        continue;
      }
      ret |= 1u << ord;
    }
  }
  return ret;
}

void ComputeCoeffOrder(SpeedTier speed, const ACImage3& acs,
                       const AcStrategyImage& ac_strategy,
                       const FrameDimensions& frame_dim, uint32_t used_orders,
                       coeff_order_t* JXL_RESTRICT order) {
  std::vector<int32_t> num_zeros(kCoeffOrderSize);
  // If compressing at high speed and only using 8x8 DCTs, only consider a
  // subset of blocks.
  double block_fraction = 1.0f;
  // TODO(veluca): figure out why sampling blocks if non-8x8s are used makes
  // encoding significantly less dense.
  if (speed >= SpeedTier::kSquirrel && used_orders == 1) {
    block_fraction = 0.5f;
  }
  // No need to compute number of zero coefficients if all orders are the
  // default.
  if (used_orders != 0) {
    uint64_t threshold =
        (std::numeric_limits<uint64_t>::max() >> 32) * block_fraction;
    uint64_t s[2] = {0x94D049BB133111EBull, 0xBF58476D1CE4E5B9ull};
    // Xorshift128+ adapted from xorshift128+-inl.h
    auto use_sample = [&]() {
      auto s1 = s[0];
      const auto s0 = s[1];
      const auto bits = s1 + s0;  // b, c
      s[0] = s0;
      s1 ^= s1 << 23;
      s1 ^= s0 ^ (s1 >> 18) ^ (s0 >> 5);
      s[1] = s1;
      return (bits >> 32) <= threshold;
    };

    // Count number of zero coefficients, separately for each DCT band.
    // TODO(veluca): precompute when doing DCT.
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
          if (!use_sample()) continue;
          size_t size = kDCTBlockSize << acs.log2_covered_blocks();
          for (size_t c = 0; c < 3; ++c) {
            const size_t order_offset =
                CoeffOrderOffset(kStrategyOrder[acs.RawStrategy()], c);
            for (size_t k = 0; k < size; k++) {
              bool is_zero = rows[c][ac_offset + k] == 0;
              num_zeros[order_offset + k] += is_zero ? 1 : 0;
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
  struct PosAndCount {
    uint32_t pos;
    uint32_t count;
  };
  auto mem = hwy::AllocateAligned<PosAndCount>(AcStrategy::kMaxCoeffArea);

  uint16_t computed = 0;
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    size_t sz = kDCTBlockSize * acs.covered_blocks_x() * acs.covered_blocks_y();
    // Ensure natural coefficient order is not permuted if the order is
    // not transmitted.
    if ((1 << ord) & ~used_orders) {
      for (size_t c = 0; c < 3; c++) {
        size_t offset = CoeffOrderOffset(ord, c);
        JXL_DASSERT(CoeffOrderOffset(ord, c + 1) - offset == sz);
        SetDefaultOrder(AcStrategy::FromRawStrategy(o), &order[offset]);
      }
      continue;
    }
    const coeff_order_t* natural_coeff_order = acs.NaturalCoeffOrder();

    for (uint8_t c = 0; c < 3; c++) {
      // Apply zig-zag order.
      PosAndCount* pos_and_val = mem.get();
      size_t offset = CoeffOrderOffset(ord, c);
      JXL_DASSERT(CoeffOrderOffset(ord, c + 1) - offset == sz);
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
        order[offset + i] = pos_and_val[i].pos;
      }
    }
  }
}

namespace {
constexpr uint32_t kPermutationContexts = 8;
uint32_t Context(uint32_t val) {
  uint32_t token, nbits, bits;
  HybridUintConfig(0, 0, 0).Encode(val, &token, &nbits, &bits);
  return std::min(token, kPermutationContexts - 1);
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
  tokens->emplace_back(Context(size), end - skip);
  uint32_t last = 0;
  for (size_t i = skip; i < end; ++i) {
    tokens->emplace_back(Context(last), lehmer[i]);
    last = lehmer[i];
  }
}

Status ReadPermutation(size_t skip, size_t size, coeff_order_t* order,
                       BitReader* br, ANSSymbolReader* reader,
                       const std::vector<uint8_t>& context_map) {
  std::vector<LehmerT> lehmer(size);
  // temp space needs to be as large as the next power of 2, so doubling the
  // allocated size is enough.
  std::vector<uint32_t> temp(size * 2);
  uint32_t end = reader->ReadHybridUint(Context(size), br, context_map) + skip;
  if (end > size) {
    return JXL_FAILURE("Invalid permutation size");
  }
  uint32_t last = 0;
  for (size_t i = skip; i < end; ++i) {
    lehmer[i] = reader->ReadHybridUint(Context(last), br, context_map);
    last = lehmer[i];
    if (lehmer[i] + i >= size) {
      return JXL_FAILURE("Invalid lehmer code");
    }
  }
  DecodeLehmerCode(lehmer.data(), temp.data(), size, order);
  return true;
}

}  // namespace

Status DecodePermutation(size_t skip, size_t size, coeff_order_t* order,
                         BitReader* br) {
  std::vector<uint8_t> context_map;
  ANSCode code;
  JXL_RETURN_IF_ERROR(
      DecodeHistograms(br, kPermutationContexts, &code, &context_map));
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
  std::vector<std::vector<Token>> tokens(1);
  TokenizePermutation(order, skip, size, &tokens[0]);
  std::vector<uint8_t> context_map;
  EntropyEncodingData codes;
  BuildAndEncodeHistograms(HistogramParams(), kPermutationContexts, tokens,
                           &codes, &context_map, writer, layer, aux_out);
  WriteTokens(tokens[0], codes, context_map, writer, layer, aux_out);
}

namespace {
void EncodeCoeffOrder(const coeff_order_t* JXL_RESTRICT order, AcStrategy acs,
                      std::vector<Token>* tokens, coeff_order_t* order_zigzag) {
  const size_t llf = acs.covered_blocks_x() * acs.covered_blocks_y();
  const size_t size = kDCTBlockSize * llf;
  const coeff_order_t* natural_coeff_order_lut = acs.NaturalCoeffOrderLut();
  for (size_t i = 0; i < size; ++i) {
    order_zigzag[i] = natural_coeff_order_lut[order[i]];
  }
  TokenizePermutation(order_zigzag, llf, size, tokens);
}
}  // namespace

void EncodeCoeffOrders(uint16_t used_orders, const coeff_order_t* order,
                       BitWriter* writer, size_t layer, AuxOut* aux_out) {
  auto mem = hwy::AllocateAligned<coeff_order_t>(AcStrategy::kMaxCoeffArea);
  uint16_t computed = 0;
  std::vector<std::vector<Token>> tokens(1);
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    if ((used_orders & (1 << ord)) == 0) continue;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    for (size_t c = 0; c < 3; c++) {
      EncodeCoeffOrder(&order[CoeffOrderOffset(ord, c)], acs, &tokens[0],
                       mem.get());
    }
  }
  // Do not write anything if no order is used.
  if (used_orders != 0) {
    std::vector<uint8_t> context_map;
    EntropyEncodingData codes;
    BuildAndEncodeHistograms(HistogramParams(), kPermutationContexts, tokens,
                             &codes, &context_map, writer, layer, aux_out);
    WriteTokens(tokens[0], codes, context_map, writer, layer, aux_out);
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
    JXL_RETURN_IF_ERROR(
        DecodeHistograms(br, kPermutationContexts, &code, &context_map));
    reader = make_unique<ANSSymbolReader>(&code, br);
  }
  for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
    uint8_t ord = kStrategyOrder[o];
    if (computed & (1 << ord)) continue;
    computed |= 1 << ord;
    AcStrategy acs = AcStrategy::FromRawStrategy(o);
    if ((used_orders & (1 << ord)) == 0) {
      for (size_t c = 0; c < 3; c++) {
        SetDefaultOrder(acs, &order[CoeffOrderOffset(ord, c)]);
      }
    } else {
      for (size_t c = 0; c < 3; c++) {
        JXL_RETURN_IF_ERROR(DecodeCoeffOrder(acs,
                                             &order[CoeffOrderOffset(ord, c)],
                                             br, reader.get(), context_map));
      }
    }
  }
  if (used_orders && !reader->CheckANSFinalState()) {
    return JXL_FAILURE("Invalid ANS stream");
  }
  return true;
}

}  // namespace jxl
