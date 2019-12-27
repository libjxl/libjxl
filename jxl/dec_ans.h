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

#ifndef JXL_DEC_ANS_H_
#define JXL_DEC_ANS_H_

// Library to decode the ANS population counts from the bit-stream and build a
// decoding table from them.

#include <stddef.h>
#include <stdint.h>

#include <hwy/static_targets.h>
#include <vector>

#include "c/dec/huffman_decode.h"
#include "jxl/ans_common.h"
#include "jxl/ans_params.h"
#include "jxl/base/byte_order.h"
#include "jxl/base/cache_aligned.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/dec_bit_reader.h"

namespace jxl {

struct ANSCode {
  CacheAlignedUniquePtr alias_tables;
  std::vector<brunsli::HuffmanDecodingData> huffman_data;
  bool use_prefix_code;
};

class ANSSymbolReader {
 public:
  // Invalid symbol reader, to be overwritten.
  ANSSymbolReader() = default;
  HWY_ATTR ANSSymbolReader(const ANSCode* code, BitReader* JXL_RESTRICT br)
      : alias_tables_(
            reinterpret_cast<AliasTable::Entry*>(code->alias_tables.get())),
        huffman_data_(&code->huffman_data),
        use_prefix_code_(code->use_prefix_code) {
    if (!use_prefix_code_) {
      state_ = static_cast<uint32_t>(br->ReadFixedBits<32>());
    } else {
      state_ = (ANS_SIGNATURE << 16u);
    }
  }

  HWY_ATTR JXL_INLINE size_t ReadSymbolANSWithoutRefill(
      const size_t histo_idx, BitReader* JXL_RESTRICT br) {
    const uint32_t res = state_ & (ANS_TAB_SIZE - 1u);

    const AliasTable::Entry* table =
        &alias_tables_[histo_idx * ANS_MAX_ALPHA_SIZE];
    const AliasTable::Symbol symbol = AliasTable::Lookup(table, res);
    state_ = symbol.freq * (state_ >> ANS_LOG_TAB_SIZE) + symbol.offset;

#if 1
    // Branchless version is about equally fast on SKX.
    const uint32_t new_state =
        (state_ << 16u) | static_cast<uint32_t>(br->PeekFixedBits<16>());
    const bool normalize = state_ < (1u << 16u);
    state_ = normalize ? new_state : state_;
    br->Consume(normalize ? 16 : 0);
#else
    if (JXL_UNLIKELY(state_ < (1u << 16u))) {
      state_ = (state_ << 16u) | br->PeekFixedBits<16>();
      br->Consume(16);
    }
#endif
    const uint32_t next_res = state_ & (ANS_TAB_SIZE - 1u);
    AliasTable::Prefetch(table, next_res);

    return symbol.value;
  }

  HWY_ATTR JXL_INLINE size_t ReadSymbolHuffWithoutRefill(
      const size_t histo_idx, BitReader* JXL_RESTRICT br) {
    // Adapted from brunsli.
    const brunsli::HuffmanCode* table = &(*huffman_data_)[histo_idx].table_[0];
    table += br->PeekFixedBits<8>();
    size_t nbits = table->bits;
    if (nbits > 8) {
      nbits -= 8;
      br->Consume(8);
      table += table->value;
      table += br->PeekBits(nbits);
    }
    br->Consume(table->bits);
    return table->value;
  }

  HWY_ATTR JXL_INLINE size_t
  ReadSymbolWithoutRefill(const size_t histo_idx, BitReader* JXL_RESTRICT br) {
    // TODO(veluca): hoist if in hotter loops.
    if (JXL_UNLIKELY(use_prefix_code_)) {
      return ReadSymbolHuffWithoutRefill(histo_idx, br);
    }
    return ReadSymbolANSWithoutRefill(histo_idx, br);
  }

  HWY_ATTR JXL_INLINE size_t ReadSymbol(const size_t histo_idx,
                                        BitReader* JXL_RESTRICT br) {
    br->Refill();
    return ReadSymbolWithoutRefill(histo_idx, br);
  }

  bool CheckANSFinalState() { return state_ == (ANS_SIGNATURE << 16u); }

 private:
  const AliasTable::Entry* JXL_RESTRICT alias_tables_;  // not owned
  const std::vector<brunsli::HuffmanDecodingData>* huffman_data_;
  bool use_prefix_code_;
  uint32_t state_ = ANS_SIGNATURE << 16u;
};

bool DecodeHistograms(BitReader* br, size_t num_contexts,
                      size_t max_alphabet_size, ANSCode* code,
                      std::vector<uint8_t>* context_map);

}  // namespace jxl

#endif  // JXL_DEC_ANS_H_
