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

#include <vector>

#include "c/dec/huffman_decode.h"
#include "jxl/ans_common.h"
#include "jxl/ans_params.h"
#include "jxl/base/byte_order.h"
#include "jxl/base/cache_aligned.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/dec_bit_reader.h"

namespace jxl {

class ANSSymbolReader;

// Experiments show that best performance is typically achieved for a
// split-exponent of 3 or 4. Trend seems to be that '4' is better
// for large-ish pictures, and '3' better for rather small-ish pictures.
// This is plausible - the more special symbols we have, the better
// statistics we need to get a benefit out of them.

// Our hybrid-encoding scheme has dedicated tokens for the smallest
// (1 << split_exponents) numbers, and for the rest
// encodes (number of bits) + (msb_in_token sub-leading binary digits) +
// (lsb_in_token lowest binary digits) in the token, with the remaining bits
// then being encoded as data.
//
// Example with split_exponent = 4, msb_in_token = 2, lsb_in_token = 0.
//
// Numbers N in [0 .. 15]:
//   These get represented as (token=N, bits='').
// Numbers N >= 16:
//   If n is such that 2**n <= N < 2**(n+1),
//   and m = N - 2**n is the 'mantissa',
//   these get represented as:
// (token=split_token +
//        ((n - split_exponent) * 4) +
//        (m >> (n - msb_in_token)),
//  bits=m & (1 << (n - msb_in_token)) - 1)
// Specifically, we would get:
// N = 0 - 15:          (token=N, nbits=0, bits='')
// N = 16 (10000):      (token=16, nbits=2, bits='00')
// N = 17 (10001):      (token=16, nbits=2, bits='01')
// N = 20 (10100):      (token=17, nbits=2, bits='00')
// N = 24 (11000):      (token=18, nbits=2, bits='00')
// N = 28 (11100):      (token=19, nbits=2, bits='00')
// N = 32 (100000):     (token=20, nbits=3, bits='000')
// N = 65535:           (token=63, nbits=13, bits='1111111111111')
struct HybridUintConfig {
  uint32_t split_exponent;
  uint32_t split_token;
  uint32_t msb_in_token;
  uint32_t lsb_in_token;
  JXL_INLINE void Encode(uint32_t value, uint32_t* JXL_RESTRICT token,
                         uint32_t* JXL_RESTRICT nbits,
                         uint32_t* JXL_RESTRICT bits) const {
    if (value < split_token) {
      *token = value;
      *nbits = 0;
      *bits = 0;
    } else {
      uint32_t n = FloorLog2Nonzero(value);
      uint32_t m = value - (1 << n);
      *token = split_token +
               ((n - split_exponent) << (msb_in_token + lsb_in_token)) +
               ((m >> (n - msb_in_token)) << lsb_in_token) +
               (m & ((1 << lsb_in_token) - 1));
      *nbits = n - msb_in_token - lsb_in_token;
      *bits = (value >> lsb_in_token) & ((1UL << *nbits) - 1);
    }
  }

  // Assumes Refill() has been called.
  template <typename BR>
  JXL_INLINE size_t Read(BR* JXL_RESTRICT br, size_t token) const {
    // Fast-track version of hybrid integer decoding.
    if (token < split_token) return token;
    uint32_t nbits = split_exponent - (msb_in_token + lsb_in_token) +
                     ((token - split_token) >> (msb_in_token + lsb_in_token));
    // Max amount of bits for ReadBits is 32 and max valid left shift is 29
    // bits. However, for speed no error is propagated here, instead limit the
    // nbits size. If nbits > 29, the code stream is invalid, but no error is
    // returned.
    nbits &= 31u;
    uint32_t low = token & ((1 << lsb_in_token) - 1);
    token >>= lsb_in_token;
    const size_t bits = br->PeekBits(nbits);
    br->Consume(nbits);
    size_t ret = (((((1 << msb_in_token) | (token & ((1 << msb_in_token) - 1)))
                    << nbits) |
                   bits)
                  << lsb_in_token) |
                 low;
    return ret;
  }
  JXL_INLINE size_t Read(size_t ctx, BitReader* JXL_RESTRICT br,
                         ANSSymbolReader* JXL_RESTRICT decoder) const;

  explicit HybridUintConfig(uint32_t split_exponent = 4,
                            uint32_t msb_in_token = 2,
                            uint32_t lsb_in_token = 0)
      : split_exponent(split_exponent),
        split_token(1 << split_exponent),
        msb_in_token(msb_in_token),
        lsb_in_token(lsb_in_token) {
    JXL_DASSERT(split_exponent >= msb_in_token + lsb_in_token);
  }
};

struct ANSCode {
  CacheAlignedUniquePtr alias_tables;
  std::vector<brunsli::HuffmanDecodingData> huffman_data;
  std::vector<HybridUintConfig> uint_config;
  bool use_prefix_code;
  uint8_t log_alpha_size;  // for ANS.
};

class ANSSymbolReader {
 public:
  // Invalid symbol reader, to be overwritten.
  ANSSymbolReader() = default;
  ANSSymbolReader(const ANSCode* code, BitReader* JXL_RESTRICT br)
      : alias_tables_(
            reinterpret_cast<AliasTable::Entry*>(code->alias_tables.get())),
        huffman_data_(&code->huffman_data),
        use_prefix_code_(code->use_prefix_code),
        configs(code->uint_config.data()) {
    if (!use_prefix_code_) {
      state_ = static_cast<uint32_t>(br->ReadFixedBits<32>());
      log_alpha_size_ = code->log_alpha_size;
      log_entry_size_ = ANS_LOG_TAB_SIZE - code->log_alpha_size;
      entry_size_minus_1_ = (1 << log_entry_size_) - 1;
    } else {
      state_ = (ANS_SIGNATURE << 16u);
    }
  }

  JXL_INLINE size_t ReadSymbolANSWithoutRefill(const size_t histo_idx,
                                               BitReader* JXL_RESTRICT br) {
    const uint32_t res = state_ & (ANS_TAB_SIZE - 1u);

    const AliasTable::Entry* table =
        &alias_tables_[histo_idx << log_alpha_size_];
    const AliasTable::Symbol symbol =
        AliasTable::Lookup(table, res, log_entry_size_, entry_size_minus_1_);
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
    AliasTable::Prefetch(table, next_res, log_entry_size_);

    return symbol.value;
  }

  JXL_INLINE size_t ReadSymbolHuffWithoutRefill(const size_t histo_idx,
                                                BitReader* JXL_RESTRICT br) {
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

  JXL_INLINE size_t ReadSymbolWithoutRefill(const size_t histo_idx,
                                            BitReader* JXL_RESTRICT br) {
    // TODO(veluca): hoist if in hotter loops.
    if (JXL_UNLIKELY(use_prefix_code_)) {
      return ReadSymbolHuffWithoutRefill(histo_idx, br);
    }
    return ReadSymbolANSWithoutRefill(histo_idx, br);
  }

  JXL_INLINE size_t ReadSymbol(const size_t histo_idx,
                               BitReader* JXL_RESTRICT br) {
    br->Refill();
    return ReadSymbolWithoutRefill(histo_idx, br);
  }

  bool CheckANSFinalState() { return state_ == (ANS_SIGNATURE << 16u); }

  // Takes a *clustered* idx.
  JXL_INLINE size_t ReadHybridUintClustered(size_t ctx,
                                            BitReader* JXL_RESTRICT br) {
    return configs[ctx].Read(ctx, br, this);
  }

  JXL_INLINE size_t ReadHybridUint(size_t ctx, BitReader* JXL_RESTRICT br,
                                   const std::vector<uint8_t>& context_map) {
    return ReadHybridUintClustered(context_map[ctx], br);
  }

  // ctx is a *clustered* context!
  bool IsSingleValue(size_t ctx, uint32_t* value) const {
    // No optimization for Huffman mode.
    if (use_prefix_code_) return false;
    const uint32_t res = state_ & (ANS_TAB_SIZE - 1u);
    const AliasTable::Entry* table = &alias_tables_[ctx << log_alpha_size_];
    AliasTable::Symbol symbol =
        AliasTable::Lookup(table, res, log_entry_size_, entry_size_minus_1_);
    if (symbol.freq != ANS_TAB_SIZE) return false;
    if (configs[ctx].split_token <= symbol.value) return false;
    *value = symbol.value;
    return true;
  }

 private:
  const AliasTable::Entry* JXL_RESTRICT alias_tables_;  // not owned
  const std::vector<brunsli::HuffmanDecodingData>* huffman_data_;
  bool use_prefix_code_;
  uint32_t state_ = ANS_SIGNATURE << 16u;
  const HybridUintConfig* JXL_RESTRICT configs;
  uint32_t log_alpha_size_;
  uint32_t log_entry_size_;
  uint32_t entry_size_minus_1_;
};

JXL_INLINE size_t
HybridUintConfig::Read(size_t ctx, BitReader* JXL_RESTRICT br,
                       ANSSymbolReader* JXL_RESTRICT decoder) const {
  br->Refill();  // covers ReadSymbolWithoutRefill + PeekBits
  size_t token = decoder->ReadSymbolWithoutRefill(ctx, br);
  return Read(br, token);
}

Status DecodeHistograms(BitReader* br, size_t num_contexts, ANSCode* code,
                        std::vector<uint8_t>* context_map);
// Exposed for tests.
Status DecodeUintConfigs(size_t log_alpha_size,
                         std::vector<HybridUintConfig>* uint_config,
                         BitReader* br);

}  // namespace jxl

#endif  // JXL_DEC_ANS_H_
