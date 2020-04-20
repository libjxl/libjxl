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

#ifndef JXL_ENC_ANS_H_
#define JXL_ENC_ANS_H_

// Library to encode the ANS population counts to the bit-stream and encode
// symbols based on the respective distributions.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "c/dec/huffman_table.h"
#include "jxl/ans_common.h"
#include "jxl/ans_params.h"
#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/status.h"
#include "jxl/dec_ans.h"
#include "jxl/enc_bit_writer.h"

namespace jxl {

const static HybridUintConfig kHybridUint420Config{4, 2, 0};

#define USE_MULT_BY_RECIPROCAL

// precision must be equal to:  #bits(state_) + #bits(freq)
#define RECIPROCAL_PRECISION (32 + ANS_LOG_TAB_SIZE)

// Data structure representing one element of the encoding table built
// from a distribution.
// TODO(veluca): split this up, or use an union.
struct ANSEncSymbolInfo {
  // ANS
  uint16_t freq_;
  std::vector<uint16_t> reverse_map_;
#ifdef USE_MULT_BY_RECIPROCAL
  uint64_t ifreq_;
#endif
  // Prefix coding.
  uint8_t depth;
  uint16_t bits;
};

class ANSCoder {
 public:
  ANSCoder() : state_(ANS_SIGNATURE << 16) {}

  uint32_t PutSymbol(const ANSEncSymbolInfo& t, uint8_t* nbits) {
    uint32_t bits = 0;
    *nbits = 0;
    if ((state_ >> (32 - ANS_LOG_TAB_SIZE)) >= t.freq_) {
      bits = state_ & 0xffff;
      state_ >>= 16;
      *nbits = 16;
    }
#ifdef USE_MULT_BY_RECIPROCAL
    // We use mult-by-reciprocal trick, but that requires 64b calc.
    const uint32_t v = (state_ * t.ifreq_) >> RECIPROCAL_PRECISION;
    const uint32_t offset = t.reverse_map_[state_ - v * t.freq_];
    state_ = (v << ANS_LOG_TAB_SIZE) + offset;
#else
    state_ = ((state_ / t.freq_) << ANS_LOG_TAB_SIZE) +
             t.reverse_map_[state_ % t.freq_];
#endif
    return bits;
  }

  uint32_t GetState() const { return state_; }

 private:
  uint32_t state_;
};

// RebalanceHistogram requires a signed type.
using ANSHistBin = int32_t;

struct EntropyEncodingData {
  std::vector<std::vector<ANSEncSymbolInfo>> encoding_info;
  bool use_prefix_code;
};

// Token to be encoded by the ANS. Uses context c (16 bits), writing symbol s
// (8 bits), and adds up to 32 (nb) extra bits (b) that are interleaved in the
// ANS stream.
struct Token {
  Token(uint32_t c, uint32_t s, uint32_t nb, uint32_t b)
      : bits(b), context(c), nbits(nb), symbol(s) {
    JXL_DASSERT(c < (1UL << 16));
    static_assert(sizeof(Token) == 8, "Token must be a 8 byte struct!");
  }
  uint32_t bits;
  uint16_t context;
  uint8_t nbits;
  uint8_t symbol;
};

// Returns an estimate of the number of bits required to encode the given
// histogram (header bits plus data bits).
float ANSPopulationCost(const ANSHistBin* data, size_t alphabet_size);

// Returns an estimate of the number of bits required to encode the given vector
// of tokens.
float TokenCost(const std::vector<Token>& tokens);

// Apply context clustering, compute histograms and encode them. Returns an
// estimate of the total bits used for encoding the stream. If `writer` ==
// nullptr, the bit estimate will not take into account the context map (which
// does not get written if `num_contexts` == 1).
size_t BuildAndEncodeHistograms(const HistogramParams& params,
                                size_t num_contexts,
                                const std::vector<std::vector<Token>>& tokens,
                                EntropyEncodingData* codes,
                                std::vector<uint8_t>* context_map,
                                BitWriter* writer, size_t layer,
                                AuxOut* aux_out);

// Write the tokens to a string.
void WriteTokens(const std::vector<Token>& tokens,
                 const EntropyEncodingData& codes,
                 const std::vector<uint8_t>& context_map, BitWriter* writer,
                 size_t layer, AuxOut* aux_out,
                 HybridUintConfig uint_config = kHybridUint420Config);

// Same as above, but assumes allotment created by caller.
size_t WriteTokens(const std::vector<Token>& tokens,
                   const EntropyEncodingData& codes,
                   const std::vector<uint8_t>& context_map,
                   const BitWriter::Allotment& allotment, BitWriter* writer,
                   HybridUintConfig uint_config = kHybridUint420Config);

}  // namespace jxl

#endif  // JXL_ENC_ANS_H_
