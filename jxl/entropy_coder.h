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

#ifndef JXL_ENTROPY_CODER_H_
#define JXL_ENTROPY_CODER_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <hwy/static_targets.h>
#include <memory>
#include <utility>
#include <vector>

#include "jxl/ac_strategy.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/fast_log.h"
#include "jxl/base/status.h"
#include "jxl/coeff_order.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/common.h"
#include "jxl/dct_util.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_ans.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/enc_cluster.h"
#include "jxl/enc_context_map.h"
#include "jxl/image.h"
#include "jxl/quantizer.h"

// Entropy coding and context modeling of DC and AC coefficients, as well as AC
// strategy and quantization field.

namespace jxl {

constexpr uint32_t kNumEpfSharpness = 8;

constexpr uint32_t kQuantFieldContexts = 12;
constexpr uint32_t kPerPredictionContexts = 2;
constexpr uint32_t kAcStrategyContexts =
    kPerPredictionContexts * AcStrategy::kNumValidStrategies;
constexpr uint32_t kARParamsContexts =
    (1 + kPerPredictionContexts) * kNumEpfSharpness;

constexpr uint32_t kNumControlFieldContexts =
    kQuantFieldContexts + kAcStrategyContexts + kARParamsContexts;

// Generate AC strategy tokens.
// Only the subset "rect" [in units of blocks] within all images.
// Appends one token per pixel to output.
// See also DecodeAcStrategy.
void TokenizeAcStrategy(const Rect& rect, const AcStrategyImage& ac_strategy,
                        std::vector<Token>* JXL_RESTRICT output,
                        size_t base_context = 0);

// Generate quantization field tokens.
// Only the subset "rect" [in units of blocks] within all images.
// Appends one token per pixel to output.
// TODO(user): quant field seems to be useful for all the AC strategies.
// perhaps, we could just have different quant_ctx based on the block type.
// See also DecodeQuantField.
void TokenizeQuantField(const Rect& rect, const ImageI& quant_field,
                        const AcStrategyImage& ac_strategy,
                        std::vector<Token>* JXL_RESTRICT output,
                        size_t base_context = 0);

// Generate DCT NxN quantized AC values tokens.
// Only the subset "rect" [in units of blocks] within all images.
// See also DecodeACVarBlock.
void TokenizeCoefficients(const coeff_order_t* JXL_RESTRICT orders,
                          const Rect& rect,
                          const ac_qcoeff_t* JXL_RESTRICT* JXL_RESTRICT ac_rows,
                          const AcStrategyImage& ac_strategy,
                          Image3I* JXL_RESTRICT tmp_num_nzeroes,
                          std::vector<Token>* JXL_RESTRICT output);

// Decode AC strategy. The `rect` argument does *not* apply to the hint!
// See also TokenizeAcStrategy.
bool DecodeAcStrategy(BitReader* JXL_RESTRICT br,
                      ANSSymbolReader* JXL_RESTRICT decoder,
                      const std::vector<uint8_t>& context_map, const Rect& rect,
                      AcStrategyImage* JXL_RESTRICT ac_strategy,
                      size_t base_context);

void TokenizeARParameters(const Rect& rect, const ImageB& epf_sharpness,
                          const AcStrategyImage& ac_strategy,
                          std::vector<Token>* JXL_RESTRICT output,
                          size_t base_context = 0);
bool DecodeARParameters(BitReader* br, ANSSymbolReader* decoder,
                        const std::vector<uint8_t>& context_map,
                        const Rect& rect, const AcStrategyImage& ac_strategy,
                        ImageB* epf_sharpness, size_t base_context = 0);

// See TokenizeQuantField.
bool DecodeQuantField(BitReader* JXL_RESTRICT br,
                      ANSSymbolReader* JXL_RESTRICT decoder,
                      const std::vector<uint8_t>& context_map,
                      const Rect& rect_qf,
                      const AcStrategyImage& JXL_RESTRICT ac_strategy,
                      ImageI* JXL_RESTRICT quant_field, size_t base_context);

// Encodes non-negative (X) into (2 * X), negative (-X) into (2 * X - 1)
constexpr uint32_t PackSigned(int32_t value) {
  return (static_cast<uint32_t>(value) << 1) ^
         ((static_cast<uint32_t>(~value) >> 31) - 1);
}

// Reverse to PackSigned, i.e. UnpackSigned(PackSigned(X)) == X.
constexpr intptr_t UnpackSigned(size_t value) {
  return static_cast<intptr_t>((value >> 1) ^ (((~value) & 1) - 1));
}

// Encode non-negative integer as a pair (N, bits), where len(bits) == N.
// 0 is encoded as (0, ''); X from range [2**N - 1, 2 * (2**N - 1)]
// is encoded as (N, X + 1 - 2**N). In detail:
// 0 -> (0, '')
// 1 -> (1, '0')
// 2 -> (1, '1')
// 3 -> (2, '00')
// 4 -> (2, '01')
// 5 -> (2, '10')
// 6 -> (2, '11')
// 7 -> (3, '000')
// ...
// 65535 -> (16, '0000000000000000')
static JXL_INLINE void EncodeVarLenUint(uint32_t value,
                                        uint32_t* JXL_RESTRICT nbits,
                                        uint32_t* JXL_RESTRICT bits) {
  if (value == 0) {
    *nbits = 0;
    *bits = 0;
  } else {
    uint32_t len = FloorLog2Nonzero(value + 1);
    *nbits = len;
    *bits = (value + 1) & ((1 << len) - 1);
  }
}

// Decode variable length non-negative value. Reverse to EncodeVarLenUint.
constexpr uint32_t DecodeVarLenUint(size_t nbits, uint32_t bits) {
  return (1u << nbits) + bits - 1;
}

// Pack signed integer and encode value.
static JXL_INLINE void EncodeVarLenInt(int32_t value,
                                       uint32_t* JXL_RESTRICT nbits,
                                       uint32_t* JXL_RESTRICT bits) {
  EncodeVarLenUint(PackSigned(value), nbits, bits);
}

// Decode value and unpack signed integer.
constexpr int32_t DecodeVarLenInt(uint32_t nbits, uint32_t bits) {
  return UnpackSigned(DecodeVarLenUint(nbits, bits));
}

// Experiments show that best performance is typically achieved for a
// split-exponent of 3 or 4. Trend seems to be that '4' is better
// for large-ish pictures, and '3' better for rather small-ish pictures.
// This is plausible - the more special symbols we have, the better
// statistics we need to get a benefit out of them.
constexpr uint32_t kHybridEncodingDirectSplitExponent = 4;
// constexpr uint32_t kHybridEncodingDirectSplitExponent = 3;
constexpr int kHybridEncodingSplitToken = 1u
                                          << kHybridEncodingDirectSplitExponent;

// Our hybrid-encoding scheme has dedicated tokens for the smallest
// (1 << kHybridEncodingDirectSplitExponent) numbers, and for the rest
// encodes (number of bits) + (2 sub-leading binary digits) in the token,
// with the remaining up to `bits` - 3 bits then being encoded as data.
//
// Numbers N in [0 .. kHybridEncodingSplitToken-1]:
//   These get represented as (token=N, bits='').
// Numbers N >= kHybridEncodingSplitToken:
//   If n is such that 2**n <= N < 2**(n+1),
//   and m = N - 2**n is the 'mantissa',
//   these get represented as:
// (token=kHybridEncodingSplitToken +
//        ((n - kHybridEncodingDirectSplitExponent) * 4) +
//        (m >> (n - 2)),
//  bits=m & (1 << (n - 2)) - 1)
// Specifically, for kHybridEncodingDirectSplitExponent = 4, i.e.
// kHybridEncodingSplitToken=16, we would get:
// N = 0 - 15:          (token=N, nbits=0, bits='')
// N = 16 (10000):      (token=16, nbits=2, bits='00')
// N = 17 (10001):      (token=16, nbits=2, bits='01')
// N = 20 (10100):      (token=17, nbits=2, bits='00')
// N = 24 (11000):      (token=18, nbits=2, bits='00')
// N = 28 (11100):      (token=19, nbits=2, bits='00')
// N = 32 (100000):     (token=20, nbits=3, bits='000')
// N = 65535:           (token=63, nbits=13, bits='1111111111111')
static JXL_INLINE void EncodeHybridVarLenUint(uint32_t value,
                                              uint32_t* JXL_RESTRICT token,
                                              uint32_t* JXL_RESTRICT nbits,
                                              uint32_t* JXL_RESTRICT bits) {
  if (value < kHybridEncodingSplitToken) {
    *token = value;
    *nbits = 0;
    *bits = 0;
  } else {
    uint32_t n = FloorLog2Nonzero(value);
    uint32_t m = value - (1 << n);
    *token = kHybridEncodingSplitToken +
             ((n - kHybridEncodingDirectSplitExponent) << 2) + (m >> (n - 2));
    *nbits = n - 2;
    *bits = value & ((1 << (n - 2)) - 1);
  }
}

static JXL_INLINE void TokenizeHybridUint(
    uint32_t ctx, uint32_t val, std::vector<Token>* JXL_RESTRICT tokens) {
  uint32_t token, nbits, bits;
  EncodeHybridVarLenUint(val, &token, &nbits, &bits);
  JXL_ASSERT(static_cast<size_t>(nbits) <= sizeof(Token::bits) * kBitsPerByte);
  tokens->emplace_back(ctx, token, nbits, bits);
}

static HWY_ATTR JXL_INLINE size_t
ReadHybridUint(size_t ctx, BitReader* JXL_RESTRICT br,
               ANSSymbolReader* JXL_RESTRICT decoder,
               const std::vector<uint8_t>& context_map) {
  br->Refill();  // covers ReadSymbolWithoutRefill + PeekBits
  size_t token = decoder->ReadSymbolWithoutRefill(context_map[ctx], br);
  // Fast-track version of hybrid integer decoding.
  if (token < kHybridEncodingSplitToken) return token;
  uint32_t nbits = kHybridEncodingDirectSplitExponent - 2 +
                   ((token - kHybridEncodingSplitToken) >> 2);
  // Max amount of bits for ReadBits is 32 and max valid left shift is 29 bits.
  // However, for speed no error is propagated here, instead limit the nbits
  // size. If nbits > 29, the code stream is invalid, but no error is returned.
  nbits &= 31u;
  const size_t bits = br->PeekBits(nbits);
  br->Consume(nbits);
  return ((4 | (token & 3)) << nbits) | bits;
}

static JXL_INLINE int32_t PredictFromTopAndLeft(
    const int32_t* const JXL_RESTRICT row_top,
    const int32_t* const JXL_RESTRICT row, size_t x, int32_t default_val) {
  if (x == 0) {
    return row_top == nullptr ? default_val : row_top[x];
  }
  if (row_top == nullptr) {
    return row[x - 1];
  }
  return (row_top[x] + row[x - 1] + 1) / 2;
}

}  // namespace jxl

#endif  // JXL_ENTROPY_CODER_H_
