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
                        size_t base_context);

// Generate quantization field tokens.
// Only the subset "rect" [in units of blocks] within all images.
// Appends one token per pixel to output.
// TODO(user): quant field seems to be useful for all the AC strategies.
// perhaps, we could just have different quant_ctx based on the block type.
// See also DecodeQuantField.
void TokenizeQuantField(const Rect& rect, const ImageI& quant_field,
                        const AcStrategyImage& ac_strategy,
                        std::vector<Token>* JXL_RESTRICT output,
                        size_t base_context);

// Generate DCT NxN quantized AC values tokens.
// Only the subset "rect" [in units of blocks] within all images.
// See also DecodeACVarBlock.
void TokenizeCoefficients(const coeff_order_t* JXL_RESTRICT orders,
                          const Rect& rect,
                          const ac_qcoeff_t* JXL_RESTRICT* JXL_RESTRICT ac_rows,
                          const AcStrategyImage& ac_strategy,
                          Image3I* JXL_RESTRICT tmp_num_nzeroes,
                          std::vector<Token>* JXL_RESTRICT output);

void TokenizeARParameters(const Rect& rect, const ImageB& epf_sharpness,
                          const AcStrategyImage& ac_strategy,
                          std::vector<Token>* JXL_RESTRICT output,
                          size_t base_context);

// Decode AC strategy. The `rect` argument does *not* apply to the hint!
// See also TokenizeAcStrategy.
Status DecodeAcStrategy(BitReader* JXL_RESTRICT br,
                        ANSSymbolReader* JXL_RESTRICT decoder,
                        const std::vector<uint8_t>& context_map,
                        const Rect& rect,
                        AcStrategyImage* JXL_RESTRICT ac_strategy,
                        size_t base_context);

Status DecodeARParameters(BitReader* br, ANSSymbolReader* decoder,
                          const std::vector<uint8_t>& context_map,
                          const Rect& rect, const AcStrategyImage& ac_strategy,
                          ImageB* epf_sharpness, size_t base_context);

// See TokenizeQuantField.
Status DecodeQuantField(BitReader* JXL_RESTRICT br,
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
