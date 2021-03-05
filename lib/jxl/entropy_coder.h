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

#ifndef LIB_JXL_ENTROPY_CODER_H_
#define LIB_JXL_ENTROPY_CODER_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_cluster.h"
#include "lib/jxl/enc_context_map.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/image.h"
#include "lib/jxl/quantizer.h"

// Entropy coding and context modeling of DC and AC coefficients, as well as AC
// strategy and quantization field.

namespace jxl {

// Generate DCT NxN quantized AC values tokens.
// Only the subset "rect" [in units of blocks] within all images.
// See also DecodeACVarBlock.
void TokenizeCoefficients(const coeff_order_t* JXL_RESTRICT orders,
                          const Rect& rect,
                          const int32_t* JXL_RESTRICT* JXL_RESTRICT ac_rows,
                          const AcStrategyImage& ac_strategy,
                          YCbCrChromaSubsampling cs,
                          Image3I* JXL_RESTRICT tmp_num_nzeroes,
                          std::vector<Token>* JXL_RESTRICT output,
                          const ImageB& qdc, const ImageI& qf,
                          const BlockCtxMap& block_ctx_map);

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

static constexpr U32Enc kDCThresholdDist(Bits(4), BitsOffset(8, 16),
                                         BitsOffset(16, 272),
                                         BitsOffset(32, 65808));

static constexpr U32Enc kQFThresholdDist(Bits(2), BitsOffset(3, 4),
                                         BitsOffset(5, 12), BitsOffset(8, 44));

Status DecodeBlockCtxMap(BitReader* br, BlockCtxMap* block_ctx_map);
}  // namespace jxl

#endif  // LIB_JXL_ENTROPY_CODER_H_
