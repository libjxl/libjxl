// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_LZ77_H_
#define LIB_JXL_ENC_LZ77_H_

// Library to encode the ANS population counts to the bit-stream and encode
// symbols based on the respective distributions.

#include <cstdint>
#include <vector>

#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_ans_params.h"

namespace jxl {

std::vector<std::vector<Token>> ApplyLZ77(
    const HistogramParams& params, size_t num_contexts,
    const std::vector<std::vector<Token>>& tokens, const LZ77Params& lz77);

}  // namespace jxl

#endif  // LIB_JXL_ENC_LZ77_H_
