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

#ifndef JXL_ENC_AC_STRATEGY_H_
#define JXL_ENC_AC_STRATEGY_H_

#include "jxl/ac_strategy.h"
#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/common.h"
#include "jxl/dec_ans.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_params.h"
#include "jxl/image.h"
#include "jxl/quant_weights.h"

// `FindBestAcStrategy` uses heuristics to choose which AC strategy should be
// used in each block, as well as the initial quantization field.

namespace jxl {

// `quant_field` will be the initial quantization field for this image.  `src`
// is the input image in the XYB color space. `ac_strategy` is the output
// strategy.
void FindBestAcStrategy(const Image3F& src,
                        PassesEncoderState* JXL_RESTRICT enc_state,
                        ThreadPool* pool, AuxOut* aux_out);

}  // namespace jxl

#endif  // JXL_ENC_AC_STRATEGY_H_
