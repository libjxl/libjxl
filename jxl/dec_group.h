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

#ifndef JXL_DEC_GROUP_H_
#define JXL_DEC_GROUP_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/status.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/dct_util.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/dec_cache.h"
#include "jxl/dec_params.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/multiframe.h"
#include "jxl/quantizer.h"

namespace jxl {

Status DecodeGroup(BitReader* JXL_RESTRICT* JXL_RESTRICT readers,
                   size_t num_passes, size_t group_idx,
                   PassesDecoderState* JXL_RESTRICT dec_state,
                   GroupDecCache* JXL_RESTRICT group_dec_cache, size_t thread,
                   Image3F* opsin, ImageBundle* JXL_RESTRICT decoded,
                   AuxOut* aux_out);

Status DecodeGroupForRoundtrip(const std::vector<ACImage3>& ac,
                               size_t group_idx,
                               PassesDecoderState* JXL_RESTRICT dec_state,
                               size_t thread, Image3F* JXL_RESTRICT opsin,
                               ImageBundle* JXL_RESTRICT decoded,
                               AuxOut* aux_out, bool save_decompressed,
                               bool apply_color_transform);

}  // namespace jxl

#endif  // JXL_DEC_GROUP_H_
