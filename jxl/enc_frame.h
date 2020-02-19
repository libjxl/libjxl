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

#ifndef JXL_ENC_FRAME_H_
#define JXL_ENC_FRAME_H_

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_params.h"
#include "jxl/frame_header.h"
#include "jxl/image_bundle.h"
#include "jxl/multiframe.h"

namespace jxl {

// Encodes a single frame (including its header) into a byte stream. A frame is
// either a single image, or animation frame (depending on multiframe),
// and consists of one or more passes. Groups may be processed in parallel by
// `pool`.
Status EncodeFrame(const CompressParams& cparams_orig,
                   const AnimationFrame* animation_frame_or_null,
                   const ImageBundle& ib, PassesEncoderState* passes_enc_state,
                   ThreadPool* pool, BitWriter* writer, AuxOut* aux_out,
                   Multiframe* multiframe);

}  // namespace jxl

#endif  // JXL_ENC_FRAME_H_
