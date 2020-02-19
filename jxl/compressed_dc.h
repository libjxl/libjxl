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

#ifndef JXL_COMPRESSED_DC_H_
#define JXL_COMPRESSED_DC_H_

#include <stddef.h>
#include <stdint.h>

#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/dec_cache.h"
#include "jxl/enc_ans.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/enc_cache.h"
#include "jxl/image.h"

// DC handling functions: encoding and decoding of DC to and from bitstream, and
// related function to initialize the per-group decoder cache.

namespace jxl {

void TokenizeDC(size_t group_index, const Image3F& dc,
                PassesEncoderState* JXL_RESTRICT enc_state, AuxOut* aux_out);

// Encodes the DC-related information from enc_state: quantized dc itself
// and gradient map.
Status EncodeDCGroup(const PassesEncoderState& enc_state, size_t group_idx,
                     BitWriter* writer, AuxOut* aux_out);

// Decodes and dequantizes DC.
Status DecodeDCGroup(BitReader* reader, size_t group_idx,
                     PassesDecoderState* dec_state, AuxOut* aux_out);

// Fill border of DC image.
void InitializeDCBorder(Image3F* JXL_RESTRICT dc);

// Smooth DC in already-smooth areas, to counteract banding.
void AdaptiveDCSmoothing(const Image3F& dc_quant_field, Image3F* dc,
                         ThreadPool* pool);

}  // namespace jxl

#endif  // JXL_COMPRESSED_DC_H_
