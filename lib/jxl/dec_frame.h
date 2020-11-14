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

#ifndef LIB_JXL_DEC_FRAME_H_
#define LIB_JXL_DEC_FRAME_H_

#include <stdint.h>

#include "lib/jxl/aux_out.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_params.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {

// `frame_header` must have nonserialized_metadata and
// nonserialized_is_preview set.
Status DecodeFrameHeader(BitReader* JXL_RESTRICT reader,
                         FrameHeader* JXL_RESTRICT frame_header);

// Decodes a frame. Groups may be processed in parallel by `pool`.
// See DecodeFile for explanation of c_decoded.
// `io` is only used for reading maximum image size. Also updates
// `dec_state` with the new frame header.
// `metadata` is the metadata that applies to all frames of the codestream
// `decoded->metadata` must already be set and must match metadata.m.
Status DecodeFrame(const DecompressParams& dparams,
                   PassesDecoderState* dec_state, ThreadPool* JXL_RESTRICT pool,
                   BitReader* JXL_RESTRICT reader, AuxOut* JXL_RESTRICT aux_out,
                   ImageBundle* decoded, const CodecMetadata& metadata,
                   const CodecInOut* io = nullptr, bool is_preview = false);

// Leaves reader in the same state as DecodeFrame would. Used to skip preview.
// Also updates `dec_state` with the new frame header.
Status SkipFrame(const CodecMetadata& metadata, BitReader* JXL_RESTRICT reader,
                 bool is_preview = false);

// Decodes the global DC info from a frame section, exposed for use by API.
Status DecodeGlobalDCInfo(size_t downsampling, BitReader* reader,
                          ImageBundle* decoded, PassesDecoderState* state,
                          ThreadPool* pool);

// Decodes the DC image, exposed for use by API.
// aux_outs may be nullptr if aux_out is nullptr.
Status DecodeDC(const FrameHeader& frame_header, PassesDecoderState* dec_state,
                ModularFrameDecoder& modular_frame_decoder,
                size_t group_codes_begin,
                const std::vector<uint64_t>& group_offsets,
                const std::vector<uint32_t>& group_sizes,
                ThreadPool* JXL_RESTRICT pool, BitReader* JXL_RESTRICT reader,
                std::vector<AuxOut>* aux_outs, AuxOut* JXL_RESTRICT aux_out);

}  // namespace jxl

#endif  // LIB_JXL_DEC_FRAME_H_
