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

#ifndef JXL_ENC_MODULAR_H_
#define JXL_ENC_MODULAR_H_

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/status.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_params.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/modular/encoding/encoding.h"
#include "jxl/modular/image/image.h"

namespace jxl {

// Encodes a single frame into a byte stream using the modular image
// sub-bitstream.
Status EncodeModularRect(const CompressParams& params, const ImageBundle& ib,
                         const Image3F& color, const Rect& rect,
                         BitWriter* writer, AuxOut* aux_out);

class ModularFrameEncoder {
 public:
  ModularFrameEncoder() = default;
  Status ComputeEncodingData(CompressParams orig_cparams,
                             const FrameHeader& frame_header,
                             const ImageBundle& ib, Image3F* JXL_RESTRICT color,
                             PassesEncoderState* JXL_RESTRICT enc_state,
                             ThreadPool* pool, AuxOut* aux_out, bool do_color);
  Status EncodeGlobalInfo(BitWriter* writer, AuxOut* aux_out);
  Status EncodeGroup(BitWriter* writer, AuxOut* aux_out, size_t layer,
                     size_t group_id);

 private:
  Status PrepareGroupParams(const Rect& rect, const CompressParams& cparams,
                            int minShift, int maxShift, size_t group_id,
                            bool do_color);
  std::vector<Image> group_images;
  std::vector<ModularOptions> group_options;

  Tree tree;
  std::vector<std::vector<Token>> tree_tokens;
  std::vector<GroupHeader> group_headers;
  std::vector<std::vector<Token>> tokens;
  EntropyEncodingData code;
  std::vector<uint8_t> context_map;
};

}  // namespace jxl

#endif  // JXL_ENC_MODULAR_H_
