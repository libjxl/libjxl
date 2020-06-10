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
  Status ComputeEncodingData(const CompressParams& orig_cparams,
                             const FrameHeader& frame_header,
                             const ImageBundle& ib, Image3F* JXL_RESTRICT color,
                             PassesEncoderState* JXL_RESTRICT enc_state,
                             bool encode_color);
  Status EncodeGlobalInfo(BitWriter* writer, AuxOut* aux_out, size_t group_id);
  Status EncodeGroup(const Rect& rect, BitWriter* writer, AuxOut* aux_out,
                     size_t minShift, size_t maxShift, size_t layer,
                     size_t group_id);

 private:
  Image full_image;
  CompressParams cparams;
  bool do_color;
  std::atomic<size_t> call{0};
};

}  // namespace jxl

#endif  // JXL_ENC_MODULAR_H_
