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

#ifndef JXL_DEC_MODULAR_H_
#define JXL_DEC_MODULAR_H_

#include <stddef.h>

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/dec_params.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/modular/encoding/encoding.h"
#include "jxl/modular/image/image.h"

namespace jxl {

Status DecodeModularRect(const DecompressParams& dparams,
                         size_t responsive_preview, ImageBundle* decoded,
                         const Rect& rect, BitReader* reader, AuxOut* aux_out,
                         size_t bytes_to_read, const FrameHeader& frame_header);

class ModularFrameDecoder {
 public:
  ModularFrameDecoder() {}
  Status DecodeGlobalInfo(BitReader* reader, const FrameHeader& frame_header,
                          ImageBundle* decoded, bool decode_color, size_t xsize,
                          size_t ysize, size_t group_id);
  Status DecodeGroup(const DecompressParams& dparams, const Rect& rect,
                     BitReader* reader, AuxOut* aux_out, size_t minShift,
                     size_t maxShift, size_t group_id);
  Status FinalizeDecoding(Image3F* color, ImageBundle* decoded,
                          jxl::ThreadPool* pool,
                          const FrameHeader& frame_header);
  bool have_dc() { return have_something; };

 private:
  Image full_image;
  bool do_color;
  bool have_something;
  Tree tree;
  ANSCode code;
  std::vector<uint8_t> context_map;
};

}  // namespace jxl

#endif  // JXL_DEC_MODULAR_H_
