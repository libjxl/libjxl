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

#include "jxl/frame_header.h"

#include "jxl/aux_out.h"
#include "jxl/fields.h"

namespace jxl {

AnimationFrame::AnimationFrame() { Bundle::Init(this); }
Passes::Passes() { Bundle::Init(this); }
FrameHeader::FrameHeader() { Bundle::Init(this); }

Status ReadFrameHeader(BitReader* JXL_RESTRICT reader,
                       FrameHeader* JXL_RESTRICT frame) {
  return Bundle::Read(reader, frame);
}

Status WriteFrameHeader(const FrameHeader& frame,
                        BitWriter* JXL_RESTRICT writer, AuxOut* aux_out) {
  return Bundle::Write(frame, writer, kLayerHeader, aux_out);
}

}  // namespace jxl
