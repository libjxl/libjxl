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

#include "jxl/passes_state.h"

#include "jxl/chroma_from_luma.h"
#include "jxl/common.h"

namespace jxl {

Status InitializePassesSharedState(const FrameHeader& frame_header,
                                   const LoopFilter& loop_filter,
                                   const ImageMetadata& image_metadata,
                                   const FrameDimensions& frame_dim,
                                   Multiframe* JXL_RESTRICT multiframe,
                                   PassesSharedState* JXL_RESTRICT shared,
                                   bool encoder) {
  shared->frame_header = frame_header;
  shared->metadata = image_metadata;
  shared->frame_dim = frame_dim;
  shared->image_features.loop_filter = loop_filter;
  shared->multiframe = multiframe;

  shared->ac_strategy =
      AcStrategyImage(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  shared->raw_quant_field =
      ImageI(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  shared->epf_sharpness =
      ImageB(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  shared->cmap = ColorCorrelationMap(frame_dim.xsize, frame_dim.ysize);

  shared->opsin_params = image_metadata.m2.opsin_inverse_matrix.ToOpsinParams();

  shared->quant_dc = ImageB(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  if (!(frame_header.flags & FrameHeader::kUseDcFrame) || encoder) {
    shared->dc_storage =
        Image3F(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  } else {
    if (frame_header.dc_level == 3) {
      return JXL_FAILURE("Invalid DC level for kUseDcFrame: %u",
                         frame_header.dc_level);
    }
    shared->dc = multiframe->SavedDc(frame_header.dc_level + 1);
    if (shared->dc->xsize() == 0) {
      return JXL_FAILURE(
          "kUseDcFrame specified for dc_level %u, but no frame was decoded "
          "with level %u",
          frame_header.dc_level, frame_header.dc_level + 1);
    }
    ZeroFillImage(&shared->quant_dc);
  }
  shared->image_features.patches.SetReferenceFrames(
      multiframe->GetReferenceFrames());

  shared->dc_storage = Image3F(frame_dim.xsize_blocks, frame_dim.ysize_blocks);

  return true;
}

}  // namespace jxl
