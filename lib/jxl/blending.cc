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

#include "lib/jxl/blending.h"

#include "lib/jxl/alpha.h"
#include "lib/jxl/image_ops.h"

namespace jxl {

Status DoBlending(const PassesSharedState& state, ImageBundle* foreground) {
  // No need to blend anything in this case.
  if (state.frame_header.frame_type != FrameType::kRegularFrame) {
    return true;
  }
  // Replace the full frame: nothing to do.
  if (state.frame_header.custom_size_or_origin == false &&
      state.frame_header.blending_info.mode == BlendMode::kReplace) {
    return true;
  }
  // This value should be 0 if there is no alpha channel.
  size_t first_alpha = 0;
  const std::vector<jxl::ExtraChannelInfo>& extra_channels =
      state.metadata->m.extra_channel_info;
  for (size_t i = 0; i < extra_channels.size(); i++) {
    if (extra_channels[i].type == jxl::ExtraChannel::kAlpha) {
      first_alpha = i;
      break;
    }
  }

  BlendingInfo info = state.frame_header.blending_info;
  const auto& ec_info = state.frame_header.extra_channel_blending_info;
  if (info.alpha_channel != first_alpha ||
      (extra_channels.size() > 1 &&
       (ec_info[first_alpha].alpha_channel != first_alpha ||
        info.source != ec_info[first_alpha].source))) {
    return JXL_FAILURE(
        "Blending from different alpha channels not yet implemented");
  }
  size_t image_xsize = state.frame_header.nonserialized_metadata->xsize();
  size_t image_ysize = state.frame_header.nonserialized_metadata->ysize();
  if (state.reference_frames[info.source].ib_is_in_xyb == true) {
    return JXL_FAILURE("Trying to blend XYB and non-XYB frames");
  }
  ImageBundle& bg = *state.reference_frames[info.source].frame;
  if (bg.xsize() != image_xsize || bg.ysize() != image_ysize ||
      bg.origin.x0 != 0 || bg.origin.y0 != 0) {
    return JXL_FAILURE("Trying to use a crop as a background");
  }
  // the rect in the canvas that needs to be updated
  Rect cropbox(0, 0, image_xsize, image_ysize);
  // the rect of this frame that overlaps with the canvas
  Rect overlap = cropbox;
  // Image to write to.
  if (state.frame_header.custom_size_or_origin) {
    FrameOrigin o = foreground->origin;
    int x0 = (o.x0 >= 0 ? o.x0 : 0);
    int y0 = (o.y0 >= 0 ? o.y0 : 0);
    int xsize = foreground->xsize();
    if (o.x0 < 0) xsize += o.x0;
    int ysize = foreground->ysize();
    if (o.y0 < 0) ysize += o.y0;
    xsize = Clamp1(xsize, 0, (int)cropbox.xsize() - x0);
    ysize = Clamp1(ysize, 0, (int)cropbox.ysize() - y0);
    if (xsize < 0) xsize = 0;
    if (ysize < 0) ysize = 0;
    cropbox = Rect(x0, y0, xsize, ysize);
    x0 = (o.x0 < 0 ? -o.x0 : 0);
    y0 = (o.y0 < 0 ? -o.y0 : 0);
    overlap = Rect(x0, y0, xsize, ysize);
  }
  // TODO(veluca): optimize memory copies here if we end up saving on the same
  // frame that we are reading from.
  ImageBundle dest = bg.Copy();
  if (info.mode == BlendMode::kAdd) {
    for (int p = 0; p < 3; p++) {
      AddTo(cropbox, foreground->color()->Plane(p), overlap,
            &dest.color()->Plane(p));
    }
    if (foreground->HasAlpha()) {
      AddTo(cropbox, *foreground->alpha(), overlap, dest.alpha());
    }
  } else if (info.mode == BlendMode::kBlend
             // blend without alpha is just replace
             && foreground->HasAlpha()) {
    size_t alpha_bits = foreground->metadata()->GetAlphaBits();
    bool is_premultiplied = foreground->AlphaIsPremultiplied();
    for (size_t y = 0; y < cropbox.ysize(); y++) {
      // Foreground.
      const uint16_t* JXL_RESTRICT a1 =
          overlap.ConstRow(*foreground->alpha(), y);
      const float* JXL_RESTRICT r1 =
          overlap.ConstRow(foreground->color()->Plane(0), y);
      const float* JXL_RESTRICT g1 =
          overlap.ConstRow(foreground->color()->Plane(1), y);
      const float* JXL_RESTRICT b1 =
          overlap.ConstRow(foreground->color()->Plane(2), y);
      // Background & destination.
      uint16_t* JXL_RESTRICT a = cropbox.Row(dest.alpha(), y);
      float* JXL_RESTRICT r = cropbox.Row(&dest.color()->Plane(0), y);
      float* JXL_RESTRICT g = cropbox.Row(&dest.color()->Plane(1), y);
      float* JXL_RESTRICT b = cropbox.Row(&dest.color()->Plane(2), y);
      PerformAlphaBlending(
          /*bg=*/
          {r, g, b, a, alpha_bits, is_premultiplied},
          /*fg=*/
          {r1, g1, b1, a1, alpha_bits, is_premultiplied},
          /*out=*/
          {r, g, b, a, alpha_bits, is_premultiplied}, cropbox.xsize());
    }
  } else if (info.mode == BlendMode::kAlphaWeightedAdd) {
    return JXL_FAILURE("BlendMode::kAlphaWeightedAdd not yet implemented");
  } else if (info.mode == BlendMode::kMul) {
    return JXL_FAILURE("BlendMode::kMul not yet implemented");
  } else {  // kReplace
    CopyImageTo(overlap, *foreground->color(), cropbox, dest.color());
    if (foreground->HasAlpha()) {
      CopyImageTo(overlap, *foreground->alpha(), cropbox, dest.alpha());
    }
  }
  *foreground = std::move(dest);
  return true;
}

}  // namespace jxl
