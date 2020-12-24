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

Status DoBlending(PassesDecoderState* dec_state, ImageBundle* foreground) {
  const PassesSharedState& state = *dec_state->shared;
  // No need to blend anything in this case.
  if (!(state.frame_header.frame_type == FrameType::kRegularFrame ||
        state.frame_header.frame_type == FrameType::kSkipProgressive)) {
    return true;
  }
  BlendingInfo info = state.frame_header.blending_info;
  bool replace_all = (info.mode == BlendMode::kReplace);
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

  const auto& ec_info = state.frame_header.extra_channel_blending_info;
  if (info.mode != BlendMode::kReplace && info.alpha_channel != first_alpha)
    return JXL_FAILURE(
        "Blending using non-first alpha channel not yet implemented");
  for (auto& ec_i : ec_info) {
    if (ec_i.mode != BlendMode::kReplace) {
      replace_all = false;
    }
    if (info.source != ec_i.source)
      return JXL_FAILURE("Blending from different sources not yet implemented");
  }

  // Replace the full frame: nothing to do.
  if (state.frame_header.custom_size_or_origin == false && replace_all) {
    return true;
  }

  size_t image_xsize = state.frame_header.nonserialized_metadata->xsize();
  size_t image_ysize = state.frame_header.nonserialized_metadata->ysize();

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
  if (overlap.xsize() == image_xsize && overlap.ysize() == image_ysize &&
      replace_all) {
    // frame is larger than image and fully replaces it, this is OK, just need
    // to crop
    ImageBundle dest = foreground->Copy();
    dest.RemoveColor();
    dest.extra_channels().clear();
    Image3F croppedcolor(image_xsize, image_ysize);
    Rect crop(-foreground->origin.x0, -foreground->origin.y0, image_xsize,
              image_ysize);
    CopyImageTo(crop, *foreground->color(), overlap, &croppedcolor);
    dest.SetFromImage(std::move(croppedcolor), foreground->c_current());
    std::vector<ImageF> ec;
    for (size_t i = 0; i < foreground->extra_channels().size(); i++) {
      ImageF cropped_ec(image_xsize, image_ysize);
      CopyImageTo(crop, foreground->extra_channels()[i], overlap, &cropped_ec);
      ec.push_back(std::move(cropped_ec));
    }
    dest.SetExtraChannels(std::move(ec));
    *foreground = std::move(dest);
    return true;
  }

  ImageBundle& bg = *state.reference_frames[info.source].frame;
  if (bg.xsize() == 0 && bg.ysize() == 0) {
    // there is no background, assume it to be all zeroes
    ImageBundle empty(foreground->metadata());
    Image3F color(image_xsize, image_ysize);
    ZeroFillImage(&color);
    empty.SetFromImage(std::move(color), foreground->c_current());
    std::vector<ImageF> ec;
    for (size_t i = 0; i < foreground->extra_channels().size(); i++) {
      ImageF eci(image_xsize, image_ysize);
      ZeroFillImage(&eci);
      ec.push_back(std::move(eci));
    }
    empty.SetExtraChannels(std::move(ec));
    bg = std::move(empty);
  } else if (state.reference_frames[info.source].ib_is_in_xyb == true) {
    return JXL_FAILURE(
        "Trying to blend XYB reference frame %i and non-XYB frame",
        info.source);
  }

  if (bg.xsize() != image_xsize || bg.ysize() != image_ysize ||
      bg.origin.x0 != 0 || bg.origin.y0 != 0) {
    return JXL_FAILURE("Trying to use a %zux%zu crop as a background",
                       bg.xsize(), bg.ysize());
  }
  // TODO(veluca): avoid doing colorspace conversion if we know it is not
  // needed.
  if (state.metadata->m.xyb_encoded) {
    if (dec_state->do_colorspace_transform == nullptr) {
      return JXL_FAILURE(
          "Blending requested but no color transform hook provided");
    }
    const auto& c_desired = state.metadata->m.color_encoding;
    JXL_RETURN_IF_ERROR(dec_state->do_colorspace_transform(
        foreground, c_desired, /*pool=*/nullptr));
  }
  // TODO(veluca): optimize memory copies here if we end up saving on the same
  // frame that we are reading from.
  ImageBundle dest = bg.Copy();
  if (info.mode == BlendMode::kAdd) {
    for (int p = 0; p < 3; p++) {
      AddTo(overlap, foreground->color()->Plane(p), cropbox,
            &dest.color()->Plane(p));
    }
    if (foreground->HasAlpha()) {
      AddTo(overlap, *foreground->alpha(), cropbox, dest.alpha());
    }
  } else if (info.mode == BlendMode::kBlend
             // blend without alpha is just replace
             && foreground->HasAlpha()) {
    bool is_premultiplied = foreground->AlphaIsPremultiplied();
    for (size_t y = 0; y < cropbox.ysize(); y++) {
      // Foreground.
      const float* JXL_RESTRICT a1 = overlap.ConstRow(*foreground->alpha(), y);
      const float* JXL_RESTRICT r1 =
          overlap.ConstRow(foreground->color()->Plane(0), y);
      const float* JXL_RESTRICT g1 =
          overlap.ConstRow(foreground->color()->Plane(1), y);
      const float* JXL_RESTRICT b1 =
          overlap.ConstRow(foreground->color()->Plane(2), y);
      // Background & destination.
      float* JXL_RESTRICT a = cropbox.Row(dest.alpha(), y);
      float* JXL_RESTRICT r = cropbox.Row(&dest.color()->Plane(0), y);
      float* JXL_RESTRICT g = cropbox.Row(&dest.color()->Plane(1), y);
      float* JXL_RESTRICT b = cropbox.Row(&dest.color()->Plane(2), y);
      PerformAlphaBlending(/*bg=*/{r, g, b, a}, /*fg=*/{r1, g1, b1, a1},
                           /*out=*/{r, g, b, a}, cropbox.xsize(),
                           is_premultiplied);
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
  for (size_t i = 0; i < ec_info.size(); i++) {
    if (i == first_alpha) continue;
    if (ec_info[i].mode == BlendMode::kAdd) {
      AddTo(overlap, (foreground->extra_channels())[i], cropbox,
            &dest.extra_channels()[i]);
    } else if (ec_info[i].mode == BlendMode::kBlend) {
      if (ec_info[i].alpha_channel != first_alpha)
        return JXL_FAILURE("Not implemented: blending using non-first alpha");
      bool is_premultiplied = foreground->AlphaIsPremultiplied();
      for (size_t y = 0; y < cropbox.ysize(); y++) {
        const float* JXL_RESTRICT a1 =
            overlap.ConstRow(*foreground->alpha(), y);
        float* JXL_RESTRICT p1 =
            overlap.Row(&foreground->extra_channels()[i], y);
        const float* JXL_RESTRICT a = cropbox.ConstRow(*dest.alpha(), y);
        float* JXL_RESTRICT p = cropbox.Row(&dest.extra_channels()[i], y);
        PerformAlphaBlending(p, a, p1, a1, p, cropbox.xsize(),
                             is_premultiplied);
      }
    } else if (ec_info[i].mode == BlendMode::kReplace) {
      CopyImageTo(overlap, foreground->extra_channels()[i], cropbox,
                  &dest.extra_channels()[i]);
    } else
      return JXL_FAILURE("Blend mode not implemented for extra channel %zu", i);
  }
  *foreground = std::move(dest);
  return true;
}

}  // namespace jxl
