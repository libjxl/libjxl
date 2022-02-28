// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_cache.h"

#include "lib/jxl/blending.h"
#include "lib/jxl/dec_reconstruct.h"
#include "lib/jxl/render_pipeline/stage_blending.h"
#include "lib/jxl/render_pipeline/stage_chroma_upsampling.h"
#include "lib/jxl/render_pipeline/stage_epf.h"
#include "lib/jxl/render_pipeline/stage_gaborish.h"
#include "lib/jxl/render_pipeline/stage_noise.h"
#include "lib/jxl/render_pipeline/stage_patches.h"
#include "lib/jxl/render_pipeline/stage_splines.h"
#include "lib/jxl/render_pipeline/stage_spot.h"
#include "lib/jxl/render_pipeline/stage_upsampling.h"
#include "lib/jxl/render_pipeline/stage_write.h"
#include "lib/jxl/render_pipeline/stage_xyb.h"
#include "lib/jxl/render_pipeline/stage_ycbcr.h"

namespace jxl {

void PassesDecoderState::EnsureBordersStorage() {
  if (!EagerFinalizeImageRect()) return;
  size_t padding = FinalizeRectPadding();
  size_t bordery = 2 * padding;
  size_t borderx = padding + RoundUpToBlockDim(padding);
  Rect horizontal = Rect(0, 0, shared->frame_dim.xsize_padded,
                         bordery * shared->frame_dim.ysize_groups * 2);
  if (!SameSize(horizontal, borders_horizontal)) {
    borders_horizontal = Image3F(horizontal.xsize(), horizontal.ysize());
  }
  Rect vertical = Rect(0, 0, borderx * shared->frame_dim.xsize_groups * 2,
                       shared->frame_dim.ysize_padded);
  if (!SameSize(vertical, borders_vertical)) {
    borders_vertical = Image3F(vertical.xsize(), vertical.ysize());
  }
}

namespace {
void SaveBorders(const Rect& block_rect, size_t hshift, size_t vshift,
                 size_t padding, const ImageF& plane_in,
                 ImageF* border_storage_h, ImageF* border_storage_v) {
  constexpr size_t kGroupDataXBorder = PassesDecoderState::kGroupDataXBorder;
  constexpr size_t kGroupDataYBorder = PassesDecoderState::kGroupDataYBorder;
  size_t x0 = DivCeil(block_rect.x0() * kBlockDim, 1 << hshift);
  size_t x1 =
      DivCeil((block_rect.x0() + block_rect.xsize()) * kBlockDim, 1 << hshift);
  size_t y0 = DivCeil(block_rect.y0() * kBlockDim, 1 << vshift);
  size_t y1 =
      DivCeil((block_rect.y0() + block_rect.ysize()) * kBlockDim, 1 << vshift);
  size_t gy = block_rect.y0() / kGroupDimInBlocks;
  size_t gx = block_rect.x0() / kGroupDimInBlocks;
  // TODO(veluca): this is too much with chroma upsampling. It's just
  // inefficient though.
  size_t borderx = RoundUpToBlockDim(padding);
  size_t bordery = padding;
  size_t borderx_write = padding + borderx;
  size_t bordery_write = padding + bordery;
  CopyImageTo(
      Rect(kGroupDataXBorder, kGroupDataYBorder, x1 - x0, bordery_write),
      plane_in, Rect(x0, (gy * 2) * bordery_write, x1 - x0, bordery_write),
      border_storage_h);
  CopyImageTo(
      Rect(kGroupDataXBorder, kGroupDataYBorder + y1 - y0 - bordery_write,
           x1 - x0, bordery_write),
      plane_in, Rect(x0, (gy * 2 + 1) * bordery_write, x1 - x0, bordery_write),
      border_storage_h);
  CopyImageTo(
      Rect(kGroupDataXBorder, kGroupDataYBorder, borderx_write, y1 - y0),
      plane_in, Rect((gx * 2) * borderx_write, y0, borderx_write, y1 - y0),
      border_storage_v);
  CopyImageTo(Rect(kGroupDataXBorder + x1 - x0 - borderx_write,
                   kGroupDataYBorder, borderx_write, y1 - y0),
              plane_in,
              Rect((gx * 2 + 1) * borderx_write, y0, borderx_write, y1 - y0),
              border_storage_v);
}

void LoadBorders(const Rect& block_rect, size_t hshift, size_t vshift,
                 const FrameDimensions& frame_dim, size_t padding,
                 const ImageF& border_storage_h, const ImageF& border_storage_v,
                 const Rect& r, ImageF* plane_out) {
  constexpr size_t kGroupDataXBorder = PassesDecoderState::kGroupDataXBorder;
  constexpr size_t kGroupDataYBorder = PassesDecoderState::kGroupDataYBorder;
  size_t x0 = DivCeil(block_rect.x0() * kBlockDim, 1 << hshift);
  size_t x1 =
      DivCeil((block_rect.x0() + block_rect.xsize()) * kBlockDim, 1 << hshift);
  size_t y0 = DivCeil(block_rect.y0() * kBlockDim, 1 << vshift);
  size_t y1 =
      DivCeil((block_rect.y0() + block_rect.ysize()) * kBlockDim, 1 << vshift);
  size_t gy = block_rect.y0() / kGroupDimInBlocks;
  size_t gx = block_rect.x0() / kGroupDimInBlocks;
  size_t borderx = RoundUpToBlockDim(padding);
  size_t bordery = padding;
  size_t borderx_write = padding + borderx;
  size_t bordery_write = padding + bordery;
  // Limits of the area to copy from, in image coordinates.
  JXL_DASSERT(r.x0() == 0 || r.x0() >= borderx);
  size_t x0src = DivCeil(r.x0() == 0 ? r.x0() : r.x0() - borderx, 1 << hshift);
  // r may be such that r.x1 (namely x0() + xsize()) is within borderx of the
  // right side of the image, so we use min() here.
  size_t x1src =
      DivCeil(std::min(r.x0() + r.xsize() + borderx, frame_dim.xsize_padded),
              1 << hshift);
  JXL_DASSERT(r.y0() == 0 || r.y0() >= bordery);
  size_t y0src = DivCeil(r.y0() == 0 ? r.y0() : r.y0() - bordery, 1 << vshift);
  // Similar to x1, y1 might be closer than bordery from the bottom.
  size_t y1src =
      DivCeil(std::min(r.y0() + r.ysize() + bordery, frame_dim.ysize_padded),
              1 << vshift);
  // Copy other groups' borders from the border storage.
  if (y0src < y0) {
    JXL_DASSERT(gy > 0);
    CopyImageTo(
        Rect(x0src, (gy * 2 - 1) * bordery_write, x1src - x0src, bordery_write),
        border_storage_h,
        Rect(kGroupDataXBorder + x0src - x0, kGroupDataYBorder - bordery_write,
             x1src - x0src, bordery_write),
        plane_out);
  }
  if (y1src > y1) {
    // When copying the bottom border we must not be on the bottom groups.
    JXL_DASSERT(gy + 1 < frame_dim.ysize_groups);
    CopyImageTo(
        Rect(x0src, (gy * 2 + 2) * bordery_write, x1src - x0src, bordery_write),
        border_storage_h,
        Rect(kGroupDataXBorder + x0src - x0, kGroupDataYBorder + y1 - y0,
             x1src - x0src, bordery_write),
        plane_out);
  }
  if (x0src < x0) {
    JXL_DASSERT(gx > 0);
    CopyImageTo(
        Rect((gx * 2 - 1) * borderx_write, y0src, borderx_write, y1src - y0src),
        border_storage_v,
        Rect(kGroupDataXBorder - borderx_write, kGroupDataYBorder + y0src - y0,
             borderx_write, y1src - y0src),
        plane_out);
  }
  if (x1src > x1) {
    // When copying the right border we must not be on the rightmost groups.
    JXL_DASSERT(gx + 1 < frame_dim.xsize_groups);
    CopyImageTo(
        Rect((gx * 2 + 2) * borderx_write, y0src, borderx_write, y1src - y0src),
        border_storage_v,
        Rect(kGroupDataXBorder + x1 - x0, kGroupDataYBorder + y0src - y0,
             borderx_write, y1src - y0src),
        plane_out);
  }
}

}  // namespace

Status PassesDecoderState::FinalizeGroup(size_t group_idx, size_t thread,
                                         Image3F* pixel_data,
                                         ImageBundle* output) {
  // Copy the group borders to the border storage.
  const Rect block_rect = shared->BlockGroupRect(group_idx);
  const YCbCrChromaSubsampling& cs = shared->frame_header.chroma_subsampling;
  size_t padding = FinalizeRectPadding();
  for (size_t c = 0; c < 3; c++) {
    SaveBorders(block_rect, cs.HShift(c), cs.VShift(c), padding,
                pixel_data->Plane(c), &borders_horizontal.Plane(c),
                &borders_vertical.Plane(c));
  }
  Rect fir_rects[GroupBorderAssigner::kMaxToFinalize];
  size_t num_fir_rects = 0;
  group_border_assigner.GroupDone(
      group_idx, RoundUpToBlockDim(FinalizeRectPadding()),
      FinalizeRectPadding(), fir_rects, &num_fir_rects);
  for (size_t i = 0; i < num_fir_rects; i++) {
    const Rect& r = fir_rects[i];
    for (size_t c = 0; c < 3; c++) {
      LoadBorders(block_rect, cs.HShift(c), cs.VShift(c), shared->frame_dim,
                  padding, borders_horizontal.Plane(c),
                  borders_vertical.Plane(c), r, &pixel_data->Plane(c));
    }
    Rect pixel_data_rect(
        kGroupDataXBorder + r.x0() - block_rect.x0() * kBlockDim,
        kGroupDataYBorder + r.y0() - block_rect.y0() * kBlockDim, r.xsize(),
        r.ysize());
    JXL_RETURN_IF_ERROR(FinalizeImageRect(pixel_data, pixel_data_rect, {}, this,
                                          thread, output, r));
  }
  return true;
}

Status PassesDecoderState::PreparePipeline(ImageBundle* decoded,
                                           PipelineOptions options) {
  const FrameHeader& frame_header = shared->frame_header;
  size_t num_c = 3 + frame_header.nonserialized_metadata->m.num_extra_channels;
  if ((frame_header.flags & FrameHeader::kNoise) != 0) {
    num_c += 3;
  }

  if (frame_header.CanBeReferenced()) {
    // Necessary so that SetInputSizes() can allocate output buffers as needed.
    frame_storage_for_referencing = ImageBundle(decoded->metadata());
  }

  RenderPipeline::Builder builder(num_c);

  if (options.use_slow_render_pipeline) {
    builder.UseSimpleImplementation();
  }

  if (!frame_header.chroma_subsampling.Is444()) {
    for (size_t c = 0; c < 3; c++) {
      if (frame_header.chroma_subsampling.HShift(c) != 0) {
        builder.AddStage(GetChromaUpsamplingStage(c, /*horizontal=*/true));
      }
      if (frame_header.chroma_subsampling.VShift(c) != 0) {
        builder.AddStage(GetChromaUpsamplingStage(c, /*horizontal=*/false));
      }
    }
  }

  if (frame_header.loop_filter.gab) {
    builder.AddStage(GetGaborishStage(frame_header.loop_filter));
  }

  {
    const LoopFilter& lf = frame_header.loop_filter;
    if (lf.epf_iters >= 3) {
      builder.AddStage(GetEPFStage(lf, filter_weights.sigma, 0));
    }
    if (lf.epf_iters >= 1) {
      builder.AddStage(GetEPFStage(lf, filter_weights.sigma, 1));
    }
    if (lf.epf_iters >= 2) {
      builder.AddStage(GetEPFStage(lf, filter_weights.sigma, 2));
    }
  }

  bool late_ec_upsample = frame_header.upsampling != 1;
  for (auto ecups : frame_header.extra_channel_upsampling) {
    if (ecups != frame_header.upsampling) {
      // If patches are applied, either frame_header.upsampling == 1 or
      // late_ec_upsample is true.
      late_ec_upsample = false;
    }
  }

  if (!late_ec_upsample) {
    for (size_t ec = 0; ec < frame_header.extra_channel_upsampling.size();
         ec++) {
      if (frame_header.extra_channel_upsampling[ec] != 1) {
        builder.AddStage(GetUpsamplingStage(
            frame_header.nonserialized_metadata->transform_data, 3 + ec,
            CeilLog2Nonzero(frame_header.extra_channel_upsampling[ec])));
      }
    }
  }

  if ((frame_header.flags & FrameHeader::kPatches) != 0) {
    builder.AddStage(GetPatchesStage(&shared->image_features.patches));
  }
  if ((frame_header.flags & FrameHeader::kSplines) != 0) {
    builder.AddStage(GetSplineStage(&shared->image_features.splines));
  }

  if (frame_header.upsampling != 1) {
    size_t nb_channels =
        3 +
        (late_ec_upsample ? frame_header.extra_channel_upsampling.size() : 0);
    for (size_t c = 0; c < nb_channels; c++) {
      builder.AddStage(GetUpsamplingStage(
          frame_header.nonserialized_metadata->transform_data, c,
          CeilLog2Nonzero(frame_header.upsampling)));
    }
  }

  if ((frame_header.flags & FrameHeader::kNoise) != 0) {
    builder.AddStage(GetConvolveNoiseStage(num_c - 3));
    builder.AddStage(GetAddNoiseStage(shared->image_features.noise_params,
                                      shared->cmap, num_c - 3));
  }
  if (frame_header.dc_level != 0) {
    builder.AddStage(GetWriteToImage3FStage(
        &shared_storage.dc_frames[frame_header.dc_level - 1]));
  }

  if (frame_header.CanBeReferenced() &&
      frame_header.save_before_color_transform) {
    builder.AddStage(GetWriteToImageBundleStage(
        &frame_storage_for_referencing, output_encoding_info.color_encoding));
  }

  bool has_alpha = false;
  size_t alpha_c = 0;
  for (size_t i = 0; i < decoded->metadata()->extra_channel_info.size(); i++) {
    if (decoded->metadata()->extra_channel_info[i].type ==
        ExtraChannel::kAlpha) {
      has_alpha = true;
      alpha_c = 3 + i;
      break;
    }
  }

  size_t width = options.coalescing
                     ? frame_header.nonserialized_metadata->xsize()
                     : shared->frame_dim.xsize_upsampled;
  size_t height = options.coalescing
                      ? frame_header.nonserialized_metadata->ysize()
                      : shared->frame_dim.ysize_upsampled;

  if (fast_xyb_srgb8_conversion) {
    JXL_ASSERT(!ImageBlender::NeedsBlending(this));
    JXL_ASSERT(!frame_header.CanBeReferenced() ||
               frame_header.save_before_color_transform);
    JXL_ASSERT(!options.render_spotcolors ||
               !decoded->metadata()->Find(ExtraChannel::kSpotColor));
    builder.AddStage(GetFastXYBTosRGB8Stage(rgb_output, rgb_stride, width,
                                            height, rgb_output_is_rgba,
                                            has_alpha, alpha_c));
  } else {
    if (frame_header.color_transform == ColorTransform::kYCbCr) {
      builder.AddStage(GetYCbCrStage());
    } else if (frame_header.color_transform == ColorTransform::kXYB) {
      builder.AddStage(GetXYBStage(output_encoding_info));
    }  // Nothing to do for kNone.

    if (options.coalescing && ImageBlender::NeedsBlending(this)) {
      builder.AddStage(
          GetBlendingStage(this, output_encoding_info.color_encoding));
    }

    if (options.coalescing && frame_header.CanBeReferenced() &&
        !frame_header.save_before_color_transform) {
      builder.AddStage(GetWriteToImageBundleStage(
          &frame_storage_for_referencing, output_encoding_info.color_encoding));
    }

    if (options.render_spotcolors &&
        frame_header.nonserialized_metadata->m.Find(ExtraChannel::kSpotColor)) {
      for (size_t i = 0; i < decoded->metadata()->extra_channel_info.size();
           i++) {
        // Don't use Find() because there may be multiple spot color channels.
        const ExtraChannelInfo& eci =
            decoded->metadata()->extra_channel_info[i];
        if (eci.type == ExtraChannel::kSpotColor) {
          builder.AddStage(GetSpotColorStage(3 + i, eci.spot_color));
        }
      }
    }

    if (pixel_callback) {
      builder.AddStage(GetWriteToPixelCallbackStage(pixel_callback, width,
                                                    height, rgb_output_is_rgba,
                                                    has_alpha, alpha_c));
    } else if (rgb_output) {
      builder.AddStage(GetWriteToU8Stage(rgb_output, rgb_stride, width, height,
                                         rgb_output_is_rgba, has_alpha,
                                         alpha_c));
    } else {
      builder.AddStage(GetWriteToImageBundleStage(
          decoded, output_encoding_info.color_encoding));
    }
  }
  render_pipeline = std::move(builder).Finalize(shared->frame_dim);
  return render_pipeline->IsInitialized();
}

}  // namespace jxl
