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

#include "jxl/dec_reconstruct.h"

#include <utility>

#include "jxl/aux_out.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/profiler.h"
#include "jxl/common.h"
#include "jxl/dec_xyb.h"
#include "jxl/epf.h"
#include "jxl/frame_header.h"
#include "jxl/gaborish.h"
#include "jxl/loop_filter.h"
#include "jxl/multiframe.h"
#include "jxl/passes_state.h"

namespace jxl {
using DF = HWY_CAPPED(float, kBlockDim);

// TODO(veluca): implement dynamic dispatch.
HWY_ATTR void ApplyImageFeatures(Image3F* JXL_RESTRICT idct, const Rect& rect,
                                 PassesDecoderState* dec_state, size_t thread,
                                 AuxOut* aux_out, bool save_decompressed,
                                 bool apply_color_transform) {
  JXL_CHECK(rect.x0() % kBlockDim == 0);
  JXL_CHECK(rect.y0() % kBlockDim == 0);
  JXL_CHECK(rect.xsize() % kBlockDim == 0);
  JXL_CHECK(rect.ysize() % kBlockDim == 0);

  const Rect block_rect(rect.x0() / kBlockDim, rect.y0() / kBlockDim,
                        rect.xsize() / kBlockDim, rect.ysize() / kBlockDim);

  const ImageFeatures& image_features = dec_state->shared->image_features;
  const FrameHeader& frame_header = dec_state->shared->frame_header;
  const LoopFilter& lf = image_features.loop_filter;
  const OpsinParams& opsin_params = dec_state->shared->opsin_params;

  if (lf.epf || lf.gab) {
    PROFILER_ZONE("EPF");
    EdgePreservingFilter(
        lf, rect, dec_state->decoded, block_rect, dec_state->sigma, rect, idct,
        &dec_state->storage1[thread], &dec_state->storage2[thread]);
  }

  // At this point, `idct:rect` holds the decoded pixels, independently of epf
  // or gaborish having been applied.

  // TODO(veluca): switch the rest of the function to row-based processing.
  image_features.patches.AddTo(idct, rect, rect);
  image_features.splines.AddTo(idct, rect, rect, dec_state->shared->cmap);

  if (dec_state->shared->multiframe->NeedsRestoring()) {
    PROFILER_ZONE("MultiframeRestore");
    for (size_t c = 0; c < 3; c++) {
      AddTo(rect, dec_state->frame_storage->Plane(c), rect, &idct->Plane(c));
    }
  }

  if (dec_state->shared->multiframe->NeedsSaving() && save_decompressed) {
    PROFILER_ZONE("MultiframeSave");
    CopyImageTo(rect, *idct, rect, dec_state->frame_storage);
  }

  if (frame_header.flags & FrameHeader::kNoise) {
    PROFILER_ZONE("AddNoise");
    AddNoise(image_features.noise_params, rect, dec_state->noise, rect,
             dec_state->shared->cmap, idct);
  }

  if (apply_color_transform &&
      frame_header.color_transform == ColorTransform::kXYB) {
    PROFILER_ZONE("ToXYB");
    for (size_t y = 0; y < rect.ysize(); y++) {
      float* JXL_RESTRICT row0 = rect.PlaneRow(idct, 0, y);
      float* JXL_RESTRICT row1 = rect.PlaneRow(idct, 1, y);
      float* JXL_RESTRICT row2 = rect.PlaneRow(idct, 2, y);

      const HWY_FULL(float) d;

      for (size_t x = 0; x < rect.xsize(); x += d.N) {
        const auto in_opsin_x = Load(d, row0 + x);
        const auto in_opsin_y = Load(d, row1 + x);
        const auto in_opsin_b = Load(d, row2 + x);
        JXL_COMPILER_FENCE;
        auto linear_r = Undefined(d);
        auto linear_g = Undefined(d);
        auto linear_b = Undefined(d);
        XybToRgb(d, in_opsin_x, in_opsin_y, in_opsin_b, opsin_params, &linear_r,
                 &linear_g, &linear_b);

        Store(linear_r, d, row0 + x);
        Store(linear_g, d, row1 + x);
        Store(linear_b, d, row2 + x);
      }
    }
  }
}

Status FinalizeFrameDecoding(Image3F* JXL_RESTRICT idct,
                             PassesDecoderState* dec_state, ThreadPool* pool,
                             AuxOut* aux_out, bool save_decompressed,
                             bool apply_color_transform) {
  std::vector<Rect> rects_to_process;

  const LoopFilter& lf = dec_state->shared->image_features.loop_filter;
  const FrameHeader& frame_header = dec_state->shared->frame_header;

  if (lf.epf || lf.gab) {
    // We can process larger tiles here, as they are at most 16 pixel wide.
    // Adding padding, this is 48 pixels, hence we can process up to
    // 96*96/48-32=160 pixels in the other dimension without exhausting L2
    // cache. We instead go with 128 pixels to keep rectangle sizes mostly
    // uniform (vertical rectangles sum up to 240 pixels as they should not
    // overlap with the horizontal rectangles, hence one rectangle of 160
    // pixels would imply another rectangle of 80).
    size_t xsize = dec_state->shared->frame_dim.xsize_padded;
    size_t ysize = dec_state->shared->frame_dim.ysize_padded;
    size_t xsize_groups = dec_state->shared->frame_dim.xsize_groups;
    size_t ysize_groups = dec_state->shared->frame_dim.ysize_groups;
    // For every gap between groups, vertically, enqueue bottom gap with next
    // group ...
    for (size_t ygroup = 0; ygroup < ysize_groups - 1; ygroup++) {
      size_t gystart = ygroup * kGroupDim;
      size_t gyend = std::min(ysize, kGroupDim * (ygroup + 1));
      // Group is processed together with another group.
      if (gyend <= gystart + kBlockDim) continue;
      for (size_t xstart = 0; xstart < xsize;
           xstart += kApplyImageFeaturesTileDim) {
        rects_to_process.emplace_back(xstart, gyend - kBlockDim,
                                      kApplyImageFeaturesTileDim, 2 * kBlockDim,
                                      xsize, ysize);
      }
    }
    // For every gap between groups, horizontally, enqueue right gap with next
    // group, carefully avoiding overlaps with the horizontal gaps enqueued
    // before...
    for (size_t xgroup = 0; xgroup < xsize_groups - 1; xgroup++) {
      size_t gxstart = xgroup == 0 ? kBlockDim : xgroup * kGroupDim;
      size_t gxend = std::min(xsize, kGroupDim * (xgroup + 1));
      // Group is processed together with another group.
      if (gxend <= gxstart + kBlockDim) continue;
      for (size_t ygroup = 0; ygroup < ysize_groups; ygroup++) {
        size_t gystart = ygroup == 0 ? 0 : ygroup * kGroupDim + kBlockDim;
        size_t gyend = ygroup == ysize_groups - 1
                           ? ysize
                           : kGroupDim * (ygroup + 1) - kBlockDim;
        if (gyend <= gystart) continue;
        for (size_t ystart = gystart; ystart < gyend;
             ystart += kApplyImageFeaturesTileDim) {
          rects_to_process.emplace_back(
              gxend - kBlockDim, ystart, 2 * kBlockDim,
              kApplyImageFeaturesTileDim, xsize, gyend);
        }
      }
    }
  }
  if (frame_header.encoding == FrameEncoding::kModularGroup) {
    for (size_t y = 0; y < idct->ysize(); y += kGroupDim) {
      for (size_t x = 0; x < idct->xsize(); x += kGroupDim) {
        rects_to_process.emplace_back(x, y, kGroupDim, kGroupDim, idct->xsize(),
                                      idct->ysize());
      }
    }
  }
  const auto allocate_storage = [&](size_t num_threads) {
    dec_state->EnsureStorage(num_threads);
    return true;
  };

  auto apply_features = [&](size_t rect_id, size_t thread) {
    ApplyImageFeatures(idct, rects_to_process[rect_id], dec_state, thread,
                       aux_out, save_decompressed, apply_color_transform);
  };

  RunOnPool(pool, 0, rects_to_process.size(), allocate_storage, apply_features,
            "ApplyFeatures");

  if (dec_state->shared->multiframe->NeedsSaving() && save_decompressed) {
    dec_state->shared->multiframe->SetDecodedFrame();
  }

  const size_t xsize = dec_state->shared->frame_dim.xsize;
  const size_t ysize = dec_state->shared->frame_dim.ysize;

  idct->ShrinkTo(xsize, ysize);

  if (apply_color_transform &&
      frame_header.color_transform == ColorTransform::kYCbCr) {
    // TODO(veluca): create per-pixel version of YcbcrToRgb for line-based
    // decoding in ApplyImageFeatures.
    YcbcrToRgb(idct->Plane(1), idct->Plane(0), idct->Plane(2),
               const_cast<ImageF*>(&idct->Plane(0)),
               const_cast<ImageF*>(&idct->Plane(1)),
               const_cast<ImageF*>(&idct->Plane(2)), pool);
  }  // otherwise no color transform needed

  return true;
}

}  // namespace jxl
