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

#include "lib/jxl/enc_comparator.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/enc_gamma_correct.h"

namespace jxl {
namespace {

// color is linear, but blending happens in gamma-compressed space using
// (gamma-compressed) grayscale background color, alpha image represents
// weights of the sRGB colors in the [0 .. (1 << bit_depth) - 1] interval,
// output image is in linear space.
void AlphaBlend(const Image3F& in, const size_t c, float background_linear255,
                const ImageU& alpha, const uint16_t opaque, Image3F* out) {
  const float background = LinearToSrgb8Direct(background_linear255);

  for (size_t y = 0; y < out->ysize(); ++y) {
    const uint16_t* JXL_RESTRICT row_a = alpha.ConstRow(y);
    const float* JXL_RESTRICT row_i = in.ConstPlaneRow(c, y);
    float* JXL_RESTRICT row_o = out->PlaneRow(c, y);
    for (size_t x = 0; x < out->xsize(); ++x) {
      const uint16_t a = row_a[x];
      if (a == 0) {
        row_o[x] = background_linear255;
      } else if (a == opaque) {
        row_o[x] = row_i[x];
      } else {
        const float w_fg = a * 1.0f / opaque;
        const float w_bg = 1.0f - w_fg;
        const float fg = w_fg * LinearToSrgb8Direct(row_i[x]);
        const float bg = w_bg * background;
        row_o[x] = Srgb8ToLinearDirect(fg + bg);
      }
    }
  }
}

const Image3F* AlphaBlend(const ImageBundle& ib, const Image3F& linear,
                          float background_linear255, Image3F* copy) {
  // No alpha => all opaque.
  if (!ib.HasAlpha()) return &linear;

  *copy = Image3F(linear.xsize(), linear.ysize());
  const uint16_t opaque = (1U << ib.metadata()->GetAlphaBits()) - 1;
  for (size_t c = 0; c < 3; ++c) {
    AlphaBlend(linear, c, background_linear255, ib.alpha(), opaque, copy);
  }
  return copy;
}

void AlphaBlend(float background_linear255, ImageBundle* io_linear_srgb) {
  // No alpha => all opaque.
  if (!io_linear_srgb->HasAlpha()) return;

  const uint16_t opaque =
      (1U << io_linear_srgb->metadata()->GetAlphaBits()) - 1;
  for (size_t c = 0; c < 3; ++c) {
    AlphaBlend(*io_linear_srgb->color(), c, background_linear255,
               *io_linear_srgb->alpha(), opaque, io_linear_srgb->color());
  }
}

float ComputeScoreImpl(const ImageBundle& rgb0, const ImageBundle& rgb1,
                       Comparator* comparator, ImageF* distmap) {
  JXL_CHECK(comparator->SetReferenceImage(rgb0));
  float score;
  JXL_CHECK(comparator->CompareWith(rgb1, distmap, &score));
  return score;
}

}  // namespace

float ComputeScore(const ImageBundle& rgb0, const ImageBundle& rgb1,
                   Comparator* comparator, ImageF* diffmap, ThreadPool* pool) {
  PROFILER_FUNC;
  // Convert to linear sRGB (unless already in that space)
  ImageMetadata metadata0 = *rgb0.metadata();
  ImageBundle store0(&metadata0);
  const ImageBundle* linear_srgb0;
  JXL_CHECK(TransformIfNeeded(rgb0, ColorEncoding::LinearSRGB(rgb0.IsGray()),
                              pool, &store0, &linear_srgb0));
  ImageMetadata metadata1 = *rgb1.metadata();
  ImageBundle store1(&metadata1);
  const ImageBundle* linear_srgb1;
  JXL_CHECK(TransformIfNeeded(rgb1, ColorEncoding::LinearSRGB(rgb1.IsGray()),
                              pool, &store1, &linear_srgb1));

  return ComputeScoreImpl(*linear_srgb0, *linear_srgb1, comparator, diffmap);
  // No alpha: skip blending, only need a single call to Butteraugli.
  if (!rgb0.HasAlpha() && !rgb1.HasAlpha()) {
    return ComputeScoreImpl(*linear_srgb0, *linear_srgb1, comparator, diffmap);
  }

  // Blend on black and white backgrounds

  const float black = 0.0f;
  ImageBundle blended_black0(&metadata0);
  blended_black0.SetFromImage(CopyImage(linear_srgb0->color()),
                              linear_srgb0->c_current());
  ImageBundle blended_black1(&metadata1);
  blended_black1.SetFromImage(CopyImage(linear_srgb1->color()),
                              linear_srgb1->c_current());
  AlphaBlend(black, &blended_black0);
  AlphaBlend(black, &blended_black1);

  // TODO(lode): this is incorrect in case of intensity multiplier, consider
  // making intensity multiplier part of comparator
  const float white = 255.0f;
  ImageBundle blended_white0(&metadata0);
  blended_white0.SetFromImage(CopyImage(linear_srgb0->color()),
                              linear_srgb0->c_current());
  ImageBundle blended_white1(&metadata1);
  blended_white1.SetFromImage(CopyImage(linear_srgb1->color()),
                              linear_srgb1->c_current());
  AlphaBlend(white, &blended_white0);
  AlphaBlend(white, &blended_white1);

  ImageF diffmap_black, diffmap_white;
  const float dist_black = ComputeScoreImpl(blended_black0, blended_black1,
                                            comparator, &diffmap_black);
  const float dist_white = ComputeScoreImpl(blended_white0, blended_white1,
                                            comparator, &diffmap_white);

  // diffmap and return values are the max of diffmap_black/white.
  if (diffmap != nullptr) {
    const size_t xsize = rgb0.xsize();
    const size_t ysize = rgb0.ysize();
    *diffmap = ImageF(xsize, ysize);
    for (size_t y = 0; y < ysize; ++y) {
      const float* JXL_RESTRICT row_black = diffmap_black.ConstRow(y);
      const float* JXL_RESTRICT row_white = diffmap_white.ConstRow(y);
      float* JXL_RESTRICT row_out = diffmap->Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = std::max(row_black[x], row_white[x]);
      }
    }
  }
  return std::max(dist_black, dist_white);
}

}  // namespace jxl
