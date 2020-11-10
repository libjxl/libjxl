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

#include "lib/jxl/alpha.h"

#include <algorithm>

#include "lib/jxl/base/status.h"

namespace jxl {

// TODO(veluca): alpha is either always premultiplied, or never.
void PerformAlphaBlending(const AlphaBlendingInputLayer& bg,
                          const AlphaBlendingInputLayer& fg,
                          const AlphaBlendingOutput& out, size_t num_pixels) {
  const float bg_max_alpha = MaxAlpha(bg.alpha_bits);
  const float bg_rmax_alpha = 1.f / bg_max_alpha;
  const float fg_max_alpha = MaxAlpha(fg.alpha_bits);
  const float fg_rmax_alpha = 1.f / fg_max_alpha;
  const float out_max_alpha = MaxAlpha(out.alpha_bits);
  bool out_alpha_is_currently_premultiplied;
  if (!bg.alpha_is_premultiplied && !fg.alpha_is_premultiplied) {
    for (size_t x = 0; x < num_pixels; ++x) {
      const float new_a = (1.f - (1.f - fg.a[x] * fg_rmax_alpha) *
                                     (1.f - bg.a[x] * bg_rmax_alpha));
      const float rnew_a = 1.f / new_a;
      out.r[x] = (fg.r[x] * fg.a[x] * fg_rmax_alpha +
                  bg.r[x] * bg.a[x] * bg_rmax_alpha *
                      (1.f - fg.a[x] * fg_rmax_alpha)) *
                 rnew_a;
      out.g[x] = (fg.g[x] * fg.a[x] * fg_rmax_alpha +
                  bg.g[x] * bg.a[x] * bg_rmax_alpha *
                      (1.f - fg.a[x] * fg_rmax_alpha)) *
                 rnew_a;
      out.b[x] = (fg.b[x] * fg.a[x] * fg_rmax_alpha +
                  bg.b[x] * bg.a[x] * bg_rmax_alpha *
                      (1.f - fg.a[x] * fg_rmax_alpha)) *
                 rnew_a;
      out.a[x] = new_a * out_max_alpha + .5f;
    }
    out_alpha_is_currently_premultiplied = false;
  } else if (!bg.alpha_is_premultiplied && fg.alpha_is_premultiplied) {
    for (size_t x = 0; x < num_pixels; ++x) {
      out.r[x] = (fg.r[x] + bg.r[x] * bg.a[x] * bg_rmax_alpha *
                                (1.f - fg.a[x] * fg_rmax_alpha));
      out.g[x] = (fg.g[x] + bg.g[x] * bg.a[x] * bg_rmax_alpha *
                                (1.f - fg.a[x] * fg_rmax_alpha));
      out.b[x] = (fg.b[x] + bg.b[x] * bg.a[x] * bg_rmax_alpha *
                                (1.f - fg.a[x] * fg_rmax_alpha));
      out.a[x] = (1.f - (1.f - fg.a[x] * fg_rmax_alpha) *
                            (1.f - bg.a[x] * bg_rmax_alpha)) *
                     out_max_alpha +
                 .5f;
    }
    out_alpha_is_currently_premultiplied = true;
  } else if (bg.alpha_is_premultiplied && !fg.alpha_is_premultiplied) {
    for (size_t x = 0; x < num_pixels; ++x) {
      out.r[x] = (fg.r[x] * fg.a[x] * fg_rmax_alpha +
                  bg.r[x] * (1.f - fg.a[x] * fg_rmax_alpha));
      out.g[x] = (fg.g[x] * fg.a[x] * fg_rmax_alpha +
                  bg.g[x] * (1.f - fg.a[x] * fg_rmax_alpha));
      out.b[x] = (fg.b[x] * fg.a[x] * fg_rmax_alpha +
                  bg.b[x] * (1.f - fg.a[x] * fg_rmax_alpha));
      out.a[x] = (1.f - (1.f - fg.a[x] * fg_rmax_alpha) *
                            (1.f - bg.a[x] * bg_rmax_alpha)) *
                     out_max_alpha +
                 .5f;
    }
    out_alpha_is_currently_premultiplied = true;
  } else {
    JXL_ASSERT(bg.alpha_is_premultiplied && fg.alpha_is_premultiplied);
    for (size_t x = 0; x < num_pixels; ++x) {
      out.r[x] = (fg.r[x] + bg.r[x] * (1.f - fg.a[x] * fg_rmax_alpha));
      out.g[x] = (fg.g[x] + bg.g[x] * (1.f - fg.a[x] * fg_rmax_alpha));
      out.b[x] = (fg.b[x] + bg.b[x] * (1.f - fg.a[x] * fg_rmax_alpha));
      out.a[x] = (1.f - (1.f - fg.a[x] * fg_rmax_alpha) *
                            (1.f - bg.a[x] * bg_rmax_alpha)) *
                     out_max_alpha +
                 .5f;
    }
    out_alpha_is_currently_premultiplied = true;
  }

  if (out_alpha_is_currently_premultiplied && !out.alpha_is_premultiplied) {
    UnpremultiplyAlpha(out.r, out.g, out.b, out.a, out.alpha_bits, num_pixels);
    out_alpha_is_currently_premultiplied = false;
  } else if (!out_alpha_is_currently_premultiplied &&
             out.alpha_is_premultiplied) {
    PremultiplyAlpha(out.r, out.g, out.b, out.a, out.alpha_bits, num_pixels);
    out_alpha_is_currently_premultiplied = true;
  }

  JXL_DASSERT(out_alpha_is_currently_premultiplied ==
              out.alpha_is_premultiplied);
}

void PremultiplyAlpha(float* JXL_RESTRICT r, float* JXL_RESTRICT g,
                      float* JXL_RESTRICT b, const uint16_t* JXL_RESTRICT a,
                      size_t alpha_bits, size_t num_pixels) {
  const float alpha_normalizer = 1.f / MaxAlpha(alpha_bits);
  for (size_t x = 0; x < num_pixels; ++x) {
    const float multiplier = std::max(kSmallAlpha, a[x] * alpha_normalizer);
    r[x] *= multiplier;
    g[x] *= multiplier;
    b[x] *= multiplier;
  }
}

void UnpremultiplyAlpha(float* JXL_RESTRICT r, float* JXL_RESTRICT g,
                        float* JXL_RESTRICT b, const uint16_t* JXL_RESTRICT a,
                        size_t alpha_bits, size_t num_pixels) {
  const float alpha_normalizer = 1.f / MaxAlpha(alpha_bits);
  for (size_t x = 0; x < num_pixels; ++x) {
    const float multiplier =
        1.f / std::max(kSmallAlpha, a[x] * alpha_normalizer);
    r[x] *= multiplier;
    g[x] *= multiplier;
    b[x] *= multiplier;
  }
}

}  // namespace jxl
