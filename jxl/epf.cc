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

// Edge-preserving smoothing: weighted average based on L1 patch similarity.

#include "jxl/epf.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/epf.cc"
#include <hwy/foreach_target.h>
// ^ must come before highway.h and any *-inl.h.

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <hwy/highway.h>
#include <mutex>
#include <numeric>  // std::accumulate
#include <vector>

#include "jxl/ac_strategy.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/convolve.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/loop_filter.h"
#include "jxl/quant_weights.h"
#include "jxl/quantizer.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Vec;

// Avoid compiler complaints about % 0 not being defined.
template <size_t m>
struct Mod {
  size_t operator()(size_t x) { return x % m; }
};
template <>
struct Mod<0> {
  size_t operator()(size_t x) { return x; }
};

// Utility struct to define input/output rows of row-based loop filters.
template <int kBorder>
struct LoopFilterRows {
  template <int row>
  const float* GetInputRow(size_t c) const {
    static_assert(-kBorder <= row && row <= kBorder, "Invalid row accessed");
    return rows_in_[c] + offsets_in_[kBorder + row];
  }

  float* GetOutputRow(size_t c) const { return rows_out_[c]; }

  template <size_t row_mod>
  void SetInput(const Image3F& in, size_t y0, size_t x0) {
    JXL_DASSERT(row_mod + y0 >= kBorder);
    for (size_t c = 0; c < 3; c++) {
      rows_in_[c] = in.ConstPlaneRow(c, 0);
    }
    for (int i = -kBorder; i <= kBorder; i++) {
      size_t y = Mod<row_mod>()(row_mod + y0 + i);
      offsets_in_[i + kBorder] = y * in.PixelsPerRow() + x0;
    }
  }

  template <size_t row_mod>
  void SetOutput(Image3F* out, size_t y0, size_t x0) {
    size_t y = Mod<row_mod>()(y0);
    for (size_t c = 0; c < 3; c++) {
      rows_out_[c] = out->PlaneRow(c, y) + x0;
    }
  }

 private:
  const float* JXL_RESTRICT rows_in_[3];
  size_t offsets_in_[2 * kBorder + 1];
  float* JXL_RESTRICT rows_out_[3];
};

using DF = HWY_CAPPED(float, 8);
using DU = HWY_CAPPED(uint32_t, 8);

// kInvSigmaNum / 0.3
constexpr float kMinSigma = -3.90524291751269967465540850526868f;

DF df;

JXL_INLINE Vec<DF> Weight(Vec<DF> sad, Vec<DF> inv_sigma, Vec<DF> thres) {
  auto v = MulAdd(sad, inv_sigma, Set(DF(), 1.0f));
  auto v2 = v * v;
  return IfThenZeroElse(v <= thres, v2);
}

template <bool aligned, int row>
JXL_INLINE void AddPixelStep1(const LoopFilterRows<2>& rows, size_t x,
                              Vec<DF> sad, Vec<DF> sad_mul, Vec<DF> inv_sigma,
                              const LoopFilter& lf, Vec<DF>* JXL_RESTRICT X,
                              Vec<DF>* JXL_RESTRICT Y, Vec<DF>* JXL_RESTRICT B,
                              Vec<DF>* JXL_RESTRICT w) {
  auto cx = aligned ? Load(DF(), rows.GetInputRow<row>(0) + x)
                    : LoadU(DF(), rows.GetInputRow<row>(0) + x);
  auto cy = aligned ? Load(DF(), rows.GetInputRow<row>(1) + x)
                    : LoadU(DF(), rows.GetInputRow<row>(1) + x);
  auto cb = aligned ? Load(DF(), rows.GetInputRow<row>(2) + x)
                    : LoadU(DF(), rows.GetInputRow<row>(2) + x);

  auto weight =
      Weight(sad * sad_mul, inv_sigma, Set(df, lf.epf_pass1_zeroflush));
  *w += weight;
  *X = MulAdd(weight, cx, *X);
  *Y = MulAdd(weight, cy, *Y);
  *B = MulAdd(weight, cb, *B);
}

template <bool aligned, int row>
JXL_INLINE void AddPixelStep2(const LoopFilterRows<1>& rows, size_t x,
                              Vec<DF> rx, Vec<DF> ry, Vec<DF> rb,
                              Vec<DF> sad_mul, Vec<DF> inv_sigma,
                              const LoopFilter& lf, Vec<DF>* JXL_RESTRICT X,
                              Vec<DF>* JXL_RESTRICT Y, Vec<DF>* JXL_RESTRICT B,
                              Vec<DF>* JXL_RESTRICT w) {
  auto cx = aligned ? Load(DF(), rows.GetInputRow<row>(0) + x)
                    : LoadU(DF(), rows.GetInputRow<row>(0) + x);
  auto cy = aligned ? Load(DF(), rows.GetInputRow<row>(1) + x)
                    : LoadU(DF(), rows.GetInputRow<row>(1) + x);
  auto cb = aligned ? Load(DF(), rows.GetInputRow<row>(2) + x)
                    : LoadU(DF(), rows.GetInputRow<row>(2) + x);

  auto sad = AbsDiff(cx, rx) * Set(df, lf.epf_channel_scale[0]);
  sad = MulAdd(AbsDiff(cy, ry), Set(df, lf.epf_channel_scale[1]), sad);
  sad = MulAdd(AbsDiff(cb, rb), Set(df, lf.epf_channel_scale[2]), sad);

  auto weight =
      Weight(sad * sad_mul, inv_sigma, Set(df, lf.epf_pass2_zeroflush));

  *w += weight;
  *X = MulAdd(weight, cx, *X);
  *Y = MulAdd(weight, cy, *Y);
  *B = MulAdd(weight, cb, *B);
}

void GaborishRow(const LoopFilterRows<1>& rows, size_t xsize,
                 const float* JXL_RESTRICT gab_weights) {
  for (size_t c = 0; c < 3; c++) {
    float* JXL_RESTRICT row_out = rows.GetOutputRow(c);
    const auto w0 = Set(df, gab_weights[3 * c]);
    const auto w1 = Set(df, gab_weights[3 * c + 1]);
    const auto w2 = Set(df, gab_weights[3 * c + 2]);
    for (size_t ix = 0; ix < xsize; ix += Lanes(df)) {
      const auto t = Load(df, rows.GetInputRow<-1>(c) + ix);
      const auto tl = LoadU(df, rows.GetInputRow<-1>(c) + ix - 1);
      const auto tr = LoadU(df, rows.GetInputRow<-1>(c) + ix + 1);
      const auto m = Load(df, rows.GetInputRow<0>(c) + ix);
      const auto l = LoadU(df, rows.GetInputRow<0>(c) + ix - 1);
      const auto r = LoadU(df, rows.GetInputRow<0>(c) + ix + 1);
      const auto b = Load(df, rows.GetInputRow<1>(c) + ix);
      const auto bl = LoadU(df, rows.GetInputRow<1>(c) + ix - 1);
      const auto br = LoadU(df, rows.GetInputRow<1>(c) + ix + 1);
      const auto sum0 = m;
      const auto sum1 = l + r + t + b;
      const auto sum2 = tl + tr + bl + br;
      auto pixels = MulAdd(sum2, w2, MulAdd(sum1, w1, sum0 * w0));
      Store(pixels, df, row_out + ix);
    }
  }
}

// Step 1: 3x3 plus-shaped kernel with 5 SADs per pixel (also 3x3
// plus-shaped).
void Epf1Row(const LoopFilterRows<2>& rows, const LoopFilter& lf, size_t iy,
             const float* JXL_RESTRICT row_sigma, size_t xsize) {
  HWY_ALIGN float sad_mul[kBlockDim] = {
      lf.epf_border_sad_mul, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
      lf.epf_border_sad_mul};

  const size_t N = Lanes(df);
  for (size_t x = kBlockDim - N; x < xsize + kBlockDim + N; x += N) {
    size_t sx = x + kBlockDim;
    size_t bx = x / kBlockDim;
    size_t ix = x - kBlockDim * bx;
    if (row_sigma[bx + 1] < kMinSigma) {
      for (size_t c = 0; c < 3; c++) {
        auto px = Load(df, rows.GetInputRow<0>(c) + sx);
        Store(px, df, rows.GetOutputRow(c) + x);
      }
      continue;
    }
    const auto inv_sigma = Set(DF(), row_sigma[bx + 1]);

    const auto sm = iy == 0 || iy == kBlockDim - 1
                        ? Set(df, lf.epf_border_sad_mul)
                        : Load(df, sad_mul + ix);
    auto sad0 = Zero(df);
    auto sad1 = Zero(df);
    auto sad2 = Zero(df);
    auto sad3 = Zero(df);

    // compute sads
    for (size_t c = 0; c < 3; c++) {
      // center px = 22, px above = 21
      auto t = Undefined(df);

      const auto p20 = Load(df, rows.GetInputRow<-2>(c) + sx);
      const auto p21 = Load(df, rows.GetInputRow<-1>(c) + sx);
      auto sad0c = AbsDiff(p20, p21);  // SAD 2, 1

      const auto p11 = LoadU(df, rows.GetInputRow<-1>(c) + sx - 1);
      auto sad1c = AbsDiff(p11, p21);  // SAD 1, 2

      const auto p31 = LoadU(df, rows.GetInputRow<-1>(c) + sx + 1);
      auto sad2c = AbsDiff(p31, p21);  // SAD 3, 2

      const auto p02 = LoadU(df, rows.GetInputRow<0>(c) + sx - 2);
      const auto p12 = LoadU(df, rows.GetInputRow<0>(c) + sx - 1);
      sad1c += AbsDiff(p02, p12);  // SAD 1, 2
      sad0c += AbsDiff(p11, p12);  // SAD 2, 1

      const auto p22 = LoadU(df, rows.GetInputRow<0>(c) + sx);
      t = AbsDiff(p12, p22);
      sad1c += t;  // SAD 1, 2
      sad2c += t;  // SAD 3, 2
      t = AbsDiff(p22, p21);
      auto sad3c = t;  // SAD 2, 3
      sad0c += t;      // SAD 2, 1

      const auto p32 = LoadU(df, rows.GetInputRow<0>(c) + sx + 1);
      sad0c += AbsDiff(p31, p32);  // SAD 2, 1
      t = AbsDiff(p22, p32);
      sad1c += t;  // SAD 1, 2
      sad2c += t;  // SAD 3, 2

      const auto p42 = LoadU(df, rows.GetInputRow<0>(c) + sx + 2);
      sad2c += AbsDiff(p42, p32);  // SAD 3, 2

      const auto p13 = LoadU(df, rows.GetInputRow<1>(c) + sx - 1);
      sad3c += AbsDiff(p13, p12);  // SAD 2, 3

      const auto p23 = Load(df, rows.GetInputRow<1>(c) + sx);
      t = AbsDiff(p22, p23);
      sad0c += t;                  // SAD 2, 1
      sad3c += t;                  // SAD 2, 3
      sad1c += AbsDiff(p13, p23);  // SAD 1, 2

      const auto p33 = LoadU(df, rows.GetInputRow<1>(c) + sx + 1);
      sad2c += AbsDiff(p33, p23);  // SAD 3, 2
      sad3c += AbsDiff(p33, p32);  // SAD 2, 3

      const auto p24 = Load(df, rows.GetInputRow<2>(c) + sx);
      sad3c += AbsDiff(p24, p23);  // SAD 2, 3

      auto scale = Set(df, lf.epf_channel_scale[c]);
      sad0 = MulAdd(sad0c, scale, sad0);
      sad1 = MulAdd(sad1c, scale, sad1);
      sad2 = MulAdd(sad2c, scale, sad2);
      sad3 = MulAdd(sad3c, scale, sad3);
    }
    const auto x_cc = Load(df, rows.GetInputRow<0>(0) + sx);
    const auto y_cc = Load(df, rows.GetInputRow<0>(1) + sx);
    const auto b_cc = Load(df, rows.GetInputRow<0>(2) + sx);

    auto w = Set(df, 1);
    auto X = x_cc;
    auto Y = y_cc;
    auto B = b_cc;

    // Top row
    AddPixelStep1</*aligned=*/true, /*row=*/-1>(rows, sx, sad0, sm, inv_sigma,
                                                lf, &X, &Y, &B, &w);
    // Center
    AddPixelStep1</*aligned=*/false, /*row=*/0>(rows, sx - 1, sad1, sm,
                                                inv_sigma, lf, &X, &Y, &B, &w);
    AddPixelStep1</*aligned=*/false, /*row=*/0>(rows, sx + 1, sad2, sm,
                                                inv_sigma, lf, &X, &Y, &B, &w);
    // Bottom
    AddPixelStep1</*aligned=*/true, /*row=*/1>(rows, sx, sad3, sm, inv_sigma,
                                               lf, &X, &Y, &B, &w);
    auto inv_w = Set(df, 1.0f) / w;
    Store(X * inv_w, df, rows.GetOutputRow(0) + x);
    Store(Y * inv_w, df, rows.GetOutputRow(1) + x);
    Store(B * inv_w, df, rows.GetOutputRow(2) + x);
  }
}

// Step 2: 3x3 plus-shaped kernel with a single reference pixel, ran on
// the output of the previous step.
void Epf2Row(const LoopFilterRows<1>& rows, const LoopFilter& lf, size_t iy,
             const float* JXL_RESTRICT row_sigma, size_t xsize) {
  HWY_ALIGN float sad_mul[kBlockDim] = {
      lf.epf_border_sad_mul, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
      lf.epf_border_sad_mul};

  for (size_t x = 0; x < xsize; x += Lanes(df)) {
    size_t sx = x + kBlockDim;
    size_t bx = x / kBlockDim;
    size_t ix = x % kBlockDim;

    if (row_sigma[bx + 2] < kMinSigma) {
      for (size_t c = 0; c < 3; c++) {
        auto px = Load(df, rows.GetInputRow<0>(c) + sx);
        Store(px, df, rows.GetOutputRow(c) + x);
      }
      continue;
    }

    const auto inv_sigma =
        Set(DF(), lf.epf_pass2_sigma_scale * row_sigma[bx + 2]);

    const auto sm = iy == 0 || iy == kBlockDim - 1
                        ? Set(df, lf.epf_border_sad_mul)
                        : Load(df, sad_mul + ix);

    const auto x_cc = Load(df, rows.GetInputRow<0>(0) + sx);
    const auto y_cc = Load(df, rows.GetInputRow<0>(1) + sx);
    const auto b_cc = Load(df, rows.GetInputRow<0>(2) + sx);

    auto w = Set(df, 1);
    auto X = x_cc;
    auto Y = y_cc;
    auto B = b_cc;

    // Top row
    AddPixelStep2</*aligned=*/true, /*row=*/-1>(rows, sx, x_cc, y_cc, b_cc, sm,
                                                inv_sigma, lf, &X, &Y, &B, &w);
    // Center
    AddPixelStep2</*aligned=*/false, /*row=*/0>(
        rows, sx - 1, x_cc, y_cc, b_cc, sm, inv_sigma, lf, &X, &Y, &B, &w);
    AddPixelStep2</*aligned=*/false, /*row=*/0>(
        rows, sx + 1, x_cc, y_cc, b_cc, sm, inv_sigma, lf, &X, &Y, &B, &w);
    // Bottom
    AddPixelStep2</*aligned=*/true, /*row=*/1>(rows, sx, x_cc, y_cc, b_cc, sm,
                                               inv_sigma, lf, &X, &Y, &B, &w);

    auto inv_w = Set(df, 1.0f) / w;
    Store(X * inv_w, df, rows.GetOutputRow(0) + x);
    Store(Y * inv_w, df, rows.GetOutputRow(1) + x);
    Store(B * inv_w, df, rows.GetOutputRow(2) + x);
  }
}

Status ApplyLoopFiltersRowImpl(const LoopFilter& lf, const Rect& in_rect,
                               const Image3F& in, const Rect& sigma_rect,
                               const ImageF& sigma, size_t y,
                               const float* JXL_RESTRICT gab_weights,
                               const Rect& out_rect, Image3F* JXL_RESTRICT out,
                               Image3F* JXL_RESTRICT storage1,
                               Image3F* JXL_RESTRICT storage2,
                               size_t* JXL_RESTRICT output_row) {
  PROFILER_ZONE("Gaborish+EPF");
  const size_t num_xborder_pixels = RoundUpTo(3, Lanes(df));
  size_t y_corr = in_rect.y0() % kBlockDim;
  const size_t xsize = in_rect.xsize();
  size_t gab_x0 = lf.epf ? 2 * kBlockDim - num_xborder_pixels : 2 * kBlockDim;
  size_t gab_xsize = lf.epf ? xsize + 2 * num_xborder_pixels : xsize;

  // First 2*lf.FirstStageRow() should not cause anything to run.
  if (y < 2 * kBlockDim - lf.PaddingRows() + 2 * lf.FirstStageRows())
    return false;
  if (y >= 2 * kBlockDim + in_rect.ysize() + lf.PaddingRows()) {
    return false;
  }

  // y is now the center row for the first stage.
  y -= lf.FirstStageRows();

  size_t first_epf1_row = lf.gab ? 2 * kBlockDim + 1 : 2 * kBlockDim - 1;
  size_t first_epf2_row = lf.gab ? 2 * kBlockDim + 3 : 2 * kBlockDim + 1;

  if (lf.gab) {
    LoopFilterRows<1> gab_rows;
    gab_rows.SetInput<0>(in, in_rect.y0() + y, in_rect.x0() + gab_x0);
    if (lf.epf) {
      gab_rows.SetOutput<kEpf1InputRows>(storage1, y, gab_x0);
    } else {
      gab_rows.SetOutput<0>(out, out_rect.y0() + y - 2 * kBlockDim,
                            out_rect.x0());
    }
    GaborishRow(gab_rows, gab_xsize, gab_weights);
  }

  if (!lf.epf) {
    *output_row = y - 2 * kBlockDim;
    return true;
  }
  if (y < first_epf1_row) return false;

  size_t sy = lf.gab ? y - 2 : y;
  size_t dy = sy - kBlockDim;
  LoopFilterRows<2> epf1_rows;
  if (lf.gab) {
    epf1_rows.SetInput<kEpf1InputRows>(*storage1, sy, 0);
  } else {
    epf1_rows.SetInput<0>(in, in_rect.y0() + sy, in_rect.x0());
  }
  epf1_rows.SetOutput<kEpf2InputRows>(storage2, dy, 0);
  Epf1Row(epf1_rows, lf, (sy + y_corr) % kBlockDim,
          sigma_rect.ConstRow(sigma, (sy + y_corr) / kBlockDim), xsize);
  if (y < first_epf2_row) return false;

  size_t sy2 = dy - 1;
  size_t dy2 = sy2 - kBlockDim;
  LoopFilterRows<1> epf2_rows;
  epf2_rows.SetInput<kEpf2InputRows>(*storage2, sy2, 0);
  epf2_rows.SetOutput<0>(out, out_rect.y0() + dy2, out_rect.x0());
  Epf2Row(epf2_rows, lf, (sy2 + y_corr) % kBlockDim,
          sigma_rect.ConstRow(sigma, (sy2 + y_corr) / kBlockDim + 1), xsize);
  *output_row = dy2;
  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(ApplyLoopFiltersRowImpl);  // Local function

Status ApplyLoopFiltersRow(PassesDecoderState* dec_state, const Rect& in_rect,
                           size_t y, size_t thread, Image3F* JXL_RESTRICT out,
                           size_t* JXL_RESTRICT output_row) {
  JXL_DASSERT(in_rect.x0() % kBlockDim == 0);
  const LoopFilter& lf = dec_state->shared->image_features.loop_filter;
  if (!lf.gab && !lf.epf) {
    if (y < 2 * kBlockDim || y >= 2 * kBlockDim + in_rect.ysize()) return false;
    *output_row = y - 2 * kBlockDim;
    return *output_row < dec_state->shared->frame_dim.ysize;
  }
  Rect sigma_rect(in_rect.x0() / kBlockDim, in_rect.y0() / kBlockDim,
                  DivCeil(in_rect.xsize(), kBlockDim),
                  DivCeil(in_rect.ysize(), kBlockDim));
  // TODO(janwas): hoist to caller
  return HWY_DYNAMIC_DISPATCH(ApplyLoopFiltersRowImpl)(
      lf, in_rect, dec_state->decoded, sigma_rect, dec_state->sigma, y,
      dec_state->gab_weights, in_rect, out, &dec_state->storage1[thread],
      &dec_state->storage2[thread], output_row);
}

void EdgePreservingFilter(const LoopFilter& lf, const Rect& in_rect,
                          const Image3F& in, const Rect& sigma_rect,
                          const ImageF& sigma, const Rect& out_rect,
                          Image3F* JXL_RESTRICT out,
                          Image3F* JXL_RESTRICT storage1,
                          Image3F* JXL_RESTRICT storage2) {
  JXL_ASSERT(SameSize(in_rect, out_rect));
  JXL_ASSERT(in_rect.xsize() == sigma_rect.xsize() * kBlockDim);
  JXL_ASSERT(in_rect.ysize() == sigma_rect.ysize() * kBlockDim);
  JXL_ASSERT(storage1->xsize() >= out_rect.xsize() + 4 * kBlockDim);
  JXL_ASSERT(storage1->ysize() >= kEpf1InputRows);
  JXL_ASSERT(storage2->xsize() >= out_rect.xsize() + 2 * kBlockDim);
  JXL_ASSERT(storage2->ysize() >= kEpf2InputRows);

  const size_t ysize = in_rect.ysize();

  float gab_weights[9];
  lf.GaborishWeights(gab_weights);

  for (size_t y = 2 * kBlockDim - lf.PaddingRows();
       y < ysize + 2 * kBlockDim + lf.PaddingRows(); y++) {
    size_t output_row;
    (void)HWY_DYNAMIC_DISPATCH(ApplyLoopFiltersRowImpl)(
        lf, in_rect, in, sigma_rect, sigma, y, gab_weights, out_rect, out,
        storage1, storage2, &output_row);
  }
}

}  // namespace jxl
#endif  // HWY_ONCE
