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

// Edge-preserving smoothing: 7x8 weighted average based on L1 patch similarity.

#include "jxl/epf.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <hwy/static_targets.h>
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

namespace jxl {

using DF = HWY_CAPPED(float, 8);
using DU = HWY_CAPPED(uint32_t, 8);

using hwy::ext::AbsDiff;

// kInvSigmaNum / 0.3
constexpr float kMinSigma = -3.90524291751269967465540850526868f;

DF df;

HWY_ATTR JXL_INLINE hwy::VT<DF> Weight(hwy::VT<DF> sad, hwy::VT<DF> inv_sigma,
                                       hwy::VT<DF> thres) {
  auto v = MulAdd(sad, inv_sigma, Set(DF(), 1.0f));
  auto v2 = v * v;
  return hwy::IfThenZeroElse(v <= thres, v2);
}

template <bool aligned>
HWY_ATTR JXL_INLINE void AddPixelStep1(
    const float* JXL_RESTRICT* JXL_RESTRICT rows, size_t offset,
    hwy::VT<DF> sad, size_t sad_id, hwy::VT<DF> sad_mul, hwy::VT<DF> inv_sigma,
    const LoopFilter& lf, hwy::VT<DF>* JXL_RESTRICT X,
    hwy::VT<DF>* JXL_RESTRICT Y, hwy::VT<DF>* JXL_RESTRICT B,
    hwy::VT<DF>* JXL_RESTRICT w) {
  auto cx =
      aligned ? Load(DF(), rows[0] + offset) : LoadU(DF(), rows[0] + offset);
  auto cy =
      aligned ? Load(DF(), rows[1] + offset) : LoadU(DF(), rows[1] + offset);
  auto cb =
      aligned ? Load(DF(), rows[2] + offset) : LoadU(DF(), rows[2] + offset);

  auto weight =
      Weight(sad * sad_mul, inv_sigma, Set(df, lf.epf_pass1_zeroflush));
  *w += weight;
  *X = MulAdd(weight, cx, *X);
  *Y = MulAdd(weight, cy, *Y);
  *B = MulAdd(weight, cb, *B);
}

template <bool aligned>
HWY_ATTR JXL_INLINE void AddPixelStep2(
    float* JXL_RESTRICT* JXL_RESTRICT rows, size_t offset, hwy::VT<DF> rx,
    hwy::VT<DF> ry, hwy::VT<DF> rb, hwy::VT<DF> sad_mul, hwy::VT<DF> inv_sigma,
    const LoopFilter& lf, hwy::VT<DF>* JXL_RESTRICT X,
    hwy::VT<DF>* JXL_RESTRICT Y, hwy::VT<DF>* JXL_RESTRICT B,
    hwy::VT<DF>* JXL_RESTRICT w) {
  auto cx =
      aligned ? Load(DF(), rows[0] + offset) : LoadU(DF(), rows[0] + offset);
  auto cy =
      aligned ? Load(DF(), rows[1] + offset) : LoadU(DF(), rows[1] + offset);
  auto cb =
      aligned ? Load(DF(), rows[2] + offset) : LoadU(DF(), rows[2] + offset);

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

HWY_ATTR void GaborishRow(const float* JXL_RESTRICT* JXL_RESTRICT rows_in,
                          size_t in_stride, size_t y_in, size_t x0_in,
                          float* JXL_RESTRICT* JXL_RESTRICT rows_out,
                          size_t out_stride, size_t y_out, size_t x0_out,
                          size_t xsize, const float* JXL_RESTRICT gab_weights) {
  for (size_t c = 0; c < 3; c++) {
    const float* JXL_RESTRICT row_in = rows_in[c] + in_stride * y_in + x0_in;
    float* JXL_RESTRICT row_out = rows_out[c] + out_stride * y_out + x0_out;
    const auto w0 = Set(df, gab_weights[3 * c]);
    const auto w1 = Set(df, gab_weights[3 * c + 1]);
    const auto w2 = Set(df, gab_weights[3 * c + 2]);
    for (size_t ix = 0; ix < xsize; ix += df.N) {
      const auto t = Load(df, row_in - in_stride + ix);
      const auto tl = LoadU(df, row_in - in_stride + ix - 1);
      const auto tr = LoadU(df, row_in - in_stride + ix + 1);
      const auto c = Load(df, row_in + ix);
      const auto l = LoadU(df, row_in + ix - 1);
      const auto r = LoadU(df, row_in + ix + 1);
      const auto b = Load(df, row_in + in_stride + ix);
      const auto bl = LoadU(df, row_in + in_stride + ix - 1);
      const auto br = LoadU(df, row_in + in_stride + ix + 1);
      const auto sum0 = c;
      const auto sum1 = l + r + t + b;
      const auto sum2 = tl + tr + bl + br;
      auto pixels = MulAdd(sum2, w2, MulAdd(sum1, w1, sum0 * w0));
      Store(pixels, df, row_out + ix);
    }
  }
}

// Step 1: 3x3 plus-shaped kernel with 5 SADs per pixel (also 3x3
// plus-shaped).
HWY_ATTR void Epf1Row(const LoopFilter& lf,
                      const float* JXL_RESTRICT* JXL_RESTRICT rows_in,
                      size_t in_stride, size_t y_in, size_t iy,
                      const float* JXL_RESTRICT rows_sigma, size_t sigma_stride,
                      size_t by_sigma,
                      float* JXL_RESTRICT* JXL_RESTRICT rows_out,
                      size_t out_stride, size_t y_out, size_t xsize) {
  size_t ym2 = y_in < 2 ? y_in + kEpf1InputRows - 2 : y_in - 2;
  size_t ym1 = y_in < 1 ? y_in + kEpf1InputRows - 1 : y_in - 1;
  size_t yc = y_in;
  size_t yp1 = y_in > kEpf1InputRows - 2 ? y_in + 1 - kEpf1InputRows : y_in + 1;
  size_t yp2 = y_in > kEpf1InputRows - 3 ? y_in + 2 - kEpf1InputRows : y_in + 2;
  y_out %= kEpf2InputRows;

  HWY_ALIGN float sad_mul[kBlockDim] = {
      lf.epf_border_sad_mul, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
      lf.epf_border_sad_mul};

  for (size_t x = kBlockDim - df.N; x < xsize + kBlockDim + df.N; x += df.N) {
    size_t bx = x / kBlockDim;
    size_t ix = x - kBlockDim * bx;
    size_t sbx = bx + 1;
    if (rows_sigma[by_sigma * sigma_stride + sbx] < kMinSigma) {
      size_t dest_offset = y_out * out_stride + bx * kBlockDim + ix;
      size_t source_offsetc = yc * in_stride + sbx * kBlockDim + ix;
      for (size_t c = 0; c < 3; c++) {
        auto px = Load(df, rows_in[c] + source_offsetc);
        Store(px, df, rows_out[c] + dest_offset);
      }
      continue;
    }
    const auto inv_sigma = Set(DF(), rows_sigma[by_sigma * sigma_stride + sbx]);

    const auto sm = iy == 0 || iy == kBlockDim - 1
                        ? Set(df, lf.epf_border_sad_mul)
                        : Load(df, sad_mul + ix);
    size_t source_offsetm2 = ym2 * in_stride + sbx * kBlockDim + ix;
    size_t source_offsetm1 = ym1 * in_stride + sbx * kBlockDim + ix;
    size_t source_offsetc = yc * in_stride + sbx * kBlockDim + ix;
    size_t source_offsetp1 = yp1 * in_stride + sbx * kBlockDim + ix;
    size_t source_offsetp2 = yp2 * in_stride + sbx * kBlockDim + ix;
    size_t dest_offset = y_out * out_stride + bx * kBlockDim + ix;

    auto sad0 = Zero(df);
    auto sad1 = Zero(df);
    auto sad2 = Zero(df);
    auto sad3 = Zero(df);

    // compute sads
    for (size_t c = 0; c < 3; c++) {
      // center px = 22, px above = 21
      auto t = Undefined(df);

      const auto p20 = Load(df, rows_in[c] + source_offsetm2);
      const auto p21 = Load(df, rows_in[c] + source_offsetm1);
      auto sad0c = AbsDiff(p20, p21);  // SAD 2, 1

      const auto p11 = LoadU(df, rows_in[c] + source_offsetm1 - 1);
      auto sad1c = AbsDiff(p11, p21);  // SAD 1, 2

      const auto p31 = LoadU(df, rows_in[c] + source_offsetm1 + 1);
      auto sad2c = AbsDiff(p31, p21);  // SAD 3, 2

      const auto p02 = LoadU(df, rows_in[c] + source_offsetc - 2);
      const auto p12 = LoadU(df, rows_in[c] + source_offsetc - 1);
      sad1c += AbsDiff(p02, p12);  // SAD 1, 2
      sad0c += AbsDiff(p11, p12);  // SAD 2, 1

      const auto p22 = LoadU(df, rows_in[c] + source_offsetc);
      t = AbsDiff(p12, p22);
      sad1c += t;  // SAD 1, 2
      sad2c += t;  // SAD 3, 2
      t = AbsDiff(p22, p21);
      auto sad3c = t;  // SAD 2, 3
      sad0c += t;      // SAD 2, 1

      const auto p32 = LoadU(df, rows_in[c] + source_offsetc + 1);
      sad0c += AbsDiff(p31, p32);  // SAD 2, 1
      t = AbsDiff(p22, p32);
      sad1c += t;  // SAD 1, 2
      sad2c += t;  // SAD 3, 2

      const auto p42 = LoadU(df, rows_in[c] + source_offsetc + 2);
      sad2c += AbsDiff(p42, p32);  // SAD 3, 2

      const auto p13 = LoadU(df, rows_in[c] + source_offsetp1 - 1);
      sad3c += AbsDiff(p13, p12);  // SAD 2, 3

      const auto p23 = Load(df, rows_in[c] + source_offsetp1);
      t = AbsDiff(p22, p23);
      sad0c += t;                  // SAD 2, 1
      sad3c += t;                  // SAD 2, 3
      sad1c += AbsDiff(p13, p23);  // SAD 1, 2

      const auto p33 = LoadU(df, rows_in[c] + source_offsetp1 + 1);
      sad2c += AbsDiff(p33, p23);  // SAD 3, 2
      sad3c += AbsDiff(p33, p32);  // SAD 2, 3

      const auto p24 = Load(df, rows_in[c] + source_offsetp2);
      sad3c += AbsDiff(p24, p23);  // SAD 2, 3

      auto scale = Set(df, lf.epf_channel_scale[c]);
      sad0 = MulAdd(sad0c, scale, sad0);
      sad1 = MulAdd(sad1c, scale, sad1);
      sad2 = MulAdd(sad2c, scale, sad2);
      sad3 = MulAdd(sad3c, scale, sad3);
    }
    const auto x_cc = Load(df, rows_in[0] + source_offsetc);
    const auto y_cc = Load(df, rows_in[1] + source_offsetc);
    const auto b_cc = Load(df, rows_in[2] + source_offsetc);

    auto w = Set(df, 1);
    auto X = x_cc;
    auto Y = y_cc;
    auto B = b_cc;

    // Top row
    AddPixelStep1</*aligned=*/true>(rows_in, source_offsetm1, sad0, 0, sm,
                                    inv_sigma, lf, &X, &Y, &B, &w);
    // Center
    AddPixelStep1</*aligned=*/false>(rows_in, source_offsetc - 1, sad1, 1, sm,
                                     inv_sigma, lf, &X, &Y, &B, &w);
    AddPixelStep1</*aligned=*/false>(rows_in, source_offsetc + 1, sad2, 2, sm,
                                     inv_sigma, lf, &X, &Y, &B, &w);
    // Bottom
    AddPixelStep1</*aligned=*/true>(rows_in, source_offsetp1, sad3, 3, sm,
                                    inv_sigma, lf, &X, &Y, &B, &w);
    auto inv_w = Set(df, 1.0f) / w;
    Store(X * inv_w, df, rows_out[0] + dest_offset);
    Store(Y * inv_w, df, rows_out[1] + dest_offset);
    Store(B * inv_w, df, rows_out[2] + dest_offset);
  }
}

// Step 2: 3x3 plus-shaped kernel with a single reference pixel, ran on
// the output of the previous step.
HWY_ATTR void Epf2Row(const LoopFilter& lf,
                      float* JXL_RESTRICT* JXL_RESTRICT rows_in,
                      size_t in_stride, size_t y_in,
                      const float* JXL_RESTRICT rows_sigma, size_t sigma_stride,
                      size_t by_sigma,
                      float* JXL_RESTRICT* JXL_RESTRICT rows_out,
                      size_t out_stride, size_t y_out, size_t xsize) {
  HWY_ALIGN float sad_mul[kBlockDim] = {
      lf.epf_border_sad_mul, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
      lf.epf_border_sad_mul};

  size_t iy = y_in % kBlockDim;
  size_t ym1 = (y_in - 1) % kEpf2InputRows;
  size_t yc = y_in % kEpf2InputRows;
  size_t yp1 = (y_in + 1) % kEpf2InputRows;

  for (size_t x = 0; x < xsize; x += df.N) {
    size_t bx = x / kBlockDim;
    size_t ix = x - kBlockDim * bx;
    size_t dbx = bx;
    size_t obx = bx + 2;

    if (rows_sigma[by_sigma * sigma_stride + obx] < kMinSigma) {
      size_t dest_offset = y_out * out_stride + dbx * kBlockDim + ix;
      size_t source_offsetc = yc * in_stride + (bx + 1) * kBlockDim + ix;
      for (size_t c = 0; c < 3; c++) {
        auto px = Load(df, rows_in[c] + source_offsetc);
        Store(px, df, rows_out[c] + dest_offset);
      }
      continue;
    }

    const auto inv_sigma =
        Set(DF(), lf.epf_pass2_sigma_scale *
                      rows_sigma[by_sigma * sigma_stride + obx]);

    const auto sm = iy == 0 || iy == kBlockDim - 1
                        ? Set(df, lf.epf_border_sad_mul)
                        : Load(df, sad_mul + ix);
    size_t source_offsetm1 = ym1 * in_stride + (bx + 1) * kBlockDim + ix;
    size_t source_offsetc = yc * in_stride + (bx + 1) * kBlockDim + ix;
    size_t source_offsetp1 = yp1 * in_stride + (bx + 1) * kBlockDim + ix;
    size_t dest_offset = y_out * out_stride + dbx * kBlockDim + ix;

    const auto x_cc = Load(df, rows_in[0] + source_offsetc);
    const auto y_cc = Load(df, rows_in[1] + source_offsetc);
    const auto b_cc = Load(df, rows_in[2] + source_offsetc);

    auto w = Set(df, 1);
    auto X = x_cc;
    auto Y = y_cc;
    auto B = b_cc;

    // Top row
    AddPixelStep2</*aligned=*/true>(rows_in, source_offsetm1, x_cc, y_cc, b_cc,
                                    sm, inv_sigma, lf, &X, &Y, &B, &w);
    // Center
    AddPixelStep2</*aligned=*/false>(rows_in, source_offsetc - 1, x_cc, y_cc,
                                     b_cc, sm, inv_sigma, lf, &X, &Y, &B, &w);
    AddPixelStep2</*aligned=*/false>(rows_in, source_offsetc + 1, x_cc, y_cc,
                                     b_cc, sm, inv_sigma, lf, &X, &Y, &B, &w);
    // Bottom
    AddPixelStep2</*aligned=*/true>(rows_in, source_offsetp1, x_cc, y_cc, b_cc,
                                    sm, inv_sigma, lf, &X, &Y, &B, &w);

    auto inv_w = Set(df, 1.0f) / w;
    Store(X * inv_w, df, rows_out[0] + dest_offset);
    Store(Y * inv_w, df, rows_out[1] + dest_offset);
    Store(B * inv_w, df, rows_out[2] + dest_offset);
  }
}

HWY_ATTR void EdgePreservingFilter(const LoopFilter& lf, const Rect& in_rect,
                                   const Image3F& in, const Rect& sigma_rect,
                                   const ImageF& sigma, const Rect& out_rect,
                                   Image3F* JXL_RESTRICT out,
                                   Image3F* JXL_RESTRICT storage1,
                                   Image3F* JXL_RESTRICT storage2) {
  JXL_CHECK(SameSize(in_rect, out_rect));
  JXL_CHECK(in_rect.xsize() == sigma_rect.xsize() * kBlockDim);
  JXL_CHECK(in_rect.ysize() == sigma_rect.ysize() * kBlockDim);
  JXL_CHECK(storage1->xsize() >= out_rect.xsize() + 4 * kBlockDim);
  JXL_CHECK(storage1->ysize() >= kEpf1InputRows);
  JXL_CHECK(storage2->xsize() >= out_rect.xsize() + 2 * kBlockDim);
  JXL_CHECK(storage2->ysize() >= kEpf2InputRows);

  const size_t xsize = in_rect.xsize();
  const size_t ysize = in_rect.ysize();

  float gab_weights[9] = {
      1, lf.gab_x_weight1, lf.gab_x_weight2,
      1, lf.gab_y_weight1, lf.gab_y_weight2,
      1, lf.gab_b_weight1, lf.gab_b_weight2,
  };
  // Normalize
  for (size_t c = 0; c < 3; c++) {
    const float mul =
        1.0f / (gab_weights[3 * c] +
                4 * (gab_weights[3 * c + 1] + gab_weights[3 * c + 2]));
    gab_weights[3 * c] *= mul;
    gab_weights[3 * c + 1] *= mul;
    gab_weights[3 * c + 2] *= mul;
  }

  const float* JXL_RESTRICT rows_sigma =
      lf.epf ? sigma_rect.ConstRow(sigma, 0) : nullptr;
  const size_t sigma_stride = sigma.PixelsPerRow();

  // Simple case first: gaborish only, no EPF.
  if (lf.gab && !lf.epf) {
    PROFILER_ZONE("Gaborish");
    const float* JXL_RESTRICT rows_in[3] = {
        in_rect.ConstPlaneRow(in, 0, 2 * kBlockDim) + 2 * kBlockDim,
        in_rect.ConstPlaneRow(in, 1, 2 * kBlockDim) + 2 * kBlockDim,
        in_rect.ConstPlaneRow(in, 2, 2 * kBlockDim) + 2 * kBlockDim,
    };
    const size_t in_stride = in.PixelsPerRow();

    float* JXL_RESTRICT rows_out[3] = {
        out_rect.PlaneRow(out, 0, 0),
        out_rect.PlaneRow(out, 1, 0),
        out_rect.PlaneRow(out, 2, 0),
    };
    const size_t out_stride = out->PixelsPerRow();

    for (size_t iy = 0; iy < ysize; iy++) {
      GaborishRow(rows_in, in_stride, iy, 0, rows_out, out_stride, iy, 0, xsize,
                  gab_weights);
    }
    return;
  }
  // EPF only, no gaborish.
  if (lf.epf && !lf.gab) {
    PROFILER_ZONE("EPF");

    const float* JXL_RESTRICT rows_in[3] = {
        in_rect.ConstPlaneRow(in, 0, 0),
        in_rect.ConstPlaneRow(in, 1, 0),
        in_rect.ConstPlaneRow(in, 2, 0),
    };
    const size_t in_stride = in.PixelsPerRow();

    float* JXL_RESTRICT rows_storage[3] = {storage2->PlaneRow(0, 0),
                                           storage2->PlaneRow(1, 0),
                                           storage2->PlaneRow(2, 0)};
    const size_t storage_stride = storage2->PixelsPerRow();
    float* JXL_RESTRICT rows_out[3] = {
        out_rect.PlaneRow(out, 0, 0),
        out_rect.PlaneRow(out, 1, 0),
        out_rect.PlaneRow(out, 2, 0),
    };
    const size_t out_stride = out->PixelsPerRow();

    // First two rows.
    Epf1Row(lf, rows_in, in_stride, 2 * kBlockDim - 1, kBlockDim - 1,
            rows_sigma, sigma_stride, 1, rows_storage, storage_stride,
            kBlockDim - 1, xsize);
    Epf1Row(lf, rows_in, in_stride, 2 * kBlockDim, 0, rows_sigma, sigma_stride,
            2, rows_storage, storage_stride, kBlockDim, xsize);

    for (size_t y = 0; y < ysize; y++) {
      size_t sy = y + 2 * kBlockDim + 1;
      size_t dy = y + kBlockDim + 1;
      size_t sby = sy / kBlockDim;
      Epf1Row(lf, rows_in, in_stride, sy, sy % kBlockDim, rows_sigma,
              sigma_stride, sby, rows_storage, storage_stride, dy, xsize);
      size_t sy2 = y + kBlockDim;
      size_t dy2 = y;
      size_t sby2 = y / kBlockDim + 2;
      Epf2Row(lf, rows_storage, storage_stride, sy2, rows_sigma, sigma_stride,
              sby2, rows_out, out_stride, dy2, xsize);
    }
    return;
  }

  // Most complex case: EPF and gaborish.
  PROFILER_ZONE("Gaborish+EPF");

  // Storage areas.
  const float* JXL_RESTRICT rows_in[3] = {
      in_rect.ConstPlaneRow(in, 0, 0),
      in_rect.ConstPlaneRow(in, 1, 0),
      in_rect.ConstPlaneRow(in, 2, 0),
  };
  const size_t in_stride = in.PixelsPerRow();

  float* JXL_RESTRICT rows_storage1[3] = {
      storage1->PlaneRow(0, 0),
      storage1->PlaneRow(1, 0),
      storage1->PlaneRow(2, 0),
  };
  const size_t storage1_stride = storage1->PixelsPerRow();

  float* JXL_RESTRICT rows_storage2[3] = {storage2->PlaneRow(0, 0),
                                          storage2->PlaneRow(1, 0),
                                          storage2->PlaneRow(2, 0)};
  const size_t storage2_stride = storage2->PixelsPerRow();

  float* JXL_RESTRICT rows_out[3] = {
      out_rect.PlaneRow(out, 0, 0),
      out_rect.PlaneRow(out, 1, 0),
      out_rect.PlaneRow(out, 2, 0),
  };
  const size_t out_stride = out->PixelsPerRow();

  constexpr size_t kNumXBorderPixels = DivCeil(3, df.N) * df.N;
  constexpr size_t x0 = 2 * kBlockDim - kNumXBorderPixels;

  // First rows.
  for (size_t iy = kBlockDim + 5; iy < kBlockDim + 10; iy++) {
    GaborishRow(rows_in, in_stride, iy, x0, rows_storage1, storage1_stride,
                iy % kEpf1InputRows, x0, xsize + 2 * kNumXBorderPixels,
                gab_weights);
  }
  Epf1Row(lf,
          const_cast<const float * JXL_RESTRICT * JXL_RESTRICT>(rows_storage1),
          storage1_stride, (2 * kBlockDim - 1) % kEpf1InputRows, kBlockDim - 1,
          rows_sigma, sigma_stride, 1, rows_storage2, storage2_stride,
          kBlockDim - 1, xsize);

  GaborishRow(rows_in, in_stride, kBlockDim + 10, x0, rows_storage1,
              storage1_stride, (kBlockDim + 10) % kEpf1InputRows, x0,
              xsize + 2 * kNumXBorderPixels, gab_weights);
  Epf1Row(lf,
          const_cast<const float * JXL_RESTRICT * JXL_RESTRICT>(rows_storage1),
          storage1_stride, (2 * kBlockDim) % kEpf1InputRows, 0, rows_sigma,
          sigma_stride, 2, rows_storage2, storage2_stride, kBlockDim, xsize);

  for (size_t y = 0; y < ysize; y++) {
    size_t sy_gab = y + kBlockDim + 11;
    size_t dy_gab = y + kBlockDim + 11;
    GaborishRow(rows_in, in_stride, sy_gab, x0, rows_storage1, storage1_stride,
                dy_gab % kEpf1InputRows, x0, xsize + 2 * kNumXBorderPixels,
                gab_weights);
    size_t sy = y + 2 * kBlockDim + 1;
    size_t dy = y + kBlockDim + 1;
    size_t sby = sy / kBlockDim;
    Epf1Row(
        lf,
        const_cast<const float * JXL_RESTRICT * JXL_RESTRICT>(rows_storage1),
        storage1_stride, sy % kEpf1InputRows, sy % kBlockDim, rows_sigma,
        sigma_stride, sby, rows_storage2, storage2_stride, dy, xsize);
    size_t sy2 = y + kBlockDim;
    size_t dy2 = y;
    size_t sby2 = y / kBlockDim + 2;
    Epf2Row(lf, rows_storage2, storage2_stride, sy2, rows_sigma, sigma_stride,
            sby2, rows_out, out_stride, dy2, xsize);
  }
}

}  // namespace jxl
