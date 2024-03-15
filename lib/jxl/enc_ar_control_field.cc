// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_ar_control_field.h"

#include <stdint.h>
#include <stdlib.h>

#include <algorithm>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_ar_control_field.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::GetLane;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::MulAdd;
using hwy::HWY_NAMESPACE::Sqrt;

Status ProcessTile(const CompressParams& cparams,
                   const FrameHeader& frame_header, const Image3F& opsin,
                   const Rect& opsin_rect, const ImageF& quant_field,
                   const AcStrategyImage& ac_strategy, ImageB* epf_sharpness,
                   const Rect& rect,
                   ArControlFieldHeuristics::TempImages* temp_image) {
  JXL_ASSERT(opsin_rect.x0() % 8 == 0);
  JXL_ASSERT(opsin_rect.y0() % 8 == 0);
  JXL_ASSERT(opsin_rect.xsize() % 8 == 0);
  JXL_ASSERT(opsin_rect.ysize() % 8 == 0);
  constexpr size_t N = kBlockDim;
  if (cparams.butteraugli_distance < kMinButteraugliForDynamicAR ||
      cparams.speed_tier > SpeedTier::kWombat ||
      frame_header.loop_filter.epf_iters == 0) {
    FillPlane(static_cast<uint8_t>(4), epf_sharpness, rect);
    return true;
  }

  // Likely better to have a higher X weight, like:
  // const float kChannelWeights[3] = {47.0f, 4.35f, 0.287f};
  const float kChannelWeights[3] = {4.35f, 4.35f, 0.287f};
  const float kChannelWeightsLapNeg[3] = {-0.125f * kChannelWeights[0],
                                          -0.125f * kChannelWeights[1],
                                          -0.125f * kChannelWeights[2]};
  const size_t sharpness_stride =
      static_cast<size_t>(epf_sharpness->PixelsPerRow());

  size_t by0 = opsin_rect.y0() / 8 + rect.y0();
  size_t by1 = by0 + rect.ysize();
  size_t bx0 = opsin_rect.x0() / 8 + rect.x0();
  size_t bx1 = bx0 + rect.xsize();
  JXL_RETURN_IF_ERROR(temp_image->InitOnce());
  ImageF& laplacian_sqrsum = temp_image->laplacian_sqrsum;
  // Calculate the L2 of the 3x3 Laplacian in an integral transform
  // (for example 32x32 dct). This relates to transforms ability
  // to propagate artefacts.
  size_t y0 = by0 == 0 ? 0 : by0 * N - 2;
  size_t y1 = by1 * N == opsin.ysize() ? by1 * N : by1 * N + 2;
  size_t x0 = bx0 == 0 ? 0 : bx0 * N - 2;
  size_t x1 = bx1 * N == opsin.xsize() ? bx1 * N : bx1 * N + 2;
  HWY_FULL(float) df;
  for (size_t y = y0; y < y1; y++) {
    float* JXL_RESTRICT laplacian_sqrsum_row =
        laplacian_sqrsum.Row(y + 2 - by0 * N);
    const float* JXL_RESTRICT in_row_t[3];
    const float* JXL_RESTRICT in_row[3];
    const float* JXL_RESTRICT in_row_b[3];
    for (size_t c = 0; c < 3; c++) {
      in_row_t[c] = opsin.ConstPlaneRow(c, y > 0 ? y - 1 : y);
      in_row[c] = opsin.ConstPlaneRow(c, y);
      in_row_b[c] = opsin.ConstPlaneRow(c, y + 1 < opsin.ysize() ? y + 1 : y);
    }
    auto compute_laplacian_scalar = [&](size_t x) {
      const size_t prevX = x >= 1 ? x - 1 : x;
      const size_t nextX = x + 1 < opsin.xsize() ? x + 1 : x;
      float sumsqr = 0;
      for (size_t c = 0; c < 3; c++) {
        float laplacian =
            kChannelWeights[c] * in_row[c][x] +
            kChannelWeightsLapNeg[c] *
                (in_row[c][prevX] + in_row[c][nextX] + in_row_b[c][prevX] +
                 in_row_b[c][x] + in_row_b[c][nextX] + in_row_t[c][prevX] +
                 in_row_t[c][x] + in_row_t[c][nextX]);
        sumsqr += laplacian * laplacian;
      }
      laplacian_sqrsum_row[x + 2 - bx0 * N] = sumsqr;
    };
    size_t x = x0;
    for (; x < 1; x++) {
      compute_laplacian_scalar(x);
    }
    // Interior. One extra pixel of border as the last pixel is special.
    for (; x + Lanes(df) <= x1 && x + Lanes(df) + 1 <= opsin.xsize();
         x += Lanes(df)) {
      auto sumsqr = Zero(df);
      for (size_t c = 0; c < 3; c++) {
        auto laplacian =
            Mul(LoadU(df, in_row[c] + x), Set(df, kChannelWeights[c]));
        auto sum_oth0 = LoadU(df, in_row[c] + x - 1);
        auto sum_oth1 = LoadU(df, in_row[c] + x + 1);
        auto sum_oth2 = LoadU(df, in_row_t[c] + x - 1);
        auto sum_oth3 = LoadU(df, in_row_t[c] + x);
        sum_oth0 = Add(sum_oth0, LoadU(df, in_row_t[c] + x + 1));
        sum_oth1 = Add(sum_oth1, LoadU(df, in_row_b[c] + x - 1));
        sum_oth2 = Add(sum_oth2, LoadU(df, in_row_b[c] + x));
        sum_oth3 = Add(sum_oth3, LoadU(df, in_row_b[c] + x + 1));
        sum_oth0 = Add(sum_oth0, sum_oth1);
        sum_oth2 = Add(sum_oth2, sum_oth3);
        sum_oth0 = Add(sum_oth0, sum_oth2);
        laplacian =
            MulAdd(Set(df, kChannelWeightsLapNeg[c]), sum_oth0, laplacian);
        sumsqr = MulAdd(laplacian, laplacian, sumsqr);
      }
      StoreU(sumsqr, df, laplacian_sqrsum_row + x + 2 - bx0 * N);
    }
    for (; x < x1; x++) {
      compute_laplacian_scalar(x);
    }
  }
  HWY_CAPPED(float, 4) df4;
  // Calculate the L2 of the 3x3 Laplacian in 4x4 blocks within the area
  // of the integral transform. Sample them within the integral transform
  // with two offsets (0,0) and (-2, -2) pixels (sqrsum_00 and sqrsum_22,
  //  respectively).
  ImageF& sqrsum_00 = temp_image->sqrsum_00;
  size_t sqrsum_00_stride = sqrsum_00.PixelsPerRow();
  float* JXL_RESTRICT sqrsum_00_row = sqrsum_00.Row(0);
  for (size_t y = 0; y < rect.ysize() * 2; y++) {
    const float* JXL_RESTRICT rows_in[4];
    for (size_t iy = 0; iy < 4; iy++) {
      rows_in[iy] = laplacian_sqrsum.ConstRow(y * 4 + iy + 2);
    }
    float* JXL_RESTRICT row_out = sqrsum_00_row + y * sqrsum_00_stride;
    for (size_t x = 0; x < rect.xsize() * 2; x++) {
      auto sum = Zero(df4);
      for (size_t iy = 0; iy < 4; iy++) {
        for (size_t ix = 0; ix < 4; ix += Lanes(df4)) {
          sum = Add(sum, LoadU(df4, rows_in[iy] + x * 4 + ix + 2));
        }
      }
      row_out[x] = GetLane(Sqrt(SumOfLanes(df4, sum))) * (1.0f / 4.0f);
    }
  }
  // Indexing iy and ix is a bit tricky as we include a 2 pixel border
  // around the block for evenness calculations. This is similar to what
  // we did in guetzli for the observability of artefacts, except there
  // the element is a sliding 5x5, not sparsely sampled 4x4 box like here.
  ImageF& sqrsum_22 = temp_image->sqrsum_22;
  size_t sqrsum_22_stride = sqrsum_22.PixelsPerRow();
  float* JXL_RESTRICT sqrsum_22_row = sqrsum_22.Row(0);
  for (size_t y = 0; y < rect.ysize() * 2 + 1; y++) {
    const float* JXL_RESTRICT rows_in[4];
    for (size_t iy = 0; iy < 4; iy++) {
      rows_in[iy] = laplacian_sqrsum.ConstRow(y * 4 + iy);
    }
    float* JXL_RESTRICT row_out = sqrsum_22_row + y * sqrsum_22_stride;
    // ignore pixels outside the image.
    // Y coordinates are relative to by0*8+y*4.
    size_t sy = y * 4 + by0 * 8 > 0 ? 0 : 2;
    size_t ey = y * 4 + by0 * 8 + 2 <= opsin.ysize()
                    ? 4
                    : opsin.ysize() - y * 4 - by0 * 8 + 2;
    for (size_t x = 0; x < rect.xsize() * 2 + 1; x++) {
      // ignore pixels outside the image.
      // X coordinates are relative to bx0*8.
      size_t sx = x * 4 + bx0 * 8 > 0 ? x * 4 : x * 4 + 2;
      size_t ex = x * 4 + bx0 * 8 + 2 <= opsin.xsize()
                      ? x * 4 + 4
                      : opsin.xsize() - bx0 * 8 + 2;
      if (ex - sx == 4 && ey - sy == 4) {
        auto sum = Zero(df4);
        for (size_t iy = sy; iy < ey; iy++) {
          for (size_t ix = sx; ix < ex; ix += Lanes(df4)) {
            sum = Add(sum, Load(df4, rows_in[iy] + ix));
          }
        }
        row_out[x] = GetLane(Sqrt(SumOfLanes(df4, sum))) * (1.0f / 4.0f);
      } else {
        float sum = 0;
        for (size_t iy = sy; iy < ey; iy++) {
          for (size_t ix = sx; ix < ex; ix++) {
            sum += rows_in[iy][ix];
          }
        }
        row_out[x] = std::sqrt(sum / ((ex - sx) * (ey - sy)));
      }
    }
  }
  for (size_t by = rect.y0(); by < rect.y1(); by++) {
    AcStrategyRow acs_row = ac_strategy.ConstRow(by);
    uint8_t* JXL_RESTRICT out_row = epf_sharpness->Row(by);
    const float* JXL_RESTRICT quant_row = quant_field.Row(by);
    for (size_t bx = rect.x0(); bx < rect.x1(); bx++) {
      AcStrategy acs = acs_row[bx];
      if (!acs.IsFirstBlock()) continue;
      // The errors are going to be linear to the quantization value in this
      // locality. We only have access to the initial quant field here.
      float quant_val = 1.0f / quant_row[bx];

      const auto sq00 = [&](size_t y, size_t x) {
        return sqrsum_00_row[((by - rect.y0()) * 2 + y) * sqrsum_00_stride +
                             (bx - rect.x0()) * 2 + x];
      };
      const auto sq22 = [&](size_t y, size_t x) {
        return sqrsum_22_row[((by - rect.y0()) * 2 + y) * sqrsum_22_stride +
                             (bx - rect.x0()) * 2 + x];
      };
      float sqrsum_integral_transform = 0;
      for (size_t iy = 0; iy < acs.covered_blocks_y() * 2; iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x() * 2; ix++) {
          sqrsum_integral_transform += sq00(iy, ix) * sq00(iy, ix);
        }
      }
      sqrsum_integral_transform /=
          4 * acs.covered_blocks_x() * acs.covered_blocks_y();
      sqrsum_integral_transform = std::sqrt(sqrsum_integral_transform);
      // If masking is high or amplitude of the artefacts is low, then no
      // smoothing is needed.
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          // Five 4x4 blocks for masking estimation, all within the
          // 8x8 area.
          float minval_1 = std::min(sq00(2 * iy + 0, 2 * ix + 0),
                                    sq00(2 * iy + 0, 2 * ix + 1));
          float minval_2 = std::min(sq00(2 * iy + 1, 2 * ix + 0),
                                    sq00(2 * iy + 1, 2 * ix + 1));
          float minval = std::min(minval_1, minval_2);
          minval = std::min(minval, sq22(2 * iy + 1, 2 * ix + 1));
          // Nine more 4x4 blocks for masking estimation, includes
          // the 2 pixel area around the 8x8 block being controlled.
          float minval2_1 = std::min(sq22(2 * iy + 0, 2 * ix + 0),
                                     sq22(2 * iy + 0, 2 * ix + 1));
          float minval2_2 = std::min(sq22(2 * iy + 0, 2 * ix + 2),
                                     sq22(2 * iy + 1, 2 * ix + 0));
          float minval2_3 = std::min(sq22(2 * iy + 1, 2 * ix + 1),
                                     sq22(2 * iy + 1, 2 * ix + 2));
          float minval2_4 = std::min(sq22(2 * iy + 2, 2 * ix + 0),
                                     sq22(2 * iy + 2, 2 * ix + 1));
          float minval2_5 = std::min(minval2_1, minval2_2);
          float minval2_6 = std::min(minval2_3, minval2_4);
          float minval2 = std::min(minval2_5, minval2_6);
          minval2 = std::min(minval2, sq22(2 * iy + 2, 2 * ix + 2));
          float minval3 = std::min(minval, minval2);
          minval *= 0.125f;
          minval += 0.625f * minval3;
          minval +=
              0.125f * std::min(1.5f * minval3, sq22(2 * iy + 1, 2 * ix + 1));
          minval += 0.125f * minval2;
          // Larger kBias, less smoothing for low intensity changes.
          float kDeltaLimit = 3.2;
          float bias = 0.0625f * quant_val;
          float delta =
              (sqrsum_integral_transform + (kDeltaLimit + 0.05) * bias) /
              (minval + bias);
          int out = 4;
          if (delta > kDeltaLimit) {
            out = 4;  // smooth
          } else {
            out = 0;
          }
          // 'threshold' is separate from 'bias' for easier tuning of these
          // heuristics.
          float threshold = 0.0625f * quant_val;
          const float kSmoothLimit = 0.085f;
          float smooth = 0.20f * (sq00(2 * iy + 0, 2 * ix + 0) +
                                  sq00(2 * iy + 0, 2 * ix + 1) +
                                  sq00(2 * iy + 1, 2 * ix + 0) +
                                  sq00(2 * iy + 1, 2 * ix + 1) + minval);
          if (smooth < kSmoothLimit * threshold) {
            out = 4;
          }
          out_row[bx + sharpness_stride * iy + ix] = out;
        }
      }
    }
  }
  return true;
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(ProcessTile);

Status ArControlFieldHeuristics::RunRect(
    const CompressParams& cparams, const FrameHeader& frame_header,
    const Rect& block_rect, const Image3F& opsin, const Rect& opsin_rect,
    const ImageF& quant_field, const AcStrategyImage& ac_strategy,
    ImageB* epf_sharpness, size_t thread) {
  return HWY_DYNAMIC_DISPATCH(ProcessTile)(
      cparams, frame_header, opsin, opsin_rect, quant_field, ac_strategy,
      epf_sharpness, block_rect, &temp_images[thread]);
}

}  // namespace jxl

#endif
