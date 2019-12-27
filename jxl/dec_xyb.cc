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

#include "jxl/dec_xyb.h"

#include <string.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/fields.h"
#include "jxl/image.h"
#include "jxl/opsin_params.h"
#include "jxl/quantizer.h"

namespace jxl {

void OpsinParams::Init() {
  InitSIMDInverseMatrix(GetOpsinAbsorbanceInverseMatrix(),
                        inverse_opsin_matrix);
  memcpy(opsin_biases, kNegOpsinAbsorbanceBiasRGB,
         sizeof(kNegOpsinAbsorbanceBiasRGB));
  memcpy(quant_biases, kDefaultQuantBias, sizeof(kDefaultQuantBias));
  for (size_t c = 0; c < 4; c++) {
    opsin_biases_cbrt[c] = std::cbrt(opsin_biases[c]);
  }
}

// In-place.
HWY_ATTR void OpsinToLinear(Image3F* JXL_RESTRICT inout, ThreadPool* pool,
                            const OpsinParams& opsin_params) {
  PROFILER_FUNC;

  const size_t xsize = inout->xsize();  // not padded
  RunOnPool(
      pool, 0, inout->ysize(), ThreadPool::SkipInit(),
      [&](const int task, const int thread) HWY_ATTR {
        const size_t y = task;

        // Faster than adding via ByteOffset at end of loop.
        float* JXL_RESTRICT row0 = inout->PlaneRow(0, y);
        float* JXL_RESTRICT row1 = inout->PlaneRow(1, y);
        float* JXL_RESTRICT row2 = inout->PlaneRow(2, y);

        const HWY_FULL(float) d;

        for (size_t x = 0; x < xsize; x += d.N) {
          const auto in_opsin_x = Load(d, row0 + x);
          const auto in_opsin_y = Load(d, row1 + x);
          const auto in_opsin_b = Load(d, row2 + x);
          JXL_COMPILER_FENCE;
          auto linear_r = Undefined(d);
          auto linear_g = Undefined(d);
          auto linear_b = Undefined(d);
          XybToRgb(d, in_opsin_x, in_opsin_y, in_opsin_b, opsin_params,
                   &linear_r, &linear_g, &linear_b);

          Store(linear_r, d, row0 + x);
          Store(linear_g, d, row1 + x);
          Store(linear_b, d, row2 + x);
        }
      },
      "OpsinToLinear");
}

// Same, but not in-place.
HWY_ATTR void OpsinToLinear(const Image3F& opsin, const Rect& rect,
                            ThreadPool* pool, Image3F* JXL_RESTRICT linear,
                            const OpsinParams& opsin_params) {
  PROFILER_FUNC;

  JXL_ASSERT(SameSize(rect, *linear));

  RunOnPool(
      pool, 0, static_cast<int>(rect.ysize()), ThreadPool::SkipInit(),
      [&](const int task, int /*thread*/) HWY_ATTR {
        const size_t y = static_cast<size_t>(task);

        // Faster than adding via ByteOffset at end of loop.
        const float* JXL_RESTRICT row_opsin_0 = rect.ConstPlaneRow(opsin, 0, y);
        const float* JXL_RESTRICT row_opsin_1 = rect.ConstPlaneRow(opsin, 1, y);
        const float* JXL_RESTRICT row_opsin_2 = rect.ConstPlaneRow(opsin, 2, y);
        float* JXL_RESTRICT row_linear_0 = linear->PlaneRow(0, y);
        float* JXL_RESTRICT row_linear_1 = linear->PlaneRow(1, y);
        float* JXL_RESTRICT row_linear_2 = linear->PlaneRow(2, y);

        const HWY_FULL(float) d;

        for (size_t x = 0; x < rect.xsize(); x += d.N) {
          const auto in_opsin_x = Load(d, row_opsin_0 + x);
          const auto in_opsin_y = Load(d, row_opsin_1 + x);
          const auto in_opsin_b = Load(d, row_opsin_2 + x);
          JXL_COMPILER_FENCE;
          auto linear_r = Undefined(d);
          auto linear_g = Undefined(d);
          auto linear_b = Undefined(d);
          XybToRgb(d, in_opsin_x, in_opsin_y, in_opsin_b, opsin_params,
                   &linear_r, &linear_g, &linear_b);

          Store(linear_r, d, row_linear_0 + x);
          Store(linear_g, d, row_linear_1 + x);
          Store(linear_b, d, row_linear_2 + x);
        }
      },
      "OpsinToLinear(Rect)");
}

// Transform YCbCr to RGB.
// Could be performed in-place (i.e. Y, Cb and Cr could alias R, B and B).
HWY_ATTR void YcbcrToRgb(const ImageF& y_plane, const ImageF& cb_plane,
                         const ImageF& cr_plane, ImageF* r_plane,
                         ImageF* g_plane, ImageF* b_plane, ThreadPool* pool) {
  const HWY_FULL(float) df;
  constexpr size_t S = df.N;  // Step.

  const size_t xsize = y_plane.xsize();
  const size_t ysize = y_plane.ysize();

  if ((xsize == 0) || (ysize == 0)) return;

  const auto c128 = Set(df, 128.0f);
  const auto crcr = Set(df, 1.402f);
  const auto cgcb = Set(df, -0.34414f);
  const auto cgcr = Set(df, -0.71414f);
  const auto cbcb = Set(df, 1.772f);

  constexpr size_t kGroupArea = kGroupDim * kGroupDim;
  const size_t lines_per_group = DivCeil(kGroupArea, xsize);
  const size_t num_stripes = DivCeil(ysize, lines_per_group);
  const auto transform = [&](int idx, int /* thread*/) HWY_ATTR {
    const size_t y0 = idx * lines_per_group;
    const size_t y1 = std::min<size_t>(y0 + lines_per_group, ysize);
    for (size_t y = y0; y < y1; ++y) {
      const float* y_row = y_plane.ConstRow(y);
      const float* cb_row = cb_plane.ConstRow(y);
      const float* cr_row = cr_plane.ConstRow(y);
      float* r_row = r_plane->Row(y);
      float* g_row = g_plane->Row(y);
      float* b_row = b_plane->Row(y);
      for (size_t x = 0; x < xsize; x += S) {
        const auto y_vec = Load(df, y_row + x) + c128;
        const auto cb_vec = Load(df, cb_row + x);
        const auto cr_vec = Load(df, cr_row + x);
        const auto r_vec = MulAdd(crcr, cr_vec, y_vec);
        const auto g_vec = MulAdd(cgcr, cr_vec, MulAdd(cgcb, cb_vec, y_vec));
        const auto b_vec = MulAdd(cbcb, cb_vec, y_vec);
        Store(r_vec, df, r_row + x);
        Store(g_vec, df, g_row + x);
        Store(b_vec, df, b_row + x);
      }
    }
  };
  RunOnPool(pool, 0, static_cast<int>(num_stripes), ThreadPool::SkipInit(),
            transform, "YcbcrToRgb");
}

/* Vertical upsampling:
 *  input:
 *   (a, b, c) := |a1 a2 a3 a4|
 *                |b1 b2 b3 b4| <- current line
 *                |c1 c2 c3 c4|
 *  intermediate:
 *   u := a + 3 * b
 *   d := c + 3 * b
 *  output:
 *  |u1 u2 u3 u4| =: (u, d)
 *  |d1 d2 d3 d4|
 */
HWY_ATTR ImageF UpsampleV2(const ImageF& src, ThreadPool* pool) {
  const HWY_FULL(float) df;
  constexpr size_t S = df.N;
  const auto c14 = Set(df, 0.25f);
  const auto c34 = Set(df, 0.75f);

  const size_t xsize = src.xsize();
  const size_t ysize = src.ysize();
  JXL_ASSERT(xsize != 0);
  JXL_ASSERT(ysize != 0);
  ImageF dst(xsize, ysize * 2);
  if (ysize == 1) {
    memcpy(dst.Row(0), src.Row(0), xsize * sizeof(*src.Row(0)));
    memcpy(dst.Row(1), src.Row(0), xsize * sizeof(*src.Row(0)));
  } else {
    constexpr size_t kGroupArea = kGroupDim * kGroupDim;
    const size_t lines_per_group = DivCeil(kGroupArea, xsize);
    const size_t num_stripes = DivCeil(ysize, lines_per_group);
    const auto upsample = [&](int idx, int /* thread*/) HWY_ATTR {
      const size_t y0 = idx * lines_per_group;
      const size_t y1 = std::min<size_t>(y0 + lines_per_group, ysize);
      for (size_t y = y0; y < y1; ++y) {
        const float* JXL_RESTRICT prev_row = src.ConstRow(y == 0 ? 1 : y - 1);
        const float* JXL_RESTRICT current_row = src.ConstRow(y);
        const float* JXL_RESTRICT next_row =
            src.ConstRow(y == ysize - 1 ? ysize - 2 : y + 1);
        float* JXL_RESTRICT dst1_row = dst.Row(2 * y);
        float* JXL_RESTRICT dst2_row = dst.Row(2 * y + 1);
        for (size_t x = 0; x < xsize; x += S) {
          const auto current34 = Load(df, current_row + x) * c34;
          const auto prev = Load(df, prev_row + x);
          const auto next = Load(df, next_row + x);
          Store(MulAdd(prev, c14, current34), df, dst1_row + x);
          Store(MulAdd(next, c14, current34), df, dst2_row + x);
        }
      }
    };
    RunOnPool(pool, 0, static_cast<int>(num_stripes), ThreadPool::SkipInit(),
              upsample, "UpsampleV2");
  }
  return dst;
}

/* Horizontal upsampling:
 *  input:
 *   (a, b, c) := |a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4|
 *                             ^^^^^^^^^^^
 *                            current block
 *  intermediate:
 *   l := (a << 3) {0001} (b >> 1) = [a4 b1 b2 b3]
 *   r := (c >> 3) {1000} (b << 1) = [b2 b3 b4 c1]
 *   o := 3 * b + l
 *   e := 3 * b + r
 *  output:
 *   |o1 e1 o2 e2 o3 e3 o4 e4| =: (o, e)
 */
HWY_ATTR ImageF UpsampleH2(const ImageF& src, ThreadPool* pool) {
  const size_t xsize = src.xsize();
  const size_t ysize = src.ysize();
  JXL_ASSERT(xsize != 0);
  JXL_ASSERT(ysize != 0);
  ImageF dst(xsize * 2, ysize);

  constexpr size_t kGroupArea = kGroupDim * kGroupDim;
  const size_t lines_per_group = DivCeil(kGroupArea, xsize);
  const size_t num_stripes = DivCeil(ysize, lines_per_group);

#if HWY_BITS == 0
  const auto upsample = [&](int idx, int /* thread*/) {
    const size_t y0 = idx * lines_per_group;
    const size_t y1 = std::min<size_t>(y0 + lines_per_group, ysize);
    for (size_t y = y0; y < y1; ++y) {
      const float* JXL_RESTRICT current_row = src.ConstRow(y);
      float* JXL_RESTRICT dst_row = dst.Row(y);
      // TODO(eustas): roll prev <- current <- next?
      for (size_t x = 1; x < xsize - 1; ++x) {
        const float current34 = current_row[x] * 0.75f;
        const float prev = current_row[x - 1];
        const float next = current_row[x + 1];
        dst_row[x * 2] = current34 + prev * 0.25f;
        dst_row[x * 2 + 1] = current34 + next * 0.25f;
      }
      if (xsize == 1) {
        dst_row[0] = dst_row[1] = current_row[0];
      } else {
        const float leftmost = current_row[0] * 0.75f + current_row[1] * 0.25f;
        dst_row[0] = dst_row[1] = leftmost;
        const float rightmost =
            current_row[xsize - 1] * 0.75f + current_row[xsize - 2] * 0.25f;
        dst_row[xsize * 2 - 2] = dst_row[xsize * 2 - 1] = rightmost;
      }
    }
  };
#else
  // TODO(eustas): Neighbors::(L|R)1 from convolve.h allows full-vector shift
  //               even for AVX2; make those helpers more independent and use
  //               both in convolve.h and here.
  constexpr size_t S = 4;  // Half of AVX2, until TODO is resolved.
  const HWY_CAPPED(float, S) df;
  const HWY_CAPPED(uint32_t, S) du;

  HWY_ALIGN static const uint32_t k1000[S] = {0u, 0u, 0u, ~0u};
  HWY_ALIGN static const uint32_t k0001[S] = {~0u, 0u, 0u, 0u};
  const auto c1000 = MaskFromVec(BitCast(df, Load(du, k1000)));
  const auto c0001 = MaskFromVec(BitCast(df, Load(du, k0001)));

  const auto upsample = [&](int idx, int /* thread*/) HWY_ATTR {
    const size_t y0 = idx * lines_per_group;
    const size_t y1 = std::min<size_t>(y0 + lines_per_group, ysize);
    for (size_t y = y0; y < y1; ++y) {
      const float* JXL_RESTRICT current_row = src.ConstRow(y);
      float* JXL_RESTRICT dst_row = dst.Row(y);
      const auto c14 = Set(df, 0.25f);
      const auto c34 = Set(df, 0.75f);
      auto prev = Undefined(df);
      auto current = Load(df, current_row);
      for (size_t x = 0; x < xsize; x += S) {
        // Image provides 2x vector size of extra space after the row.
        // So, it is valid to read one extra vector after the end.
        const auto next = Load(df, current_row + x + S);
        const auto current34 = current * c34;
        const auto l =
            IfThenElse(c0001, hwy::Broadcast<3>(prev), Shuffle2103(current));
        const auto r =
            IfThenElse(c1000, hwy::Broadcast<0>(next), Shuffle0321(current));
        const auto o = MulAdd(l, c14, current34);
        const auto e = MulAdd(r, c14, current34);
        Store(InterleaveLo(o, e), df, dst_row + x * 2);
        Store(InterleaveHi(o, e), df, dst_row + x * 2 + S);
        prev = current;
        current = next;
      }
      // Fix border values.
      if (xsize == 1) {
        dst_row[0] = dst_row[1] = current_row[0];
      } else {
        const float leftmost = current_row[0] * 0.75f + current_row[1] * 0.25f;
        dst_row[0] = dst_row[1] = leftmost;
        const float rightmost =
            current_row[xsize - 1] * 0.75f + current_row[xsize - 2] * 0.25f;
        dst_row[xsize * 2 - 2] = dst_row[xsize * 2 - 1] = rightmost;
      }
    }
  };
#endif
  RunOnPool(pool, 0, static_cast<int>(num_stripes), ThreadPool::SkipInit(),
            upsample, "UpsampleH2");
  return dst;
}

}  // namespace jxl
