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

#include "jxl/compressed_dc.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/compressed_dc.cc"
#include <hwy/foreach_target.h>
//

#include <algorithm>
#include <array>
#include <hwy/aligned_allocator.h>
#include <memory>
#include <utility>
#include <vector>

#include "jxl/ac_strategy.h"
#include "jxl/ans_params.h"
#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/common.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/dec_cache.h"
#include "jxl/enc_cache.h"
#include "jxl/entropy_coder.h"
#include "jxl/image.h"

// SIMD code
#include <hwy/before_namespace-inl.h>
namespace jxl {
#include <hwy/begin_target-inl.h>

using D = HWY_FULL(float);
using DScalar = HWY_CAPPED(float, 1);
using V = Vec<D>;

// TODO(veluca): optimize constants.
const float w1 = 0.20345139757231578f;
const float w2 = 0.0334829185968739f;
const float w0 = 1.0f - 4.0f * (w1 + w2);

template <class V>
V MaxWorkaround(V a, V b) {
#if (HWY_TARGET == HWY_AVX3) && HWY_COMPILER_CLANG <= 800
  // Prevents "Do not know how to split the result of this operator" error
  return IfThenElse(a > b, a, b);
#else
  return Max(a, b);
#endif
}

template <typename D>
JXL_INLINE void ComputePixelChannel(const D d, const float dc_factor,
                                    const float* JXL_RESTRICT row_top,
                                    const float* JXL_RESTRICT row,
                                    const float* JXL_RESTRICT row_bottom,
                                    Vec<D>* JXL_RESTRICT mc,
                                    Vec<D>* JXL_RESTRICT sm,
                                    Vec<D>* JXL_RESTRICT gap, size_t x) {
  const auto tl = LoadU(d, row_top + x - 1);
  const auto tc = Load(d, row_top + x);
  const auto tr = LoadU(d, row_top + x + 1);

  const auto ml = LoadU(d, row + x - 1);
  *mc = Load(d, row + x);
  const auto mr = LoadU(d, row + x + 1);

  const auto bl = LoadU(d, row_bottom + x - 1);
  const auto bc = Load(d, row_bottom + x);
  const auto br = LoadU(d, row_bottom + x + 1);

  const auto w_center = Set(d, w0);
  const auto w_side = Set(d, w1);
  const auto w_corner = Set(d, w2);

  const auto corner = tl + tr + bl + br;
  const auto side = ml + mr + tc + bc;
  *sm = corner * w_corner + side * w_side + *mc * w_center;

  const auto dc_quant = Set(d, dc_factor);
  *gap = MaxWorkaround(*gap, Abs((*mc - *sm) / dc_quant));
}

template <typename D>
JXL_INLINE void ComputePixel(
    const float* JXL_RESTRICT dc_factors,
    const float* JXL_RESTRICT* JXL_RESTRICT rows_top,
    const float* JXL_RESTRICT* JXL_RESTRICT rows,
    const float* JXL_RESTRICT* JXL_RESTRICT rows_bottom,
    float* JXL_RESTRICT* JXL_RESTRICT out_rows, size_t x) {
  const D d;
  auto mc_x = Undefined(d);
  auto mc_y = Undefined(d);
  auto mc_b = Undefined(d);
  auto sm_x = Undefined(d);
  auto sm_y = Undefined(d);
  auto sm_b = Undefined(d);
  auto gap = Set(d, 0.5f);
  ComputePixelChannel(d, dc_factors[0], rows_top[0], rows[0], rows_bottom[0],
                      &mc_x, &sm_x, &gap, x);
  ComputePixelChannel(d, dc_factors[1], rows_top[1], rows[1], rows_bottom[1],
                      &mc_y, &sm_y, &gap, x);
  ComputePixelChannel(d, dc_factors[2], rows_top[2], rows[2], rows_bottom[2],
                      &mc_b, &sm_b, &gap, x);
  auto factor = MulAdd(Set(d, -4.0f), gap, Set(d, 3.0f));
  factor = ZeroIfNegative(factor);

  auto out = MulAdd(sm_x - mc_x, factor, mc_x);
  Store(out, d, out_rows[0] + x);
  out = MulAdd(sm_y - mc_y, factor, mc_y);
  Store(out, d, out_rows[1] + x);
  out = MulAdd(sm_b - mc_b, factor, mc_b);
  Store(out, d, out_rows[2] + x);
}

void AdaptiveDCSmoothing(const float* dc_factors, Image3F* dc,
                         ThreadPool* pool) {
  const size_t xsize = dc->xsize();
  const size_t ysize = dc->ysize();
  if (ysize <= 2 || xsize <= 2) return;

  // TODO(veluca): use tile-based processing?
  // TODO(veluca): decide if changes to the y channel should be propagated to
  // the x and b channels through color correlation.
  JXL_ASSERT(w1 + w2 < 0.25f);

  PROFILER_FUNC;

  Image3F smoothed(xsize, ysize);
  // Fill in borders that the loop below will not. First and last are unused.
  for (size_t c = 0; c < 3; c++) {
    for (size_t y : {size_t(0), ysize - 1}) {
      memcpy(smoothed.PlaneRow(c, y), dc->PlaneRow(c, y),
             xsize * sizeof(float));
    }
  }
  auto process_row = [&](int y, int /*thread*/) {
    const float* JXL_RESTRICT rows_top[3]{
        dc->ConstPlaneRow(0, y - 1),
        dc->ConstPlaneRow(1, y - 1),
        dc->ConstPlaneRow(2, y - 1),
    };
    const float* JXL_RESTRICT rows[3] = {
        dc->ConstPlaneRow(0, y),
        dc->ConstPlaneRow(1, y),
        dc->ConstPlaneRow(2, y),
    };
    const float* JXL_RESTRICT rows_bottom[3] = {
        dc->ConstPlaneRow(0, y + 1),
        dc->ConstPlaneRow(1, y + 1),
        dc->ConstPlaneRow(2, y + 1),
    };
    float* JXL_RESTRICT rows_out[3] = {
        smoothed.PlaneRow(0, y),
        smoothed.PlaneRow(1, y),
        smoothed.PlaneRow(2, y),
    };
    for (size_t x : {size_t(0), xsize - 1}) {
      for (size_t c = 0; c < 3; c++) {
        rows_out[c][x] = rows[c][x];
      }
    }

    size_t x = 1;
    // First pixels
    const size_t N = Lanes(D());
    for (; x < std::min(N, xsize - 1); x++) {
      ComputePixel<DScalar>(dc_factors, rows_top, rows, rows_bottom, rows_out,
                            x);
    }
    // Full vectors.
    for (; x + N <= xsize - 1; x += N) {
      ComputePixel<D>(dc_factors, rows_top, rows, rows_bottom, rows_out, x);
    }
    // Last pixels.
    for (; x < xsize - 1; x++) {
      ComputePixel<DScalar>(dc_factors, rows_top, rows, rows_bottom, rows_out,
                            x);
    }
  };
  RunOnPool(pool, 1, ysize - 1, ThreadPool::SkipInit(), process_row,
            "DCSmoothingRow");
  dc->Swap(smoothed);
}

// DC dequantization.
void DequantDC(const Rect& r, Image3F* dc, const Image& in,
               const float* dc_factors, float mul, const float* cfl_factors) {
  const HWY_FULL(float) df;
  const HWY_CAPPED(pixel_type, MaxLanes(df)) di;  // assumes pixel_type <= float
  const auto fac_x = Set(df, dc_factors[0] * mul);
  const auto fac_y = Set(df, dc_factors[1] * mul);
  const auto fac_b = Set(df, dc_factors[2] * mul);
  const auto cfl_fac_x = Set(df, cfl_factors[0]);
  const auto cfl_fac_b = Set(df, cfl_factors[2]);
  for (size_t y = 0; y < r.ysize(); y++) {
    float* dec_row_x = r.PlaneRow(dc, 0, y);
    float* dec_row_y = r.PlaneRow(dc, 1, y);
    float* dec_row_b = r.PlaneRow(dc, 2, y);
    const int32_t* quant_row_x = in.channel[1].plane.Row(y);
    const int32_t* quant_row_y = in.channel[0].plane.Row(y);
    const int32_t* quant_row_b = in.channel[2].plane.Row(y);
    for (size_t x = 0; x < r.xsize(); x += Lanes(di)) {
      const auto in_x = ConvertTo(df, Load(di, quant_row_x + x)) * fac_x;
      const auto in_y = ConvertTo(df, Load(di, quant_row_y + x)) * fac_y;
      const auto in_b = ConvertTo(df, Load(di, quant_row_b + x)) * fac_b;
      Store(in_y, df, dec_row_y + x);
      Store(MulAdd(in_y, cfl_fac_x, in_x), df, dec_row_x + x);
      Store(MulAdd(in_y, cfl_fac_b, in_b), df, dec_row_b + x);
    }
  }
}

#include <hwy/end_target-inl.h>
}  // namespace jxl
#include <hwy/after_namespace-inl.h>

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(DequantDC)
HWY_EXPORT(AdaptiveDCSmoothing)
void AdaptiveDCSmoothing(const float* dc_factors, Image3F* dc,
                         ThreadPool* pool) {
  return HWY_DYNAMIC_DISPATCH(AdaptiveDCSmoothing)(dc_factors, dc, pool);
}

void DequantDC(const Rect& r, Image3F* dc, const Image& in,
               const float* dc_factors, float mul, const float* cfl_factors) {
  return HWY_DYNAMIC_DISPATCH(DequantDC)(r, dc, in, dc_factors, mul,
                                         cfl_factors);
}

}  // namespace jxl
#endif  // HWY_ONCE
