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

#include "jxl/brunsli.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

// Brunsli public headers:
#include <brunsli/brunsli_decode.h>
#include <brunsli/brunsli_encode.h>

// Brunsli internal headers:
#include "c/common/constants.h"
#include "c/common/context.h"
#include "c/dec/state.h"
#include "c/enc/state.h"
//

#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/common.h"
#include "jxl/dct_scales.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/dec_dct.h"
#include "jxl/dec_xyb.h"
#include "jxl/enc_dct.h"
#include "jxl/enc_xyb.h"
#include "jxl/gaborish.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/luminance.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/brunsli.cc"
#include <hwy/foreach_target.h>

// Definitions required by SIMD. Only define once.
#ifndef JXL_BRUNSLI
#define JXL_BRUNSLI
namespace jxl {

struct BrunsliExtensions {
  std::string hdr_orig_colorspace;
  std::string hdr_colorspace;

  BrunsliDccParams dcc;
  BrunsliGaborishParams gab;
};

struct GetDcCoeff {
  const ::brunsli::coeff_t* JXL_RESTRICT src;
  size_t xsize_blocks;

  ::brunsli::coeff_t operator()(size_t u, size_t v) const {
    return src[(u + v * xsize_blocks) * kDCTBlockSize];
  }
};

struct GetAcCoeff {
  const ::brunsli::coeff_t* JXL_RESTRICT src;
  size_t xsize_blocks;

  ::brunsli::coeff_t operator()(size_t u, size_t v) const {
    return src[(u + v * xsize_blocks) * kDCTBlockSize + 1];
  }
};

struct GetDcCoeffTransposed {
  const ::brunsli::coeff_t* JXL_RESTRICT src;
  size_t xsize_blocks;

  ::brunsli::coeff_t operator()(size_t u, size_t v) const {
    return src[(v + u * xsize_blocks) * kDCTBlockSize];
  }
};

struct GetAcCoeffTransposed {
  const ::brunsli::coeff_t* JXL_RESTRICT src;
  size_t xsize_blocks;

  ::brunsli::coeff_t operator()(size_t u, size_t v) const {
    return src[(v + u * xsize_blocks) * kDCTBlockSize + 8];
  }
};

struct MakeSlope {
  float max_gap;
  float min_step;

  template <typename GetDcCoeffFn, typename GetAcCoeffFn>
  bool operator()(const GetDcCoeffFn& get_dc_coeff,
                  const GetAcCoeffFn& get_ac_coeff, float q_ac_dc, size_t y,
                  size_t x0, size_t x1, size_t xsize, float* JXL_RESTRICT from,
                  float* JXL_RESTRICT to) const {
    const float current = static_cast<float>(get_dc_coeff(x0, y));
    float ll = current;
    const float lr = current;
    if (x0 != 0) {
      ll = get_dc_coeff(x0 - 1, y) - q_ac_dc * get_ac_coeff(x0 - 1, y);
    }
    const float l = (ll + lr) * 0.5f;

    const float rl = current;
    float rr = current;
    if (x1 != xsize - 1) {
      rr = get_dc_coeff(x1 + 1, y) + q_ac_dc * get_ac_coeff(x1 + 1, y);
    }
    const float r = (rl + rr) * 0.5f;

    const float l_delta = l - current;
    const float r_delta = r - current;
    const float l_delta_abs = std::abs(l_delta);
    const float r_delta_abs = std::abs(r_delta);
    bool is_sane = (std::abs(l - r) > min_step) && (l_delta_abs <= max_gap) &&
                   (r_delta_abs <= max_gap);
    *from = l;
    *to = r;
    return is_sane;
  }
};

// First and last indices (inclusive) of the range of repeated DC values.
struct Range {
  size_t x0;
  size_t x1;
  Range(size_t left, size_t right) : x0(left), x1(right){};
};
typedef std::vector<Range> RowRanges;
typedef std::vector<RowRanges> Ranges;

template <typename GetDcCoeffFn, typename GetAcCoeffFn>
Ranges FindPlateaux(const GetDcCoeffFn& get_dc_coeff,
                    const GetAcCoeffFn& get_ac_coeff, size_t xsize,
                    size_t ysize) {
  Ranges result(ysize);
  for (size_t y = 0; y < ysize; ++y) {
    RowRanges& row = result[y];
    ::brunsli::coeff_t span_dc = get_dc_coeff(0, y);
    size_t span_start = 0;
    for (size_t x = 0; x <= xsize; ++x) {
      bool end = (x == xsize);
      ::brunsli::coeff_t dc = end ? (span_dc + 8192) : get_dc_coeff(x, y);
      bool step = span_dc != dc;
      bool bump = end || (get_ac_coeff(x, y) != 0);
      if (step || bump) {
        if (span_start < x) row.emplace_back(span_start, x - 1);
        span_start = x + (bump ? 1 : 0);
        span_dc = dc;
      }
    }
  }
  return result;
}

template <typename GetDcCoeffFn, typename GetAcCoeffFn, typename SlopeFn>
void BuildSlopes(const GetDcCoeffFn& get_dc_coeff,
                 const GetAcCoeffFn& get_ac_coeff, SlopeFn* make_slope,
                 const float q_ac_dc, const Ranges& ranges, ImageF* approx,
                 ImageB* presence, ImageF* slope) {
  const size_t xsize = approx->xsize();
  const size_t ysize = approx->ysize();

  for (size_t y = 0; y < ysize; ++y) {
    float* JXL_RESTRICT approx_row = approx->Row(y);
    float* JXL_RESTRICT slope_row = slope->Row(y);
    uint8_t* JXL_RESTRICT presence_row = presence->Row(y);
    std::fill_n(presence_row, xsize, 0);
    for (auto& range : ranges[y]) {
      const size_t x0 = range.x0;
      const size_t x1 = range.x1;
      float from;
      float to;
      bool has_slope = (*make_slope)(get_dc_coeff, get_ac_coeff, q_ac_dc, y, x0,
                                     x1, xsize, &from, &to);
      if (!has_slope) continue;
      const size_t span = x1 - x0 + 1;
      const float delta = to - from;
      const float step = delta / span;
      const float base = from + 0.5f * step;
      for (size_t x = x0; x <= x1; ++x) {
        presence_row[x] = 1;
        const float diff = static_cast<float>(x - x0);
        approx_row[x] = base + diff * step;
        // Base function f(u, k) = k * cos(u / Pi) -> f(0, k) = k, f(1, k) = -k.
        // To stitch consecutive "steps" we want
        // f(0) = -0.5 * step, f(1) = 0.5 * step
        slope_row[x] = -0.5f * step;
      }
    }
  }
}

void FixDc(const ::brunsli::coeff_t* JXL_RESTRICT coeffs,
           const MakeSlope& make_slope, int q_dc, int q_ach, int q_acv,
           size_t xsize_blocks, size_t ysize_blocks, float* JXL_RESTRICT dst,
           size_t dst_stride) {
  constexpr size_t N = kBlockDim;
  static_assert(N == 8, "JPEG block dim must be 8");
  static_assert(kDCTBlockSize == N * N, "JPEG block size must be 64");
  const float dequant_mult =
      static_cast<float>(q_dc) * IDCTScales<N>()[0] * IDCTScales<N>()[0];
  const float q_ach_dc = static_cast<float>(q_ach) / static_cast<float>(q_dc);
  const float q_acv_dc = static_cast<float>(q_acv) / static_cast<float>(q_dc);

  const GetDcCoeff get_dc_coeff = {coeffs, xsize_blocks};
  const GetAcCoeff get_ac_coeff = {coeffs, xsize_blocks};
  Ranges ranges_h =
      FindPlateaux(get_dc_coeff, get_ac_coeff, xsize_blocks, ysize_blocks);
  ImageF approx_h(xsize_blocks, ysize_blocks);
  ImageF slope_h(xsize_blocks, ysize_blocks);
  ImageB has_h(xsize_blocks, ysize_blocks);
  BuildSlopes(get_dc_coeff, get_ac_coeff, &make_slope, q_ach_dc, ranges_h,
              &approx_h, &has_h, &slope_h);

  const GetDcCoeffTransposed get_dc_t_coeff = {coeffs, xsize_blocks};
  const GetAcCoeffTransposed get_ac_t_coeff = {coeffs, xsize_blocks};
  Ranges ranges_v =
      FindPlateaux(get_dc_t_coeff, get_ac_t_coeff, ysize_blocks, xsize_blocks);
  // NB: transposed.
  ImageF approx_v(ysize_blocks, xsize_blocks);
  ImageF slope_v(ysize_blocks, xsize_blocks);
  ImageB has_v(ysize_blocks, xsize_blocks);
  BuildSlopes(get_dc_t_coeff, get_ac_t_coeff, &make_slope, q_acv_dc, ranges_v,
              &approx_v, &has_v, &slope_v);

  for (size_t by = 0; by < ysize_blocks; ++by) {
    const ::brunsli::coeff_t* JXL_RESTRICT coeffs_row =
        &coeffs[by * xsize_blocks * kDCTBlockSize];
    const uint8_t* JXL_RESTRICT has_h_row = has_h.ConstRow(by);
    const float* JXL_RESTRICT approx_h_row = approx_h.ConstRow(by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      const bool has_h_val = (has_h_row[bx] != 0);
      const bool has_v_val = (has_v.ConstRow(bx)[by] != 0);
      float fixed_val;
      if (has_h_val && has_v_val) {
        fixed_val = 0.5f * (approx_h_row[bx] + approx_v.ConstRow(bx)[by]);
      } else if (has_h_val) {
        fixed_val = approx_h_row[bx];
      } else if (has_v_val) {
        fixed_val = approx_v.ConstRow(bx)[by];
      } else {
        fixed_val = coeffs_row[bx * kDCTBlockSize];
      }
      dst[by * dst_stride + bx * kDCTBlockSize] = fixed_val * dequant_mult;
    }
  }

  for (size_t by = 0; by < ysize_blocks; ++by) {
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      const int16_t* JXL_RESTRICT src_coeffs =
          &coeffs[(by * xsize_blocks + bx) * kDCTBlockSize];
      const bool has_v_val = (has_v.ConstRow(bx)[by] != 0);
      const float slope_v_val = slope_v.ConstRow(bx)[by];
      if (has_v_val && (src_coeffs[8] == 0)) {
        dst[by * dst_stride + bx * kDCTBlockSize + 8] =
            slope_v_val * dequant_mult;
      }
    }
  }
  for (size_t by = 0; by < ysize_blocks; ++by) {
    const uint8_t* JXL_RESTRICT has_h_row = has_h.ConstRow(by);
    const float* JXL_RESTRICT slope_h_row = slope_h.ConstRow(by);
    for (size_t bx = 0; bx < xsize_blocks; ++bx) {
      const bool has_h_val = (has_h_row[bx] != 0);
      const float slope_h_val = slope_h_row[bx];
      const int16_t* JXL_RESTRICT src_coeffs =
          &coeffs[(by * xsize_blocks + bx) * kDCTBlockSize];
      if (has_h_val && (src_coeffs[1] == 0)) {
        dst[by * dst_stride + bx * kDCTBlockSize + 1] =
            slope_h_val * dequant_mult;
      }
    }
  }
}

}  // namespace jxl
#endif  // JXL_BRUNSLI

#include <hwy/before_namespace-inl.h>
namespace jxl {
#include <hwy/begin_target-inl.h>

Status JpegDataToPixels(const brunsli::JPEGData& src,
                        const BrunsliExtensions& extensions, Image3F* out,
                        ThreadPool* pool) {
  constexpr size_t N = kBlockDim;
  static_assert(N == 8, "JPEG block dim must be 8");
  static_assert(kDCTBlockSize == N * N, "JPEG block size must be 64");
  const size_t xsize = src.width;
  const size_t ysize = src.height;
  const size_t num_components = src.components.size();

  const HWY_FULL(float) df;
  const size_t S = Lanes(df);  // Step.
  JXL_DASSERT(kDCTBlockSize % S == 0);
  const HWY_FULL(int32_t) di32;
  const HWY_CAPPED(int16_t, MaxLanes(df)) di16;
  JXL_RETURN_IF_ERROR(num_components == 1 || num_components == 3);
  const bool is_gray = (num_components == 1);

  ImageF planes[3];

  const size_t xmcu = N * src.max_h_samp_factor;
  const size_t ymcu = N * src.max_v_samp_factor;

  for (size_t c = 0; c < num_components; ++c) {
    const brunsli::JPEGComponent& component = src.components[c];
    const size_t factor_x = src.max_h_samp_factor / component.h_samp_factor;
    const size_t factor_y = src.max_v_samp_factor / component.v_samp_factor;
    JXL_RETURN_IF_ERROR(src.max_h_samp_factor % component.h_samp_factor == 0);
    JXL_RETURN_IF_ERROR(src.max_v_samp_factor % component.v_samp_factor == 0);
    JXL_RETURN_IF_ERROR(component.quant_idx <
                        static_cast<int>(src.quant.size()));
    const size_t comp_xsize = component.width_in_blocks * N * factor_x;
    const size_t comp_ysize = component.height_in_blocks * N * factor_y;
    JXL_RETURN_IF_ERROR(xsize <= comp_xsize);
    JXL_RETURN_IF_ERROR(xsize > comp_xsize - xmcu);
    JXL_RETURN_IF_ERROR(ysize <= comp_ysize);
    JXL_RETURN_IF_ERROR(ysize > comp_ysize - ymcu);
    if (c == 0) {
      JXL_RETURN_IF_ERROR((factor_x == 1) && (factor_y == 1));
    } else {
      JXL_RETURN_IF_ERROR((factor_x <= 2) && (factor_y <= 2));
    }
  }

  for (size_t c = 0; c < num_components; ++c) {
    const brunsli::JPEGComponent& component = src.components[c];
    const brunsli::JPEGQuantTable& quant_table = src.quant[component.quant_idx];
    const size_t xsize_blocks = component.width_in_blocks;
    const size_t ysize_blocks = component.height_in_blocks;
    ImageF pixels(xsize_blocks * N, ysize_blocks * N);
    const size_t pixels_stride = pixels.PixelsPerRow();
    const size_t factor_x = src.max_h_samp_factor / component.h_samp_factor;
    const size_t factor_y = src.max_v_samp_factor / component.v_samp_factor;

    HWY_ALIGN float quant[kDCTBlockSize];
    for (size_t y = 0; y < N; ++y) {
      for (size_t x = 0; x < N; ++x) {
        const size_t i = y * N + x;
        quant[i] = static_cast<float>(quant_table.values[i]) *
                   IDCTScales<N>()[x] * IDCTScales<N>()[y];
      }
    }

    // Dequantize.
    constexpr size_t group_dim = kGroupDimInBlocks;
    ImageF dequantized(xsize_blocks * kDCTBlockSize, ysize_blocks);
    const size_t xsize_groups = DivCeil(xsize_blocks, group_dim);
    const size_t ysize_groups = DivCeil(ysize_blocks, group_dim);
    const auto dequantize = [&](int idx, int /* thread */) {
      HWY_ALIGN int16_t coeffs[kDCTBlockSize];
      const size_t gx = idx % xsize_groups;
      const size_t gy = idx / xsize_groups;
      const Rect group_rect_blocks(gx * group_dim, gy * group_dim, group_dim,
                                   group_dim, xsize_blocks, ysize_blocks);
      const size_t bx0 = group_rect_blocks.x0();
      const size_t bx1 = bx0 + group_rect_blocks.xsize();
      const size_t by0 = group_rect_blocks.y0();
      const size_t by1 = by0 + group_rect_blocks.ysize();
      for (size_t by = by0; by < by1; ++by) {
        float* JXL_RESTRICT dequantized_row = dequantized.Row(by);
        for (size_t bx = bx0; bx < bx1; ++bx) {
          float* JXL_RESTRICT dequantized_block =
              dequantized_row + bx * kDCTBlockSize;
          const int16_t* JXL_RESTRICT src_coeffs =
              &component.coeffs[(by * xsize_blocks + bx) * kDCTBlockSize];
          memcpy(coeffs, src_coeffs, sizeof(int16_t) * kDCTBlockSize);

          for (size_t i = 0; i < kDCTBlockSize; i += S) {
            const auto coeff =
                ConvertTo(df, PromoteTo(di32, Load(di16, coeffs + i)));
            const auto mult = Load(df, quant + i);
            Store(coeff * mult, df, dequantized_block + i);
          }
        }
      }
    };
    RunOnPool(pool, 0, static_cast<int>(xsize_groups * ysize_groups),
              ThreadPool::SkipInit(), dequantize, "Brunsli:Dequantize");

    // TODO(eustas): turn to pipeline and parallelize.
    if (extensions.dcc.active) {
      const BrunsliDccParams& dcc = extensions.dcc;
      JXL_RETURN_IF_ERROR(c < 3);
      const int q_dc = src.quant[component.quant_idx].values[0];
      const int q_ach = src.quant[component.quant_idx].values[1];
      const int q_acv = src.quant[component.quant_idx].values[8];
      MakeSlope make_slope{dcc.max_gap[c] / 64.0f, dcc.min_step[c] / 64.0f};
      FixDc(component.coeffs.data(), make_slope, q_dc, q_ach, q_acv,
            xsize_blocks, ysize_blocks, dequantized.Row(0),
            dequantized.PixelsPerRow());
    }

    IDct8(xsize_blocks, ysize_blocks, dequantized, pool, &pixels);

    // TODO: before or after upsampling?
    if (extensions.gab.active) {
      const BrunsliGaborishParams& gab = extensions.gab;
      JXL_RETURN_IF_ERROR(c < 3);
      ImageF gab_pixels(pixels.xsize(), pixels.ysize());
      ConvolveGaborish(pixels, gab.w1[c] / 1024.0f, gab.w2[c] / 1024.0f, pool,
                       &gab_pixels);
      float threshold = gab.threshold[c] / 16.0f;
      size_t limit = gab.limit[c];
      for (size_t by = 0; by < ysize_blocks; ++by) {
        for (size_t bx = 0; bx < xsize_blocks; ++bx) {
          const float* JXL_RESTRICT gab_pixels_row =
              gab_pixels.Row(by * N) + bx * N;
          float* JXL_RESTRICT pixels_row = pixels.Row(by * N) + bx * N;
          size_t count = 0;
          for (size_t u = 1; u < N - 1; ++u) {
            for (size_t v = 1; v < N - 1; ++v) {
              float a = gab_pixels_row[u + pixels_stride * v];
              float b = pixels_row[u + pixels_stride * v];
              if (std::abs(a - b) > threshold) count++;
            }
          }
          if (count < limit) {
            // TODO: vectorize
            for (size_t u = 0; u < N; ++u) {
              for (size_t v = 0; v < N; ++v) {
                pixels_row[u + pixels_stride * v] =
                    gab_pixels_row[u + pixels_stride * v];
              }
            }
          }
        }
      }
    }

    if (factor_x == 1) {
      if (factor_y == 1) {
        planes[c] = std::move(pixels);
      } else {
        planes[c] = UpsampleV2(pixels, pool);
      }
    } else {
      pixels.InitializePaddingForUnalignedAccesses();
      if (factor_y == 1) {
        planes[c] = UpsampleH2(pixels, pool);
      } else {
        planes[c] = UpsampleV2(UpsampleH2(pixels, pool), pool);
      }
    }
  }

  if (is_gray) {
    const auto c128 = Set(df, 128.0f);
    for (size_t y = 0; y < ysize; ++y) {
      float* JXL_RESTRICT y_row = planes[0].Row(y);
      for (size_t x = 0; x < xsize; x += S) {
        Store(Load(df, y_row + x) + c128, df, y_row + x);
      }
    }
  } else {
    YcbcrToRgb(planes[0], planes[1], planes[2], &planes[0], &planes[1],
               &planes[2], pool);
  }

  if (is_gray) {
    planes[1] = CopyImage(planes[0]);
    planes[2] = CopyImage(planes[0]);
  }
  Image3F rgb(std::move(planes[0]), std::move(planes[1]), std::move(planes[2]));
  rgb.ShrinkTo(xsize, ysize);
  *out = std::move(rgb);

  return true;
}

#include <hwy/end_target-inl.h>
}  // namespace jxl
#include <hwy/after_namespace-inl.h>

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(JpegDataToPixels)  // Local function.

namespace {

const uint8_t kBrunsliMagic[] = {0x0A, 0x04, 'B', 0xd2, 0xd5, 'N', 0x12};
constexpr size_t kBrunsliMagicSize = sizeof(kBrunsliMagic);

const uint8_t kBrunsliXHdr[] = {'J', 'X', 'L', ':', 'H', 'D', 'R'};
const uint8_t kBrunsliXDcc[] = {'J', 'X', 'L', ':', 'D', 'C', 'C'};
const uint8_t kBrunsliXGab[] = {'J', 'X', 'L', ':', 'G', 'A', 'B'};

const uint8_t kIccProfileTag[] = {'I', 'C', 'C', '_', 'P', 'R',
                                  'O', 'F', 'I', 'L', 'E', 0x00};

// JPEG markers.
constexpr uint8_t kSof0 = 0xC0;
constexpr uint8_t kDht = 0xC4;
constexpr uint8_t kEoi = 0xD9;
constexpr uint8_t kSos = 0xDA;
constexpr uint8_t kDqt = 0xDB;
constexpr uint8_t kApp0 = 0xE0;
constexpr uint8_t kApp2 = 0xE2;
constexpr uint8_t kApp11 = 0xEB;

constexpr uint32_t kHuffmanCodeDcSlotOffset = 0x00;
constexpr uint32_t kHuffmanCodeAcSlotOffset = 0x10;

// JFIF / v1.1 / no units / square pixel / no thumbnail.
const uint8_t kApp0Template[] = {kApp0, 0x00, 0x10, 0x4A, 0x46, 0x49,
                                 0x46,  0x00, 0x01, 0x01, 0x00, 0x00,
                                 0x01,  0x00, 0x01, 0x00, 0x00};
constexpr size_t kApp0TemplateSize = sizeof(kApp0Template);

using ByteSpan = Span<const uint8_t>;

}  // namespace

Status JpegDataToCoefficients(const brunsli::JPEGData& jpg, Image3F* out,
                              std::vector<int32_t>* out_quant_table,
                              ThreadPool* pool) {
  static_assert(kBlockDim == 8, "JPEG block dim must be 8");

  out_quant_table->resize(jpg.components.size() * kDCTBlockSize);

  for (size_t c = 0; c < jpg.components.size(); ++c) {
    const brunsli::JPEGComponent& component = jpg.components[c];

    const brunsli::JPEGQuantTable& quant_table = jpg.quant[component.quant_idx];
    std::copy(quant_table.values.begin(), quant_table.values.end(),
              out_quant_table->begin() + c * kDCTBlockSize);

    // Copy each DCT strip (64x1) to 8x8 squares.
    // Limit output to block-padded size (not MCU-padded).
    JXL_DASSERT((out->ysize() % kBlockDim) == 0);
    const size_t hib = std::min(static_cast<size_t>(component.height_in_blocks),
                                out->ysize() / kBlockDim);
    JXL_DASSERT((out->xsize() % kBlockDim) == 0);
    const size_t wib = std::min(static_cast<size_t>(component.width_in_blocks),
                                out->xsize() / kBlockDim);
    const size_t out_stride = static_cast<size_t>(out->PixelsPerRow());
    for (size_t by = 0; by < hib; ++by) {
      const int16_t* JXL_RESTRICT src_blocks_row =
          &component.coeffs[by * wib * kDCTBlockSize];
      const size_t ch = c < 2 ? 1 - c : 2;
      float* JXL_RESTRICT out_top_row = out->PlaneRow(ch, by * kBlockDim);

      for (size_t bx = 0; bx < wib; ++bx) {
        const int16_t* JXL_RESTRICT src_block =
            src_blocks_row + bx * kDCTBlockSize;

        for (size_t iy = 0; iy < kBlockDim; ++iy) {
          float* JXL_RESTRICT out_top_left =
              out_top_row + iy * out_stride + bx * kBlockDim;

          for (size_t ix = 0; ix < kBlockDim; ++ix) {
            out_top_left[ix] = src_block[iy * kBlockDim + ix];
          }
        }
      }
    }
  }

  return true;
}

namespace {

bool GetMarkerPayload(const uint8_t* data, size_t size, ByteSpan* payload) {
  if (size < 3) {
    return false;
  }
  size_t hi = data[1];
  size_t lo = data[2];
  size_t internal_size = (hi << 8u) | lo;
  // Second byte of marker is not counted towards size.
  if (internal_size != size - 1) {
    return false;
  }
  // cut second marker byte and "length" from payload.
  *payload = ByteSpan(data, size);
  payload->remove_prefix(3);
  return true;
}

Status ParseChunkedMarker(const brunsli::JPEGData& src, uint8_t marker_type,
                          const ByteSpan& tag, PaddedBytes* output) {
  output->clear();

  std::vector<ByteSpan> chunks;
  std::vector<bool> presence;
  size_t expected_number_of_parts = 0;
  bool is_first_chunk = true;
  for (const auto& marker : src.app_data) {
    if (marker.empty() || marker[0] != marker_type) {
      continue;
    }
    ByteSpan payload;
    if (!GetMarkerPayload(marker.data(), marker.size(), &payload)) {
      // Something is wrong with this marker; does not care.
      continue;
    }
    if ((payload.size() < tag.size()) ||
        memcmp(payload.data(), tag.data(), tag.size()) != 0) {
      continue;
    }
    payload.remove_prefix(tag.size());
    if (payload.size() < 2) {
      return JXL_FAILURE("Chunk is too small.");
    }
    uint8_t index = payload[0];
    uint8_t total = payload[1];
    payload.remove_prefix(2);

    JXL_RETURN_IF_ERROR(total != 0);
    if (is_first_chunk) {
      is_first_chunk = false;
      expected_number_of_parts = total;
      // 1-based indices; 0-th element is added for convenience.
      chunks.resize(total + 1);
      presence.resize(total + 1);
    } else {
      JXL_RETURN_IF_ERROR(expected_number_of_parts == total);
    }

    if (index == 0 || index > total) {
      return JXL_FAILURE("Invalid chunk index.");
    }

    if (presence[index]) {
      return JXL_FAILURE("Duplicate chunk.");
    }
    presence[index] = true;
    chunks[index] = payload;
  }

  for (size_t i = 0; i < expected_number_of_parts; ++i) {
    // 0-th element is not used.
    size_t index = i + 1;
    if (!presence[index]) {
      return JXL_FAILURE("Missing chunk.");
    }
    output->append(chunks[index]);
  }

  return true;
}

Status AddChunkedMarker(brunsli::JPEGData* out, uint8_t marker_type,
                        const ByteSpan& tag, const PaddedBytes& payload) {
  // 2 bytes encode size itself, 2 bytes are index / total.
  const size_t kChunkOverhead = 4 + tag.size();
  // Chunk size is an uint16_t.
  constexpr size_t kChunkSizeBits = 16;
  const size_t kMaxChunkSize = (1u << kChunkSizeBits) - kChunkOverhead;

  const size_t full_size = payload.size();
  const size_t num_chunks = DivCeil(full_size, kMaxChunkSize);
  if (num_chunks == 0 || num_chunks > 255) {
    return false;
  }
  for (size_t i = 0; i < num_chunks; ++i) {
    const size_t start = i * kMaxChunkSize;
    const size_t end = std::min(start + kMaxChunkSize, full_size);
    const size_t part_length = end - start;
    const size_t chunk_length = part_length + kChunkOverhead;

    JXL_ASSERT((chunk_length >> kChunkSizeBits) == 0);
    const uint8_t hi = chunk_length >> 8u;
    const uint8_t lo = chunk_length & 0xFFu;

    std::vector<uint8_t> extension_chunk;
    extension_chunk.push_back(marker_type);
    extension_chunk.push_back(hi);
    extension_chunk.push_back(lo);
    extension_chunk.insert(extension_chunk.end(), tag.data(),
                           tag.data() + tag.size());
    extension_chunk.push_back(i + 1);
    extension_chunk.push_back(num_chunks);
    extension_chunk.insert(extension_chunk.end(), payload.data() + start,
                           payload.data() + start + part_length);

    // 1 extra for marker byte.
    JXL_ASSERT(extension_chunk.size() == (chunk_length + 1));
    out->app_data.push_back(extension_chunk);
    out->marker_order.push_back(marker_type);
  }
  return true;
}

BrunsliExtensions ParseBrunsliExtensions(const brunsli::JPEGData& src) {
  BrunsliExtensions result{};

  PaddedBytes hdr_payload;
  if (!ParseChunkedMarker(src, kApp11, ByteSpan(kBrunsliXHdr), &hdr_payload)) {
    hdr_payload.clear();
    JXL_WARNING("ReJPEG: corrupted HDR extension payload\n");
  }
  if (!hdr_payload.empty()) {
    std::string colorspaces = std::string(
        reinterpret_cast<const char*>(hdr_payload.data()), hdr_payload.size());
    const size_t pos = colorspaces.find('>', 0);
    if (pos != std::string::npos) {
      result.hdr_orig_colorspace = colorspaces.substr(0, pos);
      result.hdr_colorspace = colorspaces.substr(pos + 1);
    }
  }

  PaddedBytes dcc_payload;
  if (!ParseChunkedMarker(src, kApp11, ByteSpan(kBrunsliXDcc), &dcc_payload)) {
    dcc_payload.clear();
    JXL_WARNING("ReJPEG: corrupted DCC extension payload\n");
  }
  if (!dcc_payload.empty()) {
    // TODO(eustas): pass number of components.
    const size_t num_components = 3;
    BrunsliDccParams& dcc = result.dcc;
    if (dcc_payload.size() == num_components * 2) {
      const uint8_t* data = dcc_payload.data();
      for (size_t c = 0; c < num_components; ++c) {
        dcc.max_gap[c] = data[c * 2];
        dcc.min_step[c] = data[c * 2 + 1];
      }
      dcc.active = true;
    } else {
      JXL_WARNING("ReJPEG: corrupted DCC extension data\n");
    }
  }

  PaddedBytes gab_payload;
  if (!ParseChunkedMarker(src, kApp11, ByteSpan(kBrunsliXGab), &gab_payload)) {
    gab_payload.clear();
    JXL_WARNING("ReJPEG: corrupted GAB extension payload\n");
  }
  if (!gab_payload.empty()) {
    // TODO(eustas): pass number of components.
    const size_t num_components = 3;
    BrunsliGaborishParams& gab = result.gab;
    if (gab_payload.size() == num_components * 4) {
      const uint8_t* data = gab_payload.data();
      for (size_t c = 0; c < num_components; ++c) {
        gab.w1[c] = data[c * 4];
        gab.w2[c] = data[c * 4 + 1];
        gab.threshold[c] = data[c * 4 + 2];
        gab.limit[c] = data[c * 4 + 3];
      }
      gab.active = true;
    } else {
      JXL_WARNING("ReJPEG: corrupted GAB extension data\n");
    }
  }

  return result;
}

// Update |data| to point to the start of the next section.
bool SkipSection(const uint8_t** data, size_t len) {
  size_t section_len = 0;
  uint64_t b = 0x80;
  size_t off = 1;
  for (size_t i = 0; (i < 9) && (b & 0x80u); ++i) {
    if (off >= len) return false;
    b = (*data)[off++];
    section_len |= (b & 0x7Fu) << (i * 7);
  }
  if ((b & 0x80u) != 0) return false;
  off += section_len;
  if (off > len) return false;
  *data += off;
  return true;
}

}  // namespace

YCbCrChromaSubsampling GetSubsamplingFromJpegData(
    const brunsli::JPEGData& jpg) {
  if (jpg.components.size() != 3) {
    return YCbCrChromaSubsampling::kAuto;
  }
  const size_t max_v_samp_factor = jpg.max_v_samp_factor;
  const size_t max_h_samp_factor = jpg.max_h_samp_factor;

  // Check consistency of chroma subsampling.
  for (size_t c = 0; c < 3; ++c) {
    const ::brunsli::JPEGComponent& component = jpg.components[c];
    if (c == 0) {  // Luma
      if (component.h_samp_factor != max_h_samp_factor) {
        return YCbCrChromaSubsampling::kAuto;
      }
      if (component.v_samp_factor != max_v_samp_factor) {
        return YCbCrChromaSubsampling::kAuto;
      }
    } else {  // Chroma
      if (component.h_samp_factor != 1) {
        return YCbCrChromaSubsampling::kAuto;
      }
      if (component.v_samp_factor != 1) {
        return YCbCrChromaSubsampling::kAuto;
      }
    }
  }

  if (max_h_samp_factor == 1 && max_v_samp_factor == 1) {
    return YCbCrChromaSubsampling::k444;
  }
  if (max_h_samp_factor == 2 && max_v_samp_factor == 2) {
    return YCbCrChromaSubsampling::k420;
  }
  if (max_h_samp_factor == 2 && max_v_samp_factor == 1) {
    return YCbCrChromaSubsampling::k422;
  }
  if (max_h_samp_factor == 4 && max_v_samp_factor == 1) {
    return YCbCrChromaSubsampling::k411;
  }
  return YCbCrChromaSubsampling::kAuto;
}

void SetColorEncodingFromJpegData(const brunsli::JPEGData& jpg,
                                  ColorEncoding* color_encoding) {
  PaddedBytes icc_profile;
  if (!ParseChunkedMarker(jpg, kApp2, ByteSpan(kIccProfileTag), &icc_profile)) {
    JXL_WARNING("ReJPEG: corrupted ICC profile\n");
    icc_profile.clear();
  }

  if (!color_encoding->SetICC(std::move(icc_profile))) {
    bool is_gray = (jpg.components.size() == 1);
    *color_encoding = ColorEncoding::SRGB(is_gray);
  }
}

Status BrunsliToPixels(const brunsli::JPEGData& jpg,
                       jxl::CodecInOut* JXL_RESTRICT io,
                       const BrunsliDecoderOptions& options,
                       BrunsliDecoderMeta* metadata, jxl::ThreadPool* pool) {
  BrunsliExtensions extensions = ParseBrunsliExtensions(jpg);
  if (options.fix_dc_staircase) extensions.dcc.active = true;
  if (options.gaborish) extensions.gab.active = true;

  Image3F rgb;
  JXL_RETURN_IF_ERROR(
      HWY_DYNAMIC_DISPATCH(JpegDataToPixels)(jpg, extensions, &rgb, pool));

  ColorEncoding color_encoding;
  if (!extensions.hdr_colorspace.empty()) {
    JXL_RETURN_IF_ERROR(
        ParseDescription(extensions.hdr_colorspace, &color_encoding));
    JXL_RETURN_IF_ERROR(color_encoding.CreateICC());
  } else {
    SetColorEncodingFromJpegData(jpg, &color_encoding);
  }

  // TODO(eustas): also import EXIF, etc.
  metadata->hdr_orig_colorspace = extensions.hdr_orig_colorspace;

  if (extensions.hdr_orig_colorspace.empty()) {
    io->metadata.SetUintSamples(8);
  } else {
    io->metadata.SetFloat32Samples();
  }
  io->metadata.color_encoding = color_encoding;
  io->SetFromImage(std::move(rgb), color_encoding);

  io->dec_pixels += jpg.width * jpg.height;

  return Map255ToTargetNits(io, pool);
}

// TODO(eustas): use VerifySignature from Brunsli library.
BrunsliFileSignature IsBrunsliFile(jxl::Span<const uint8_t> compressed) {
  if (memcmp(compressed.data(), kBrunsliMagic,
             std::min(kBrunsliMagicSize, compressed.size())) != 0) {
    return BrunsliFileSignature::kInvalid;
  }
  if (compressed.size() < kBrunsliMagicSize) {
    return BrunsliFileSignature::kNotEnoughData;
  }
  return BrunsliFileSignature::kBrunsli;
}

namespace {

const uint8_t kDefaultQuantMatrix[2][64] = {
    {16, 11, 10, 16, 24,  40,  51,  61,  12, 12, 14, 19, 26,  58,  60,  55,
     14, 13, 16, 24, 40,  57,  69,  56,  14, 17, 22, 29, 51,  87,  80,  62,
     18, 22, 37, 56, 68,  109, 103, 77,  24, 35, 55, 64, 81,  104, 113, 92,
     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99},
    {17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
     24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
     99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99}};

void FillQuantMatrix(bool is_chroma, float scale, int32_t* dst) {
  const uint8_t* const in = kDefaultQuantMatrix[is_chroma];
  for (uint32_t i = 0; i < kDCTBlockSize; ++i) {
    const uint32_t v = static_cast<float>(in[i]) * scale;
    dst[i] = (v < 1) ? 1 : (v > 255) ? 255u : v;
  }
}

// Convert YCbCr pixels to JPEGData components.
void ConvertPixels(const Image3F& from, brunsli::JPEGData* to,
                   const int quant_map[3], jxl::ThreadPool* pool) {
  constexpr size_t N = kBlockDim;
  // TODO(eustas): consider grayscale / CMYK.
  const size_t num_c = 3;

  for (size_t c = 0; c < num_c; ++c) {
    // TODO(eustas): use pool.
    const ImageF& plane = Dct8(from.Plane(c));
    ::brunsli::JPEGComponent component;
    component.id = static_cast<int>(c) + 1;  // YCbCr
    component.h_samp_factor = 1;
    component.v_samp_factor = 1;
    component.quant_idx = quant_map[c];
    const int* JXL_RESTRICT quant_table =
        to->quant[component.quant_idx].values.data();
    const size_t xsize_blocks = component.h_samp_factor * to->MCU_cols;
    component.width_in_blocks = xsize_blocks;
    const size_t ysize_blocks = component.v_samp_factor * to->MCU_rows;
    component.height_in_blocks = ysize_blocks;
    const size_t num_blocks = xsize_blocks * ysize_blocks;
    component.num_blocks = num_blocks;
    std::vector<::brunsli::coeff_t> coeffs(num_blocks * kDCTBlockSize);
    // TODO(eustas): use pool.
    for (size_t by = 0; by < ysize_blocks; ++by) {
      const float* JXL_RESTRICT from_row = plane.ConstRow(by);
      ::brunsli::coeff_t* JXL_RESTRICT to_row =
          &coeffs[by * xsize_blocks * kDCTBlockSize];
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        const float* JXL_RESTRICT from_block = &from_row[bx * kDCTBlockSize];
        ::brunsli::coeff_t* JXL_RESTRICT to_block = &to_row[bx * kDCTBlockSize];
        // TODO(eustas): SIMDify
        for (size_t u = 0; u < N; ++u) {
          for (size_t v = 0; v < N; ++v) {
            size_t to_idx = u * N + v;
            size_t from_idx = v * N + u;
            float raw = (static_cast<float>(N * N) * DCTScales<N>()[u] *
                         DCTScales<N>()[v]) *
                        from_block[from_idx];
            raw += std::copysign(0.5f * static_cast<float>(quant_table[to_idx]),
                                 raw);
            to_block[to_idx] = raw / static_cast<float>(quant_table[to_idx]);
          }
        }
      }
    }

    component.coeffs = std::move(coeffs);
    to->components.emplace_back(std::move(component));
  }
}

}  // namespace

Status PixelsToBrunsli(const jxl::CodecInOut* JXL_RESTRICT io,
                       jxl::PaddedBytes* compressed,
                       const BrunsliEncoderOptions& options,
                       jxl::ThreadPool* pool) {
  constexpr size_t N = kBlockDim;
  static_assert(N == 8, "JPEG block dim must be 8");
  static_assert(kDCTBlockSize == N * N, "JPEG block size must be 64");

  jxl::ImageBundle ib = io->Main().Copy();
  if (!ib.HasColor()) {
    return false;
  }
  JXL_RETURN_IF_ERROR(MapTargetNitsTo255(&ib, pool));
  const size_t num_c = 3;

  const size_t xsize = io->xsize();
  const size_t ysize = io->ysize();

  const Image3F& src = PadImageToMultiple(ib.color(), N);
  ImageF planes[3];
  for (auto& plane : planes) {
    plane = ImageF(src.xsize(), src.ysize());
  }
  RgbToYcbcr(src.Plane(0), src.Plane(1), src.Plane(2), &planes[0], &planes[1],
             &planes[2], pool);

  ::brunsli::JPEGData out;
  out.width = xsize;
  out.height = ysize;
  out.version = 0;           // regular Brunsli
  out.restart_interval = 0;  // don't care
  out.max_h_samp_factor = 1;
  out.max_v_samp_factor = 1;
  out.MCU_rows = DivCeil(out.height, out.max_v_samp_factor * N);
  out.MCU_cols = DivCeil(out.width, out.max_h_samp_factor * N);
  out.has_zero_padding_bit = false;

  out.app_data.emplace_back(kApp0Template, kApp0Template + kApp0TemplateSize);
  out.marker_order.push_back(kApp0);

  if (options.hdr_orig_colorspace.empty()) {
    // TODO(eustas): use "predefined profile" optimization, when eligible.
    if (!AddChunkedMarker(&out, kApp2, ByteSpan(kIccProfileTag),
                          io->metadata.color_encoding.ICC())) {
      JXL_ABORT("Brunsli: failed to add ICC profile\n");
    }
  } else {
    PaddedBytes hdr_payload;
    std::string colorspace = Description(ib.c_current());
    std::string payload = options.hdr_orig_colorspace + ">" + colorspace;
    hdr_payload.append(payload);
    if (!AddChunkedMarker(&out, kApp11, ByteSpan(kBrunsliXHdr), hdr_payload)) {
      JXL_ABORT("Brunsli: failed to add HDR extension\n");
    }
  }

  if (options.dcc.active) {
    const BrunsliDccParams& dcc = options.dcc;
    PaddedBytes dcc_payload;
    for (size_t c = 0; c < num_c; ++c) {
      dcc_payload.push_back(dcc.max_gap[c]);
      dcc_payload.push_back(dcc.min_step[c]);
    }
    if (!AddChunkedMarker(&out, kApp11, ByteSpan(kBrunsliXDcc), dcc_payload)) {
      JXL_ABORT("Brunsli: failed to add DCC extension\n");
    }
  }

  if (options.gab.active) {
    const BrunsliGaborishParams& gab = options.gab;
    PaddedBytes gab_payload;
    for (size_t c = 0; c < num_c; ++c) {
      gab_payload.push_back(gab.w1[c]);
      gab_payload.push_back(gab.w2[c]);
      gab_payload.push_back(gab.threshold[c]);
      gab_payload.push_back(gab.limit[c]);
    }
    if (!AddChunkedMarker(&out, kApp11, ByteSpan(kBrunsliXGab), gab_payload)) {
      JXL_ABORT("Brunsli: failed to add GAB extension\n");
    }
  }

  ::brunsli::JPEGQuantTable quant_luma;
  FillQuantMatrix(false, options.quant_scale, quant_luma.values.data());
  quant_luma.precision = 0;
  quant_luma.index = 0;
  quant_luma.is_last = false;
  out.quant.emplace_back(std::move(quant_luma));
  ::brunsli::JPEGQuantTable quant_chroma;
  FillQuantMatrix(true, options.quant_scale, quant_chroma.values.data());
  quant_chroma.precision = 0;
  quant_chroma.index = 1;
  quant_chroma.is_last = true;
  out.quant.emplace_back(std::move(quant_chroma));
  out.marker_order.push_back(kDqt);

  {
    Image3F pixels(std::move(planes[0]), std::move(planes[1]),
                   std::move(planes[2]));
    constexpr int kQuantMap[] = {0, 1, 1};
    ConvertPixels(pixels, &out, kQuantMap, pool);
  }

  out.marker_order.push_back(kSof0);

  ::brunsli::JPEGHuffmanCode dc_huff;
  // kJpegDCAlphabetSize + 1 = 13 = 3 + 10; 3/8 + 10/16 = 1
  dc_huff.counts[3] = 3;
  dc_huff.counts[4] = 10;
  for (size_t i = 0; i < ::brunsli::kJpegDCAlphabetSize; ++i) {
    dc_huff.values[i] = i;
  }
  dc_huff.values[::brunsli::kJpegDCAlphabetSize] =
      ::brunsli::kJpegHuffmanAlphabetSize;
  dc_huff.slot_id = 0 + kHuffmanCodeDcSlotOffset;
  dc_huff.is_last = false;
  out.huffman_code.emplace_back(std::move(dc_huff));
  ::brunsli::JPEGHuffmanCode ac_huff;
  // kJpegHuffmanAlphabetSize + 1 = 257 = 255 + 2; 255 / 256 + 2 / 512 = 1
  ac_huff.counts[8] = 255;
  ac_huff.counts[9] = 2;
  for (size_t i = 0; i < ::brunsli::kJpegHuffmanAlphabetSize; ++i) {
    ac_huff.values[i] = i;
  }
  ac_huff.values[::brunsli::kJpegHuffmanAlphabetSize] =
      ::brunsli::kJpegHuffmanAlphabetSize;
  ac_huff.slot_id = 0 + kHuffmanCodeAcSlotOffset;
  ac_huff.is_last = true;
  out.huffman_code.emplace_back(std::move(ac_huff));
  out.marker_order.push_back(kDht);

  ::brunsli::JPEGScanInfo scan_info;
  scan_info.Ss = 0;
  scan_info.Se = 63;
  scan_info.Ah = 0;
  scan_info.Al = 0;
  scan_info.num_components = num_c;
  for (size_t i = 0; i < num_c; ++i) {
    scan_info.components[i].comp_idx = i;
    scan_info.components[i].dc_tbl_idx = 0;
    scan_info.components[i].ac_tbl_idx = 0;
  }
  out.scan_info.emplace_back(std::move(scan_info));
  out.marker_order.push_back(kSos);

  out.marker_order.push_back(kEoi);

  size_t output_size = ::brunsli::GetMaximumBrunsliEncodedSize(out);
  std::vector<uint8_t> output(output_size);
  // TODO(eustas): introduce streaming API?
  if (!::brunsli::BrunsliEncodeJpeg(out, output.data(), &output_size)) {
    return false;
  }
  compressed->append(Span<uint8_t>(output.data(), output_size));

  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
