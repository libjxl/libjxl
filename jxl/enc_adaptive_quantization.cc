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

#include "jxl/enc_adaptive_quantization.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "jxl/ac_strategy.h"
#include "jxl/aux_out.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/fast_log.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/butteraugli/butteraugli.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/common.h"
#include "jxl/convolve.h"
#include "jxl/dct_scales.h"
#include "jxl/dec_cache.h"
#include "jxl/dec_group.h"
#include "jxl/dec_reconstruct.h"
#include "jxl/enc_butteraugli_comparator.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_dct.h"
#include "jxl/enc_group.h"
#include "jxl/enc_params.h"
#include "jxl/gauss_blur.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/multiframe.h"
#include "jxl/opsin_params.h"
#include "jxl/quant_weights.h"

bool FLAGS_log_search_state = false;
// If true, prints the quantization maps at each iteration.
bool FLAGS_dump_quant_state = false;

namespace jxl {
namespace {

static const float kQuant64[64] = {
    0.0,
    1.3627035631195761,
    3.75394603948046884,
    3.7552562789889619,
    3.738923115258169,
    3.2658460689961202,
    3.3644109735743761,
    3.407927898760664,
    3.3435304472886553,
    0.912932235161943,
    0.86013076305762415,
    3.2181857425503768,
    2.5344889514395561,
    1.1030971121374131,
    0.86654135268900234,
    1.1790809890137122,
    1.1325855042326589,
    2.1036916840563111,
    2.1662829197292539,
    1.0855127071756325,
    1.4837636416580766,
    0.79528165577401788,
    0.84472454415339238,
    1.4156418573711551,
    1.4477267104680289,
    1.3169293134214741,
    1.1435971405444745,
    0.44922571728240968,
    0.66437130514818066,
    0.56811538880030854,
    0.88663683723756648,
    0.75797820670535709,
    1.2141376049363837,
    1.8082860500557918,
    0.34785192011414331,
    0.86658716645380518,
    0.84457566277095242,
    0.365886424294040194,
    0.27942131498024103,
    0.48621389791680419,
    0.2848216363151751,
    0.37182998793445576,
    0.37191251587301016,
    0.3434673885351725,
    0.50423064557670205,
    0.30742301163233954,
    0.38866381308020725,
    0.46912603786452162,
    0.38233313062142565,
    0.41786679581133063,
    0.3695788988347995,
    0.63197592721834761,
    0.73351761078380862,
    0.40450080085057888,
    0.39291869306279764,
    0.51847970794880838,
    0.43559898503478928,
    0.32837421314561244,
    0.51318960017980164,
    0.79565824960993681,
    0.21456893657335676,
    0.47552975726356284,
    0.4411023380122629,
    0.32950033621891689,
};

void ComputeMask(float* JXL_RESTRICT out_pos) {
  constexpr float kBase = 1.522f;
  constexpr float kMul1 = 0.011521457315309454f;
  constexpr float kOffset1 = 0.0079611186521877063f;
  constexpr float kMul2 = -0.19590586155132378f;
  constexpr float kOffset2 = 0.074575093726693686f;
  const float val = *out_pos;
  // Avoid division by zero.
  const float div = std::max(val + kOffset1, 1e-3f);
  *out_pos = kBase + kMul1 / div + kMul2 / (val * val + kOffset2);
}

// Increase precision in 8x8 blocks that are complicated in DCT space.
void DctModulation(const size_t x, const size_t y, const ImageF& xyb,
                   const coeff_order_t* natural_coeff_order,
                   const float* JXL_RESTRICT dct_rescale,
                   float* JXL_RESTRICT out_pos) {
  HWY_ALIGN_MAX float dct[kDCTBlockSize] = {0};
  for (size_t dy = 0; dy < 8; ++dy) {
    const size_t yclamp = std::min(y + dy, xyb.ysize() - 1);
    const float* const JXL_RESTRICT row_in = xyb.Row(yclamp);
    for (size_t dx = 0; dx < 8; ++dx) {
      const size_t xclamp = std::min(x + dx, xyb.xsize() - 1);
      dct[dy * 8 + dx] = row_in[xclamp];
    }
  }
  TransposedScaledDCT8(dct);
  float entropyQL2 = 0.0f;
  float entropyQL4 = 0.0f;
  float entropyQL8 = 0.0f;
  for (size_t k = 1; k < kDCTBlockSize; ++k) {
    const coeff_order_t i = natural_coeff_order[k];
    float v = dct[i] * dct_rescale[i];
    v *= v;
    static const float kPow = 1.7656070913325459f;
    float q = std::pow(kQuant64[k], kPow);
    entropyQL2 += q * v;
    v *= v;
    entropyQL4 += q * v;
    v *= v;
    entropyQL8 += q * v;
  }
  entropyQL2 = std::sqrt(entropyQL2);
  entropyQL4 = std::sqrt(std::sqrt(entropyQL4));
  entropyQL8 = std::pow(entropyQL8, 0.125f);
  constexpr float mulQL2 = 0.00064095761586667813f;
  constexpr float mulQL4 = -0.93103691258798293f;
  constexpr float mulQL8 = 0.20682345500923968f;
  float v = mulQL2 * entropyQL2 + mulQL4 * entropyQL4 + mulQL8 * entropyQL8;
  constexpr float kMul = 1.0833857206487167f;
  *out_pos += kMul * v;
}

// Increase precision in 8x8 blocks that have high dynamic range.
void RangeModulation(const size_t x, const size_t y, const ImageF& xyb_x,
                     const ImageF& xyb_y, float* JXL_RESTRICT out_pos) {
  float minval_x = 1e30f;
  float minval_y = 1e30f;
  float maxval_x = -1e30f;
  float maxval_y = -1e30f;
  for (size_t dy = 0; dy < 8 && y + dy < xyb_x.ysize(); ++dy) {
    const float* const JXL_RESTRICT row_in_x = xyb_x.Row(y + dy);
    const float* const JXL_RESTRICT row_in_y = xyb_y.Row(y + dy);
    for (size_t dx = 0; dx < 8 && x + dx < xyb_x.xsize(); ++dx) {
      float vx = row_in_x[x + dx];
      float vy = row_in_y[x + dx];
      if (minval_x > vx) {
        minval_x = vx;
      }
      if (maxval_x < vx) {
        maxval_x = vx;
      }
      if (minval_y > vy) {
        minval_y = vy;
      }
      if (maxval_y < vy) {
        maxval_y = vy;
      }
    }
  }
  float range_x = maxval_x - minval_x;
  float range_y = maxval_y - minval_y;
  // This is not really a sound approach but it seems to yield better results
  // than the previous approach of just using range_y.
  float range = std::sqrt(range_x * range_y);
  constexpr float mul = 0.66697599699046262f;
  *out_pos += mul * range;
}

// Change precision in 8x8 blocks that have high frequency content.
void HfModulation(const size_t x, const size_t y, const ImageF& xyb,
                  float* JXL_RESTRICT out_pos) {
  float sum = 0;
  int n = 0;
  for (size_t dy = 0; dy < 8 && y + dy < xyb.ysize(); ++dy) {
    const float* JXL_RESTRICT row_in = xyb.Row(y + dy);
    for (size_t dx = 0; dx < 7 && x + dx + 1 < xyb.xsize(); ++dx) {
      float v = std::fabs(row_in[x + dx] - row_in[x + dx + 1]);
      sum += v;
      ++n;
    }
  }
  for (size_t dy = 0; dy < 7 && y + dy + 1 < xyb.ysize(); ++dy) {
    const float* JXL_RESTRICT row_in = xyb.Row(y + dy);
    const float* JXL_RESTRICT row_in_next = xyb.Row(y + dy + 1);
    for (size_t dx = 0; dx < 8 && x + dx < xyb.xsize(); ++dx) {
      float v = std::abs(row_in[x + dx] - row_in_next[x + dx]);
      sum += v;
      ++n;
    }
  }
  if (n != 0) {
    sum /= n;
  }
  constexpr float kMul = 0.70810081505707823f;
  sum *= kMul;
  *out_pos += sum;
}

void PerBlockModulations(const ImageF& xyb_x, const ImageF& xyb_y,
                         const float scale, ThreadPool* pool, ImageF* out) {
  JXL_ASSERT(DivCeil(xyb_x.xsize(), kBlockDim) == out->xsize());
  JXL_ASSERT(DivCeil(xyb_x.ysize(), kBlockDim) == out->ysize());
  JXL_ASSERT(DivCeil(xyb_y.xsize(), kBlockDim) == out->xsize());
  JXL_ASSERT(DivCeil(xyb_y.ysize(), kBlockDim) == out->ysize());
  const coeff_order_t* natural_coeff_order =
      AcStrategy::FromRawStrategy(AcStrategy::Type::DCT).NaturalCoeffOrder();
  float dct_rescale[kDCTBlockSize] = {0};
  {
    const float* dct_scale = DCTScales<8>();
    for (size_t i = 0; i < kDCTBlockSize; ++i) {
      dct_rescale[i] = dct_scale[i / 8] * dct_scale[i % 8];
    }
  }

  RunOnPool(
      pool, 0, static_cast<int>(DivCeil(xyb_x.ysize(), kBlockDim)),
      ThreadPool::SkipInit(),
      [&](const int task, const int /*thread*/) {
        const size_t iy = static_cast<size_t>(task);
        const size_t y = iy * 8;
        float* const JXL_RESTRICT row_out = out->Row(iy);
        for (size_t x = 0; x < xyb_x.xsize(); x += 8) {
          float* JXL_RESTRICT out_pos = row_out + x / 8;
          ComputeMask(out_pos);
          DctModulation(x, y, xyb_y, natural_coeff_order, dct_rescale, out_pos);
          RangeModulation(x, y, xyb_x, xyb_y, out_pos);
          HfModulation(x, y, xyb_y, out_pos);

          // We want multiplicative quantization field, so everything
          // until this point has been modulating the exponent.
          *out_pos = std::exp(*out_pos) * scale;
        }
      },
      "AQ per block modulation");
}

static float SimpleGamma(float v) {
  // A simple HDR compatible gamma function.
  // mul and mul2 represent a scaling difference between jxl and butteraugli.
  static const float mul = 103.72874071313939f;
  static const float mul2 = 1 / 67.781877502322729f;

  v *= mul;

  // Includes correction factor for std::log -> log2.
  static const float kRetMul = mul2 * 18.6580932135f * 0.693147181f;
  static const float kRetAdd = mul2 * -20.2789020414f;
  static const float kVOffset = 7.14672470003f;

  if (v < 0) {
    // This should happen rarely, but may lead to a NaN, which is rather
    // undesirable. Since negative photons don't exist we solve the NaNs by
    // clamping here.
    v = 0;
  }
  return kRetMul * FastLog2f(v + kVOffset) + kRetAdd;
}

static float RatioOfCubicRootToSimpleGamma(float v) {
  // The opsin space in jxl is the cubic root of photons, i.e., v * v * v
  // is related to the number of photons.
  //
  // SimpleGamma(v * v * v) is the psychovisual space in butteraugli.
  // This ratio allows quantization to move from jxl's opsin space to
  // butteraugli's log-gamma space.
  return v / SimpleGamma(v * v * v);
}

// Returns image (padded to multiple of 8x8) of local pixel differences.
ImageF DiffPrecompute(const Image3F& xyb, const FrameDimensions& frame_dim,
                      float cutoff, ThreadPool* pool) {
  PROFILER_ZONE("aq DiffPrecompute");
  const size_t xsize = frame_dim.xsize;
  const size_t ysize = frame_dim.ysize;
  const size_t padded_xsize = RoundUpToBlockDim(xsize);
  const size_t padded_ysize = RoundUpToBlockDim(ysize);
  ImageF padded_diff(padded_xsize, padded_ysize);
  constexpr float mul0 = 0.046072108343079003f;

  // The XYB gamma is 3.0 to be able to decode faster with two muls.
  // Butteraugli's gamma is matching the gamma of human eye, around 2.6.
  // We approximate the gamma difference by adding one cubic root into
  // the adaptive quantization. This gives us a total gamma of 2.6666
  // for quantization uses.
  constexpr float match_gamma_offset = 0.25084239333070085f;

  RunOnPool(
      pool, 0, static_cast<int>(ysize), ThreadPool::SkipInit(),
      [&](const int task, int /*thread*/) {
        const size_t y = static_cast<size_t>(task);
        size_t y2;
        if (y + 1 < ysize) {
          y2 = y + 1;
        } else if (y > 0) {
          y2 = y - 1;
        } else {
          y2 = y;
        }
        size_t y1;
        if (y == 0 && ysize >= 2) {
          y1 = y + 1;
        } else if (y > 0) {
          y1 = y - 1;
        } else {
          y1 = y;
        }
        const float* row_in = xyb.PlaneRow(1, y);
        const float* row_in1 = xyb.PlaneRow(1, y1);
        const float* row_in2 = xyb.PlaneRow(1, y2);
        float* JXL_RESTRICT row_out = padded_diff.Row(y);

        size_t x = 0;
        // First pixel of the row.
        {
          const size_t x2 = (xsize < 1) ? 0 : 1;
          const size_t x1 = x2;
          float diff = mul0 * (std::abs(row_in[x] - row_in[x2]) +
                               std::abs(row_in[x] - row_in2[x]) +
                               std::abs(row_in[x] - row_in[x1]) +
                               std::abs(row_in[x] - row_in1[x]) +
                               3 * (std::abs(row_in2[x] - row_in1[x]) +
                                    std::abs(row_in[x1] - row_in[x2])));
          diff *= RatioOfCubicRootToSimpleGamma(row_in[x] + match_gamma_offset);
          row_out[x] = std::min(cutoff, diff);
          ++x;
        }
        for (; x + 1 < xsize; ++x) {
          const size_t x2 = x + 1;
          const size_t x1 = x - 1;
          float diff = mul0 * (std::abs(row_in[x] - row_in[x2]) +
                               std::abs(row_in[x] - row_in2[x]) +
                               std::abs(row_in[x] - row_in[x1]) +
                               std::abs(row_in[x] - row_in1[x]) +
                               3 * (std::abs(row_in2[x] - row_in1[x]) +
                                    std::abs(row_in[x1] - row_in[x2])));
          diff *= RatioOfCubicRootToSimpleGamma(row_in[x] + match_gamma_offset);
          row_out[x] = std::min(cutoff, diff);
        }
        // Last pixel of the row.
        {
          float diff = 7.0f * mul0 * (std::abs(row_in[x] - row_in2[x]));
          diff *= RatioOfCubicRootToSimpleGamma(row_in[x] + match_gamma_offset);
          row_out[x] = std::min(cutoff, diff);
          ++x;
        }

        // Extend to multiple of 8 columns
        float lastval = row_out[xsize - 1];
        if (xsize >= 3) {
          lastval += row_out[xsize - 3];
          lastval += row_out[xsize - 2];
          lastval *= 1.0f / 3;
        } else if (xsize >= 2) {
          lastval += row_out[xsize - 2];
          lastval *= 0.5f;
        }
        for (; x < padded_diff.xsize(); ++x) {
          row_out[x] = lastval;
        }
      },
      "AQ DiffPrecompute");

  // Last row.
  {
    const size_t y = ysize - 1;
    const float* const JXL_RESTRICT row_in = xyb.PlaneRow(1, y);
    float* const JXL_RESTRICT row_out = padded_diff.Row(y);
    for (size_t x = 0; x + 1 < xsize; ++x) {
      const size_t x2 = x + 1;
      float diff = 7.0f * mul0 * std::abs(row_in[x] - row_in[x2]);
      diff *= RatioOfCubicRootToSimpleGamma(row_in[x] + match_gamma_offset);
      row_out[x] = std::min(cutoff, diff);
    }
    // Last pixel of the last row.
    {
      const size_t x = xsize - 1;
      if (x > 0) {
        row_out[x] = row_out[x - 1];
      }
    }
  }
  // Extend to multiple of 8 rows
  if (ysize != padded_diff.ysize()) {
    const float* JXL_RESTRICT last_row = padded_diff.Row(ysize - 1);
    for (size_t x = 0; x < padded_diff.xsize(); ++x) {
      float lastval = last_row[x];
      if (ysize >= 3) {
        lastval += padded_diff.Row(ysize - 2)[x];
        lastval += padded_diff.Row(ysize - 3)[x];
        lastval *= 1.0f / 3;
      } else if (ysize >= 2) {
        lastval += padded_diff.Row(ysize - 2)[x];
        lastval *= 0.5f;
      }
      for (size_t y = ysize; y < padded_diff.ysize(); ++y) {
        padded_diff.Row(y)[x] = lastval;
      }
    }
  }

  return padded_diff;
}

ImageF TileDistMap(const ImageF& distmap, int tile_size, int margin,
                   const AcStrategyImage& ac_strategy) {
  PROFILER_FUNC;
  const int tile_xsize = (distmap.xsize() + tile_size - 1) / tile_size;
  const int tile_ysize = (distmap.ysize() + tile_size - 1) / tile_size;
  ImageF tile_distmap(tile_xsize, tile_ysize);
  size_t distmap_stride = tile_distmap.PixelsPerRow();
  for (int tile_y = 0; tile_y < tile_ysize; ++tile_y) {
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(tile_y);
    float* JXL_RESTRICT dist_row = tile_distmap.Row(tile_y);
    for (int tile_x = 0; tile_x < tile_xsize; ++tile_x) {
      AcStrategy acs = ac_strategy_row[tile_x];
      if (!acs.IsFirstBlock()) continue;
      int this_tile_xsize = acs.covered_blocks_x() * tile_size;
      int this_tile_ysize = acs.covered_blocks_y() * tile_size;
      int y_begin = std::max<int>(0, tile_size * tile_y - margin);
      int y_end = std::min<int>(distmap.ysize(),
                                tile_size * tile_y + this_tile_ysize + margin);
      int x_begin = std::max<int>(0, tile_size * tile_x - margin);
      int x_end = std::min<int>(distmap.xsize(),
                                tile_size * tile_x + this_tile_xsize + margin);
      float dist_norm = 0.0;
      double pixels = 0;
      for (int y = y_begin; y < y_end; ++y) {
        float ymul = 1.0;
        static const float kBorderMul = 0.98f;
        static const float kCornerMul = 0.7f;
        if (margin != 0 && (y == y_begin || y == y_end - 1)) {
          ymul = kBorderMul;
        }
        const float* const JXL_RESTRICT row = distmap.Row(y);
        for (int x = x_begin; x < x_end; ++x) {
          float xmul = ymul;
          if (margin != 0 && (x == x_begin || x == x_end - 1)) {
            if (xmul == 1.0) {
              xmul = kBorderMul;
            } else {
              xmul = kCornerMul;
            }
          }
          float v = row[x];
          v *= v;
          v *= v;
          v *= v;
          v *= v;
          dist_norm += xmul * v;
          pixels += xmul;
        }
      }
      if (pixels == 0) pixels = 1;
      // 16th norm is less than the max norm, we reduce the difference
      // with this normalization factor.
      static const double kTileNorm = 1.2;
      const double tile_dist =
          kTileNorm * std::pow(dist_norm / pixels, 1.0 / 16);
      dist_row[tile_x] = tile_dist;
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          dist_row[tile_x + distmap_stride * iy + ix] = tile_dist;
        }
      }
    }
  }
  return tile_distmap;
}

ImageF DistToPeakMap(const ImageF& field, float peak_min, int local_radius,
                     float peak_weight) {
  ImageF result(field.xsize(), field.ysize());
  FillImage(-1.0f, &result);
  for (size_t y0 = 0; y0 < field.ysize(); ++y0) {
    for (size_t x0 = 0; x0 < field.xsize(); ++x0) {
      int x_min = std::max<int>(0, static_cast<int>(x0) - local_radius);
      int y_min = std::max<int>(0, static_cast<int>(y0) - local_radius);
      int x_max = std::min<size_t>(field.xsize(), x0 + 1 + local_radius);
      int y_max = std::min<size_t>(field.ysize(), y0 + 1 + local_radius);
      float local_max = peak_min;
      for (int y = y_min; y < y_max; ++y) {
        for (int x = x_min; x < x_max; ++x) {
          local_max = std::max(local_max, field.Row(y)[x]);
        }
      }
      if (field.Row(y0)[x0] >
          (1.0f - peak_weight) * peak_min + peak_weight * local_max) {
        for (int y = y_min; y < y_max; ++y) {
          for (int x = x_min; x < x_max; ++x) {
            float dist = std::max(std::abs(y - static_cast<int>(y0)),
                                  std::abs(x - static_cast<int>(x0)));
            float cur_dist = result.Row(y)[x];
            if (cur_dist < 0.0 || cur_dist > dist) {
              result.Row(y)[x] = dist;
            }
          }
        }
      }
    }
  }
  return result;
}

bool AdjustQuantVal(float* const JXL_RESTRICT q, const float d,
                    const float factor, const float quant_max) {
  if (*q >= 0.999f * quant_max) return false;
  const float inv_q = 1.0f / *q;
  const float adj_inv_q = inv_q - factor / (d + 1.0f);
  *q = 1.0f / std::max(1.0f / quant_max, adj_inv_q);
  return true;
}

void DumpHeatmap(const AuxOut* aux_out, const std::string& label,
                 const ImageF& image, float good_threshold,
                 float bad_threshold) {
  Image3B heatmap = CreateHeatMapImage(image, good_threshold, bad_threshold);
  char filename[200];
  snprintf(filename, sizeof(filename), "%s%05d", label.c_str(),
           aux_out->num_butteraugli_iters);
  aux_out->DumpImage(filename, heatmap);
}

void DumpHeatmaps(const AuxOut* aux_out, float ba_target,
                  const ImageF& quant_field, const ImageF& tile_heatmap) {
  if (!WantDebugOutput(aux_out)) return;
  ImageF inv_qmap(quant_field.xsize(), quant_field.ysize());
  for (size_t y = 0; y < quant_field.ysize(); ++y) {
    const float* JXL_RESTRICT row_q = quant_field.ConstRow(y);
    float* JXL_RESTRICT row_inv_q = inv_qmap.Row(y);
    for (size_t x = 0; x < quant_field.xsize(); ++x) {
      row_inv_q[x] = 1.0f / row_q[x];  // never zero
    }
  }
  DumpHeatmap(aux_out, "quant_heatmap", inv_qmap, 4.0f * ba_target,
              6.0f * ba_target);
  DumpHeatmap(aux_out, "tile_heatmap", tile_heatmap, ba_target,
              1.5f * ba_target);
}

void AdjustQuantField(const AcStrategyImage& ac_strategy, ImageF* quant_field) {
  // Replace the whole quant_field in non-8x8 blocks with the maximum of each
  // 8x8 block.
  size_t stride = quant_field->PixelsPerRow();
  for (size_t y = 0; y < quant_field->ysize(); ++y) {
    AcStrategyRow ac_strategy_row = ac_strategy.ConstRow(y);
    float* JXL_RESTRICT quant_row = quant_field->Row(y);
    for (size_t x = 0; x < quant_field->xsize(); ++x) {
      AcStrategy acs = ac_strategy_row[x];
      if (!acs.IsFirstBlock()) continue;
      JXL_ASSERT(x + acs.covered_blocks_x() <= quant_field->xsize());
      JXL_ASSERT(y + acs.covered_blocks_y() <= quant_field->ysize());
      float max = quant_row[x];
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          max = std::max(quant_row[x + ix + iy * stride], max);
        }
      }
      for (size_t iy = 0; iy < acs.covered_blocks_y(); iy++) {
        for (size_t ix = 0; ix < acs.covered_blocks_x(); ix++) {
          quant_row[x + ix + iy * stride] = max;
        }
      }
    }
  }
}

static const float kDcQuantPow = 0.55;
static const float kDcQuant = 1.13;
static const float kAcQuant = 0.84;

void FindBestQuantization(const ImageBundle& linear, const Image3F& opsin,
                          PassesEncoderState* enc_state, ThreadPool* pool,
                          AuxOut* aux_out) {
  const CompressParams& cparams = enc_state->cparams;
  Quantizer& quantizer = enc_state->shared.quantizer;
  ImageI& raw_quant_field = enc_state->shared.raw_quant_field;
  ImageF& quant_field = enc_state->initial_quant_field;

  const float butteraugli_target = cparams.butteraugli_distance;
  JxlButteraugliComparator comparator(cparams.hf_asymmetry);
  ImageMetadata metadata;
  JXL_CHECK(comparator.SetReferenceImage(linear));
  bool lower_is_better =
      (comparator.GoodQualityScore() < comparator.BadQualityScore());
  const float initial_quant_dc = InitialQuantDC(butteraugli_target);
  AdjustQuantField(enc_state->shared.ac_strategy, &quant_field);
  ImageF tile_distmap;
  ImageF tile_distmap_localopt;
  ImageF initial_quant_field = CopyImage(quant_field);
  ImageF last_quant_field = CopyImage(initial_quant_field);
  ImageF last_tile_distmap_localopt;

  float initial_qf_min, initial_qf_max;
  ImageMinMax(initial_quant_field, &initial_qf_min, &initial_qf_max);
  float initial_qf_ratio = initial_qf_max / initial_qf_min;
  float qf_max_deviation_low = std::sqrt(250 / initial_qf_ratio);
  float asymmetry = 2;
  if (qf_max_deviation_low < asymmetry) asymmetry = qf_max_deviation_low;
  float qf_lower = initial_qf_min / (asymmetry * qf_max_deviation_low);
  float qf_higher = initial_qf_max * (qf_max_deviation_low / asymmetry);

  JXL_ASSERT(qf_higher / qf_lower < 253);

  constexpr int kOriginalComparisonRound = 1;
  constexpr float kMaximumDistanceIncreaseFactor = 1.015;

  for (int i = 0; i < cparams.max_butteraugli_iters + 1; ++i) {
    if (FLAGS_dump_quant_state) {
      printf("\nQuantization field:\n");
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
    }

    quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field);
    ImageMetadata metadata;
    metadata.bits_per_sample = 32;
    metadata.floating_point_sample = true;
    metadata.color_encoding = ColorEncoding::LinearSRGB();
    ImageBundle linear(&metadata);
    linear.SetFromImage(RoundtripImage(opsin, enc_state, pool),
                        metadata.color_encoding);
    PROFILER_ZONE("enc Butteraugli");
    float score;
    ImageF diffmap;
    JXL_CHECK(comparator.CompareWith(linear, &diffmap, &score));
    if (!lower_is_better) {
      score = -score;
      diffmap = ScaleImage(-1.0f, diffmap);
    }
    static const int kMargins[100] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    tile_distmap =
        TileDistMap(diffmap, 8, kMargins[i], enc_state->shared.ac_strategy);
    tile_distmap_localopt =
        TileDistMap(diffmap, 8, 2, enc_state->shared.ac_strategy);
    if (WantDebugOutput(aux_out)) {
      DumpHeatmaps(aux_out, butteraugli_target, quant_field, tile_distmap);
    }
    if (aux_out != nullptr) ++aux_out->num_butteraugli_iters;
    if (FLAGS_log_search_state) {
      float minval, maxval;
      ImageMinMax(quant_field, &minval, &maxval);
      printf("\nButteraugli iter: %d/%d\n", i, cparams.max_butteraugli_iters);
      printf("Butteraugli distance: %f\n", score);
      printf("quant range: %f ... %f  DC quant: %f\n", minval, maxval,
             initial_quant_dc);
      if (FLAGS_dump_quant_state) {
        quantizer.DumpQuantizationMap(raw_quant_field);
      }
    }

    if (i > kOriginalComparisonRound) {
      // Undo last round if it made things worse (i.e. increased the quant value
      // AND the distance in nearby pixels by at least some percentage).
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        const float* const JXL_RESTRICT row_dist = tile_distmap_localopt.Row(y);
        const float* const JXL_RESTRICT row_last_dist =
            last_tile_distmap_localopt.Row(y);
        const float* const JXL_RESTRICT row_last_q = last_quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          if (row_q[x] > row_last_q[x] &&
              row_dist[x] > kMaximumDistanceIncreaseFactor * row_last_dist[x]) {
            row_q[x] = row_last_q[x];
          }
        }
      }
    }
    last_quant_field = CopyImage(quant_field);
    last_tile_distmap_localopt = CopyImage(tile_distmap_localopt);
    if (i == cparams.max_butteraugli_iters) break;

    double kPow[8] = {
        0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    double kPowMod[8] = {
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };
    if (i == kOriginalComparisonRound) {
      // Don't allow optimization to make the quant field a lot worse than
      // what the initial guess was. This allows the AC field to have enough
      // precision to reduce the oscillations due to the dc reconstruction.
      double kInitMul = 0.6;
      const double kOneMinusInitMul = 1.0 - kInitMul;
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        const float* const JXL_RESTRICT row_init = initial_quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          double clamp = kOneMinusInitMul * row_q[x] + kInitMul * row_init[x];
          if (row_q[x] < clamp) {
            row_q[x] = clamp;
            if (row_q[x] > qf_higher) row_q[x] = qf_higher;
            if (row_q[x] < qf_lower) row_q[x] = qf_lower;
          }
        }
      }
    }

    double cur_pow = 0.0;
    if (i < 7) {
      cur_pow = kPow[i] + (butteraugli_target - 1.0) * kPowMod[i];
      if (cur_pow < 0) {
        cur_pow = 0;
      }
    }
    // pow(x, 0) == 1, so skip pow.
    if (cur_pow == 0.0) {
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        const float* const JXL_RESTRICT row_dist = tile_distmap.Row(y);
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          const float diff = row_dist[x] / butteraugli_target;
          if (diff > 1.0f) {
            float old = row_q[x];
            row_q[x] *= diff;
            int qf_old = old * quantizer.InvGlobalScale() + 0.5;
            int qf_new = row_q[x] * quantizer.InvGlobalScale() + 0.5;
            if (qf_old == qf_new) {
              row_q[x] = old + quantizer.Scale();
            }
          }
          if (row_q[x] > qf_higher) row_q[x] = qf_higher;
          if (row_q[x] < qf_lower) row_q[x] = qf_lower;
        }
      }
    } else {
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        const float* const JXL_RESTRICT row_dist = tile_distmap.Row(y);
        float* const JXL_RESTRICT row_q = quant_field.Row(y);
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          const float diff = row_dist[x] / butteraugli_target;
          if (diff <= 1.0f) {
            row_q[x] *= std::pow(diff, cur_pow);
          } else {
            float old = row_q[x];
            row_q[x] *= diff;
            int qf_old = old * quantizer.InvGlobalScale() + 0.5;
            int qf_new = row_q[x] * quantizer.InvGlobalScale() + 0.5;
            if (qf_old == qf_new) {
              row_q[x] = old + quantizer.Scale();
            }
          }
          if (row_q[x] > qf_higher) row_q[x] = qf_higher;
          if (row_q[x] < qf_lower) row_q[x] = qf_lower;
        }
      }
    }
  }
  quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field);
}

void FindBestQuantizationMaxError(const Image3F& opsin,
                                  PassesEncoderState* enc_state,
                                  ThreadPool* pool, AuxOut* aux_out) {
  const CompressParams& cparams = enc_state->cparams;
  Quantizer& quantizer = enc_state->shared.quantizer;
  ImageI& raw_quant_field = enc_state->shared.raw_quant_field;
  ImageF& quant_field = enc_state->initial_quant_field;

  // TODO(veluca): better choice of this value.
  const float initial_quant_dc =
      16 * std::sqrt(0.1f / cparams.butteraugli_distance);
  AdjustQuantField(enc_state->shared.ac_strategy, &quant_field);

  const float inv_max_err[3] = {1.0f / enc_state->cparams.max_error[0],
                                1.0f / enc_state->cparams.max_error[1],
                                1.0f / enc_state->cparams.max_error[2]};

  for (int i = 0; i < cparams.max_butteraugli_iters + 1; ++i) {
    quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field);
    if (aux_out)
      aux_out->DumpXybImage(("ops" + std::to_string(i)).c_str(), opsin);
    Image3F decoded =
        RoundtripImage(opsin, enc_state, pool, /*save_decompressed=*/false,
                       /*apply_color_transform=*/false);
    if (aux_out)
      aux_out->DumpXybImage(("dec" + std::to_string(i)).c_str(), decoded);

    for (size_t by = 0; by < enc_state->shared.frame_dim.ysize_blocks; by++) {
      AcStrategyRow ac_strategy_row =
          enc_state->shared.ac_strategy.ConstRow(by);
      for (size_t bx = 0; bx < enc_state->shared.frame_dim.xsize_blocks; bx++) {
        AcStrategy acs = ac_strategy_row[bx];
        if (!acs.IsFirstBlock()) continue;
        float max_error = 0;
        for (size_t c = 0; c < 3; c++) {
          for (size_t y = by * kBlockDim;
               y < (by + acs.covered_blocks_y()) * kBlockDim; y++) {
            if (y >= decoded.ysize()) continue;
            const float* JXL_RESTRICT in_row = opsin.ConstPlaneRow(c, y);
            const float* JXL_RESTRICT dec_row = decoded.ConstPlaneRow(c, y);
            for (size_t x = bx * kBlockDim;
                 x < (bx + acs.covered_blocks_x()) * kBlockDim; x++) {
              if (x >= decoded.xsize()) continue;
              max_error = std::max(
                  std::abs(in_row[x] - dec_row[x]) * inv_max_err[c], max_error);
            }
          }
        }
        // Target an error between max_error/2 and max_error.
        // If the error in the varblock is above the target, increase the qf to
        // compensate. If the error is below the target, decrease the qf.
        // However, to avoid an excessive increase of the qf, only do so if the
        // error is less than half the maximum allowed error.
        float qf_mul = max_error < 0.5f ? max_error * 2.0f
                                        : max_error > 1.0f ? max_error : 1.0f;
        for (size_t qy = by; qy < by + acs.covered_blocks_y(); qy++) {
          float* JXL_RESTRICT quant_field_row = quant_field.Row(qy);
          for (size_t qx = bx; qx < bx + acs.covered_blocks_x(); qx++) {
            quant_field_row[qx] *= qf_mul;
          }
        }
      }
    }
  }
  quantizer.SetQuantField(initial_quant_dc, quant_field, &raw_quant_field);
}

void FindBestQuantizationHQ(const ImageBundle& linear, const Image3F& opsin,
                            PassesEncoderState* enc_state, ThreadPool* pool,
                            AuxOut* aux_out) {
  const CompressParams& cparams = enc_state->cparams;
  Quantizer& quantizer = enc_state->shared.quantizer;
  ImageI& raw_quant_field = enc_state->shared.raw_quant_field;
  ImageF& quant_field = enc_state->initial_quant_field;
  const AcStrategyImage& ac_strategy = enc_state->shared.ac_strategy;

  JxlButteraugliComparator comparator(cparams.hf_asymmetry);
  ImageMetadata metadata;
  JXL_CHECK(comparator.SetReferenceImage(linear));
  AdjustQuantField(ac_strategy, &quant_field);
  ImageF best_quant_field = CopyImage(quant_field);
  bool lower_is_better =
      (comparator.GoodQualityScore() < comparator.BadQualityScore());
  float best_score = 1000000.0f;
  ImageF tile_distmap;
  static const int kMaxOuterIters = 2;
  int outer_iter = 0;
  int butteraugli_iter = 0;
  int search_radius = 0;
  float quant_ceil = 5.0f;
  float quant_dc = 1.2f;
  float best_quant_dc = quant_dc;
  int num_stalling_iters = 0;
  int max_iters = cparams.max_butteraugli_iters_guetzli_mode;
  const float butteraugli_target = cparams.butteraugli_distance;

  for (;;) {
    if (FLAGS_dump_quant_state) {
      printf("\nQuantization field:\n");
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          printf(" %.5f", quant_field.Row(y)[x]);
        }
        printf("\n");
      }
    }
    float qmin, qmax;
    ImageMinMax(quant_field, &qmin, &qmax);
    ++butteraugli_iter;
    float score = 0.0;
    ImageF diffmap;
    quantizer.SetQuantField(quant_dc, quant_field, &raw_quant_field);
    ImageMetadata metadata;
    metadata.bits_per_sample = 32;
    metadata.floating_point_sample = true;
    metadata.color_encoding = ColorEncoding::LinearSRGB();
    ImageBundle linear(&metadata);
    linear.SetFromImage(RoundtripImage(opsin, enc_state, pool),
                        metadata.color_encoding);
    JXL_CHECK(comparator.CompareWith(linear, &diffmap, &score));

    if (!lower_is_better) {
      score = -score;
      ScaleImage(-1.0f, &diffmap);
    }
    bool best_quant_updated = false;
    if (score <= best_score) {
      best_quant_field = CopyImage(quant_field);
      best_score = std::max<float>(score, butteraugli_target);
      best_quant_updated = true;
      best_quant_dc = quant_dc;
      num_stalling_iters = 0;
    } else if (outer_iter == 0) {
      ++num_stalling_iters;
    }
    tile_distmap = TileDistMap(diffmap, 8, 0, ac_strategy);
    if (WantDebugOutput(aux_out)) {
      DumpHeatmaps(aux_out, butteraugli_target, quant_field, tile_distmap);
    }
    if (aux_out) {
      ++aux_out->num_butteraugli_iters;
    }
    if (FLAGS_log_search_state) {
      float minval, maxval;
      ImageMinMax(quant_field, &minval, &maxval);
      printf("\nButteraugli iter: %d/%d%s\n", butteraugli_iter, max_iters,
             best_quant_updated ? " (*)" : "");
      printf("Butteraugli distance: %f\n", score);
      printf(
          "quant range: %f ... %f  DC quant: "
          "%f\n",
          minval, maxval, quant_dc);
      printf("search radius: %d\n", search_radius);
      if (FLAGS_dump_quant_state) {
        quantizer.DumpQuantizationMap(raw_quant_field);
      }
    }
    if (butteraugli_iter >= max_iters) {
      break;
    }
    bool changed = false;
    while (!changed && score > butteraugli_target) {
      for (int radius = 0; radius <= search_radius && !changed; ++radius) {
        ImageF dist_to_peak_map =
            DistToPeakMap(tile_distmap, butteraugli_target, radius, 0.0);
        for (size_t y = 0; y < quant_field.ysize(); ++y) {
          float* const JXL_RESTRICT row_q = quant_field.Row(y);
          const float* const JXL_RESTRICT row_dist = dist_to_peak_map.Row(y);
          for (size_t x = 0; x < quant_field.xsize(); ++x) {
            if (row_dist[x] >= 0.0f) {
              static const float kAdjSpeed[kMaxOuterIters] = {0.1f, 0.04f};
              const float factor =
                  kAdjSpeed[outer_iter] * tile_distmap.Row(y)[x];
              if (AdjustQuantVal(&row_q[x], row_dist[x], factor, quant_ceil)) {
                changed = true;
              }
            }
          }
        }
      }
      if (!changed || num_stalling_iters >= 3) {
        // Try to extend the search parameters.
        if ((search_radius < 4) &&
            (qmax < 0.99f * quant_ceil || quant_ceil >= 3.0f + search_radius)) {
          ++search_radius;
          continue;
        }
        if (quant_dc < 0.4f * quant_ceil - 0.8f) {
          quant_dc += 0.2f;
          changed = true;
          continue;
        }
        if (quant_ceil < 8.0f) {
          quant_ceil += 0.5f;
          continue;
        }
        break;
      }
    }
    if (!changed) {
      if (++outer_iter == kMaxOuterIters) break;
      static const float kQuantScale = 0.75f;
      for (size_t y = 0; y < quant_field.ysize(); ++y) {
        for (size_t x = 0; x < quant_field.xsize(); ++x) {
          quant_field.Row(y)[x] *= kQuantScale;
        }
      }
      num_stalling_iters = 0;
    }
  }
  quantizer.SetQuantField(best_quant_dc, best_quant_field, &raw_quant_field);
}

ImageF AdaptiveQuantizationMap(const Image3F& opsin,
                               const ImageF& intensity_ac_x,
                               const ImageF& intensity_ac_y,
                               const FrameDimensions& frame_dim, float scale,
                               ThreadPool* pool) {
  PROFILER_ZONE("aq AdaptiveQuantMap");
  const float kSigma = 8.2553856725566153f;
  static const int kRadius = static_cast<int>(2 * kSigma + 0.5f);
  std::vector<float> kernel = GaussianKernel(kRadius, kSigma);

  constexpr float kDiffCutoff = 0.11883287948847132f;
  ImageF out = DiffPrecompute(opsin, frame_dim, kDiffCutoff, pool);
  JXL_ASSERT(out.xsize() % kBlockDim == 0 && out.ysize() % kBlockDim == 0);
  out = ConvolveAndSample(out, kernel, kBlockDim);
  PerBlockModulations(intensity_ac_x, intensity_ac_y, scale, pool, &out);
  return out;
}

const WeightsSymmetric3& WeightsSymmetric3GaussianDC() {
  constexpr float w0 = 0.320356f;
  constexpr float w1 = 0.122822f;
  constexpr float w2 = 0.047089f;
  static constexpr WeightsSymmetric3 weights = {
      {HWY_REP4(w0)}, {HWY_REP4(w1)}, {HWY_REP4(w2)}};
  return weights;
}

ImageF IntensityAcEstimate(const ImageF& opsin_y,
                           const FrameDimensions& frame_dim, ThreadPool* pool) {
  const Rect rect(0, 0, frame_dim.xsize, frame_dim.ysize);  // not padded
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();

  const WeightsSymmetric3& weights = WeightsSymmetric3GaussianDC();
  ImageF smoothed(xsize, ysize);
  Symmetric3(opsin_y, rect, weights, pool, &smoothed);

  RunOnPool(
      pool, 0, static_cast<int>(ysize), ThreadPool::SkipInit(),
      [xsize, &opsin_y, &smoothed](const int task, int /*thread*/) {
        const size_t y = static_cast<size_t>(task);
        const float* JXL_RESTRICT row_opsin = opsin_y.ConstRow(y);
        float* JXL_RESTRICT row_smooth = smoothed.Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row_smooth[x] = row_opsin[x] - row_smooth[x];
        }
      },
      "AQ subtract");
  return smoothed;
}

}  // namespace

float InitialQuantDC(float butteraugli_target) {
  const float butteraugli_target_dc =
      std::min<float>(butteraugli_target,
                      2.5 * std::pow(0.4 * butteraugli_target, kDcQuantPow));
  // We want the maximum DC value to be at most 2**15 * kInvDCQuant / quant_dc.
  // The maximum DC value might not be in the kXybRange because of inverse
  // gaborish, so we add some slack to the maximum theoretical quant obtained
  // this way (64).
  return std::min(kDcQuant / butteraugli_target_dc, 50.f);
}

ImageF InitialQuantField(const float butteraugli_target, const Image3F& opsin,
                         const FrameDimensions& frame_dim, ThreadPool* pool,
                         float rescale) {
  PROFILER_FUNC;
  const float quant_ac = kAcQuant / butteraugli_target;
  ImageF intensity_ac_x = IntensityAcEstimate(opsin.Plane(0), frame_dim, pool);
  ImageF intensity_ac_y = IntensityAcEstimate(opsin.Plane(1), frame_dim, pool);
  return AdaptiveQuantizationMap(opsin, intensity_ac_x, intensity_ac_y,
                                 frame_dim, quant_ac * rescale, pool);
}

void FindBestQuantizer(const ImageBundle* linear, const Image3F& opsin,
                       PassesEncoderState* enc_state, ThreadPool* pool,
                       AuxOut* aux_out, double rescale) {
  const CompressParams& cparams = enc_state->cparams;
  Quantizer& quantizer = enc_state->shared.quantizer;
  ImageI& raw_quant_field = enc_state->shared.raw_quant_field;
  if (cparams.max_error_mode) {
    PROFILER_ZONE("enc find best maxerr");
    FindBestQuantizationMaxError(opsin, enc_state, pool, aux_out);
  } else if (cparams.speed_tier == SpeedTier::kFalcon) {
    const float quant_dc = InitialQuantDC(cparams.butteraugli_distance);
    // TODO(veluca): tune constant.
    const float quant_ac = kAcQuant / cparams.butteraugli_distance;
    quantizer.SetQuant(quant_dc, quant_ac, &raw_quant_field);
  } else if (cparams.uniform_quant > 0.0) {
    quantizer.SetQuant(cparams.uniform_quant * rescale,
                       cparams.uniform_quant * rescale, &raw_quant_field);
  } else if (cparams.speed_tier > SpeedTier::kKitten) {
    PROFILER_ZONE("enc fast quant");
    const float quant_dc = InitialQuantDC(cparams.butteraugli_distance);
    AdjustQuantField(enc_state->shared.ac_strategy,
                     &enc_state->initial_quant_field);
    quantizer.SetQuantField(quant_dc, enc_state->initial_quant_field,
                            &raw_quant_field);
  } else {
    // Normal encoding to a butteraugli score.
    PROFILER_ZONE("enc find best2");
    if (cparams.speed_tier == SpeedTier::kTortoise) {
      FindBestQuantizationHQ(*linear, opsin, enc_state, pool, aux_out);
    } else {
      FindBestQuantization(*linear, opsin, enc_state, pool, aux_out);
    }
  }
}

Image3F RoundtripImage(const Image3F& opsin, PassesEncoderState* enc_state,
                       ThreadPool* pool, bool save_decompressed,
                       bool apply_color_transform) {
  PROFILER_ZONE("enc roundtrip");
  PassesDecoderState dec_state;
  dec_state.shared = &enc_state->shared;
  JXL_ASSERT(opsin.ysize() % kBlockDim == 0);

  const size_t xsize_groups = DivCeil(opsin.xsize(), kGroupDim);
  const size_t ysize_groups = DivCeil(opsin.ysize(), kGroupDim);
  const size_t num_groups = xsize_groups * ysize_groups;

  // Dummy metadata with grayscale = off.
  ImageMetadata metadata;
  metadata.color_encoding = ColorEncoding::SRGB();

  InitializePassesEncoder(opsin, pool, enc_state, nullptr);
  dec_state.Init(pool);

  Image3F idct(opsin.xsize(), opsin.ysize());
  ImageBundle decoded(&metadata);

  const auto allocate_storage = [&](size_t num_threads) {
    dec_state.EnsureStorage(num_threads);
    return true;
  };
  const auto process_group = [&](const int group_index, const int thread) {
    ComputeCoefficients(group_index, enc_state, nullptr);
    JXL_CHECK(DecodeGroupForRoundtrip(
        enc_state->coeffs, group_index, &dec_state, thread, &idct, &decoded,
        nullptr, save_decompressed, apply_color_transform));
  };
  RunOnPool(pool, 0, num_groups, allocate_storage, process_group, "AQ loop");

  // Fine to do a JXL_ASSERT instead of error handling, since this only happens
  // on the encoder side where we can't be fed with invalid data.
  JXL_CHECK(FinalizeFrameDecoding(&idct, &dec_state, pool, nullptr,
                                  save_decompressed, apply_color_transform));
  return idct;
}

}  // namespace jxl
