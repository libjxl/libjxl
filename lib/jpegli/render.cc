// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/render.h"

#include <string.h>

#include <cmath>

#include "hwy/aligned_allocator.h"
#include "lib/jpegli/color_transform.h"
#include "lib/jpegli/decode_internal.h"
#include "lib/jpegli/idct.h"
#include "lib/jpegli/upsample.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

#ifdef MEMORY_SANITIZER
#define JXL_MEMORY_SANITIZER 1
#elif defined(__has_feature)
#if __has_feature(memory_sanitizer)
#define JXL_MEMORY_SANITIZER 1
#else
#define JXL_MEMORY_SANITIZER 0
#endif
#else
#define JXL_MEMORY_SANITIZER 0
#endif

#if JXL_MEMORY_SANITIZER
#include "sanitizer/msan_interface.h"
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jpegli/render.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace jpegli {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Abs;
using hwy::HWY_NAMESPACE::Add;
using hwy::HWY_NAMESPACE::Clamp;
using hwy::HWY_NAMESPACE::Gt;
using hwy::HWY_NAMESPACE::IfThenElseZero;
using hwy::HWY_NAMESPACE::Mul;
using hwy::HWY_NAMESPACE::NearestInt;
using hwy::HWY_NAMESPACE::Rebind;
using hwy::HWY_NAMESPACE::Vec;

using D = HWY_FULL(float);
using DI = HWY_FULL(int32_t);
constexpr D d;
constexpr DI di;

void GatherBlockStats(const int16_t* JXL_RESTRICT coeffs,
                      const size_t coeffs_size, int32_t* JXL_RESTRICT nonzeros,
                      int32_t* JXL_RESTRICT sumabs) {
  for (size_t i = 0; i < coeffs_size; i += Lanes(d)) {
    size_t k = i % DCTSIZE2;
    const Rebind<int16_t, DI> di16;
    const Vec<DI> coeff = PromoteTo(di, Load(di16, coeffs + i));
    const auto abs_coeff = Abs(coeff);
    const auto not_0 = Gt(abs_coeff, Zero(di));
    const auto nzero = IfThenElseZero(not_0, Set(di, 1));
    Store(Add(nzero, Load(di, nonzeros + k)), di, nonzeros + k);
    Store(Add(abs_coeff, Load(di, sumabs + k)), di, sumabs + k);
  }
}

void DecenterRow(float* row, size_t xsize) {
  const HWY_FULL(float) df;
  const auto c128 = Set(df, 128.0f / 255);
  for (size_t x = 0; x < xsize; x += Lanes(df)) {
    Store(Add(Load(df, row + x), c128), df, row + x);
  }
}

template <typename T>
void StoreUnsignedRow(float* JXL_RESTRICT input[3], size_t x0, size_t len,
                      size_t num_channels, float multiplier, T* output) {
  const HWY_FULL(float) d;
  auto zero = Zero(d);
  auto one = Set(d, 1.0f);
  auto mul = Set(d, multiplier);
  const Rebind<T, decltype(d)> du;
#if JXL_MEMORY_SANITIZER
  const size_t padding = hwy::RoundUpTo(len, Lanes(d)) - len;
  for (size_t c = 0; c < num_channels; ++c) {
    __msan_unpoison(input[c] + x0 + len, sizeof(input[c][0]) * padding);
  }
#endif
  if (num_channels == 1) {
    for (size_t i = 0; i < len; i += Lanes(d)) {
      auto v0 = Mul(Clamp(zero, Load(d, &input[0][x0 + i]), one), mul);
      Store(DemoteTo(du, NearestInt(v0)), du, &output[i]);
    }
  } else if (num_channels == 3) {
    for (size_t i = 0; i < len; i += Lanes(d)) {
      auto v0 = Mul(Clamp(zero, Load(d, &input[0][x0 + i]), one), mul);
      auto v1 = Mul(Clamp(zero, Load(d, &input[1][x0 + i]), one), mul);
      auto v2 = Mul(Clamp(zero, Load(d, &input[2][x0 + i]), one), mul);
      StoreInterleaved3(DemoteTo(du, NearestInt(v0)),
                        DemoteTo(du, NearestInt(v1)),
                        DemoteTo(du, NearestInt(v2)), du, &output[3 * i]);
    }
  }
#if JXL_MEMORY_SANITIZER
  __msan_poison(output + num_channels * len,
                sizeof(output[0]) * num_channels * padding);
#endif
}

void WriteToOutput(float* JXL_RESTRICT rows[3], size_t xoffset, size_t x0,
                   size_t len, size_t num_channels, size_t bit_depth,
                   uint8_t* JXL_RESTRICT scratch_space,
                   uint8_t* JXL_RESTRICT output) {
  const float mul = (1u << bit_depth) - 1;
  if (bit_depth <= 8) {
    size_t offset = x0 * num_channels;
    StoreUnsignedRow(rows, xoffset + x0, len, num_channels, mul, scratch_space);
    memcpy(output + offset, scratch_space, len * num_channels);
  } else {
    size_t offset = x0 * num_channels * 2;
    uint16_t* tmp = reinterpret_cast<uint16_t*>(scratch_space);
    StoreUnsignedRow(rows, xoffset + x0, len, num_channels, mul, tmp);
    // TODO(szabadka) Handle endianness.
    memcpy(output + offset, tmp, len * num_channels * 2);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jpegli
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jpegli {

HWY_EXPORT(GatherBlockStats);
HWY_EXPORT(WriteToOutput);
HWY_EXPORT(DecenterRow);

void GatherBlockStats(const int16_t* JXL_RESTRICT coeffs,
                      const size_t coeffs_size, int32_t* JXL_RESTRICT nonzeros,
                      int32_t* JXL_RESTRICT sumabs) {
  return HWY_DYNAMIC_DISPATCH(GatherBlockStats)(coeffs, coeffs_size, nonzeros,
                                                sumabs);
}

void WriteToOutput(float* JXL_RESTRICT rows[3], size_t xoffset, size_t x0,
                   size_t len, size_t num_channels, size_t bit_depth,
                   uint8_t* JXL_RESTRICT scratch_space,
                   uint8_t* JXL_RESTRICT output) {
  return HWY_DYNAMIC_DISPATCH(WriteToOutput)(
      rows, xoffset, x0, len, num_channels, bit_depth, scratch_space, output);
}

void DecenterRow(float* row, size_t xsize) {
  return HWY_DYNAMIC_DISPATCH(DecenterRow)(row, xsize);
}

// Padding for horizontal chroma upsampling.
constexpr size_t kPaddingLeft = 64;
constexpr size_t kPaddingRight = 64;
constexpr size_t kTempOutputLen = 1024;

// See the following article for the details:
// J. R. Price and M. Rabbani, "Dequantization bias for JPEG decompression"
// Proceedings International Conference on Information Technology: Coding and
// Computing (Cat. No.PR00540), 2000, pp. 30-35, doi: 10.1109/ITCC.2000.844179.
void ComputeOptimalLaplacianBiases(const int num_blocks, const int* nonzeros,
                                   const int* sumabs, float* biases) {
  for (size_t k = 1; k < DCTSIZE2; ++k) {
    // Notation adapted from the article
    size_t N = num_blocks;
    size_t N1 = nonzeros[k];
    size_t N0 = num_blocks - N1;
    size_t S = sumabs[k];
    // Compute gamma from N0, N1, N, S (eq. 11), with A and B being just
    // temporary grouping of terms.
    float A = 4.0 * S + 2.0 * N;
    float B = 4.0 * S - 2.0 * N1;
    float gamma = (-1.0 * N0 + std::sqrt(N0 * N0 * 1.0 + A * B)) / A;
    float gamma2 = gamma * gamma;
    // The bias is computed from gamma with (eq. 5), where the quantization
    // multiplier Q can be factored out and thus the bias can be applied
    // directly on the quantized coefficient.
    biases[k] =
        0.5 * (((1.0 + gamma2) / (1.0 - gamma2)) + 1.0 / std::log(gamma));
  }
}

void PrepareForOutput(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  m->MCU_row_stride_ = m->iMCU_cols_ * cinfo->max_h_samp_factor * DCTSIZE +
                       kPaddingLeft + kPaddingRight;
  m->MCU_plane_size_ = m->MCU_row_stride_ * cinfo->max_v_samp_factor * DCTSIZE;
  m->MCU_row_buf_ = hwy::AllocateAligned<float>(3 * m->MCU_plane_size_);
  m->num_chroma_ = 0;
  m->chroma_plane_size_ = m->MCU_row_stride_ * 3 * DCTSIZE;
  m->chroma_ = hwy::AllocateAligned<float>(3 * m->chroma_plane_size_);
  for (int c = 0; c < cinfo->num_components; ++c) {
    if (cinfo->comp_info[c].v_samp_factor < cinfo->max_v_samp_factor) {
      m->component_order_.emplace_back(c);
      ++m->num_chroma_;
    }
  }
  for (int c = 0; c < cinfo->num_components; ++c) {
    if (cinfo->comp_info[c].v_samp_factor == cinfo->max_v_samp_factor) {
      m->component_order_.emplace_back(c);
    }
  }
  m->idct_scratch_ = hwy::AllocateAligned<float>(DCTSIZE2 * 2);
  m->upsample_scratch_ = hwy::AllocateAligned<float>(m->MCU_row_stride_);
  size_t bytes_per_channel = DivCeil(m->output_bit_depth_, 8);
  size_t bytes_per_sample = cinfo->out_color_components * bytes_per_channel;
  m->output_scratch_ =
      hwy::AllocateAligned<uint8_t>(bytes_per_sample * kTempOutputLen);
  size_t coeffs_per_block = cinfo->num_components * DCTSIZE2;
  m->nonzeros_ = hwy::AllocateAligned<int>(coeffs_per_block);
  m->sumabs_ = hwy::AllocateAligned<int>(coeffs_per_block);
  memset(m->nonzeros_.get(), 0, coeffs_per_block * sizeof(m->nonzeros_[0]));
  memset(m->sumabs_.get(), 0, coeffs_per_block * sizeof(m->sumabs_[0]));
  m->num_processed_blocks_.resize(cinfo->num_components);
  m->biases_ = hwy::AllocateAligned<float>(coeffs_per_block);
  memset(m->biases_.get(), 0, coeffs_per_block * sizeof(m->biases_[0]));
  cinfo->output_iMCU_row = 0;
  m->output_ci_ = 0;
  cinfo->output_scanline = 0;
  m->MCU_buf_ready_rows_ = 0;
  const float kDequantScale = 1.0f / (8 * 255);
  m->dequant_ = hwy::AllocateAligned<float>(coeffs_per_block);
  for (int c = 0; c < cinfo->num_components; c++) {
    const auto& comp = cinfo->comp_info[c];
    const int32_t* quant = m->quant_[comp.quant_tbl_no].values.data();
    for (size_t k = 0; k < DCTSIZE2; ++k) {
      m->dequant_[c * DCTSIZE2 + k] = quant[k] * kDequantScale;
    }
  }
}

void ProcessOutput(j_decompress_ptr cinfo, size_t* num_output_rows,
                   JSAMPARRAY scanlines, size_t max_output_rows) {
  jpeg_decomp_master* m = cinfo->master;
  size_t xsize_blocks = DivCeil(cinfo->image_width, DCTSIZE);
  size_t mcu_y = cinfo->output_iMCU_row;
  for (; m->output_ci_ < cinfo->num_components; ++m->output_ci_) {
    size_t c = m->component_order_[m->output_ci_];
    size_t k0 = c * DCTSIZE2;
    auto& comp = m->components_[c];
    auto& compinfo = cinfo->comp_info[c];
    bool hups = compinfo.h_samp_factor < cinfo->max_h_samp_factor;
    bool vups = compinfo.v_samp_factor < cinfo->max_v_samp_factor;
    size_t nblocks_y = compinfo.v_samp_factor;
    float* output;
    size_t output_ysize;
    if (vups) {
      output = m->chroma_.get() + m->output_ci_ * m->chroma_plane_size_;
      output_ysize = 3 * DCTSIZE;
    } else {
      output = m->MCU_row_buf_.get() + c * m->MCU_plane_size_;
      output_ysize = cinfo->max_v_samp_factor * DCTSIZE;
    }
    size_t mcu_y0 = vups ? (mcu_y * DCTSIZE) % output_ysize : 0;
    if (m->output_ci_ == m->num_chroma_ && mcu_y > 0) {
      // For the previous MCU row we have everything we need at this point,
      // including the chroma components for the current MCU row that was used
      // in upsampling, so we can do the color conversion and the interleaved
      // output.
      if (m->MCU_buf_ready_rows_ == 0) {
        m->MCU_buf_ready_rows_ = cinfo->max_v_samp_factor * DCTSIZE;
        m->MCU_buf_current_row_ = 0;
      }
      while (m->MCU_buf_current_row_ < m->MCU_buf_ready_rows_ &&
             *num_output_rows < max_output_rows &&
             cinfo->output_scanline < cinfo->output_height) {
        // TODO(szabadka) Support 4 components JPEGs.
        size_t offsets[3];
        float* rows[3];
        for (int c = 0; c < cinfo->out_color_components; ++c) {
          offsets[c] = c * m->MCU_plane_size_ +
                       m->MCU_buf_current_row_ * m->MCU_row_stride_ +
                       kPaddingLeft;
        }
        for (int c = 0; c < cinfo->out_color_components; ++c) {
          rows[c] = m->MCU_row_buf_.get() + offsets[c];
        }
        if (cinfo->jpeg_color_space == JCS_YCbCr) {
          YCbCrToRGB(rows[0], rows[1], rows[2], xsize_blocks * DCTSIZE);
        } else {
          for (int c = 0; c < cinfo->out_color_components; ++c) {
            // Libjpeg encoder converts all unsigned input values to signed
            // ones, i.e. for 8 bit input from [0..255] to [-128..127]. For
            // YCbCr jpegs this is undone in the YCbCr -> RGB conversion above
            // by adding 128 to Y channel, but for grayscale and RGB jpegs we
            // need to undo it here channel by channel.
            DecenterRow(rows[c], xsize_blocks * DCTSIZE);
          }
        }
        for (size_t x0 = 0; x0 < cinfo->output_width; x0 += kTempOutputLen) {
          size_t len = std::min(cinfo->output_width - x0, kTempOutputLen);
          if (scanlines) {
            uint8_t* output = scanlines[*num_output_rows];
            WriteToOutput(rows, m->xoffset_, x0, len,
                          cinfo->out_color_components, m->output_bit_depth_,
                          m->output_scratch_.get(), output);
          }
        }
        ++cinfo->output_scanline;
        ++(*num_output_rows);
        ++m->MCU_buf_current_row_;
      }
      if (cinfo->output_scanline == cinfo->output_height ||
          *num_output_rows == max_output_rows) {
        return;
      }
      m->MCU_buf_ready_rows_ = 0;
    }
    if (mcu_y < cinfo->total_iMCU_rows) {
      if (!hups && !vups) {
        size_t num_coeffs = compinfo.width_in_blocks * DCTSIZE2;
        size_t offset = mcu_y * compinfo.width_in_blocks * DCTSIZE2;
        // Update statistics for this MCU row.
        GatherBlockStats(&comp.coeffs[offset], num_coeffs, &m->nonzeros_[k0],
                         &m->sumabs_[k0]);
        m->num_processed_blocks_[c] += compinfo.width_in_blocks;
        if (mcu_y % 4 == 3) {
          // Re-compute optimal biases every few MCU-rows.
          ComputeOptimalLaplacianBiases(m->num_processed_blocks_[c],
                                        &m->nonzeros_[k0], &m->sumabs_[k0],
                                        &m->biases_[k0]);
        }
      }
      for (size_t iy = 0; iy < nblocks_y; ++iy) {
        size_t by = mcu_y * nblocks_y + iy;
        size_t y0 = mcu_y0 + iy * DCTSIZE;
        int16_t* JXL_RESTRICT row_in =
            &comp.coeffs[by * compinfo.width_in_blocks * DCTSIZE2];
        float* JXL_RESTRICT row_out =
            output + y0 * m->MCU_row_stride_ + kPaddingLeft;
        for (size_t bx = 0; bx < compinfo.width_in_blocks; ++bx) {
          InverseTransformBlock(&row_in[bx * DCTSIZE2], &m->dequant_[k0],
                                &m->biases_[k0], m->idct_scratch_.get(),
                                &row_out[bx * DCTSIZE], m->MCU_row_stride_);
        }
        if (hups) {
          for (size_t y = 0; y < DCTSIZE; ++y) {
            float* JXL_RESTRICT row =
                output + (y0 + y) * m->MCU_row_stride_ + kPaddingLeft;
            Upsample2Horizontal(row, m->upsample_scratch_.get(),
                                xsize_blocks * DCTSIZE);
            memcpy(row, m->upsample_scratch_.get(),
                   xsize_blocks * DCTSIZE * sizeof(row[0]));
          }
        }
      }
    }
    if (vups) {
      auto y_idx = [&](size_t mcu_y, ssize_t y) {
        return (output_ysize + mcu_y * DCTSIZE + y) % output_ysize;
      };
      if (mcu_y == 0) {
        // Copy the first row of the current MCU row to the last row of the
        // previous one.
        memcpy(output + y_idx(mcu_y, -1) * m->MCU_row_stride_,
               output + y_idx(mcu_y, 0) * m->MCU_row_stride_,
               m->MCU_row_stride_ * sizeof(output[0]));
      }
      if (mcu_y == cinfo->total_iMCU_rows) {
        // Copy the last row of the current MCU row to the  first row of the
        // next  one.
        memcpy(output + y_idx(mcu_y + 1, 0) * m->MCU_row_stride_,
               output + y_idx(mcu_y, DCTSIZE - 1) * m->MCU_row_stride_,
               m->MCU_row_stride_ * sizeof(output[0]));
      }
      if (mcu_y > 0) {
        for (size_t y = 0; y < DCTSIZE; ++y) {
          size_t y_top = y_idx(mcu_y - 1, y - 1);
          size_t y_cur = y_idx(mcu_y - 1, y);
          size_t y_bot = y_idx(mcu_y - 1, y + 1);
          size_t y_out0 = 2 * y;
          size_t y_out1 = 2 * y + 1;
          Upsample2Vertical(output + y_top * m->MCU_row_stride_ + kPaddingLeft,
                            output + y_cur * m->MCU_row_stride_ + kPaddingLeft,
                            output + y_bot * m->MCU_row_stride_ + kPaddingLeft,
                            m->MCU_row_buf_.get() + c * m->MCU_plane_size_ +
                                y_out0 * m->MCU_row_stride_ + kPaddingLeft,
                            m->MCU_row_buf_.get() + c * m->MCU_plane_size_ +
                                y_out1 * m->MCU_row_stride_ + kPaddingLeft,
                            xsize_blocks * DCTSIZE);
        }
      }
    }
  }
  ++cinfo->output_iMCU_row;
  m->output_ci_ = 0;
  JXL_DASSERT(cinfo->output_iMCU_row <= cinfo->total_iMCU_rows);
}

}  // namespace jpegli
#endif  // HWY_ONCE
