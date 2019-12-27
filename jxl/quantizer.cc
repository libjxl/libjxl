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

#include "jxl/quantizer.h"

#include <stdio.h>
#include <string.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/field_encodings.h"
#include "jxl/fields.h"
#include "jxl/image.h"
#include "jxl/image_ops.h"
#include "jxl/quant_weights.h"

namespace jxl {

static const int kDefaultQuant = 64;

Quantizer::Quantizer(const DequantMatrices* dequant)
    : Quantizer(dequant, kDefaultQuant, kGlobalScaleDenom / kDefaultQuant) {}

Quantizer::Quantizer(const DequantMatrices* dequant, int quant_dc,
                     int global_scale)
    : global_scale_(global_scale), quant_dc_(quant_dc), dequant_(dequant) {
  JXL_ASSERT(dequant_ != nullptr);
  RecomputeFromGlobalScale();
  inv_quant_dc_ = inv_global_scale_ / quant_dc_;

  memcpy(zero_bias_, kZeroBiasDefault, sizeof(kZeroBiasDefault));
}

void Quantizer::ComputeGlobalScaleAndQuant(float quant_dc, float quant_median,
                                           float quant_median_absd) {
  // Target value for the median value in the quant field.
  const float kQuantFieldTarget = 3.80987740592518214386f;
  // We reduce the median of the quant field by the median absolute deviation:
  // higher resolution on highly varying quant fields.
  int new_global_scale =
      static_cast<int>(kGlobalScaleDenom * (quant_median - quant_median_absd) /
                       kQuantFieldTarget);
  // Ensure that quant_dc_ will always be at least
  // kGlobalScaleDenom/kGlobalScaleNumerator.
  const int scaled_quant_dc =
      static_cast<int>(quant_dc * kGlobalScaleNumerator);
  if (new_global_scale > scaled_quant_dc) {
    new_global_scale = scaled_quant_dc;
  }
  // Ensure that new_global_scale is positive and no more than 1<<15.
  if (new_global_scale <= 0) new_global_scale = 1;
  if (new_global_scale > (1 << 15)) new_global_scale = 1 << 15;
  global_scale_ = new_global_scale;
  // Code below uses inv_global_scale_.
  RecomputeFromGlobalScale();

  float fval = quant_dc * inv_global_scale_ + 0.5f;
  fval = std::min<float>(1 << 16, fval);
  const int new_quant_dc = static_cast<int>(fval);
  quant_dc_ = new_quant_dc;

  // quant_dc_ was updated, recompute values.
  RecomputeFromGlobalScale();
}

void Quantizer::SetQuantField(const float quant_dc, const ImageF& qf,
                              ImageI* JXL_RESTRICT raw_quant_field) {
  JXL_CHECK(SameSize(*raw_quant_field, qf));
  std::vector<float> data(qf.xsize() * qf.ysize());
  for (size_t y = 0; y < qf.ysize(); ++y) {
    const float* JXL_RESTRICT row_qf = qf.Row(y);
    for (size_t x = 0; x < qf.xsize(); ++x) {
      float quant = row_qf[x];
      data[qf.xsize() * y + x] = quant;
    }
  }
  const float quant_median = Median(&data);
  const float quant_median_absd = MedianAbsoluteDeviation(data, quant_median);
  ComputeGlobalScaleAndQuant(quant_dc, quant_median, quant_median_absd);
  for (size_t y = 0; y < qf.ysize(); ++y) {
    const float* JXL_RESTRICT row_qf = qf.Row(y);
    int32_t* JXL_RESTRICT row_qi = raw_quant_field->Row(y);
    for (size_t x = 0; x < qf.xsize(); ++x) {
      int val = ClampVal(row_qf[x] * inv_global_scale_ + 0.5f);
      row_qi[x] = val;
    }
  }
}

void Quantizer::SetQuant(float quant_dc, float quant_ac,
                         ImageI* JXL_RESTRICT raw_quant_field) {
  ComputeGlobalScaleAndQuant(quant_dc, quant_ac, 0);
  int val = ClampVal(quant_ac * inv_global_scale_ + 0.5f);
  FillImage(val, raw_quant_field);
}

HWY_ATTR void Quantizer::QuantizeBlockAC(
    const bool error_diffusion, size_t c, int32_t quant, float qm_multiplier,
    size_t quant_kind, size_t xsize, size_t ysize,
    const float* JXL_RESTRICT block_in,
    ac_qcoeff_t* JXL_RESTRICT block_out) const {
  PROFILER_FUNC;
  const float* JXL_RESTRICT qm = dequant_->InvMatrix(quant_kind, c);
  const float qac = Scale() * quant;
  // Not SIMD-fied for now.
  float thres[4] = {0.5f, 0.6f, 0.6f, 0.65f};
  if (c != 1) {
    for (int i = 1; i < 4; ++i) {
      thres[i] = 0.75f;
    }
  }

  CoefficientLayout(&ysize, &xsize);

  if (!error_diffusion) {
    HWY_CAPPED(float, kBlockDim) df;
    HWY_CAPPED(uint32_t, kBlockDim) du;
    const auto quant = Set(df, qac * qm_multiplier);
    const auto abs_mask = BitCast(df, Set(du, 0x7FFFFFFFu));

    for (size_t y = 0; y < ysize * kBlockDim; y++) {
      size_t yfix = static_cast<size_t>(y >= ysize * kBlockDim / 2) * 2;
      const size_t off = y * kBlockDim * xsize;
      for (size_t x = 0; x < xsize * kBlockDim; x += df.N) {
        auto thr = Zero(df);
        if (xsize == 1) {
          HWY_ALIGN uint32_t kMask[kBlockDim] = {0,   0,   0,   0,
                                                 ~0u, ~0u, ~0u, ~0u};
          const auto mask = MaskFromVec(BitCast(df, Load(du, kMask + x)));
          thr =
              IfThenElse(mask, Set(df, thres[yfix + 1]), Set(df, thres[yfix]));
        } else {
          // Same for all lanes in the vector.
          thr = Set(
              df,
              thres[yfix + static_cast<size_t>(x >= xsize * kBlockDim / 2)]);
        }

        const auto q = Load(df, qm + off + x) * quant;
        const auto in = Load(df, block_in + off + x);
        const auto val = q * in;
        const auto nzero_mask = (abs_mask & val) >= thr;
        const auto v = IfThenElseZero(nzero_mask, Round(val));
        Store(v, df, block_out + off + x);
      }
    }
    return;
  }

retry:
  int hfNonZeros[4] = {};
  float hfError[4] = {};
  float hfMaxError[4] = {};
  size_t hfMaxErrorIx[4] = {};
  for (size_t y = 0; y < ysize * kBlockDim; y++) {
    for (size_t x = 0; x < xsize * kBlockDim; x++) {
      const size_t pos = y * kBlockDim * xsize + x;
      if (x < xsize && y < ysize) {
        // Ensure block is initialized
        block_out[pos] = 0;
        continue;
      }
      const size_t hfix = (static_cast<size_t>(y >= ysize * kBlockDim / 2) * 2 +
                           static_cast<size_t>(x >= xsize * kBlockDim / 2));
      const float val = block_in[pos] * (qm[pos] * qac * qm_multiplier);
      float v = (std::abs(val) < thres[hfix]) ? 0 : std::rint(val);
      const float error = std::abs(val) - std::abs(v);
      hfError[hfix] += error;
      if (hfMaxError[hfix] < error) {
        hfMaxError[hfix] = error;
        hfMaxErrorIx[hfix] = pos;
      }
      if (v != 0.0f) {
        hfNonZeros[hfix] += std::abs(v);
      }
      block_out[pos] = static_cast<ac_qcoeff_t>(std::rint(v));
    }
  }
  if (c != 1) return;
  // TODO(veluca): include AFV?
  const size_t kPartialBlockKinds =
      (1 << AcStrategy::Type::IDENTITY) | (1 << AcStrategy::Type::DCT2X2) |
      (1 << AcStrategy::Type::DCT4X4) | (1 << AcStrategy::Type::DCT4X8) |
      (1 << AcStrategy::Type::DCT8X4);
  if ((1 << quant_kind) & kPartialBlockKinds) return;
  float hfErrorLimit = 0.1f * (xsize * ysize) * kDCTBlockSize * 0.25f;
  bool goretry = false;
  for (int i = 1; i < 4; ++i) {
    if (hfError[i] >= hfErrorLimit &&
        hfNonZeros[i] <= (xsize + ysize) * 0.25f) {
      if (thres[i] >= 0.4f) {
        thres[i] -= 0.01f;
        goretry = true;
      }
    }
  }
  if (goretry) goto retry;
  for (int i = 1; i < 4; ++i) {
    if (hfError[i] >= hfErrorLimit && hfNonZeros[i] == 0) {
      const size_t pos = hfMaxErrorIx[i];
      if (hfMaxError[i] >= 0.4f) {
        block_out[pos] = block_in[pos] > 0.0f ? 1.0f : -1.0f;
      }
    }
  }
}

HWY_ATTR void Quantizer::QuantizeRoundtripYBlockAC(
    const bool error_diffusion, int32_t quant, size_t quant_kind, size_t xsize,
    size_t ysize, const float* JXL_RESTRICT biases,
    const float* JXL_RESTRICT in, ac_qcoeff_t* JXL_RESTRICT quantized,
    float* JXL_RESTRICT out) const {
  QuantizeBlockAC(error_diffusion, 1, quant, 1.0f, quant_kind, xsize, ysize, in,
                  quantized);

  PROFILER_ZONE("enc quant adjust bias");
  const float* JXL_RESTRICT dequant_matrix = DequantMatrix(quant_kind, 1);

  HWY_CAPPED(float, kDCTBlockSize) df;
  const auto inv_qac = Set(df, inv_quant_ac(quant));
  for (size_t k = 0; k < kDCTBlockSize * xsize * ysize; k += df.N) {
    const auto quant = Load(df, quantized + k);
    const auto adj_quant = AdjustQuantBias(df, 1, quant, biases);
    const auto dequantm = Load(df, dequant_matrix + k);
    Store(adj_quant * dequantm * inv_qac, df, out + k);
  }
}

struct QuantizerParams {
  QuantizerParams() { Bundle::Init(this); }
  static const char* Name() { return "QuantizerParams"; }

  template <class Visitor>
  Status VisitFields(Visitor* JXL_RESTRICT visitor) {
    visitor->U32(Bits(11), BitsOffset(11, 2048), BitsOffset(12, 4096),
                 BitsOffset(15, 8192), 0, &global_scale_minus_1);
    visitor->U32(Val(15), Bits(5), Bits(8), Bits(16), 0, &quant_dc_minus_1);
    return true;
  }

  uint32_t global_scale_minus_1;
  uint32_t quant_dc_minus_1;
};

void TestQuantizerParams() {
  for (uint32_t i = 1; i < 10000; ++i) {
    QuantizerParams p;
    p.global_scale_minus_1 = i - 1;
    size_t extension_bits = 0, total_bits = 0;
    JXL_CHECK(Bundle::CanEncode(p, &extension_bits, &total_bits));
    JXL_CHECK(extension_bits == 0);
    JXL_CHECK(total_bits >= 4);
  }
}

Status Quantizer::Encode(BitWriter* writer, size_t layer,
                         AuxOut* aux_out) const {
  QuantizerParams params;
  params.global_scale_minus_1 = global_scale_ - 1;
  params.quant_dc_minus_1 = quant_dc_ - 1;
  return Bundle::Write(params, writer, layer, aux_out);
}

Status Quantizer::Decode(BitReader* reader) {
  QuantizerParams params;
  JXL_RETURN_IF_ERROR(Bundle::Read(reader, &params));
  global_scale_ = static_cast<int>(params.global_scale_minus_1 + 1);
  quant_dc_ = static_cast<int>(params.quant_dc_minus_1 + 1);
  RecomputeFromGlobalScale();
  return true;
}

void Quantizer::DumpQuantizationMap(const ImageI& raw_quant_field) const {
  printf("Global scale: %d (%.7f)\nDC quant: %d\n", global_scale_,
         global_scale_ * 1.0 / kGlobalScaleDenom, quant_dc_);
  printf("AC quantization Map:\n");
  for (size_t y = 0; y < raw_quant_field.ysize(); ++y) {
    for (size_t x = 0; x < raw_quant_field.xsize(); ++x) {
      printf(" %3d", raw_quant_field.Row(y)[x]);
    }
    printf("\n");
  }
}

static constexpr JXL_INLINE int QuantizeValue(float value, float inv_step) {
  return static_cast<int>(value * inv_step + (value >= 0 ? .5f : -.5f));
}

}  // namespace jxl
