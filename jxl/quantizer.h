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

#ifndef JXL_QUANTIZER_H_
#define JXL_QUANTIZER_H_

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <hwy/static_targets.h>
#include <utility>
#include <vector>

#include "jxl/ac_strategy.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/profiler.h"
#include "jxl/base/robust_statistics.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/dct_util.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/image.h"
#include "jxl/linalg.h"
#include "jxl/quant_weights.h"

// Quantizes DC and AC coefficients, with separate quantization tables according
// to the quant_kind (which is currently computed from the AC strategy and the
// block index inside that strategy).

namespace jxl {

static constexpr int kGlobalScaleDenom = 1 << 16;
static constexpr int kGlobalScaleNumerator = 4096;

// zero-biases for quantizing channels X, Y, B
static constexpr float kZeroBiasDefault[3] = {0.5f, 0.5f, 0.5f};

// Returns adjusted version of a quantized integer, such that its value is
// closer to the expected value of the original.
// The residuals of AC coefficients that we quantize are not uniformly
// distributed. Numerical experiments show that they have a distribution with
// the "shape" of 1/(1+x^2) [up to some coefficients]. This means that the
// expected value of a coefficient that gets quantized to x will not be x
// itself, but (at least with reasonable approximation):
// - 0 if x is 0
// - x * biases[c] if x is 1 or -1
// - x - biases[3]/x otherwise
// This follows from computing the distribution of the quantization bias, which
// can be approximated fairly well by <constant>/x when |x| is at least two.
static constexpr float kBiasNumerator = 0.145f;

static constexpr float kDefaultQuantBias[4] = {
    1.0f - 0.05465007330715401f,
    1.0f - 0.07005449891748593f,
    1.0f - 0.049935103337343655f,
    0.145f,
};

template <class DF>
HWY_ATTR JXL_INLINE hwy::VT<DF> AdjustQuantBias(
    DF df, const size_t c, const hwy::VT<DF> quant,
    const float* JXL_RESTRICT biases) {
  const hwy::Desc<int32_t, DF::LanesOr0()> di;

  // Compare |quant|, keep sign bit for negating result.
  const auto kSign = BitCast(df, Set(di, INT32_MIN));
  const auto sign = quant & kSign;  // TODO(janwas): = abs ^ orig
  const auto abs_quant = AndNot(kSign, quant);

  // If |x| is 1, kZeroBias creates a different bias for each channel.
  // We're implementing the following:
  // if (quant == 0) return 0;
  // if (quant == 1) return biases[c];
  // if (quant == -1) return -biases[c];
  // return quant - biases[3] / quant;

  // Integer comparison is not helpful because Clang incurs bypass penalties
  // from unnecessarily mixing integer and float.
  const auto is_01 = abs_quant < Set(df, 1.125f);
  const auto not_0 = abs_quant > Zero(df);

  // Bitwise logic is faster than quant * biases[3].
  const auto one_bias = IfThenElseZero(not_0, Set(df, biases[c]) ^ sign);

  // About 2E-5 worse than ReciprocalNR or division.
  const auto bias =
      NegMulAdd(Set(df, biases[3]), ApproximateReciprocal(quant), quant);

  return IfThenElse(is_01, one_bias, bias);
}

class Quantizer {
 public:
  explicit Quantizer(const DequantMatrices* dequant);
  Quantizer(const DequantMatrices* dequant, int quant_dc, int global_scale);

  static JXL_INLINE int ClampVal(float val) {
    static const int kQuantMax = 256;
    return std::min(static_cast<int>(std::max(1.0f, val)), kQuantMax);
  }

  // Recomputes other derived fields after global_scale_ has changed.
  void RecomputeFromGlobalScale() {
    global_scale_float_ = global_scale_ * (1.0 / kGlobalScaleDenom);
    inv_global_scale_ = 1.0 * kGlobalScaleDenom / global_scale_;
    inv_quant_dc_ = inv_global_scale_ / quant_dc_;
    for (size_t c = 0; c < 3; c++) {
      mul_dc_[c] = GetDcStep(c);
      inv_mul_dc_[c] = GetInvDcStep(c);
    }
  }

  // Returns scaling factor such that Scale() * (RawDC() or RawQuantField())
  // pixels yields the same float values returned by GetQuantField.
  JXL_INLINE float Scale() const { return global_scale_float_; }

  // Reciprocal of Scale().
  JXL_INLINE float InvGlobalScale() const { return inv_global_scale_; }

  void SetQuantField(const float quant_dc, const ImageF& qf,
                     ImageI* JXL_RESTRICT raw_quant_field);

  void SetQuant(float quant_dc, float quant_ac,
                ImageI* JXL_RESTRICT raw_quant_field);

  // Returns the DC quantization base value, which is currently global (not
  // adaptive). The actual scale factor used to dequantize pixels in channel c
  // is: inv_quant_dc() * dequant_->DCQuant(c).
  float inv_quant_dc() const { return inv_quant_dc_; }

  // Dequantize by multiplying with this times dequant_matrix.
  float inv_quant_ac(int32_t quant) const { return inv_global_scale_ / quant; }

  // NOTE: caller takes care of extracting quant from rect of RawQuantField.
  HWY_ATTR void QuantizeBlockAC(const bool error_diffusion, size_t c,
                                int32_t quant, float qm_multiplier,
                                size_t quant_kind, size_t xsize, size_t ysize,
                                const float* JXL_RESTRICT block_in,
                                ac_qcoeff_t* JXL_RESTRICT block_out) const;

  // NOTE: caller takes care of extracting quant from rect of RawQuantField.
  HWY_ATTR void QuantizeRoundtripYBlockAC(const bool error_diffusion,
                                          int32_t quant, size_t quant_kind,
                                          size_t xsize, size_t ysize,
                                          const float* JXL_RESTRICT biases,
                                          const float* JXL_RESTRICT in,
                                          ac_qcoeff_t* JXL_RESTRICT quantized,
                                          float* JXL_RESTRICT out) const;

  Status Encode(BitWriter* writer, size_t layer, AuxOut* aux_out) const;

  Status Decode(BitReader* reader);

  void DumpQuantizationMap(const ImageI& raw_quant_field) const;

  JXL_INLINE const float* DequantMatrix(size_t quant_kind, size_t c) const {
    return dequant_->Matrix(quant_kind, c);
  }

  JXL_INLINE size_t DequantMatrixOffset(size_t quant_kind, size_t c) const {
    return dequant_->MatrixOffset(quant_kind, c);
  }

  // Calculates DC quantization step.
  JXL_INLINE float GetDcStep(size_t c) const {
    return inv_quant_dc_ * dequant_->DCQuant(c);
  }
  JXL_INLINE float GetInvDcStep(size_t c) const {
    return dequant_->InvDCQuant(c) * (global_scale_float_ * quant_dc_);
  }

  JXL_INLINE const float* MulDC() const { return mul_dc_; }
  JXL_INLINE const float* InvMulDC() const { return inv_mul_dc_; }

  JXL_INLINE void ClearDCMul() {
    std::fill(mul_dc_, mul_dc_ + 4, 1);
    std::fill(inv_mul_dc_, inv_mul_dc_ + 4, 1);
  }

 private:
  void ComputeGlobalScaleAndQuant(float quant_dc, float quant_median,
                                  float quant_absd);

  // These are serialized:
  int global_scale_;
  int quant_dc_;

  // These are derived from global_scale_:
  float inv_global_scale_;
  float global_scale_float_;  // reciprocal of inv_global_scale_
  float inv_quant_dc_;

  float zero_bias_[3];
  HWY_ALIGN float mul_dc_[4];
  HWY_ALIGN float inv_mul_dc_[4];
  const DequantMatrices* dequant_;
};

void TestQuantizerParams();

}  // namespace jxl

#endif  // JXL_QUANTIZER_H_
