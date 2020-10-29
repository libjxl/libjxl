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

// Transfer functions for color encodings.

#if defined(LIB_JXL_TRANSFER_FUNCTIONS_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_TRANSFER_FUNCTIONS_INL_H_
#undef LIB_JXL_TRANSFER_FUNCTIONS_INL_H_
#else
#define LIB_JXL_TRANSFER_FUNCTIONS_INL_H_
#endif

#include <algorithm>
#include <cmath>
#include <hwy/highway.h>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/rational_polynomial-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// Definitions for BT.2100-2 transfer functions (used inside/outside SIMD):
// "display" is linear light (nits) normalized to [0, 1].
// "encoded" is a nonlinear encoding (e.g. PQ) in [0, 1].
// "scene" is a linear function of photon counts, normalized to [0, 1].

// Despite the stated ranges, we need unbounded transfer functions: see
// http://www.littlecms.com/CIC18_UnboundedCMM.pdf. Inputs can be negative or
// above 1 due to chromatic adaptation. To avoid severe round-trip errors caused
// by clamping, we mirror negative inputs via copysign (f(-x) = -f(x), see
// https://developer.apple.com/documentation/coregraphics/cgcolorspace/1644735-extendedsrgb)
// and extend the function domains above 1.

// Hybrid Log-Gamma.
class TF_HLG {
 public:
  // EOTF. e = encoded.
  JXL_INLINE double DisplayFromEncoded(const double e) const {
    const double lifted = e * (1.0 - kBeta) + kBeta;
    return OOTF(InvOETF(lifted));
  }

  // Inverse EOTF. d = display.
  JXL_INLINE double EncodedFromDisplay(const double d) const {
    const double lifted = OETF(InvOOTF(d));
    const double e = (lifted - kBeta) * (1.0 / (1.0 - kBeta));
    return e;
  }

 private:
  // OETF (defines the HLG approach). s = scene, returns encoded.
  JXL_INLINE double OETF(double s) const {
    if (s == 0.0) return 0.0;
    const double original_sign = s;
    s = std::abs(s);

    if (s <= kDiv12) return std::copysign(std::sqrt(3.0 * s), original_sign);

    const double e = kA * std::log(12 * s - kB) + kC;
    JXL_ASSERT(e > 0.0);
    return std::copysign(e, original_sign);
  }

  // e = encoded, returns scene.
  JXL_INLINE double InvOETF(double e) const {
    if (e == 0.0) return 0.0;
    const double original_sign = e;
    e = std::abs(e);

    if (e <= 0.5) return std::copysign(e * e * (1.0 / 3), original_sign);

    const double s = (std::exp((e - kC) * kRA) + kB) * kDiv12;
    JXL_ASSERT(s >= 0);
    return std::copysign(s, original_sign);
  }

  // s = scene, returns display.
  JXL_INLINE double OOTF(const double s) const {
    // The actual (red channel) OOTF is RD = alpha * YS^(gamma-1) * RS, where
    // YS = 0.2627 * RS + 0.6780 * GS + 0.0593 * BS. Let alpha = 1 so we return
    // "display" (normalized [0, 1]) instead of nits. Our transfer function
    // interface does not allow a dependency on YS. Fortunately, the system
    // gamma at 334 nits is 1.0, so this reduces to RD = RS.
    return s;
  }

  // d = display, returns scene.
  JXL_INLINE double InvOOTF(const double d) const {
    return d;  // see OOTF().
  }

  // Assume 1000:1 contrast @ 200 nits => gamma 0.9
  static constexpr double kBeta = 0.04;  // = sqrt(3 * contrast^(1/gamma))

  static constexpr double kA = 0.17883277;
  static constexpr double kRA = 1.0 / kA;
  static constexpr double kB = 1 - 4 * kA;
  static constexpr double kC = 0.5599107295;
  static constexpr double kDiv12 = 1.0 / 12;
};

// Perceptual Quantization
class TF_PQ {
 public:
  // EOTF (defines the PQ approach). e = encoded.
  JXL_INLINE double DisplayFromEncoded(double e) const {
    if (e == 0.0) return 0.0;
    const double original_sign = e;
    e = std::abs(e);

    const double xp = std::pow(e, 1.0 / kM2);
    const double num = std::max(xp - kC1, 0.0);
    const double den = kC2 - kC3 * xp;
    JXL_ASSERT(den != 0.0);
    const double d = std::pow(num / den, 1.0 / kM1);
    JXL_ASSERT(d >= 0.0);  // Equal for e ~= 1E-9
    return std::copysign(d, original_sign);
  }

  // Inverse EOTF. d = display.
  JXL_INLINE double EncodedFromDisplay(double d) const {
    if (d == 0.0) return 0.0;
    const double original_sign = d;
    d = std::abs(d);

    const double xp = std::pow(d, kM1);
    const double num = kC1 + xp * kC2;
    const double den = 1.0 + xp * kC3;
    const double e = std::pow(num / den, kM2);
    JXL_ASSERT(e > 0.0);
    return std::copysign(e, original_sign);
  }

 private:
  static constexpr double kM1 = 2610.0 / 16384;
  static constexpr double kM2 = (2523.0 / 4096) * 128;
  static constexpr double kC1 = 3424.0 / 4096;
  static constexpr double kC2 = (2413.0 / 4096) * 32;
  static constexpr double kC3 = (2392.0 / 4096) * 32;
};

// sRGB
class TF_SRGB {
 public:
  template <typename V>
  JXL_INLINE V DisplayFromEncoded(V x) const {
    const HWY_FULL(float) d;
    const HWY_FULL(uint32_t) du;
    const V kSign = BitCast(d, Set(du, 0x80000000u));
    const V original_sign = And(x, kSign);
    x = AndNot(kSign, x);  // abs

    // TODO(janwas): range reduction
    // Computed via af_cheb_rational (k=100); replicated 4x.
    HWY_ALIGN constexpr float p[(4 + 1) * 4] = {
        2.200248328e-04f, 2.200248328e-04f, 2.200248328e-04f, 2.200248328e-04f,
        1.043637593e-02f, 1.043637593e-02f, 1.043637593e-02f, 1.043637593e-02f,
        1.624820318e-01f, 1.624820318e-01f, 1.624820318e-01f, 1.624820318e-01f,
        7.961564959e-01f, 7.961564959e-01f, 7.961564959e-01f, 7.961564959e-01f,
        8.210152774e-01f, 8.210152774e-01f, 8.210152774e-01f, 8.210152774e-01f,
    };
    HWY_ALIGN constexpr float q[(4 + 1) * 4] = {
        2.631846970e-01f,  2.631846970e-01f,  2.631846970e-01f,
        2.631846970e-01f,  1.076976492e+00f,  1.076976492e+00f,
        1.076976492e+00f,  1.076976492e+00f,  4.987528350e-01f,
        4.987528350e-01f,  4.987528350e-01f,  4.987528350e-01f,
        -5.512498495e-02f, -5.512498495e-02f, -5.512498495e-02f,
        -5.512498495e-02f, 6.521209011e-03f,  6.521209011e-03f,
        6.521209011e-03f,  6.521209011e-03f,
    };
    const V linear = x * Set(d, kLowDivInv);
    const V poly = EvalRationalPolynomial(d, x, p, q);
    const V magnitude =
        IfThenElse(x > Set(d, kThreshSRGBToLinear), poly, linear);
    return Or(AndNot(kSign, magnitude), original_sign);
  }

  template <class V>
  JXL_INLINE V EncodedFromDisplay(V x) const {
    const HWY_FULL(float) d;
    const HWY_FULL(uint32_t) du;
    const V kSign = BitCast(d, Set(du, 0x80000000u));
    const V original_sign = And(x, kSign);
    x = AndNot(kSign, x);  // abs

    // Computed via af_cheb_rational (k=100); replicated 4x.
    HWY_ALIGN constexpr float p[(4 + 1) * 4] = {
        -5.135152395e-04f, -5.135152395e-04f, -5.135152395e-04f,
        -5.135152395e-04f, 5.287254571e-03f,  5.287254571e-03f,
        5.287254571e-03f,  5.287254571e-03f,  3.903842876e-01f,
        3.903842876e-01f,  3.903842876e-01f,  3.903842876e-01f,
        1.474205315e+00f,  1.474205315e+00f,  1.474205315e+00f,
        1.474205315e+00f,  7.352629620e-01f,  7.352629620e-01f,
        7.352629620e-01f,  7.352629620e-01f,
    };
    HWY_ALIGN constexpr float q[(4 + 1) * 4] = {
        1.004519624e-02f, 1.004519624e-02f, 1.004519624e-02f, 1.004519624e-02f,
        3.036675394e-01f, 3.036675394e-01f, 3.036675394e-01f, 3.036675394e-01f,
        1.340816930e+00f, 1.340816930e+00f, 1.340816930e+00f, 1.340816930e+00f,
        9.258482155e-01f, 9.258482155e-01f, 9.258482155e-01f, 9.258482155e-01f,
        2.424867759e-02f, 2.424867759e-02f, 2.424867759e-02f, 2.424867759e-02f,
    };
    const V linear = x * Set(d, kLowDiv);
    const V poly = EvalRationalPolynomial(d, Sqrt(x), p, q);
    const V magnitude =
        IfThenElse(x > Set(d, kThreshLinearToSRGB), poly, linear);
    return Or(AndNot(kSign, magnitude), original_sign);
  }

 private:
  static constexpr float kThreshSRGBToLinear = 0.04045f;
  static constexpr float kThreshLinearToSRGB = 0.0031308f;
  static constexpr float kLowDiv = 12.92f;
  static constexpr float kLowDivInv = 1.0f / kLowDiv;
};

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_TRANSFER_FUNCTIONS_INL_H_
