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

// Fast SIMD log2

#if defined(JXL_FAST_LOG_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef JXL_FAST_LOG_INL_H_
#undef JXL_FAST_LOG_INL_H_
#else
#define JXL_FAST_LOG_INL_H_
#endif

#include <hwy/highway.h>
#include "jxl/base/status.h"
#include "jxl/rational_polynomial-inl.h"

namespace jxl {

#include <hwy/begin_target-inl.h>

// L1 error ~3.9E-6 (see fast_log_test).
template <class V>
HWY_ATTR HWY_MAYBE_UNUSED V FastLog2f_18bits(V x) {
  // 2,2 rational polynomial approximation of std::log1p(x) / std::log(2).
  HWY_ALIGN const float p[4 * (2 + 1)] = {HWY_REP4(-1.8503833400518310E-06f),
                                          HWY_REP4(1.4287160470083755E+00f),
                                          HWY_REP4(7.4245873327820566E-01f)};
  HWY_ALIGN const float q[4 * (2 + 1)] = {HWY_REP4(9.9032814277590719E-01f),
                                          HWY_REP4(1.0096718572241148E+00f),
                                          HWY_REP4(1.7409343003366853E-01f)};

  HWY_FULL(float) df;
  HWY_FULL(int32_t) di;
  const auto x_bits = BitCast(di, x);
  // Cannot handle negative numbers / NaN.
  JXL_DASSERT(AllTrue(Abs(x_bits) == x_bits));

  // Range reduction to [-1/3, 1/3] - 3 integer, 2 float ops
  const auto exp_bits = x_bits - Set(di, 0x3f2aaaab);  // = 2/3
  // Shifted exponent = log2; also used to clear mantissa.
  const auto exp_shifted = ShiftRight<23>(exp_bits);
  const auto mantissa = BitCast(df, x_bits - ShiftLeft<23>(exp_shifted));
  const auto exp_val = ConvertTo(df, exp_shifted);
  return EvalRationalPolynomial(mantissa - Set(df, 1.0f), p, q) + exp_val;
}

#include <hwy/end_target-inl.h>

}  // namespace jxl

#endif  // include guard
