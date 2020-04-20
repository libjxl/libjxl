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

// Fast SIMD evaluation of rational polynomials for approximating functions.

#if defined(JXL_RATIONAL_POLYNOMIAL_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef JXL_RATIONAL_POLYNOMIAL_INL_H_
#undef JXL_RATIONAL_POLYNOMIAL_INL_H_
#else
#define JXL_RATIONAL_POLYNOMIAL_INL_H_
#endif

#include <hwy/highway.h>
#include <stddef.h>

namespace jxl {

#include <hwy/begin_target-inl.h>

// Primary template: default to actual division.
template <typename T, class V>
struct FastDivision {
  HWY_ATTR HWY_INLINE V operator()(const V n, const V d) const { return n / d; }
};
// Partial specialization for float vectors.
template <class V>
struct FastDivision<float, V> {
  // One Newton-Raphson iteration.
  static HWY_ATTR HWY_INLINE V ReciprocalNR(const V x) {
    const auto rcp = ApproximateReciprocal(x);
    const auto sum = rcp + rcp;
    const auto x_rcp = x * rcp;
    return NegMulAdd(x_rcp, rcp, sum);
  }

  HWY_ATTR V operator()(const V n, const V d) const {
#if 1  // Faster on SKX
    return n / d;
#else
    return n * ReciprocalNR(d);
#endif
  }
};

// Approximates smooth functions via rational polynomials (i.e. dividing two
// polynomials). Evaluates polynomials via Horner's scheme, which is faster than
// Clenshaw recurrence for Chebyshev polynomials. LoadDup128 allows us to
// specify constants (replicated 4x) independently of the lane count.
template <size_t NP, size_t NQ, class V, typename T>
HWY_FUNC V EvalRationalPolynomial(const V x, const T (&p)[NP],
                                  const T (&q)[NQ]) {
  const HWY_FULL(T) d;
  constexpr size_t kDegP = NP / 4 - 1;
  constexpr size_t kDegQ = NQ / 4 - 1;
  auto yp = LoadDup128(d, &p[kDegP * 4]);
  auto yq = LoadDup128(d, &q[kDegQ * 4]);
  HWY_FENCE;
  if (kDegP >= 1) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 1) * 4]));
  if (kDegQ >= 1) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 1) * 4]));
  HWY_FENCE;
  if (kDegP >= 2) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 2) * 4]));
  if (kDegQ >= 2) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 2) * 4]));
  HWY_FENCE;
  if (kDegP >= 3) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 3) * 4]));
  if (kDegQ >= 3) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 3) * 4]));
  HWY_FENCE;
  if (kDegP >= 4) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 4) * 4]));
  if (kDegQ >= 4) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 4) * 4]));
  HWY_FENCE;
  if (kDegP >= 5) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 5) * 4]));
  if (kDegQ >= 5) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 5) * 4]));
  HWY_FENCE;
  if (kDegP >= 6) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 6) * 4]));
  if (kDegQ >= 6) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 6) * 4]));
  HWY_FENCE;
  if (kDegP >= 7) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 7) * 4]));
  if (kDegQ >= 7) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 7) * 4]));

  return FastDivision<T, V>()(yp, yq);
}

#include <hwy/end_target-inl.h>

}  // namespace jxl
#endif  // include guard
