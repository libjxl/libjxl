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

#ifndef JXL_RATIONAL_POLYNOMIAL_H_
#define JXL_RATIONAL_POLYNOMIAL_H_

// Fast SIMD evaluation of rational polynomials for approximating functions.

#include <hwy/static_targets.h>

#include "jxl/base/compiler_specific.h"

namespace jxl {

// One Newton-Raphson iteration.
template <class V>
static HWY_ATTR HWY_INLINE V ReciprocalNR(const V x) {
  const auto rcp = ApproximateReciprocal(x);
  const auto sum = rcp + rcp;
  const auto x_rcp = x * rcp;
  return NegMulAdd(x_rcp, rcp, sum);
}

// Primary template: default to actual division.
template <typename T, class V>
struct FastDivision {
  HWY_ATTR V operator()(const V n, const V d) const { return n / d; }
};
// Partial specialization for float vectors.
template <class V>
struct FastDivision<float, V> {
  HWY_ATTR V operator()(const V n, const V d) const {
    return n * ReciprocalNR(d);
  }
};

// Approximates smooth functions via rational polynomials (i.e. dividing two
// polynomials). Supports V = SIMD or Scalar<T> inputs.

// Evaluates the polynomial using Horner's method, which is faster than
// Clenshaw recurrence for Chebyshev polynomials.
//
// "kDeg" is the degree of the numerator and denominator polynomials;
// kDegP == kDegQ + 1 = 3 or 4 is usually a good choice.
template <class D, int kDegP, int kDegQ>
class RationalPolynomial {
  using T = typename D::T;
  using V = hwy::VT<D>;
  static_assert(kDegP <= 7, "Unroll more iterations");
  static_assert(kDegQ <= 7, "Unroll more iterations");

 public:
  template <typename U>
  HWY_ATTR void SetCoefficients(const U (&p)[kDegP + 1],
                                const U (&q)[kDegQ + 1]) {
    for (int i = 0; i <= kDegP; ++i) {
      p_[i] = Set(D(), static_cast<T>(p[i]));
    }
    for (int i = 0; i <= kDegQ; ++i) {
      q_[i] = Set(D(), static_cast<T>(q[i]));
    }
  }

  HWY_ATTR void GetCoefficients(T (*p)[kDegP + 1], T (*q)[kDegQ + 1]) const {
    for (int i = 0; i <= kDegP; ++i) {
      p[i] = GetLane(p_[i]);
    }
    for (int i = 0; i <= kDegQ; ++i) {
      q[i] = GetLane(q_[i]);
    }
  }

  template <typename U>
  HWY_ATTR RationalPolynomial(const U (&p)[kDegP + 1],
                              const U (&q)[kDegQ + 1]) {
    SetCoefficients(p, q);
  }

  // Evaluates the polynomial at x.
  HWY_ATTR JXL_INLINE V operator()(const V x) const {
    V yp = p_[kDegP];
    V yq = q_[kDegQ];
    JXL_COMPILER_FENCE;
    if (kDegP >= 1) yp = MulAdd(yp, x, p_[kDegP - 1]);
    if (kDegQ >= 1) yq = MulAdd(yq, x, q_[kDegQ - 1]);
    JXL_COMPILER_FENCE;
    if (kDegP >= 2) yp = MulAdd(yp, x, p_[kDegP - 2]);
    if (kDegQ >= 2) yq = MulAdd(yq, x, q_[kDegQ - 2]);
    JXL_COMPILER_FENCE;
    if (kDegP >= 3) yp = MulAdd(yp, x, p_[kDegP - 3]);
    if (kDegQ >= 3) yq = MulAdd(yq, x, q_[kDegQ - 3]);
    JXL_COMPILER_FENCE;
    if (kDegP >= 4) yp = MulAdd(yp, x, p_[kDegP - 4]);
    if (kDegQ >= 4) yq = MulAdd(yq, x, q_[kDegQ - 4]);
    JXL_COMPILER_FENCE;
    if (kDegP >= 5) yp = MulAdd(yp, x, p_[kDegP - 5]);
    if (kDegQ >= 5) yq = MulAdd(yq, x, q_[kDegQ - 5]);
    JXL_COMPILER_FENCE;
    if (kDegP >= 6) yp = MulAdd(yp, x, p_[kDegP - 6]);
    if (kDegQ >= 6) yq = MulAdd(yq, x, q_[kDegQ - 6]);
    JXL_COMPILER_FENCE;
    if (kDegP >= 7) yp = MulAdd(yp, x, p_[kDegP - 7]);
    if (kDegQ >= 7) yq = MulAdd(yq, x, q_[kDegQ - 7]);

    // Division is faster for a single evaluation but the Triple below are
    // much faster with NR, and we use the same approach to here so that we
    // compute the same max error as reached below.
    return FastDivision<T, V>()(yp, yq);
  }

 private:
  // Horner coefficients in ascending order.
  V p_[kDegP + 1];
  V q_[kDegQ + 1];
};

// Evaluates a rational polynomial via Horner's scheme. Equivalent to
// RationalPolynomial poly(p, q); return poly(x). This can be more efficient
// because the coefficients are loaded directly from memory, whereas Set
// can result in copying them from RIP+x to stack frame. LoadDup128 allows us
// to specify constants (replicated 4x) independently of the lane count.
template <int NP, int NQ, class V, typename T>
HWY_ATTR JXL_INLINE V EvalRationalPolynomial(const V x, const T (&p)[NP],
                                             const T (&q)[NQ]) {
  const HWY_FULL(T) d;
  constexpr int kDegP = NP / 4 - 1;
  constexpr int kDegQ = NQ / 4 - 1;
  auto yp = LoadDup128(d, &p[kDegP * 4]);
  auto yq = LoadDup128(d, &q[kDegQ * 4]);
  JXL_COMPILER_FENCE;
  if (kDegP >= 1) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 1) * 4]));
  if (kDegQ >= 1) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 1) * 4]));
  JXL_COMPILER_FENCE;
  if (kDegP >= 2) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 2) * 4]));
  if (kDegQ >= 2) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 2) * 4]));
  JXL_COMPILER_FENCE;
  if (kDegP >= 3) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 3) * 4]));
  if (kDegQ >= 3) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 3) * 4]));
  JXL_COMPILER_FENCE;
  if (kDegP >= 4) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 4) * 4]));
  if (kDegQ >= 4) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 4) * 4]));
  JXL_COMPILER_FENCE;
  if (kDegP >= 5) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 5) * 4]));
  if (kDegQ >= 5) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 5) * 4]));
  JXL_COMPILER_FENCE;
  if (kDegP >= 6) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 6) * 4]));
  if (kDegQ >= 6) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 6) * 4]));
  JXL_COMPILER_FENCE;
  if (kDegP >= 7) yp = MulAdd(yp, x, LoadDup128(d, &p[(kDegP - 7) * 4]));
  if (kDegQ >= 7) yq = MulAdd(yq, x, LoadDup128(d, &q[(kDegQ - 7) * 4]));

  return FastDivision<T, V>()(yp, yq);
}

// Evaluates three at once for better FMA utilization and fewer loads.
template <int NP, int NQ, class V, typename T>
HWY_ATTR void EvalRationalPolynomialTriple(const V x0, const V x1, const V x2,
                                           const T (&p)[NP], const T (&q)[NQ],
                                           V* JXL_RESTRICT y0,
                                           V* JXL_RESTRICT y1,
                                           V* JXL_RESTRICT y2) {
  // Computing both polynomials in parallel is slightly faster than sequential
  // (better utilization of FMA slots despite higher register pressure).
  const HWY_FULL(T) d;
  constexpr int kDegP = NP / 4 - 1;
  constexpr int kDegQ = NQ / 4 - 1;
  V yp0 = LoadDup128(d, &p[kDegP * 4]);
  V yq0 = LoadDup128(d, &q[kDegQ * 4]);
  V yp1 = yp0;
  V yq1 = yq0;
  V yp2 = yp0;
  V yq2 = yq0;
  V c;
  if (kDegP >= 1) {
    c = LoadDup128(d, &p[(kDegP - 1) * 4]);
    yp0 = MulAdd(yp0, x0, c);
    yp1 = MulAdd(yp1, x1, c);
    yp2 = MulAdd(yp2, x2, c);
  }
  if (kDegQ >= 1) {
    c = LoadDup128(d, &q[(kDegQ - 1) * 4]);
    yq0 = MulAdd(yq0, x0, c);
    yq1 = MulAdd(yq1, x1, c);
    yq2 = MulAdd(yq2, x2, c);
  }
  if (kDegP >= 2) {
    c = LoadDup128(d, &p[(kDegP - 2) * 4]);
    yp0 = MulAdd(yp0, x0, c);
    yp1 = MulAdd(yp1, x1, c);
    yp2 = MulAdd(yp2, x2, c);
  }
  if (kDegQ >= 2) {
    c = LoadDup128(d, &q[(kDegQ - 2) * 4]);
    yq0 = MulAdd(yq0, x0, c);
    yq1 = MulAdd(yq1, x1, c);
    yq2 = MulAdd(yq2, x2, c);
  }
  if (kDegP >= 3) {
    c = LoadDup128(d, &p[(kDegP - 3) * 4]);
    yp0 = MulAdd(yp0, x0, c);
    yp1 = MulAdd(yp1, x1, c);
    yp2 = MulAdd(yp2, x2, c);
  }
  if (kDegQ >= 3) {
    c = LoadDup128(d, &q[(kDegQ - 3) * 4]);
    yq0 = MulAdd(yq0, x0, c);
    yq1 = MulAdd(yq1, x1, c);
    yq2 = MulAdd(yq2, x2, c);
  }
  if (kDegP >= 4) {
    c = LoadDup128(d, &p[(kDegP - 4) * 4]);
    yp0 = MulAdd(yp0, x0, c);
    yp1 = MulAdd(yp1, x1, c);
    yp2 = MulAdd(yp2, x2, c);
  }
  if (kDegQ >= 4) {
    c = LoadDup128(d, &q[(kDegQ - 4) * 4]);
    yq0 = MulAdd(yq0, x0, c);
    yq1 = MulAdd(yq1, x1, c);
    yq2 = MulAdd(yq2, x2, c);
  }
  if (kDegP >= 5) {
    c = LoadDup128(d, &p[(kDegP - 5) * 4]);
    yp0 = MulAdd(yp0, x0, c);
    yp1 = MulAdd(yp1, x1, c);
    yp2 = MulAdd(yp2, x2, c);
  }
  if (kDegQ >= 5) {
    c = LoadDup128(d, &q[(kDegQ - 5) * 4]);
    yq0 = MulAdd(yq0, x0, c);
    yq1 = MulAdd(yq1, x1, c);
    yq2 = MulAdd(yq2, x2, c);
  }
  if (kDegP >= 6) {
    c = LoadDup128(d, &p[(kDegP - 6) * 4]);
    yp0 = MulAdd(yp0, x0, c);
    yp1 = MulAdd(yp1, x1, c);
    yp2 = MulAdd(yp2, x2, c);
  }
  if (kDegQ >= 6) {
    c = LoadDup128(d, &q[(kDegQ - 6) * 4]);
    yq0 = MulAdd(yq0, x0, c);
    yq1 = MulAdd(yq1, x1, c);
    yq2 = MulAdd(yq2, x2, c);
  }
  if (kDegP >= 7) {
    c = LoadDup128(d, &p[(kDegP - 7) * 4]);
    yp0 = MulAdd(yp0, x0, c);
    yp1 = MulAdd(yp1, x1, c);
    yp2 = MulAdd(yp2, x2, c);
  }
  if (kDegQ >= 7) {
    c = LoadDup128(d, &q[(kDegQ - 7) * 4]);
    yq0 = MulAdd(yq0, x0, c);
    yq1 = MulAdd(yq1, x1, c);
    yq2 = MulAdd(yq2, x2, c);
  }

  // Much faster than division when computing three at once.
  *y0 = FastDivision<T, V>()(yp0, yq0);
  *y1 = FastDivision<T, V>()(yp1, yq1);
  *y2 = FastDivision<T, V>()(yp2, yq2);
}

}  // namespace jxl

#endif  // JXL_RATIONAL_POLYNOMIAL_H_
