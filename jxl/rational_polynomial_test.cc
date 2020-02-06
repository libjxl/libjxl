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

#ifndef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/rational_polynomial_test.cc"

#include <stdio.h>

#include <cmath>
#include <string>

#define HWY_USE_GTEST
#include <hwy/tests/test_util.h>

#include "jxl/base/descriptive_statistics.h"
#include "jxl/common.h"

struct RationalPolynomialTest {
  HWY_DECLARE(void, ())
};
TEST(RationalPolynomialTest, Run) { hwy::RunTests<RationalPolynomialTest>(); }

#endif  // HWY_TARGET_INCLUDE
#include <hwy/tests/test_target_util.h>

namespace jxl {
namespace HWY_NAMESPACE {
namespace {

#include "jxl/rational_polynomial-inl.h"

#if HWY_HAS_DOUBLE
using T = double;
#else
using T = float;
#endif
using D = HWY_FULL(T);

// Functions to approximate:

T LinearToSrgb8Direct(T val) {
  if (val < 0.0) return 0.0;
  if (val >= 255.0) return 255.0;
  if (val <= 10.0 / 12.92) return val * 12.92;
  return 255.0 * (std::pow(val / 255.0, 1.0 / 2.4) * 1.055 - 0.055);
}

T SimpleGamma(T v) {
  static const T kGamma = 0.387494322593;
  static const T limit = 43.01745241042018;
  T bright = v - limit;
  if (bright >= 0) {
    static const T mul = 0.0383723643799;
    v -= bright * mul;
  }
  static const T limit2 = 94.68634353321337;
  T bright2 = v - limit2;
  if (bright2 >= 0) {
    static const T mul = 0.22885405968;
    v -= bright2 * mul;
  }
  static const T offset = 0.156775786057;
  static const T scale = 8.898059160493739;
  T retval = scale * (offset + pow(v, kGamma));
  return retval;
}

// Runs CaratheodoryFejer and verifies the polynomial using a lot of samples to
// return the biggest error.
template <size_t NP, size_t NQ>
HWY_ATTR T RunApproximation(T x0, T x1, const T (&p)[NP], const T (&q)[NQ],
                            T func_to_approx(T)) {
  Stats err;

  T lastPrint = 0;
  // NOLINTNEXTLINE(clang-analyzer-security.FloatLoopCounter)
  for (T x = x0; x <= x1; x += (x1 - x0) / 10000.0) {
    const T f = func_to_approx(x);
    const HWY_FULL(T) d;
    HWY_ALIGN T g_lanes[d.N];
    const auto approx = EvalRationalPolynomial(Set(d, x), p, q);
    Store(approx, d, g_lanes);
    const T g = g_lanes[0];
    err.Notify(fabs(g - f));
    if (x == x0 || x - lastPrint > (x1 - x0) / 20.0) {
      printf("x: %11.6f, f: %11.6f, g: %11.6f, e: %11.6f\n", x, f, g,
             fabs(g - f));
      lastPrint = x;
    }
  }
  printf("%s\n", err.ToString().c_str());

  return err.Max();
}

HWY_ATTR void TestSimpleGamma() {
  const T p[4 * (6 + 1)] = {
      HWY_REP4(-5.0646949363741811E-05), HWY_REP4(6.7369380528439771E-05),
      HWY_REP4(8.9376652530412794E-05),  HWY_REP4(2.1153513301520462E-06),
      HWY_REP4(-6.9130322970386449E-08), HWY_REP4(3.9424752749293728E-10),
      HWY_REP4(1.2360288207619576E-13)};

  const T q[4 * (6 + 1)] = {
      HWY_REP4(-6.6389733798591366E-06), HWY_REP4(1.3299859726565908E-05),
      HWY_REP4(3.8538748358398873E-06),  HWY_REP4(-2.8707687262928236E-08),
      HWY_REP4(-6.6897385800005434E-10), HWY_REP4(6.1428748869186003E-12),
      HWY_REP4(-2.5475738169252870E-15)};

  const T err = RunApproximation(0.77, 274.579999999999984, p, q, SimpleGamma);
  EXPECT_LT(err, 0.05);
}

HWY_ATTR void TestLinearToSrgb8Direct() {
  const T p[4 * (5 + 1)] = {
      HWY_REP4(-9.5357499040105154E-05), HWY_REP4(4.6761186249798248E-04),
      HWY_REP4(2.5708174333943594E-04),  HWY_REP4(1.5250087770436082E-05),
      HWY_REP4(1.1946768008931187E-07),  HWY_REP4(5.9916446295972850E-11)};

  const T q[4 * (4 + 1)] = {
      HWY_REP4(1.8932479758079768E-05), HWY_REP4(2.7312342474687321E-05),
      HWY_REP4(4.3901204783327006E-06), HWY_REP4(1.0417787306920273E-07),
      HWY_REP4(3.0084206762140419E-10)};

  const T err = RunApproximation(0.77, 255, p, q, LinearToSrgb8Direct);
  EXPECT_LT(err, 0.05);
}

HWY_ATTR void TestExp() {
  const T p[4 * (2 + 1)] = {HWY_REP4(9.6266879665530902E-01),
                            HWY_REP4(4.8961265681586763E-01),
                            HWY_REP4(8.2619259189548433E-02)};
  const T q[4 * (2 + 1)] = {HWY_REP4(9.6259895571622622E-01),
                            HWY_REP4(-4.7272457588933831E-01),
                            HWY_REP4(7.4802088567547664E-02)};
  const T err = RunApproximation(-1, 1, p, q, [](T x) { return T(exp(x)); });
  EXPECT_LT(err, 1E-4);
}

HWY_ATTR void TestNegExp() {
  // 4,3 is the min required for monotonicity; max error in 0,10: 751 ppm
  // no benefit for k>50.
  const T p[4 * (4 + 1)] = {
      HWY_REP4(5.9580258551150123E-02), HWY_REP4(-2.5073728806886408E-02),
      HWY_REP4(4.1561830213689248E-03), HWY_REP4(-3.1815408488900372E-04),
      HWY_REP4(9.3866690094906802E-06)};
  const T q[4 * (3 + 1)] = {
      HWY_REP4(5.9579108238812878E-02), HWY_REP4(3.4542074345478582E-02),
      HWY_REP4(8.7263562483501714E-03), HWY_REP4(1.4095109143061216E-03)};

  const T err = RunApproximation(0, 10, p, q, [](T x) { return T(exp(-x)); });
#if HWY_HAS_DOUBLE
  EXPECT_LT(err, 2E-5);
#else
  EXPECT_LT(err, 3E-5);
#endif
}

HWY_ATTR void TestSin() {
  const T p[4 * (6 + 1)] = {
      HWY_REP4(1.5518122109203780E-05),  HWY_REP4(2.3388958643675966E+00),
      HWY_REP4(-8.6705520940849157E-01), HWY_REP4(-1.9702294764873535E-01),
      HWY_REP4(1.2193404314472320E-01),  HWY_REP4(-1.7373966109788839E-02),
      HWY_REP4(7.8829435883034796E-04)};
  const T q[4 * (5 + 1)] = {
      HWY_REP4(2.3394371422557279E+00), HWY_REP4(-8.7028221081288615E-01),
      HWY_REP4(2.0052872219658430E-01), HWY_REP4(-3.2460335995264836E-02),
      HWY_REP4(3.1546157932479282E-03), HWY_REP4(-1.6692542019380155E-04)};

  const T err =
      RunApproximation(0, Pi<T>(1) * 2, p, q, [](T x) { return T(sin(x)); });
#if HWY_HAS_DOUBLE
  EXPECT_LT(err, 5E-4);
#else
  EXPECT_LT(err, 7E-4);
#endif
}

HWY_ATTR HWY_NOINLINE void RunAll() {
  TestSimpleGamma();
  TestLinearToSrgb8Direct();
  TestExp();
  TestNegExp();
  TestSin();
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl

// Instantiate for the current target.
void RationalPolynomialTest::HWY_FUNC() { jxl::HWY_NAMESPACE::RunAll(); }
