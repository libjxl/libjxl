// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/testing.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/cms/jxl_cms_dec_test.cc"
#include <hwy/foreach_target.h>

#include "lib/jxl/cms/jxl_cms_dec.h"
#include "lib/jxl/cms/tone_mapping-inl.h"
#include "lib/jxl/cms/transfer_functions-inl.h"

// Test utils
#include <hwy/highway.h>
#include <hwy/tests/hwy_gtest.h>
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

template <size_t N>
std::array<uint16_t, N> CreateTableCurve(bool tf_is_pq, bool tone_map) {
  // The generated PQ curve will make room for highlights up to this luminance.
  // TODO(sboukortt): make this variable?
  static constexpr float kPQIntensityTarget = 10000;

  JXL_ASSERT(N <= 4096);  // ICC MFT2 only allows 4K entries
  bool tf_is_hlg = !tf_is_pq;

  static constexpr float kLuminances[] = {1.f / 3, 1.f / 3, 1.f / 3};
  using D = HWY_CAPPED(float, 1);
  Rec2408ToneMapper<D> tone_mapper({0, kPQIntensityTarget},
                                   {0, kDefaultIntensityTarget}, kLuminances);
  // No point using float - LCMS converts to 16-bit for A2B/MFT.
  std::array<uint16_t, N> table;
  TF_PQ tf_pq(/*display_intensity_target=*/10000.0);
  for (uint32_t i = 0; i < N; ++i) {
    const float x = static_cast<float>(i) / (N - 1);  // 1.0 at index N - 1.
    const double dx = static_cast<double>(x);
    // LCMS requires EOTF (e.g. 2.4 exponent).
    double y = tf_is_hlg ? TF_HLG().DisplayFromEncoded(dx)
                         : tf_pq.DisplayFromEncoded(dx);
    if (tone_map && tf_is_pq && kPQIntensityTarget > kDefaultIntensityTarget) {
      D df;
      auto r = Set(df, y * 10000 / kPQIntensityTarget), g = r, b = r;
      tone_mapper.ToneMap(&r, &g, &b);
      float fy;
      StoreU(r, df, &fy);
      y = fy;
    }
    JXL_ASSERT(y >= 0.0);
    // Clamp to table range - necessary for HLG.
    if (y > 1.0) y = 1.0;
    // 1.0 corresponds to table value 0xFFFF.
    table[i] = static_cast<uint16_t>(roundf(y * 65535));
  }
  return table;
}

namespace {
int Delta(std::array<uint16_t, 64> a, std::array<uint16_t, 64> b) {
  int total = 0;
  for (size_t i = 0; i < 64; ++i) {
    total += std::abs(a[i] - b[i]);
  }
  return total;
}
}  // namespace

HWY_NOINLINE void TestCurves() {
  std::array<uint16_t, 64> pq_tm =
      CreateTableCurve<64>(/* pq */ true, /* tone_map */ true);
  EXPECT_LE(Delta(kPqTmCurve, pq_tm), 1);
  std::array<uint16_t, 64> pq =
      CreateTableCurve<64>(/* pq */ true, /* tone_map */ false);
  EXPECT_LE(Delta(kPqCurve, pq), 1);
  std::array<uint16_t, 64> hlg_tm =
      CreateTableCurve<64>(/* pq */ false, /* tone_map */ true);
  EXPECT_LE(Delta(kHlgCurve, hlg_tm), 1);
  std::array<uint16_t, 64> hlg =
      CreateTableCurve<64>(/* pq */ false, /* tone_map */ false);
  EXPECT_LE(Delta(kHlgCurve, hlg), 1);
}

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

class JxlCmsDecTargetTest : public hwy::TestWithParamTarget {};
HWY_TARGET_INSTANTIATE_TEST_SUITE_P(JxlCmsDecTargetTest);

HWY_EXPORT_AND_TEST_P(JxlCmsDecTargetTest, TestCurves);

}  // namespace jxl
#endif  // HWY_ONCE
