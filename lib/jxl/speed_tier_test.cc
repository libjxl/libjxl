// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lib/extras/codec.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

struct SpeedTierTestParams {
  explicit SpeedTierTestParams(const SpeedTier speed_tier,
                               const bool shrink8 = false)
      : speed_tier(speed_tier), shrink8(shrink8) {}
  SpeedTier speed_tier;
  bool shrink8;
};

std::ostream& operator<<(std::ostream& os, SpeedTierTestParams params) {
  auto previous_flags = os.flags();
  os << std::boolalpha;
  os << "SpeedTierTestParams{" << static_cast<size_t>(params.speed_tier)
     << ", /*shrink8=*/" << params.shrink8 << "}";
  os.flags(previous_flags);
  return os;
}

class SpeedTierTest : public testing::TestWithParam<SpeedTierTestParams> {};

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(
    SpeedTierTestInstantiation, SpeedTierTest,
    testing::Values(SpeedTierTestParams{SpeedTier::kCheetah,
                                        /*shrink8=*/true},
                    SpeedTierTestParams{SpeedTier::kCheetah,
                                        /*shrink8=*/false},
                    SpeedTierTestParams{SpeedTier::kThunder,
                                        /*shrink8=*/true},
                    SpeedTierTestParams{SpeedTier::kThunder,
                                        /*shrink8=*/false},
                    SpeedTierTestParams{SpeedTier::kLightning,
                                        /*shrink8=*/true},
                    SpeedTierTestParams{SpeedTier::kLightning,
                                        /*shrink8=*/false},
                    SpeedTierTestParams{SpeedTier::kFalcon,
                                        /*shrink8=*/true},
                    SpeedTierTestParams{SpeedTier::kFalcon,
                                        /*shrink8=*/false},
                    SpeedTierTestParams{SpeedTier::kHare,
                                        /*shrink8=*/true},
                    SpeedTierTestParams{SpeedTier::kHare,
                                        /*shrink8=*/false},
                    SpeedTierTestParams{SpeedTier::kWombat,
                                        /*shrink8=*/true},
                    SpeedTierTestParams{SpeedTier::kWombat,
                                        /*shrink8=*/false},
                    SpeedTierTestParams{SpeedTier::kSquirrel,
                                        /*shrink8=*/true},
                    SpeedTierTestParams{SpeedTier::kSquirrel,
                                        /*shrink8=*/false},
                    SpeedTierTestParams{SpeedTier::kKitten,
                                        /*shrink8=*/false},
                    // Only downscaled image for Tortoise mode.
                    SpeedTierTestParams{SpeedTier::kTortoise,
                                        /*shrink8=*/true},
                    SpeedTierTestParams{SpeedTier::kGlacier,
                                        /*shrink8=*/true}));

TEST_P(SpeedTierTest, Roundtrip) {
  const std::vector<uint8_t> orig = jxl::test::ReadTestData(
      "external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  test::ThreadPoolForTests pool(8);
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io, &pool));

  const SpeedTierTestParams& params = GetParam();

  if (params.speed_tier == SpeedTier::kGlacier) {
    // just a few pixels will already take enough time at this setting
    io.ShrinkTo(8, 8);
  } else if (params.shrink8) {
    io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);
  }

  CompressParams cparams;
  cparams.speed_tier = params.speed_tier;
  cparams.SetCms(*JxlGetDefaultCms());

  CodecInOut io2, io3;
  JXL_EXPECT_OK(test::Roundtrip(&io, cparams, {}, &io2, _));
  EXPECT_LE(ButteraugliDistance(io.frames, io2.frames, ButteraugliParams(),
                                *JxlGetDefaultCms(),
                                /*distmap=*/nullptr, /*pool=*/nullptr),
            1.6);

  if (params.shrink8) {
    cparams.SetLossless();
    JXL_EXPECT_OK(test::Roundtrip(&io, cparams, {}, &io3, _));

    JXL_EXPECT_OK(SamePixels(*io.Main().color(), *io3.Main().color(), _));
  }
}
}  // namespace
}  // namespace jxl
