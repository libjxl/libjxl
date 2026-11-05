// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_photon_noise.h"

#include "lib/jxl/noise.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

constexpr double kTolerance = 1e-6;

TEST(EncPhotonNoiseTest, ISO100) {
  EXPECT_ARRAY_NEAR(
      SimulatePhotonNoise(/*xsize=*/6000, /*ysize=*/4000, /*iso=*/100).lut,
      (NoiseParams::Lut{0.00185044, 0.00475817, 0.00445666, 0.00489137,
                        0.00566186, 0.00658993, 0.00759693, 0.00864657}),
      kTolerance);
}

TEST(EncPhotoNoiseTest, ISO800) {
  EXPECT_ARRAY_NEAR(
      SimulatePhotonNoise(/*xsize=*/6000, /*ysize=*/4000, /*iso=*/800).lut,
      (NoiseParams::Lut{0.01480355, 0.01285268, 0.01028639, 0.00923201,
                        0.00892596, 0.00907445, 0.00951939, 0.01016191}),
      kTolerance);
}

TEST(EncPhotoNoiseTest, ISO6400) {
  EXPECT_ARRAY_NEAR(
      SimulatePhotonNoise(/*xsize=*/6000, /*ysize=*/4000, /*iso=*/6400).lut,
      (NoiseParams::Lut{0.11842841, 0.03781911, 0.02844290, 0.02407629,
                        0.02149543, 0.01985800, 0.01882002, 0.01820604}),
      kTolerance);
}

TEST(EncPhotoNoiseTest, ISO6400LowRes) {
  // Lower when measured on a per-pixel basis as there are fewer of them.
  EXPECT_ARRAY_NEAR(
      SimulatePhotonNoise(/*xsize=*/4000, /*ysize=*/3000, /*iso=*/6400).lut,
      (NoiseParams::Lut{0.05921420, 0.02608126, 0.02009425, 0.01721223,
                        0.01559596, 0.01469410, 0.01426560, 0.01417822}),
      kTolerance);
}

}  // namespace
}  // namespace jxl
