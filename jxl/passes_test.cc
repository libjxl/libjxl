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

#include <stddef.h>

#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "jxl/aux_out.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/override.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/color_encoding.h"
#include "jxl/common.h"
#include "jxl/dec_file.h"
#include "jxl/dec_params.h"
#include "jxl/enc_butteraugli_comparator.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_file.h"
#include "jxl/enc_params.h"
#include "jxl/extras/codec.h"
#include "jxl/image_bundle.h"
#include "jxl/test_utils.h"
#include "jxl/testdata_path.h"

namespace jxl {
namespace {
using test::Roundtrip;

TEST(PassesTest, RoundtripSmallPasses) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  cparams.progressive_mode = true;
  DecompressParams dparams;

  CodecInOut io2;
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.5);
}

TEST(PassesTest, RoundtripUnalignedPasses) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 12, io.ysize() / 7);

  CompressParams cparams;
  cparams.butteraugli_distance = 2.0;
  cparams.progressive_mode = true;
  DecompressParams dparams;

  CodecInOut io2;
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            3.2);
}

TEST(PassesTest, RoundtripMultiGroupPasses) {
  ThreadPoolInternal pool(4);
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));
  io.ShrinkTo(600, 1024);  // partial X, full Y group

  CompressParams cparams;
  DecompressParams dparams;

  cparams.butteraugli_distance = 1.0f;
  cparams.progressive_mode = true;
  CodecInOut io2;
  Roundtrip(&io, cparams, dparams, &pool, &io2);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, &pool),
            1.99f);

  cparams.butteraugli_distance = 2.0f;
  CodecInOut io3;
  Roundtrip(&io, cparams, dparams, &pool, &io3);
  EXPECT_LE(ButteraugliDistance(io, io3, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, &pool),
            3.0f);
}

TEST(PassesTest, RoundtripLargeFastPasses) {
  ThreadPoolInternal pool(8);
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_mode = true;
  DecompressParams dparams;

  CodecInOut io2;
  Roundtrip(&io, cparams, dparams, &pool, &io2);
}

// Checks for differing size/distance in two consecutive runs of distance 2,
// which involves additional processing including adaptive reconstruction.
// Failing this may be a sign of race conditions or invalid memory accesses.
TEST(PassesTest, RoundtripProgressiveConsistent) {
  ThreadPoolInternal pool(8);
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_mode = true;
  cparams.butteraugli_distance = 2.0;
  DecompressParams dparams;

  // Try each xsize mod kBlockDim to verify right border handling.
  for (size_t xsize = 48; xsize > 40; --xsize) {
    io.ShrinkTo(xsize, 15);

    CodecInOut io2;
    const size_t size2 = Roundtrip(&io, cparams, dparams, &pool, &io2);

    CodecInOut io3;
    const size_t size3 = Roundtrip(&io, cparams, dparams, &pool, &io3);

    // Exact same compressed size.
    EXPECT_EQ(size2, size3);

    // Exact same distance.
    const float dist2 = ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                            /*distmap=*/nullptr, &pool);
    const float dist3 = ButteraugliDistance(io, io3, cparams.hf_asymmetry,
                                            /*distmap=*/nullptr, &pool);
    EXPECT_EQ(dist2, dist3);
  }
}

TEST(PassesTest, AllDownsampleFeasible) {
  ThreadPoolInternal pool(8);
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));

  PaddedBytes compressed;
  AuxOut aux;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_mode = true;
  cparams.butteraugli_distance = 1.0;
  PassesEncoderState enc_state;
  ASSERT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, &aux, &pool));

  EXPECT_LE(compressed.size(), 240000);
  float target_butteraugli[9] = {};
  target_butteraugli[1] = 2.5f;
  target_butteraugli[2] = 13.0f;
  target_butteraugli[4] = 20.0f;
  target_butteraugli[8] = 60.0f;

  // The default progressive encoding scheme should make all these downsampling
  // factors achievable.
  // TODO(veluca): re-enable downsampling 16.
  std::vector<size_t> downsamplings = {1, 2, 4, 8};  //, 16};

  auto check = [&](uint32_t task, uint32_t /* thread */) -> void {
    const size_t downsampling = downsamplings[task];
    DecompressParams dparams;
    dparams.max_downsampling = downsampling;
    AuxOut aux_downsampled = aux;
    CodecInOut output;
    ASSERT_TRUE(
        DecodeFile(dparams, compressed, &output, &aux_downsampled, nullptr));
    EXPECT_EQ(aux_downsampled.downsampling, downsampling);
    EXPECT_EQ(output.xsize(), io.xsize()) << "downsampling = " << downsampling;
    EXPECT_EQ(output.ysize(), io.ysize()) << "downsampling = " << downsampling;
    EXPECT_LE(ButteraugliDistance(io, output, cparams.hf_asymmetry,
                                  /*distmap=*/nullptr, nullptr),
              target_butteraugli[downsampling])
        << "downsampling: " << downsampling;
  };
  pool.Run(0, downsamplings.size(), ThreadPool::SkipInit(), check);
}

TEST(PassesTest, AllDownsampleFeasibleQProgressive) {
  ThreadPoolInternal pool(8);
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));

  PaddedBytes compressed;
  AuxOut aux;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.qprogressive_mode = true;
  cparams.butteraugli_distance = 1.0;
  PassesEncoderState enc_state;
  ASSERT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, &aux, &pool));

  EXPECT_LE(compressed.size(), 220000);

  float target_butteraugli[9] = {};
  target_butteraugli[1] = 3.0f;
  target_butteraugli[2] = 6.0f;
  target_butteraugli[4] = 10.0f;
  target_butteraugli[8] = 60.0f;

  // The default progressive encoding scheme should make all these downsampling
  // factors achievable.
  std::vector<size_t> downsamplings = {1, 2, 4, 8};

  auto check = [&](uint32_t task, uint32_t /* thread */) -> void {
    const size_t downsampling = downsamplings[task];
    DecompressParams dparams;
    dparams.max_downsampling = downsampling;
    CodecInOut output;
    AuxOut aux_downsampled = aux;
    ASSERT_TRUE(
        DecodeFile(dparams, compressed, &output, &aux_downsampled, nullptr));
    EXPECT_EQ(aux_downsampled.downsampling, downsampling);
    EXPECT_EQ(output.xsize(), io.xsize()) << "downsampling = " << downsampling;
    EXPECT_EQ(output.ysize(), io.ysize()) << "downsampling = " << downsampling;
    EXPECT_LE(ButteraugliDistance(io, output, cparams.hf_asymmetry,
                                  /*distmap=*/nullptr, nullptr),
              target_butteraugli[downsampling])
        << "downsampling: " << downsampling;
  };
  pool.Run(0, downsamplings.size(), ThreadPool::SkipInit(), check);
}

TEST(PassesTest, NonProgressiveDCImage) {
  ThreadPoolInternal pool(8);
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));

  PaddedBytes compressed;
  AuxOut aux;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_mode = false;
  cparams.butteraugli_distance = 2.0;
  PassesEncoderState enc_state;
  ASSERT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, &aux, &pool));

  // Even in non-progressive mode, it should be possible to return a DC-only
  // image.
  DecompressParams dparams;
  dparams.max_downsampling = 100;
  CodecInOut output;
  ASSERT_TRUE(DecodeFile(dparams, compressed, &output, &aux, &pool));
  constexpr decltype(output.xsize()) expected_downscale = 8;
  EXPECT_EQ(aux.downsampling, expected_downscale);
  EXPECT_EQ(output.xsize(), io.xsize());
  EXPECT_EQ(output.ysize(), io.ysize());
}

TEST(PassesTest, RoundtripSmallNoGaborishPasses) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.gaborish = Override::kOff;
  cparams.butteraugli_distance = 1.0;
  cparams.progressive_mode = true;
  DecompressParams dparams;

  CodecInOut io2;
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.7);
}

}  // namespace
}  // namespace jxl
