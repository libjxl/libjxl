// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <stddef.h>

#include <cstdint>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "lib/extras/codec.h"
#include "lib/extras/dec/jxl.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {

using test::ButteraugliDistance;
using test::ReadTestData;
using test::Roundtrip;
using test::ThreadPoolForTests;

namespace {

TEST(PassesTest, RoundtripSmallPasses) {
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  cparams.progressive_mode = Override::kOn;
  cparams.SetCms(*JxlGetDefaultCms());

  CodecInOut io2;
  JXL_EXPECT_OK(Roundtrip(&io, cparams, {}, &io2, _));
  EXPECT_THAT(ButteraugliDistance(io.frames, io2.frames, ButteraugliParams(),
                                  *JxlGetDefaultCms(),
                                  /*distmap=*/nullptr),
              IsSlightlyBelow(0.8222));
}

TEST(PassesTest, RoundtripUnalignedPasses) {
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io));
  io.ShrinkTo(io.xsize() / 12, io.ysize() / 7);

  CompressParams cparams;
  cparams.butteraugli_distance = 2.0;
  cparams.progressive_mode = Override::kOn;
  cparams.SetCms(*JxlGetDefaultCms());

  CodecInOut io2;
  JXL_EXPECT_OK(Roundtrip(&io, cparams, {}, &io2, _));
  EXPECT_THAT(ButteraugliDistance(io.frames, io2.frames, ButteraugliParams(),
                                  *JxlGetDefaultCms(),
                                  /*distmap=*/nullptr),
              IsSlightlyBelow(1.72));
}

TEST(PassesTest, RoundtripMultiGroupPasses) {
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  {
    ThreadPoolForTests pool(4);
    ASSERT_TRUE(SetFromBytes(Bytes(orig), &io, &pool));
  }
  io.ShrinkTo(600, 1024);  // partial X, full Y group

  auto test = [&](float target_distance, float threshold) {
    ThreadPoolForTests pool(4);
    CompressParams cparams;
    cparams.butteraugli_distance = target_distance;
    cparams.progressive_mode = Override::kOn;
    cparams.SetCms(*JxlGetDefaultCms());
    CodecInOut io2;
    JXL_EXPECT_OK(Roundtrip(&io, cparams, {}, &io2, _,
                            /* compressed_size */ nullptr, &pool));
    EXPECT_THAT(ButteraugliDistance(io.frames, io2.frames, ButteraugliParams(),
                                    *JxlGetDefaultCms(),
                                    /*distmap=*/nullptr, &pool),
                IsSlightlyBelow(target_distance + threshold));
  };

  auto run1 = std::async(std::launch::async, test, 1.0f, 0.15f);
  auto run2 = std::async(std::launch::async, test, 2.0f, 0.0f);
}

TEST(PassesTest, RoundtripLargeFastPasses) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io, &pool));

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_mode = Override::kOn;
  cparams.SetCms(*JxlGetDefaultCms());

  CodecInOut io2;
  JXL_EXPECT_OK(Roundtrip(&io, cparams, {}, &io2, _,
                          /* compressed_size */ nullptr, &pool));
}

// Checks for differing size/distance in two consecutive runs of distance 2,
// which involves additional processing including adaptive reconstruction.
// Failing this may be a sign of race conditions or invalid memory accesses.
TEST(PassesTest, RoundtripProgressiveConsistent) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io, &pool));

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_mode = Override::kOn;
  cparams.butteraugli_distance = 2.0;
  cparams.SetCms(*JxlGetDefaultCms());

  // Try each xsize mod kBlockDim to verify right border handling.
  for (size_t xsize = 48; xsize > 40; --xsize) {
    io.ShrinkTo(xsize, 15);

    CodecInOut io2;
    size_t size2;
    JXL_EXPECT_OK(Roundtrip(&io, cparams, {}, &io2, _, &size2, &pool));

    CodecInOut io3;
    size_t size3;
    JXL_EXPECT_OK(Roundtrip(&io, cparams, {}, &io3, _, &size3, &pool));

    // Exact same compressed size.
    EXPECT_EQ(size2, size3);

    // Exact same distance.
    const float dist2 = ButteraugliDistance(
        io.frames, io2.frames, ButteraugliParams(), *JxlGetDefaultCms(),
        /*distmap=*/nullptr, &pool);
    const float dist3 = ButteraugliDistance(
        io.frames, io3.frames, ButteraugliParams(), *JxlGetDefaultCms(),
        /*distmap=*/nullptr, &pool);
    EXPECT_EQ(dist2, dist3);
  }
}

TEST(PassesTest, AllDownsampleFeasible) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io, &pool));

  std::vector<uint8_t> compressed;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_mode = Override::kOn;
  cparams.butteraugli_distance = 1.0;
  ASSERT_TRUE(test::EncodeFile(cparams, &io, &compressed, &pool));

  EXPECT_LE(compressed.size(), 240000u);
  float target_butteraugli[9] = {};
  target_butteraugli[1] = 2.5f;
  target_butteraugli[2] = 16.0f;
  target_butteraugli[4] = 20.0f;
  target_butteraugli[8] = 80.0f;

  // The default progressive encoding scheme should make all these downsampling
  // factors achievable.
  // TODO(veluca): re-enable downsampling 16.
  std::vector<size_t> downsamplings = {1, 2, 4, 8};  //, 16};

  auto check = [&](const uint32_t task, size_t /* thread */) -> void {
    const size_t downsampling = downsamplings[task];
    extras::JXLDecompressParams dparams;
    dparams.max_downsampling = downsampling;
    CodecInOut output;
    ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output));
    EXPECT_EQ(output.xsize(), io.xsize()) << "downsampling = " << downsampling;
    EXPECT_EQ(output.ysize(), io.ysize()) << "downsampling = " << downsampling;
    EXPECT_LE(ButteraugliDistance(io.frames, output.frames, ButteraugliParams(),
                                  *JxlGetDefaultCms(),
                                  /*distmap=*/nullptr, nullptr),
              target_butteraugli[downsampling])
        << "downsampling: " << downsampling;
  };
  EXPECT_TRUE(RunOnPool(&pool, 0, downsamplings.size(), ThreadPool::NoInit,
                        check, "TestDownsampling"));
}

TEST(PassesTest, AllDownsampleFeasibleQProgressive) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io, &pool));

  std::vector<uint8_t> compressed;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.qprogressive_mode = Override::kOn;
  cparams.butteraugli_distance = 1.0;
  ASSERT_TRUE(test::EncodeFile(cparams, &io, &compressed, &pool));

  EXPECT_LE(compressed.size(), 220000u);

  float target_butteraugli[9] = {};
  target_butteraugli[1] = 3.0f;
  target_butteraugli[2] = 6.0f;
  target_butteraugli[4] = 10.0f;
  target_butteraugli[8] = 80.0f;

  // The default progressive encoding scheme should make all these downsampling
  // factors achievable.
  std::vector<size_t> downsamplings = {1, 2, 4, 8};

  auto check = [&](const uint32_t task, size_t /* thread */) -> void {
    const size_t downsampling = downsamplings[task];
    extras::JXLDecompressParams dparams;
    dparams.max_downsampling = downsampling;
    CodecInOut output;
    ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output));
    EXPECT_EQ(output.xsize(), io.xsize()) << "downsampling = " << downsampling;
    EXPECT_EQ(output.ysize(), io.ysize()) << "downsampling = " << downsampling;
    EXPECT_LE(ButteraugliDistance(io.frames, output.frames, ButteraugliParams(),
                                  *JxlGetDefaultCms(),
                                  /*distmap=*/nullptr),
              target_butteraugli[downsampling])
        << "downsampling: " << downsampling;
  };
  EXPECT_TRUE(RunOnPool(&pool, 0, downsamplings.size(), ThreadPool::NoInit,
                        check, "TestQProgressive"));
}

TEST(PassesTest, ProgressiveDownsample2DegradesCorrectlyGrayscale) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData(
      "external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  CodecInOut io_orig;
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io_orig, &pool));
  Rect rect(0, 0, io_orig.xsize(), 128);
  // need 2 DC groups for the DC frame to actually be progressive.
  JXL_ASSIGN_OR_DIE(Image3F large, Image3F::Create(4242, rect.ysize()));
  ZeroFillImage(&large);
  CopyImageTo(rect, *io_orig.Main().color(), rect, &large);
  CodecInOut io;
  io.metadata = io_orig.metadata;
  io.SetFromImage(std::move(large), io_orig.Main().c_current());

  std::vector<uint8_t> compressed;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_dc = 1;
  cparams.responsive = true;
  cparams.qprogressive_mode = Override::kOn;
  cparams.butteraugli_distance = 1.0;
  ASSERT_TRUE(test::EncodeFile(cparams, &io, &compressed, &pool));

  EXPECT_LE(compressed.size(), 10000u);

  extras::JXLDecompressParams dparams;
  dparams.max_downsampling = 1;
  CodecInOut output;
  ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output));

  dparams.max_downsampling = 2;
  CodecInOut output_d2;
  ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output_d2));

  // 0 if reading all the passes, ~15 if skipping the 8x pass.
  float butteraugli_distance_down2_full = ButteraugliDistance(
      output.frames, output_d2.frames, ButteraugliParams(), *JxlGetDefaultCms(),
      /*distmap=*/nullptr);

  EXPECT_LE(butteraugli_distance_down2_full, 3.2f);
  EXPECT_GE(butteraugli_distance_down2_full, 1.0f);
}

TEST(PassesTest, ProgressiveDownsample2DegradesCorrectly) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io_orig;
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io_orig, &pool));
  Rect rect(0, 0, io_orig.xsize(), 128);
  // need 2 DC groups for the DC frame to actually be progressive.
  JXL_ASSIGN_OR_DIE(Image3F large, Image3F::Create(4242, rect.ysize()));
  ZeroFillImage(&large);
  CopyImageTo(rect, *io_orig.Main().color(), rect, &large);
  CodecInOut io;
  io.SetFromImage(std::move(large), io_orig.Main().c_current());

  std::vector<uint8_t> compressed;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_dc = 1;
  cparams.responsive = true;
  cparams.qprogressive_mode = Override::kOn;
  cparams.butteraugli_distance = 1.0;
  ASSERT_TRUE(test::EncodeFile(cparams, &io, &compressed, &pool));

  EXPECT_LE(compressed.size(), 220000u);

  extras::JXLDecompressParams dparams;
  dparams.max_downsampling = 1;
  CodecInOut output;
  ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output));

  dparams.max_downsampling = 2;
  CodecInOut output_d2;
  ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output_d2));

  // 0 if reading all the passes, ~15 if skipping the 8x pass.
  float butteraugli_distance_down2_full = ButteraugliDistance(
      output.frames, output_d2.frames, ButteraugliParams(), *JxlGetDefaultCms(),
      /*distmap=*/nullptr);

  EXPECT_LE(butteraugli_distance_down2_full, 3.0f);
  EXPECT_GE(butteraugli_distance_down2_full, 1.0f);
}

TEST(PassesTest, NonProgressiveDCImage) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io, &pool));

  std::vector<uint8_t> compressed;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_mode = Override::kOff;
  cparams.butteraugli_distance = 2.0;
  ASSERT_TRUE(test::EncodeFile(cparams, &io, &compressed, &pool));

  // Even in non-progressive mode, it should be possible to return a DC-only
  // image.
  extras::JXLDecompressParams dparams;
  dparams.max_downsampling = 100;
  CodecInOut output;
  ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output, &pool));
  EXPECT_EQ(output.xsize(), io.xsize());
  EXPECT_EQ(output.ysize(), io.ysize());
}

TEST(PassesTest, RoundtripSmallNoGaborishPasses) {
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.gaborish = Override::kOff;
  cparams.butteraugli_distance = 1.0;
  cparams.progressive_mode = Override::kOn;
  cparams.SetCms(*JxlGetDefaultCms());

  CodecInOut io2;
  JXL_EXPECT_OK(Roundtrip(&io, cparams, {}, &io2, _));
  EXPECT_THAT(ButteraugliDistance(io.frames, io2.frames, ButteraugliParams(),
                                  *JxlGetDefaultCms(),
                                  /*distmap=*/nullptr),
              IsSlightlyBelow(1.2));
}

}  // namespace
}  // namespace jxl
