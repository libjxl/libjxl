// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/jxl.h"

#include <stdint.h>
#include <stdio.h>

#include <array>
#include <future>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "lib/extras/codec.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/codec_y4m_testonly.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/enc_butteraugli_pnorm.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_file.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/fake_parallel_runner_testonly.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/jpeg/dec_jpeg_data.h"
#include "lib/jxl/jpeg/dec_jpeg_data_writer.h"
#include "lib/jxl/jpeg/enc_jpeg_data.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testdata.h"
#include "tools/box/box.h"

namespace jxl {
namespace {
using extras::JXLCompressParams;
using extras::JXLDecompressParams;
using extras::PackedPixelFile;
using test::ButteraugliDistance;
using test::ComputeDistance2;
using test::Roundtrip;
using test::TestImage;

#define JXL_TEST_NL 0  // Disabled in code

TEST(JxlTest, RoundtripSinglePixel) {
  TestImage t;
  t.SetDimensions(1, 1).AddFrame().ZeroFill();
  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), {}, {}, nullptr, &ppf_out), 58);
}

TEST(JxlTest, RoundtripSinglePixelWithAlpha) {
  TestImage t;
  t.SetDimensions(1, 1).SetChannels(4).AddFrame().ZeroFill();
  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), {}, {}, nullptr, &ppf_out), 63);
}

// Changing serialized signature causes Decode to fail.
#ifndef JXL_CRASH_ON_ERROR
TEST(JxlTest, RoundtripMarker) {
  TestImage t;
  t.SetDimensions(1, 1).AddFrame().ZeroFill();
  for (size_t i = 0; i < 2; ++i) {
    std::vector<uint8_t> compressed;
    EXPECT_TRUE(extras::EncodeImageJXL({}, t.ppf(), /*jpeg_bytes=*/nullptr,
                                       &compressed));
    compressed[i] ^= 0xFF;
    PackedPixelFile ppf_out;
    EXPECT_FALSE(extras::DecodeImageJXL(compressed.data(), compressed.size(),
                                        {}, /*decodec_bytes=*/nullptr,
                                        &ppf_out));
  }
}
#endif

TEST(JxlTest, RoundtripTinyFast) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata().SetDimensions(32, 32);

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.distance = 4.0f;

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 202);
}

TEST(JxlTest, RoundtripSmallD1) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();
  size_t xsize = t.ppf().info.xsize / 8;
  size_t ysize = t.ppf().info.ysize / 8;
  t.SetDimensions(xsize, ysize);

  {
    PackedPixelFile ppf_out;
    EXPECT_EQ(Roundtrip(t.ppf(), {}, {}, pool, &ppf_out), 772);
    EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(1.0));
  }

  // With a lower intensity target than the default, the bitrate should be
  // smaller.
  t.ppf().info.intensity_target = 100.0f;

  {
    PackedPixelFile ppf_out;
    EXPECT_EQ(Roundtrip(t.ppf(), {}, {}, pool, &ppf_out), 626);
    EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(1.1));
    EXPECT_EQ(ppf_out.info.intensity_target, t.ppf().info.intensity_target);
  }
}
TEST(JxlTest, RoundtripResample2) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 3);  // kFalcon

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 16599);
  EXPECT_THAT(ComputeDistance2(t.ppf(), ppf_out), IsSlightlyBelow(90));
}

TEST(JxlTest, RoundtripResample2Slow) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 9);  // kTortoise
  cparams.distance = 10.0;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 4118, 80);
  EXPECT_THAT(ComputeDistance2(t.ppf(), ppf_out), IsSlightlyBelow(250));
}

TEST(JxlTest, RoundtripResample2MT) {
  ThreadPoolInternal pool(4);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  // image has to be large enough to have multiple groups after downsampling
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 3);  // kFalcon

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), 192785);
  EXPECT_THAT(ComputeDistance2(t.ppf(), ppf_out), IsSlightlyBelow(340));
}

// Roundtrip the image using a parallel runner that executes single-threaded but
// in random order.
TEST(JxlTest, RoundtripOutOfOrderProcessing) {
  FakeParallelRunner fake_pool(/*order_seed=*/123, /*num_threads=*/8);
  ThreadPool pool(&JxlFakeParallelRunner, &fake_pool);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();
  // Image size is selected so that the block border needed is larger than the
  // amount of pixels available on the next block.
  t.SetDimensions(513, 515);

  JXLCompressParams cparams;
  // Force epf so we end up needing a lot of border.
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EPF, 3);

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), 22686, 50);
  EXPECT_LE(ButteraugliDistance(t.ppf(), ppf_out), 1.5);
}

TEST(JxlTest, RoundtripOutOfOrderProcessingBorder) {
  FakeParallelRunner fake_pool(/*order_seed=*/47, /*num_threads=*/8);
  ThreadPool pool(&JxlFakeParallelRunner, &fake_pool);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();
  // Image size is selected so that the block border needed is larger than the
  // amount of pixels available on the next block.
  t.SetDimensions(513, 515);

  JXLCompressParams cparams;
  // Force epf so we end up needing a lot of border.
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EPF, 3);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 2);

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), 9906, 30);
  EXPECT_LE(ButteraugliDistance(t.ppf(), ppf_out), 2.8);
}

TEST(JxlTest, RoundtripResample4) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 4);

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 5628, 20);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(22));
}

TEST(JxlTest, RoundtripResample8) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 8);

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 1997, 20);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(50));
}

TEST(JxlTest, RoundtripUnalignedD2) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();
  size_t xsize = t.ppf().info.xsize / 12;
  size_t ysize = t.ppf().info.ysize / 7;
  t.SetDimensions(xsize, ysize);

  JXLCompressParams cparams;
  cparams.distance = 2.0;

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 494);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(1.7));
}

TEST(JxlTest, RoundtripMultiGroup) {
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata().SetDimensions(600, 1024);

  auto test = [&](jxl::SpeedTier speed_tier, float target_distance,
                  size_t expected_size, float expected_distance) {
    ThreadPoolInternal pool(4);
    JXLCompressParams cparams;
    int64_t effort = 10 - static_cast<int>(speed_tier);
    cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, effort);
    cparams.distance = target_distance;

    PackedPixelFile ppf_out;
    EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), expected_size,
                70);
    EXPECT_THAT(ComputeDistance2(t.ppf(), ppf_out),
                IsSlightlyBelow(expected_distance));
  };

  auto run_kitten = std::async(std::launch::async, test, SpeedTier::kKitten,
                               1.0f, 52571u, 11);
  auto run_wombat = std::async(std::launch::async, test, SpeedTier::kWombat,
                               2.0f, 33666u, 18);
}

TEST(JxlTest, RoundtripRGBToGrayscale) {
  ThreadPoolInternal pool(4);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  io.ShrinkTo(600, 1024);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0f;
  cparams.speed_tier = SpeedTier::kFalcon;

  JXLDecompressParams dparams;
  dparams.color_space = "Gra_D65_Rel_SRG";

  CodecInOut io2;
  EXPECT_FALSE(io.Main().IsGray());
  EXPECT_LE(Roundtrip(&io, cparams, dparams, &pool, &io2), 55000u);
  EXPECT_TRUE(io2.Main().IsGray());

  // Convert original to grayscale here, because TransformTo refuses to
  // convert between grayscale and RGB.
  ColorEncoding srgb_lin = ColorEncoding::LinearSRGB(/*is_gray=*/false);
  ASSERT_TRUE(io.TransformTo(srgb_lin, GetJxlCms(), &pool));
  Image3F* color = io.Main().color();
  for (size_t y = 0; y < color->ysize(); ++y) {
    float* row_r = color->PlaneRow(0, y);
    float* row_g = color->PlaneRow(1, y);
    float* row_b = color->PlaneRow(2, y);
    for (size_t x = 0; x < color->xsize(); ++x) {
      float luma = 0.2126 * row_r[x] + 0.7152 * row_g[x] + 0.0722 * row_b[x];
      row_r[x] = row_g[x] = row_b[x] = luma;
    }
  }
  ColorEncoding srgb_gamma = ColorEncoding::SRGB(/*is_gray=*/false);
  ASSERT_TRUE(io.TransformTo(srgb_gamma, GetJxlCms(), &pool));
  io.metadata.m.color_encoding = io2.Main().c_current();
  io.Main().OverrideProfile(io2.Main().c_current());
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, &pool),
              IsSlightlyBelow(1.7));
}

TEST(JxlTest, RoundtripLargeFast) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel

  PackedPixelFile ppf_out;
  // TODO(szabadka) Investigate size difference on SCALAR build.
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), 450746, 300);
  EXPECT_THAT(ComputeDistance2(t.ppf(), ppf_out), IsSlightlyBelow(90));
}

TEST(JxlTest, RoundtripDotsForceEpf) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/cvo9xd_keong_macan_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EPF, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_DOTS, 1);

  PackedPixelFile ppf_out;
  // TODO(szabadka) Investigate size difference on SCALAR build.
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), 38729, 130);
  EXPECT_THAT(ComputeDistance2(t.ppf(), ppf_out), IsSlightlyBelow(18));
}

// Checks for differing size/distance in two consecutive runs of distance 2,
// which involves additional processing including adaptive reconstruction.
// Failing this may be a sign of race conditions or invalid memory accesses.
TEST(JxlTest, RoundtripD2Consistent) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.distance = 2.0;

  // Try each xsize mod kBlockDim to verify right border handling.
  for (size_t xsize = 48; xsize > 40; --xsize) {
    t.SetDimensions(xsize, 15);

    PackedPixelFile ppf2;
    const size_t size2 = Roundtrip(t.ppf(), cparams, {}, &pool, &ppf2);

    PackedPixelFile ppf3;
    const size_t size3 = Roundtrip(t.ppf(), cparams, {}, &pool, &ppf3);

    // Exact same compressed size.
    EXPECT_EQ(size2, size3);

    // Exact same distance.
    const float dist2 = ComputeDistance2(t.ppf(), ppf2);
    const float dist3 = ComputeDistance2(t.ppf(), ppf3);
    EXPECT_EQ(dist2, dist3);
  }
}

// Same as above, but for full image, testing multiple groups.
TEST(JxlTest, RoundtripLargeConsistent) {
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.distance = 2.0;

  auto roundtrip_and_compare = [&]() {
    ThreadPoolInternal pool(8);
    PackedPixelFile ppf2;
    size_t size = Roundtrip(t.ppf(), cparams, {}, &pool, &ppf2);
    double dist = ComputeDistance2(t.ppf(), ppf2);
    return std::tuple<size_t, double>(size, dist);
  };

  // Try each xsize mod kBlockDim to verify right border handling.
  auto future2 = std::async(std::launch::async, roundtrip_and_compare);
  auto future3 = std::async(std::launch::async, roundtrip_and_compare);

  const auto result2 = future2.get();
  const auto result3 = future3.get();

  // Exact same compressed size.
  EXPECT_EQ(std::get<0>(result2), std::get<0>(result3));

  // Exact same distance.
  EXPECT_EQ(std::get<1>(result2), std::get<1>(result3));
}

TEST(JxlTest, RoundtripSmallNL) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();
  size_t xsize = t.ppf().info.xsize / 8;
  size_t ysize = t.ppf().info.ysize / 8;
  t.SetDimensions(xsize, ysize);

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), {}, {}, pool, &ppf_out), 772);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(1.0));
}

TEST(JxlTest, RoundtripNoGaborishNoAR) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EPF, 0);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_GABORISH, 0);

  PackedPixelFile ppf_out;
  // TODO(szabadka) Investigate size difference on SCALAR build.
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 36756, 100);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(2.0));
}

TEST(JxlTest, RoundtripSmallNoGaborish) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();
  size_t xsize = t.ppf().info.xsize / 8;
  size_t ysize = t.ppf().info.ysize / 8;
  t.SetDimensions(xsize, ysize);

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_GABORISH, 0);

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 813);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(1.2));
}

TEST(JxlTest, RoundtripSmallPatchesAlpha) {
  ThreadPool* pool = nullptr;
  TestImage t;
  t.SetDimensions(256, 256).SetChannels(4);
  t.SetColorEncoding("RGB_D65_SRG_Rel_Lin");
  TestImage::Frame frame = t.AddFrame();
  frame.ZeroFill();
  // This pattern should be picked up by the patch detection heuristics.
  for (size_t y = 0; y < t.ppf().info.ysize; ++y) {
    for (size_t x = 0; x < t.ppf().info.xsize; ++x) {
      if (x % 4 == 0 && (y / 32) % 4 == 0) {
        frame.SetValue(y, x, 1, 127.0f / 255.0f);
      }
      frame.SetValue(y, x, 3, 1.0f);
    }
  }

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.distance = 0.1f;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 806, 15);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(0.03f));
}

TEST(JxlTest, RoundtripSmallPatches) {
  ThreadPool* pool = nullptr;
  TestImage t;
  t.SetDimensions(256, 256);
  t.SetColorEncoding("RGB_D65_SRG_Rel_Lin");
  TestImage::Frame frame = t.AddFrame();
  frame.ZeroFill();
  // This pattern should be picked up by the patch detection heuristics.
  for (size_t y = 0; y < t.ppf().info.ysize; ++y) {
    for (size_t x = 0; x < t.ppf().info.xsize; ++x) {
      if (x % 4 == 0 && (y / 32) % 4 == 0) {
        frame.SetValue(y, x, 1, 127.0f / 255.0f);
      }
    }
  }

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);  // kSquirrel
  cparams.distance = 0.1f;

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 701, 10);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(0.03f));
}

// TODO(szabadka) Add encoder and decoder API functions that accept frame
// buffers in arbitrary unsigned and floating point formats, and then roundtrip
// test the lossless codepath to make sure the exact binary representations
// are preserved.
#if 0
TEST(JxlTest, RoundtripImageBundleOriginalBits) {
  ThreadPool* pool = nullptr;

  // Image does not matter, only io.metadata.m and io2.metadata.m are tested.
  Image3F image(1, 1);
  ZeroFillImage(&image);
  CodecInOut io;
  io.metadata.m.color_encoding = ColorEncoding::LinearSRGB();
  io.SetFromImage(std::move(image), ColorEncoding::LinearSRGB());

  CompressParams cparams;

  // Test unsigned integers from 1 to 32 bits
  for (uint32_t bit_depth = 1; bit_depth <= 32; bit_depth++) {
    if (bit_depth == 32) {
      // TODO(lode): allow testing 32, however the code below ends up in
      // enc_modular which does not support 32. We only want to test the header
      // encoding though, so try without modular.
      break;
    }

    io.metadata.m.SetUintSamples(bit_depth);
    CodecInOut io2;
    Roundtrip(&io, cparams, {}, pool, &io2);

    EXPECT_EQ(bit_depth, io2.metadata.m.bit_depth.bits_per_sample);
    EXPECT_FALSE(io2.metadata.m.bit_depth.floating_point_sample);
    EXPECT_EQ(0u, io2.metadata.m.bit_depth.exponent_bits_per_sample);
    EXPECT_EQ(0u, io2.metadata.m.GetAlphaBits());
  }

  // Test various existing and non-existing floating point formats
  for (uint32_t bit_depth = 8; bit_depth <= 32; bit_depth++) {
    if (bit_depth != 32) {
      // TODO: test other float types once they work
      break;
    }

    uint32_t exponent_bit_depth;
    if (bit_depth < 10) {
      exponent_bit_depth = 2;
    } else if (bit_depth < 12) {
      exponent_bit_depth = 3;
    } else if (bit_depth < 16) {
      exponent_bit_depth = 4;
    } else if (bit_depth < 20) {
      exponent_bit_depth = 5;
    } else if (bit_depth < 24) {
      exponent_bit_depth = 6;
    } else if (bit_depth < 28) {
      exponent_bit_depth = 7;
    } else {
      exponent_bit_depth = 8;
    }

    io.metadata.m.bit_depth.bits_per_sample = bit_depth;
    io.metadata.m.bit_depth.floating_point_sample = true;
    io.metadata.m.bit_depth.exponent_bits_per_sample = exponent_bit_depth;

    CodecInOut io2;
    Roundtrip(&io, cparams, {}, pool, &io2);

    EXPECT_EQ(bit_depth, io2.metadata.m.bit_depth.bits_per_sample);
    EXPECT_TRUE(io2.metadata.m.bit_depth.floating_point_sample);
    EXPECT_EQ(exponent_bit_depth,
              io2.metadata.m.bit_depth.exponent_bits_per_sample);
    EXPECT_EQ(0u, io2.metadata.m.GetAlphaBits());
  }
}
#endif

TEST(JxlTest, RoundtripGrayscale) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig = ReadTestData(
      "external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  ASSERT_NE(io.xsize(), 0u);
  io.ShrinkTo(128, 128);
  EXPECT_TRUE(io.Main().IsGray());
  EXPECT_EQ(8u, io.metadata.m.bit_depth.bits_per_sample);
  EXPECT_FALSE(io.metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(0u, io.metadata.m.bit_depth.exponent_bits_per_sample);
  EXPECT_TRUE(io.metadata.m.color_encoding.tf.IsSRGB());

  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;

  {
    CompressParams cparams;
    cparams.butteraugli_distance = 1.0;

    PaddedBytes compressed;
    EXPECT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, GetJxlCms(),
                           aux_out, pool));
    CodecInOut io2;
    EXPECT_TRUE(test::DecodeFile({}, compressed, &io2, pool));
    EXPECT_TRUE(io2.Main().IsGray());

    EXPECT_LE(compressed.size(), 7000u);
    EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                    /*distmap=*/nullptr, pool),
                IsSlightlyBelow(1.6));
  }

  // Test with larger butteraugli distance and other settings enabled so
  // different jxl codepaths trigger.
  {
    CompressParams cparams;
    cparams.butteraugli_distance = 8.0;

    PaddedBytes compressed;
    EXPECT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, GetJxlCms(),
                           aux_out, pool));
    CodecInOut io2;
    EXPECT_TRUE(test::DecodeFile({}, compressed, &io2, pool));
    EXPECT_TRUE(io2.Main().IsGray());

    EXPECT_LE(compressed.size(), 1300u);
    EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                    /*distmap=*/nullptr, pool),
                IsSlightlyBelow(6.0));
  }

  {
    CompressParams cparams;
    cparams.butteraugli_distance = 1.0;

    PaddedBytes compressed;
    EXPECT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, GetJxlCms(),
                           aux_out, pool));

    CodecInOut io2;
    JXLDecompressParams dparams;
    dparams.color_space = "RGB_D65_SRG_Rel_SRG";
    EXPECT_TRUE(test::DecodeFile(dparams, compressed, &io2, pool));
    EXPECT_FALSE(io2.Main().IsGray());

    EXPECT_LE(compressed.size(), 7000u);
    EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                    /*distmap=*/nullptr, pool),
                IsSlightlyBelow(1.6));
  }
}

TEST(JxlTest, RoundtripAlpha) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));

  ASSERT_NE(io.xsize(), 0u);
  ASSERT_TRUE(io.metadata.m.HasAlpha());
  ASSERT_TRUE(io.Main().HasAlpha());
  io.ShrinkTo(300, 300);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;

  EXPECT_EQ(8u, io.metadata.m.bit_depth.bits_per_sample);
  EXPECT_FALSE(io.metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(0u, io.metadata.m.bit_depth.exponent_bits_per_sample);
  EXPECT_TRUE(io.metadata.m.color_encoding.tf.IsSRGB());
  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, GetJxlCms(),
                         aux_out, pool));

  for (bool use_image_callback : {false, true}) {
    for (bool unpremul_alpha : {false, true}) {
      CodecInOut io2;
      JXLDecompressParams dparams;
      dparams.use_image_callback = use_image_callback;
      dparams.unpremultiply_alpha = unpremul_alpha;
      EXPECT_TRUE(test::DecodeFile(dparams, compressed, &io2, pool));

      EXPECT_LE(compressed.size(), 10077u);

      EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                      /*distmap=*/nullptr, pool),
                  IsSlightlyBelow(1.2));
    }
  }
}

TEST(JxlTest, RoundtripAlphaPremultiplied) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  CodecInOut io, io_nopremul;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io_nopremul, pool));

  ASSERT_NE(io.xsize(), 0u);
  ASSERT_TRUE(io.metadata.m.HasAlpha());
  ASSERT_TRUE(io.Main().HasAlpha());
  io.ShrinkTo(300, 300);
  io_nopremul.ShrinkTo(300, 300);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;

  EXPECT_FALSE(io.Main().AlphaIsPremultiplied());
  EXPECT_TRUE(io.PremultiplyAlpha());
  EXPECT_TRUE(io.Main().AlphaIsPremultiplied());

  EXPECT_FALSE(io_nopremul.Main().AlphaIsPremultiplied());

  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, GetJxlCms(),
                         aux_out, pool));

  for (bool use_image_callback : {false, true}) {
    for (bool unpremul_alpha : {false, true}) {
      for (bool use_uint8 : {false, true}) {
        printf(
            "Testing premultiplied alpha using %s %s requesting "
            "%spremultiplied output.\n",
            use_uint8 ? "uint8" : "float",
            use_image_callback ? "image callback" : "image_buffer",
            unpremul_alpha ? "un" : "");
        CodecInOut io2;
        JXLDecompressParams dparams;
        dparams.use_image_callback = use_image_callback;
        dparams.unpremultiply_alpha = unpremul_alpha;
        if (use_uint8) {
          dparams.accepted_formats = {
              {4, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0}};
        }
        EXPECT_TRUE(test::DecodeFile(dparams, compressed, &io2, pool));

        EXPECT_LE(compressed.size(), 10000u);
        EXPECT_EQ(unpremul_alpha, !io2.Main().AlphaIsPremultiplied());
        if (!unpremul_alpha) {
          EXPECT_THAT(
              ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, pool),
              IsSlightlyBelow(1.25));
          EXPECT_TRUE(io2.UnpremultiplyAlpha());
          EXPECT_FALSE(io2.Main().AlphaIsPremultiplied());
        }
        EXPECT_THAT(ButteraugliDistance(io_nopremul, io2, cparams.ba_params,
                                        GetJxlCms(),
                                        /*distmap=*/nullptr, pool),
                    IsSlightlyBelow(1.35));
      }
    }
  }
}

TEST(JxlTest, RoundtripAlphaResampling) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();
  ASSERT_NE(t.ppf().info.xsize, 0);
  ASSERT_TRUE(t.ppf().info.alpha_bits > 0);

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 5);  // kHare
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESAMPLING, 2);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING, 2);

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 12268, 30);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(4.7));
}

TEST(JxlTest, RoundtripAlphaResamplingOnlyAlpha) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();
  ASSERT_NE(t.ppf().info.xsize, 0);
  ASSERT_TRUE(t.ppf().info.alpha_bits > 0);

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 3);  // kFalcon
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EXTRA_CHANNEL_RESAMPLING, 2);

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, pool, &ppf_out), 30278, 20);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(1.85));
}

TEST(JxlTest, RoundtripAlphaNonMultipleOf8) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata().SetDimensions(12, 12);
  ASSERT_NE(t.ppf().info.xsize, 0);
  ASSERT_TRUE(t.ppf().info.alpha_bits > 0);
  EXPECT_EQ(t.ppf().frames[0].color.format.data_type, JXL_TYPE_UINT8);

  PackedPixelFile ppf_out;
  EXPECT_NEAR(Roundtrip(t.ppf(), {}, {}, pool, &ppf_out), 107, 10);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(0.9));
}

TEST(JxlTest, RoundtripAlpha16) {
  ThreadPoolInternal pool(4);
  // The image is wider than 512 pixels to ensure multiple groups are tested.
  size_t xsize = 1200, ysize = 160;
  TestImage t;
  t.SetDimensions(xsize, ysize).SetChannels(4).SetAllBitDepths(16);
  TestImage::Frame frame = t.AddFrame();
  // Generate 16-bit pattern that uses various colors and alpha values.
  const float mul = 1.0f / 65535;
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      uint16_t r = y * 65535 / ysize;
      uint16_t g = x * 65535 / xsize;
      uint16_t b = (y + x) * 65535 / (xsize + ysize);
      frame.SetValue(y, x, 0, r * mul);
      frame.SetValue(y, x, 1, g * mul);
      frame.SetValue(y, x, 2, b * mul);
      frame.SetValue(y, x, 3, g * mul);
    }
  }

  ASSERT_NE(t.ppf().info.xsize, 0);
  ASSERT_EQ(t.ppf().info.alpha_bits, 16);

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 6);  // kWombat
  cparams.distance = 0.5;

  PackedPixelFile ppf_out;
  // TODO(szabadka) Investigate big size difference on i686
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), 3794, 150);
  EXPECT_THAT(ButteraugliDistance(t.ppf(), ppf_out), IsSlightlyBelow(0.8));
}

namespace {
CompressParams CParamsForLossless() {
  CompressParams cparams;
  cparams.modular_mode = true;
  cparams.color_transform = jxl::ColorTransform::kNone;
  cparams.butteraugli_distance = 0.f;
  cparams.options.predictor = {Predictor::Weighted};
  return cparams;
}
JXLCompressParams CompressParamsForLossless() {
  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_COLOR_TRANSFORM, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 6);  // Weighted
  cparams.distance = 0;
  return cparams;
}
}  // namespace

TEST(JxlTest, JXL_SLOW_TEST(RoundtripLossless8)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams = CompressParamsForLossless();

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), 222167);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
}

TEST(JxlTest, JXL_SLOW_TEST(RoundtripLossless8ThunderGradient)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams = CompressParamsForLossless();
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 2);             // kThunder
  cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 5);  // Gradient

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), 261684);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
}

TEST(JxlTest, JXL_SLOW_TEST(RoundtripLossless8LightningGradient)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams = CompressParamsForLossless();
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 1);             // kLightning
  cparams.AddOption(JXL_ENC_FRAME_SETTING_MODULAR_PREDICTOR, 5);  // Gradient

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), 265418);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
}

TEST(JxlTest, JXL_SLOW_TEST(RoundtripLossless8Falcon)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_srgb8.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();

  JXLCompressParams cparams = CompressParamsForLossless();
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 3);  // kFalcon

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), cparams, {}, &pool, &ppf_out), 230766);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
}

TEST(JxlTest, RoundtripLossless8Alpha) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/tmshre_riaphotographs_alpha.png");
  TestImage t;
  t.DecodeFromBytes(orig).ClearMetadata();
  ASSERT_EQ(t.ppf().info.alpha_bits, 8);
  EXPECT_EQ(t.ppf().frames[0].color.format.data_type, JXL_TYPE_UINT8);

  JXLCompressParams cparams = CompressParamsForLossless();

  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  EXPECT_EQ(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 248817);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
  EXPECT_EQ(ppf_out.info.alpha_bits, 8);
  EXPECT_TRUE(test::SameAlpha(t.ppf(), ppf_out));
}

TEST(JxlTest, RoundtripLossless16Alpha) {
  ThreadPool* pool = nullptr;
  size_t xsize = 1200, ysize = 160;
  TestImage t;
  t.SetDimensions(xsize, ysize).SetChannels(4).SetAllBitDepths(16);
  TestImage::Frame frame = t.AddFrame();
  // Generate 16-bit pattern that uses various colors and alpha values.
  const float mul = 1.0f / 65535;
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      uint16_t r = y * 65535 / ysize;
      uint16_t g = x * 65535 / xsize + 37;
      uint16_t b = (y + x) * 65535 / (xsize + ysize);
      frame.SetValue(y, x, 0, r * mul);
      frame.SetValue(y, x, 1, g * mul);
      frame.SetValue(y, x, 2, b * mul);
      frame.SetValue(y, x, 3, g * mul);
    }
  }
  ASSERT_EQ(t.ppf().info.bits_per_sample, 16);
  ASSERT_EQ(t.ppf().info.alpha_bits, 16);

  JXLCompressParams cparams = CompressParamsForLossless();

  JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(t.ppf().frames[0].color.format);

  PackedPixelFile ppf_out;
  // TODO(szabadka) Investigate big size difference on i686
  EXPECT_NEAR(Roundtrip(t.ppf(), cparams, dparams, pool, &ppf_out), 4849, 100);
  EXPECT_EQ(ComputeDistance2(t.ppf(), ppf_out), 0.0);
  EXPECT_EQ(ppf_out.info.alpha_bits, 16);
  EXPECT_TRUE(test::SameAlpha(t.ppf(), ppf_out));
}

TEST(JxlTest, RoundtripLossless16AlphaNotMisdetectedAs8Bit) {
  ThreadPool* pool = nullptr;

  size_t xsize = 128, ysize = 128;
  Image3F color(xsize, ysize);
  ImageF alpha(xsize, ysize);
  // All 16-bit values, both color and alpha, of this image are below 64.
  // This allows testing if a code path wrongly concludes it's an 8-bit instead
  // of 16-bit image (or even 6-bit).
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      color.PlaneRow(0, y)[x] = (y * 64 / ysize) * (1.0f / 65535);
      color.PlaneRow(1, y)[x] = (x * 64 / xsize) * (1.0f / 65535);
      color.PlaneRow(2, y)[x] =
          ((y + x) * 64 / (xsize + ysize)) * (1.0f / 65535);
      alpha.Row(y)[x] = (64 * x / xsize) * (1.0f / 65535);
    }
  }
  const bool is_gray = false;
  CodecInOut io;
  io.metadata.m.SetUintSamples(16);
  io.metadata.m.SetAlphaBits(16);
  io.metadata.m.color_encoding = ColorEncoding::SRGB(is_gray);
  io.SetFromImage(std::move(color), io.metadata.m.color_encoding);
  io.Main().SetAlpha(std::move(alpha), /*alpha_is_premultiplied=*/false);

  EXPECT_EQ(16u, io.metadata.m.GetAlphaBits());
  EXPECT_EQ(16u, io.metadata.m.bit_depth.bits_per_sample);
  EXPECT_FALSE(io.metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(0u, io.metadata.m.bit_depth.exponent_bits_per_sample);

  CompressParams cparams = CParamsForLossless();
  JXLDecompressParams dparams;

  CodecInOut io2;
  dparams.accepted_formats = {{4, JXL_TYPE_UINT16, JXL_LITTLE_ENDIAN, 0}};
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 3100u);
  EXPECT_EQ(16u, io2.metadata.m.GetAlphaBits());
  EXPECT_EQ(16u, io2.metadata.m.bit_depth.bits_per_sample);
  EXPECT_FALSE(io2.metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(0u, io2.metadata.m.bit_depth.exponent_bits_per_sample);
  // If fails, see note about floating point in RoundtripLossless8.
  EXPECT_EQ(ComputeDistance2(io.Main(), io2.Main(), GetJxlCms()), 0.0);
  EXPECT_TRUE(SamePixels(*io.Main().alpha(), *io2.Main().alpha()));
}

TEST(JxlTest, RoundtripYCbCr420) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  const PaddedBytes yuv420 = ReadTestData("jxl/flower/flower.png.ffmpeg.y4m");
  CodecInOut io2;
  ASSERT_TRUE(test::DecodeImageY4M(Span<const uint8_t>(yuv420), &io2));

  CompressParams cparams = CParamsForLossless();
  cparams.speed_tier = SpeedTier::kThunder;

  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(EncodeFile(cparams, &io2, &enc_state, &compressed, GetJxlCms(),
                         aux_out, pool));
  CodecInOut io3;
  EXPECT_TRUE(test::DecodeFile({}, compressed, &io3, pool));

  EXPECT_LE(compressed.size(), 2000000u);

  // we're comparing an original PNG with a YCbCr 4:2:0 version
  EXPECT_THAT(ComputeDistance2(io.Main(), io3.Main(), GetJxlCms()),
              IsSlightlyBelow(4.3));
}

TEST(JxlTest, RoundtripDots) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/cvo9xd_keong_macan_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));

  ASSERT_NE(io.xsize(), 0u);

  CompressParams cparams;
  cparams.dots = Override::kOn;
  cparams.butteraugli_distance = 0.04;
  cparams.speed_tier = SpeedTier::kSquirrel;

  EXPECT_EQ(8u, io.metadata.m.bit_depth.bits_per_sample);
  EXPECT_EQ(0u, io.metadata.m.bit_depth.exponent_bits_per_sample);
  EXPECT_FALSE(io.metadata.m.bit_depth.floating_point_sample);
  EXPECT_TRUE(io.metadata.m.color_encoding.tf.IsSRGB());
  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, GetJxlCms(),
                         aux_out, pool));
  CodecInOut io2;
  EXPECT_TRUE(test::DecodeFile({}, compressed, &io2, pool));

  EXPECT_LE(compressed.size(), 400000u);
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, pool),
              IsSlightlyBelow(0.3));
}

TEST(JxlTest, RoundtripNoise) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("external/wesaturate/500px/cvo9xd_keong_macan_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));

  ASSERT_NE(io.xsize(), 0u);

  CompressParams cparams;
  cparams.noise = Override::kOn;
  cparams.speed_tier = SpeedTier::kSquirrel;

  EXPECT_EQ(8u, io.metadata.m.bit_depth.bits_per_sample);
  EXPECT_EQ(0u, io.metadata.m.bit_depth.exponent_bits_per_sample);
  EXPECT_FALSE(io.metadata.m.bit_depth.floating_point_sample);
  EXPECT_TRUE(io.metadata.m.color_encoding.tf.IsSRGB());
  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, GetJxlCms(),
                         aux_out, pool));
  CodecInOut io2;
  EXPECT_TRUE(test::DecodeFile({}, compressed, &io2, pool));

  EXPECT_LE(compressed.size(), 40000u);
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, pool),
              IsSlightlyBelow(1.6));
}

TEST(JxlTest, RoundtripLossless8Gray) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig = ReadTestData(
      "external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));

  CompressParams cparams = CParamsForLossless();
  JXLDecompressParams dparams;

  EXPECT_TRUE(io.Main().IsGray());
  EXPECT_EQ(8u, io.metadata.m.bit_depth.bits_per_sample);
  EXPECT_FALSE(io.metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(0u, io.metadata.m.bit_depth.exponent_bits_per_sample);
  CodecInOut io2;
  dparams.accepted_formats = {{1, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0}};
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 130000u);
  // If fails, see note about floating point in RoundtripLossless8.
  EXPECT_EQ(ComputeDistance2(io.Main(), io2.Main(), GetJxlCms()), 0);
  EXPECT_EQ(0.0, ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                     /*distmap=*/nullptr, pool));
  EXPECT_TRUE(io2.Main().IsGray());
  EXPECT_EQ(8u, io2.metadata.m.bit_depth.bits_per_sample);
  EXPECT_FALSE(io2.metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(0u, io2.metadata.m.bit_depth.exponent_bits_per_sample);
}

#if JPEGXL_ENABLE_GIF

TEST(JxlTest, RoundtripAnimation) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig = ReadTestData("jxl/traffic_light.gif");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  ASSERT_EQ(4u, io.frames.size());

  CompressParams cparams;
  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, {}, pool, &io2), 3000u);

  EXPECT_EQ(io2.frames.size(), io.frames.size());
  test::CoalesceGIFAnimationWithAlpha(&io);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                /*distmap=*/nullptr, pool),
#if JXL_HIGH_PRECISION
            1.55);
#else
            1.75);
#endif
}

TEST(JxlTest, RoundtripLosslessAnimation) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig = ReadTestData("jxl/traffic_light.gif");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  ASSERT_EQ(4u, io.frames.size());

  CompressParams cparams = CParamsForLossless();
  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, {}, pool, &io2), 1200u);

  EXPECT_EQ(io2.frames.size(), io.frames.size());
  test::CoalesceGIFAnimationWithAlpha(&io);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                /*distmap=*/nullptr, pool),
            5e-4);
}

TEST(JxlTest, RoundtripAnimationPatches) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig = ReadTestData("jxl/animation_patches.gif");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));
  ASSERT_EQ(2u, io.frames.size());

  CompressParams cparams;
  cparams.patches = Override::kOn;
  CodecInOut io2;
  // 40k with no patches, 27k with patch frames encoded multiple times.
  EXPECT_LE(Roundtrip(&io, cparams, {}, pool, &io2), 24000u);

  EXPECT_EQ(io2.frames.size(), io.frames.size());
  // >10 with broken patches
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, pool),
              IsSlightlyBelow(1.2));
}

#endif  // JPEGXL_ENABLE_GIF

size_t RoundtripJpeg(const PaddedBytes& jpeg_in, ThreadPool* pool) {
  CodecInOut io;
  EXPECT_TRUE(jpeg::DecodeImageJPG(Span<const uint8_t>(jpeg_in), &io));
  CompressParams cparams;
  cparams.color_transform = jxl::ColorTransform::kYCbCr;

  PassesEncoderState passes_enc_state;
  PaddedBytes compressed, codestream;

  EXPECT_TRUE(EncodeFile(cparams, &io, &passes_enc_state, &codestream,
                         GetJxlCms(),
                         /*aux_out=*/nullptr, pool));
  jpegxl::tools::JpegXlContainer enc_container;
  enc_container.codestream = std::move(codestream);
  jpeg::JPEGData data_in = *io.Main().jpeg_data;
  jxl::PaddedBytes jpeg_data;
  EXPECT_TRUE(EncodeJPEGData(data_in, &jpeg_data, cparams));
  enc_container.jpeg_reconstruction = jpeg_data.data();
  enc_container.jpeg_reconstruction_size = jpeg_data.size();
  EXPECT_TRUE(EncodeJpegXlContainerOneShot(enc_container, &compressed));

  jxl::JXLDecompressParams dparams;
  dparams.runner = pool->runner();
  dparams.runner_opaque = pool->runner_opaque();
  std::vector<uint8_t> out;
  jxl::PackedPixelFile ppf;
  EXPECT_TRUE(DecodeImageJXL(compressed.data(), compressed.size(), dparams,
                             nullptr, &ppf, &out));
  EXPECT_EQ(out.size(), jpeg_in.size());
  size_t failures = 0;
  for (size_t i = 0; i < std::min(out.size(), jpeg_in.size()); i++) {
    if (out[i] != jpeg_in[i]) {
      EXPECT_EQ(out[i], jpeg_in[i])
          << "byte mismatch " << i << " " << out[i] << " != " << jpeg_in[i];
      if (++failures > 4) {
        return compressed.size();
      }
    }
  }
  return compressed.size();
}

TEST(JxlTest, JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompression444)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png.im_q85_444.jpg");
  // JPEG size is 696,659 bytes.
  EXPECT_LE(RoundtripJpeg(orig, &pool), 570000u);
}

#if JPEGXL_ENABLE_JPEG

TEST(JxlTest, JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompressionToPixels)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png.im_q85_444.jpg");
  CodecInOut io;
  ASSERT_TRUE(jpeg::DecodeImageJPG(Span<const uint8_t>(orig), &io));

  CodecInOut io2;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io2, &pool));

  CompressParams cparams;
  cparams.color_transform = jxl::ColorTransform::kYCbCr;

  CodecInOut io3;
  Roundtrip(&io, cparams, {}, &pool, &io3);

  // TODO(eustas): investigate, why SJPEG and JpegRecompression pixels are
  // different.
  EXPECT_THAT(ComputeDistance2(io2.Main(), io3.Main(), GetJxlCms()),
              IsSlightlyBelow(12));
}

TEST(JxlTest, JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompressionToPixels420)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png.im_q85_420.jpg");
  CodecInOut io;
  ASSERT_TRUE(jpeg::DecodeImageJPG(Span<const uint8_t>(orig), &io));

  CodecInOut io2;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io2, &pool));

  CompressParams cparams;
  cparams.color_transform = jxl::ColorTransform::kYCbCr;

  CodecInOut io3;
  Roundtrip(&io, cparams, {}, &pool, &io3);

  EXPECT_THAT(ComputeDistance2(io2.Main(), io3.Main(), GetJxlCms()),
              IsSlightlyBelow(11));
}

TEST(JxlTest,
     JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompressionToPixels420EarlyFlush)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png.im_q85_420.jpg");
  CodecInOut io;
  ASSERT_TRUE(jpeg::DecodeImageJPG(Span<const uint8_t>(orig), &io));

  CodecInOut io2;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io2, &pool));

  CompressParams cparams;
  cparams.color_transform = jxl::ColorTransform::kYCbCr;

  JXLDecompressParams dparams;
  dparams.max_downsampling = 8;

  CodecInOut io3;
  Roundtrip(&io, cparams, dparams, &pool, &io3);

  EXPECT_THAT(ComputeDistance2(io2.Main(), io3.Main(), GetJxlCms()),
              IsSlightlyBelow(4410));
}

TEST(JxlTest,
     JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompressionToPixels420Mul16)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower_cropped.jpg");
  CodecInOut io;
  ASSERT_TRUE(jpeg::DecodeImageJPG(Span<const uint8_t>(orig), &io));

  CodecInOut io2;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io2, &pool));

  CompressParams cparams;
  cparams.color_transform = jxl::ColorTransform::kYCbCr;

  CodecInOut io3;
  Roundtrip(&io, cparams, {}, &pool, &io3);

  EXPECT_THAT(ComputeDistance2(io2.Main(), io3.Main(), GetJxlCms()),
              IsSlightlyBelow(4));
}

TEST(JxlTest,
     JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompressionToPixels_asymmetric)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("jxl/flower/flower.png.im_q85_asymmetric.jpg");
  CodecInOut io;
  ASSERT_TRUE(jpeg::DecodeImageJPG(Span<const uint8_t>(orig), &io));

  CodecInOut io2;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io2, &pool));

  CompressParams cparams;
  cparams.color_transform = jxl::ColorTransform::kYCbCr;

  CodecInOut io3;
  Roundtrip(&io, cparams, {}, &pool, &io3);

  EXPECT_THAT(ComputeDistance2(io2.Main(), io3.Main(), GetJxlCms()),
              IsSlightlyBelow(10));
}

#endif

TEST(JxlTest, JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompressionGray)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("jxl/flower/flower.png.im_q85_gray.jpg");
  // JPEG size is 456,528 bytes.
  EXPECT_LE(RoundtripJpeg(orig, &pool), 390000u);
}

TEST(JxlTest, JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompression420)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png.im_q85_420.jpg");
  // JPEG size is 546,797 bytes.
  EXPECT_LE(RoundtripJpeg(orig, &pool), 460000u);
}

TEST(JxlTest,
     JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompression_luma_subsample)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("jxl/flower/flower.png.im_q85_luma_subsample.jpg");
  // JPEG size is 400,724 bytes.
  EXPECT_LE(RoundtripJpeg(orig, &pool), 330000u);
}

TEST(JxlTest, JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompression444_12)) {
  // 444 JPEG that has an interesting sampling-factor (1x2, 1x2, 1x2).
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("jxl/flower/flower.png.im_q85_444_1x2.jpg");
  // JPEG size is 703,874 bytes.
  EXPECT_LE(RoundtripJpeg(orig, &pool), 570000u);
}

TEST(JxlTest, JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompression422)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png.im_q85_422.jpg");
  // JPEG size is 522,057 bytes.
  EXPECT_LE(RoundtripJpeg(orig, &pool), 500000u);
}

TEST(JxlTest, JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompression440)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png.im_q85_440.jpg");
  // JPEG size is 603,623 bytes.
  EXPECT_LE(RoundtripJpeg(orig, &pool), 510000u);
}

TEST(JxlTest, JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompression_asymmetric)) {
  // 2x vertical downsample of one chroma channel, 2x horizontal downsample of
  // the other.
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("jxl/flower/flower.png.im_q85_asymmetric.jpg");
  // JPEG size is 604,601 bytes.
  EXPECT_LE(RoundtripJpeg(orig, &pool), 510000u);
}

TEST(JxlTest, JXL_TRANSCODE_JPEG_TEST(RoundtripJpegRecompression420Progr)) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig =
      ReadTestData("jxl/flower/flower.png.im_q85_420_progr.jpg");
  // JPEG size is 522,057 bytes.
  EXPECT_LE(RoundtripJpeg(orig, &pool), 460000u);
}

TEST(JxlTest, RoundtripProgressive) {
  ThreadPoolInternal pool(4);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  io.ShrinkTo(600, 1024);

  CompressParams cparams;

  cparams.butteraugli_distance = 1.0f;
  cparams.progressive_dc = 1;
  cparams.responsive = true;
  cparams.progressive_mode = true;
  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, {}, &pool, &io2), 61700u);
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, &pool),
              IsSlightlyBelow(1.17f));
}

TEST(JxlTest, RoundtripProgressiveLevel2Slow) {
  ThreadPoolInternal pool(8);
  const PaddedBytes orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  io.ShrinkTo(600, 1024);

  CompressParams cparams;

  cparams.butteraugli_distance = 1.0f;
  cparams.progressive_dc = 2;
  cparams.speed_tier = SpeedTier::kTortoise;
  cparams.responsive = true;
  cparams.progressive_mode = true;
  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, {}, &pool, &io2), 71000u);
  EXPECT_THAT(ButteraugliDistance(io, io2, cparams.ba_params, GetJxlCms(),
                                  /*distmap=*/nullptr, &pool),
              IsSlightlyBelow(1.2f));
}

}  // namespace
}  // namespace jxl
