// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "lib/extras/codec.h"
#include "lib/extras/dec/jxl.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {

using ::jxl::test::ButteraugliDistance;
using ::jxl::test::ReadTestData;
using ::jxl::test::Roundtrip;
using ::jxl::test::ThreadPoolForTests;

namespace {

TEST(PassesTest, RoundtripSmallPasses) {
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  extras::PackedPixelFile ppf;
  ASSERT_TRUE(DecodeBytes(Bytes(orig), extras::ColorHints(), &ppf));
  ASSERT_TRUE(ppf.ShrinkTo(ppf.xsize() / 8, ppf.ysize() / 8));

  extras::JXLCompressParams cparams;
  cparams.distance = 1.0;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);

  extras::PackedPixelFile ppf2;
  size_t compressed_size = Roundtrip(ppf, cparams, {}, {}, &ppf2);
  EXPECT_LE(compressed_size, 6000);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(ppf, ppf2, /*pool=*/nullptr), 1.0);
}

TEST(PassesTest, RoundtripUnalignedPasses) {
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  extras::PackedPixelFile ppf;
  ASSERT_TRUE(DecodeBytes(Bytes(orig), extras::ColorHints(), &ppf));
  ASSERT_TRUE(ppf.ShrinkTo(ppf.xsize() / 12, ppf.ysize() / 7));

  extras::JXLCompressParams cparams;
  cparams.distance = 2.0;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);

  extras::PackedPixelFile ppf2;
  size_t compressed_size = Roundtrip(ppf, cparams, {}, {}, &ppf2);
  EXPECT_LE(compressed_size, 6000);

  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(ppf, ppf2, /*pool=*/nullptr), 1.72);
}

TEST(PassesTest, RoundtripMultiGroupPasses) {
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  extras::PackedPixelFile ppf;
  ASSERT_TRUE(DecodeBytes(Bytes(orig), extras::ColorHints(), &ppf));
  ASSERT_TRUE(ppf.ShrinkTo(600, 1024));  // partial X, full Y group

  auto test = [&](float target_distance, float threshold,
                  size_t expected_size) {
    ThreadPoolForTests pool(4);
    extras::JXLCompressParams cparams;
    cparams.distance = target_distance;
    cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);

    extras::PackedPixelFile ppf2;
    size_t compressed_size = Roundtrip(ppf, cparams, {}, {}, &ppf2);
    EXPECT_SLIGHTLY_BELOW(compressed_size, expected_size);
    EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(ppf, ppf2, /*pool=*/nullptr),
                          target_distance + threshold);
  };

  auto run1 = std::async(std::launch::async, test, 1.0f, 0.25f, 83000);
  auto run2 = std::async(std::launch::async, test, 2.0f, 0.0f, 55000);
}

TEST(PassesTest, RoundtripLargeFastPasses) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  extras::PackedPixelFile ppf;
  ASSERT_TRUE(DecodeBytes(Bytes(orig), extras::ColorHints(), &ppf));

  extras::JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);

  extras::PackedPixelFile ppf2;
  size_t compressed_size = Roundtrip(ppf, cparams, {}, {}, &ppf2);
  EXPECT_SLIGHTLY_BELOW(compressed_size, 550000);
}

// Checks for differing size/distance in two consecutive runs of distance 2,
// which involves additional processing including adaptive reconstruction.
// Failing this may be a sign of race conditions or invalid memory accesses.
TEST(PassesTest, RoundtripProgressiveConsistent) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  extras::PackedPixelFile ppf;
  ASSERT_TRUE(DecodeBytes(Bytes(orig), jxl::extras::ColorHints(), &ppf));

  extras::JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);
  cparams.distance = 2.0;

  // Try each xsize mod kBlockDim to verify right border handling.
  for (size_t xsize = 48; xsize > 40; --xsize) {
    ASSERT_TRUE(ppf.ShrinkTo(xsize, 15));

    extras::PackedPixelFile ppf2;

    size_t size2 = Roundtrip(ppf, cparams, {}, pool.get(), &ppf2);

    extras::PackedPixelFile ppf3;
    size_t size3 = Roundtrip(ppf, cparams, {}, pool.get(), &ppf3);

    // Exact same compressed size.
    EXPECT_EQ(size2, size3);

    // Exact same distance.
    const float dist2 = ButteraugliDistance(ppf, ppf2, pool.get());
    const float dist3 = ButteraugliDistance(ppf, ppf2, pool.get());
    EXPECT_EQ(dist2, dist3);
  }
}

TEST(PassesTest, AllDownsampleFeasible) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  extras::PackedPixelFile ppf;
  ASSERT_TRUE(DecodeBytes(Bytes(orig), jxl::extras::ColorHints(), &ppf));

  std::vector<uint8_t> compressed;

  extras::JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);
  cparams.distance = 1.0;

  ASSERT_TRUE(extras::EncodeImageJXL(cparams, ppf, /*jpeg_bytes=*/nullptr,
                                     &compressed));

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

  auto check = [&](const uint32_t task, size_t /* thread */) -> Status {
    const size_t downsampling = downsamplings[task];
    extras::JXLDecompressParams dparams;
    dparams.max_downsampling = downsampling;

    extras::PackedPixelFile output;
    dparams.accepted_formats = {{3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0}};

    JXL_RETURN_IF_ERROR(extras::DecodeImageJXL(
        compressed.data(), compressed.size(), dparams, nullptr, &output));
    EXPECT_EQ(output.xsize(), ppf.xsize()) << "downsampling = " << downsampling;
    EXPECT_EQ(output.ysize(), ppf.ysize()) << "downsampling = " << downsampling;
    EXPECT_LE(ButteraugliDistance(ppf, output, nullptr),
              target_butteraugli[downsampling])
        << "downsampling: " << downsampling;
    return true;
  };
  EXPECT_TRUE(RunOnPool(pool.get(), 0, downsamplings.size(), ThreadPool::NoInit,
                        check, "TestDownsampling"));
}

// TODO(firsching): Make this a parameterized test together with
// AllDownsampleFeasible
TEST(PassesTest, AllDownsampleFeasibleQProgressive) {
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  extras::PackedPixelFile ppf;
  ASSERT_TRUE(DecodeBytes(Bytes(orig), jxl::extras::ColorHints(), &ppf));

  std::vector<uint8_t> compressed;

  extras::JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC, 1);
  cparams.distance = 1.0;

  ASSERT_TRUE(extras::EncodeImageJXL(cparams, ppf, /*jpeg_bytes=*/nullptr,
                                     &compressed));

  EXPECT_LE(compressed.size(), 220000u);

  float target_butteraugli[9] = {};
  target_butteraugli[1] = 3.0f;
  target_butteraugli[2] = 6.0f;
  target_butteraugli[4] = 10.0f;
  target_butteraugli[8] = 80.0f;

  // The default progressive encoding scheme should make all these downsampling
  // factors achievable.
  std::vector<size_t> downsamplings = {1, 2, 4, 8};

  auto check = [&](const uint32_t task, size_t /* thread */) -> Status {
    const size_t downsampling = downsamplings[task];
    extras::JXLDecompressParams dparams;
    dparams.max_downsampling = downsampling;

    extras::PackedPixelFile output;
    dparams.accepted_formats = {{3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0}};

    JXL_RETURN_IF_ERROR(extras::DecodeImageJXL(
        compressed.data(), compressed.size(), dparams, nullptr, &output));
    EXPECT_EQ(output.xsize(), ppf.xsize()) << "downsampling = " << downsampling;
    EXPECT_EQ(output.ysize(), ppf.ysize()) << "downsampling = " << downsampling;
    EXPECT_LE(ButteraugliDistance(ppf, output, nullptr),
              target_butteraugli[downsampling])
        << "downsampling: " << downsampling;
    return true;
  };
  EXPECT_TRUE(RunOnPool(pool.get(), 0, downsamplings.size(), ThreadPool::NoInit,
                        check, "TestQProgressive"));
}

TEST(PassesTest, ProgressiveDownsample2DegradesCorrectlyGrayscale) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData(
      "external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  CodecInOut io_orig{memory_manager};
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io_orig, pool.get()));
  Rect rect(0, 0, io_orig.xsize(), 128);
  // need 2 DC groups for the DC frame to actually be progressive.
  JXL_TEST_ASSIGN_OR_DIE(Image3F large,
                         Image3F::Create(memory_manager, 4242, rect.ysize()));
  ZeroFillImage(&large);
  ASSERT_TRUE(CopyImageTo(rect, *io_orig.Main().color(), rect, &large));
  CodecInOut io{memory_manager};
  io.metadata = io_orig.metadata;
  ASSERT_TRUE(io.SetFromImage(std::move(large), io_orig.Main().c_current()));

  std::vector<uint8_t> compressed;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_dc = 1;
  cparams.responsive = JXL_TRUE;
  cparams.qprogressive_mode = Override::kOn;
  cparams.butteraugli_distance = 1.0;
  ASSERT_TRUE(test::EncodeFile(cparams, &io, &compressed, pool.get()));

  EXPECT_LE(compressed.size(), 10000u);

  extras::JXLDecompressParams dparams;
  dparams.max_downsampling = 1;
  CodecInOut output{memory_manager};
  ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output));

  dparams.max_downsampling = 2;
  CodecInOut output_d2{memory_manager};
  ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output_d2));

  // 0 if reading all the passes, ~15 if skipping the 8x pass.
  float butteraugli_distance_down2_full = ButteraugliDistance(
      output.frames, output_d2.frames, ButteraugliParams(), *JxlGetDefaultCms(),
      /*distmap=*/nullptr);

  EXPECT_LE(butteraugli_distance_down2_full, 3.2f);
  EXPECT_GE(butteraugli_distance_down2_full, 1.0f);
}

TEST(PassesTest, ProgressiveDownsample2DegradesCorrectly) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io_orig{memory_manager};
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io_orig, pool.get()));
  Rect rect(0, 0, io_orig.xsize(), 128);
  // need 2 DC groups for the DC frame to actually be progressive.
  JXL_TEST_ASSIGN_OR_DIE(Image3F large,
                         Image3F::Create(memory_manager, 4242, rect.ysize()));
  ZeroFillImage(&large);
  ASSERT_TRUE(CopyImageTo(rect, *io_orig.Main().color(), rect, &large));
  CodecInOut io{memory_manager};
  ASSERT_TRUE(io.SetFromImage(std::move(large), io_orig.Main().c_current()));

  std::vector<uint8_t> compressed;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_dc = 1;
  cparams.responsive = JXL_TRUE;
  cparams.qprogressive_mode = Override::kOn;
  cparams.butteraugli_distance = 1.0;
  ASSERT_TRUE(test::EncodeFile(cparams, &io, &compressed, pool.get()));

  EXPECT_LE(compressed.size(), 220000u);

  extras::JXLDecompressParams dparams;
  dparams.max_downsampling = 1;
  CodecInOut output{memory_manager};
  ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output));

  dparams.max_downsampling = 2;
  CodecInOut output_d2{memory_manager};
  ASSERT_TRUE(test::DecodeFile(dparams, Bytes(compressed), &output_d2));

  // 0 if reading all the passes, ~15 if skipping the 8x pass.
  float butteraugli_distance_down2_full = ButteraugliDistance(
      output.frames, output_d2.frames, ButteraugliParams(), *JxlGetDefaultCms(),
      /*distmap=*/nullptr);

  EXPECT_LE(butteraugli_distance_down2_full, 3.0f);
  EXPECT_GE(butteraugli_distance_down2_full, 1.0f);
}

TEST(PassesTest, NonProgressiveDCImage) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  ThreadPoolForTests pool(8);
  const std::vector<uint8_t> orig = ReadTestData("jxl/flower/flower.png");
  CodecInOut io{memory_manager};
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io, pool.get()));

  std::vector<uint8_t> compressed;

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.progressive_mode = Override::kOff;
  cparams.butteraugli_distance = 2.0;
  ASSERT_TRUE(test::EncodeFile(cparams, &io, &compressed, pool.get()));

  // Even in non-progressive mode, it should be possible to return a DC-only
  // image.
  extras::JXLDecompressParams dparams;
  dparams.max_downsampling = 100;
  CodecInOut output{memory_manager};
  ASSERT_TRUE(
      test::DecodeFile(dparams, Bytes(compressed), &output, pool.get()));
  EXPECT_EQ(output.xsize(), io.xsize());
  EXPECT_EQ(output.ysize(), io.ysize());
}

TEST(PassesTest, RoundtripSmallNoGaborishPasses) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const std::vector<uint8_t> orig =
      ReadTestData("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io{memory_manager};
  ASSERT_TRUE(SetFromBytes(Bytes(orig), &io));
  ASSERT_TRUE(io.ShrinkTo(io.xsize() / 8, io.ysize() / 8));

  CompressParams cparams;
  cparams.gaborish = Override::kOff;
  cparams.butteraugli_distance = 1.0;
  cparams.progressive_mode = Override::kOn;
  cparams.SetCms(*JxlGetDefaultCms());

  CodecInOut io2{memory_manager};
  JXL_EXPECT_OK(Roundtrip(&io, cparams, {}, &io2, _));
  EXPECT_SLIGHTLY_BELOW(
      ButteraugliDistance(io.frames, io2.frames, ButteraugliParams(),
                          *JxlGetDefaultCms(),
                          /*distmap=*/nullptr),
      1.0);
}

}  // namespace
}  // namespace jxl
