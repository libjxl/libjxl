// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/encode.h>
#include <jxl/memory_manager.h>
#include <jxl/types.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <string>
#include <utility>
#include <vector>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/dec/jxl.h"
#include "lib/extras/enc/jxl.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {

using ::jxl::extras::ColorHints;
using ::jxl::extras::DecodeImageJXL;
using ::jxl::extras::EncodeImageJXL;
using ::jxl::extras::JXLCompressParams;
using ::jxl::extras::JXLDecompressParams;
using ::jxl::extras::PackedFrame;
using ::jxl::extras::PackedImage;
using ::jxl::extras::PackedPixelFile;
using ::jxl::test::ButteraugliDistance;
using ::jxl::test::DefaultAcceptedFormats;
using ::jxl::test::ReadTestData;
using ::jxl::test::Roundtrip;
using ::jxl::test::SetThreadParallelRunner;
using ::jxl::test::ThreadPoolForTests;

namespace {

PackedPixelFile LoadTestImage(const std::string& path) {
  const std::vector<uint8_t> orig = ReadTestData(path);
  PackedPixelFile ppf;
  Check(DecodeBytes(Bytes(orig), ColorHints(), &ppf));
  return ppf;
}

void ShrinkImage(PackedPixelFile& ppf, size_t new_xsize, size_t new_ysize) {
  Check(ppf.ShrinkTo(new_xsize, new_ysize));
  std::vector<PackedFrame> frames;
  frames.reserve(ppf.frames.size());
  // TODO(eustas): remove when not necessary.
  // ButteraugliDistance does a conversion that checks that physical size
  // is the same as logical; thus we need reallocation here.
  for (const auto& frame : ppf.frames) {
    JXL_TEST_ASSIGN_OR_DIE(PackedFrame pf, frame.Copy());
    frames.emplace_back(std::move(pf));
  }
  ppf.frames.swap(frames);
}

TEST(PassesTest, RoundtripSmallPasses) {
  ThreadPool* pool = nullptr;
  PackedPixelFile ppf =
      LoadTestImage("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  ShrinkImage(ppf, ppf.xsize() / 8, ppf.ysize() / 8);

  JXLCompressParams cparams;
  cparams.distance = 1.0;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);

  PackedPixelFile ppf2;
  size_t compressed_size = Roundtrip(ppf, cparams, {}, pool, &ppf2);
  EXPECT_LE(compressed_size, 5600u);
  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(ppf, ppf2, pool), 1.0);
}

TEST(PassesTest, RoundtripUnalignedPasses) {
  ThreadPool* pool = nullptr;
  PackedPixelFile ppf =
      LoadTestImage("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  ShrinkImage(ppf, ppf.xsize() / 12, ppf.ysize() / 7);

  JXLCompressParams cparams;
  cparams.distance = 2.0;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);

  PackedPixelFile ppf2;
  size_t compressed_size = Roundtrip(ppf, cparams, {}, pool, &ppf2);
  EXPECT_LE(compressed_size, 5200u);

  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(ppf, ppf2, pool), 1.72);
}

TEST(PassesTest, RoundtripMultiGroupPasses) {
  PackedPixelFile ppf = LoadTestImage("jxl/flower/flower.png");
  ShrinkImage(ppf, 600, 1024);  // partial X, full Y group

  auto test = [&](float target_distance, float threshold,
                  size_t expected_size) {
    ThreadPoolForTests pool(4);
    JXLCompressParams cparams;
    cparams.distance = target_distance;
    cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);

    PackedPixelFile ppf2;
    size_t compressed_size = Roundtrip(ppf, cparams, {}, pool.get(), &ppf2);
    EXPECT_SLIGHTLY_BELOW(compressed_size, expected_size);
    EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(ppf, ppf2, pool.get()),
                          target_distance + threshold);
  };

  auto run1 = std::async(std::launch::async, test, 1.0f, 0.25f, 76350);
  auto run2 = std::async(std::launch::async, test, 2.05f, 0.0f, 50900);
  (void)run1;
  (void)run2;
}

TEST(PassesTest, RoundtripLargeFastPasses) {
  ThreadPoolForTests pool(8);
  PackedPixelFile ppf = LoadTestImage("jxl/flower/flower.png");

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);

  PackedPixelFile ppf2;
  size_t compressed_size = Roundtrip(ppf, cparams, {}, pool.get(), &ppf2);
  EXPECT_SLIGHTLY_BELOW(compressed_size, 497250);
}

// Checks for differing size/distance in two consecutive runs of distance 2,
// which involves additional processing including adaptive reconstruction.
// Failing this may be a sign of race conditions or invalid memory accesses.
TEST(PassesTest, RoundtripProgressiveConsistent) {
  ThreadPoolForTests pool(8);
  PackedPixelFile ppf = LoadTestImage("jxl/flower/flower.png");

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);
  cparams.distance = 2.0;

  // Try each xsize mod kBlockDim to verify right border handling.
  for (size_t xsize = 48; xsize > 40; --xsize) {
    ShrinkImage(ppf, xsize, 15);

    PackedPixelFile ppf2;

    size_t size2 = Roundtrip(ppf, cparams, {}, pool.get(), &ppf2);

    PackedPixelFile ppf3;
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
  PackedPixelFile ppf =
      LoadTestImage("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");

  std::vector<uint8_t> compressed;

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);
  cparams.distance = 1.0;
  SetThreadParallelRunner(cparams, pool.get());

  ASSERT_TRUE(
      EncodeImageJXL(cparams, ppf, /*jpeg_bytes=*/nullptr, &compressed));

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
    JXLDecompressParams dparams;
    DefaultAcceptedFormats(dparams);
    dparams.max_downsampling = downsampling;

    PackedPixelFile output;

    JXL_RETURN_IF_ERROR(DecodeImageJXL(compressed.data(), compressed.size(),
                                       dparams, /*decoded_bytes=*/nullptr,
                                       &output));
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
  PackedPixelFile ppf =
      LoadTestImage("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");

  std::vector<uint8_t> compressed;

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC, 1);
  cparams.distance = 1.0;
  SetThreadParallelRunner(cparams, pool.get());
  ASSERT_TRUE(
      EncodeImageJXL(cparams, ppf, /*jpeg_bytes=*/nullptr, &compressed));

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
    JXLDecompressParams dparams;
    dparams.max_downsampling = downsampling;

    PackedPixelFile output;
    DefaultAcceptedFormats(dparams);

    JXL_RETURN_IF_ERROR(DecodeImageJXL(compressed.data(), compressed.size(),
                                       dparams, /*decoded_bytes=*/nullptr,
                                       &output));
    EXPECT_EQ(output.xsize(), ppf.xsize()) << "downsampling = " << downsampling;
    EXPECT_EQ(output.ysize(), ppf.ysize()) << "downsampling = " << downsampling;
    EXPECT_LE(ButteraugliDistance(ppf, output, /*pool=*/nullptr),
              target_butteraugli[downsampling])
        << "downsampling: " << downsampling;
    return true;
  };
  EXPECT_TRUE(RunOnPool(pool.get(), 0, downsamplings.size(), ThreadPool::NoInit,
                        check, "TestQProgressive"));
}

TEST(PassesTest, ProgressiveDownsample2DegradesCorrectlyGrayscale) {
  ThreadPoolForTests pool(8);
  std::vector<uint8_t> compressed;
  PackedPixelFile ppf = LoadTestImage(
      "external/wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  size_t xsize_src = ppf.xsize();
  size_t ysize_src = 128;
  // need 2 DC groups for the DC frame to actually be progressive.
  size_t xsize = 4242;
  size_t ysize = ysize_src;
  const PackedImage& src = ppf.frames[0].color;
  JXL_TEST_ASSIGN_OR_DIE(PackedFrame pf,
                         PackedFrame::Create(xsize, ysize, src.format));
  PackedImage& pfc = pf.color;
  memset(pfc.pixels(), 0, pfc.pixels_size);
  size_t row_stride = pfc.pixels(0, xsize_src, 0) - pfc.pixels(0, 0, 0);
  for (size_t y = 0; y < ysize_src; ++y) {
    memcpy(pfc.pixels(y, 0, 0), src.pixels(y, 0, 0), row_stride);
  }
  ppf.frames.clear();
  ppf.frames.emplace_back(std::move(pf));
  ppf.info.xsize = xsize;
  ppf.info.ysize = ysize;

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESPONSIVE, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC, 1);
  cparams.distance = 1.0;
  SetThreadParallelRunner(cparams, pool.get());

  ASSERT_TRUE(
      EncodeImageJXL(cparams, ppf, /*jpeg_bytes=*/nullptr, &compressed));
  EXPECT_LE(compressed.size(), 18300u);

  JXLDecompressParams dparams;
  DefaultAcceptedFormats(dparams);
  SetThreadParallelRunner(dparams, pool.get());

  dparams.max_downsampling = 1;
  PackedPixelFile output;
  ASSERT_TRUE(DecodeImageJXL(compressed.data(), compressed.size(), dparams,
                             /*decoded_bytes=*/nullptr, &output));

  dparams.max_downsampling = 2;
  PackedPixelFile output_d2;
  ASSERT_TRUE(DecodeImageJXL(compressed.data(), compressed.size(), dparams,
                             /*decoded_bytes=*/nullptr, &output_d2));

  // 0 if reading all the passes, ~15 if skipping the 8x pass.
  float butteraugli_distance_down2_full =
      ButteraugliDistance(output, output_d2, pool.get());

  EXPECT_LE(butteraugli_distance_down2_full, 3.2f);
  EXPECT_GE(butteraugli_distance_down2_full, 1.0f);
}

TEST(PassesTest, ProgressiveDownsample2DegradesCorrectly) {
  ThreadPoolForTests pool(8);
  PackedPixelFile ppf = LoadTestImage("jxl/flower/flower.png");
  std::vector<uint8_t> compressed;

  size_t xsize_src = ppf.xsize();
  size_t ysize_src = 128;
  // need 2 DC groups for the DC frame to actually be progressive.
  size_t xsize = 4242;
  size_t ysize = ysize_src;
  const PackedImage& src = ppf.frames[0].color;
  JXL_TEST_ASSIGN_OR_DIE(PackedFrame pf,
                         PackedFrame::Create(xsize, ysize, src.format));
  PackedImage& pfc = pf.color;
  memset(pfc.pixels(), 0, pfc.pixels_size);
  size_t row_stride = pfc.pixels(0, xsize_src, 0) - pfc.pixels(0, 0, 0);
  for (size_t y = 0; y < ysize_src; ++y) {
    memcpy(pfc.pixels(y, 0, 0), src.pixels(y, 0, 0), row_stride);
  }
  ppf.frames.clear();
  ppf.frames.emplace_back(std::move(pf));
  ppf.info.xsize = xsize;
  ppf.info.ysize = ysize;

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_RESPONSIVE, 1);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_QPROGRESSIVE_AC, 1);
  cparams.distance = 1.0;
  SetThreadParallelRunner(cparams, pool.get());

  ASSERT_TRUE(
      EncodeImageJXL(cparams, ppf, /*jpeg_bytes=*/nullptr, &compressed));
  EXPECT_LE(compressed.size(), 55750u);

  JXLDecompressParams dparams;
  DefaultAcceptedFormats(dparams);
  SetThreadParallelRunner(dparams, pool.get());

  dparams.max_downsampling = 1;
  PackedPixelFile output;
  ASSERT_TRUE(DecodeImageJXL(compressed.data(), compressed.size(), dparams,
                             /*decoded_bytes=*/nullptr, &output));

  dparams.max_downsampling = 2;
  PackedPixelFile output_d2;
  ASSERT_TRUE(DecodeImageJXL(compressed.data(), compressed.size(), dparams,
                             /*decoded_bytes=*/nullptr, &output_d2));

  // 0 if reading all the passes, ~15 if skipping the 8x pass.
  float butteraugli_distance_down2_full =
      ButteraugliDistance(output, output_d2, pool.get());

  EXPECT_LE(butteraugli_distance_down2_full, 3.0f);
  EXPECT_GE(butteraugli_distance_down2_full, 1.0f);
}

TEST(PassesTest, NonProgressiveDCImage) {
  ThreadPoolForTests pool(8);
  PackedPixelFile ppf = LoadTestImage("jxl/flower/flower.png");

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_EFFORT, 7);
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_DC, 0);
  cparams.distance = 1.0;
  std::vector<uint8_t> compressed;

  JXLDecompressParams dparams;
  dparams.max_downsampling = 100;

  PackedPixelFile output;
  size_t decoded_size = 0;
  size_t compressed_size =
      Roundtrip(ppf, cparams, dparams, pool.get(), &output, &decoded_size);
  EXPECT_SLIGHTLY_BELOW(compressed_size, 499300);
  EXPECT_SLIGHTLY_BELOW(decoded_size, 119000);

  EXPECT_EQ(output.xsize(), ppf.xsize());
  EXPECT_EQ(output.ysize(), ppf.ysize());
}

TEST(PassesTest, RoundtripSmallNoGaborishPasses) {
  ThreadPool* pool = nullptr;
  PackedPixelFile ppf =
      LoadTestImage("external/wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  ShrinkImage(ppf, ppf.xsize() / 8, ppf.ysize() / 8);

  JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_GABORISH, 0);
  cparams.distance = 1.0;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PROGRESSIVE_AC, 1);

  PackedPixelFile ppf2;
  size_t compressed_size = Roundtrip(ppf, cparams, {}, pool, &ppf2);
  EXPECT_LE(compressed_size, 5700u);

  EXPECT_SLIGHTLY_BELOW(ButteraugliDistance(ppf, ppf2, pool), 1.0);
}

}  // namespace
}  // namespace jxl
