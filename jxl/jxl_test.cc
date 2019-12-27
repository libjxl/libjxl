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

#include <stdint.h>
#include <stdio.h>

#include <array>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "jxl/aux_out.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/override.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/dec_file.h"
#include "jxl/dec_params.h"
#include "jxl/enc_butteraugli_comparator.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_file.h"
#include "jxl/enc_params.h"
#include "jxl/extras/codec.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/image_test_utils.h"
#include "jxl/modular/encoding/context_predict.h"
#include "jxl/test_utils.h"
#include "jxl/testdata_path.h"

namespace jxl {
namespace {
using test::Roundtrip;

#define JXL_TEST_NL 0  // Disabled in code

void CreateImage1x1(CodecInOut* io) {
  Image3F image(1, 1);
  ZeroFillImage(&image);
  io->metadata.bits_per_sample = 8;
  io->metadata.color_encoding = ColorManagement::SRGB();
  io->SetFromImage(std::move(image), io->metadata.color_encoding);
}

TEST(JxlTest, HeaderSize) {
  CodecInOut io;
  CreateImage1x1(&io);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.3;
  DecompressParams dparams;
  ThreadPool* pool = nullptr;

  {
    CodecInOut io2;
    AuxOut aux_out;
    Roundtrip(&io, cparams, dparams, pool, &io2, &aux_out);
    EXPECT_LE(aux_out.layers[kLayerHeader].total_bits, 51);
  }

  {
    CodecInOut io2;
    io.metadata.alpha_bits = 8;
    ImageU alpha(1, 1);
    alpha.Row(0)[0] = 1;
    io.Main().SetAlpha(std::move(alpha));
    AuxOut aux_out;
    Roundtrip(&io, cparams, dparams, pool, &io2, &aux_out);
    EXPECT_LE(aux_out.layers[kLayerHeader].total_bits, 60);
  }
}

TEST(JxlTest, RoundtripSinglePixel) {
  CodecInOut io;
  CreateImage1x1(&io);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;
  ThreadPool* pool = nullptr;
  CodecInOut io2;
  Roundtrip(&io, cparams, dparams, pool, &io2);
}

// Changing serialized signature causes Decode to fail.
#ifndef JXL_CRASH_ON_ERROR
TEST(JxlTest, RoundtripMarker) {
  CodecInOut io;
  CreateImage1x1(&io);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;
  AuxOut* aux_out = nullptr;
  ThreadPool* pool = nullptr;

  PassesEncoderState enc_state;
  for (size_t i = 0; i < 2; ++i) {
    PaddedBytes compressed;
    EXPECT_TRUE(
        EncodeFile(cparams, &io, &enc_state, &compressed, aux_out, pool));
    compressed[i] ^= 0xFF;
    CodecInOut io2;
    EXPECT_FALSE(DecodeFile(dparams, compressed, &io2, aux_out, pool));
  }
}
#endif

TEST(JxlTest, RoundtripTinyFast) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(32, 32);

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.butteraugli_distance = 4.0f;
  DecompressParams dparams;

  CodecInOut io2;
  const size_t enc_bytes = Roundtrip(&io, cparams, dparams, pool, &io2);
  printf("32x32 image size %zu bytes\n", enc_bytes);
}

TEST(JxlTest, RoundtripSmallD1) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 1000);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.5);
}

TEST(JxlTest, RoundtripUnalignedD2) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 12, io.ysize() / 7);

  CompressParams cparams;
  cparams.butteraugli_distance = 2.0;
  DecompressParams dparams;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 700);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            3.2);
}

#if JXL_TEST_NL

TEST(JxlTest, RoundtripMultiGroupNL) {
  ThreadPoolInternal pool(4);
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));
  io.ShrinkTo(600, 1024);  // partial X, full Y group

  CompressParams cparams;
  DecompressParams dparams;

  cparams.fast_mode = true;
  cparams.butteraugli_distance = 1.0f;
  CodecInOut io2;
  Roundtrip(&io, cparams, dparams, &pool, &io2);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, &pool),
            0.9f);

  cparams.butteraugli_distance = 2.0f;
  CodecInOut io3;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, &pool, &io3), 80000);
  EXPECT_LE(ButteraugliDistance(io, io3, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, &pool),
            1.5f);
}

#endif

TEST(JxlTest, RoundtripMultiGroup) {
  ThreadPoolInternal pool(4);
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));
  io.ShrinkTo(600, 1024);  // partial X, full Y group

  CompressParams cparams;
  DecompressParams dparams;

  cparams.butteraugli_distance = 1.0f;
  CodecInOut io2;
  Roundtrip(&io, cparams, dparams, &pool, &io2);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, &pool),
            1.99f);

  cparams.butteraugli_distance = 2.0f;
  CodecInOut io3;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, &pool, &io3), 20000);
  EXPECT_LE(ButteraugliDistance(io, io3, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, &pool),
            3.0f);
}

TEST(JxlTest, RoundtripLargeFast) {
  ThreadPoolInternal pool(8);
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  DecompressParams dparams;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, &pool, &io2), 265000);
}

// Checks for differing size/distance in two consecutive runs of distance 2,
// which involves additional processing including adaptive reconstruction.
// Failing this may be a sign of race conditions or invalid memory accesses.
TEST(JxlTest, RoundtripD2Consistent) {
  ThreadPoolInternal pool(8);
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
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

// Same as above, but for full image, testing multiple groups.
TEST(JxlTest, RoundtripLargeConsistent) {
  ThreadPoolInternal pool(8);
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.butteraugli_distance = 2.0;
  DecompressParams dparams;

  // Try each xsize mod kBlockDim to verify right border handling.
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

#if JXL_TEST_NL

TEST(JxlTest, RoundtripSmallNL) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 1500);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.7);
}

#endif

TEST(JxlTest, RoundtripSmallNoGaborish) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  CompressParams cparams;
  cparams.gaborish = Override::kOff;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 900);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            1.7);
}

TEST(JxlTest, RoundtripSmallPatches) {
  ThreadPool* pool = nullptr;
  CodecInOut io;
  io.metadata.color_encoding = ColorManagement::LinearSRGB();
  Image3F black_with_small_lines(256, 256);
  ZeroFillImage(&black_with_small_lines);
  // This pattern should be picked up by the patch detection heuristics.
  for (size_t y = 0; y < black_with_small_lines.ysize(); y++) {
    float* JXL_RESTRICT row = black_with_small_lines.PlaneRow(1, y);
    for (size_t x = 0; x < black_with_small_lines.xsize(); x++) {
      if (x % 4 == 0 && (y / 32) % 4 == 0) row[x] = 127.0f;
    }
  }
  io.SetFromImage(std::move(black_with_small_lines),
                  ColorManagement::LinearSRGB());

  CompressParams cparams;
  cparams.speed_tier = SpeedTier::kSquirrel;
  cparams.butteraugli_distance = 0.1f;
  DecompressParams dparams;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 2000);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            0.5f);
}

TEST(JxlTest, RoundtripGrayscale) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  ASSERT_NE(io.xsize(), 0);
  io.ShrinkTo(128, 128);
  EXPECT_TRUE(io.Main().IsGray());
  EXPECT_EQ(8, io.metadata.bits_per_sample);
  EXPECT_TRUE(io.metadata.color_encoding.tf.IsSRGB());

  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;

  {
    CompressParams cparams;
    cparams.butteraugli_distance = 1.0;
    DecompressParams dparams;

    PaddedBytes compressed;
    EXPECT_TRUE(
        EncodeFile(cparams, &io, &enc_state, &compressed, aux_out, pool));
    CodecInOut io2;
    EXPECT_TRUE(DecodeFile(dparams, compressed, &io2, aux_out, pool));
    EXPECT_TRUE(io2.Main().IsGray());

    EXPECT_LE(compressed.size(), 7000);
    EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                  /*distmap=*/nullptr, pool),
              1.7777777);
  }

  // Test with larger butteraugli distance and other settings enabled so
  // different jxl codepaths trigger.
  {
    CompressParams cparams;
    cparams.butteraugli_distance = 8.0;
    DecompressParams dparams;

    PaddedBytes compressed;
    EXPECT_TRUE(
        EncodeFile(cparams, &io, &enc_state, &compressed, aux_out, pool));
    CodecInOut io2;
    EXPECT_TRUE(DecodeFile(dparams, compressed, &io2, aux_out, pool));
    EXPECT_TRUE(io2.Main().IsGray());

    EXPECT_LE(compressed.size(), 1300);
    EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                  /*distmap=*/nullptr, pool),
              9.0);
  }
}

TEST(JxlTest, RoundtripAlpha) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/tmshre_riaphotographs_alpha.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));

  ASSERT_NE(io.xsize(), 0);
  ASSERT_TRUE(io.metadata.HasAlpha());
  ASSERT_TRUE(io.Main().HasAlpha());
  io.ShrinkTo(128, 128);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;

  EXPECT_EQ(8, io.metadata.bits_per_sample);
  EXPECT_TRUE(io.metadata.color_encoding.tf.IsSRGB());
  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, aux_out, pool));
  CodecInOut io2;
  EXPECT_TRUE(DecodeFile(dparams, compressed, &io2, aux_out, pool));

  EXPECT_LE(compressed.size(), 5500);

  // TODO(robryk): Fix the following line in presence of different alpha_bits in
  // the two contexts.
  // EXPECT_TRUE(SamePixels(io.Main().alpha(), io2.Main().alpha()));
  // TODO(robryk): Fix the distance estimate used in the encoder.
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            6.3);
}

TEST(JxlTest, RoundtripAlphaNonMultipleOf8) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/tmshre_riaphotographs_alpha.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));

  ASSERT_NE(io.xsize(), 0);
  ASSERT_TRUE(io.metadata.HasAlpha());
  ASSERT_TRUE(io.Main().HasAlpha());
  io.ShrinkTo(12, 12);

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  DecompressParams dparams;

  EXPECT_EQ(8, io.metadata.bits_per_sample);
  EXPECT_TRUE(io.metadata.color_encoding.tf.IsSRGB());
  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, aux_out, pool));
  CodecInOut io2;
  EXPECT_TRUE(DecodeFile(dparams, compressed, &io2, aux_out, pool));

  EXPECT_LE(compressed.size(), 120);

  // TODO(robryk): Fix the following line in presence of different alpha_bits in
  // the two contexts.
  // EXPECT_TRUE(SamePixels(io.Main().alpha(), io2.Main().alpha()));
  // TODO(robryk): Fix the distance estimate used in the encoder.
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            6.3);
}

TEST(JxlTest, RoundtripAlpha16) {
  ThreadPoolInternal pool(4);

  size_t xsize = 1200, ysize = 160;
  std::vector<uint16_t> pixels(xsize * ysize * 4);
  // Generate 16-bit pattern that uses various colors and alpha values.
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      size_t i = y * xsize + x;
      pixels[i * 4 + 0] = y * 65535 / ysize;
      pixels[i * 4 + 1] = x * 65535 / xsize;
      pixels[i * 4 + 2] = (y + x) * 65535 / (xsize + ysize);
      pixels[i * 4 + 3] = 65535 * x / xsize;
    }
  }
  const bool is_gray = false;
  CodecInOut io;
  io.metadata.bits_per_sample = 16;
  io.metadata.alpha_bits = 16;
  io.metadata.color_encoding = ColorManagement::SRGB(is_gray);
  ASSERT_TRUE(io.Main().SetFromSRGB(xsize, ysize, is_gray,
                                    /*has_alpha=*/true, pixels.data(),
                                    pixels.data() + pixels.size(), &pool));

  // The image is wider than 512 pixels to ensure multiple groups are tested.

  ASSERT_NE(io.xsize(), 0);
  ASSERT_TRUE(io.metadata.HasAlpha());
  ASSERT_TRUE(io.Main().HasAlpha());

  CompressParams cparams;
  cparams.butteraugli_distance = 1.0;
  // Prevent the test to be too slow, does not affect alpha
  cparams.speed_tier = SpeedTier::kSquirrel;
  DecompressParams dparams;

  EXPECT_EQ(16, io.metadata.bits_per_sample);
  EXPECT_TRUE(io.metadata.color_encoding.tf.IsSRGB());
  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(
      EncodeFile(cparams, &io, &enc_state, &compressed, aux_out, &pool));
  CodecInOut io2;
  EXPECT_TRUE(DecodeFile(dparams, compressed, &io2, aux_out, &pool));

  EXPECT_TRUE(SamePixels(io.Main().alpha(), io2.Main().alpha()));
}

namespace {
CompressParams CParamsForLossless() {
  CompressParams cparams;
  cparams.modular_group_mode = true;
  cparams.color_transform = jxl::ColorTransform::kNone;
  cparams.quality_pair = {100, 100};
  cparams.options.predictor = {int(Predictor::Weighted)};
  return cparams;
}
};  // namespace

TEST(JxlTest, RoundtripLossless8) {
  ThreadPoolInternal pool(8);
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, &pool));

  CompressParams cparams = CParamsForLossless();
  DecompressParams dparams;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, &pool, &io2), 3500000);
  // If this test fails with a very close to 0.0 but not exactly 0.0 butteraugli
  // distance, then there is likely a floating point issue, that could be
  // happening either in io or io2. The values of io are generated by
  // external_image.cc, and those in io2 by the jxl decoder. If they use
  // slightly different floating point operations (say, one casts int to float
  // while other divides the int through 255.0f and later multiplies it by
  // 255 again) they will get slightly different values. To fix, ensure both
  // sides do the following formula for converting integer range 0-255 to
  // floating point range 0.0f-255.0f: static_cast<float>(i)
  // without any further intermediate operations.
  // Note that this precision issue is not a problem in practice if the values
  // are equal when rounded to 8-bit int, but currently full exact precision is
  // tested.
  EXPECT_EQ(0.0, ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                     /*distmap=*/nullptr, &pool));
}

TEST(JxlTest, RoundtripLossless8Alpha) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/tmshre_riaphotographs_alpha.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  EXPECT_EQ(8, io.metadata.alpha_bits);
  EXPECT_EQ(8, io.metadata.bits_per_sample);

  CompressParams cparams = CParamsForLossless();
  DecompressParams dparams;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 350000);
  // If fails, see note about floating point in RoundtripLossless8.
  EXPECT_EQ(0.0, ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                     /*distmap=*/nullptr, pool));
  EXPECT_TRUE(SamePixels(io.Main().alpha(), io2.Main().alpha()));
  EXPECT_EQ(8, io2.metadata.alpha_bits);
  EXPECT_EQ(8, io2.metadata.bits_per_sample);
}

TEST(JxlTest, RoundtripLossless16Alpha) {
  ThreadPool* pool = nullptr;

  size_t xsize = 1200, ysize = 160;
  std::vector<uint16_t> pixels(xsize * ysize * 4);
  // Generate 16-bit pattern that uses various colors and alpha values.
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      size_t i = y * xsize + x;
      pixels[i * 4 + 0] = y * 65535 / ysize;
      pixels[i * 4 + 1] = x * 65535 / xsize;
      pixels[i * 4 + 2] = (y + x) * 65535 / (xsize + ysize);
      pixels[i * 4 + 3] = 65535 * y / ysize;
    }
  }
  const bool is_gray = false;
  CodecInOut io;
  io.metadata.bits_per_sample = 16;
  io.metadata.alpha_bits = 16;
  io.metadata.color_encoding = ColorManagement::SRGB(is_gray);
  ASSERT_TRUE(io.Main().SetFromSRGB(xsize, ysize, is_gray,
                                    /*has_alpha=*/true, pixels.data(),
                                    pixels.data() + pixels.size(), pool));

  EXPECT_EQ(16, io.metadata.alpha_bits);
  EXPECT_EQ(16, io.metadata.bits_per_sample);

  CompressParams cparams = CParamsForLossless();
  DecompressParams dparams;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 7000);
  // If this test fails with a very close to 0.0 but not exactly 0.0 butteraugli
  // distance, then there is likely a floating point issue, that could be
  // happening either in io or io2. The values of io are generated by
  // external_image.cc, and those in io2 by the jxl decoder. If they use
  // slightly different floating point operations (say, one does "i / 257.0f"
  // while the other does "i * (1.0f / 257)" they will get slightly different
  // values. To fix, ensure both sides do the following formula for converting
  // integer range 0-65535 to Image3F floating point range 0.0f-255.0f:
  // "i * (1.0f / 257)".
  // Note that this precision issue is not a problem in practice if the values
  // are equal when rounded to 16-bit int, but currently full exact precision is
  // tested.
  EXPECT_EQ(0.0, ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                     /*distmap=*/nullptr, pool));
  EXPECT_TRUE(SamePixels(io.Main().alpha(), io2.Main().alpha()));
  EXPECT_EQ(16, io2.metadata.alpha_bits);
  EXPECT_EQ(16, io2.metadata.bits_per_sample);
}

TEST(JxlTest, RoundtripLossless16AlphaNotMisdetectedAs8Bit) {
  ThreadPool* pool = nullptr;

  size_t xsize = 128, ysize = 128;
  std::vector<uint16_t> pixels(xsize * ysize * 4);
  // All 16-bit values, both color and alpha, of this image are below 64.
  // This allows testing if a code path wrongly concludes it's an 8-bit instead
  // of 16-bit image (or even 6-bit).
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      size_t i = y * xsize + x;
      pixels[i * 4 + 0] = y * 64 / ysize;
      pixels[i * 4 + 1] = x * 64 / xsize;
      pixels[i * 4 + 2] = (y + x) * 64 / (xsize + ysize);
      pixels[i * 4 + 3] = 64 * y / ysize;
    }
  }
  const bool is_gray = false;
  CodecInOut io;
  io.metadata.bits_per_sample = 16;
  io.metadata.alpha_bits = 16;
  io.metadata.color_encoding = ColorManagement::SRGB(is_gray);
  ASSERT_TRUE(io.Main().SetFromSRGB(xsize, ysize, /*is_gray=*/false,
                                    /*has_alpha=*/true, pixels.data(),
                                    pixels.data() + pixels.size(), pool));

  EXPECT_EQ(16, io.metadata.alpha_bits);
  EXPECT_EQ(16, io.metadata.bits_per_sample);

  CompressParams cparams = CParamsForLossless();
  DecompressParams dparams;

  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 2000);
  EXPECT_EQ(16, io2.metadata.alpha_bits);
  EXPECT_EQ(16, io2.metadata.bits_per_sample);
  // If fails, see note about floating point in RoundtripLossless8.
  EXPECT_EQ(0.0, ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                     /*distmap=*/nullptr, pool));
  EXPECT_TRUE(SamePixels(io.Main().alpha(), io2.Main().alpha()));
}

TEST(JxlTest, RoundtripDots) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/cvo9xd_keong_macan_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));

  ASSERT_NE(io.xsize(), 0);

  CompressParams cparams;
  cparams.dots = Override::kOn;
  cparams.butteraugli_distance = 0.04;
  cparams.speed_tier = SpeedTier::kSquirrel;
  DecompressParams dparams;

  EXPECT_EQ(8, io.metadata.bits_per_sample);
  EXPECT_TRUE(io.metadata.color_encoding.tf.IsSRGB());
  PassesEncoderState enc_state;
  AuxOut* aux_out = nullptr;
  PaddedBytes compressed;
  EXPECT_TRUE(EncodeFile(cparams, &io, &enc_state, &compressed, aux_out, pool));
  CodecInOut io2;
  EXPECT_TRUE(DecodeFile(dparams, compressed, &io2, aux_out, pool));

  EXPECT_LE(compressed.size(), 400000);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            2.2);
}

TEST(JxlTest, RoundtripLossless8Gray) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/cvo9xd_keong_macan_grayscale.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));

  CompressParams cparams = CParamsForLossless();
  DecompressParams dparams;

  EXPECT_TRUE(io.Main().IsGray());
  EXPECT_EQ(8, io.metadata.bits_per_sample);
  CodecInOut io2;
  EXPECT_LE(Roundtrip(&io, cparams, dparams, pool, &io2), 130000);
  // If fails, see note about floating point in RoundtripLossless8.
  EXPECT_EQ(0.0, ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                     /*distmap=*/nullptr, pool));
  EXPECT_TRUE(io2.Main().IsGray());
  EXPECT_EQ(8, io2.metadata.bits_per_sample);
}

}  // namespace
}  // namespace jxl
