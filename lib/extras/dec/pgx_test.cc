// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/pgx.h"

#include <jxl/memory_manager.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "lib/extras/packed_image.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace extras {
namespace {

Span<const uint8_t> MakeSpan(const char* str) {
  return Bytes(reinterpret_cast<const uint8_t*>(str), strlen(str));
}

TEST(CodecPGXTest, Test8bits) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  std::string pgx = "PG ML + 8 2 3\npixels";

  PackedPixelFile ppf;
  ThreadPool* pool = nullptr;

  EXPECT_TRUE(DecodeImagePGX(MakeSpan(pgx.c_str()), ColorHints(), &ppf));
  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  EXPECT_TRUE(ConvertPackedPixelFileToCodecInOut(ppf, pool, io.get()));

  ScaleImage(255.f, io->Main().color());

  EXPECT_FALSE(io->metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(8u, io->metadata.m.bit_depth.bits_per_sample);
  EXPECT_TRUE(io->metadata.m.color_encoding.IsGray());
  EXPECT_EQ(2u, io->xsize());
  EXPECT_EQ(3u, io->ysize());

  float eps = 1e-5;
  std::array<std::array<char, 2>, 3> expected = {
      {{'p', 'i'}, {'x', 'e'}, {'l', 's'}}};
  for (size_t y = 0; y < 3; ++y) {
    for (size_t x = 0; x < 2; ++x) {
      EXPECT_NEAR(expected[y][x], io->Main().color()->Plane(0).Row(y)[x], eps);
    }
  }
}

TEST(CodecPGXTest, Test16bits) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  std::string pgx = "PG ML + 16 2 3\np_i_x_e_l_s_";

  PackedPixelFile ppf;
  ThreadPool* pool = nullptr;

  EXPECT_TRUE(DecodeImagePGX(MakeSpan(pgx.c_str()), ColorHints(), &ppf));
  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  EXPECT_TRUE(ConvertPackedPixelFileToCodecInOut(ppf, pool, io.get()));

  ScaleImage(255.f, io->Main().color());

  EXPECT_FALSE(io->metadata.m.bit_depth.floating_point_sample);
  EXPECT_EQ(16u, io->metadata.m.bit_depth.bits_per_sample);
  EXPECT_TRUE(io->metadata.m.color_encoding.IsGray());
  EXPECT_EQ(2u, io->xsize());
  EXPECT_EQ(3u, io->ysize());

  // Comparing ~16-bit numbers in floats, only ~7 bits left.
  float eps = 1e-3;
  std::array<std::array<char, 2>, 3> expected = {
      {{'p', 'i'}, {'x', 'e'}, {'l', 's'}}};
  for (size_t y = 0; y < 3; ++y) {
    for (size_t x = 0; x < 2; ++x) {
      float expected_value = 256.0f * expected[y][x] + '_';
      EXPECT_NEAR(expected_value, io->Main().color()->Plane(0).Row(y)[x] * 257,
                  eps);
    }
  }
}

// 12-bit PGX: the declared bit depth, not the 2-byte storage type, defines
// the sample range. A full-scale sample (2^12 - 1) must survive a round trip
// through the encoder as full scale.
TEST(CodecPGXTest, Test12bits) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  const std::string pgx =
      "PG ML + 12 2 1\n\x0f\xff\x0f\xff";
  PackedPixelFile ppf;
  ThreadPool* pool = nullptr;

  ASSERT_TRUE(DecodeImagePGX(Bytes(reinterpret_cast<const uint8_t*>(pgx.data()),
                                   pgx.size()),
                             ColorHints(), &ppf));
  EXPECT_EQ(12u, ppf.info.bits_per_sample);
  auto io = jxl::make_unique<CodecInOut>(memory_manager);
  ASSERT_TRUE(ConvertPackedPixelFileToCodecInOut(ppf, pool, io.get()));
  EXPECT_EQ(12u, io->metadata.m.bit_depth.bits_per_sample);

  // Full scale stays full scale; before the input bit depth was declared,
  // this came back as 1/16th of full scale.
  const float eps = 1e-5;
  for (const size_t x : {size_t{0}, size_t{1}}) {
    EXPECT_NEAR(1.0f, io->Main().color()->Plane(0).Row(0)[x], eps);
  }
}

// PGX samples must be in [0, 2^bits_per_sample - 1]; a sample above the
// declared depth would be encoded into a codestream that promises a smaller
// depth, which 16-bit-buffered decoders read as a wrapped negative value.
TEST(CodecPGXTest, RejectSamplesAboveBitDepth) {
  const auto decode = [](const std::string& header,
                         const std::vector<uint8_t>& samples) -> bool {
    std::vector<uint8_t> pgx(header.begin(), header.end());
    pgx.insert(pgx.end(), samples.begin(), samples.end());
    PackedPixelFile ppf;
    return static_cast<bool>(DecodeImagePGX(
        jxl::Bytes(pgx.data(), pgx.size()), ColorHints(), &ppf));
  };

  // Boundary samples are valid, in both byte orders.
  EXPECT_TRUE(decode("PG ML + 12 2 1\n", {0x0F, 0xFF, 0x0F, 0xFF}));
  EXPECT_TRUE(decode("PG LM + 12 2 1\n", {0xFF, 0x0F, 0xFF, 0x0F}));
  EXPECT_TRUE(decode("PG ML + 4 2 1\n", {15, 15}));
  // One value above the depth is rejected.
  EXPECT_FALSE(decode("PG ML + 12 2 1\n", {0x0F, 0xFF, 0x10, 0x00}));
  // Above-depth bits in the most significant byte, which for little-endian
  // 16-bit storage is the second byte of each sample.
  EXPECT_FALSE(decode("PG LM + 12 2 1\n", {0xFF, 0x0F, 0xFF, 0x10}));
  EXPECT_FALSE(decode("PG ML + 12 2 1\n", {0xFF, 0xFF, 0xFF, 0xFF}));
  EXPECT_FALSE(decode("PG LM + 12 2 1\n", {0xFF, 0xFF, 0xFF, 0xFF}));
  EXPECT_FALSE(decode("PG ML + 4 2 1\n", {15, 16}));
  // Check every sample position of an image longer than 8 bytes, so both the
  // 8-byte-chunk OR and the tail handling are covered.
  const std::vector<uint8_t> max4(19, 15);
  std::vector<uint8_t> max12;
  for (size_t i = 0; i < 19; ++i) max12.insert(max12.end(), {0x0F, 0xFF});
  EXPECT_TRUE(decode("PG ML + 4 19 1\n", max4));
  EXPECT_TRUE(decode("PG ML + 12 19 1\n", max12));
  for (size_t i = 0; i < 19; ++i) {
    std::vector<uint8_t> samples4 = max4;
    samples4[i] = 16;
    EXPECT_FALSE(decode("PG ML + 4 19 1\n", samples4)) << i;
    std::vector<uint8_t> samples12 = max12;
    samples12[2 * i] = 0x10;
    samples12[2 * i + 1] = 0x00;
    EXPECT_FALSE(decode("PG ML + 12 19 1\n", samples12)) << i;
  }
}

// Crafted PGX with > 19 digit dimension: without the ParseUnsigned overflow
// guard, the size_t multiplication during ASCII parsing silently wraps. The
// decoder must reject this before any size arithmetic runs.
TEST(CodecPGXTest, RejectsDimensionsThatOverflowParseUnsigned) {
  std::string pgx = "PG ML + 8 99999999999999999999999 2\np";
  PackedPixelFile ppf;
  EXPECT_FALSE(DecodeImagePGX(MakeSpan(pgx.c_str()), ColorHints(), &ppf));
}

// Crafted PGX whose individual dimensions fit in size_t but whose product
// xsize*ysize wraps. Without SafeMul on the data-size check, the data-too-
// small test can be bypassed and the decoder relies entirely on
// VerifyDimensions to catch it. With SafeMul, the parser rejects up front.
TEST(CodecPGXTest, RejectsDimensionProductOverflow) {
  // 2^33 * 2^33 = 2^66, wraps to 0 in size_t (on 64-bit). On 32-bit either
  // dimension alone already overflows ParseUnsigned and is rejected there.
  std::string pgx = "PG ML + 8 8589934592 8589934592\n";
  PackedPixelFile ppf;
  EXPECT_FALSE(DecodeImagePGX(MakeSpan(pgx.c_str()), ColorHints(), &ppf));
}

}  // namespace
}  // namespace extras
}  // namespace jxl
