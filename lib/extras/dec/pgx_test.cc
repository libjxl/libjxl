// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/pgx.h"

#include <cstdint>
#include <cstring>
#include <string>

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

}  // namespace
}  // namespace extras
}  // namespace jxl
