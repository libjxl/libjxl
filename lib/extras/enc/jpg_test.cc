// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/enc/jpg.h"

#include <jxl/codestream_header.h>
#include <jxl/types.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include "lib/extras/enc/encode.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace extras {
namespace {

PackedPixelFile MakeTestImage(size_t xsize, size_t ysize,
                              uint32_t num_color_channels) {
  PackedPixelFile ppf;
  ppf.info.xsize = xsize;
  ppf.info.ysize = ysize;
  ppf.info.bits_per_sample = 8;
  ppf.info.exponent_bits_per_sample = 0;
  ppf.info.num_color_channels = num_color_channels;
  ppf.info.alpha_bits = 0;
  JxlPixelFormat format = {num_color_channels, JXL_TYPE_UINT8,
                           JXL_BIG_ENDIAN, /*align=*/0};
  JXL_TEST_ASSIGN_OR_DIE(PackedFrame frame,
                         PackedFrame::Create(xsize, ysize, format));
  memset(frame.color.pixels(), 0x80, frame.color.pixels_size);
  ppf.frames.emplace_back(std::move(frame));
  return ppf;
}

// Control cell: a regular small image must still encode to a JPEG bitstream
// (SOI marker), identically with and without the fix.
TEST(JPEGEncoderTest, EncodesSmallImage) {
  if (!GetJPEGEncoder()) GTEST_SKIP() << "JPEG encoder not available";
  PackedPixelFile ppf = MakeTestImage(8, 8, /*num_color_channels=*/3);
  auto encoder = GetJPEGEncoder();
  EncodedImage encoded;
  ASSERT_TRUE(encoder->Encode(ppf, &encoded, nullptr));
  ASSERT_EQ(encoded.bitstreams.size(), 1u);
  const std::vector<uint8_t>& out = encoded.bitstreams[0];
  ASSERT_GE(out.size(), 2u);
  EXPECT_EQ(out[0], 0xFF);
  EXPECT_EQ(out[1], 0xD8);
}

// JPEG cannot represent images wider or taller than JPEG_MAX_DIMENSION
// (65500 px). The encoder must fail with an error return; before the fix,
// libjpeg's default fatal error handler terminated the whole process
// (exit(EXIT_FAILURE)) inside jpeg_start_compress.
TEST(JPEGEncoderTest, OversizedImageFailsCleanly) {
  if (!GetJPEGEncoder()) GTEST_SKIP() << "JPEG encoder not available";
  PackedPixelFile ppf = MakeTestImage(65536, 2, /*num_color_channels=*/1);
  auto encoder = GetJPEGEncoder();
  EncodedImage encoded;
  EXPECT_FALSE(encoder->Encode(ppf, &encoded, nullptr));
}

// An Exif payload that does not fit in a single JPEG APP1 marker (65533
// bytes of data) must be rejected; before the fix, jpeg_write_m_header()
// aborted the process through the same fatal error handler.
TEST(JPEGEncoderTest, OversizedExifFailsCleanly) {
  if (!GetJPEGEncoder()) GTEST_SKIP() << "JPEG encoder not available";
  PackedPixelFile ppf = MakeTestImage(8, 8, /*num_color_channels=*/3);
  ppf.metadata.exif.assign(70000, 0x2A);
  auto encoder = GetJPEGEncoder();
  EncodedImage encoded;
  EXPECT_FALSE(encoder->Encode(ppf, &encoded, nullptr));
}

}  // namespace
}  // namespace extras
}  // namespace jxl
