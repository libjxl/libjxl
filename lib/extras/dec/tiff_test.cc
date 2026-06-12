// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/tiff.h"

#include <jxl/types.h>
#include <stdint.h>
#include <string.h>

#include <vector>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/jxl.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace extras {
namespace {

void AppendU16LE(uint16_t value, std::vector<uint8_t>* bytes) {
  bytes->push_back(static_cast<uint8_t>(value));
  bytes->push_back(static_cast<uint8_t>(value >> 8));
}

void AppendU32LE(uint32_t value, std::vector<uint8_t>* bytes) {
  bytes->push_back(static_cast<uint8_t>(value));
  bytes->push_back(static_cast<uint8_t>(value >> 8));
  bytes->push_back(static_cast<uint8_t>(value >> 16));
  bytes->push_back(static_cast<uint8_t>(value >> 24));
}

void AppendIFDEntry(uint16_t tag, uint16_t type, uint32_t count, uint32_t value,
                    std::vector<uint8_t>* bytes) {
  AppendU16LE(tag, bytes);
  AppendU16LE(type, bytes);
  AppendU32LE(count, bytes);
  if (type == 3 && count == 1) {
    AppendU16LE(static_cast<uint16_t>(value), bytes);
    AppendU16LE(0, bytes);
  } else {
    AppendU32LE(value, bytes);
  }
}

uint32_t LoadBE32(const uint8_t* p) {
  return (static_cast<uint32_t>(p[0]) << 24) |
         (static_cast<uint32_t>(p[1]) << 16) |
         (static_cast<uint32_t>(p[2]) << 8) | static_cast<uint32_t>(p[3]);
}

bool FindBox(const std::vector<uint8_t>& jxl, const char type[4],
             size_t* payload_offset, size_t* payload_size) {
  static const uint8_t kSignature[] = {0,   0,   0,    12,   'J',  'X',
                                       'L', ' ', 0x0D, 0x0A, 0x87, 0x0A};
  if (jxl.size() < sizeof(kSignature) ||
      memcmp(jxl.data(), kSignature, sizeof(kSignature)) != 0) {
    return false;
  }

  size_t pos = sizeof(kSignature);
  while (pos + 8 <= jxl.size()) {
    const uint32_t box_size = LoadBE32(jxl.data() + pos);
    if (box_size < 8 || pos + box_size > jxl.size()) return false;
    const uint8_t* box_type = jxl.data() + pos + 4;
    if (memcmp(box_type, type, 4) == 0) {
      *payload_offset = pos + 8;
      *payload_size = box_size - 8;
      return true;
    }
    pos += box_size;
  }
  return false;
}

std::vector<uint8_t> MakeTestTIFF(uint32_t width = 256, uint32_t height = 256,
                                  uint16_t orientation = 1) {
  constexpr uint16_t kEntryCount = 14;
  const char kMake[] = "libjxl";
  const char kModel[] = "tiff-metadata-preservation";
  const char kSoftware[] = "tiff_test";

  const uint32_t ifd_offset = 8;
  const uint32_t data_offset = ifd_offset + 2 + kEntryCount * 12 + 4;
  const uint32_t bits_offset = data_offset;
  const uint32_t make_offset = bits_offset + 6;
  const uint32_t model_offset = make_offset + sizeof(kMake);
  const uint32_t software_offset = model_offset + sizeof(kModel);
  const uint32_t pixel_offset = software_offset + sizeof(kSoftware);
  const uint32_t pixel_bytes = width * height * 3;

  std::vector<uint8_t> tiff;
  tiff.reserve(pixel_offset + pixel_bytes);
  tiff.push_back('I');
  tiff.push_back('I');
  AppendU16LE(42, &tiff);
  AppendU32LE(ifd_offset, &tiff);

  AppendU16LE(kEntryCount, &tiff);
  AppendIFDEntry(256, 4, 1, width, &tiff);        // ImageWidth
  AppendIFDEntry(257, 4, 1, height, &tiff);       // ImageLength
  AppendIFDEntry(258, 3, 3, bits_offset, &tiff);  // BitsPerSample
  AppendIFDEntry(259, 3, 1, 1, &tiff);            // Compression: none
  AppendIFDEntry(262, 3, 1, 2, &tiff);            // Photometric: RGB
  AppendIFDEntry(271, 2, sizeof(kMake), make_offset, &tiff);
  AppendIFDEntry(272, 2, sizeof(kModel), model_offset, &tiff);
  AppendIFDEntry(273, 4, 1, pixel_offset, &tiff);  // StripOffsets
  AppendIFDEntry(274, 3, 1, orientation, &tiff);   // Orientation
  AppendIFDEntry(277, 3, 1, 3, &tiff);             // SamplesPerPixel
  AppendIFDEntry(278, 4, 1, height, &tiff);        // RowsPerStrip
  AppendIFDEntry(279, 4, 1, pixel_bytes, &tiff);   // StripByteCounts
  AppendIFDEntry(284, 3, 1, 1, &tiff);             // PlanarConfiguration
  AppendIFDEntry(305, 2, sizeof(kSoftware), software_offset, &tiff);
  AppendU32LE(0, &tiff);

  AppendU16LE(8, &tiff);
  AppendU16LE(8, &tiff);
  AppendU16LE(8, &tiff);
  tiff.insert(tiff.end(), kMake, kMake + sizeof(kMake));
  tiff.insert(tiff.end(), kModel, kModel + sizeof(kModel));
  tiff.insert(tiff.end(), kSoftware, kSoftware + sizeof(kSoftware));
  EXPECT_EQ(pixel_offset, tiff.size());

  for (uint32_t y = 0; y < height; ++y) {
    for (uint32_t x = 0; x < width; ++x) {
      tiff.push_back(x == 0 ? 255 : 0);
      tiff.push_back(y == 0 ? 255 : 0);
      tiff.push_back(0);
    }
  }
  return tiff;
}

std::vector<uint8_t> MakeSeparateStraightAlphaTIFF() {
  constexpr uint32_t kWidth = 1;
  constexpr uint32_t kHeight = 1;
  constexpr uint16_t kEntryCount = 11;
  const uint32_t ifd_offset = 8;
  const uint32_t data_offset = ifd_offset + 2 + kEntryCount * 12 + 4;
  const uint32_t bits_offset = data_offset;
  const uint32_t strip_offsets_offset = bits_offset + 8;
  const uint32_t strip_counts_offset = strip_offsets_offset + 16;
  const uint32_t pixel_offset = strip_counts_offset + 16;

  std::vector<uint8_t> tiff;
  tiff.reserve(pixel_offset + 4);
  tiff.push_back('I');
  tiff.push_back('I');
  AppendU16LE(42, &tiff);
  AppendU32LE(ifd_offset, &tiff);

  AppendU16LE(kEntryCount, &tiff);
  AppendIFDEntry(256, 4, 1, kWidth, &tiff);       // ImageWidth
  AppendIFDEntry(257, 4, 1, kHeight, &tiff);      // ImageLength
  AppendIFDEntry(258, 3, 4, bits_offset, &tiff);  // BitsPerSample
  AppendIFDEntry(259, 3, 1, 1, &tiff);            // Compression: none
  AppendIFDEntry(262, 3, 1, 2, &tiff);            // Photometric: RGB
  AppendIFDEntry(273, 4, 4, strip_offsets_offset, &tiff);
  AppendIFDEntry(277, 3, 1, 4, &tiff);        // SamplesPerPixel
  AppendIFDEntry(278, 4, 1, kHeight, &tiff);  // RowsPerStrip
  AppendIFDEntry(279, 4, 4, strip_counts_offset, &tiff);
  AppendIFDEntry(284, 3, 1, 2, &tiff);  // PlanarConfiguration
  AppendIFDEntry(338, 3, 1, 2, &tiff);  // Unassociated alpha
  AppendU32LE(0, &tiff);

  AppendU16LE(8, &tiff);
  AppendU16LE(8, &tiff);
  AppendU16LE(8, &tiff);
  AppendU16LE(8, &tiff);
  AppendU32LE(pixel_offset, &tiff);
  AppendU32LE(pixel_offset + 1, &tiff);
  AppendU32LE(pixel_offset + 2, &tiff);
  AppendU32LE(pixel_offset + 3, &tiff);
  AppendU32LE(1, &tiff);
  AppendU32LE(1, &tiff);
  AppendU32LE(1, &tiff);
  AppendU32LE(1, &tiff);
  EXPECT_EQ(pixel_offset, tiff.size());

  tiff.push_back(200);
  tiff.push_back(100);
  tiff.push_back(50);
  tiff.push_back(128);
  return tiff;
}

std::vector<uint8_t> MakeOrientationRotateTIFF() {
  constexpr uint32_t kWidth = 2;
  constexpr uint32_t kHeight = 3;
  constexpr uint16_t kEntryCount = 11;
  const uint32_t ifd_offset = 8;
  const uint32_t data_offset = ifd_offset + 2 + kEntryCount * 12 + 4;
  const uint32_t bits_offset = data_offset;
  const uint32_t pixel_offset = bits_offset + 6;
  const uint32_t pixel_bytes = kWidth * kHeight * 3;

  std::vector<uint8_t> tiff;
  tiff.reserve(pixel_offset + pixel_bytes);
  tiff.push_back('I');
  tiff.push_back('I');
  AppendU16LE(42, &tiff);
  AppendU32LE(ifd_offset, &tiff);

  AppendU16LE(kEntryCount, &tiff);
  AppendIFDEntry(256, 4, 1, kWidth, &tiff);
  AppendIFDEntry(257, 4, 1, kHeight, &tiff);
  AppendIFDEntry(258, 3, 3, bits_offset, &tiff);
  AppendIFDEntry(259, 3, 1, 1, &tiff);
  AppendIFDEntry(262, 3, 1, 2, &tiff);
  AppendIFDEntry(273, 4, 1, pixel_offset, &tiff);
  AppendIFDEntry(274, 3, 1, 6, &tiff);  // Rotate 90 degrees clockwise.
  AppendIFDEntry(277, 3, 1, 3, &tiff);
  AppendIFDEntry(278, 4, 1, kHeight, &tiff);
  AppendIFDEntry(279, 4, 1, pixel_bytes, &tiff);
  AppendIFDEntry(284, 3, 1, 1, &tiff);
  AppendU32LE(0, &tiff);

  AppendU16LE(8, &tiff);
  AppendU16LE(8, &tiff);
  AppendU16LE(8, &tiff);
  EXPECT_EQ(pixel_offset, tiff.size());

  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      tiff.push_back(static_cast<uint8_t>(10 + x + 10 * y));
      tiff.push_back(0);
      tiff.push_back(0);
    }
  }
  return tiff;
}

TEST(TIFFTest, ReadsPixelsWithoutEmbeddingInputAsExif) {
  if (!CanDecodeTIFF()) GTEST_SKIP() << "TIFF support is not enabled";

  const std::vector<uint8_t> tiff = MakeTestTIFF();
  EXPECT_EQ(Codec::kTIFF, DetectCodec(Bytes(tiff)));

  PackedPixelFile ppf;
  Codec codec = Codec::kUnknown;
  ASSERT_TRUE(DecodeBytes(Bytes(tiff), ColorHints(), &ppf, nullptr, &codec));
  EXPECT_EQ(Codec::kTIFF, codec);
  EXPECT_EQ(256u, ppf.info.xsize);
  EXPECT_EQ(256u, ppf.info.ysize);
  EXPECT_EQ(3u, ppf.info.num_color_channels);
  EXPECT_EQ(8u, ppf.info.bits_per_sample);
  ASSERT_EQ(1u, ppf.frames.size());

  const PackedImage& image = ppf.frames.front().color;
  EXPECT_EQ(255u, image.const_pixels(0, 0, 0)[0]);
  EXPECT_EQ(255u, image.const_pixels(0, 0, 1)[0]);
  EXPECT_EQ(0u, image.const_pixels(0, 0, 2)[0]);
  EXPECT_EQ(0u, image.const_pixels(1, 1, 0)[0]);
  EXPECT_EQ(0u, image.const_pixels(1, 1, 1)[0]);
  EXPECT_TRUE(ppf.metadata.exif.empty());

  JXLCompressParams params;
  params.distance = 0.0f;
  params.compress_boxes = false;
  std::vector<uint8_t> compressed;
  ASSERT_TRUE(EncodeImageJXL(params, ppf, nullptr, &compressed));

  size_t payload_offset = 0;
  size_t payload_size = 0;
  EXPECT_FALSE(FindBox(compressed, "Exif", &payload_offset, &payload_size));
}

TEST(TIFFTest, AppliesDirectPathOrientation) {
  if (!CanDecodeTIFF()) GTEST_SKIP() << "TIFF support is not enabled";

  const std::vector<uint8_t> tiff = MakeTestTIFF(2, 1, 3);
  PackedPixelFile ppf;
  Codec codec = Codec::kUnknown;
  ASSERT_TRUE(DecodeBytes(Bytes(tiff), ColorHints(), &ppf, nullptr, &codec));
  EXPECT_EQ(Codec::kTIFF, codec);
  EXPECT_EQ(2u, ppf.info.xsize);
  EXPECT_EQ(1u, ppf.info.ysize);
  EXPECT_EQ(JXL_ORIENT_IDENTITY, ppf.info.orientation);
  ASSERT_EQ(1u, ppf.frames.size());

  const PackedImage& image = ppf.frames.front().color;
  EXPECT_EQ(0u, image.const_pixels(0, 0, 0)[0]);
  EXPECT_EQ(255u, image.const_pixels(0, 0, 1)[0]);
  EXPECT_EQ(255u, image.const_pixels(0, 1, 0)[0]);
  EXPECT_EQ(255u, image.const_pixels(0, 1, 1)[0]);
}

TEST(TIFFTest, AppliesDirectPathRotatedOrientation) {
  if (!CanDecodeTIFF()) GTEST_SKIP() << "TIFF support is not enabled";

  const std::vector<uint8_t> tiff = MakeOrientationRotateTIFF();
  PackedPixelFile ppf;
  Codec codec = Codec::kUnknown;
  ASSERT_TRUE(DecodeBytes(Bytes(tiff), ColorHints(), &ppf, nullptr, &codec));
  EXPECT_EQ(Codec::kTIFF, codec);
  EXPECT_EQ(3u, ppf.info.xsize);
  EXPECT_EQ(2u, ppf.info.ysize);
  EXPECT_EQ(JXL_ORIENT_IDENTITY, ppf.info.orientation);
  ASSERT_EQ(1u, ppf.frames.size());

  const PackedImage& image = ppf.frames.front().color;
  EXPECT_EQ(30u, image.const_pixels(0, 0, 0)[0]);
  EXPECT_EQ(20u, image.const_pixels(0, 1, 0)[0]);
  EXPECT_EQ(10u, image.const_pixels(0, 2, 0)[0]);
  EXPECT_EQ(31u, image.const_pixels(1, 0, 0)[0]);
  EXPECT_EQ(21u, image.const_pixels(1, 1, 0)[0]);
  EXPECT_EQ(11u, image.const_pixels(1, 2, 0)[0]);
}

TEST(TIFFTest, MarksRGBAFallbackAlphaAsPremultiplied) {
  if (!CanDecodeTIFF()) GTEST_SKIP() << "TIFF support is not enabled";

  const std::vector<uint8_t> tiff = MakeSeparateStraightAlphaTIFF();
  PackedPixelFile ppf;
  Codec codec = Codec::kUnknown;
  ASSERT_TRUE(DecodeBytes(Bytes(tiff), ColorHints(), &ppf, nullptr, &codec));
  EXPECT_EQ(Codec::kTIFF, codec);
  EXPECT_EQ(8u, ppf.info.alpha_bits);
  EXPECT_EQ(JXL_TRUE, ppf.info.alpha_premultiplied);
  ASSERT_EQ(1u, ppf.frames.size());

  const PackedImage& image = ppf.frames.front().color;
  ASSERT_EQ(4u, image.format.num_channels);
  EXPECT_EQ(100u, image.const_pixels(0, 0, 0)[0]);
  EXPECT_EQ(50u, image.const_pixels(0, 0, 1)[0]);
  EXPECT_EQ(25u, image.const_pixels(0, 0, 2)[0]);
  EXPECT_EQ(128u, image.const_pixels(0, 0, 3)[0]);
}

}  // namespace
}  // namespace extras
}  // namespace jxl
