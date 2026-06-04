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

std::vector<uint8_t> MakeTestTIFF() {
  constexpr uint32_t kWidth = 256;
  constexpr uint32_t kHeight = 256;
  constexpr uint16_t kEntryCount = 13;
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
  const uint32_t pixel_bytes = kWidth * kHeight * 3;

  std::vector<uint8_t> tiff;
  tiff.reserve(pixel_offset + pixel_bytes);
  tiff.push_back('I');
  tiff.push_back('I');
  AppendU16LE(42, &tiff);
  AppendU32LE(ifd_offset, &tiff);

  AppendU16LE(kEntryCount, &tiff);
  AppendIFDEntry(256, 4, 1, kWidth, &tiff);       // ImageWidth
  AppendIFDEntry(257, 4, 1, kHeight, &tiff);      // ImageLength
  AppendIFDEntry(258, 3, 3, bits_offset, &tiff);  // BitsPerSample
  AppendIFDEntry(259, 3, 1, 1, &tiff);            // Compression: none
  AppendIFDEntry(262, 3, 1, 2, &tiff);            // Photometric: RGB
  AppendIFDEntry(271, 2, sizeof(kMake), make_offset, &tiff);
  AppendIFDEntry(272, 2, sizeof(kModel), model_offset, &tiff);
  AppendIFDEntry(273, 4, 1, pixel_offset, &tiff);  // StripOffsets
  AppendIFDEntry(277, 3, 1, 3, &tiff);             // SamplesPerPixel
  AppendIFDEntry(278, 4, 1, kHeight, &tiff);       // RowsPerStrip
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

  for (uint32_t y = 0; y < kHeight; ++y) {
    for (uint32_t x = 0; x < kWidth; ++x) {
      tiff.push_back(x == 0 ? 255 : 0);
      tiff.push_back(y == 0 ? 255 : 0);
      tiff.push_back(0);
    }
  }
  return tiff;
}

TEST(TIFFTest, ReadsPixelsAndPreservesOriginalTiffAsExif) {
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
  EXPECT_EQ(tiff, ppf.metadata.exif);

  JXLCompressParams params;
  params.distance = 0.0f;
  params.compress_boxes = false;
  std::vector<uint8_t> compressed;
  ASSERT_TRUE(EncodeImageJXL(params, ppf, nullptr, &compressed));

  size_t payload_offset = 0;
  size_t payload_size = 0;
  ASSERT_TRUE(FindBox(compressed, "Exif", &payload_offset, &payload_size));
  ASSERT_EQ(tiff.size() + 4, payload_size);
  EXPECT_EQ(0, memcmp(compressed.data() + payload_offset, "\0\0\0\0", 4));
  EXPECT_EQ(0, memcmp(compressed.data() + payload_offset + 4, tiff.data(),
                      tiff.size()));
}

}  // namespace
}  // namespace extras
}  // namespace jxl
