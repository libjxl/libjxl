// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/apng.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace extras {
namespace {

void AppendBE32(std::vector<uint8_t>* out, uint32_t v) {
  out->push_back(static_cast<uint8_t>(v >> 24));
  out->push_back(static_cast<uint8_t>(v >> 16));
  out->push_back(static_cast<uint8_t>(v >> 8));
  out->push_back(static_cast<uint8_t>(v));
}

// Appends a PNG chunk. The decoder configures libpng with
// PNG_CRC_QUIET_USE, so chunk CRCs are not validated and a placeholder is
// enough.
void AppendChunk(std::vector<uint8_t>* out, const char* type,
                 const uint8_t* data, size_t size) {
  AppendBE32(out, static_cast<uint32_t>(size));
  out->insert(out->end(), type, type + 4);
  out->insert(out->end(), data, data + size);
  out->insert(out->end(), 4, 0);
}

std::vector<uint8_t> MakePng(uint32_t xsize, uint32_t ysize,
                             uint8_t bit_depth, uint8_t color_type) {
  std::vector<uint8_t> png = {137, 'P', 'N', 'G', '\r', '\n', 26, '\n'};
  std::vector<uint8_t> ihdr(13);
  ihdr[0] = static_cast<uint8_t>(xsize >> 24);
  ihdr[1] = static_cast<uint8_t>(xsize >> 16);
  ihdr[2] = static_cast<uint8_t>(xsize >> 8);
  ihdr[3] = static_cast<uint8_t>(xsize);
  ihdr[4] = static_cast<uint8_t>(ysize >> 24);
  ihdr[5] = static_cast<uint8_t>(ysize >> 16);
  ihdr[6] = static_cast<uint8_t>(ysize >> 8);
  ihdr[7] = static_cast<uint8_t>(ysize);
  ihdr[8] = bit_depth;
  ihdr[9] = color_type;
  ihdr[10] = 0;  // compression
  ihdr[11] = 0;  // filter
  ihdr[12] = 0;  // interlace
  AppendChunk(&png, "IHDR", ihdr.data(), ihdr.size());
  // The row buffer is allocated when the first IDAT chunk is seen, before
  // any pixel data is decoded, so a stub payload reaches that point.
  const uint8_t stub_idat[6] = {0x78, 0x01, 0x00, 0x00, 0xff, 0xff};
  AppendChunk(&png, "IDAT", stub_idat, sizeof(stub_idat));
  AppendChunk(&png, "IEND", nullptr, 0);
  return png;
}

TEST(CodecAPNGTest, HugeDimensionsFailGracefully) {
  if (!CanDecodeAPNG()) {
    fprintf(stderr, "Skipping test because of missing codec support.\n");
    return;
  }
  // 1000000 x 1000000 RGBA16 pixels describe a ~8 TiB row buffer. Decoding
  // must fail cleanly instead of attempting the allocation.
  const std::vector<uint8_t> png = MakePng(1000000, 1000000, 16, 6);
  PackedPixelFile ppf;
  EXPECT_FALSE(DecodeImageAPNG(Bytes(png), ColorHints(), &ppf, nullptr));
}

}  // namespace
}  // namespace extras
}  // namespace jxl
