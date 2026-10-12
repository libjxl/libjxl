// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/exr.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace extras {
namespace {

void AppendLE32(std::vector<uint8_t>* out, uint32_t v) {
  for (int i = 0; i < 4; ++i) out->push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void AppendLE64(std::vector<uint8_t>* out, uint64_t v) {
  AppendLE32(out, static_cast<uint32_t>(v));
  AppendLE32(out, static_cast<uint32_t>(v >> 32));
}

void AppendAttr(std::vector<uint8_t>* out, const char* name, const char* type,
                const std::vector<uint8_t>& value) {
  out->insert(out->end(), name, name + strlen(name) + 1);
  out->insert(out->end(), type, type + strlen(type) + 1);
  AppendLE32(out, static_cast<uint32_t>(value.size()));
  out->insert(out->end(), value.begin(), value.end());
}

std::vector<uint8_t> LE32Box(int32_t x1, int32_t y1) {
  std::vector<uint8_t> box;
  for (int32_t v : {0, 0, x1, y1}) AppendLE32(&box, static_cast<uint32_t>(v));
  return box;
}

// A valid single-part scanline EXR file: one HALF "Y" channel, no compression,
// `width` x 1 pixels, with a single line block that declares and contains only
// `block_bytes` bytes of pixel data. A full scanline needs 2 * `width` bytes.
std::vector<uint8_t> MakeShortBlockExr(uint32_t width, uint32_t block_bytes) {
  std::vector<uint8_t> out = {0x76, 0x2f, 0x31, 0x01};  // magic
  AppendLE32(&out, 2);                                // version 2, no flags

  std::vector<uint8_t> chlist = {'Y', 0};
  AppendLE32(&chlist, 1);                     // HALF
  chlist.insert(chlist.end(), {0, 0, 0, 0});  // pLinear + reserved
  AppendLE32(&chlist, 1);                     // xSampling
  AppendLE32(&chlist, 1);                     // ySampling
  chlist.push_back(0);                        // end of channel list
  AppendAttr(&out, "channels", "chlist", chlist);
  AppendAttr(&out, "compression", "compression", {0});  // NO_COMPRESSION
  const std::vector<uint8_t> window = LE32Box(width - 1, 0);
  AppendAttr(&out, "dataWindow", "box2i", window);
  AppendAttr(&out, "displayWindow", "box2i", window);
  AppendAttr(&out, "lineOrder", "lineOrder", {0});
  const std::vector<uint8_t> one = {0, 0, 0x80, 0x3f};  // 1.0f
  AppendAttr(&out, "pixelAspectRatio", "float", one);
  AppendAttr(&out, "screenWindowCenter", "v2f", std::vector<uint8_t>(8, 0));
  AppendAttr(&out, "screenWindowWidth", "float", one);
  out.push_back(0);  // end of header

  AppendLE64(&out, out.size() + 8);  // offset table: start of the only block
  AppendLE32(&out, 0);               // block's y coordinate
  AppendLE32(&out, block_bytes);     // block's packed size
  out.insert(out.end(), block_bytes, 0x3c);
  return out;
}

// A line block that claims far fewer bytes than one scanline must not cause a
// read past the end of the input buffer. OpenEXR sizes the per-line read from
// the image header, so when the input stream reports itself as memory-mapped
// it reads `2 * width` bytes starting inside our buffer even though only
// `block_bytes` are present. With a wide line that is a large out-of-bounds
// read of our input, reachable from cjxl on an untrusted file. The read is in
// OpenEXR (a system library that is normally not sanitizer-instrumented), so it
// surfaces as a crash once the over-read reaches an unmapped page rather than as
// an ASan report; the width here makes the over-read (~0.5 MB past a small
// input) large enough to fault deterministically on the stock decoder. The
// decoded pixels of such a corrupt file are unspecified, so this does not assert
// on the result, only that decoding returns without walking off the input.
TEST(CodecEXRTest, ShortLineBlockDoesNotReadPastInput) {
  if (!CanDecodeEXR()) {
    fprintf(stderr, "Skipping test because of missing EXR support.\n");
    return;
  }
  const std::vector<uint8_t> exr =
      MakeShortBlockExr(/*width=*/262144, /*block_bytes=*/16);
  PackedPixelFile ppf;
  (void)DecodeImageEXR(Bytes(exr), ColorHints(), &ppf, nullptr);
}

// A block that declares a full scanline still decodes, so the handling above is
// about the short block and not a regression in reading valid files.
TEST(CodecEXRTest, FullLineBlockDecodes) {
  if (!CanDecodeEXR()) {
    fprintf(stderr, "Skipping test because of missing EXR support.\n");
    return;
  }
  const uint32_t width = 8;
  const std::vector<uint8_t> exr =
      MakeShortBlockExr(width, /*block_bytes=*/2 * width);
  PackedPixelFile ppf;
  ASSERT_TRUE(DecodeImageEXR(Bytes(exr), ColorHints(), &ppf, nullptr));
  EXPECT_EQ(ppf.info.xsize, width);
  EXPECT_EQ(ppf.info.ysize, 1u);
}

}  // namespace
}  // namespace extras
}  // namespace jxl
