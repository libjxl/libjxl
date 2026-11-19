// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/exr.h"

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

void AppendLE32(std::vector<uint8_t>* out, uint32_t v) {
  out->push_back(static_cast<uint8_t>(v));
  out->push_back(static_cast<uint8_t>(v >> 8));
  out->push_back(static_cast<uint8_t>(v >> 16));
  out->push_back(static_cast<uint8_t>(v >> 24));
}

void AppendLE64(std::vector<uint8_t>* out, uint64_t v) {
  AppendLE32(out, static_cast<uint32_t>(v));
  AppendLE32(out, static_cast<uint32_t>(v >> 32));
}

void AppendAttr(std::vector<uint8_t>* out, const char* name, const char* type,
                const uint8_t* value, uint32_t size) {
  out->insert(out->end(), name, name + strlen(name) + 1);
  out->insert(out->end(), type, type + strlen(type) + 1);
  AppendLE32(out, size);
  out->insert(out->end(), value, value + size);
}

// Appends a channel description; EXR requires channels to be sorted
// alphabetically by name.
void AppendChannel(std::vector<uint8_t>* chlist, const char* name,
                   int32_t pixel_type) {
  chlist->insert(chlist->end(), name, name + strlen(name) + 1);
  AppendLE32(chlist, static_cast<uint32_t>(pixel_type));
  chlist->push_back(0);  // pLinear
  chlist->push_back(0);
  chlist->push_back(0);
  chlist->push_back(0);  // reserved
  AppendLE32(chlist, 1);  // xSampling
  AppendLE32(chlist, 1);  // ySampling
}

// Builds a scanline EXR file with an offset table for a single 1-line data
// window. `extra_channel` adds a FLOAT channel besides HALF B/G/R.
// The data window is 2^20 x 1 and the display window is 1 x 2^20, so the
// decoder-side buffer size is driven by image_height * data_width while
// each window individually stays well within the dimension limits.
std::vector<uint8_t> MakeExr(bool extra_channel) {
  std::vector<uint8_t> out;
  out.insert(out.end(), {0x76, 0x2f, 0x31, 0x01});  // magic
  AppendLE32(&out, 2);                               // version 2, no flags

  std::vector<uint8_t> chlist;
  AppendChannel(&chlist, "B", 1);  // HALF
  if (extra_channel) AppendChannel(&chlist, "depth", 2);  // FLOAT
  AppendChannel(&chlist, "G", 1);
  AppendChannel(&chlist, "R", 1);
  chlist.push_back(0);  // end of channel list
  AppendAttr(&out, "channels", "chlist", chlist.data(),
             static_cast<uint32_t>(chlist.size()));
  const uint8_t kCompressionNone = 0;
  AppendAttr(&out, "compression", "compression", &kCompressionNone, 1);
  const uint8_t kDataWindow[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0x0f,
                                   0, 0, 0, 0, 0};  // (0,0)-(2^20-1, 0)
  AppendAttr(&out, "dataWindow", "box2i", kDataWindow, 16);
  const uint8_t kDisplayWindow[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0xff, 0xff, 0x0f, 0};  // (0,0)-(0, 2^20-1)
  AppendAttr(&out, "displayWindow", "box2i", kDisplayWindow, 16);
  const uint8_t kLineOrder[1] = {0};
  AppendAttr(&out, "lineOrder", "lineOrder", kLineOrder, 1);
  const uint8_t kOne[4] = {0, 0, 0x80, 0x3f};  // 1.0f
  AppendAttr(&out, "pixelAspectRatio", "float", kOne, 4);
  const uint8_t kZeroV2f[8] = {0};
  AppendAttr(&out, "screenWindowCenter", "v2f", kZeroV2f, 8);
  AppendAttr(&out, "screenWindowWidth", "float", kOne, 4);
  out.push_back(0);  // end of header

  // Chunk offset table for the 1-line data window: a single entry. Point it
  // at the end of the file; it is not dereferenced before the allocation
  // that this test exercises. A zero entry would make OpenEXR attempt
  // offset reconstruction, which reads past the end of the file.
  AppendLE64(&out, out.size() + 8);
  return out;
}

TEST(CodecEXRTest, HugeBufferFailGracefully) {
  if (!CanDecodeEXR()) {
    fprintf(stderr, "Skipping test because of missing codec support.\n");
    return;
  }
  // image_height * data_width = 2^40 pixels; the interleaved color buffer
  // would be ~6 TiB. Decoding must fail cleanly instead of attempting the
  // allocation.
  const std::vector<uint8_t> exr = MakeExr(/*extra_channel=*/false);
  PackedPixelFile ppf;
  EXPECT_FALSE(DecodeImageEXR(Bytes(exr), ColorHints(), &ppf, nullptr));
}

TEST(CodecEXRTest, HugeExtraChannelBufferFailGracefully) {
  if (!CanDecodeEXR()) {
    fprintf(stderr, "Skipping test because of missing codec support.\n");
    return;
  }
  // Same as above, but the oversized buffer is the extra channel plane.
  const std::vector<uint8_t> exr = MakeExr(/*extra_channel=*/true);
  PackedPixelFile ppf;
  EXPECT_FALSE(DecodeImageEXR(Bytes(exr), ColorHints(), &ppf, nullptr));
}

}  // namespace
}  // namespace extras
}  // namespace jxl
