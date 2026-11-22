// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tests for the bound on the total amount of decoded pixel memory in the
// APNG decoder. VerifyDimensions only bounds a single image, so without a
// global budget a tiny file with many (possibly truncated) frames can make the
// decoder hold an arbitrary amount of memory.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/extras/dec/apng.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/size_constraints.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/testing.h"
#include "zlib.h"

namespace jxl {
namespace extras {
namespace {

void AppendU16(std::vector<uint8_t>* out, uint32_t v) {
  out->push_back(static_cast<uint8_t>(v >> 8));
  out->push_back(static_cast<uint8_t>(v));
}

void AppendU32(std::vector<uint8_t>* out, uint32_t v) {
  out->push_back(static_cast<uint8_t>(v >> 24));
  out->push_back(static_cast<uint8_t>(v >> 16));
  out->push_back(static_cast<uint8_t>(v >> 8));
  out->push_back(static_cast<uint8_t>(v));
}

void AppendChunk(std::vector<uint8_t>* out, const char* type,
                 const std::vector<uint8_t>& payload) {
  AppendU32(out, static_cast<uint32_t>(payload.size()));
  const size_t start = out->size();
  out->insert(out->end(), type, type + 4);
  out->insert(out->end(), payload.begin(), payload.end());
  const uLong crc = crc32(0, out->data() + start, payload.size() + 4);
  AppendU32(out, static_cast<uint32_t>(crc));
}

std::vector<uint8_t> Deflate(const std::vector<uint8_t>& raw) {
  uLongf out_size = compressBound(raw.size());
  std::vector<uint8_t> out(out_size);
  EXPECT_EQ(Z_OK,
            compress2(out.data(), &out_size, raw.data(), raw.size(), /*level=*/9));
  out.resize(out_size);
  return out;
}

// Builds an 8-bit RGBA APNG with `num_frames` full-canvas frames. If
// `with_rows` is false every frame carries an empty deflate stream, i.e. no
// pixel rows at all; the decoder zero-fills those rows (see #5083), so each
// frame still costs a full frame buffer.
std::vector<uint8_t> MakeApng(size_t xsize, size_t ysize, size_t num_frames,
                              bool with_rows) {
  std::vector<uint8_t> out = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
  std::vector<uint8_t> ihdr;
  AppendU32(&ihdr, static_cast<uint32_t>(xsize));
  AppendU32(&ihdr, static_cast<uint32_t>(ysize));
  ihdr.push_back(8);  // bit depth
  ihdr.push_back(6);  // color type: RGBA
  ihdr.push_back(0);  // compression
  ihdr.push_back(0);  // filter
  ihdr.push_back(0);  // interlace
  AppendChunk(&out, "IHDR", ihdr);
  std::vector<uint8_t> actl;
  AppendU32(&actl, static_cast<uint32_t>(num_frames));
  AppendU32(&actl, 0);
  AppendChunk(&out, "acTL", actl);

  std::vector<uint8_t> raw;
  if (with_rows) raw.assign((1 + 4 * xsize) * ysize, 0);
  const std::vector<uint8_t> idat = Deflate(raw);

  uint32_t seq = 0;
  for (size_t i = 0; i < num_frames; ++i) {
    std::vector<uint8_t> fctl;
    AppendU32(&fctl, seq++);
    AppendU32(&fctl, static_cast<uint32_t>(xsize));
    AppendU32(&fctl, static_cast<uint32_t>(ysize));
    AppendU32(&fctl, 0);  // x offset
    AppendU32(&fctl, 0);  // y offset
    AppendU16(&fctl, 1);  // delay numerator
    AppendU16(&fctl, 10);  // delay denominator
    fctl.push_back(0);    // dispose op: none
    fctl.push_back(0);    // blend op: source
    AppendChunk(&out, "fcTL", fctl);
    if (i == 0) {
      AppendChunk(&out, "IDAT", idat);
    } else {
      std::vector<uint8_t> fdat;
      AppendU32(&fdat, seq++);
      fdat.insert(fdat.end(), idat.begin(), idat.end());
      AppendChunk(&out, "fdAT", fdat);
    }
  }
  AppendChunk(&out, "IEND", {});
  return out;
}

size_t DecodedBytes(const PackedPixelFile& ppf) {
  size_t total = 0;
  for (const auto& frame : ppf.frames) total += frame.color.pixels_size;
  return total;
}

TEST(DecodedSizeTest, ApngRejectsTotalFrameAmplification) {
  // 2 frames x 4096x4096 RGBA from ~200 bytes of input.
  const std::vector<uint8_t> apng = MakeApng(4096, 4096, 2, /*with_rows=*/false);
  ASSERT_LT(apng.size(), 1000u);
  PackedPixelFile ppf;
  EXPECT_FALSE(
      DecodeImageAPNG(Bytes(apng.data(), apng.size()), ColorHints(), &ppf,
                      /*constraints=*/nullptr));
}

TEST(DecodedSizeTest, ApngAcceptsLargeAnimation) {
  // 4 frames x 2048x2048 RGBA = 64 MiB of decoded pixels, which is above the
  // floor of the budget and must be accepted.
  const std::vector<uint8_t> apng = MakeApng(2048, 2048, 4, /*with_rows=*/true);
  PackedPixelFile ppf;
  ASSERT_TRUE(DecodeImageAPNG(Bytes(apng.data(), apng.size()), ColorHints(),
                              &ppf, /*constraints=*/nullptr));
  ASSERT_EQ(4u, ppf.frames.size());
  EXPECT_EQ(4u * 2048u * 2048u * 4u, DecodedBytes(ppf));
}

TEST(DecodedSizeTest, ApngHonorsTotalPixelLimit) {
  // The total number of decoded pixels is checked against dec_max_pixels, as
  // the GIF decoder does.
  const std::vector<uint8_t> apng = MakeApng(256, 256, 8, /*with_rows=*/true);
  PackedPixelFile ppf;
  ASSERT_TRUE(DecodeImageAPNG(Bytes(apng.data(), apng.size()), ColorHints(),
                              &ppf, /*constraints=*/nullptr));
  SizeConstraints constraints;
  constraints.dec_max_pixels = 4 * 256 * 256;  // room for 4 frames only
  PackedPixelFile ppf2;
  EXPECT_FALSE(DecodeImageAPNG(Bytes(apng.data(), apng.size()), ColorHints(),
                               &ppf2, &constraints));
}

}  // namespace
}  // namespace extras
}  // namespace jxl
