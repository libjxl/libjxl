// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/wasm_demo/no_png.h"

#include <array>
#include <memory>

extern "C" {

namespace {

static std::array<uint32_t, 256> makeCrc32Lut() {
  std::array<uint32_t, 256> result;
  for (uint32_t i = 0; i < 256; ++i) {
    constexpr uint32_t poly = 0xEDB88320;
    uint32_t v = i;
    for (size_t i = 0; i < 8; ++i) {
      uint32_t mask = ~((v & 1) - 1);
      v = (v >> 1) ^ (poly & mask);
    }
    result[i] = v;
  }
  return result;
}

const std::array<uint32_t, 256> kCrc32Lut = makeCrc32Lut();

const std::array<uint32_t, 8> kPngMagic = {137, 80, 78, 71, 13, 10, 26, 10};

// No need to SIMDify it, only small blocks are actually checksummed.
uint32_t CalculateCrc32(const uint8_t* start, const uint8_t* end) {
  uint32_t result = ~0;
  for (const uint8_t* data = start; data < end; ++data) {
    result ^= *data;
    result = (result >> 8) ^ kCrc32Lut[result & 0xFF];
  }
  return ~result;
}

void AdlerCopy(const uint8_t* src, uint8_t* dst, size_t length, uint32_t* s1,
               uint32_t* s2) {
  // TODO: SIMD-ify and use multithreading.

  // Precondition: s1, s2 normalized; length <= 65535
  uint32_t a = *s1;
  uint32_t b = *s2;

  for (size_t i = 0; i < length; ++i) {
    const uint8_t v = src[i];
    a += v;
    b += a;
    dst[i] = v;
  }

  // Postcondition: s1, s2 normalized.
  *s1 = a % 65521;
  *s2 = b % 65521;
}

constexpr size_t kMaxDeflateBlock = 65535;
constexpr uint32_t kIhdrSize = 13;

}  // namespace

uint8_t* WrapPixelsToPng(size_t width, size_t height, size_t bit_depth,
                         bool has_alpha, const uint8_t* input,
                         uint32_t* output_size) {
  size_t row_size = width * (bit_depth / 8) * (3 + has_alpha);
  size_t data_size = height * (row_size + 1);
  size_t num_deflate_blocks =
      (data_size + kMaxDeflateBlock - 1) / kMaxDeflateBlock;
  size_t idat_size = data_size + num_deflate_blocks * 5 + 6;
  size_t total_size = 0;
  total_size += kPngMagic.size();
  total_size += 12 + kIhdrSize;
  total_size += 12 + idat_size;
  total_size += 12;  // IEND

  uint8_t* output = static_cast<uint8_t*>(malloc(total_size));
  if (!output) {
    return nullptr;
  }
  uint8_t* dst = output;
  *output_size = total_size;

  for (size_t i = 0; i < kPngMagic.size(); ++i) {
    *(dst++) = kPngMagic[i];
  }

  // IHDR
  *reinterpret_cast<uint32_t*>(dst) = __builtin_bswap32(kIhdrSize);
  dst += 4;
  uint8_t* chunk_start = dst;
  *reinterpret_cast<uint32_t*>(dst) = 0x52444849;
  dst += 4;
  *reinterpret_cast<uint32_t*>(dst) = __builtin_bswap32(width);
  dst += 4;
  *reinterpret_cast<uint32_t*>(dst) = __builtin_bswap32(height);
  dst += 4;
  *(dst++) = bit_depth;
  *(dst++) = has_alpha ? 6 : 2;
  *(dst++) = 0;  // deflate
  *(dst++) = 0;  // standard filters
  *(dst++) = 0;  // no interlace
  uint32_t crc32 = CalculateCrc32(chunk_start, dst);
  *reinterpret_cast<uint32_t*>(dst) = __builtin_bswap32(crc32);
  dst += 4;

  // IDAT
  *reinterpret_cast<uint32_t*>(dst) = __builtin_bswap32(idat_size);
  dst += 4;
  *reinterpret_cast<uint32_t*>(dst) = 0x54414449;
  dst += 4;
  size_t offset = 0;
  size_t bytes_to_next_row = 0;
  uint32_t adler_s1 = 1;
  uint32_t adler_s2 = 0;
  *(dst++) = 0x08;  // CM = 8 (deflate), CINFO = 0 (window size = 2**(0+8))
  *(dst++) = 29;    // FCHECK; (FCHECK + 256* CMF) % 31 = 0
  for (size_t i = 0; i < num_deflate_blocks; ++i) {
    size_t block_size = data_size - offset;
    if (block_size > kMaxDeflateBlock) {
      block_size = kMaxDeflateBlock;
    }
    bool is_last = ((i + 1) == num_deflate_blocks);
    *(dst++) = is_last;  // btype = 00 (uncompressed)
    offset += block_size;
    *reinterpret_cast<uint16_t*>(dst) = block_size;
    dst += 2;
    *reinterpret_cast<uint16_t*>(dst) = ~block_size;
    dst += 2;
    while (block_size > 0) {
      if (bytes_to_next_row == 0) {
        *(dst++) = 0;
        adler_s2 += adler_s1;
        bytes_to_next_row = row_size;
        block_size--;
        continue;
      }
      size_t bytes_to_copy = std::min(block_size, bytes_to_next_row);
      AdlerCopy(input, dst, bytes_to_copy, &adler_s1, &adler_s2);
      dst += bytes_to_copy;
      input += bytes_to_copy;
      block_size -= bytes_to_copy;
      bytes_to_next_row -= bytes_to_copy;
    }
  }
  uint32_t adler = (adler_s2 << 8) | adler_s1;
  *reinterpret_cast<uint32_t*>(dst) = __builtin_bswap32(adler);
  dst += 4;
  *reinterpret_cast<uint32_t*>(dst) = 0;  // Fake CRC32
  dst += 4;

  // IEND
  *reinterpret_cast<uint32_t*>(dst) = __builtin_bswap32(0);
  dst += 4;
  chunk_start = dst;
  *reinterpret_cast<uint32_t*>(dst) = 0x444E4549;
  dst += 4;
  crc32 = CalculateCrc32(chunk_start, dst);
  *reinterpret_cast<uint32_t*>(dst) = __builtin_bswap32(crc32);
  dst += 4;

  return output;
}

}  // extern "C"
