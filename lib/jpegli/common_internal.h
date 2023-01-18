// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_COMMON_INTERNAL_H_
#define LIB_JPEGLI_COMMON_INTERNAL_H_

#include <stddef.h>
#include <stdint.h>

namespace jpegli {

template <typename T1, typename T2>
constexpr inline T1 DivCeil(T1 a, T2 b) {
  return (a + b - 1) / b;
}

constexpr size_t kDCTBlockSize = 64;
constexpr int kMaxComponents = 4;
constexpr int kMaxQuantTables = 4;
constexpr int kMaxHuffmanTables = 4;
constexpr size_t kJpegHuffmanMaxBitLength = 16;
constexpr int kJpegHuffmanAlphabetSize = 256;
constexpr int kJpegDCAlphabetSize = 12;
constexpr int kMaxDHTMarkers = 512;
constexpr int kMaxDimPixels = 65535;
constexpr uint8_t kApp1 = 0xE1;
constexpr uint8_t kApp2 = 0xE2;
const uint8_t kIccProfileTag[12] = "ICC_PROFILE";
const uint8_t kExifTag[6] = "Exif\0";
const uint8_t kXMPTag[29] = "http://ns.adobe.com/xap/1.0/";

/* clang-format off */
constexpr uint32_t kJPEGNaturalOrder[80] = {
  0,   1,  8, 16,  9,  2,  3, 10,
  17, 24, 32, 25, 18, 11,  4,  5,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13,  6,  7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63,
  // extra entries for safety in decoder
  63, 63, 63, 63, 63, 63, 63, 63,
  63, 63, 63, 63, 63, 63, 63, 63
};

constexpr uint32_t kJPEGZigZagOrder[64] = {
  0,   1,  5,  6, 14, 15, 27, 28,
  2,   4,  7, 13, 16, 26, 29, 42,
  3,   8, 12, 17, 25, 30, 41, 43,
  9,  11, 18, 24, 31, 40, 44, 53,
  10, 19, 23, 32, 39, 45, 52, 54,
  20, 22, 33, 38, 46, 51, 55, 60,
  21, 34, 37, 47, 50, 56, 59, 61,
  35, 36, 48, 49, 57, 58, 62, 63
};
/* clang-format on */

}  // namespace jpegli

#endif  // LIB_JPEGLI_COMMON_INTERNAL_H_
