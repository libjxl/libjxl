// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LIB_JXL_JPEG_JPEG_CONSTANTS_H_
#define LIB_JXL_JPEG_JPEG_CONSTANTS_H_

#include <stddef.h>
#include <stdint.h>

#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace jpeg {

// Refuse to allocate more than 1 GB of memory for the coefficients,
// that is 2M blocks x 64 coeffs x 2 bytes per coeff x max 4 components.
// TODO(eustas): this should be the minimal guaranteed limit, rather than
//               hard limit; 4GPx images should be eligible, if users would
//               want them.
static const size_t kBrunsliMaxNumBlocks = 1ull << 21;

// The maximum absolute value brunsli can encode is 2054 (8 values for direct
// codes and num bits from 1 to 10, so a total of 8 + 2 + 4 + ... + 1024).
static const int kBrunsliMaxDCAbsVal = 2054;

// We use only the context map alphabet in brunsli, where the maximum alphabet
// size is 256 + 16 = 272. (We can have 256 clusters and 16 run length codes).
static const size_t kMaxContextMapAlphabetSize = 272;

static const size_t kHuffmanTableBits = 8u;
static const size_t kMaxHuffmanBits = 15u;

// Total number of short markers allowed. Short marker represents popular marker
// and is encoded with 1 or 2 bytes in brunsli, but expanded into 15..3161 bytes
// in JPEG (see AppData_0x and GenerateApp0Marker). This poses "ZIP BOMB" threat
// because sequence of N bytes with value 0x80 will expand to 1580.5 * N bytes,
// but it could be compressed by brotli into just few bytes...
// On the other side, there is no reason to repeat any of those markers.
// Software that generates JPEG files might contain issues that would place
// repeated markers; to mitigate this, brunsli allows repetition of short
// markers, but sets the limit: the number of all unique marker variants.
static const int kBrunsliShortMarkerLimit = 0x40 + 3 * 0x100;

static const uint8_t kBrunsliWiringTypeVarint = 0x0;
static const uint8_t kBrunsliWiringTypeLengthDelimited = 0x2;

// The maximum supported V / H sampling factor.
static const int kBrunsliMaxSampling = 15;

constexpr uint8_t ValueMarker(uint8_t tag) {
  return (tag << 3) | kBrunsliWiringTypeVarint;
}

constexpr uint8_t SectionMarker(uint8_t tag) {
  return (tag << 3) | kBrunsliWiringTypeLengthDelimited;
}

static const uint8_t kBrunsliSignatureTag = 0x1;
static const uint8_t kBrunsliHeaderTag = 0x2;
static const uint8_t kBrunsliMetaDataTag = 0x3;
static const uint8_t kBrunsliJPEGInternalsTag = 0x4;
static const uint8_t kBrunsliQuantDataTag = 0x5;
static const uint8_t kBrunsliHistogramDataTag = 0x6;
static const uint8_t kBrunsliDCDataTag = 0x7;
static const uint8_t kBrunsliACDataTag = 0x8;
static const uint8_t kBrunsliOriginalJpgTag = 0x9;

// Header section. All fields are varints.
static const uint8_t kBrunsliHeaderWidthTag = 0x1;
static const uint8_t kBrunsliHeaderHeightTag = 0x2;
static const uint8_t kBrunsliHeaderVersionCompTag = 0x3;
static const uint8_t kBrunsliHeaderSubsamplingTag = 0x4;

static const size_t kBrunsliSignatureSize = 6;
extern const uint8_t kBrunsliSignature[kBrunsliSignatureSize];

static const size_t kMaxApp0Densities = 8;
static const uint16_t kApp0Densities[kMaxApp0Densities] = {
  1, 72, 96, 100, 150, 180, 240, 300
};

// TODO(eustas): reintroduce as a bit-length constant.
static const int kNumStockQuantTables = 8;
static const uint8_t
    kStockQuantizationTables[2][kNumStockQuantTables][64] = {
  {  // LUMA
    {  3,  2,  2,  3,  5,  8, 10, 12,
       2,  2,  3,  4,  5, 12, 12, 11,
       3,  3,  3,  5,  8, 11, 14, 11,
       3,  3,  4,  6, 10, 17, 16, 12,
       4,  4,  7, 11, 14, 22, 21, 15,
       5,  7, 11, 13, 16, 21, 23, 18,
      10, 13, 16, 17, 21, 24, 24, 20,
      14, 18, 19, 20, 22, 20, 21, 20 },
    {  8,  6,  5,  8, 12, 20, 26, 31,
       6,  6,  7, 10, 13, 29, 30, 28,
       7,  7,  8, 12, 20, 29, 35, 28,
       7,  9, 11, 15, 26, 44, 40, 31,
       9, 11, 19, 28, 34, 55, 52, 39,
      12, 18, 28, 32, 41, 52, 57, 46,
      25, 32, 39, 44, 52, 61, 60, 51,
      36, 46, 48, 49, 56, 50, 52, 50 },
    {  6,  4,  4,  6, 10, 16, 20, 24,
       5,  5,  6,  8, 10, 23, 24, 22,
       6,  5,  6, 10, 16, 23, 28, 22,
       6,  7,  9, 12, 20, 35, 32, 25,
       7,  9, 15, 22, 27, 44, 41, 31,
      10, 14, 22, 26, 32, 42, 45, 37,
      20, 26, 31, 35, 41, 48, 48, 40,
      29, 37, 38, 39, 45, 40, 41, 40 },
    {  5,  3,  3,  5,  7, 12, 15, 18,
       4,  4,  4,  6,  8, 17, 18, 17,
       4,  4,  5,  7, 12, 17, 21, 17,
       4,  5,  7,  9, 15, 26, 24, 19,
       5,  7, 11, 17, 20, 33, 31, 23,
       7, 11, 17, 19, 24, 31, 34, 28,
      15, 19, 23, 26, 31, 36, 36, 30,
      22, 28, 29, 29, 34, 30, 31, 30 },
    {  1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1 },
    {  2,  1,  1,  2,  2,  4,  5,  6,
       1,  1,  1,  2,  3,  6,  6,  6,
       1,  1,  2,  2,  4,  6,  7,  6,
       1,  2,  2,  3,  5,  9,  8,  6,
       2,  2,  4,  6,  7, 11, 10,  8,
       2,  4,  6,  6,  8, 10, 11,  9,
       5,  6,  8,  9, 10, 12, 12, 10,
       7,  9, 10, 10, 11, 10, 10, 10 },
    {  1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  2,
       1,  1,  1,  1,  1,  1,  2,  2,
       1,  1,  1,  1,  1,  2,  2,  3,
       1,  1,  1,  1,  2,  2,  3,  3,
       1,  1,  1,  2,  2,  3,  3,  3,
       1,  1,  2,  2,  3,  3,  3,  3 },
    { 10,  7,  6, 10, 14, 24, 31, 37,
       7,  7,  8, 11, 16, 35, 36, 33,
       8,  8, 10, 14, 24, 34, 41, 34,
       8, 10, 13, 17, 31, 52, 48, 37,
      11, 13, 22, 34, 41, 65, 62, 46,
      14, 21, 33, 38, 49, 62, 68, 55,
      29, 38, 47, 52, 62, 73, 72, 61,
      43, 55, 57, 59, 67, 60, 62, 59 }
  },
  {   // CHROMA
    {  9,  9,  9, 12, 11, 12, 24, 13,
      13, 24, 50, 33, 28, 33, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50 },
    {  3,  4,  5,  9, 20, 20, 20, 20,
       4,  4,  5, 13, 20, 20, 20, 20,
       5,  5, 11, 20, 20, 20, 20, 20,
       9, 13, 20, 20, 20, 20, 20, 20,
      20, 20, 20, 20, 20, 20, 20, 20,
      20, 20, 20, 20, 20, 20, 20, 20,
      20, 20, 20, 20, 20, 20, 20, 20,
      20, 20, 20, 20, 20, 20, 20, 20 },
    {  9,  9, 12, 24, 50, 50, 50, 50,
       9, 11, 13, 33, 50, 50, 50, 50,
      12, 13, 28, 50, 50, 50, 50, 50,
      24, 33, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50 },
    {  5,  5,  7, 14, 30, 30, 30, 30,
       5,  6,  8, 20, 30, 30, 30, 30,
       7,  8, 17, 30, 30, 30, 30, 30,
      14, 20, 30, 30, 30, 30, 30, 30,
      30, 30, 30, 30, 30, 30, 30, 30,
      30, 30, 30, 30, 30, 30, 30, 30,
      30, 30, 30, 30, 30, 30, 30, 30,
      30, 30, 30, 30, 30, 30, 30, 30 },
    {  7,  7, 10, 19, 40, 40, 40, 40,
       7,  8, 10, 26, 40, 40, 40, 40,
      10, 10, 22, 40, 40, 40, 40, 40,
      19, 26, 40, 40, 40, 40, 40, 40,
      40, 40, 40, 40, 40, 40, 40, 40,
      40, 40, 40, 40, 40, 40, 40, 40,
      40, 40, 40, 40, 40, 40, 40, 40,
      40, 40, 40, 40, 40, 40, 40, 40 },
    {  1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1,
       1,  1,  1,  1,  1,  1,  1,  1 },
    {  2,  2,  2,  5, 10, 10, 10, 10,
       2,  2,  3,  7, 10, 10, 10, 10,
       2,  3,  6, 10, 10, 10, 10, 10,
       5,  7, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10,
      10, 10, 10, 10, 10, 10, 10, 10 },
    { 10, 11, 14, 28, 59, 59, 59, 59,
      11, 13, 16, 40, 59, 59, 59, 59,
      14, 16, 34, 59, 59, 59, 59, 59,
      28, 40, 59, 59, 59, 59, 59, 59,
      59, 59, 59, 59, 59, 59, 59, 59,
      59, 59, 59, 59, 59, 59, 59, 59,
      59, 59, 59, 59, 59, 59, 59, 59,
      59, 59, 59, 59, 59, 59, 59, 59 }
  }
};

// TODO(eustas): reintroduce with bit-length.
static const int kComponentIds123 = 0;
static const int kComponentIdsGray = 1;
static const int kComponentIdsRGB = 2;
static const int kComponentIdsCustom = 3;

// TODO(eustas): reintroduce as bit-length
static const int kNumStockDCHuffmanCodes = 2;
static const int kStockDCHuffmanCodeCounts[kNumStockDCHuffmanCodes][
    kJpegHuffmanMaxBitLength] = {
  { 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, },
  { 0, 1, 5, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, },
};
// TODO(eustas): replace the final "256" with marker constant.
static const int kStockDCHuffmanCodeValues[kNumStockDCHuffmanCodes][
    kJpegDCAlphabetSize + 1] = {
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 256 },
  { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 256 },
};

// TODO(eustas): reintroduce as bit-length
static const int kNumStockACHuffmanCodes = 2;
static const int kStockACHuffmanCodeCounts[kNumStockACHuffmanCodes][
    kJpegHuffmanMaxBitLength] = {
  { 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 126, },
  { 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 120, },
};
static const int kStockACHuffmanCodeTotalCount = 163;
// TODO(eustas): replace the final "256" with marker constant.
static const int kStockACHuffmanCodeValues[kNumStockACHuffmanCodes][
    kStockACHuffmanCodeTotalCount] = {
  {
      1,   2,   3,   0,   4,  17,   5,  18,
     33,  49,  65,   6,  19,  81,  97,   7,
     34, 113,  20,  50, 129, 145, 161,   8,
     35,  66, 177, 193,  21,  82, 209, 240,
     36,  51,  98, 114, 130,   9,  10,  22,
     23,  24,  25,  26,  37,  38,  39,  40,
      41, 42,  52,  53,  54,  55,  56,  57,
     58,  67,  68,  69,  70,  71,  72,  73,
     74,  83,  84,  85,  86,  87,  88,  89,
     90,  99, 100, 101, 102, 103, 104, 105,
    106, 115, 116, 117, 118, 119, 120, 121,
    122, 131, 132, 133, 134, 135, 136, 137,
    138, 146, 147, 148, 149, 150, 151, 152,
    153, 154, 162, 163, 164, 165, 166, 167,
    168, 169, 170, 178, 179, 180, 181, 182,
    183, 184, 185, 186, 194, 195, 196, 197,
    198, 199, 200, 201, 202, 210, 211, 212,
    213, 214, 215, 216, 217, 218, 225, 226,
    227, 228, 229, 230, 231, 232, 233, 234,
    241, 242, 243, 244, 245, 246, 247, 248,
    249, 250, 256,
  },
  {
      0,   1,   2,   3,  17,   4,   5,  33,
     49,   6,  18,  65,  81,   7,  97, 113,
     19,  34,  50, 129,   8,  20,  66, 145,
    161, 177, 193,   9,  35,  51,  82, 240,
     21,  98, 114, 209,  10,  22,  36,  52,
    225,  37, 241,  23,  24,  25,  26,  38,
     39,  40,  41,  42,  53,  54,  55,  56,
     57,  58,  67,  68,  69,  70,  71,  72,
     73,  74,  83,  84,  85,  86,  87,  88,
     89,  90,  99, 100, 101, 102, 103, 104,
    105, 106, 115, 116, 117, 118, 119, 120,
    121, 122, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 146, 147, 148, 149, 150,
    151, 152, 153, 154, 162, 163, 164, 165,
    166, 167, 168, 169, 170, 178, 179, 180,
    181, 182, 183, 184, 185, 186, 194, 195,
    196, 197, 198, 199, 200, 201, 202, 210,
    211, 212, 213, 214, 215, 216, 217, 218,
    226, 227, 228, 229, 230, 231, 232, 233,
    234, 242, 243, 244, 245, 246, 247, 248,
    249, 250, 256,
  },
};

// Pre-defined tables for PermutationCoder.
static const uint8_t kDefaultDCValues[16] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
};
static const uint8_t kDefaultACValues[256] = {
  1, 0, 2, 3, 17, 4, 5, 33, 18, 49, 65, 6, 81, 19, 97, 7,
  34, 113, 50, 129, 20, 145, 161, 8, 35, 66, 177, 193, 21, 82, 209, 240,
  36, 51, 98, 114, 9, 130, 10, 22, 52, 225, 23, 37, 241, 24, 25, 26,
  38, 39, 40, 41, 42, 53, 54, 55, 56, 57, 58, 67, 68, 69, 70, 71,
  72, 73, 74, 83, 84, 85, 86, 87, 88, 89, 90, 99, 100, 101, 102, 103,
  104, 105, 106, 115, 116, 117, 118, 119, 120, 121, 122, 131, 132, 133, 134,
  135, 136, 137, 138, 146, 147, 148, 149, 150, 151, 152, 153, 154, 162, 163,
  164, 165, 166, 167, 168, 169, 170, 178, 179, 180, 181, 182, 183, 184, 185,
  186, 194, 195, 196, 197, 198, 199, 200, 201, 202, 210, 211, 212, 213, 214,
  215, 216, 217, 218, 226, 227, 228, 229, 230, 231, 232, 233, 234, 242, 243,
  244, 245, 246, 247, 248, 249, 250, 16, 32, 48, 64, 80, 96, 112, 128, 144,
  160, 176, 192, 208,
  // extra fill-in entries for missing values
  11, 12, 13, 14, 15, 27, 28, 29, 30, 31, 43, 44, 45, 46, 47, 59, 60,
  61, 62, 63, 75, 76, 77, 78, 79, 91, 92, 93, 94, 95, 107, 108, 109,
  110, 111, 123, 124, 125, 126, 127, 139, 140, 141, 142, 143, 155, 156,
  157, 158, 159, 171, 172, 173, 174, 175, 187, 188, 189, 190, 191, 203,
  204, 205, 206, 207, 219, 220, 221, 222, 223, 224, 235, 236, 237, 238,
  239, 251, 252, 253, 254, 255
};

// Common app-data chunks
extern const uint8_t AppData_0xe0[17];
extern const uint8_t AppData_0xe2[3161];  // special byte at offset 84
extern const uint8_t AppData_0xec[18];    // special byte at offset 15
extern const uint8_t AppData_0xee[15];    // special byte at offset 10

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_JPEG_CONSTANTS_H_
