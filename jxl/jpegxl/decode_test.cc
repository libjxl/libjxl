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

#include "jpegxl/decode.h"

#include <stdint.h>
#include <stdlib.h>

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

TEST(DecodeTest, JpegxlSignatureCheckTest) {
  std::vector<std::pair<int, std::vector<uint8_t>>> tests = {
      // No JPEGXL header starts with 'a'.
      {JPEGXL_SIG_INVALID, {'a'}},
      {JPEGXL_SIG_INVALID, {'a', 'b', 'c', 'd', 'e', 'f'}},

      // JPEGXL headers.
      {JPEGXL_SIG_NOT_ENOUGH_BYTES, {0xff}},  // Part of a signature.
      {JPEGXL_SIG_JPEGXL, {0xff, 0x58}},
      {JPEGXL_SIG_JPEGXL, {0xff, 0x0a}},

      // A header could start with 0x0a, but it is not a complete signature.
      {JPEGXL_SIG_NOT_ENOUGH_BYTES, {0x0a}},

      // This is the beginning of a Brunsli file, but not a complete signature.
      {JPEGXL_SIG_NOT_ENOUGH_BYTES, {0x0a, 0x04, 'B', 0xd2}},
      {JPEGXL_SIG_BRUNSLI, {0x0a, 0x04, 'B', 0xd2, 0xd5, 'N', 0x12}},
      {JPEGXL_SIG_INVALID, {0x0a, 0x04, 'B', 0xd2, 0xd5, 'N', 0x13}},
  };
  for (const auto& test : tests) {
    EXPECT_EQ(test.first,
              JpegxlSignatureCheck(test.second.data(), test.second.size()))
        << "Where test data is " << ::testing::PrintToString(test.second);
  }
}

TEST(DecodeTest, DefaultAllocTest) {
  JpegxlDecoder* dec = JpegxlDecoderCreate(nullptr);
  EXPECT_NE(nullptr, dec);
  JpegxlDecoderDestroy(dec);
}

TEST(DecodeTest, CustomAllocTest) {
  struct CalledCounters {
    int allocs = 0;
    int frees = 0;
  } counters;

  JpegxlMemoryManager mm;
  mm.opaque = &counters;
  mm.alloc = [](void* opaque, size_t size) {
    reinterpret_cast<CalledCounters*>(opaque)->allocs++;
    return malloc(size);
  };
  mm.free = [](void* opaque, void* address) {
    reinterpret_cast<CalledCounters*>(opaque)->frees++;
    free(address);
  };

  JpegxlDecoder* dec = JpegxlDecoderCreate(&mm);
  EXPECT_NE(nullptr, dec);
  EXPECT_LE(1, counters.allocs);
  EXPECT_EQ(0, counters.frees);
  JpegxlDecoderDestroy(dec);
  EXPECT_LE(1, counters.frees);
}
