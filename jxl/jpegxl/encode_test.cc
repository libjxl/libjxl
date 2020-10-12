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

#include "jpegxl/encode.h"

#include "gtest/gtest.h"

TEST(EncodeTest, DefaultAllocTest) {
  JpegxlEncoder* enc = JpegxlEncoderCreate(nullptr);
  EXPECT_NE(nullptr, enc);
  JpegxlEncoderDestroy(enc);
}

TEST(EncodeTest, CustomAllocTest) {
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

  JpegxlEncoder* enc = JpegxlEncoderCreate(&mm);
  EXPECT_NE(nullptr, enc);
  EXPECT_LE(1, counters.allocs);
  EXPECT_EQ(0, counters.frees);
  JpegxlEncoderDestroy(enc);
  EXPECT_LE(1, counters.frees);
}

// TODO(zond): add multi-threaded test when multithreaded pixel encoding from
// API is implemented.
TEST(EncodeTest, DefaultParallelRunnerTest) {
  JpegxlEncoder* enc = JpegxlEncoderCreate(nullptr);
  EXPECT_NE(nullptr, enc);
  EXPECT_EQ(JPEGXL_ENC_SUCCESS,
            JpegxlEncoderSetParallelRunner(enc, nullptr, nullptr));
  JpegxlEncoderDestroy(enc);
}

