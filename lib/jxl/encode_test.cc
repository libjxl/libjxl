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

#include "jxl/encode.h"

#include "gtest/gtest.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/test_utils.h"

TEST(EncodeTest, DefaultAllocTest) {
  JxlEncoder* enc = JxlEncoderCreate(nullptr);
  EXPECT_NE(nullptr, enc);
  JxlEncoderDestroy(enc);
}

TEST(EncodeTest, CustomAllocTest) {
  struct CalledCounters {
    int allocs = 0;
    int frees = 0;
  } counters;

  JxlMemoryManager mm;
  mm.opaque = &counters;
  mm.alloc = [](void* opaque, size_t size) {
    reinterpret_cast<CalledCounters*>(opaque)->allocs++;
    return malloc(size);
  };
  mm.free = [](void* opaque, void* address) {
    reinterpret_cast<CalledCounters*>(opaque)->frees++;
    free(address);
  };

  JxlEncoder* enc = JxlEncoderCreate(&mm);
  EXPECT_NE(nullptr, enc);
  EXPECT_LE(1, counters.allocs);
  EXPECT_EQ(0, counters.frees);
  JxlEncoderDestroy(enc);
  EXPECT_LE(1, counters.frees);
}

TEST(EncodeTest, DefaultParallelRunnerTest) {
  JxlEncoder* enc = JxlEncoderCreate(nullptr);
  EXPECT_NE(nullptr, enc);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetParallelRunner(enc, nullptr, nullptr));
  JxlEncoderDestroy(enc);
}

TEST(EncodeTest, FrameEncodingTest) {
  JxlEncoder* enc = JxlEncoderCreate(nullptr);
  EXPECT_NE(nullptr, enc);

  JxlPixelFormat pixel_format = {4, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};
  uint32_t width = 63;
  uint32_t height = 129;
  JxlFrameFormat frame_format = JxlFrameFormat{pixel_format, width, height};
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(width, height, 4);

  jxl::CodecInOut input_io =
      jxl::test::SomeTestImageToCodecInOut(pixels, width, height);

  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddImageFrame(enc, &frame_format, pixels.data(),
                                    pixels.size()));
  JxlEncoderCloseInput(enc);

  std::vector<uint8_t> compressed = std::vector<uint8_t>(64);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size() - (next_out - compressed.data());
  JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
  while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
    process_result = JxlEncoderProcessOutput(enc, &next_out, &avail_out);
    if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - compressed.data();
      compressed.resize(compressed.size() * 2);
      next_out = compressed.data() + offset;
      avail_out = compressed.size() - offset;
    }
  }
  compressed.resize(next_out - compressed.data());
  EXPECT_EQ(JXL_ENC_SUCCESS, process_result);

  jxl::DecompressParams dparams;
  jxl::CodecInOut decoded_io;
  EXPECT_TRUE(jxl::DecodeFile(
      dparams, jxl::Span<const uint8_t>(compressed.data(), compressed.size()),
      &decoded_io, /*aux_out=*/nullptr, /*pool=*/nullptr));

  jxl::ButteraugliParams ba;
  EXPECT_LE(ButteraugliDistance(input_io, decoded_io, ba,
                                /*distmap=*/nullptr, nullptr),
            2.0f);

  JxlEncoderDestroy(enc);
}
