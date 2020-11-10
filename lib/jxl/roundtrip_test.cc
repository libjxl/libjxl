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

#include "jxl/decode.h"
#include "jxl/encode.h"

#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/test_utils.h"

#include "gtest/gtest.h"

TEST(RoundtripTest, FrameRoundtripTest) {
  JxlEncoder* enc = JxlEncoderCreate(nullptr);
  EXPECT_NE(nullptr, enc);

  uint32_t channels = 4;
  JxlPixelFormat pixel_format = {channels, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};
  uint32_t width = 63;
  uint32_t height = 129;
  JxlFrameFormat original_frame_format = JxlFrameFormat{pixel_format, width, height};
  std::vector<uint8_t> original_pixels = jxl::test::GetSomeTestImage(width, height, channels);

  jxl::CodecInOut original_io =
      jxl::test::SomeTestImageToCodecInOut(original_pixels, width, height);

  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddImageFrame(enc, &original_frame_format, original_pixels.data(),
                                    original_pixels.size()));
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

  JxlEncoderDestroy(enc);

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  EXPECT_NE(nullptr, dec);

  const uint8_t* next_in = compressed.data();
  size_t avail_in = compressed.size();

  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSubscribeEvents(
                                 dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE));

  EXPECT_EQ(JXL_DEC_BASIC_INFO,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));
  size_t buffer_size;
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderImageOutBufferSize(dec, &pixel_format, &buffer_size));
  EXPECT_EQ(buffer_size, original_pixels.size());

  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec, &info));
  EXPECT_EQ(width, info.xsize);
  EXPECT_EQ(height, info.ysize);

  std::vector<uint8_t> decoded_pixels(buffer_size);

  EXPECT_EQ(JXL_DEC_NEED_IMAGE_OUT_BUFFER,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetImageOutBuffer(
                                 dec, &pixel_format, decoded_pixels.data(), decoded_pixels.size()));

  EXPECT_EQ(JXL_DEC_FULL_IMAGE,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  JxlDecoderDestroy(dec);

  jxl::CodecInOut decoded_io =
      jxl::test::SomeTestImageToCodecInOut(decoded_pixels, width, height);

  jxl::ButteraugliParams ba;
  EXPECT_LE(ButteraugliDistance(original_io, decoded_io, ba,
                                /*distmap=*/nullptr, nullptr),
            2.0f);
}
