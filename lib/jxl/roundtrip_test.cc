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

namespace {

// Returns whether the JxlEndianness value indicates little endian. If not,
// then big endian is assumed.
bool IsLittleEndian(const JxlEndianness& endianness) {
  switch (endianness) {
    case JXL_LITTLE_ENDIAN:
      return true;
    case JXL_BIG_ENDIAN:
      return false;
    case JXL_NATIVE_ENDIAN: {
      // JXL_BYTE_ORDER_LITTLE from byte_order.h cannot be used because it only
      // distinguishes between little endian and unknown.
      uint32_t u = 1;
      char c[4];
      memcpy(c, &u, 4);
      return c[0] == 1;
    }
  }

  JXL_ASSERT(false);
  return false;
}

// Returns a 3 channel, 32 bit float, native endian image.
std::vector<float> GetFloatTestImage(size_t xsize, size_t ysize) {
  size_t num_pixels = xsize * ysize;
  // One float per channel, 3 channels
  std::vector<float> pixels(num_pixels * 3);
  // Create pixel content to test, actual content does not matter as long as it
  // can be compared after roundtrip.
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      float r = static_cast<float>(y) / static_cast<float>(ysize);
      float g = static_cast<float>(x) / static_cast<float>(xsize);
      float b = static_cast<float>(x + y) / static_cast<float>(xsize + ysize);
      size_t i = (y * xsize + x) * 3;
      pixels[i] = r;
      pixels[i + 1] = g;
      pixels[i + 2] = b;
    }
  }
  return pixels;
}

// Compresses some pixels using some frame_format, verifying that the decoded
// version is similar to some original CodecInOut.
void VerifyRoundtripCompression(
    const std::vector<uint8_t>& original_bytes,
    const JxlFrameFormat& original_frame_format,
    const std::function<jxl::CodecInOut(const std::vector<uint8_t>&)>&
        converter) {
  jxl::CodecInOut original_io = converter(original_bytes);

  JxlEncoder* enc = JxlEncoderCreate(nullptr);
  EXPECT_NE(nullptr, enc);

  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddImageFrame(enc, &original_frame_format,
                                    (void*)original_bytes.data(),
                                    original_bytes.size()));
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
            JxlDecoderImageOutBufferSize(
                dec, &original_frame_format.pixel_format, &buffer_size));
  EXPECT_EQ(buffer_size, original_bytes.size());

  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec, &info));
  EXPECT_EQ(original_frame_format.xsize, info.xsize);
  EXPECT_EQ(original_frame_format.ysize, info.ysize);

  std::vector<uint8_t> decoded_bytes(buffer_size);

  EXPECT_EQ(JXL_DEC_NEED_IMAGE_OUT_BUFFER,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetImageOutBuffer(
                                 dec, &original_frame_format.pixel_format,
                                 decoded_bytes.data(), decoded_bytes.size()));

  EXPECT_EQ(JXL_DEC_FULL_IMAGE,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  JxlDecoderDestroy(dec);

  jxl::CodecInOut decoded_io = converter(decoded_bytes);

  jxl::ButteraugliParams ba;
  float butteraugli_score = ButteraugliDistance(original_io, decoded_io, ba,
                                                /*distmap=*/nullptr, nullptr);
  EXPECT_LE(butteraugli_score, 2.0f);
}

}  // namespace

TEST(RoundtripTest, FloatFrameRoundtripTest) {
  JxlPixelFormat pixel_format = {3, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
  uint32_t xsize = 63;
  uint32_t ysize = 129;
  JxlFrameFormat original_frame_format =
      JxlFrameFormat{pixel_format, xsize, ysize};
  std::vector<float> original_pixels = GetFloatTestImage(xsize, ysize);
  std::vector<uint8_t> original_bytes(original_pixels.size() * sizeof(float));
  memcpy(original_bytes.data(), original_pixels.data(),
         sizeof(float) * original_pixels.size());

  VerifyRoundtripCompression(
      original_bytes, original_frame_format,
      [xsize, ysize](const std::vector<uint8_t>& buf) {
        jxl::CodecInOut io;
        io.SetSize(xsize, ysize);
        EXPECT_TRUE(ConvertImage(
            jxl::Span<const uint8_t>(buf.data(), buf.size()), xsize, ysize,
            jxl::ColorEncoding::SRGB(/*is_gray=*/false),
            /*has_alpha=*/false, /*alpha_is_premultiplied=*/false,
            /*bits_per_alpha=*/0, /*bitdepth=*/32,
            /*big_endian=*/!IsLittleEndian(JXL_NATIVE_ENDIAN),
            /*flipped_y=*/false, /*pool=*/nullptr,
            /*ib=*/&io.Main()));
        return io;
      });
}

TEST(RoundtripTest, Uint16FrameRoundtripTest) {
  uint32_t channels = 4;
  JxlPixelFormat pixel_format = {channels, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};
  uint32_t xsize = 63;
  uint32_t ysize = 129;
  JxlFrameFormat original_frame_format =
      JxlFrameFormat{pixel_format, xsize, ysize};
  std::vector<uint8_t> original_bytes =
      jxl::test::GetSomeTestImage(xsize, ysize, channels);

  VerifyRoundtripCompression(original_bytes, original_frame_format,
                             [xsize, ysize](const std::vector<uint8_t>& buf) {
                               return jxl::test::SomeTestImageToCodecInOut(
                                   buf, xsize, ysize);
                             });
}
