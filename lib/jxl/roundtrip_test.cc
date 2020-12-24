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

// Converts a test image to a CodecInOut.
jxl::CodecInOut ConvertTestImage(const std::vector<uint8_t>& buf,
                                 const size_t xsize, const size_t ysize,
                                 const JxlPixelFormat& pixel_format) {
  jxl::CodecInOut io;
  io.SetSize(xsize, ysize);
  io.metadata.m.color_encoding.SetColorSpace(
      pixel_format.num_channels == 1 || pixel_format.num_channels == 2
          ? jxl::ColorSpace::kGray
          : jxl::ColorSpace::kRGB);
  if (pixel_format.num_channels == 2 || pixel_format.num_channels == 4) {
    // Note: alpha > 16 not yet supported by the C++ codec
    io.metadata.m.SetAlphaBits(16);
  }
  size_t bitdepth = 0;
  switch (pixel_format.data_type) {
    case JXL_TYPE_FLOAT:
      bitdepth = 32;
      io.metadata.m.SetFloat32Samples();
      break;
    case JXL_TYPE_UINT8:
      bitdepth = 8;
      io.metadata.m.SetUintSamples(8);
      break;
    case JXL_TYPE_UINT16:
      bitdepth = 16;
      io.metadata.m.SetUintSamples(16);
      break;
    default:
      EXPECT_TRUE(false) << "Roundtrip tests for data type "
                         << pixel_format.data_type << " not yet implemented.";
  }
  EXPECT_TRUE(ConvertImage(
      jxl::Span<const uint8_t>(buf.data(), buf.size()), xsize, ysize,
      /*c_current=*/pixel_format.data_type == JXL_TYPE_FLOAT
          ? jxl::ColorEncoding::LinearSRGB(
                /*is_gray=*/pixel_format.num_channels < 3)
          : jxl::ColorEncoding::SRGB(
                /*is_gray=*/pixel_format.num_channels < 3),
      /*has_alpha=*/pixel_format.num_channels == 2 ||
          pixel_format.num_channels == 4,
      /*alpha_is_premultiplied=*/false, /*bitdepth=*/bitdepth,
      pixel_format.endianness, /*flipped_y=*/false, /*pool=*/nullptr,
      /*ib=*/&io.Main()));
  return io;
}

// Stores a float in big endian
void StoreBEFloat(float value, uint8_t* p) {
  uint32_t u;
  memcpy(&u, &value, 4);
  StoreBE32(u, p);
}

// Stores a float in little endian
void StoreLEFloat(float value, uint8_t* p) {
  uint32_t u;
  memcpy(&u, &value, 4);
  StoreLE32(u, p);
}

// Loads a float in big endian
float LoadBEFloat(const uint8_t* p) {
  float value;
  const uint32_t u = LoadBE32(p);
  memcpy(&value, &u, 4);
  return value;
}

// Loads a float in little endian
float LoadLEFloat(const uint8_t* p) {
  float value;
  const uint32_t u = LoadLE32(p);
  memcpy(&value, &u, 4);
  return value;
}

template <typename T>
T ConvertTestPixel(const float val);

template <>
float ConvertTestPixel<float>(const float val) {
  return val;
}

template <>
uint16_t ConvertTestPixel<uint16_t>(const float val) {
  return (uint16_t)(val * UINT16_MAX);
}

template <>
uint8_t ConvertTestPixel<uint8_t>(const float val) {
  return (uint8_t)(val * UINT8_MAX);
}

// Returns a test image.
template <typename T>
std::vector<uint8_t> GetTestImage(const size_t xsize, const size_t ysize,
                                  const JxlPixelFormat& pixel_format) {
  std::vector<T> pixels(xsize * ysize * pixel_format.num_channels);
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      for (size_t chan = 0; chan < pixel_format.num_channels; chan++) {
        float val;
        switch (chan % 4) {
          case 0:
            val = static_cast<float>(y) / static_cast<float>(ysize);
            break;
          case 1:
            val = static_cast<float>(x) / static_cast<float>(xsize);
            break;
          case 2:
            val = static_cast<float>(x + y) / static_cast<float>(xsize + ysize);
            break;
          case 3:
            val = static_cast<float>(x * y) / static_cast<float>(xsize * ysize);
            break;
        }
        pixels[(y * xsize + x) * pixel_format.num_channels + chan] =
            ConvertTestPixel<T>(val);
      }
    }
    std::vector<uint8_t> bytes(pixels.size() * sizeof(T));
    memcpy(bytes.data(), pixels.data(), sizeof(T) * pixels.size());
    return bytes;
  }
  return {};
}

// Generates some pixels using using some dimensions and pixel_format,
// compresses them, and verifies that the decoded version is similar to the
// original pixels.
template <typename T>
void VerifyRoundtripCompression(const size_t xsize, const size_t ysize,
                                const JxlPixelFormat& pixel_format,
                                const bool lossless) {
  const std::vector<uint8_t> original_bytes =
      GetTestImage<T>(xsize, ysize, pixel_format);
  jxl::CodecInOut original_io =
      ConvertTestImage(original_bytes, xsize, ysize, pixel_format);

  JxlEncoder* enc = JxlEncoderCreate(nullptr);
  EXPECT_NE(nullptr, enc);

  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetDimensions(enc, xsize, ysize));
  JxlEncoderOptions* opts = JxlEncoderOptionsCreate(enc, nullptr);
  JxlEncoderOptionsSetLossless(opts, lossless);
  EXPECT_EQ(
      JXL_ENC_SUCCESS,
      JxlEncoderAddImageFrame(opts, &pixel_format, (void*)original_bytes.data(),
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
            JxlDecoderImageOutBufferSize(dec, &pixel_format, &buffer_size));
  EXPECT_EQ(buffer_size, original_bytes.size());

  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec, &info));
  EXPECT_EQ(xsize, info.xsize);
  EXPECT_EQ(ysize, info.ysize);

  std::vector<uint8_t> decoded_bytes(buffer_size);

  EXPECT_EQ(JXL_DEC_NEED_IMAGE_OUT_BUFFER,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetImageOutBuffer(dec, &pixel_format,
                                                         decoded_bytes.data(),
                                                         decoded_bytes.size()));

  EXPECT_EQ(JXL_DEC_FULL_IMAGE,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  JxlDecoderDestroy(dec);

  jxl::CodecInOut decoded_io =
      ConvertTestImage(decoded_bytes, xsize, ysize, pixel_format);

  jxl::ButteraugliParams ba;
  float butteraugli_score = ButteraugliDistance(original_io, decoded_io, ba,
                                                /*distmap=*/nullptr, nullptr);
  if (lossless) {
    EXPECT_LE(butteraugli_score, 0.0f);
  } else {
    EXPECT_LE(butteraugli_score, 2.0f);
  }
}

}  // namespace

TEST(RoundtripTest, FloatFrameRoundtripTest) {
  // TODO(zond): Add a lossless test here as well.
  for (uint32_t num_channels = 1; num_channels < 5; num_channels++) {
    JxlPixelFormat pixel_format =
        JxlPixelFormat{num_channels, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN, 0};
    VerifyRoundtripCompression<float>(63, 129, pixel_format, false);
  }
}

TEST(RoundtripTest, Uint16FrameRoundtripTest) {
  for (int lossless = 0; lossless < 2; lossless++) {
    for (uint32_t num_channels = 1; num_channels < 5; num_channels++) {
      JxlPixelFormat pixel_format =
          JxlPixelFormat{num_channels, JXL_TYPE_UINT16, JXL_NATIVE_ENDIAN, 0};
      VerifyRoundtripCompression<uint16_t>(63, 129, pixel_format,
                                           (bool)lossless);
    }
  }
}

TEST(RoundtripTest, Uint8FrameRoundtripTest) {
  for (int lossless = 0; lossless < 2; lossless++) {
    for (uint32_t num_channels = 1; num_channels < 5; num_channels++) {
      JxlPixelFormat pixel_format =
          JxlPixelFormat{num_channels, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
      VerifyRoundtripCompression<uint8_t>(63, 129, pixel_format,
                                          (bool)lossless);
    }
  }
}
