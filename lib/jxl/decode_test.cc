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

#include <stdint.h>
#include <stdlib.h>

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "jxl/thread_parallel_runner.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/enc_gamma_correct.h"
#include "lib/jxl/external_image.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/icc_codec.h"
#include "lib/jxl/test_utils.h"

////////////////////////////////////////////////////////////////////////////////

namespace jxl {
namespace {
// Input pixels always given as 16-bit RGBA, 8 bytes per pixel.
// include_alpha determines if the encoded image should contain the alpha
// channel.
PaddedBytes CreateTestJXLCodestream(Span<const uint8_t> pixels, size_t xsize,
                                    size_t ysize, size_t num_channels,
                                    const CompressParams& cparams) {
  // Compress the pixels with JPEG XL.
  bool grayscale = (num_channels <= 2);
  bool include_alpha = !(num_channels & 1);
  // Always add for now, tests extra non-full displayed frames.
  bool add_preview = true;
  size_t bitdepth = 16;
  CodecInOut io;
  io.SetSize(xsize, ysize);
  ColorEncoding color_encoding = jxl::test::ColorEncodingFromDescriptor(
      {grayscale ? ColorSpace::kGray : ColorSpace::kRGB, WhitePoint::kD65,
       Primaries::kSRGB, TransferFunction::kSRGB, RenderingIntent::kRelative});
  ThreadPool pool(nullptr, nullptr);
  io.metadata.m.SetUintSamples(bitdepth);
  if (include_alpha) {
    io.metadata.m.SetAlphaBits(bitdepth);
  }
  // Make the grayscale-ness of the io metadata color_encoding and the packed
  // image match.
  io.metadata.m.color_encoding = color_encoding;
  EXPECT_TRUE(ConvertImage(
      pixels, xsize, ysize, color_encoding, /*has_alpha=*/include_alpha,
      /*alpha_is_premultiplied=*/false, include_alpha ? bitdepth : 0, bitdepth,
      /*big_endian=*/true, /*flipped_y=*/false, &pool, &io.Main()));
  if (add_preview) {
    io.preview_frame = io.Main().Copy();
    io.preview_frame.ShrinkTo(xsize / 16, ysize / 16);
    io.metadata.m.have_preview = true;
    EXPECT_TRUE(io.metadata.m.preview_size.Set(io.preview_frame.xsize(),
                                               io.preview_frame.ysize()));
  }
  AuxOut aux_out;
  PaddedBytes compressed;
  PassesEncoderState enc_state;
  EXPECT_TRUE(
      EncodeFile(cparams, &io, &enc_state, &compressed, &aux_out, &pool));
  return compressed;
}

std::vector<uint8_t> DecodeWithAPI(Span<const uint8_t> compressed,
                                   const JxlPixelFormat& format) {
  // Test decoding with the API.

  JxlDecoder* dec = JxlDecoderCreate(NULL);
  const uint8_t* next_in = compressed.data();
  size_t avail_in = compressed.size();

  void* runner = JxlThreadParallelRunnerCreate(
      NULL, JxlThreadParallelRunnerDefaultNumWorkerThreads());
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetParallelRunner(dec, JxlThreadParallelRunner, runner));

  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSubscribeEvents(
                                 dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE));

  EXPECT_EQ(JXL_DEC_BASIC_INFO,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));
  size_t buffer_size;
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderImageOutBufferSize(dec, &format, &buffer_size));
  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec, &info));
  std::vector<uint8_t> pixels(buffer_size);

  EXPECT_EQ(JXL_DEC_NEED_IMAGE_OUT_BUFFER,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetImageOutBuffer(
                                 dec, &format, pixels.data(), pixels.size()));

  EXPECT_EQ(JXL_DEC_FULL_IMAGE,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  JxlThreadParallelRunnerDestroy(runner);
  JxlDecoderDestroy(dec);

  return pixels;
}

}  // namespace
}  // namespace jxl

namespace {
bool Near(double expected, double value, double max_dist) {
  double dist = expected > value ? expected - value : value - expected;
  return dist <= max_dist;
}

// Loads a Big-Endian float
// TODO(lode): support little endian here and in the API
float LoadBEFloat(const uint8_t* p) {
  uint32_t u = LoadBE32(p);
  float result;
  memcpy(&result, &u, 4);
  return result;
}

// Loads a Little-Endian float
float LoadLEFloat(const uint8_t* p) {
  uint32_t u = LoadLE32(p);
  float result;
  memcpy(&result, &u, 4);
  return result;
}

size_t GetPrecision(JxlDataType data_type) {
  switch (data_type) {
    case JXL_TYPE_BOOLEAN:
      return 1;
    case JXL_TYPE_UINT8:
      return 8;
    case JXL_TYPE_UINT16:
      return 16;
    case JXL_TYPE_UINT32:
      return 32;
    case JXL_TYPE_FLOAT:
      // Floating point mantissa precision
      return 24;
    default:
      JXL_ASSERT(false);  // unknown type
  }
}

// Procedure to convert pixels to double precision, not efficient, but
// well-controlled for testing. It uses double, to be able to represent all
// precisions needed for the maximum data types the API supports: uint32_t
// integers, and, single precision float. The values are in range 0-255 for
// SDR.
std::vector<double> ConvertToRGBA32(const uint8_t* pixels, size_t xsize,
                                    size_t ysize,
                                    const JxlPixelFormat& format) {
  std::vector<double> result(xsize * ysize * 4);
  size_t num_channels = format.num_channels;
  bool gray = num_channels == 1 || num_channels == 2;
  bool alpha = num_channels == 2 || num_channels == 4;
  if (format.data_type == JXL_TYPE_BOOLEAN) {
    size_t row_size = (xsize * num_channels + 7) >> 3;
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t j = (y * xsize + x) * 4;
        size_t i = y * row_size + ((x * num_channels + 7) >> 3);
        uint8_t byte = pixels[i];
        size_t bit = (x * num_channels) & 7;
        if (bit != 0) byte >>= (8 - bit);
        double r = (byte & 1);
        double g = gray ? r : ((byte & 2) >> 1);
        double b = gray ? r : ((byte & 4) >> 2);
        double a = alpha ? ((byte >> (num_channels - 1)) & 1) : 1;
        result[j + 0] = r;
        result[j + 1] = g;
        result[j + 2] = b;
        result[j + 3] = a;
      }
    }
  } else if (format.data_type == JXL_TYPE_UINT8) {
    double mul = 1.0 / 255.0;  // Multiplier to bring to 0-1.0 range
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t j = (y * xsize + x) * 4;
        size_t i = (y * xsize + x) * num_channels;
        double r = pixels[i];
        double g = gray ? r : pixels[i + 1];
        double b = gray ? r : pixels[i + 2];
        double a = alpha ? pixels[i + num_channels - 1] : 255;
        result[j + 0] = r * mul;
        result[j + 1] = g * mul;
        result[j + 2] = b * mul;
        result[j + 3] = a * mul;
      }
    }
  } else if (format.data_type == JXL_TYPE_UINT16) {
    double mul = 1.0 / 65535.0;  // Multiplier to bring to 0-1.0 range
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t j = (y * xsize + x) * 4;
        size_t i = (y * xsize + x) * num_channels * 2;
        double r, g, b, a;
        if (format.endianness == JXL_BIG_ENDIAN) {
          r = (pixels[i + 0] << 8) + pixels[i + 1];
          g = gray ? r : (pixels[i + 2] << 8) + pixels[i + 3];
          b = gray ? r : (pixels[i + 4] << 8) + pixels[i + 5];
          a = alpha ? (pixels[i + num_channels * 2 - 2] << 8) +
                          pixels[i + num_channels * 2 - 1]
                    : 65535;
        } else {
          r = (pixels[i + 1] << 8) + pixels[i + 0];
          g = gray ? r : (pixels[i + 3] << 8) + pixels[i + 2];
          b = gray ? r : (pixels[i + 5] << 8) + pixels[i + 4];
          a = alpha ? (pixels[i + num_channels * 2 - 1] << 8) +
                          pixels[i + num_channels * 2 - 2]
                    : 65535;
        }
        result[j + 0] = r * mul;
        result[j + 1] = g * mul;
        result[j + 2] = b * mul;
        result[j + 3] = a * mul;
      }
    }
  } else if (format.data_type == JXL_TYPE_UINT32) {
    double mul = 1.0 / 4294967295.0;  // Multiplier to bring to 0-1.0 range
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t j = (y * xsize + x) * 4;
        size_t i = (y * xsize + x) * num_channels * 4;
        double r, g, b, a;
        if (format.endianness == JXL_BIG_ENDIAN) {
          r = LoadBE32(pixels + i);
          g = gray ? r : LoadBE32(pixels + i + 4);
          b = gray ? r : LoadBE32(pixels + i + 8);
          a = alpha ? LoadBE32(pixels + i + num_channels * 2 - 4) : 4294967295;

        } else {
          r = LoadLE32(pixels + i);
          g = gray ? r : LoadLE32(pixels + i + 4);
          b = gray ? r : LoadLE32(pixels + i + 8);
          a = alpha ? LoadLE32(pixels + i + num_channels * 2 - 4) : 4294967295;
        }
        result[j + 0] = r * mul;
        result[j + 1] = g * mul;
        result[j + 2] = b * mul;
        result[j + 3] = a * mul;
      }
    }
  } else if (format.data_type == JXL_TYPE_FLOAT) {
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t j = (y * xsize + x) * 4;
        size_t i = (y * xsize + x) * num_channels * 4;
        double r, g, b, a;
        if (format.endianness == JXL_BIG_ENDIAN) {
          r = LoadBEFloat(pixels + i);
          g = gray ? r : LoadBEFloat(pixels + i + 4);
          b = gray ? r : LoadBEFloat(pixels + i + 8);
          a = alpha ? LoadBEFloat(pixels + i + num_channels * 4 - 4) : 1.0;
        } else {
          r = LoadLEFloat(pixels + i);
          g = gray ? r : LoadLEFloat(pixels + i + 4);
          b = gray ? r : LoadLEFloat(pixels + i + 8);
          a = alpha ? LoadLEFloat(pixels + i + num_channels * 4 - 4) : 1.0;
        }
        result[j + 0] = r;
        result[j + 1] = g;
        result[j + 2] = b;
        result[j + 3] = a;
      }
    }
  } else {
    JXL_ASSERT(false);  // Unsupported type
  }
  return result;
}

// Returns amount of pixels which differ between the two pictures. Image b is
// the image after roundtrip after roundtrip, image a before roundtrip. There
// are more strict requirements for the alpha channel and grayscale values of
// the output image.
size_t ComparePixels(const uint8_t* a, const uint8_t* b, size_t xsize,
                     size_t ysize, const JxlPixelFormat& format_a,
                     const JxlPixelFormat& format_b) {
  // Convert both images to equal full precision for comparison.
  std::vector<double> a_full = ConvertToRGBA32(a, xsize, ysize, format_a);
  std::vector<double> b_full = ConvertToRGBA32(b, xsize, ysize, format_b);
  bool gray_a = format_a.num_channels < 3;
  bool gray_b = format_b.num_channels < 3;
  bool alpha_a = !(format_a.num_channels & 1);
  bool alpha_b = !(format_b.num_channels & 1);
  size_t bits_a = GetPrecision(format_a.data_type);
  size_t bits_b = GetPrecision(format_b.data_type);
  size_t bits = std::min(bits_a, bits_b);
  // How much distance is allowed in case of pixels with lower bit depths, given
  // that the double precision float images use range 0-1.0.
  // E.g. in case of 1-bit this is 0.5 since 0.499 must map to 0 and 0.501 must
  // map to 1.
  double precision = 0.5 / ((1ull << bits) - 1ull);
  size_t numdiff = 0;
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      size_t i = (y * xsize + x) * 4;
      bool ok = true;
      if (gray_a || gray_b) {
        if (!Near(a_full[i + 0], b_full[i + 0], precision)) ok = false;
        // If the input was grayscale and the output not, then the output must
        // have all channels equal.
        if (gray_a && b_full[i + 0] != b_full[i + 1] &&
            b_full[i + 2] != b_full[i + 2]) {
          ok = false;
        }
      } else {
        if (!Near(a_full[i + 0], b_full[i + 0], precision) ||
            !Near(a_full[i + 1], b_full[i + 1], precision) ||
            !Near(a_full[i + 2], b_full[i + 2], precision)) {
          ok = false;
        }
      }
      if (alpha_a && alpha_b) {
        if (!Near(a_full[i + 3], b_full[i + 3], precision)) ok = false;
      } else {
        // If the input had no alpha channel, the output should be opaque
        // after roundtrip.
        if (alpha_b && !Near(1.0, b_full[i + 3], precision)) ok = false;
      }
      if (!ok) numdiff++;
    }
  }
  return numdiff;
}

}  // namespace

////////////////////////////////////////////////////////////////////////////////

TEST(DecodeTest, JxlSignatureCheckTest) {
  std::vector<std::pair<int, std::vector<uint8_t>>> tests = {
      // No JPEGXL header starts with 'a'.
      {JXL_SIG_INVALID, {'a'}},
      {JXL_SIG_INVALID, {'a', 'b', 'c', 'd', 'e', 'f'}},

      // Empty file is not enough bytes.
      {JXL_SIG_NOT_ENOUGH_BYTES, {}},

      // JPEGXL headers.
      {JXL_SIG_NOT_ENOUGH_BYTES, {0xff}},  // Part of a signature.
      {JXL_SIG_INVALID, {0xff, 0xD8}},     // JPEG-1
      {JXL_SIG_CODESTREAM, {0xff, 0x0a}},

      // JPEGXL container file.
      {JXL_SIG_CONTAINER,
       {0, 0, 0, 0xc, 'J', 'X', 'L', ' ', 0xD, 0xA, 0x87, 0xA}},
      // Ending with invalid byte.
      {JXL_SIG_INVALID, {0, 0, 0, 0xc, 'J', 'X', 'L', ' ', 0xD, 0xA, 0x87, 0}},
      // Part of signature.
      {JXL_SIG_NOT_ENOUGH_BYTES,
       {0, 0, 0, 0xc, 'J', 'X', 'L', ' ', 0xD, 0xA, 0x87}},
      {JXL_SIG_NOT_ENOUGH_BYTES, {0}},
  };
  for (const auto& test : tests) {
    EXPECT_EQ(test.first,
              JxlSignatureCheck(test.second.data(), test.second.size()))
        << "Where test data is " << ::testing::PrintToString(test.second);
  }
}

TEST(DecodeTest, DefaultAllocTest) {
  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  EXPECT_NE(nullptr, dec);
  JxlDecoderDestroy(dec);
}

TEST(DecodeTest, CustomAllocTest) {
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

  JxlDecoder* dec = JxlDecoderCreate(&mm);
  EXPECT_NE(nullptr, dec);
  EXPECT_LE(1, counters.allocs);
  EXPECT_EQ(0, counters.frees);
  JxlDecoderDestroy(dec);
  EXPECT_LE(1, counters.frees);
}

// TODO(lode): add multi-threaded test when multithreaded pixel decoding from
// API is implemented.
TEST(DecodeTest, DefaultParallelRunnerTest) {
  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  EXPECT_NE(nullptr, dec);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetParallelRunner(dec, nullptr, nullptr));
  JxlDecoderDestroy(dec);
}

// Creates the header of a JPEG XL file with various custom parameters for
// testing.
// xsize, ysize: image dimentions to store in the SizeHeader, max 512.
// bits_per_sample, orientation: a selection of header parameters to test with.
// orientation: image orientation to set in the metadata
// alpha_bits: if non-0, alpha extra channel bits to set in the metadata. Also
//   gives the alpha channel the name "alpha_test"
// have_container: add box container format around the codestream.
// metadata_default: if true, ImageMetadata is set to default and
//   bits_per_sample, orientation and alpha_bits are ignored.
// insert_box: insert an extra box before the codestream box, making the header
// farther away from the front than is ideal. Only used if have_container.
std::vector<uint8_t> GetTestHeader(size_t xsize, size_t ysize,
                                   size_t bits_per_sample, size_t orientation,
                                   size_t alpha_bits, bool xyb_encoded,
                                   bool have_container, bool metadata_default,
                                   bool insert_extra_box,
                                   const std::vector<uint8_t>& icc_profile) {
  jxl::BitWriter writer;
  jxl::BitWriter::Allotment allotment(&writer, 65536);  // Large enough

  if (have_container) {
    const std::vector<uint8_t> signature_box = {0,   0,   0,   0xc, 'J',  'X',
                                                'L', ' ', 0xd, 0xa, 0x87, 0xa};
    const std::vector<uint8_t> filetype_box = {
        0,   0,   0, 0x14, 'f', 't', 'y', 'p', 'j', 'x',
        'l', ' ', 0, 0,    0,   0,   'j', 'x', 'l', ' '};
    const std::vector<uint8_t> extra_box_header = {0,   0,   0,   0xff,
                                                   't', 'e', 's', 't'};
    // Beginning of codestream box, with an arbitrary size certainly large
    // enough to contain the header
    const std::vector<uint8_t> codestream_box_header = {0,   0,   0,   0xff,
                                                        'j', 'x', 'l', 'c'};

    for (size_t i = 0; i < signature_box.size(); i++) {
      writer.Write(8, signature_box[i]);
    }
    for (size_t i = 0; i < filetype_box.size(); i++) {
      writer.Write(8, filetype_box[i]);
    }
    if (insert_extra_box) {
      for (size_t i = 0; i < extra_box_header.size(); i++) {
        writer.Write(8, extra_box_header[i]);
      }
      for (size_t i = 0; i < 255 - 8; i++) {
        writer.Write(8, 0);
      }
    }
    for (size_t i = 0; i < codestream_box_header.size(); i++) {
      writer.Write(8, codestream_box_header[i]);
    }
  }

  // JXL signature
  writer.Write(8, 0xff);
  writer.Write(8, 0x0a);

  // SizeHeader
  jxl::CodecMetadata metadata;
  EXPECT_TRUE(metadata.size.Set(xsize, ysize));
  EXPECT_TRUE(WriteSizeHeader(metadata.size, &writer, 0, nullptr));

  if (!metadata_default) {
    metadata.m.SetUintSamples(bits_per_sample);
    metadata.m.orientation = orientation;
    metadata.m.SetAlphaBits(alpha_bits);
    metadata.m.xyb_encoded = xyb_encoded;
    if (alpha_bits != 0) {
      metadata.m.extra_channel_info[0].name = "alpha_test";
    }
  }

  jxl::PaddedBytes icc_padded(icc_profile.size());
  memcpy(icc_padded.data(), icc_profile.data(), icc_profile.size());

  if (!icc_profile.empty()) {
    jxl::PaddedBytes copy = icc_padded;
    EXPECT_TRUE(metadata.m.color_encoding.SetICC(std::move(copy)));
  }

  EXPECT_TRUE(jxl::Bundle::Write(metadata.m, &writer, 0, nullptr));
  metadata.transform_data.nonserialized_xyb_encoded = metadata.m.xyb_encoded;
  EXPECT_TRUE(jxl::Bundle::Write(metadata.transform_data, &writer, 0, nullptr));

  if (!icc_profile.empty()) {
    EXPECT_TRUE(metadata.m.color_encoding.WantICC());
    EXPECT_TRUE(jxl::WriteICC(icc_padded, &writer, 0, nullptr));
  }

  writer.ZeroPadToByte();
  ReclaimAndCharge(&writer, &allotment, 0, nullptr);
  return std::vector<uint8_t>(
      writer.GetSpan().data(),
      writer.GetSpan().data() + writer.GetSpan().size());
}

TEST(DecodeTest, BasicInfoTest) {
  size_t xsize[2] = {50, 33};
  size_t ysize[2] = {50, 77};
  size_t bits_per_sample[2] = {8, 23};
  size_t orientation[2] = {3, 5};
  size_t alpha_bits[2] = {0, 8};
  size_t have_container[2] = {0, 1};
  bool xyb_encoded = false;

  std::vector<std::vector<uint8_t>> test_samples;
  // Test with direct codestream
  test_samples.push_back(GetTestHeader(
      xsize[0], ysize[0], bits_per_sample[0], orientation[0], alpha_bits[0],
      xyb_encoded, have_container[0], /*metadata_default=*/false,
      /*insert_extra_box=*/false, {}));
  // Test with container and different parameters
  test_samples.push_back(GetTestHeader(
      xsize[1], ysize[1], bits_per_sample[1], orientation[1], alpha_bits[1],
      xyb_encoded, have_container[1], /*metadata_default=*/false,
      /*insert_extra_box=*/false, {}));

  for (size_t i = 0; i < test_samples.size(); ++i) {
    const std::vector<uint8_t>& data = test_samples[i];
    // Test decoding too small header first, until we reach the final byte.
    for (size_t size = 0; size <= data.size(); ++size) {
      // Test with a new decoder for each tested byte size.
      JxlDecoder* dec = JxlDecoderCreate(nullptr);
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO));
      const uint8_t* next_in = data.data();
      size_t avail_in = size;
      JxlDecoderStatus status =
          JxlDecoderProcessInput(dec, &next_in, &avail_in);

      JxlBasicInfo info;
      bool have_basic_info = !JxlDecoderGetBasicInfo(dec, &info);

      if (size == data.size()) {
        EXPECT_EQ(JXL_DEC_BASIC_INFO, status);

        // All header bytes given so the decoder must have the basic info.
        EXPECT_EQ(true, have_basic_info);
        EXPECT_EQ(have_container[i], info.have_container);
        EXPECT_EQ(alpha_bits[i], info.alpha_bits);
        // Orientations 5..8 swap the dimensions
        if (orientation[i] >= 5) {
          EXPECT_EQ(xsize[i], info.ysize);
          EXPECT_EQ(ysize[i], info.xsize);
        } else {
          EXPECT_EQ(xsize[i], info.xsize);
          EXPECT_EQ(ysize[i], info.ysize);
        }
        // The API should set the orientation to identity by default since it
        // already applies the transformation internally by default.
        EXPECT_EQ(1, info.orientation);

        if (alpha_bits[i] != 0) {
          // Expect an extra channel
          EXPECT_EQ(1, info.num_extra_channels);
          JxlExtraChannelInfo extra;
          EXPECT_EQ(0, JxlDecoderGetExtraChannelInfo(dec, 0, &extra));
          EXPECT_EQ(alpha_bits[i], extra.bits_per_sample);
          EXPECT_EQ(JXL_CHANNEL_ALPHA, extra.type);
          EXPECT_EQ(0, extra.alpha_associated);
          // Verify the name "alpha_test" given to the alpha channel
          EXPECT_EQ(10, extra.name_length);
          char name[11];
          EXPECT_EQ(0,
                    JxlDecoderGetExtraChannelName(dec, 0, name, sizeof(name)));
          EXPECT_EQ(std::string("alpha_test"), std::string(name));
        } else {
          EXPECT_EQ(0, info.num_extra_channels);
        }

        EXPECT_EQ(JXL_DEC_SUCCESS,
                  JxlDecoderProcessInput(dec, &next_in, &avail_in));
      } else {
        // If we did not give the full header, the basic info should not be
        // available. Allow a few bytes of slack due to some bits for default
        // opsinmatrix/extension bits.
        if (size + 2 < data.size()) {
          EXPECT_EQ(false, have_basic_info);
          EXPECT_EQ(JXL_DEC_NEED_MORE_INPUT, status);
        }
      }

      // Test that decoder doesn't allow setting a setting required at beginning
      // unless it's reset
      EXPECT_EQ(JXL_DEC_ERROR,
                JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO));
      JxlDecoderReset(dec);
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO));

      JxlDecoderDestroy(dec);
    }
  }
}

TEST(DecodeTest, BufferSizeTest) {
  size_t xsize = 33;
  size_t ysize = 77;
  size_t bits_per_sample = 8;
  size_t orientation = 1;
  size_t alpha_bits = 8;
  bool have_container = false;
  bool xyb_encoded = false;

  std::vector<uint8_t> header =
      GetTestHeader(xsize, ysize, bits_per_sample, orientation, alpha_bits,
                    xyb_encoded, have_container, /*metadata_default=*/false,
                    /*insert_extra_box=*/false, {});

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO));
  const uint8_t* next_in = header.data();
  size_t avail_in = header.size();
  JxlDecoderStatus status = JxlDecoderProcessInput(dec, &next_in, &avail_in);
  EXPECT_EQ(JXL_DEC_BASIC_INFO, status);

  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec, &info));
  EXPECT_EQ(xsize, info.xsize);
  EXPECT_EQ(ysize, info.ysize);

  JxlPixelFormat format = {4, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};
  size_t image_out_size;
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderImageOutBufferSize(dec, &format, &image_out_size));
  EXPECT_EQ(xsize * ysize * 4, image_out_size);

  size_t dc_out_size;
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderDCOutBufferSize(dec, &format, &dc_out_size));
  // expected dc size: ceil(33 / 8) * ceil(77 / 8) * 4 channels
  EXPECT_EQ(5 * 10 * 4, dc_out_size);

  JxlDecoderDestroy(dec);
}

TEST(DecodeTest, BasicInfoSizeHintTest) {
  // Test on a file where the size hint is too small initially due to inserting
  // a box before the codestream (something that is normally not recommended)
  size_t xsize = 50;
  size_t ysize = 50;
  size_t bits_per_sample = 16;
  size_t orientation = 1;
  size_t alpha_bits = 0;
  bool xyb_encoded = false;
  std::vector<uint8_t> data = GetTestHeader(
      xsize, ysize, bits_per_sample, orientation, alpha_bits, xyb_encoded,
      /*have_container=*/true, /*metadata_default=*/false,
      /*insert_extra_box=*/true, {});

  JxlDecoderStatus status;
  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO));

  size_t hint0 = JxlDecoderSizeHintBasicInfo(dec);
  // Test that the test works as intended: we construct a file on purpose to
  // be larger than the first hint by having that extra box.
  EXPECT_LT(hint0, data.size());
  const uint8_t* next_in = data.data();
  // Do as if we have only as many bytes as indicated by the hint available
  size_t avail_in = std::min(hint0, data.size());
  status = JxlDecoderProcessInput(dec, &next_in, &avail_in);
  EXPECT_EQ(JXL_DEC_NEED_MORE_INPUT, status);
  // Basic info cannot be available yet due to the extra inserted box.
  EXPECT_EQ(false, !JxlDecoderGetBasicInfo(dec, nullptr));

  size_t num_read = next_in - data.data();
  EXPECT_LT(num_read, data.size());

  size_t hint1 = JxlDecoderSizeHintBasicInfo(dec);
  // The hint must be larger than the previouw hint (taking already processed
  // bytes into account, the hint is a hint for the next avail_in) since the
  // decoder now knows there is a box in between.
  EXPECT_GT(hint1 + num_read, hint0);
  avail_in = std::min<size_t>(hint1, data.size() - num_read);

  status = JxlDecoderProcessInput(dec, &next_in, &avail_in);
  EXPECT_EQ(JXL_DEC_BASIC_INFO, status);
  JxlBasicInfo info;
  // We should have the basic info now, since we only added one box in-between,
  // and the decoder should have known its size, its implementation can return
  // a correct hint.
  EXPECT_EQ(true, !JxlDecoderGetBasicInfo(dec, &info));

  // Also test if the basic info is correct.
  EXPECT_EQ(1, info.have_container);
  EXPECT_EQ(xsize, info.xsize);
  EXPECT_EQ(ysize, info.ysize);
  EXPECT_EQ(orientation, info.orientation);
  EXPECT_EQ(bits_per_sample, info.bits_per_sample);

  JxlDecoderDestroy(dec);
}

// Returns an ICC profile output by the JPEG XL decoder for RGB_D65_SRG_Rel_Lin,
// but with, on purpose, rXYZ, bXYZ and gXYZ (the RGB primaries) switched to a
// different order to ensure the profile does not match any known profile, so
// the encoder cannot encode it in a compact struct instead.
std::vector<uint8_t> GetIccTestProfile() {
  const uint8_t* profile = reinterpret_cast<const uint8_t*>(
      "\0\0\3\200lcms\0040\0\0mntrRGB XYZ "
      "\a\344\0\a\0\27\0\21\0$"
      "\0\37acspAPPL\0\0\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\1\0\0\366"
      "\326\0\1\0\0\0\0\323-lcms\372c\207\36\227\200{"
      "\2\232s\255\327\340\0\n\26\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
      "\0\0\0\0\0\0\0\0\rdesc\0\0\1 "
      "\0\0\0Bcprt\0\0\1d\0\0\1\0wtpt\0\0\2d\0\0\0\24chad\0\0\2x\0\0\0,"
      "bXYZ\0\0\2\244\0\0\0\24gXYZ\0\0\2\270\0\0\0\24rXYZ\0\0\2\314\0\0\0\24rTR"
      "C\0\0\2\340\0\0\0 gTRC\0\0\2\340\0\0\0 bTRC\0\0\2\340\0\0\0 "
      "chrm\0\0\3\0\0\0\0$dmnd\0\0\3$\0\0\0("
      "dmdd\0\0\3L\0\0\0002mluc\0\0\0\0\0\0\0\1\0\0\0\fenUS\0\0\0&"
      "\0\0\0\34\0R\0G\0B\0_\0D\0006\0005\0_\0S\0R\0G\0_\0R\0e\0l\0_"
      "\0L\0i\0n\0\0mluc\0\0\0\0\0\0\0\1\0\0\0\fenUS\0\0\0\344\0\0\0\34\0C\0o\0"
      "p\0y\0r\0i\0g\0h\0t\0 \0002\0000\0001\08\0 \0G\0o\0o\0g\0l\0e\0 "
      "\0L\0L\0C\0,\0 \0C\0C\0-\0B\0Y\0-\0S\0A\0 \0003\0.\0000\0 "
      "\0U\0n\0p\0o\0r\0t\0e\0d\0 "
      "\0l\0i\0c\0e\0n\0s\0e\0(\0h\0t\0t\0p\0s\0:\0/\0/"
      "\0c\0r\0e\0a\0t\0i\0v\0e\0c\0o\0m\0m\0o\0n\0s\0.\0o\0r\0g\0/"
      "\0l\0i\0c\0e\0n\0s\0e\0s\0/\0b\0y\0-\0s\0a\0/\0003\0.\0000\0/"
      "\0l\0e\0g\0a\0l\0c\0o\0d\0e\0)XYZ "
      "\0\0\0\0\0\0\366\326\0\1\0\0\0\0\323-"
      "sf32\0\0\0\0\0\1\fB\0\0\5\336\377\377\363%"
      "\0\0\a\223\0\0\375\220\377\377\373\241\377\377\375\242\0\0\3\334\0\0\300"
      "nXYZ \0\0\0\0\0\0o\240\0\08\365\0\0\3\220XYZ "
      "\0\0\0\0\0\0$\237\0\0\17\204\0\0\266\304XYZ "
      "\0\0\0\0\0\0b\227\0\0\267\207\0\0\30\331para\0\0\0\0\0\3\0\0\0\1\0\0\0\1"
      "\0\0\0\0\0\0\0\1\0\0\0\0\0\0chrm\0\0\0\0\0\3\0\0\0\0\243\327\0\0T|"
      "\0\0L\315\0\0\231\232\0\0&"
      "g\0\0\17\\mluc\0\0\0\0\0\0\0\1\0\0\0\fenUS\0\0\0\f\0\0\0\34\0G\0o\0o\0g"
      "\0l\0emluc\0\0\0\0\0\0\0\1\0\0\0\fenUS\0\0\0\26\0\0\0\34\0I\0m\0a\0g\0e"
      "\0 \0c\0o\0d\0e\0c\0\0");
  size_t profile_size = 896;
  std::vector<uint8_t> icc_profile;
  icc_profile.assign(profile, profile + profile_size);
  return icc_profile;
}

std::vector<uint8_t> GetIccTestHeader(const std::vector<uint8_t>& icc_profile,
                                      bool xyb_encoded) {
  size_t xsize = 50;
  size_t ysize = 50;
  size_t bits_per_sample = 16;
  size_t orientation = 1;
  size_t alpha_bits = 0;
  return GetTestHeader(xsize, ysize, bits_per_sample, orientation, alpha_bits,
                       xyb_encoded,
                       /*have_container=*/false, /*metadata_default=*/false,
                       /*insert_extra_box=*/false, icc_profile);
}

// Tests the case where pixels and metadata ICC profile are the same
TEST(DecodeTest, IccProfileTestOriginal) {
  std::vector<uint8_t> icc_profile = GetIccTestProfile();
  bool xyb_encoded = false;
  std::vector<uint8_t> data = GetIccTestHeader(icc_profile, xyb_encoded);
  JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_LITTLE_ENDIAN, 0};

  const uint8_t* next_in = data.data();
  size_t avail_in = data.size();

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(
                dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING));

  EXPECT_EQ(JXL_DEC_BASIC_INFO,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  // Expect the opposite of xyb_encoded for uses_original_profile
  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec, &info));
  EXPECT_EQ(JXL_TRUE, info.uses_original_profile);

  EXPECT_EQ(JXL_DEC_COLOR_ENCODING,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  // the encoded color profile expected to be not available, since the image
  // has an ICC profile instead
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderGetColorAsEncodedProfile(
                dec, &format, JXL_COLOR_PROFILE_TARGET_ORIGINAL, nullptr));

  size_t dec_profile_size;
  EXPECT_EQ(
      JXL_DEC_SUCCESS,
      JxlDecoderGetICCProfileSize(
          dec, &format, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &dec_profile_size));

  // Check that can get return status with NULL size
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderGetICCProfileSize(
                dec, &format, JXL_COLOR_PROFILE_TARGET_ORIGINAL, nullptr));

  // The profiles must be equal. This requires they have equal size, and if
  // they do, we can get the profile and compare the contents.
  EXPECT_EQ(icc_profile.size(), dec_profile_size);
  if (icc_profile.size() == dec_profile_size) {
    std::vector<uint8_t> icc_profile2(icc_profile.size());
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderGetColorAsICCProfile(
                  dec, &format, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                  icc_profile2.data(), icc_profile2.size()));
    EXPECT_EQ(icc_profile, icc_profile2);
  }

  // the data is not xyb_encoded, so same result expected for the pixel data
  // color profile
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderGetColorAsEncodedProfile(
                dec, &format, JXL_COLOR_PROFILE_TARGET_DATA, nullptr));

  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetICCProfileSize(
                                 dec, &format, JXL_COLOR_PROFILE_TARGET_DATA,
                                 &dec_profile_size));
  EXPECT_EQ(icc_profile.size(), dec_profile_size);

  JxlDecoderDestroy(dec);
}

// Tests the case where pixels and metadata ICC profile are different
TEST(DecodeTest, IccProfileTestXybEncoded) {
  std::vector<uint8_t> icc_profile = GetIccTestProfile();
  bool xyb_encoded = true;
  std::vector<uint8_t> data = GetIccTestHeader(icc_profile, xyb_encoded);
  JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_LITTLE_ENDIAN, 0};
  JxlPixelFormat format_int = {4, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};

  const uint8_t* next_in = data.data();
  size_t avail_in = data.size();

  JxlDecoder* dec = JxlDecoderCreate(nullptr);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(
                dec, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING));

  EXPECT_EQ(JXL_DEC_BASIC_INFO,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  // Expect the opposite of xyb_encoded for uses_original_profile
  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec, &info));
  EXPECT_EQ(JXL_FALSE, info.uses_original_profile);

  EXPECT_EQ(JXL_DEC_COLOR_ENCODING,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  // the encoded color profile expected to be not available, since the image
  // has an ICC profile instead
  EXPECT_EQ(JXL_DEC_ERROR,
            JxlDecoderGetColorAsEncodedProfile(
                dec, &format, JXL_COLOR_PROFILE_TARGET_ORIGINAL, nullptr));

  // Check that can get return status with NULL size
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderGetICCProfileSize(
                dec, &format, JXL_COLOR_PROFILE_TARGET_ORIGINAL, nullptr));

  size_t dec_profile_size;
  EXPECT_EQ(
      JXL_DEC_SUCCESS,
      JxlDecoderGetICCProfileSize(
          dec, &format, JXL_COLOR_PROFILE_TARGET_ORIGINAL, &dec_profile_size));

  // The profiles must be equal. This requires they have equal size, and if
  // they do, we can get the profile and compare the contents.
  EXPECT_EQ(icc_profile.size(), dec_profile_size);
  if (icc_profile.size() == dec_profile_size) {
    std::vector<uint8_t> icc_profile2(icc_profile.size());
    EXPECT_EQ(JXL_DEC_SUCCESS,
              JxlDecoderGetColorAsICCProfile(
                  dec, &format, JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                  icc_profile2.data(), icc_profile2.size()));
    EXPECT_EQ(icc_profile, icc_profile2);
  }

  // Data is xyb_encoded, so the data profile is a different profile, encoded
  // as structured profile.
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderGetColorAsEncodedProfile(
                dec, &format, JXL_COLOR_PROFILE_TARGET_DATA, nullptr));
  JxlColorEncoding pixel_encoding;
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderGetColorAsEncodedProfile(
                dec, &format, JXL_COLOR_PROFILE_TARGET_DATA, &pixel_encoding));
  EXPECT_EQ(JXL_PRIMARIES_SRGB, pixel_encoding.primaries);
  // This is JXL_TRANSFER_FUNCTION_LINEAR because the format is float, for
  // uint8 and uint16 this must be JXL_TRANSFER_FUNCTION_SRGB instead.
  EXPECT_EQ(JXL_TRANSFER_FUNCTION_LINEAR, pixel_encoding.transfer_function);

  // Test the same but with integer format, which gives different transfer
  // function.
  EXPECT_EQ(
      JXL_DEC_SUCCESS,
      JxlDecoderGetColorAsEncodedProfile(
          dec, &format_int, JXL_COLOR_PROFILE_TARGET_DATA, &pixel_encoding));
  EXPECT_EQ(JXL_PRIMARIES_SRGB, pixel_encoding.primaries);
  EXPECT_EQ(JXL_TRANSFER_FUNCTION_SRGB, pixel_encoding.transfer_function);

  // The decoder can also output this as a generated ICC profile anyway, and
  // we're certain that it will differ from the above defined profile since
  // the sRGB data should not have swapped R/G/B primaries.

  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetICCProfileSize(
                                 dec, &format, JXL_COLOR_PROFILE_TARGET_DATA,
                                 &dec_profile_size));
  // We don't need to dictate exactly what size the generated ICC profile
  // must be (since there are many ways to represent the same color space),
  // but it should not be zero.
  EXPECT_NE(0, dec_profile_size);
  if (0 != dec_profile_size) {
    std::vector<uint8_t> icc_profile2(dec_profile_size);
    EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetColorAsICCProfile(
                                   dec, &format, JXL_COLOR_PROFILE_TARGET_DATA,
                                   icc_profile2.data(), icc_profile2.size()));
    // expected not equal
    EXPECT_NE(icc_profile, icc_profile2);
  }

  JxlDecoderDestroy(dec);
}

TEST(DecodeTest, PixelTest) {
  size_t xsize = 123, ysize = 77;
  size_t num_pixels = xsize * ysize;
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 4);
  JxlPixelFormat format_orig = {4, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};

  jxl::CompressParams cparams;
  cparams.SetLossless();  // Lossless to verify pixels exactly after roundtrip.
  jxl::PaddedBytes compressed = jxl::CreateTestJXLCodestream(
      jxl::Span<const uint8_t>(pixels.data(), pixels.size()), xsize, ysize, 4,
      cparams);

  for (int big_endian = 0; big_endian <= 1; ++big_endian) {
    JxlEndianness endianness = big_endian ? JXL_BIG_ENDIAN : JXL_LITTLE_ENDIAN;

    for (uint32_t channels = 3; channels <= 4; ++channels) {
      {
        JxlPixelFormat format = {channels, JXL_TYPE_UINT8, endianness, 0};

        std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
            jxl::Span<const uint8_t>(compressed.data(), compressed.size()),
            format);
        EXPECT_EQ(num_pixels * channels, pixels2.size());
        EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                                   format_orig, format));
      }

      {
        JxlPixelFormat format = {channels, JXL_TYPE_UINT16, endianness, 0};

        std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
            jxl::Span<const uint8_t>(compressed.data(), compressed.size()),
            format);
        EXPECT_EQ(num_pixels * channels * 2, pixels2.size());
        EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                                   format_orig, format));
      }

#if 0  // Disabled since external_image doesn't currently support uint32_t
      {
        JxlPixelFormat format = {channels, JXL_TYPE_UINT32, endianness, 0};

        std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
            jxl::Span<const uint8_t>(compressed.data(),
                compressed.size()), format);
        EXPECT_EQ(num_pixels * channels * 4, pixels2.size());
        EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                                  format_orig, format));
      }
#endif

      {
        JxlPixelFormat format = {channels, JXL_TYPE_FLOAT, endianness, 0};

        std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
            jxl::Span<const uint8_t>(compressed.data(), compressed.size()),
            format);
        EXPECT_EQ(num_pixels * channels * 4, pixels2.size());
        EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                                   format_orig, format));
      }
    }
  }
}

TEST(DecodeTest, GrayscaleTest) {
  size_t xsize = 123, ysize = 77;
  size_t num_pixels = xsize * ysize;
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 2);
  JxlPixelFormat format_orig = {2, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};

  jxl::CompressParams cparams;
  cparams.SetLossless();  // Lossless to verify pixels exactly after roundtrip.
  jxl::PaddedBytes compressed = jxl::CreateTestJXLCodestream(
      jxl::Span<const uint8_t>(pixels.data(), pixels.size()), xsize, ysize, 2,
      cparams);

  for (int big_endian = 0; big_endian <= 1; ++big_endian) {
    JxlEndianness endianness = big_endian ? JXL_BIG_ENDIAN : JXL_LITTLE_ENDIAN;
    // The compressed image is grayscale, but the output can be tested with
    // up to 4 channels (RGBA)
    for (uint32_t channels = 1; channels <= 4; ++channels) {
      {
        JxlPixelFormat format = {channels, JXL_TYPE_UINT8, endianness, 0};

        std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
            jxl::Span<const uint8_t>(compressed.data(), compressed.size()),
            format);
        EXPECT_EQ(num_pixels * channels, pixels2.size());
        EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                                   format_orig, format));
      }

      {
        JxlPixelFormat format = {channels, JXL_TYPE_UINT16, endianness, 0};

        std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
            jxl::Span<const uint8_t>(compressed.data(), compressed.size()),
            format);
        EXPECT_EQ(num_pixels * channels * 2, pixels2.size());
        EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                                   format_orig, format));
      }

#if 0  // Disabled since external_image doesn't currently support uint32_t
      {
        JxlPixelFormat format = {channels, JXL_TYPE_UINT32, endianness, 0};

        std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
            jxl::Span<const uint8_t>(compressed.data(),
                compressed.size()), format);
        EXPECT_EQ(num_pixels * channels * 4, pixels2.size());
        EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                                  format_orig, format));
      }
#endif

      {
        JxlPixelFormat format = {channels, JXL_TYPE_FLOAT, endianness, 0};

        std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
            jxl::Span<const uint8_t>(compressed.data(), compressed.size()),
            format);
        EXPECT_EQ(num_pixels * channels * 4, pixels2.size());
        EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                                   format_orig, format));
      }
    }
  }
}

// Tests the return status when trying to decode pixels on incomplete file: it
// should return JXL_DEC_NEED_MORE_INPUT, not error.
TEST(DecodeTest, PixelPartialTest) {
  size_t xsize = 123, ysize = 77;
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 4);
  jxl::CompressParams cparams;
  cparams.SetLossless();  // Lossless to verify pixels exactly after roundtrip.
  jxl::PaddedBytes data = jxl::CreateTestJXLCodestream(
      jxl::Span<const uint8_t>(pixels.data(), pixels.size()), xsize, ysize, 4,
      cparams);
  JxlPixelFormat format_orig = {4, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};

  std::vector<uint8_t> pixels2;
  pixels2.resize(pixels.size());

  const uint8_t* next_in = data.data();
  size_t avail_in = 0;

  JxlDecoder* dec = JxlDecoderCreate(nullptr);

  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSubscribeEvents(
                                 dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE));

  bool seen_basic_info = false;
  bool seen_full_image = false;

  size_t total_size = 0;

  for (;;) {
    JxlDecoderStatus status = JxlDecoderProcessInput(dec, &next_in, &avail_in);

    if (status == JXL_DEC_NEED_MORE_INPUT) {
      if (total_size >= data.size()) {
        // End of test data reached, it should have successfully decoded the
        // image now.
        FAIL();
        break;
      }

      size_t increment = 1;
      // Go faster once we're past the headers to speed up the test: testing
      // with increments of 1 byte during header and TOC parsing is interesting,
      // in the much larger pixel region testing just a few partial spots is
      // sufficient.
      if (total_size > 200) increment = total_size / 4;
      // End of the file reached, should be the final test.
      if (total_size + increment > data.size()) {
        increment = data.size() - total_size;
      }
      total_size += increment;
      avail_in += increment;
    } else if (status == JXL_DEC_BASIC_INFO) {
      // This event should happen exactly once
      EXPECT_FALSE(seen_basic_info);
      if (seen_basic_info) break;
      seen_basic_info = true;
      JxlBasicInfo info;
      EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec, &info));
      EXPECT_EQ(info.xsize, xsize);
      EXPECT_EQ(info.ysize, ysize);
      EXPECT_EQ(JXL_DEC_SUCCESS,
                JxlDecoderSetImageOutBuffer(dec, &format_orig, pixels2.data(),
                                            pixels2.size()));
    } else if (status == JXL_DEC_FULL_IMAGE) {
      // This event should happen exactly once
      EXPECT_FALSE(seen_full_image);
      if (seen_full_image) break;
      // This event should happen after basic info
      EXPECT_TRUE(seen_basic_info);
      seen_full_image = true;
      EXPECT_EQ(pixels, pixels2);
    } else if (status == JXL_DEC_SUCCESS) {
      EXPECT_TRUE(seen_full_image);
      break;
    } else {
      // We do not expect any other events or errors
      FAIL();
      break;
    }
  }

  // Ensure the decoder emitted the basic info and full image events
  EXPECT_TRUE(seen_basic_info);
  EXPECT_TRUE(seen_full_image);

  JxlDecoderDestroy(dec);
}

TEST(DecodeTest, DCTest) {
  using jxl::kBlockDim;

  // TODO(lode): test with a completely black image, with alpha channel
  // 65536, since that gave an error during debuging for getting DC
  // image (namely: "Failed to decode AC metadata")

  // Ensure a dimension is larger than 256 so that there are multiple groups,
  // otherwise getting DC does not work due to how TOC is then laid out.
  size_t xsize = 260, ysize = 77;
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 4);

  // Set the params to lossy, since getting DC with API is only supported for
  // lossy at this time.
  jxl::CompressParams cparams;
  jxl::PaddedBytes compressed = jxl::CreateTestJXLCodestream(
      jxl::Span<const uint8_t>(pixels.data(), pixels.size()), xsize, ysize, 4,
      cparams);

  JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};

  JxlDecoder* dec = JxlDecoderCreate(NULL);
  const uint8_t* next_in = compressed.data();
  size_t avail_in = compressed.size();

  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderSubscribeEvents(
                                 dec, JXL_DEC_BASIC_INFO | JXL_DEC_DC_IMAGE));

  EXPECT_EQ(JXL_DEC_BASIC_INFO,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));
  size_t buffer_size;
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderDCOutBufferSize(dec, &format, &buffer_size));

  size_t xsize_dc = (xsize + kBlockDim - 1) / kBlockDim;
  size_t ysize_dc = (ysize + kBlockDim - 1) / kBlockDim;
  EXPECT_EQ(xsize_dc * ysize_dc * 3, buffer_size);

  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec, &info));

  EXPECT_EQ(JXL_DEC_NEED_DC_OUT_BUFFER,
            JxlDecoderProcessInput(dec, &next_in, &avail_in));

  std::vector<uint8_t> dc(buffer_size);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetDCOutBuffer(dec, &format, dc.data(), dc.size()));

  EXPECT_EQ(JXL_DEC_DC_IMAGE, JxlDecoderProcessInput(dec, &next_in, &avail_in));

  jxl::Image3F dc0(xsize_dc, ysize_dc);
  jxl::Image3F dc1(xsize_dc, ysize_dc);

  // Downscale the original image 8x8 to allow comparing with the DC.
  std::vector<uint8_t> dc_orig(buffer_size);
  for (size_t y = 0; y < ysize_dc; y++) {
    for (size_t x = 0; x < xsize_dc; x++) {
      double r = 0, g = 0, b = 0;
      size_t num = 0;
      for (size_t by = 0; by < kBlockDim; by++) {
        size_t y2 = y * kBlockDim + by;
        if (y2 >= ysize) break;
        for (size_t bx = 0; bx < kBlockDim; bx++) {
          size_t x2 = x * kBlockDim + bx;
          if (x2 >= xsize) break;
          // Use linear RGB for correct downscaling.
          r += jxl::Srgb8ToLinearDirect(pixels[(y2 * xsize + x2) * 8 + 0]);
          g += jxl::Srgb8ToLinearDirect(pixels[(y2 * xsize + x2) * 8 + 2]);
          b += jxl::Srgb8ToLinearDirect(pixels[(y2 * xsize + x2) * 8 + 4]);
          num++;
        }
      }
      // Take average per block.
      double mul = 1.0 / num;
      r *= mul;
      g *= mul;
      b *= mul;
      dc0.PlaneRow(0, y)[x] = r;
      dc0.PlaneRow(1, y)[x] = g;
      dc0.PlaneRow(2, y)[x] = b;
      dc1.PlaneRow(0, y)[x] = (dc[(y * xsize_dc + x) * 3 + 0]);
      dc1.PlaneRow(1, y)[x] = (dc[(y * xsize_dc + x) * 3 + 1]);
      dc1.PlaneRow(2, y)[x] = (dc[(y * xsize_dc + x) * 3 + 2]);
    }
  }

  // dc0 is in linear sRGB because we converted it to linear in the downscaling
  // above.
  jxl::CodecInOut dc0_io;
  dc0_io.SetFromImage(std::move(dc0), jxl::ColorEncoding::LinearSRGB(false));
  // dc1 is in non-linear sRGB because the C decoding API outputs non-linear
  // sRGB for VarDCT to integer output types
  jxl::CodecInOut dc1_io;
  dc1_io.SetFromImage(std::move(dc1), jxl::ColorEncoding::SRGB(false));

  // Check with butteraugli that the DC is close to the 8x8 downscaled original
  // image. We don't expect a score of 0, since the downscaling done may not
  // 100% match what is stored for the DC, and the lossy codec is used.
  // A reasonable butteraugli distance shows that the DC works, the color
  // encoding (transfer function) is correct and geometry (shifts, ...) is
  // correct.
  jxl::ButteraugliParams ba;
  EXPECT_LE(ButteraugliDistance(dc0_io, dc1_io, ba,
                                /*distmap=*/nullptr, nullptr),
            3.0f);

  JxlDecoderDestroy(dec);
}
