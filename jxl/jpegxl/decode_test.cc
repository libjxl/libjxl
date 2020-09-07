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
#include "jxl/base/byte_order.h"
#include "jxl/base/file_io.h"
#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/brunsli.h"
#include "jxl/dec_file.h"
#include "jxl/external_image.h"
#include "jxl/fields.h"
#include "jxl/headers.h"
#include "jxl/icc_codec.h"
#include "jxl/test_utils.h"

TEST(DecodeTest, JpegxlSignatureCheckTest) {
  std::vector<std::pair<int, std::vector<uint8_t>>> tests = {
      // No JPEGXL header starts with 'a'.
      {JPEGXL_SIG_INVALID, {'a'}},
      {JPEGXL_SIG_INVALID, {'a', 'b', 'c', 'd', 'e', 'f'}},

      // Empty file is not enough bytes.
      {JPEGXL_SIG_NOT_ENOUGH_BYTES, {}},

      // JPEGXL headers.
      {JPEGXL_SIG_NOT_ENOUGH_BYTES, {0xff}},  // Part of a signature.
      {JPEGXL_SIG_VALID, {0xff, 0xD8}},
      {JPEGXL_SIG_VALID, {0xff, 0x0a}},

      // A header could start with 0x0a, but it is not a complete signature.
      {JPEGXL_SIG_NOT_ENOUGH_BYTES, {0x0a}},

      // This is the beginning of a Brunsli file, but not a complete signature.
      {JPEGXL_SIG_NOT_ENOUGH_BYTES, {0x0a, 0x04, 'B', 0xd2}},
      {JPEGXL_SIG_VALID, {0x0a, 0x04, 'B', 0xd2, 0xd5, 'N', 0x12}},
      {JPEGXL_SIG_INVALID, {0x0a, 0x04, 'B', 0xd2, 0xd5, 'N', 0x13}},

      // JPEGXL container file.
      {JPEGXL_SIG_VALID,
       {0, 0, 0, 0xc, 'J', 'X', 'L', ' ', 0xD, 0xA, 0x87, 0xA}},
      // Ending with invalid byte.
      {JPEGXL_SIG_INVALID,
       {0, 0, 0, 0xc, 'J', 'X', 'L', ' ', 0xD, 0xA, 0x87, 0}},
      // Part of signature.
      {JPEGXL_SIG_NOT_ENOUGH_BYTES,
       {0, 0, 0, 0xc, 'J', 'X', 'L', ' ', 0xD, 0xA, 0x87}},
      {JPEGXL_SIG_NOT_ENOUGH_BYTES, {0}},
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

// TODO(lode): add multi-threaded test when multithreaded pixel decoding from
// API is implemented.
TEST(DecodeTest, DefaultParallelRunnerTest) {
  JpegxlDecoder* dec = JpegxlDecoderCreate(nullptr);
  EXPECT_NE(nullptr, dec);
  EXPECT_EQ(JPEGXL_DEC_SUCCESS,
            JpegxlDecoderSetParallelRunner(dec, nullptr, nullptr));
  JpegxlDecoderDestroy(dec);
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
                                   size_t alpha_bits, bool have_container,
                                   bool metadata_default, bool insert_extra_box,
                                   const jxl::PaddedBytes& icc_profile) {
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
  jxl::SizeHeader size;
  EXPECT_TRUE(size.Set(xsize, ysize));
  EXPECT_TRUE(WriteSizeHeader(size, &writer, 0, nullptr));

  jxl::ImageMetadata metadata;
  if (!metadata_default) {
    metadata.SetUintSamples(bits_per_sample);
    metadata.m2.orientation_minus_1 = orientation - 1;
    metadata.SetAlphaBits(alpha_bits);
    if (alpha_bits != 0) {
      metadata.m2.extra_channel_info[0].name = "alpha_test";
    }
  }

  if (!icc_profile.empty()) {
    jxl::PaddedBytes copy = icc_profile;
    EXPECT_TRUE(metadata.color_encoding.SetICC(std::move(copy)));
  }

  EXPECT_TRUE(jxl::Bundle::Write(metadata, &writer, 0, nullptr));

  if (!icc_profile.empty()) {
    EXPECT_TRUE(jxl::WriteICC(icc_profile, &writer, 0, nullptr));
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

  std::vector<std::vector<uint8_t>> test_samples;
  // Test with direct codestream
  test_samples.push_back(GetTestHeader(
      xsize[0], ysize[0], bits_per_sample[0], orientation[0], alpha_bits[0],
      have_container[0], /*metadata_default=*/false,
      /*insert_extra_box=*/false, {}));
  // Test with container and different parameters
  test_samples.push_back(GetTestHeader(
      xsize[1], ysize[1], bits_per_sample[1], orientation[1], alpha_bits[1],
      have_container[1], /*metadata_default=*/false,
      /*insert_extra_box=*/false, {}));

  for (size_t i = 0; i < test_samples.size(); ++i) {
    const std::vector<uint8_t>& data = test_samples[i];
    // Test decoding too small header first, until we reach the final byte.
    for (size_t size = 0; size <= data.size(); ++size) {
      JpegxlDecoder* dec = JpegxlDecoderCreate(nullptr);
      EXPECT_EQ(JPEGXL_DEC_SUCCESS,
                JpegxlDecoderSubscribeEvents(dec, JPEGXL_DEC_BASIC_INFO));
      const uint8_t* next_in = data.data();
      size_t avail_in = size;
      JpegxlDecoderStatus status =
          JpegxlDecoderProcessInput(dec, &next_in, &avail_in);

      JpegxlBasicInfo info;
      bool have_basic_info = !JpegxlDecoderGetBasicInfo(dec, &info);

      if (size == data.size()) {
        EXPECT_EQ(JPEGXL_DEC_BASIC_INFO, status);

        // All header bytes given so the decoder must have the basic info.
        EXPECT_EQ(true, have_basic_info);
        EXPECT_EQ(JPEGXL_SIG_TYPE_JPEGXL, info.signature_type);
        EXPECT_EQ(have_container[i], info.have_container);
        EXPECT_EQ(xsize[i], info.xsize);
        EXPECT_EQ(ysize[i], info.ysize);
        EXPECT_EQ(alpha_bits[i], info.alpha_bits);
        EXPECT_EQ(orientation[i], info.orientation);

        if (alpha_bits[i] != 0) {
          // Expect an extra channel
          EXPECT_EQ(1, info.num_extra_channels);
          JpegxlExtraChannelInfo extra;
          EXPECT_EQ(0, JpegxlDecoderGetExtraChannelInfo(dec, 0, &extra));
          EXPECT_EQ(alpha_bits[i], extra.bits_per_sample);
          EXPECT_EQ(JPEGXL_CHANNEL_ALPHA, extra.type);
          EXPECT_EQ(0, extra.alpha_associated);
          // Verify the name "alpha_test" given to the alpha channel
          EXPECT_EQ(10, extra.name_length);
          char name[11];
          EXPECT_EQ(
              0, JpegxlDecoderGetExtraChannelName(dec, 0, name, sizeof(name)));
          EXPECT_EQ(std::string("alpha_test"), std::string(name));
        } else {
          EXPECT_EQ(0, info.num_extra_channels);
        }
      } else {
        // If we did not give the full header, the basic info should not be
        // available. Allow a few bytes of slack due to some bits for default
        // opsinmatrix/extension bits.
        if (size + 2 < data.size()) {
          EXPECT_EQ(false, have_basic_info);
          EXPECT_EQ(JPEGXL_DEC_NEED_MORE_INPUT, status);
        }
      }

      JpegxlDecoderDestroy(dec);
    }
  }
}

TEST(DecodeTest, BasicInfoSizeHintTest) {
  // Test on a file where the size hint is too small initially due to inserting
  // a box before the codestream (something that is normally not recommended)
  size_t xsize = 50;
  size_t ysize = 50;
  size_t bits_per_sample = 16;
  size_t orientation = 1;
  size_t alpha_bits = 0;
  std::vector<uint8_t> data =
      GetTestHeader(xsize, ysize, bits_per_sample, orientation, alpha_bits,
                    /*have_container=*/true, /*metadata_default=*/false,
                    /*insert_extra_box=*/true, {});

  JpegxlDecoderStatus status;
  JpegxlDecoder* dec = JpegxlDecoderCreate(nullptr);
  EXPECT_EQ(JPEGXL_DEC_SUCCESS,
            JpegxlDecoderSubscribeEvents(dec, JPEGXL_DEC_BASIC_INFO));

  size_t hint0 = JpegxlDecoderSizeHintBasicInfo(dec);
  // Test that the test works as intended: we construct a file on purpose to
  // be larger than the first hint by having that extra box.
  EXPECT_LT(hint0, data.size());
  const uint8_t* next_in = data.data();
  // Do as if we have only as many bytes as indicated by the hint available
  size_t avail_in = std::min(hint0, data.size());
  status = JpegxlDecoderProcessInput(dec, &next_in, &avail_in);
  EXPECT_EQ(JPEGXL_DEC_NEED_MORE_INPUT, status);
  // Basic info cannot be available yet due to the extra inserted box.
  EXPECT_EQ(false, !JpegxlDecoderGetBasicInfo(dec, nullptr));

  size_t num_read = next_in - data.data();
  EXPECT_LT(num_read, data.size());

  size_t hint1 = JpegxlDecoderSizeHintBasicInfo(dec);
  // The hint must be larger than the previouw hint (taking already processed
  // bytes into account, the hint is a hint for the next avail_in) since the
  // decoder now knows there is a box in between.
  EXPECT_GT(hint1 + num_read, hint0);
  avail_in = std::min<size_t>(hint1, data.size() - num_read);

  status = JpegxlDecoderProcessInput(dec, &next_in, &avail_in);
  EXPECT_EQ(JPEGXL_DEC_BASIC_INFO, status);
  JpegxlBasicInfo info;
  // We should have the basic info now, since we only added one box in-between,
  // and the decoder should have known its size, its implementation can return
  // a correct hint.
  EXPECT_EQ(true, !JpegxlDecoderGetBasicInfo(dec, &info));

  // Also test if the basic info is correct.
  EXPECT_EQ(JPEGXL_SIG_TYPE_JPEGXL, info.signature_type);
  EXPECT_EQ(1, info.have_container);
  EXPECT_EQ(xsize, info.xsize);
  EXPECT_EQ(ysize, info.ysize);
  EXPECT_EQ(orientation, info.orientation);
  EXPECT_EQ(bits_per_sample, info.bits_per_sample);

  JpegxlDecoderDestroy(dec);
}

TEST(DecodeTest, IccProfileTest) {
  // An ICC profile output by the JPEG XL decoder for RGB_D65_SRG_Rel_Lin.
  const uint8_t* profile = reinterpret_cast<const uint8_t*>(
      "\0\0\3\200lcms\0040\0\0mntrRGB XYZ "
      "\a\344\0\a\0\27\0\21\0$"
      "\0\37acspAPPL\0\0\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\1\0\0\366"
      "\326\0\1\0\0\0\0\323-lcms\372c\207\36\227\200{"
      "\2\232s\255\327\340\0\n\26\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
      "\0\0\0\0\0\0\0\0\rdesc\0\0\1 "
      "\0\0\0Bcprt\0\0\1d\0\0\1\0wtpt\0\0\2d\0\0\0\24chad\0\0\2x\0\0\0,"
      "rXYZ\0\0\2\244\0\0\0\24bXYZ\0\0\2\270\0\0\0\24gXYZ\0\0\2\314\0\0\0\24rTR"
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
  jxl::PaddedBytes icc_profile;
  icc_profile.assign(profile, profile + profile_size);

  size_t xsize = 50;
  size_t ysize = 50;
  size_t bits_per_sample = 16;
  size_t orientation = 1;
  size_t alpha_bits = 0;
  std::vector<uint8_t> data =
      GetTestHeader(xsize, ysize, bits_per_sample, orientation, alpha_bits,
                    /*have_container=*/false, /*metadata_default=*/false,
                    /*insert_extra_box=*/false, icc_profile);
  const uint8_t* next_in = data.data();
  size_t avail_in = data.size();

  JpegxlDecoder* dec = JpegxlDecoderCreate(nullptr);
  EXPECT_EQ(JPEGXL_DEC_SUCCESS,
            JpegxlDecoderSubscribeEvents(
                dec, JPEGXL_DEC_BASIC_INFO | JPEGXL_DEC_COLOR_ENCODING));

  EXPECT_EQ(JPEGXL_DEC_BASIC_INFO,
            JpegxlDecoderProcessInput(dec, &next_in, &avail_in));

  EXPECT_EQ(JPEGXL_DEC_COLOR_ENCODING,
            JpegxlDecoderProcessInput(dec, &next_in, &avail_in));

  JpegxlColorProfileSource color_info;
  EXPECT_EQ(JPEGXL_DEC_SUCCESS,
            JpegxlDecoderGetColorProfileSource(dec, &color_info));

  EXPECT_EQ(profile_size, color_info.icc_profile_size);
  jxl::PaddedBytes icc_profile2(profile_size);
  EXPECT_EQ(JPEGXL_DEC_SUCCESS,
            JpegxlDecoderGetICCProfile(dec, icc_profile2.data(),
                                       icc_profile2.size()));

  JpegxlDecoderDestroy(dec);
}

namespace jxl {
namespace {
// Input pixels always given as 16-bit RGBA, 8 bytes per pixel.
// include_alpha determines if the encoded image should contain the alpha
// channel.
PaddedBytes CreateTestJXLCodestream(Span<const uint8_t> pixels, size_t xsize,
                                    size_t ysize, const CompressParams& cparams,
                                    bool include_alpha, bool grayscale) {
  // Compress the pixels with JPEG XL.
  size_t bitdepth = 16;
  CodecInOut io;
  ColorEncoding color_encoding = jxl::test::ColorEncodingFromDescriptor(
      {grayscale ? ColorSpace::kGray : ColorSpace::kRGB, WhitePoint::kD65,
       Primaries::kSRGB, TransferFunction::kSRGB, RenderingIntent::kRelative});
  ThreadPool pool(nullptr, nullptr);
  const PackedImage desc(
      xsize, ysize, color_encoding, /*has_alpha=*/include_alpha,
      /*alpha_is_premultiplied=*/false, include_alpha ? bitdepth : 0, bitdepth,
      /*big_endian=*/true, /*flipped_y=*/false);
  io.metadata.SetUintSamples(bitdepth);
  if (include_alpha) {
    io.metadata.SetAlphaBits(bitdepth);
  }
  // Make the grayscale-ness of the io metadata color_encoding and the packed
  // image match.
  io.metadata.color_encoding = color_encoding;
  EXPECT_TRUE(CopyTo(desc, pixels, &pool, &io.Main()));
  AuxOut aux_out;
  PaddedBytes compressed;
  PassesEncoderState enc_state;
  EXPECT_TRUE(
      EncodeFile(cparams, &io, &enc_state, &compressed, &aux_out, &pool));
  return compressed;
}

std::vector<uint8_t> DecodeWithAPI(Span<const uint8_t> compressed,
                                   const JpegxlPixelFormat& format) {
  // Test decoding with the API.

  JpegxlDecoder* dec = JpegxlDecoderCreate(NULL);
  const uint8_t* next_in = compressed.data();
  size_t avail_in = compressed.size();

  EXPECT_EQ(JPEGXL_DEC_SUCCESS,
            JpegxlDecoderSubscribeEvents(
                dec, JPEGXL_DEC_BASIC_INFO | JPEGXL_DEC_FULL_IMAGE));

  EXPECT_EQ(JPEGXL_DEC_BASIC_INFO,
            JpegxlDecoderProcessInput(dec, &next_in, &avail_in));
  size_t buffer_size;
  EXPECT_EQ(JPEGXL_DEC_SUCCESS,
            JpegxlDecoderImageOutBufferSize(dec, &format, &buffer_size));
  JpegxlBasicInfo info;
  EXPECT_EQ(JPEGXL_DEC_SUCCESS, JpegxlDecoderGetBasicInfo(dec, &info));
  std::vector<uint8_t> pixels(buffer_size);
  EXPECT_EQ(JPEGXL_DEC_SUCCESS,
            JpegxlDecoderSetImageOutBuffer(dec, &format, pixels.data(),
                                           pixels.size()));

  EXPECT_EQ(JPEGXL_DEC_FULL_IMAGE,
            JpegxlDecoderProcessInput(dec, &next_in, &avail_in));

  JpegxlDecoderDestroy(dec);

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

size_t GetPrecision(JpegxlDataType data_type) {
  switch (data_type) {
    case JPEGXL_TYPE_BOOLEAN:
      return 1;
    case JPEGXL_TYPE_UINT8:
      return 8;
    case JPEGXL_TYPE_UINT16:
      return 16;
    case JPEGXL_TYPE_UINT32:
      return 32;
    case JPEGXL_TYPE_FLOAT:
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
                                    const JpegxlPixelFormat& format) {
  std::vector<double> result(xsize * ysize * 4);
  size_t num_channels = format.num_channels;
  bool gray = num_channels == 1 || num_channels == 2;
  bool alpha = num_channels == 2 || num_channels == 4;
  if (format.data_type == JPEGXL_TYPE_BOOLEAN) {
    size_t row_size = (xsize * num_channels + 7) >> 3;
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t j = (y * xsize + x) * 4;
        size_t i = y * row_size + ((x * num_channels + 7) >> 3);
        uint8_t byte = pixels[i];
        size_t bit = (x * num_channels) & 7;
        if (bit != 0) byte >>= (8 - bit);
        uint32_t r = (byte & 1);
        uint32_t g = gray ? r : ((byte & 2) >> 1);
        uint32_t b = gray ? r : ((byte & 4) >> 2);
        uint32_t a = alpha ? ((byte >> (num_channels - 1)) & 1) : 1;
        result[j + 0] = r;
        result[j + 1] = g;
        result[j + 2] = b;
        result[j + 3] = a;
      }
    }
  } else if (format.data_type == JPEGXL_TYPE_UINT8) {
    double mul = 1.0 / 255.0;  // Multiplier to bring to 0-1.0 range
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t j = (y * xsize + x) * 4;
        size_t i = (y * xsize + x) * num_channels;
        uint32_t r = pixels[i];
        uint32_t g = gray ? r : pixels[i + 1];
        uint32_t b = gray ? r : pixels[i + 2];
        uint32_t a = alpha ? pixels[i + num_channels - 1] : 255;
        result[j + 0] = r * mul;
        result[j + 1] = g * mul;
        result[j + 2] = b * mul;
        result[j + 3] = a * mul;
      }
    }
  } else if (format.data_type == JPEGXL_TYPE_UINT16) {
    double mul = 1.0 / 65535.0;  // Multiplier to bring to 0-1.0 range
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t j = (y * xsize + x) * 4;
        size_t i = (y * xsize + x) * num_channels * 2;
        uint32_t r = (pixels[i + 0] << 8) + pixels[i + 1];
        uint32_t g = gray ? r : (pixels[i + 2] << 8) + pixels[i + 3];
        uint32_t b = gray ? r : (pixels[i + 4] << 8) + pixels[i + 5];
        uint32_t a = alpha ? (pixels[i + num_channels * 2 - 2] << 8) +
                                 pixels[i + num_channels * 2 - 1]
                           : 65535;
        result[j + 0] = r * mul;
        result[j + 1] = g * mul;
        result[j + 2] = b * mul;
        result[j + 3] = a * mul;
      }
    }
  } else if (format.data_type == JPEGXL_TYPE_UINT32) {
    double mul = 1.0 / 4294967295.0;  // Multiplier to bring to 0-1.0 range
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t j = (y * xsize + x) * 4;
        size_t i = (y * xsize + x) * num_channels * 4;
        uint32_t r = LoadBE32(pixels + i);
        uint32_t g = gray ? r : LoadBE32(pixels + i + 4);
        uint32_t b = gray ? r : LoadBE32(pixels + i + 8);
        uint32_t a =
            alpha ? LoadBE32(pixels + i + num_channels * 2 - 4) : 4294967295;
        result[j + 0] = r * mul;
        result[j + 1] = g * mul;
        result[j + 2] = b * mul;
        result[j + 3] = a * mul;
      }
    }
  } else if (format.data_type == JPEGXL_TYPE_FLOAT) {
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        size_t j = (y * xsize + x) * 4;
        size_t i = (y * xsize + x) * num_channels * 4;
        float r = LoadBEFloat(pixels + i);
        float g = gray ? r : LoadBEFloat(pixels + i + 4);
        float b = gray ? r : LoadBEFloat(pixels + i + 8);
        float a = alpha ? LoadBEFloat(pixels + i + num_channels * 4 - 4) : 1.0;
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
                     size_t ysize, const JpegxlPixelFormat& format_a,
                     const JpegxlPixelFormat& format_b) {
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
        if (alpha_b && !Near(255.0, b_full[i + 3], precision)) ok = false;
      }
      if (!ok) numdiff++;
    }
  }
  return numdiff;
}
}  // namespace

TEST(DecodeTest, PixelTest) {
  size_t xsize = 123, ysize = 77;
  size_t num_pixels = xsize * ysize;
  // 16 bits per channel, big endian, 4 channels
  size_t orig_bytes_per_channel = 8;
  std::vector<uint8_t> pixels(num_pixels * 8);
  // Create pixel content to test, actual content does not matter as long as it
  // can be compared after roundtrip.
  for (size_t y = 0; y < ysize; y++) {
    for (size_t x = 0; x < xsize; x++) {
      uint16_t r = 65535 - x * y;
      uint16_t g = (x << 8) + y;
      uint16_t b = (y << 8) + x;
      uint16_t a = 32768 + x * 256 - y;
      size_t i = (y * xsize + x) * orig_bytes_per_channel;
      pixels[i + 0] = (r >> 8);
      pixels[i + 1] = (r & 255);
      pixels[i + 2] = (g >> 8);
      pixels[i + 3] = (g & 255);
      pixels[i + 4] = (b >> 8);
      pixels[i + 5] = (b & 255);
      pixels[i + 6] = (a >> 8);
      pixels[i + 7] = (a & 255);
    }
  }
  JpegxlPixelFormat format_orig = {4, JPEGXL_TYPE_UINT16};

  jxl::CompressParams cparams;
  cparams.SetLossless();  // Lossless to verify pixels exactly after roundtrip.
  jxl::PaddedBytes compressed = jxl::CreateTestJXLCodestream(
      jxl::Span<const uint8_t>(pixels.data(), pixels.size()), xsize, ysize,
      cparams, true, false);

  {
    JpegxlPixelFormat format = {3, JPEGXL_TYPE_UINT8};

    std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
        jxl::Span<const uint8_t>(compressed.data(), compressed.size()), format);
    EXPECT_EQ(num_pixels * 3, pixels2.size());
    EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                               format_orig, format));
  }

  {
    JpegxlPixelFormat format = {4, JPEGXL_TYPE_UINT8};
    std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
        jxl::Span<const uint8_t>(compressed.data(), compressed.size()), format);
    EXPECT_EQ(num_pixels * 4, pixels2.size());
    EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                               format_orig, format));
  }

  {
    JpegxlPixelFormat format = {3, JPEGXL_TYPE_UINT16};

    std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
        jxl::Span<const uint8_t>(compressed.data(), compressed.size()), format);
    EXPECT_EQ(num_pixels * 6, pixels2.size());
    EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                               format_orig, format));
  }

  {
    JpegxlPixelFormat format = {4, JPEGXL_TYPE_UINT16};

    std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
        jxl::Span<const uint8_t>(compressed.data(), compressed.size()), format);
    EXPECT_EQ(num_pixels * 8, pixels2.size());
    EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                               format_orig, format));
  }

#if 0  // Disabled since external_image doesn't currently support uint32_t
  {
    JpegxlPixelFormat format = {4, JPEGXL_TYPE_UINT32};

    std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
        jxl::Span<const uint8_t>(compressed.data(), compressed.size()), format);
    EXPECT_EQ(num_pixels * 16, pixels2.size());
    EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                               format_orig, format));
  }
#endif

  {
    JpegxlPixelFormat format = {3, JPEGXL_TYPE_FLOAT};

    std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
        jxl::Span<const uint8_t>(compressed.data(), compressed.size()), format);
    EXPECT_EQ(num_pixels * 12, pixels2.size());
    EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                               format_orig, format));
  }

  {
    JpegxlPixelFormat format = {4, JPEGXL_TYPE_FLOAT};

    std::vector<uint8_t> pixels2 = jxl::DecodeWithAPI(
        jxl::Span<const uint8_t>(compressed.data(), compressed.size()), format);
    EXPECT_EQ(num_pixels * 16, pixels2.size());
    EXPECT_EQ(0, ComparePixels(pixels.data(), pixels2.data(), xsize, ysize,
                               format_orig, format));
  }
}
