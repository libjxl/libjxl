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
#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/brunsli.h"
#include "jxl/fields.h"
#include "jxl/headers.h"

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
                                   bool metadata_default,
                                   bool insert_extra_box) {
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

  EXPECT_TRUE(jxl::Bundle::Write(metadata, &writer, 0, nullptr));

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
      /*insert_extra_box=*/false));
  // Test with container and different parameters
  test_samples.push_back(GetTestHeader(
      xsize[1], ysize[1], bits_per_sample[1], orientation[1], alpha_bits[1],
      have_container[1], /*metadata_default=*/false,
      /*insert_extra_box=*/false));

  for (size_t i = 0; i < test_samples.size(); ++i) {
    const std::vector<uint8_t>& data = test_samples[i];
    // Test decoding too small header first, until we reach the final byte.
    for (size_t size = 0; size <= data.size(); ++size) {
      JpegxlDecoder* dec = JpegxlDecoderCreate(NULL);
      const uint8_t* next_in = data.data();
      size_t avail_in = size;
      JpegxlDecoderStatus status =
          JpegxlDecoderProcessInput(dec, &next_in, &avail_in);
      // Since the header is a partial JPEG XL file, the function should always
      // return 'need more input'.
      EXPECT_EQ(JPEGXL_DEC_NEED_MORE_INPUT, status);

      JpegxlBasicInfo info;
      bool have_basic_info = !JpegxlDecoderGetBasicInfo(dec, &info);

      if (size == data.size()) {
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
              0, JpegxlDecoderGetExtraChannelName(dec, 0, sizeof(name), name));
          EXPECT_EQ(std::string("alpha_test"), std::string(name));
        } else {
          EXPECT_EQ(0, info.num_extra_channels);
        }
      } else {
        // If we did not give the full header, the basic info should not be
        // available. Allow a few bytes of slack due to some bits for default
        // opsinmatrix/extension bits.
        if (size + 2 < data.size()) EXPECT_EQ(false, have_basic_info);
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
                    /*insert_extra_box=*/true);

  JpegxlDecoderStatus status;
  JpegxlDecoder* dec = JpegxlDecoderCreate(NULL);

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
  EXPECT_EQ(JPEGXL_DEC_NEED_MORE_INPUT, status);
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
