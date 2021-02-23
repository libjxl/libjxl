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
#include "jxl/encode_cxx.h"
#include "lib/extras/codec.h"
#include "lib/jxl/box.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/enc_butteraugli_comparator.h"
#include "lib/jxl/encode_internal.h"
#include "lib/jxl/jpeg/dec_jpeg_data.h"
#include "lib/jxl/jpeg/dec_jpeg_data_writer.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testdata.h"

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

  {
    JxlEncoderPtr enc = JxlEncoderMake(&mm);
    EXPECT_NE(nullptr, enc.get());
    EXPECT_LE(1, counters.allocs);
    EXPECT_EQ(0, counters.frees);
  }
  EXPECT_LE(1, counters.frees);
}

TEST(EncodeTest, DefaultParallelRunnerTest) {
  JxlEncoderPtr enc = JxlEncoderMake(nullptr);
  EXPECT_NE(nullptr, enc.get());
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetParallelRunner(enc.get(), nullptr, nullptr));
}

void VerifyFrameEncoding(size_t xsize, size_t ysize, JxlEncoder* enc,
                         const JxlEncoderOptions* options) {
  JxlPixelFormat pixel_format = {4, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};
  std::vector<uint8_t> pixels = jxl::test::GetSomeTestImage(xsize, ysize, 4, 0);

  jxl::CodecInOut input_io =
      jxl::test::SomeTestImageToCodecInOut(pixels, 4, xsize, ysize);

  JxlBasicInfo basic_info;
  jxl::test::JxlBasicInfoSetFromPixelFormat(&basic_info, &pixel_format);
  basic_info.xsize = xsize;
  basic_info.ysize = ysize;
  if (options->values.lossless) {
    basic_info.uses_original_profile = true;
  } else {
    basic_info.uses_original_profile = false;
  }
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc, &basic_info));
  JxlColorEncoding color_encoding;
  JxlColorEncodingSetToSRGB(&color_encoding,
                            /*is_gray=*/pixel_format.num_channels < 3);
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetColorEncoding(enc, &color_encoding));
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddImageFrame(options, &pixel_format, pixels.data(),
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
            3.0f);
}

void VerifyFrameEncoding(JxlEncoder* enc, const JxlEncoderOptions* options) {
  VerifyFrameEncoding(63, 129, enc, options);
}

TEST(EncodeTest, FrameEncodingTest) {
  JxlEncoderPtr enc = JxlEncoderMake(nullptr);
  EXPECT_NE(nullptr, enc.get());
  VerifyFrameEncoding(enc.get(), JxlEncoderOptionsCreate(enc.get(), nullptr));
}

TEST(EncodeTest, EncoderResetTest) {
  JxlEncoderPtr enc = JxlEncoderMake(nullptr);
  EXPECT_NE(nullptr, enc.get());
  VerifyFrameEncoding(50, 200, enc.get(),
                      JxlEncoderOptionsCreate(enc.get(), nullptr));
  // Encoder should become reusable for a new image from scratch after using
  // reset.
  JxlEncoderReset(enc.get());
  VerifyFrameEncoding(157, 77, enc.get(),
                      JxlEncoderOptionsCreate(enc.get(), nullptr));
}

TEST(EncodeTest, OptionsTest) {
  {
    JxlEncoderPtr enc = JxlEncoderMake(nullptr);
    EXPECT_NE(nullptr, enc.get());
    JxlEncoderOptions* options = JxlEncoderOptionsCreate(enc.get(), NULL);
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderOptionsSetEffort(options, 5));
    VerifyFrameEncoding(enc.get(), options);
    EXPECT_EQ(jxl::SpeedTier::kHare, enc->last_used_cparams.speed_tier);
  }

  {
    JxlEncoderPtr enc = JxlEncoderMake(nullptr);
    EXPECT_NE(nullptr, enc.get());
    JxlEncoderOptions* options = JxlEncoderOptionsCreate(enc.get(), NULL);
    // Lower than currently supported values
    EXPECT_EQ(JXL_ENC_ERROR, JxlEncoderOptionsSetEffort(options, 2));
    // Higher than currently supported values
    EXPECT_EQ(JXL_ENC_ERROR, JxlEncoderOptionsSetEffort(options, 10));
  }

  {
    JxlEncoderPtr enc = JxlEncoderMake(nullptr);
    EXPECT_NE(nullptr, enc.get());
    JxlEncoderOptions* options = JxlEncoderOptionsCreate(enc.get(), NULL);
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderOptionsSetLossless(options, JXL_TRUE));
    VerifyFrameEncoding(enc.get(), options);
    EXPECT_EQ(true, enc->last_used_cparams.IsLossless());
  }

  {
    JxlEncoderPtr enc = JxlEncoderMake(nullptr);
    EXPECT_NE(nullptr, enc.get());
    JxlEncoderOptions* options = JxlEncoderOptionsCreate(enc.get(), NULL);
    EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderOptionsSetDistance(options, 0.5));
    VerifyFrameEncoding(enc.get(), options);
    EXPECT_EQ(0.5, enc->last_used_cparams.butteraugli_distance);
  }

  {
    JxlEncoderPtr enc = JxlEncoderMake(nullptr);
    JxlEncoderOptions* options = JxlEncoderOptionsCreate(enc.get(), NULL);
    // Disallowed negative distance
    EXPECT_EQ(JXL_ENC_ERROR, JxlEncoderOptionsSetDistance(options, -1));
  }
}

TEST(EncodeTest, JPEGReconstructionTest) {
  const std::string jpeg_path =
      "imagecompression.info/flower_foveon.png.im_q85_420.jpg";
  const jxl::PaddedBytes orig = jxl::ReadTestData(jpeg_path);
  jxl::CodecInOut orig_io;
  ASSERT_TRUE(
      SetFromBytes(jxl::Span<const uint8_t>(orig), &orig_io, /*pool=*/nullptr));

  JxlEncoderPtr enc = JxlEncoderMake(nullptr);
  JxlEncoderOptions* options = JxlEncoderOptionsCreate(enc.get(), NULL);

  JxlBasicInfo basic_info;
  basic_info.exponent_bits_per_sample = 0;
  basic_info.bits_per_sample = 8;
  basic_info.alpha_bits = 0;
  basic_info.alpha_exponent_bits = 0;
  basic_info.xsize = orig_io.xsize();
  basic_info.ysize = orig_io.ysize();
  basic_info.uses_original_profile = true;
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc.get(), &basic_info));
  JxlColorEncoding color_encoding;
  JxlColorEncodingSetToSRGB(&color_encoding, /*is_gray=*/false);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetColorEncoding(enc.get(), &color_encoding));
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderUseContainer(enc.get(), JXL_TRUE));
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderStoreJPEGMetadata(enc.get(), JXL_TRUE));
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddJPEGFrame(options, orig.data(), orig.size()));
  JxlEncoderCloseInput(enc.get());

  std::vector<uint8_t> compressed = std::vector<uint8_t>(64);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size() - (next_out - compressed.data());
  JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
  while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
    process_result = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
    if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - compressed.data();
      compressed.resize(compressed.size() * 2);
      next_out = compressed.data() + offset;
      avail_out = compressed.size() - offset;
    }
  }
  compressed.resize(next_out - compressed.data());
  EXPECT_EQ(JXL_ENC_SUCCESS, process_result);

  jxl::JxlContainer container = {};
  jxl::Span<uint8_t> encoded_span =
      jxl::Span<uint8_t>(compressed.data(), compressed.size());
  EXPECT_TRUE(container.Decode(&encoded_span));
  EXPECT_EQ(0, encoded_span.size());
  EXPECT_EQ(0, memcmp("jbrd", container.boxes[0].type, 4));
  EXPECT_EQ(0, memcmp("jxlc", container.boxes[1].type, 4));

  jxl::CodecInOut decoded_io;
  decoded_io.Main().jpeg_data = jxl::make_unique<jxl::jpeg::JPEGData>();
  EXPECT_TRUE(jxl::jpeg::DecodeJPEGData(container.boxes[0].data,
                                        decoded_io.Main().jpeg_data.get()));

  jxl::DecompressParams dparams;
  dparams.keep_dct = true;
  EXPECT_TRUE(jxl::DecodeFile(dparams, container.boxes[1].data, &decoded_io,
                              nullptr, nullptr));

  std::vector<uint8_t> decoded_jpeg_bytes;
  auto write = [&decoded_jpeg_bytes](const uint8_t* buf, size_t len) {
    decoded_jpeg_bytes.insert(decoded_jpeg_bytes.end(), buf, buf + len);
    return len;
  };
  EXPECT_TRUE(jxl::jpeg::WriteJpeg(*decoded_io.Main().jpeg_data, write));

  EXPECT_EQ(decoded_jpeg_bytes.size(), orig.size());
  EXPECT_EQ(0, memcmp(decoded_jpeg_bytes.data(), orig.data(), orig.size()));
}

TEST(EncodeTest, JPEGFrameTest) {
  const std::string jpeg_path =
      "imagecompression.info/flower_foveon.png.im_q85_420.jpg";
  const jxl::PaddedBytes orig = jxl::ReadTestData(jpeg_path);
  jxl::CodecInOut orig_io;
  ASSERT_TRUE(
      SetFromBytes(jxl::Span<const uint8_t>(orig), &orig_io, /*pool=*/nullptr));

  JxlEncoderPtr enc = JxlEncoderMake(nullptr);
  JxlEncoderOptions* options = JxlEncoderOptionsCreate(enc.get(), NULL);

  JxlBasicInfo basic_info;
  basic_info.exponent_bits_per_sample = 0;
  basic_info.bits_per_sample = 8;
  basic_info.alpha_bits = 0;
  basic_info.alpha_exponent_bits = 0;
  basic_info.xsize = orig_io.xsize();
  basic_info.ysize = orig_io.ysize();
  basic_info.uses_original_profile = true;
  EXPECT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc.get(), &basic_info));
  JxlColorEncoding color_encoding;
  JxlColorEncodingSetToSRGB(&color_encoding, /*is_gray=*/false);
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetColorEncoding(enc.get(), &color_encoding));
  EXPECT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddJPEGFrame(options, orig.data(), orig.size()));
  JxlEncoderCloseInput(enc.get());

  std::vector<uint8_t> compressed = std::vector<uint8_t>(64);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size() - (next_out - compressed.data());
  JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
  while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
    process_result = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
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
  EXPECT_LE(ButteraugliDistance(orig_io, decoded_io, ba,
                                /*distmap=*/nullptr, nullptr),
            2.5f);
}
