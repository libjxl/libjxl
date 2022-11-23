// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/* clang-format off */
#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <stddef.h>
/* clang-format on */

#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "lib/jpegli/test_utils.h"
#include "lib/jxl/base/file_io.h"

namespace jpegli {
namespace {

static constexpr uint8_t kFakeEoiMarker[2] = {0xff, 0xd9};

// Custom source manager that refills the input buffer in chunks, simulating
// a file reader with a fixed buffer size.
struct TestJpegSourceManager {
  jpeg_source_mgr pub;
  const uint8_t* data;
  size_t len;
  size_t pos;
  size_t chunk_size;

  TestJpegSourceManager(const uint8_t* buf, size_t buf_size,
                        size_t max_chunk_size) {
    pub.next_input_byte = nullptr;
    pub.bytes_in_buffer = 0;
    pub.init_source = init_source;
    pub.fill_input_buffer = fill_input_buffer;
    pub.skip_input_data = skip_input_data;
    pub.resync_to_restart = resync_to_restart;
    pub.term_source = term_source;
    data = buf;
    len = buf_size;
    pos = 0;
    chunk_size = max_chunk_size;
  }

  static void init_source(j_decompress_ptr cinfo) {}

  static boolean fill_input_buffer(j_decompress_ptr cinfo) {
    auto src = reinterpret_cast<TestJpegSourceManager*>(cinfo->src);
    if (src->pos < src->len) {
      src->pub.next_input_byte = src->data + src->pos;
      src->pub.bytes_in_buffer = std::min(src->len - src->pos, src->chunk_size);
      src->pos += src->pub.bytes_in_buffer;
    } else {
      src->pub.next_input_byte = kFakeEoiMarker;
      src->pub.bytes_in_buffer = 2;
    }
    return TRUE;
  }

  static void skip_input_data(j_decompress_ptr cinfo, long num_bytes) {}

  static boolean resync_to_restart(j_decompress_ptr cinfo, int desired) {
    return FALSE;
  }

  static void term_source(j_decompress_ptr cinfo) {}
};

struct TestConfig {
  std::string fn;
  std::string fn_desc;
  std::string origfn;
  size_t chunk_size;
  size_t max_output_lines;
  size_t output_bit_depth;
  float max_distance;
};

class DecodeAPITestParam : public ::testing::TestWithParam<TestConfig> {};

TEST_P(DecodeAPITestParam, TestAPI) {
  TestConfig config = GetParam();
  const std::vector<uint8_t> compressed = ReadTestData(config.fn.c_str());
  const std::vector<uint8_t> origdata = ReadTestData(config.origfn.c_str());

  size_t xsize, ysize, num_channels, bitdepth;
  std::vector<uint8_t> orig;
  ASSERT_TRUE(
      ReadPNM(origdata, &xsize, &ysize, &num_channels, &bitdepth, &orig));
  ASSERT_EQ(8, bitdepth);

  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jmp_buf env;
  if (setjmp(env)) {
    FAIL();
  }
  cinfo.client_data = static_cast<void*>(&env);
  cinfo.err->error_exit = [](j_common_ptr cinfo) {
    (*cinfo->err->output_message)(cinfo);
    jmp_buf* env = static_cast<jmp_buf*>(cinfo->client_data);
    longjmp(*env, 1);
  };

  jpeg_create_decompress(&cinfo);

  size_t chunk_size = config.chunk_size;
  if (chunk_size == 0) chunk_size = compressed.size();
  TestJpegSourceManager jsrc(compressed.data(), compressed.size(), chunk_size);
  cinfo.src = reinterpret_cast<jpeg_source_mgr*>(&jsrc);

  ASSERT_EQ(JPEG_HEADER_OK, jpeg_read_header(&cinfo, /*require_image=*/TRUE));

  EXPECT_EQ(xsize, cinfo.image_width);
  EXPECT_EQ(ysize, cinfo.image_height);
  EXPECT_EQ(num_channels, cinfo.num_components);

  cinfo.quantize_colors = FALSE;
  cinfo.desired_number_of_colors = 1 << config.output_bit_depth;
  ASSERT_TRUE(jpeg_start_decompress(&cinfo));
  EXPECT_EQ(xsize, cinfo.output_width);
  EXPECT_EQ(ysize, cinfo.output_height);
  EXPECT_EQ(num_channels, cinfo.out_color_components);

  size_t bytes_per_sample = config.output_bit_depth <= 8 ? 1 : 2;
  size_t stride = cinfo.output_width * cinfo.num_components * bytes_per_sample;
  std::vector<uint8_t> output(cinfo.output_height * stride);
  size_t max_output_lines = config.max_output_lines;
  if (max_output_lines == 0) max_output_lines = cinfo.output_height;
  size_t total_output_lines = 0;
  while (cinfo.output_scanline < cinfo.output_height) {
    std::vector<JSAMPROW> scanlines(max_output_lines);
    for (size_t i = 0; i < max_output_lines; ++i) {
      scanlines[i] = &output[(cinfo.output_scanline + i) * stride];
    }
    size_t num_output_lines =
        jpeg_read_scanlines(&cinfo, &scanlines[0], max_output_lines);
    total_output_lines += num_output_lines;
    EXPECT_EQ(total_output_lines, cinfo.output_scanline);
    if (cinfo.output_scanline < cinfo.output_height) {
      EXPECT_EQ(num_output_lines, max_output_lines);
    }
  }

  ASSERT_TRUE(jpeg_finish_decompress(&cinfo));

  jpeg_destroy_decompress(&cinfo);

  ASSERT_EQ(output.size(), orig.size() * bytes_per_sample);
  const double mul_orig = 1.0 / 255.0;
  const double mul_output = 1.0 / ((1u << config.output_bit_depth) - 1);
  double diff2 = 0.0;
  for (size_t i = 0; i < orig.size(); ++i) {
    double sample_orig = orig[i] * mul_orig;
    double sample_output;
    if (bytes_per_sample == 1) {
      sample_output = output[i];
    } else {
      sample_output = output[2 * i] + (output[2 * i + 1] << 8);
    }
    sample_output *= mul_output;
    double diff = sample_orig - sample_output;
    diff2 += diff * diff;
  }
  double rms = std::sqrt(diff2 / orig.size());

  EXPECT_LE(rms / mul_orig, config.max_distance);
}

std::vector<TestConfig> GenerateTests() {
  std::vector<TestConfig> all_tests;
  {
    std::vector<std::pair<std::string, std::string>> testfiles({
        {"jxl/flower/flower.png.im_q85_444.jpg", "Q85YUV444"},
        {"jxl/flower/flower.png.im_q85_420.jpg", "Q85YUV420"},
        {"jxl/flower/flower.png.im_q85_420_progr.jpg", "Q85YUV420PROGR"},
        {"jxl/flower/flower.png.im_q85_420_R13B.jpg", "Q85YUV420R13B"},
    });
    for (const auto& it : testfiles) {
      for (size_t chunk_size : {0, 1, 64, 65536}) {
        for (size_t max_output_lines : {0, 1, 8, 16}) {
          for (size_t output_bit_depth : {8, 16}) {
            TestConfig config;
            config.fn = it.first;
            config.fn_desc = it.second;
            config.chunk_size = chunk_size;
            config.output_bit_depth = output_bit_depth;
            config.max_output_lines = max_output_lines;
            config.origfn = "jxl/flower/flower.pnm";
            config.max_distance = 2.2;
            if (config.output_bit_depth == 16) {
              config.max_distance = 2.1;
            }
            all_tests.push_back(config);
          }
        }
      }
    }
  }
  {
    std::vector<std::pair<std::string, std::string>> testfiles({
        {"jxl/flower/flower.png.im_q85_422.jpg", "Q85YUV422"},
        {"jxl/flower/flower.png.im_q85_440.jpg", "Q85YUV440"},
        {"jxl/flower/flower.png.im_q85_444_1x2.jpg", "Q85YUV444_1x2"},
        {"jxl/flower/flower.png.im_q85_asymmetric.jpg", "Q85Asymmetric"},
        {"jxl/flower/flower.png.im_q85_gray.jpg", "Q85Gray"},
        {"jxl/flower/flower.png.im_q85_luma_subsample.jpg", "Q85LumaSubsample"},
        {"jxl/flower/flower.png.im_q85_rgb.jpg", "Q85RGB"},
        {"jxl/flower/flower.png.im_q85_rgb_subsample_blue.jpg",
         "Q85RGBSubsampleBlue"},
    });
    for (const auto& it : testfiles) {
      for (size_t chunk_size : {0, 64}) {
        for (size_t max_output_lines : {0, 16}) {
          TestConfig config;
          config.fn = it.first;
          config.fn_desc = it.second;
          config.chunk_size = chunk_size;
          config.output_bit_depth = 8;
          config.max_output_lines = max_output_lines;
          config.origfn = "jxl/flower/flower.pnm";
          config.max_distance = 3.5;
          if (config.fn_desc == "Q85Gray") {
            config.origfn = "jxl/flower/flower.pgm";
            config.max_distance = 1.5;
          }
          all_tests.push_back(config);
        }
      }
    }
  }
  return all_tests;
}

std::ostream& operator<<(std::ostream& os, const TestConfig& c) {
  os << c.fn_desc;
  if (c.chunk_size == 0) {
    os << "CompleteInput";
  } else {
    os << "InputChunks" << c.chunk_size;
  }
  if (c.max_output_lines == 0) {
    os << "CompleteOutput";
  } else {
    os << "OutputLines" << c.max_output_lines;
  }
  os << "BitDepth" << c.output_bit_depth;
  return os;
}

std::string TestDescription(
    const testing::TestParamInfo<DecodeAPITestParam::ParamType>& info) {
  std::stringstream name;
  name << info.param;
  return name.str();
}

JPEGLI_INSTANTIATE_TEST_SUITE_P(DecodeAPITest, DecodeAPITestParam,
                                testing::ValuesIn(GenerateTests()),
                                TestDescription);

}  // namespace
}  // namespace jpegli
