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
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "lib/jpegli/test_utils.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/status.h"

namespace jpegli {
namespace {

static constexpr uint8_t kFakeEoiMarker[2] = {0xff, 0xd9};

class SourceManager {
 public:
  SourceManager(const uint8_t* data, size_t len, size_t max_chunk_size)
      : data_(data), len_(len), pos_(0), max_chunk_size_(max_chunk_size) {
    pub_.next_input_byte = nullptr;
    pub_.bytes_in_buffer = 0;
    pub_.skip_input_data = skip_input_data;
    pub_.resync_to_restart = resync_to_restart;
    pub_.term_source = term_source;
  }

  size_t TotalBytes() const { return pos_; }
  size_t UnprocessedBytes() const { return pub_.bytes_in_buffer; }

 protected:
  jpeg_source_mgr pub_;
  const uint8_t* data_;
  size_t len_;
  size_t pos_;
  size_t max_chunk_size_;

 private:
  static void skip_input_data(j_decompress_ptr cinfo, long num_bytes) {}

  static boolean resync_to_restart(j_decompress_ptr cinfo, int desired) {
    return FALSE;
  }

  static void term_source(j_decompress_ptr cinfo) {}
};

// Custom source manager that refills the input buffer in chunks, simulating
// a file reader with a fixed buffer size.
class ChunkedSourceManager : public SourceManager {
 public:
  ChunkedSourceManager(const uint8_t* data, size_t len, size_t max_chunk_size)
      : SourceManager(data, len, max_chunk_size) {
    pub_.init_source = init_source;
    pub_.fill_input_buffer = fill_input_buffer;
  }

 private:
  static void init_source(j_decompress_ptr cinfo) { fill_input_buffer(cinfo); }

  static boolean fill_input_buffer(j_decompress_ptr cinfo) {
    auto src = reinterpret_cast<ChunkedSourceManager*>(cinfo->src);
    if (src->pos_ < src->len_) {
      size_t chunk_size = std::min(src->len_ - src->pos_, src->max_chunk_size_);
      src->pub_.next_input_byte = src->data_ + src->pos_;
      src->pub_.bytes_in_buffer = chunk_size;
    } else {
      src->pub_.next_input_byte = kFakeEoiMarker;
      src->pub_.bytes_in_buffer = 2;
    }
    src->pos_ += src->pub_.bytes_in_buffer;
    return TRUE;
  }
};

class SuspendingSourceManager : public SourceManager {
 public:
  SuspendingSourceManager(const uint8_t* data, size_t len,
                          size_t max_chunk_size)
      : SourceManager(data, len, max_chunk_size) {
    pub_.init_source = init_source;
    pub_.fill_input_buffer = fill_input_buffer;
  }

  bool LoadNextChunk() {
    if (pos_ >= len_) {
      return false;
    }
    if (pub_.bytes_in_buffer > 0) {
      EXPECT_LE(pub_.bytes_in_buffer, buffer_.size());
      memmove(&buffer_[0], pub_.next_input_byte, pub_.bytes_in_buffer);
    }
    size_t chunk_size = std::min(len_ - pos_, max_chunk_size_);
    buffer_.resize(pub_.bytes_in_buffer + chunk_size);
    memcpy(&buffer_[pub_.bytes_in_buffer], data_ + pos_, chunk_size);
    pub_.next_input_byte = &buffer_[0];
    pub_.bytes_in_buffer += chunk_size;
    pos_ += chunk_size;
    return true;
  }

 private:
  std::vector<uint8_t> buffer_;

  static void init_source(j_decompress_ptr cinfo) {
    auto src = reinterpret_cast<SuspendingSourceManager*>(cinfo->src);
    src->pub_.next_input_byte = nullptr;
    src->pub_.bytes_in_buffer = 0;
  }
  static boolean fill_input_buffer(j_decompress_ptr cinfo) { return FALSE; }
};

enum SourceManagerType {
  SOURCE_MGR_CHUNKED,
  SOURCE_MGR_SUSPENDING,
};

struct TestConfig {
  std::string fn;
  std::string fn_desc;
  std::string origfn;
  size_t chunk_size;
  size_t max_output_lines;
  size_t output_bit_depth;
  float max_distance;
  SourceManagerType source_mgr;
  bool pre_consume_input = false;
};

void LoadNextChunk(const TestConfig& config, SourceManager* src) {
  ASSERT_EQ(config.source_mgr, SOURCE_MGR_SUSPENDING);
  JXL_CHECK(reinterpret_cast<SuspendingSourceManager*>(src)->LoadNextChunk());
}

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
  std::unique_ptr<SourceManager> jsrc;
  if (config.source_mgr == SOURCE_MGR_CHUNKED) {
    jsrc.reset(new ChunkedSourceManager(compressed.data(), compressed.size(),
                                        chunk_size));
  } else if (config.source_mgr == SOURCE_MGR_SUSPENDING) {
    jsrc.reset(new SuspendingSourceManager(compressed.data(), compressed.size(),
                                           chunk_size));
  }
  cinfo.src = reinterpret_cast<jpeg_source_mgr*>(jsrc.get());

  if (config.pre_consume_input) {
    for (;;) {
      int status = jpeg_consume_input(&cinfo);
      if (status == JPEG_SUSPENDED) {
        LoadNextChunk(config, jsrc.get());
      } else if (status == JPEG_REACHED_SOS) {
        break;
      }
    }
  } else {
    for (;;) {
      int status = jpeg_read_header(&cinfo, /*require_image=*/TRUE);
      if (status == JPEG_SUSPENDED) {
        LoadNextChunk(config, jsrc.get());
      } else {
        ASSERT_EQ(status, JPEG_HEADER_OK);
        break;
      }
    }
  }

  ASSERT_EQ(JPEG_REACHED_SOS, jpeg_consume_input(&cinfo));

  EXPECT_EQ(xsize, cinfo.image_width);
  EXPECT_EQ(ysize, cinfo.image_height);
  EXPECT_EQ(num_channels, cinfo.num_components);

  cinfo.quantize_colors = FALSE;
  cinfo.desired_number_of_colors = 1 << config.output_bit_depth;

  if (config.pre_consume_input) {
    jpeg_start_decompress(&cinfo);
  } else {
    while (!jpeg_start_decompress(&cinfo)) {
      LoadNextChunk(config, jsrc.get());
    }
  }

  EXPECT_EQ(xsize, cinfo.output_width);
  EXPECT_EQ(ysize, cinfo.output_height);
  EXPECT_EQ(num_channels, cinfo.out_color_components);

  if (config.pre_consume_input) {
    for (;;) {
      int status = jpeg_consume_input(&cinfo);
      if (status == JPEG_SUSPENDED) {
        LoadNextChunk(config, jsrc.get());
      } else if (status == JPEG_REACHED_EOI) {
        break;
      }
    }
  }

  size_t bytes_per_sample = config.output_bit_depth <= 8 ? 1 : 2;
  size_t stride = cinfo.output_width * cinfo.num_components * bytes_per_sample;
  std::vector<uint8_t> output(cinfo.output_height * stride);
  size_t max_output_lines = config.max_output_lines;
  if (max_output_lines == 0) max_output_lines = cinfo.output_height;
  size_t total_output_lines = 0;
  for (;;) {
    std::vector<JSAMPROW> scanlines(max_output_lines);
    for (size_t i = 0; i < max_output_lines; ++i) {
      scanlines[i] = &output[(cinfo.output_scanline + i) * stride];
    }
    size_t num_output_lines =
        jpeg_read_scanlines(&cinfo, &scanlines[0], max_output_lines);
    total_output_lines += num_output_lines;
    EXPECT_EQ(total_output_lines, cinfo.output_scanline);
    if (cinfo.output_scanline >= cinfo.output_height) {
      break;
    }
    if (config.pre_consume_input) {
      EXPECT_EQ(num_output_lines, max_output_lines);
    } else if (num_output_lines < max_output_lines) {
      LoadNextChunk(config, jsrc.get());
    }
  }
  EXPECT_EQ(cinfo.input_iMCU_row, cinfo.total_iMCU_rows);

  if (config.pre_consume_input) {
    jpeg_finish_decompress(&cinfo);
  } else {
    while (!jpeg_finish_decompress(&cinfo)) {
      LoadNextChunk(config, jsrc.get());
    }
  }
  EXPECT_EQ(0, jsrc->UnprocessedBytes());
  EXPECT_EQ(jsrc->TotalBytes(), compressed.size());

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
            config.source_mgr = SOURCE_MGR_CHUNKED;
            all_tests.push_back(config);
            if (config.chunk_size != 0) {
              config.source_mgr = SOURCE_MGR_SUSPENDING;
              all_tests.push_back(config);
            }
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
          for (bool pre_consume : {false, true}) {
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
            config.source_mgr = SOURCE_MGR_CHUNKED;
            config.pre_consume_input = pre_consume;
            all_tests.push_back(config);
            if (config.chunk_size != 0) {
              config.source_mgr = SOURCE_MGR_SUSPENDING;
              all_tests.push_back(config);
            }
          }
        }
      }
    }
  }
  {
    std::vector<std::pair<std::string, std::string>> testfiles({
        {"jxl/flower/flower_small.q85_444_non_interleaved.jpg",
         "Q85YUV444NonInterleaved"},
        {"jxl/flower/flower_small.q85_420_non_interleaved.jpg",
         "Q85YUV420NonInterleaved"},
        {"jxl/flower/flower_small.q85_444_partially_interleaved.jpg",
         "Q85YUV444PartiallyInterleaved"},
        {"jxl/flower/flower_small.q85_420_partially_interleaved.jpg",
         "Q85YUV420PartiallyInterleaved"},
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
          config.origfn = "jxl/flower/flower_small.rgb.depth8.ppm";
          config.max_distance = 3.5;
          config.source_mgr = SOURCE_MGR_CHUNKED;
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
    if (c.source_mgr == SOURCE_MGR_SUSPENDING) {
      os << "Suspending";
    }
  }
  if (c.pre_consume_input) {
    os << "PreConsume";
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
