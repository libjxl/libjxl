// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>

#include <cmath>
#include <vector>

#include "lib/jpegli/decode.h"
#include "lib/jpegli/test_utils.h"
#include "lib/jpegli/testing.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/sanitizers.h"

namespace jpegli {
namespace {

void DecodeWithLibJpeg(const std::vector<uint8_t>& compressed,
                       volatile size_t* xsize, volatile size_t* ysize,
                       volatile size_t* num_channels,
                       std::vector<uint8_t>* pixels) {
  jpeg_decompress_struct cinfo = {};
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
  jpeg_mem_src(&cinfo, compressed.data(), compressed.size());
  jpeg_read_header(&cinfo, TRUE);
  *xsize = cinfo.image_width;
  *ysize = cinfo.image_height;
  jpeg_start_decompress(&cinfo);
  *num_channels = cinfo.output_components;
  const size_t stride = cinfo.output_components * cinfo.image_width;
  pixels->resize(cinfo.image_height * stride);
  for (size_t y = 0; y < cinfo.image_height; ++y) {
    JSAMPROW rows[] = {&(*pixels)[stride * y]};
    jpeg_read_scanlines(&cinfo, rows, 1);
    jxl::msan::UnpoisonMemory(rows[0], stride);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
}

static constexpr uint8_t kFakeEoiMarker[2] = {0xff, 0xd9};

class SourceManager {
 public:
  SourceManager(const uint8_t* data, size_t len, size_t max_chunk_size)
      : data_(data), len_(len), pos_(0), max_chunk_size_(max_chunk_size) {
    pub_.next_input_byte = nullptr;
    pub_.bytes_in_buffer = 0;
    pub_.skip_input_data = skip_input_data;
    pub_.resync_to_restart = jpegli_resync_to_restart;
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
  SOURCE_MGR_STDIO,
};

struct TestConfig {
  std::string fn;
  std::string fn_desc;
  size_t chunk_size = 0;
  size_t max_output_lines = 0;
  float max_distance = 1.0;
  SourceManagerType source_mgr = SOURCE_MGR_CHUNKED;
  bool pre_consume_input = false;
  bool buffered_image_mode = false;
  bool crop = false;
  bool raw_output = false;
  JpegliDataType data_type = JPEGLI_TYPE_UINT8;
  JpegliEndianness endianness = JPEGLI_NATIVE_ENDIAN;
};

bool LoadNextChunk(const TestConfig& config, j_decompress_ptr cinfo) {
  if (config.source_mgr == SOURCE_MGR_SUSPENDING) {
    auto src = reinterpret_cast<SuspendingSourceManager*>(cinfo->src);
    return src->LoadNextChunk();
  }
  return false;
}

std::string DataTypeString(JpegliDataType type) {
  switch (type) {
    case JPEGLI_TYPE_UINT8:
      return "UINT8";
    case JPEGLI_TYPE_UINT16:
      return "UINT16";
    case JPEGLI_TYPE_FLOAT:
      return "FLOAT";
    default:
      return "UNKNOWN";
  }
}

std::string EndiannessString(JpegliEndianness endianness) {
  if (endianness == JPEGLI_BIG_ENDIAN) {
    return "BE";
  } else if (endianness == JPEGLI_LITTLE_ENDIAN) {
    return "LE";
  }
  return "";
}

class DecodeAPITestParam : public ::testing::TestWithParam<TestConfig> {};

TEST_P(DecodeAPITestParam, TestAPI) {
  TestConfig config = GetParam();
  const std::vector<uint8_t> compressed = ReadTestData(config.fn.c_str());

  // These has to be volatile to make setjmp/longjmp work.
  volatile size_t xsize, ysize, num_channels;
  std::vector<uint8_t> orig;
  DecodeWithLibJpeg(compressed, &xsize, &ysize, &num_channels, &orig);

  jpeg_decompress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpegli_std_error(&jerr);
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

  jpegli_create_decompress(&cinfo);

  size_t chunk_size = config.chunk_size;
  if (chunk_size == 0) chunk_size = compressed.size();
  ChunkedSourceManager src_chunked(compressed.data(), compressed.size(),
                                   chunk_size);
  SuspendingSourceManager src_susp(compressed.data(), compressed.size(),
                                   chunk_size);
  std::string jpg_full_path = GetTestDataPath(config.fn);
  jxl::FileWrapper testfile(jpg_full_path, "rb");
  ASSERT_TRUE(testfile != nullptr);
  if (config.source_mgr == SOURCE_MGR_CHUNKED) {
    cinfo.src = reinterpret_cast<jpeg_source_mgr*>(&src_chunked);
  } else if (config.source_mgr == SOURCE_MGR_SUSPENDING) {
    cinfo.src = reinterpret_cast<jpeg_source_mgr*>(&src_susp);
  } else if (config.source_mgr == SOURCE_MGR_STDIO) {
    jpegli_stdio_src(&cinfo, testfile);
  }

  if (config.pre_consume_input) {
    for (;;) {
      int status = jpegli_consume_input(&cinfo);
      if (status == JPEG_SUSPENDED) {
        ASSERT_TRUE(LoadNextChunk(config, &cinfo));
      } else if (status == JPEG_REACHED_SOS) {
        break;
      }
    }
  } else {
    for (;;) {
      int status = jpegli_read_header(&cinfo, /*require_image=*/TRUE);
      if (status == JPEG_SUSPENDED) {
        ASSERT_TRUE(LoadNextChunk(config, &cinfo));
      } else {
        ASSERT_EQ(status, JPEG_HEADER_OK);
        break;
      }
    }
  }

  ASSERT_EQ(JPEG_REACHED_SOS, jpegli_consume_input(&cinfo));

  EXPECT_EQ(xsize, cinfo.image_width);
  EXPECT_EQ(ysize, cinfo.image_height);
  EXPECT_EQ(num_channels, cinfo.num_components);

  jpegli_set_output_format(&cinfo, config.data_type, config.endianness);

  if (jpegli_has_multiple_scans(&cinfo) && config.buffered_image_mode) {
    cinfo.buffered_image = TRUE;
  }

  if (cinfo.max_v_samp_factor > 1 && config.raw_output) {
    cinfo.raw_data_out = TRUE;
  }

  if (config.pre_consume_input) {
    jpegli_start_decompress(&cinfo);
  } else if (cinfo.buffered_image) {
    EXPECT_TRUE(jpegli_start_decompress(&cinfo));
  } else {
    while (!jpegli_start_decompress(&cinfo)) {
      ASSERT_TRUE(LoadNextChunk(config, &cinfo));
    }
  }

  EXPECT_EQ(xsize, cinfo.output_width);
  EXPECT_EQ(ysize, cinfo.output_height);
  EXPECT_EQ(num_channels, cinfo.out_color_components);

  JDIMENSION xoffset = 0;
  JDIMENSION yoffset = 0;
  JDIMENSION xsize_cropped = xsize;
  JDIMENSION ysize_cropped = ysize;
  if (config.crop && !config.raw_output) {
    xoffset = xsize_cropped = xsize / 3;
    yoffset = ysize_cropped = ysize / 3;
    jpegli_crop_scanline(&cinfo, &xoffset, &xsize_cropped);
  }

  std::vector<uint8_t> cropped(xsize_cropped * ysize_cropped * num_channels);
  for (size_t y = 0; y < ysize_cropped; ++y) {
    for (size_t x = 0; x < xsize_cropped; ++x) {
      size_t crop_ix = y * xsize_cropped + x;
      size_t orig_ix = (yoffset + y) * xsize + xoffset + x;
      for (size_t c = 0; c < num_channels; ++c) {
        cropped[crop_ix * num_channels + c] = orig[orig_ix * num_channels + c];
      }
    }
  }

  size_t bytes_per_sample = jpegli_bytes_per_sample(config.data_type);
  size_t stride = cinfo.output_width * cinfo.num_components * bytes_per_sample;
  size_t max_output_lines = config.max_output_lines;
  if (max_output_lines == 0) max_output_lines = cinfo.output_height;

  if (config.pre_consume_input) {
    for (;;) {
      int status = jpegli_consume_input(&cinfo);
      if (status == JPEG_SUSPENDED) {
        ASSERT_TRUE(LoadNextChunk(config, &cinfo));
      } else if (status == JPEG_REACHED_EOI) {
        break;
      }
    }
  }

  while (!jpegli_input_complete(&cinfo)) {
    if (cinfo.buffered_image) {
      EXPECT_TRUE(jpegli_start_output(&cinfo, cinfo.input_scan_number));
    }

    if (cinfo.raw_data_out) {
      std::vector<std::vector<uint8_t>> planes;
      std::vector<size_t> strides(cinfo.num_components);
      for (int c = 0; c < cinfo.num_components; ++c) {
        size_t xsize = cinfo.comp_info[c].width_in_blocks * DCTSIZE;
        strides[c] = xsize * bytes_per_sample;
        size_t ysize = cinfo.comp_info[c].height_in_blocks * DCTSIZE;
        std::vector<uint8_t> plane(ysize * stride);
        planes.emplace_back(std::move(plane));
      }
      size_t total_output_lines = 0;
      for (;;) {
        size_t max_lines = cinfo.max_v_samp_factor * DCTSIZE;
        std::vector<std::vector<JSAMPROW>> rowdata(cinfo.num_components);
        std::vector<JSAMPARRAY> data(cinfo.num_components);
        for (int c = 0; c < cinfo.num_components; ++c) {
          size_t vfactor = cinfo.comp_info[c].v_samp_factor;
          size_t cheight = cinfo.comp_info[c].height_in_blocks * DCTSIZE;
          size_t num_lines = vfactor * DCTSIZE;
          rowdata[c].resize(num_lines);
          size_t y0 = cinfo.output_iMCU_row * num_lines;
          for (size_t i = 0; i < num_lines; ++i) {
            rowdata[c][i] = y0 + i < cheight ? &planes[c][y0 + i] : nullptr;
          }
          data[c] = &rowdata[c][0];
        }
        size_t num_output_lines =
            jpegli_read_raw_data(&cinfo, &data[0], max_lines);
        total_output_lines += num_output_lines;
        EXPECT_EQ(total_output_lines, cinfo.output_scanline);
        if (cinfo.output_scanline >= cinfo.output_height) {
          break;
        }
        if (config.pre_consume_input) {
          EXPECT_EQ(num_output_lines, max_lines);
        } else if (num_output_lines < max_lines) {
          ASSERT_TRUE(LoadNextChunk(config, &cinfo));
        }
        // TODO(szabadka) Add expectations on the raw data.
      }
    } else {
      std::vector<uint8_t> output(ysize_cropped * stride);
      size_t total_output_lines = 0;
      for (;;) {
        size_t num_output_lines;
        size_t max_lines;
        if (cinfo.output_scanline < yoffset) {
          max_lines = yoffset - cinfo.output_scanline;
          num_output_lines = jpegli_skip_scanlines(&cinfo, max_lines);
        } else if (cinfo.output_scanline >= yoffset + ysize_cropped) {
          max_lines = cinfo.output_height - cinfo.output_scanline;
          num_output_lines = jpegli_skip_scanlines(&cinfo, max_lines);
        } else {
          size_t lines_left = yoffset + ysize_cropped - cinfo.output_scanline;
          max_lines = std::min<size_t>(max_output_lines, lines_left);
          std::vector<JSAMPROW> scanlines(max_lines);
          for (size_t i = 0; i < max_lines; ++i) {
            size_t yidx = cinfo.output_scanline - yoffset + i;
            scanlines[i] = &output[yidx * stride];
          }
          num_output_lines =
              jpegli_read_scanlines(&cinfo, &scanlines[0], max_lines);
        }
        total_output_lines += num_output_lines;
        EXPECT_EQ(total_output_lines, cinfo.output_scanline);
        if (cinfo.output_scanline >= cinfo.output_height) {
          break;
        }
        if (config.pre_consume_input) {
          EXPECT_EQ(num_output_lines, max_lines);
        } else if (num_output_lines < max_lines) {
          ASSERT_TRUE(LoadNextChunk(config, &cinfo));
        }
      }
      ASSERT_EQ(output.size(), cropped.size() * bytes_per_sample);
      bool is_little_endian =
          (config.endianness == JPEGLI_LITTLE_ENDIAN ||
           (config.endianness == JPEGLI_NATIVE_ENDIAN && IsLittleEndian()));
      const double kMul8 = 1.0 / 255.0;
      const double kMul16 = 1.0 / 65535.0;
      double diff2 = 0.0;
      for (size_t i = 0; i < cropped.size(); ++i) {
        double sample_orig = cropped[i] * kMul8;
        double sample_output = 0.0;
        if (config.data_type == JPEGLI_TYPE_UINT8) {
          sample_output = output[i] * kMul8;
        } else if (config.data_type == JPEGLI_TYPE_UINT16) {
          sample_output = is_little_endian ? LoadLE16(&output[2 * i])
                                           : LoadBE16(&output[2 * i]);
          sample_output *= kMul16;
        } else if (config.data_type == JPEGLI_TYPE_FLOAT) {
          sample_output = is_little_endian ? LoadLEFloat(&output[4 * i])
                                           : LoadBEFloat(&output[4 * i]);
        }
        double diff = sample_orig - sample_output;
        diff2 += diff * diff;
      }
      double rms = std::sqrt(diff2 / cropped.size()) / kMul8;
      double max_dist = config.max_distance;
      if (!cinfo.buffered_image || jpegli_input_complete(&cinfo)) {
        EXPECT_LE(rms, max_dist);
      } else {
        EXPECT_LE(rms, max_dist * 20.0);
      }
    }
    if (!cinfo.buffered_image || jpegli_input_complete(&cinfo)) {
      EXPECT_EQ(cinfo.input_iMCU_row, cinfo.total_iMCU_rows);
    }
    if (cinfo.buffered_image) {
      if (config.pre_consume_input) {
        EXPECT_TRUE(jpegli_finish_output(&cinfo));
      } else {
        while (!jpegli_finish_output(&cinfo)) {
          ASSERT_TRUE(LoadNextChunk(config, &cinfo));
        }
      }
    } else {
      break;
    }
  }

  if (config.pre_consume_input || cinfo.buffered_image) {
    EXPECT_TRUE(jpegli_finish_decompress(&cinfo));
  } else {
    while (!jpegli_finish_decompress(&cinfo)) {
      ASSERT_TRUE(LoadNextChunk(config, &cinfo));
    }
  }
  EXPECT_TRUE(jpegli_input_complete(&cinfo));
  if (config.source_mgr == SOURCE_MGR_CHUNKED) {
    EXPECT_EQ(0, src_chunked.UnprocessedBytes());
    EXPECT_EQ(src_chunked.TotalBytes(), compressed.size());
  } else if (config.source_mgr == SOURCE_MGR_SUSPENDING) {
    EXPECT_EQ(0, src_susp.UnprocessedBytes());
    EXPECT_EQ(src_susp.TotalBytes(), compressed.size());
  }

  jpegli_destroy_decompress(&cinfo);
}

std::vector<TestConfig> GenerateTests() {
  std::vector<TestConfig> all_tests;
  {
    std::vector<std::pair<std::string, std::string>> testfiles({
        {"jxl/flower/flower.png.im_q85_444.jpg", "Q85YUV444"},
        {"jxl/flower/flower.png.im_q85_420.jpg", "Q85YUV420"},
        {"jxl/flower/flower.png.im_q85_420_R13B.jpg", "Q85YUV420R13B"},
    });
    for (const auto& it : testfiles) {
      for (size_t chunk_size : {0, 1, 64, 65536}) {
        for (size_t max_output_lines : {0, 1, 8, 16}) {
          TestConfig config;
          config.fn = it.first;
          config.fn_desc = it.second;
          config.chunk_size = chunk_size;
          config.max_output_lines = max_output_lines;
          all_tests.push_back(config);
          if (config.chunk_size != 0) {
            config.source_mgr = SOURCE_MGR_SUSPENDING;
            all_tests.push_back(config);
          } else {
            config.source_mgr = SOURCE_MGR_STDIO;
            all_tests.push_back(config);
          }
        }
      }
    }
  }

  {
    for (size_t chunk_size : {0, 65536}) {
      for (size_t max_output_lines : {0, 16}) {
        for (bool pre_consume : {false, true}) {
          for (bool buffered : {false, true}) {
            for (bool crop : {false, true}) {
              TestConfig config;
              config.fn = "jxl/flower/flower.png.im_q85_420_progr.jpg";
              config.fn_desc = "Q85YUV420PROGR";
              config.chunk_size = chunk_size;
              config.max_output_lines = max_output_lines;
              config.pre_consume_input = pre_consume;
              config.buffered_image_mode = buffered;
              config.crop = crop;
              all_tests.push_back(config);
              if (config.chunk_size != 0) {
                config.source_mgr = SOURCE_MGR_SUSPENDING;
                all_tests.push_back(config);
              }
              if (config.max_output_lines == 0 && !config.crop) {
                config.raw_output = true;
                all_tests.push_back(config);
              }
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
            config.max_output_lines = max_output_lines;
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
          config.max_output_lines = max_output_lines;
          all_tests.push_back(config);
        }
      }
    }
  }
  {
    for (JpegliDataType type : {JPEGLI_TYPE_UINT16, JPEGLI_TYPE_FLOAT}) {
      for (JpegliEndianness endianness :
           {JPEGLI_NATIVE_ENDIAN, JPEGLI_LITTLE_ENDIAN, JPEGLI_BIG_ENDIAN}) {
        TestConfig config;
        config.fn = "jxl/flower/flower.png.im_q85_444.jpg";
        config.fn_desc = "Q85YUV444";
        config.data_type = type;
        config.endianness = endianness;
        all_tests.push_back(config);
      }
    }
  }
  {
    TestConfig config;
    config.fn = "jxl/flower/flower_small.cmyk.jpg";
    config.fn_desc = "CMYK";
    all_tests.push_back(config);
  }
  return all_tests;
}

std::ostream& operator<<(std::ostream& os, const TestConfig& c) {
  os << c.fn_desc;
  if (c.source_mgr == SOURCE_MGR_STDIO) {
    os << "Stdio";
  } else if (c.chunk_size == 0) {
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
  if (c.buffered_image_mode) {
    os << "Buffered";
  }
  if (c.crop) {
    os << "Crop";
  } else if (c.raw_output) {
    os << "Raw";
  }
  os << DataTypeString(c.data_type);
  if (c.data_type != JPEGLI_TYPE_UINT8) {
    os << EndiannessString(c.endianness);
  }
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
