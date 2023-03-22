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

static constexpr uint8_t kFakeEoiMarker[2] = {0xff, 0xd9};

// Custom source manager that refills the input buffer in chunks, simulating
// a file reader with a fixed buffer size.
class SourceManager {
 public:
  SourceManager(const uint8_t* data, size_t len, size_t max_chunk_size)
      : data_(data), len_(len), pos_(0), max_chunk_size_(max_chunk_size) {
    pub_.next_input_byte = nullptr;
    pub_.bytes_in_buffer = 0;
    pub_.skip_input_data = skip_input_data;
    pub_.resync_to_restart = jpegli_resync_to_restart;
    pub_.term_source = term_source;
    pub_.init_source = init_source;
    pub_.fill_input_buffer = fill_input_buffer;
    if (max_chunk_size_ == 0) max_chunk_size_ = len;
  }

  ~SourceManager() {
    EXPECT_EQ(0, pub_.bytes_in_buffer);
    EXPECT_EQ(len_, pos_);
  }

 private:
  jpeg_source_mgr pub_;
  const uint8_t* data_;
  size_t len_;
  size_t pos_;
  size_t max_chunk_size_;

  static void init_source(j_decompress_ptr cinfo) {}

  static boolean fill_input_buffer(j_decompress_ptr cinfo) {
    auto src = reinterpret_cast<SourceManager*>(cinfo->src);
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

  static void skip_input_data(j_decompress_ptr cinfo, long num_bytes) {}

  static void term_source(j_decompress_ptr cinfo) {}
};

void SetDecompressParams(const DecompressParams& dparams,
                         j_decompress_ptr cinfo) {
  jpegli_set_output_format(cinfo, dparams.data_type, dparams.endianness);
  cinfo->raw_data_out = dparams.output_mode == RAW_DATA;
  cinfo->do_block_smoothing = dparams.do_block_smoothing;
  if (dparams.set_out_color_space) {
    cinfo->out_color_space = dparams.out_color_space;
    if (dparams.out_color_space == JCS_UNKNOWN) {
      cinfo->jpeg_color_space = JCS_UNKNOWN;
    }
  }
  cinfo->scale_num = dparams.scale_num;
  cinfo->scale_denom = dparams.scale_denom;
}

void ReadOutputImage(const DecompressParams& dparams, j_decompress_ptr cinfo,
                     TestImage* output) {
  JDIMENSION xoffset = 0;
  JDIMENSION yoffset = 0;
  JDIMENSION xsize_cropped = cinfo->output_width;
  JDIMENSION ysize_cropped = cinfo->output_height;
  if (dparams.crop_output) {
    xoffset = xsize_cropped = cinfo->output_width / 3;
    yoffset = ysize_cropped = cinfo->output_height / 3;
    jpegli_crop_scanline(cinfo, &xoffset, &xsize_cropped);
  }
  output->ysize = ysize_cropped;
  output->xsize = cinfo->output_width;
  output->components = cinfo->num_components;
  output->data_type = dparams.data_type;
  output->endianness = dparams.endianness;
  size_t bytes_per_sample = jpegli_bytes_per_sample(dparams.data_type);
  if (cinfo->raw_data_out) {
    output->color_space = cinfo->jpeg_color_space;
    for (int c = 0; c < cinfo->num_components; ++c) {
      size_t xsize = cinfo->comp_info[c].width_in_blocks * DCTSIZE;
      size_t ysize = cinfo->comp_info[c].height_in_blocks * DCTSIZE;
      std::vector<uint8_t> plane(ysize * xsize * bytes_per_sample);
      output->raw_data.emplace_back(std::move(plane));
    }
  } else {
    output->color_space = cinfo->out_color_space;
    output->AllocatePixels();
  }
  size_t total_output_lines = 0;
  while (cinfo->output_scanline < cinfo->output_height) {
    size_t max_lines;
    size_t num_output_lines;
    if (cinfo->raw_data_out) {
      size_t iMCU_height = cinfo->max_v_samp_factor * DCTSIZE;
      EXPECT_EQ(cinfo->output_scanline, cinfo->output_iMCU_row * iMCU_height);
      max_lines = iMCU_height;
      std::vector<std::vector<JSAMPROW>> rowdata(cinfo->num_components);
      std::vector<JSAMPARRAY> data(cinfo->num_components);
      for (int c = 0; c < cinfo->num_components; ++c) {
        size_t xsize = cinfo->comp_info[c].width_in_blocks * DCTSIZE;
        size_t ysize = cinfo->comp_info[c].height_in_blocks * DCTSIZE;
        size_t num_lines = cinfo->comp_info[c].v_samp_factor * DCTSIZE;
        rowdata[c].resize(num_lines);
        size_t y0 = cinfo->output_iMCU_row * num_lines;
        for (size_t i = 0; i < num_lines; ++i) {
          rowdata[c][i] =
              y0 + i < ysize ? &output->raw_data[c][(y0 + i) * xsize] : nullptr;
        }
        data[c] = &rowdata[c][0];
      }
      num_output_lines = jpegli_read_raw_data(cinfo, &data[0], max_lines);
    } else {
      size_t max_output_lines = dparams.max_output_lines;
      if (max_output_lines == 0) max_output_lines = cinfo->output_height;
      if (cinfo->output_scanline < yoffset) {
        max_lines = yoffset - cinfo->output_scanline;
        num_output_lines = jpegli_skip_scanlines(cinfo, max_lines);
      } else if (cinfo->output_scanline >= yoffset + ysize_cropped) {
        max_lines = cinfo->output_height - cinfo->output_scanline;
        num_output_lines = jpegli_skip_scanlines(cinfo, max_lines);
      } else {
        size_t lines_left = yoffset + ysize_cropped - cinfo->output_scanline;
        max_lines = std::min<size_t>(max_output_lines, lines_left);
        size_t stride =
            cinfo->output_width * cinfo->num_components * bytes_per_sample;
        std::vector<JSAMPROW> scanlines(max_lines);
        for (size_t i = 0; i < max_lines; ++i) {
          size_t yidx = cinfo->output_scanline - yoffset + i;
          scanlines[i] = &output->pixels[yidx * stride];
        }
        num_output_lines =
            jpegli_read_scanlines(cinfo, &scanlines[0], max_lines);
      }
    }
    total_output_lines += num_output_lines;
    EXPECT_EQ(total_output_lines, cinfo->output_scanline);
    EXPECT_EQ(num_output_lines, max_lines);
  }
  EXPECT_EQ(cinfo->total_iMCU_rows,
            DivCeil(cinfo->image_height, cinfo->max_v_samp_factor * DCTSIZE));
}

struct TestConfig {
  std::string fn;
  std::string fn_desc;
  TestImage input;
  CompressParams jparams;
  DecompressParams dparams;
  bool compare_to_orig = false;
};

std::vector<uint8_t> GetTestJpegData(TestConfig& config) {
  if (!config.fn.empty()) {
    return ReadTestData(config.fn.c_str());
  }
  GeneratePixels(&config.input);
  std::vector<uint8_t> compressed;
  JXL_CHECK(EncodeWithJpegli(config.input, config.jparams, &compressed));
  return compressed;
}

class DecodeAPITestParam : public ::testing::TestWithParam<TestConfig> {};

TEST_P(DecodeAPITestParam, TestAPI) {
  TestConfig config = GetParam();
  const DecompressParams& dparams = config.dparams;
  const std::vector<uint8_t> compressed = GetTestJpegData(config);
  SourceManager src(compressed.data(), compressed.size(), dparams.chunk_size);
  TestImage output0;
  jpeg_decompress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_decompress(&cinfo);
    cinfo.src = reinterpret_cast<jpeg_source_mgr*>(&src);
    if (config.jparams.add_marker) {
      jpegli_save_markers(&cinfo, kSpecialMarker, 0xffff);
    }
    jpegli_read_header(&cinfo, /*require_image=*/TRUE);
    SetDecompressParams(dparams, &cinfo);
    VerifyHeader(config.jparams, &cinfo);
    jpegli_start_decompress(&cinfo);
    VerifyScanHeader(config.jparams, &cinfo);
    ReadOutputImage(dparams, &cinfo, &output0);
    jpegli_finish_decompress(&cinfo);
    return true;
  };
  ASSERT_TRUE(try_catch_block());
  jpegli_destroy_decompress(&cinfo);

  TestImage output1;
  DecodeWithLibjpeg(CompressParams(), dparams, compressed, &output1);

  if (config.compare_to_orig) {
    double rms0 = DistanceRms(config.input, output0);
    double rms1 = DistanceRms(config.input, output1);
    EXPECT_LE(rms0, rms1 * 1.01);
  } else {
    VerifyOutputImage(output1, output0, 1.0f);
  }
}

class DecodeAPITestParamBuffered : public ::testing::TestWithParam<TestConfig> {
};

TEST_P(DecodeAPITestParamBuffered, TestAPI) {
  TestConfig config = GetParam();
  const DecompressParams& dparams = config.dparams;
  const std::vector<uint8_t> compressed = GetTestJpegData(config);
  SourceManager src(compressed.data(), compressed.size(), dparams.chunk_size);
  std::vector<TestImage> output_progression0;
  jpeg_decompress_struct cinfo;
  const auto try_catch_block = [&]() -> bool {
    ERROR_HANDLER_SETUP(jpegli);
    jpegli_create_decompress(&cinfo);
    cinfo.src = reinterpret_cast<jpeg_source_mgr*>(&src);
    EXPECT_EQ(JPEG_REACHED_SOS,
              jpegli_read_header(&cinfo, /*require_image=*/TRUE));
    SetDecompressParams(dparams, &cinfo);
    VerifyHeader(config.jparams, &cinfo);
    cinfo.buffered_image = TRUE;
    EXPECT_TRUE(jpegli_start_decompress(&cinfo));
    // start decompress should not read the whole input in buffered image mode
    EXPECT_FALSE(jpegli_input_complete(&cinfo));
    EXPECT_EQ(0, cinfo.output_scan_number);
    int sos_marker_cnt = 1;  // read_header reads the first SOS marker
    while (!jpegli_input_complete(&cinfo)) {
      EXPECT_EQ(cinfo.input_scan_number, sos_marker_cnt);
      EXPECT_TRUE(jpegli_start_output(&cinfo, cinfo.input_scan_number));
      // start output sets output_scan_number, but does not change
      // input_scan_number
      EXPECT_EQ(cinfo.output_scan_number, cinfo.input_scan_number);
      EXPECT_EQ(cinfo.input_scan_number, sos_marker_cnt);
      VerifyScanHeader(config.jparams, &cinfo);
      TestImage output;
      ReadOutputImage(dparams, &cinfo, &output);
      output_progression0.emplace_back(std::move(output));
      // read scanlines/read raw data does not change input/output scan number
      EXPECT_EQ(cinfo.input_scan_number, sos_marker_cnt);
      EXPECT_EQ(cinfo.output_scan_number, cinfo.input_scan_number);
      EXPECT_TRUE(jpegli_finish_output(&cinfo));
      ++sos_marker_cnt;  // finish output reads the next SOS marker or EOI
    }
    jpegli_finish_decompress(&cinfo);
    return true;
  };
  ASSERT_TRUE(try_catch_block());
  jpegli_destroy_decompress(&cinfo);

  std::vector<TestImage> output_progression1;
  DecodeAllScansWithLibjpeg(CompressParams(), dparams, compressed,
                            &output_progression1);
  ASSERT_EQ(output_progression0.size(), output_progression1.size());
  for (size_t i = 0; i < output_progression0.size(); ++i) {
    const TestImage& output = output_progression0[i];
    const TestImage& expected = output_progression1[i];
    VerifyOutputImage(expected, output, 1.0);
  }
}

std::vector<TestConfig> GenerateTests(bool buffered) {
  std::vector<TestConfig> all_tests;
  {
    std::vector<std::pair<std::string, std::string>> testfiles({
        {"jxl/flower/flower.png.im_q85_420_progr.jpg", "Q85YUV420PROGR"},
        {"jxl/flower/flower.png.im_q85_420_R13B.jpg", "Q85YUV420R13B"},
        {"jxl/flower/flower.png.im_q85_444.jpg", "Q85YUV444"},
    });
    for (size_t i = 0; i < (buffered ? 1u : testfiles.size()); ++i) {
      TestConfig config;
      config.fn = testfiles[i].first;
      config.fn_desc = testfiles[i].second;
      for (size_t chunk_size : {0, 1, 64, 65536}) {
        config.dparams.chunk_size = chunk_size;
        for (size_t max_output_lines : {0, 1, 8, 16}) {
          config.dparams.max_output_lines = max_output_lines;
          config.dparams.output_mode = PIXELS;
          all_tests.push_back(config);
        }
        {
          config.dparams.max_output_lines = 16;
          config.dparams.output_mode = RAW_DATA;
          all_tests.push_back(config);
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
        {"jxl/flower/flower.png.im_q85_422.jpg", "Q85YUV422"},
        {"jxl/flower/flower.png.im_q85_440.jpg", "Q85YUV440"},
        {"jxl/flower/flower.png.im_q85_444_1x2.jpg", "Q85YUV444_1x2"},
        {"jxl/flower/flower.png.im_q85_asymmetric.jpg", "Q85Asymmetric"},
        {"jxl/flower/flower.png.im_q85_gray.jpg", "Q85Gray"},
        {"jxl/flower/flower.png.im_q85_luma_subsample.jpg", "Q85LumaSubsample"},
        {"jxl/flower/flower.png.im_q85_rgb.jpg", "Q85RGB"},
        {"jxl/flower/flower.png.im_q85_rgb_subsample_blue.jpg",
         "Q85RGBSubsampleBlue"},
        {"jxl/flower/flower_small.cmyk.jpg", "CMYK"},
    });
    for (size_t i = 0; i < (buffered ? 4u : testfiles.size()); ++i) {
      for (JpegIOMode output_mode : {PIXELS, RAW_DATA}) {
        TestConfig config;
        config.fn = testfiles[i].first;
        config.fn_desc = testfiles[i].second;
        config.dparams.output_mode = output_mode;
        all_tests.push_back(config);
      }
    }
  }

  if (buffered) {
    return all_tests;
  }

  for (int h_samp : {1, 2}) {
    for (int v_samp : {1, 2}) {
      TestConfig config;
      config.jparams.h_sampling = {h_samp, 1, 1};
      config.jparams.v_sampling = {v_samp, 1, 1};
      all_tests.push_back(config);
    }
  }
  for (JpegliDataType type :
       {JPEGLI_TYPE_UINT8, JPEGLI_TYPE_UINT16, JPEGLI_TYPE_FLOAT}) {
    for (JpegliEndianness endianness :
         {JPEGLI_NATIVE_ENDIAN, JPEGLI_LITTLE_ENDIAN, JPEGLI_BIG_ENDIAN}) {
      if (type == JPEGLI_TYPE_UINT8 && endianness != JPEGLI_NATIVE_ENDIAN) {
        continue;
      }
      for (int channels = 1; channels <= 4; ++channels) {
        TestConfig config;
        config.dparams.data_type = type;
        config.dparams.endianness = endianness;
        config.input.color_space = JCS_UNKNOWN;
        config.input.components = channels;
        config.dparams.set_out_color_space = true;
        config.dparams.out_color_space = JCS_UNKNOWN;
        all_tests.push_back(config);
      }
    }
  }
  {
    TestConfig config;
    config.dparams.crop_output = true;
    all_tests.push_back(config);
  }
  for (J_COLOR_SPACE jpeg_color_space : {JCS_RGB, JCS_YCbCr}) {
    for (J_COLOR_SPACE out_color_space : {JCS_RGB, JCS_YCbCr}) {
      if (jpeg_color_space == JCS_RGB && out_color_space == JCS_YCbCr) continue;
      TestConfig config;
      config.jparams.set_jpeg_colorspace = true;
      config.jparams.jpeg_color_space = jpeg_color_space;
      config.dparams.set_out_color_space = true;
      config.dparams.out_color_space = out_color_space;
      all_tests.push_back(config);
    }
  }
  for (J_COLOR_SPACE jpeg_color_space : {JCS_CMYK, JCS_YCCK}) {
    for (J_COLOR_SPACE out_color_space : {JCS_CMYK, JCS_YCCK}) {
      if (jpeg_color_space == JCS_CMYK && out_color_space == JCS_YCCK) continue;
      TestConfig config;
      config.input.color_space = JCS_CMYK;
      config.jparams.set_jpeg_colorspace = true;
      config.jparams.jpeg_color_space = jpeg_color_space;
      config.dparams.set_out_color_space = true;
      config.dparams.out_color_space = out_color_space;
      all_tests.push_back(config);
    }
  }
  for (int p = 0; p < kNumTestScripts; ++p) {
    TestConfig config;
    config.jparams.progressive_id = p + 1;
    all_tests.push_back(config);
  }
  for (size_t l = 0; l <= 2; ++l) {
    TestConfig config;
    config.jparams.progressive_level = l;
    all_tests.push_back(config);
  }
  for (size_t r : {1, 17, 1024}) {
    for (size_t chunk_size : {1, 65536}) {
      TestConfig config;
      config.dparams.chunk_size = chunk_size;
      config.jparams.restart_interval = r;
      all_tests.push_back(config);
    }
  }
  for (size_t rr : {1, 3, 8, 100}) {
    TestConfig config;
    config.jparams.restart_in_rows = rr;
    all_tests.push_back(config);
  }
  for (int type : {0, 1, 10, 100, 10000}) {
    for (int scale : {1, 50, 100, 200, 500}) {
      for (bool add_raw : {false, true}) {
        for (bool baseline : {true, false}) {
          if (!baseline && (add_raw || type * scale < 25500)) continue;
          TestConfig config;
          config.input.xsize = 64;
          config.input.ysize = 64;
          CustomQuantTable table;
          table.table_type = type;
          table.scale_factor = scale;
          table.force_baseline = baseline;
          table.add_raw = add_raw;
          table.Generate();
          config.jparams.quant_tables.push_back(table);
          config.jparams.quant_indexes = {0, 0, 0};
          config.compare_to_orig = true;
          all_tests.push_back(config);
        }
      }
    }
  }
  for (int qidx = 0; qidx < 8; ++qidx) {
    if (qidx == 3) continue;
    TestConfig config;
    config.input.xsize = 256;
    config.input.ysize = 256;
    config.jparams.quant_indexes = {(qidx >> 2) & 1, (qidx >> 1) & 1,
                                    (qidx >> 0) & 1};
    all_tests.push_back(config);
  }
  for (int qidx = 0; qidx < 8; ++qidx) {
    for (int slot_idx = 0; slot_idx < 2; ++slot_idx) {
      if (qidx == 0 && slot_idx == 0) continue;
      TestConfig config;
      config.input.xsize = 256;
      config.input.ysize = 256;
      config.jparams.quant_indexes = {(qidx >> 2) & 1, (qidx >> 1) & 1,
                                      (qidx >> 0) & 1};
      CustomQuantTable table;
      table.slot_idx = slot_idx;
      table.Generate();
      config.jparams.quant_tables.push_back(table);
      all_tests.push_back(config);
    }
  }
  for (int qidx = 0; qidx < 8; ++qidx) {
    for (bool xyb : {false, true}) {
      TestConfig config;
      config.input.xsize = 256;
      config.input.ysize = 256;
      config.jparams.xyb_mode = xyb;
      config.jparams.quant_indexes = {(qidx >> 2) & 1, (qidx >> 1) & 1,
                                      (qidx >> 0) & 1};
      {
        CustomQuantTable table;
        table.slot_idx = 0;
        table.Generate();
        config.jparams.quant_tables.push_back(table);
      }
      {
        CustomQuantTable table;
        table.slot_idx = 1;
        table.table_type = 20;
        table.Generate();
        config.jparams.quant_tables.push_back(table);
      }
      config.compare_to_orig = true;
      all_tests.push_back(config);
    }
  }
  for (bool xyb : {false, true}) {
    TestConfig config;
    config.input.xsize = 256;
    config.input.ysize = 256;
    config.jparams.xyb_mode = xyb;
    config.jparams.quant_indexes = {0, 1, 2};
    {
      CustomQuantTable table;
      table.slot_idx = 0;
      table.Generate();
      config.jparams.quant_tables.push_back(table);
    }
    {
      CustomQuantTable table;
      table.slot_idx = 1;
      table.table_type = 20;
      table.Generate();
      config.jparams.quant_tables.push_back(table);
    }
    {
      CustomQuantTable table;
      table.slot_idx = 2;
      table.table_type = 30;
      table.Generate();
      config.jparams.quant_tables.push_back(table);
    }
    config.compare_to_orig = true;
    all_tests.push_back(config);
  }
  for (J_COLOR_SPACE jpeg_color_space : {JCS_RGB, JCS_YCbCr}) {
    for (bool flat_dc_luma : {false, true}) {
      TestConfig config;
      config.jparams.set_jpeg_colorspace = true;
      config.jparams.jpeg_color_space = jpeg_color_space;
      config.jparams.progressive_level = 0;
      config.jparams.optimize_coding = false;
      config.jparams.use_flat_dc_luma_code = flat_dc_luma;
      all_tests.push_back(config);
    }
  }
  for (J_COLOR_SPACE jpeg_color_space : {JCS_CMYK, JCS_YCCK}) {
    for (bool flat_dc_luma : {false, true}) {
      TestConfig config;
      config.input.color_space = JCS_CMYK;
      config.jparams.set_jpeg_colorspace = true;
      config.jparams.jpeg_color_space = jpeg_color_space;
      config.jparams.progressive_level = 0;
      config.jparams.optimize_coding = false;
      config.jparams.use_flat_dc_luma_code = flat_dc_luma;
      all_tests.push_back(config);
    }
  }
  {
    TestConfig config;
    config.input.xsize = config.input.ysize = 128;
    config.jparams.comp_ids = {7, 17, 177};
    all_tests.push_back(config);
  }
  for (int override_JFIF : {-1, 0, 1}) {
    for (int override_Adobe : {-1, 0, 1}) {
      if (override_JFIF == -1 && override_Adobe == -1) continue;
      TestConfig config;
      config.input.xsize = config.input.ysize = 128;
      config.jparams.override_JFIF = override_JFIF;
      config.jparams.override_Adobe = override_Adobe;
      all_tests.push_back(config);
    }
  }
  for (int xsize : {1, 7, 8, 9, 15, 16, 17}) {
    for (int ysize : {1, 7, 8, 9, 15, 16, 17}) {
      TestConfig config;
      config.input.xsize = xsize;
      config.input.ysize = ysize;
      all_tests.push_back(config);
    }
  }
  {
    TestConfig config;
    config.input.xsize = config.input.ysize = 256;
    config.jparams.add_marker = true;
    all_tests.push_back(config);
  }
  for (int h0_samp : {1, 2, 3, 4}) {
    for (int v0_samp : {1, 2, 3, 4}) {
      for (int dxb = 0; dxb < h0_samp; ++dxb) {
        for (int dyb = 0; dyb < v0_samp; ++dyb) {
          for (int dx = 0; dx < 2; ++dx) {
            for (int dy = 0; dy < 2; ++dy) {
              TestConfig config;
              config.input.xsize = 128 + dyb * 8 + dy;
              config.input.ysize = 256 + dxb * 8 + dx;
              config.jparams.h_sampling = {h0_samp, 1, 1};
              config.jparams.v_sampling = {v0_samp, 1, 1};
              config.compare_to_orig = true;
              all_tests.push_back(config);
            }
          }
        }
      }
    }
  }
  for (int h0_samp : {1, 2, 4}) {
    for (int v0_samp : {1, 2, 4}) {
      for (int h2_samp : {1, 2, 4}) {
        for (int v2_samp : {1, 2, 4}) {
          TestConfig config;
          config.input.xsize = 137;
          config.input.ysize = 75;
          config.jparams.h_sampling = {h0_samp, 1, h2_samp};
          config.jparams.v_sampling = {v0_samp, 1, v2_samp};
          config.compare_to_orig = true;
          all_tests.push_back(config);
        }
      }
    }
  }
  for (int h0_samp : {1, 3}) {
    for (int v0_samp : {1, 3}) {
      for (int h2_samp : {1, 3}) {
        for (int v2_samp : {1, 3}) {
          TestConfig config;
          config.input.xsize = 205;
          config.input.ysize = 99;
          config.jparams.h_sampling = {h0_samp, 1, h2_samp};
          config.jparams.v_sampling = {v0_samp, 1, v2_samp};
          all_tests.push_back(config);
        }
      }
    }
  }
  for (int scale_num = 1; scale_num <= 16; ++scale_num) {
    if (scale_num == 8) continue;
    for (bool crop : {false, true}) {
      for (int samp : {1, 2}) {
        TestConfig config;
        config.jparams.h_sampling = {samp, 1, 1};
        config.jparams.v_sampling = {samp, 1, 1};
        config.dparams.scale_num = scale_num;
        config.dparams.scale_denom = 8;
        config.dparams.crop_output = crop;
        all_tests.push_back(config);
      }
    }
  }
  return all_tests;
}

std::ostream& operator<<(std::ostream& os, const DecompressParams& dparams) {
  if (dparams.chunk_size == 0) {
    os << "CompleteInput";
  } else {
    os << "InputChunks" << dparams.chunk_size;
  }
  if (dparams.max_output_lines == 0) {
    os << "CompleteOutput";
  } else {
    os << "OutputLines" << dparams.max_output_lines;
  }
  if (dparams.output_mode == RAW_DATA) {
    os << "RawDataOut";
  }
  os << IOMethodName(dparams.data_type, dparams.endianness);
  if (dparams.set_out_color_space) {
    os << "OutColor" << ColorSpaceName(dparams.out_color_space);
  }
  if (dparams.crop_output) {
    os << "Crop";
  }
  if (dparams.do_block_smoothing) {
    os << "BlockSmoothing";
  }
  if (dparams.scale_num != 1 || dparams.scale_denom != 1) {
    os << "Scale" << dparams.scale_num << "_" << dparams.scale_denom;
  }
  return os;
}

std::ostream& operator<<(std::ostream& os, const TestConfig& c) {
  if (!c.fn.empty()) {
    os << c.fn_desc;
  } else {
    os << c.input;
  }
  os << c.jparams;
  os << c.dparams;
  return os;
}

std::string TestDescription(const testing::TestParamInfo<TestConfig>& info) {
  std::stringstream name;
  name << info.param;
  return name.str();
}

JPEGLI_INSTANTIATE_TEST_SUITE_P(DecodeAPITest, DecodeAPITestParam,
                                testing::ValuesIn(GenerateTests(false)),
                                TestDescription);

JPEGLI_INSTANTIATE_TEST_SUITE_P(DecodeAPITestBuffered,
                                DecodeAPITestParamBuffered,
                                testing::ValuesIn(GenerateTests(true)),
                                TestDescription);

}  // namespace
}  // namespace jpegli
