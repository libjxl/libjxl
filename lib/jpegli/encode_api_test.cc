// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <algorithm>
#include <cmath>
#include <vector>

#include "lib/jpegli/encode.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/test_utils.h"
#include "lib/jpegli/testing.h"
#include "lib/jxl/sanitizers.h"

namespace jpegli {
namespace {

struct TestConfig {
  TestImage input;
  CompressParams jparams;
  JpegIOMode input_mode = PIXELS;
  double max_bpp;
  double max_dist;
};

class EncodeAPITestParam : public ::testing::TestWithParam<TestConfig> {};

TEST_P(EncodeAPITestParam, TestAPI) {
  TestConfig config = GetParam();
  GeneratePixels(&config.input);
  if (config.input_mode == RAW_DATA) {
    GenerateRawData(config.jparams, &config.input);
  } else if (config.input_mode == COEFFICIENTS) {
    GenerateCoeffs(config.jparams, &config.input);
  }
  std::vector<uint8_t> compressed;
  ASSERT_TRUE(EncodeWithJpegli(config.input, config.jparams, &compressed));
  double bpp =
      compressed.size() * 8.0 / (config.input.xsize * config.input.ysize);
  printf("bpp: %f\n", bpp);
  EXPECT_LT(bpp, config.max_bpp);
  DecompressParams dparams;
  dparams.output_mode =
      config.input_mode == COEFFICIENTS ? COEFFICIENTS : PIXELS;
  dparams.set_out_color_space = true;
  dparams.out_color_space = config.input.color_space;
  TestImage output;
  DecodeWithLibjpeg(config.jparams, dparams, compressed, &output);
  VerifyOutputImage(config.input, output, config.max_dist);
}

TEST(EncodeAPITest, AbbreviatedStreams) {
  uint8_t* table_stream = nullptr;
  unsigned long table_stream_size = 0;
  uint8_t* data_stream = nullptr;
  unsigned long data_stream_size = 0;
  {
    jpeg_compress_struct cinfo;
    const auto try_catch_block = [&]() -> bool {
      ERROR_HANDLER_SETUP(jpegli);
      jpegli_create_compress(&cinfo);
      jpegli_mem_dest(&cinfo, &table_stream, &table_stream_size);
      cinfo.input_components = 3;
      cinfo.in_color_space = JCS_RGB;
      jpegli_set_defaults(&cinfo);
      jpegli_write_tables(&cinfo);
      jpegli_mem_dest(&cinfo, &data_stream, &data_stream_size);
      cinfo.image_width = 1;
      cinfo.image_height = 1;
      cinfo.optimize_coding = false;
      jpegli_set_progressive_level(&cinfo, 0);
      jpegli_start_compress(&cinfo, FALSE);
      JSAMPLE image[3] = {0};
      JSAMPROW row[] = {image};
      jpegli_write_scanlines(&cinfo, row, 1);
      jpegli_finish_compress(&cinfo);
      return true;
    };
    EXPECT_TRUE(try_catch_block());
    EXPECT_LT(data_stream_size, 50);
    jpegli_destroy_compress(&cinfo);
  }
  {
    jpeg_decompress_struct cinfo = {};
    const auto try_catch_block = [&]() -> bool {
      ERROR_HANDLER_SETUP(jpeg);
      jpeg_create_decompress(&cinfo);
      jpeg_mem_src(&cinfo, table_stream, table_stream_size);
      jpeg_read_header(&cinfo, FALSE);
      jpeg_mem_src(&cinfo, data_stream, data_stream_size);
      jpeg_read_header(&cinfo, TRUE);
      EXPECT_EQ(1, cinfo.image_width);
      EXPECT_EQ(1, cinfo.image_height);
      EXPECT_EQ(3, cinfo.num_components);
      jpeg_start_decompress(&cinfo);
      JSAMPLE image[3] = {0};
      JSAMPROW row[] = {image};
      jpeg_read_scanlines(&cinfo, row, 1);
      jxl::msan::UnpoisonMemory(image, 3);
      EXPECT_EQ(0, image[0]);
      EXPECT_EQ(0, image[1]);
      EXPECT_EQ(0, image[2]);
      jpeg_finish_decompress(&cinfo);
      return true;
    };
    EXPECT_TRUE(try_catch_block());
    jpeg_destroy_decompress(&cinfo);
  }
  if (table_stream) free(table_stream);
  if (data_stream) free(data_stream);
}

std::vector<TestConfig> GenerateTests() {
  std::vector<TestConfig> all_tests;
  {
    TestConfig config;
    config.max_bpp = 1.45;
    config.max_dist = 2.2;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.jparams.quality = 100;
    config.max_bpp = 5.9;
    config.max_dist = 0.6;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.jparams.quality = 80;
    config.max_bpp = 0.95;
    config.max_dist = 2.9;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.jparams.h_sampling = {2, 1, 1};
    config.jparams.v_sampling = {2, 1, 1};
    config.max_bpp = 1.25;
    config.max_dist = 2.9;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.jparams.h_sampling = {1, 1, 1};
    config.jparams.v_sampling = {2, 1, 1};
    config.max_bpp = 1.35;
    config.max_dist = 2.5;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.jparams.h_sampling = {2, 1, 1};
    config.jparams.v_sampling = {1, 1, 1};
    config.max_bpp = 1.35;
    config.max_dist = 2.5;
    all_tests.push_back(config);
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
          config.max_bpp = 2.5;
          config.max_dist = 12.0;
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
          config.max_bpp = 2.5;
          config.max_dist = 10.0;
          all_tests.push_back(config);
        }
      }
    }
  }
  for (int h0_samp : {1, 2, 3, 4}) {
    for (int v0_samp : {1, 2, 3, 4}) {
      TestConfig config;
      config.input.xsize = 217;
      config.input.ysize = 129;
      config.jparams.h_sampling = {h0_samp, 1, 1};
      config.jparams.v_sampling = {v0_samp, 1, 1};
      config.max_bpp = 2.0;
      config.max_dist = 5.5;
      all_tests.push_back(config);
    }
  }
  for (int p = 0; p < kNumTestScripts; ++p) {
    TestConfig config;
    config.jparams.progressive_id = p + 1;
    config.max_bpp = 1.5;
    config.max_dist = 2.2;
    all_tests.push_back(config);
  }
  for (size_t l = 0; l <= 2; ++l) {
    TestConfig config;
    config.jparams.progressive_level = l;
    config.max_bpp = 1.5;
    config.max_dist = 2.2;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.jparams.xyb_mode = true;
    config.max_bpp = 1.5;
    config.max_dist = 3.5;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.jparams.libjpeg_mode = true;
    config.max_bpp = 2.1;
    config.max_dist = 1.7;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.input.color_space = JCS_YCbCr;
    config.max_bpp = 1.45;
    config.max_dist = 1.35;
    all_tests.push_back(config);
  }
  for (bool xyb : {false, true}) {
    TestConfig config;
    config.input.color_space = JCS_GRAYSCALE;
    config.jparams.xyb_mode = xyb;
    config.max_bpp = 1.25;
    config.max_dist = 1.4;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.input.color_space = JCS_RGB;
    config.jparams.set_jpeg_colorspace = true;
    config.jparams.jpeg_color_space = JCS_RGB;
    config.jparams.xyb_mode = false;
    config.max_bpp = 3.75;
    config.max_dist = 1.4;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.input.color_space = JCS_CMYK;
    config.max_bpp = 3.75;
    config.max_dist = 1.4;
    all_tests.push_back(config);
    config.jparams.set_jpeg_colorspace = true;
    config.jparams.jpeg_color_space = JCS_YCCK;
    config.max_bpp = 3.2;
    config.max_dist = 1.7;
    all_tests.push_back(config);
  }
  for (int channels = 1; channels <= 4; ++channels) {
    TestConfig config;
    config.input.color_space = JCS_UNKNOWN;
    config.input.components = channels;
    config.max_bpp = 1.25 * channels;
    config.max_dist = 1.4;
    all_tests.push_back(config);
  }
  for (size_t r : {1, 3, 17, 1024}) {
    TestConfig config;
    config.jparams.restart_interval = r;
    config.max_bpp = 1.5 + 5.5 / r;
    config.max_dist = 2.2;
    all_tests.push_back(config);
  }
  for (size_t rr : {1, 3, 8, 100}) {
    TestConfig config;
    config.jparams.restart_in_rows = rr;
    config.max_bpp = 1.5;
    config.max_dist = 2.2;
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
          float q = (type == 0 ? 16 : type) * scale * 0.01f;
          if (baseline && !add_raw) q = std::max(1.0f, std::min(255.0f, q));
          config.max_bpp = 1.3f + 25.0f / q;
          config.max_dist = 0.6f + 0.25f * q;
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
    config.max_bpp = 2.6;
    config.max_dist = 2.5;
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
      config.max_bpp = 2.6;
      config.max_dist = 2.75;
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
      config.max_bpp = 1.9;
      config.max_dist = 3.75;
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
    config.max_bpp = 1.5;
    config.max_dist = 3.75;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.jparams.comp_ids = {7, 17, 177};
    config.input.xsize = config.input.ysize = 128;
    config.max_bpp = 2.1;
    config.max_dist = 2.4;
    all_tests.push_back(config);
  }
  for (int override_JFIF : {-1, 0, 1}) {
    for (int override_Adobe : {-1, 0, 1}) {
      if (override_JFIF == -1 && override_Adobe == -1) continue;
      TestConfig config;
      config.input.xsize = config.input.ysize = 128;
      config.jparams.override_JFIF = override_JFIF;
      config.jparams.override_Adobe = override_Adobe;
      config.max_bpp = 2.1;
      config.max_dist = 2.4;
      all_tests.push_back(config);
    }
  }
  {
    TestConfig config;
    config.input.xsize = config.input.ysize = 256;
    config.max_bpp = 1.7;
    config.max_dist = 2.3;
    config.jparams.add_marker = true;
    all_tests.push_back(config);
  }
  for (JpegIOMode input_mode : {PIXELS, RAW_DATA}) {
    TestConfig config;
    config.input.xsize = config.input.ysize = 256;
    config.input_mode = input_mode;
    if (input_mode == RAW_DATA) {
      config.input.color_space = JCS_YCbCr;
    }
    config.jparams.progressive_level = 0;
    config.jparams.optimize_coding = false;
    config.max_bpp = 1.8;
    config.max_dist = 2.3;
    all_tests.push_back(config);
    config.jparams.use_flat_dc_luma_code = true;
    all_tests.push_back(config);
  }
  for (int xsize : {640, 641, 648, 649}) {
    for (int ysize : {640, 641, 648, 649}) {
      for (int h_sampling : {1, 2}) {
        for (int v_sampling : {1, 2}) {
          if (h_sampling == 1 && v_sampling == 1) continue;
          TestConfig config;
          config.input.xsize = xsize;
          config.input.ysize = ysize;
          config.input.color_space = JCS_YCbCr;
          config.jparams.h_sampling = {h_sampling, 1, 1};
          config.jparams.v_sampling = {v_sampling, 1, 1};
          config.input_mode = RAW_DATA;
          config.max_bpp = 1.7;
          config.max_dist = 2.0;
          all_tests.push_back(config);
          config.input_mode = COEFFICIENTS;
          if (xsize & 1) {
            config.jparams.add_marker = true;
          }
          config.max_bpp = 24.0;
          all_tests.push_back(config);
        }
      }
    }
  }
  for (JpegliDataType data_type : {JPEGLI_TYPE_UINT16, JPEGLI_TYPE_FLOAT}) {
    for (JpegliEndianness endianness :
         {JPEGLI_LITTLE_ENDIAN, JPEGLI_BIG_ENDIAN, JPEGLI_NATIVE_ENDIAN}) {
      J_COLOR_SPACE colorspace[4] = {JCS_GRAYSCALE, JCS_UNKNOWN, JCS_RGB,
                                     JCS_CMYK};
      float max_bpp[4] = {1.25, 2.5, 1.45, 3.75};
      for (int channels = 1; channels <= 4; ++channels) {
        TestConfig config;
        config.input.data_type = data_type;
        config.input.endianness = endianness;
        config.input.components = channels;
        config.input.color_space = colorspace[channels - 1];
        config.max_bpp = max_bpp[channels - 1];
        config.max_dist = 2.2;
        all_tests.push_back(config);
      }
    }
  }
  return all_tests;
};

std::ostream& operator<<(std::ostream& os, const TestConfig& c) {
  os << c.input;
  os << c.jparams;
  if (c.input_mode == RAW_DATA) {
    os << "RawDataIn";
  } else if (c.input_mode == COEFFICIENTS) {
    os << "WriteCoeffs";
  }
  return os;
}

std::string TestDescription(
    const testing::TestParamInfo<EncodeAPITestParam::ParamType>& info) {
  std::stringstream name;
  name << info.param;
  return name.str();
}

JPEGLI_INSTANTIATE_TEST_SUITE_P(EncodeAPITest, EncodeAPITestParam,
                                testing::ValuesIn(GenerateTests()),
                                TestDescription);
}  // namespace
}  // namespace jpegli
