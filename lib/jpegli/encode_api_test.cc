// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/* clang-format off */
#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>
/* clang-format on */

#include <algorithm>
#include <cmath>
#include <vector>

#include "gtest/gtest.h"
#include "lib/jpegli/encode.h"
#include "lib/jpegli/test_utils.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/sanitizers.h"

#define ARRAYSIZE(X) (sizeof(X) / sizeof((X)[0]))

namespace jpegli {
namespace {

static constexpr jpeg_scan_info kScript1[] = {
    {3, {0, 1, 2}, 0, 0, 0, 0},
    {1, {0}, 1, 63, 0, 0},
    {1, {1}, 1, 63, 0, 0},
    {1, {2}, 1, 63, 0, 0},
};
static constexpr jpeg_scan_info kScript2[] = {
    {1, {0}, 0, 0, 0, 0},  {1, {1}, 0, 0, 0, 0},  {1, {2}, 0, 0, 0, 0},
    {1, {0}, 1, 63, 0, 0}, {1, {1}, 1, 63, 0, 0}, {1, {2}, 1, 63, 0, 0},
};
static constexpr jpeg_scan_info kScript3[] = {
    {3, {0, 1, 2}, 0, 0, 0, 0}, {1, {0}, 1, 63, 0, 1}, {1, {1}, 1, 63, 0, 1},
    {1, {2}, 1, 63, 0, 1},      {1, {0}, 1, 63, 1, 0}, {1, {1}, 1, 63, 1, 0},
    {1, {2}, 1, 63, 1, 0},
};

struct ScanScript {
  size_t num_scans;
  const jpeg_scan_info* scans;
};

static constexpr ScanScript kTestScript[] = {
    {ARRAYSIZE(kScript1), kScript1},
    {ARRAYSIZE(kScript2), kScript2},
    {ARRAYSIZE(kScript3), kScript3},
};
static constexpr size_t kNumTestScripts = ARRAYSIZE(kTestScript);

struct TestConfig {
  size_t xsize = 0;
  size_t ysize = 0;
  J_COLOR_SPACE in_color_space = JCS_RGB;
  bool set_jpeg_colorspace = false;
  J_COLOR_SPACE jpeg_color_space = JCS_UNKNOWN;
  size_t input_components = 3;
  std::vector<uint8_t> pixels;
  int quality = 90;
  bool custom_sampling = false;
  int h_sampling[3] = {1, 1, 1};
  int v_sampling[3] = {1, 1, 1};
  int progressive_id = 0;
  int progressive_level = -1;
  int restart_interval = 0;
  bool xyb_mode = false;
  bool libjpeg_mode = false;
  double max_bpp;
  double max_dist;
};

void SetNumChannels(J_COLOR_SPACE colorspace, size_t* channels) {
  if (colorspace == JCS_GRAYSCALE) {
    *channels = 1;
  } else if (colorspace == JCS_RGB || colorspace == JCS_YCbCr) {
    *channels = 3;
  } else if (colorspace == JCS_UNKNOWN) {
    ASSERT_LE(*channels, 3);
  } else {
    FAIL();
  }
}

void ConvertPixel(const uint8_t* input_rgb, uint8_t* out,
                  J_COLOR_SPACE colorspace, size_t num_channels) {
  const float r = input_rgb[0];
  const float g = input_rgb[1];
  const float b = input_rgb[2];
  if (colorspace == JCS_GRAYSCALE) {
    const float Y = 0.299f * r + 0.587f * g + 0.114f * b;
    out[0] = static_cast<uint8_t>(std::round(Y));
  } else if (colorspace == JCS_RGB || colorspace == JCS_UNKNOWN) {
    memcpy(out, input_rgb, num_channels);
  } else if (colorspace == JCS_YCbCr) {
    float Y = 0.299f * r + 0.587f * g + 0.114f * b;
    float Cb = -0.168736f * r - 0.331264f * g + 0.5f * b + 128.0f;
    float Cr = 0.5f * r - 0.418688f * g - 0.081312f * b + 128.0f;
    out[0] = static_cast<uint8_t>(std::round(Y));
    out[1] = static_cast<uint8_t>(std::round(Cb));
    out[2] = static_cast<uint8_t>(std::round(Cr));
  } else {
    JXL_ABORT("Colorspace %d not supported", colorspace);
  }
}

void GeneratePixels(TestConfig* config) {
  const std::vector<uint8_t> imgdata = ReadTestData("jxl/flower/flower.pnm");
  size_t xsize, ysize, channels, bitdepth;
  std::vector<uint8_t> pixels;
  ASSERT_TRUE(ReadPNM(imgdata, &xsize, &ysize, &channels, &bitdepth, &pixels));
  if (config->xsize == 0) config->xsize = xsize;
  if (config->ysize == 0) config->ysize = ysize;
  ASSERT_LE(config->xsize, xsize);
  ASSERT_LE(config->ysize, ysize);
  ASSERT_EQ(3, channels);
  ASSERT_EQ(8, bitdepth);
  size_t in_bytes_per_pixel = channels;
  size_t in_stride = xsize * in_bytes_per_pixel;
  size_t x0 = (xsize - config->xsize) / 2;
  size_t y0 = (ysize - config->ysize) / 2;
  SetNumChannels(config->in_color_space, &config->input_components);
  size_t out_bytes_per_pixel = config->input_components;
  size_t out_stride = config->xsize * out_bytes_per_pixel;
  config->pixels.resize(config->ysize * out_stride);
  for (size_t iy = 0; iy < config->ysize; ++iy) {
    size_t y = y0 + iy;
    for (size_t ix = 0; ix < config->xsize; ++ix) {
      size_t x = x0 + ix;
      size_t idx_in = y * in_stride + x * in_bytes_per_pixel;
      size_t idx_out = iy * out_stride + ix * out_bytes_per_pixel;
      ConvertPixel(&pixels[idx_in], &config->pixels[idx_out],
                   config->in_color_space, config->input_components);
    }
  }
}

// Verifies that an image encoded with libjpegli can be decoded with libjpeg.
void TestDecodedImage(const TestConfig& config,
                      const std::vector<uint8_t>& compressed) {
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
  EXPECT_EQ(JPEG_REACHED_SOS, jpeg_read_header(&cinfo, /*require_image=*/TRUE));
  EXPECT_EQ(config.xsize, cinfo.image_width);
  EXPECT_EQ(config.ysize, cinfo.image_height);
  EXPECT_EQ(config.input_components, cinfo.num_components);
  cinfo.buffered_image = TRUE;
  cinfo.out_color_space = config.in_color_space;
  if (config.in_color_space == JCS_UNKNOWN) {
    cinfo.jpeg_color_space = JCS_UNKNOWN;
  }
  EXPECT_TRUE(jpeg_start_decompress(&cinfo));
#if !JXL_MEMORY_SANITIZER
  if (config.custom_sampling) {
    for (int i = 0; i < cinfo.num_components; ++i) {
      EXPECT_EQ(cinfo.comp_info[i].h_samp_factor, config.h_sampling[i]);
      EXPECT_EQ(cinfo.comp_info[i].v_samp_factor, config.v_sampling[i]);
    }
  }
#endif
  while (!jpeg_input_complete(&cinfo)) {
    EXPECT_GT(cinfo.input_scan_number, 0);
    EXPECT_TRUE(jpeg_start_output(&cinfo, cinfo.input_scan_number));
    if (config.progressive_id > 0) {
      ASSERT_LE(config.progressive_id, kNumTestScripts);
      const ScanScript& script = kTestScript[config.progressive_id - 1];
      ASSERT_LE(cinfo.input_scan_number, script.num_scans);
      const jpeg_scan_info& scan = script.scans[cinfo.input_scan_number - 1];
      ASSERT_EQ(cinfo.comps_in_scan, scan.comps_in_scan);
#if !JXL_MEMORY_SANITIZER
      for (int i = 0; i < cinfo.comps_in_scan; ++i) {
        EXPECT_EQ(cinfo.cur_comp_info[i]->component_index,
                  scan.component_index[i]);
      }
#endif
      EXPECT_EQ(cinfo.Ss, scan.Ss);
      EXPECT_EQ(cinfo.Se, scan.Se);
      EXPECT_EQ(cinfo.Ah, scan.Ah);
      EXPECT_EQ(cinfo.Al, scan.Al);
    }
    EXPECT_TRUE(jpeg_finish_output(&cinfo));
  }
  EXPECT_TRUE(jpeg_start_output(&cinfo, cinfo.input_scan_number));
  size_t stride = cinfo.image_width * cinfo.out_color_components;
  std::vector<uint8_t> output(cinfo.image_height * stride);
  for (size_t y = 0; y < cinfo.image_height; ++y) {
    JSAMPROW rows[] = {reinterpret_cast<JSAMPLE*>(&output[y * stride])};
    jxl::msan::UnpoisonMemory(
        rows[0], sizeof(JSAMPLE) * cinfo.output_components * cinfo.image_width);
    EXPECT_EQ(1, jpeg_read_scanlines(&cinfo, rows, 1));
  }
  EXPECT_TRUE(jpeg_finish_output(&cinfo));
  EXPECT_TRUE(jpeg_finish_decompress(&cinfo));
  jpeg_destroy_decompress(&cinfo);

  double bpp = compressed.size() * 8.0 / (config.xsize * config.ysize);
  printf("bpp: %f\n", bpp);
  EXPECT_LT(bpp, config.max_bpp);

  ASSERT_EQ(output.size(), config.pixels.size());
  const double mul = 1.0 / 255.0;
  double diff2 = 0.0;
  for (size_t i = 0; i < config.pixels.size(); ++i) {
    double sample_orig = config.pixels[i] * mul;
    double sample_output = output[i] * mul;
    double diff = sample_orig - sample_output;
    diff2 += diff * diff;
  }
  double rms = std::sqrt(diff2 / config.pixels.size()) / mul;
  printf("rms: %f\n", rms);
  EXPECT_LE(rms, config.max_dist);
}

bool EncodeWithJpegli(const TestConfig& config,
                      std::vector<uint8_t>* compressed) {
  unsigned char* buffer = nullptr;
  unsigned long size = 0;
  jpeg_compress_struct cinfo;
  jpeg_error_mgr jerr;
  cinfo.err = jpegli_std_error(&jerr);
  jmp_buf env;
  if (setjmp(env)) {
    if (buffer) std::free(buffer);
    return false;
  }
  cinfo.client_data = static_cast<void*>(&env);
  cinfo.err->error_exit = [](j_common_ptr cinfo) {
    (*cinfo->err->output_message)(cinfo);
    jmp_buf* env = static_cast<jmp_buf*>(cinfo->client_data);
    longjmp(*env, 1);
  };
  jpegli_create_compress(&cinfo);
  jpegli_mem_dest(&cinfo, &buffer, &size);
  cinfo.image_width = config.xsize;
  cinfo.image_height = config.ysize;
  cinfo.input_components = config.input_components;
  cinfo.in_color_space = config.in_color_space;
  if (config.xyb_mode) {
    jpegli_set_xyb_mode(&cinfo);
  }
  jpegli_set_defaults(&cinfo);
  if (config.set_jpeg_colorspace) {
    jpegli_set_colorspace(&cinfo, config.jpeg_color_space);
  }
  if (config.custom_sampling) {
    for (int c = 0; c < cinfo.num_components; ++c) {
      cinfo.comp_info[c].h_samp_factor = config.h_sampling[c];
      cinfo.comp_info[c].v_samp_factor = config.v_sampling[c];
    }
  }
  if (config.progressive_id > 0) {
    const ScanScript& script = kTestScript[config.progressive_id - 1];
    cinfo.scan_info = script.scans;
    cinfo.num_scans = script.num_scans;
  } else if (config.progressive_level >= 0) {
    jpegli_set_progressive_level(&cinfo, config.progressive_level);
  }
  cinfo.restart_interval = config.restart_interval;
  cinfo.optimize_coding = TRUE;
  jpegli_set_quality(&cinfo, config.quality, TRUE);
  if (config.libjpeg_mode) {
    jpegli_enable_adaptive_quantization(&cinfo, FALSE);
    jpegli_use_standard_quant_tables(&cinfo);
    jpegli_set_progressive_level(&cinfo, 0);
  }
  jpegli_start_compress(&cinfo, TRUE);
  size_t stride = cinfo.image_width * cinfo.input_components;
  std::vector<uint8_t> row_bytes(stride);
  for (size_t y = 0; y < cinfo.image_height; ++y) {
    memcpy(&row_bytes[0], &config.pixels[y * stride], stride);
    JSAMPROW row[] = {row_bytes.data()};
    jpegli_write_scanlines(&cinfo, row, 1);
  }
  jpegli_finish_compress(&cinfo);
  jpegli_destroy_compress(&cinfo);
  compressed->resize(size);
  std::copy_n(buffer, size, compressed->data());
  std::free(buffer);
  return true;
}

class EncodeAPITestParam : public ::testing::TestWithParam<TestConfig> {};

TEST_P(EncodeAPITestParam, TestAPI) {
  TestConfig config = GetParam();
  GeneratePixels(&config);
  std::vector<uint8_t> compressed;
  ASSERT_TRUE(EncodeWithJpegli(config, &compressed));
  TestDecodedImage(config, compressed);
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
    config.quality = 100;
    config.max_bpp = 5.9;
    config.max_dist = 0.6;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.quality = 80;
    config.max_bpp = 0.95;
    config.max_dist = 2.9;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.custom_sampling = true;
    config.h_sampling[0] = 2;
    config.v_sampling[0] = 2;
    config.max_bpp = 1.25;
    config.max_dist = 2.9;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.custom_sampling = true;
    config.h_sampling[0] = 1;
    config.v_sampling[0] = 2;
    config.max_bpp = 1.35;
    config.max_dist = 2.5;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.custom_sampling = true;
    config.h_sampling[0] = 2;
    config.v_sampling[0] = 1;
    config.max_bpp = 1.35;
    config.max_dist = 2.5;
    all_tests.push_back(config);
  }
  {
    for (size_t p = 0; p < kNumTestScripts; ++p) {
      TestConfig config;
      config.progressive_id = p + 1;
      config.max_bpp = 1.5;
      config.max_dist = 2.2;
      all_tests.push_back(config);
    }
  }
  {
    for (size_t l = 0; l <= 2; ++l) {
      TestConfig config;
      config.progressive_level = l;
      config.max_bpp = 1.5;
      config.max_dist = 2.2;
      all_tests.push_back(config);
    }
  }
  {
    TestConfig config;
    config.xyb_mode = true;
    config.max_bpp = 1.5;
    config.max_dist = 3.5;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.libjpeg_mode = true;
    config.max_bpp = 2.1;
    config.max_dist = 1.7;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.in_color_space = JCS_YCbCr;
    config.max_bpp = 1.45;
    config.max_dist = 1.3;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.in_color_space = JCS_GRAYSCALE;
    for (bool xyb : {false, true}) {
      config.xyb_mode = xyb;
      config.max_bpp = 1.25;
      config.max_dist = 1.4;
      all_tests.push_back(config);
    }
  }
  {
    TestConfig config;
    config.in_color_space = JCS_RGB;
    config.set_jpeg_colorspace = true;
    config.jpeg_color_space = JCS_RGB;
    config.xyb_mode = false;
    config.max_bpp = 3.75;
    config.max_dist = 1.4;
    all_tests.push_back(config);
  }
  {
    TestConfig config;
    config.in_color_space = JCS_UNKNOWN;
    for (int channels = 1; channels <= 3; ++channels) {
      config.input_components = channels;
      config.max_bpp = 1.25 * channels;
      config.max_dist = 1.4;
      all_tests.push_back(config);
    }
  }
  {
    for (size_t r : {1, 3, 17, 1024}) {
      TestConfig config;
      config.restart_interval = r;
      config.max_bpp = 1.5 + 5.5 / r;
      config.max_dist = 2.2;
      all_tests.push_back(config);
    }
  }
  return all_tests;
};

std::string ColorSpaceName(J_COLOR_SPACE colorspace) {
  switch (colorspace) {
    case JCS_UNKNOWN:
      return "UNKNOWN";
    case JCS_GRAYSCALE:
      return "GRAYSCALE";
    case JCS_RGB:
      return "RGB";
    case JCS_YCbCr:
      return "YCbCr";
    default:
      return "";
  }
}

std::ostream& operator<<(std::ostream& os, const TestConfig& c) {
  os << ColorSpaceName(c.in_color_space);
  if (c.in_color_space == JCS_UNKNOWN) {
    os << c.input_components;
  }
  if (c.set_jpeg_colorspace) {
    os << "to" << ColorSpaceName(c.jpeg_color_space);
  }
  os << "Q" << c.quality;
  if (c.custom_sampling) {
    os << "SAMP";
    for (int i = 0; i < 3; ++i) {
      os << "_";
      os << c.h_sampling[i] << "x" << c.v_sampling[i];
    }
  }
  if (c.progressive_id > 0) {
    os << "P" << c.progressive_id;
  }
  if (c.restart_interval > 0) {
    os << "R" << c.restart_interval;
  }
  if (c.progressive_level >= 0) {
    os << "PL" << c.progressive_level;
  }
  if (c.xyb_mode) {
    os << "XYB";
  } else if (c.libjpeg_mode) {
    os << "Libjpeg";
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
