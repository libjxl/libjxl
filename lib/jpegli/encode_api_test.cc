// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/* clang-format off */
#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>
/* clang-format on */

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

enum InputColor {
  COLOR_SRGB,
  COLOR_GRAY,
};

struct TestConfig {
  InputColor color = COLOR_SRGB;
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

// Verifies that an image encoded with libjpegli can be decoded with libjpeg.
void TestDecodedImage(const TestConfig& config,
                      const std::vector<uint8_t>& compressed,
                      const std::vector<uint8_t>& orig, size_t xsize,
                      size_t ysize, size_t num_channels) {
  jpeg_decompress_struct cinfo;
  // cinfo is initialized by libjpeg, which we are not instrumenting with
  // msan, therefore we need to initialize cinfo here.
  jxl::msan::UnpoisonMemory(&cinfo, sizeof(cinfo));
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
  EXPECT_EQ(xsize, cinfo.image_width);
  EXPECT_EQ(ysize, cinfo.image_height);
  EXPECT_EQ(num_channels, cinfo.num_components);
  cinfo.buffered_image = TRUE;
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
  size_t stride = xsize * num_channels;
  std::vector<uint8_t> output(ysize * stride);
  for (size_t y = 0; y < cinfo.image_height; ++y) {
    JSAMPROW rows[] = {reinterpret_cast<JSAMPLE*>(&output[y * stride])};
    jxl::msan::UnpoisonMemory(
        rows[0], sizeof(JSAMPLE) * cinfo.output_components * cinfo.image_width);
    EXPECT_EQ(1, jpeg_read_scanlines(&cinfo, rows, 1));
  }
  EXPECT_TRUE(jpeg_finish_output(&cinfo));
  EXPECT_TRUE(jpeg_finish_decompress(&cinfo));
  jpeg_destroy_decompress(&cinfo);

  ASSERT_EQ(output.size(), orig.size());
  const double mul = 1.0 / 255.0;
  double diff2 = 0.0;
  for (size_t i = 0; i < orig.size(); ++i) {
    double sample_orig = orig[i] * mul;
    double sample_output = output[i] * mul;
    double diff = sample_orig - sample_output;
    diff2 += diff * diff;
  }
  double rms = std::sqrt(diff2 / orig.size()) / mul;
  printf("rms: %f\n", rms);
  EXPECT_LE(rms, config.max_dist);
}

class EncodeAPITestParam : public ::testing::TestWithParam<TestConfig> {};

TEST_P(EncodeAPITestParam, TestAPI) {
  TestConfig config = GetParam();
  std::string testimage = (config.color == COLOR_SRGB)
                              ? "jxl/flower/flower.pnm"
                              : "jxl/flower/flower.pgm";
  const std::vector<uint8_t> origdata = ReadTestData(testimage);
  // These has to be volatile to make setjmp/longjmp work.
  volatile size_t xsize, ysize, num_channels, bitdepth;
  std::vector<uint8_t> orig;
  ASSERT_TRUE(
      ReadPNM(origdata, &xsize, &ysize, &num_channels, &bitdepth, &orig));
  ASSERT_EQ(8, bitdepth);
  if (config.color == COLOR_GRAY) {
    ASSERT_EQ(1, num_channels);
  }
  jpeg_compress_struct cinfo;
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
  jpegli_create_compress(&cinfo);
  unsigned char* buffer = nullptr;
  unsigned long size = 0;
  jpegli_mem_dest(&cinfo, &buffer, &size);
  cinfo.image_width = xsize;
  cinfo.image_height = ysize;
  cinfo.input_components = num_channels;
  cinfo.in_color_space = num_channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
  if (config.xyb_mode) {
    jpegli_set_xyb_mode(&cinfo);
  }
  jpegli_set_defaults(&cinfo);
  if (config.custom_sampling) {
    for (size_t c = 0; c < num_channels; ++c) {
      cinfo.comp_info[c].h_samp_factor = config.h_sampling[c];
      cinfo.comp_info[c].v_samp_factor = config.v_sampling[c];
    }
  }
  if (config.progressive_id > 0) {
    ASSERT_LE(config.progressive_id, kNumTestScripts);
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
  size_t stride = xsize * cinfo.input_components;
  for (size_t y = 0; y < ysize; ++y) {
    JSAMPROW row[] = {orig.data() + y * stride};
    jpegli_write_scanlines(&cinfo, row, 1);
  }
  jpegli_finish_compress(&cinfo);
  jpegli_destroy_compress(&cinfo);
  std::vector<uint8_t> compressed;
  compressed.resize(size);
  std::copy_n(buffer, size, compressed.data());
  std::free(buffer);
  double bpp = compressed.size() * 8.0 / (xsize * ysize);
  printf("bpp: %f\n", bpp);
  EXPECT_LT(bpp, config.max_bpp);
  TestDecodedImage(config, compressed, orig, xsize, ysize, num_channels);
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
    config.color = COLOR_GRAY;
    config.max_bpp = 1.15;
    config.max_dist = 1.35;
    all_tests.push_back(config);
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

std::ostream& operator<<(std::ostream& os, const TestConfig& c) {
  if (c.color == COLOR_SRGB) {
    os << "SRGB";
  } else if (c.color == COLOR_GRAY) {
    os << "GRAY";
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
