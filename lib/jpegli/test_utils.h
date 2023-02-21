// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_TEST_UTILS_H_
#define LIB_JPEGLI_TEST_UTILS_H_

/* clang-format off */
#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>
/* clang-format on */

#include <cmath>
#include <vector>

#include "lib/jpegli/encode.h"
#include "lib/jpegli/testing.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/sanitizers.h"

#if !defined(TEST_DATA_PATH)
#include "tools/cpp/runfiles/runfiles.h"
#endif

namespace jpegli {

template <typename T1, typename T2>
constexpr inline T1 DivCeil(T1 a, T2 b) {
  return (a + b - 1) / b;
}

struct MyClientData {
  jmp_buf env;
  unsigned char* buffer = nullptr;
  unsigned long size = 0;
  unsigned char* table_stream = nullptr;
  unsigned long table_stream_size = 0;
};

#define ERROR_HANDLER_SETUP(action)                                           \
  jpeg_error_mgr jerr;                                                        \
  cinfo.err = jpegli_std_error(&jerr);                                        \
  if (setjmp(data.env)) {                                                     \
    action;                                                                   \
  }                                                                           \
  cinfo.client_data = reinterpret_cast<void*>(&data);                         \
  cinfo.err->error_exit = [](j_common_ptr cinfo) {                            \
    (*cinfo->err->output_message)(cinfo);                                     \
    MyClientData* data = reinterpret_cast<MyClientData*>(cinfo->client_data); \
    if (data->buffer) free(data->buffer);                                     \
    jpegli_destroy(cinfo);                                                    \
    longjmp(data->env, 1);                                                    \
  };

static constexpr int kSpecialMarker = 0xe5;
static constexpr uint8_t kMarkerData[] = {0, 1, 255, 0, 17};

#define ARRAY_SIZE(X) (sizeof(X) / sizeof((X)[0]))

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
    {ARRAY_SIZE(kScript1), kScript1},
    {ARRAY_SIZE(kScript2), kScript2},
    {ARRAY_SIZE(kScript3), kScript3},
};
static constexpr size_t kNumTestScripts = ARRAY_SIZE(kTestScript);

enum JpegIOMode {
  PIXELS,
  RAW_DATA,
  COEFFICIENTS,
};

struct CustomQuantTable {
  int slot_idx = 0;
  uint16_t table_type = 0;
  int scale_factor = 100;
  bool add_raw = false;
  bool force_baseline = true;
  std::vector<unsigned int> basic_table;
  std::vector<unsigned int> quantval;
  void Generate() {
    basic_table.resize(DCTSIZE2);
    quantval.resize(DCTSIZE2);
    switch (table_type) {
      case 0: {
        for (int k = 0; k < DCTSIZE2; ++k) {
          basic_table[k] = k + 1;
        }
        break;
      }
      default:
        for (int k = 0; k < DCTSIZE2; ++k) {
          basic_table[k] = table_type;
        }
    }
    for (int k = 0; k < DCTSIZE2; ++k) {
      quantval[k] = (basic_table[k] * scale_factor + 50U) / 100U;
      quantval[k] = std::max(quantval[k], 1U);
      quantval[k] = std::min(quantval[k], 65535U);
      if (!add_raw) {
        quantval[k] = std::min(quantval[k], force_baseline ? 255U : 32767U);
      }
    }
  }
};

struct TestImage {
  size_t xsize = 0;
  size_t ysize = 0;
  J_COLOR_SPACE color_space = JCS_RGB;
  size_t components = 3;
  std::vector<uint8_t> pixels;
  std::vector<std::vector<uint8_t>> raw_data;
  std::vector<std::vector<JCOEF>> coeffs;
};

struct CompressParams {
  int quality = 90;
  bool set_jpeg_colorspace = false;
  J_COLOR_SPACE jpeg_color_space = JCS_UNKNOWN;
  std::vector<int> quant_indexes;
  std::vector<CustomQuantTable> quant_tables;
  std::vector<int> h_sampling;
  std::vector<int> v_sampling;
  std::vector<int> comp_ids;
  int override_JFIF = -1;
  int override_Adobe = -1;
  bool add_marker = false;
  int progressive_id = 0;
  int progressive_level = -1;
  int restart_interval = 0;
  int restart_in_rows = 0;
  bool optimize_coding = true;
  bool use_flat_dc_luma_code = false;
  bool xyb_mode = false;
  bool libjpeg_mode = false;

  int h_samp(int c) const { return h_sampling.empty() ? 1 : h_sampling[c]; }
  int v_samp(int c) const { return v_sampling.empty() ? 1 : v_sampling[c]; }
  int max_h_sample() const {
    auto it = std::max_element(h_sampling.begin(), h_sampling.end());
    return it == h_sampling.end() ? 1 : *it;
  }
  int max_v_sample() const {
    auto it = std::max_element(v_sampling.begin(), v_sampling.end());
    return it == v_sampling.end() ? 1 : *it;
  }
  int comp_width(const TestImage& input, int c) const {
    return DivCeil(input.xsize * h_samp(c), max_h_sample() * DCTSIZE) * DCTSIZE;
  }
  int comp_height(const TestImage& input, int c) const {
    return DivCeil(input.ysize * v_samp(c), max_v_sample() * DCTSIZE) * DCTSIZE;
  }
};

#if defined(TEST_DATA_PATH)
template <typename S>
std::string GetTestDataPath(S filename) {
  return std::string(TEST_DATA_PATH "/") + filename;
}
#else
using bazel::tools::cpp::runfiles::Runfiles;
const std::unique_ptr<Runfiles> kRunfiles(Runfiles::Create(""));
template <typename S>
std::string GetTestDataPath(S filename) {
  return kRunfiles->Rlocation("__main__/testdata/" + std::string(filename));
}
#endif

static inline std::vector<uint8_t> ReadTestData(const std::string& filename) {
  std::string full_path = GetTestDataPath(filename);
  std::vector<uint8_t> data;
  fprintf(stderr, "ReadTestData %s\n", full_path.c_str());
  JXL_CHECK(jxl::ReadFile(full_path, &data));
  printf("Test data %s is %d bytes long.\n", filename.c_str(),
         static_cast<int>(data.size()));
  return data;
}

class PNMParser {
 public:
  explicit PNMParser(const uint8_t* data, const size_t len)
      : pos_(data), end_(data + len) {}

  // Sets "pos" to the first non-header byte/pixel on success.
  bool ParseHeader(const uint8_t** pos, size_t* xsize, size_t* ysize,
                   size_t* num_channels, size_t* bitdepth) {
    if (pos_[0] != 'P' || (pos_[1] != '5' && pos_[1] != '6')) {
      fprintf(stderr, "Invalid PNM header.");
      return false;
    }
    *num_channels = (pos_[1] == '5' ? 1 : 3);
    pos_ += 2;

    size_t maxval;
    if (!SkipWhitespace() || !ParseUnsigned(xsize) || !SkipWhitespace() ||
        !ParseUnsigned(ysize) || !SkipWhitespace() || !ParseUnsigned(&maxval) ||
        !SkipWhitespace()) {
      return false;
    }
    if (maxval == 0 || maxval >= 65536) {
      fprintf(stderr, "Invalid maxval value.\n");
      return false;
    }
    bool found_bitdepth = false;
    for (int bits = 1; bits <= 16; ++bits) {
      if (maxval == (1u << bits) - 1) {
        *bitdepth = bits;
        found_bitdepth = true;
        break;
      }
    }
    if (!found_bitdepth) {
      fprintf(stderr, "Invalid maxval value.\n");
      return false;
    }

    *pos = pos_;
    return true;
  }

 private:
  static bool IsLineBreak(const uint8_t c) { return c == '\r' || c == '\n'; }
  static bool IsWhitespace(const uint8_t c) {
    return IsLineBreak(c) || c == '\t' || c == ' ';
  }

  bool ParseUnsigned(size_t* number) {
    if (pos_ == end_ || *pos_ < '0' || *pos_ > '9') {
      fprintf(stderr, "Expected unsigned number.\n");
      return false;
    }
    *number = 0;
    while (pos_ < end_ && *pos_ >= '0' && *pos_ <= '9') {
      *number *= 10;
      *number += *pos_ - '0';
      ++pos_;
    }

    return true;
  }

  bool SkipWhitespace() {
    if (pos_ == end_ || !IsWhitespace(*pos_)) {
      fprintf(stderr, "Expected whitespace.\n");
      return false;
    }
    while (pos_ < end_ && IsWhitespace(*pos_)) {
      ++pos_;
    }
    return true;
  }

  const uint8_t* pos_;
  const uint8_t* const end_;
};

static inline bool ReadPNM(const std::vector<uint8_t>& data, size_t* xsize,
                           size_t* ysize, size_t* num_channels,
                           size_t* bitdepth, std::vector<uint8_t>* pixels) {
  if (data.size() < 2) {
    fprintf(stderr, "PNM file too small.\n");
    return false;
  }
  PNMParser parser(data.data(), data.size());
  const uint8_t* pos = nullptr;
  if (!parser.ParseHeader(&pos, xsize, ysize, num_channels, bitdepth)) {
    return false;
  }
  pixels->resize(data.data() + data.size() - pos);
  memcpy(&(*pixels)[0], pos, pixels->size());
  return true;
}

void SetNumChannels(J_COLOR_SPACE colorspace, size_t* channels) {
  if (colorspace == JCS_GRAYSCALE) {
    *channels = 1;
  } else if (colorspace == JCS_RGB || colorspace == JCS_YCbCr) {
    *channels = 3;
  } else if (colorspace == JCS_CMYK || colorspace == JCS_YCCK) {
    *channels = 4;
  } else if (colorspace == JCS_UNKNOWN) {
    ASSERT_LE(*channels, 4);
  } else {
    FAIL();
  }
}

void ConvertPixel(const uint8_t* input_rgb, uint8_t* out,
                  J_COLOR_SPACE colorspace, size_t num_channels) {
  const float kMul = 255.0f;
  const float r = input_rgb[0] / kMul;
  const float g = input_rgb[1] / kMul;
  const float b = input_rgb[2] / kMul;
  if (colorspace == JCS_GRAYSCALE) {
    const float Y = 0.299f * r + 0.587f * g + 0.114f * b;
    out[0] = static_cast<uint8_t>(std::round(Y * kMul));
  } else if (colorspace == JCS_RGB || colorspace == JCS_UNKNOWN) {
    for (size_t c = 0; c < num_channels; ++c) {
      out[c] = input_rgb[std::min<size_t>(2, c)];
    }
  } else if (colorspace == JCS_YCbCr) {
    float Y = 0.299f * r + 0.587f * g + 0.114f * b;
    float Cb = -0.168736f * r - 0.331264f * g + 0.5f * b + 0.5f;
    float Cr = 0.5f * r - 0.418688f * g - 0.081312f * b + 0.5f;
    out[0] = static_cast<uint8_t>(std::round(Y * kMul));
    out[1] = static_cast<uint8_t>(std::round(Cb * kMul));
    out[2] = static_cast<uint8_t>(std::round(Cr * kMul));
  } else if (colorspace == JCS_CMYK) {
    float K = 1.0f - std::max(r, std::max(g, b));
    float scaleK = 1.0f / (1.0f - K);
    float C = (1.0f - K - r) * scaleK;
    float M = (1.0f - K - g) * scaleK;
    float Y = (1.0f - K - b) * scaleK;
    out[0] = static_cast<uint8_t>(std::round(C * kMul));
    out[1] = static_cast<uint8_t>(std::round(M * kMul));
    out[2] = static_cast<uint8_t>(std::round(Y * kMul));
    out[3] = static_cast<uint8_t>(std::round(K * kMul));
  } else {
    JXL_ABORT("Colorspace %d not supported", colorspace);
  }
}

void GeneratePixels(TestImage* img) {
  const std::vector<uint8_t> imgdata = ReadTestData("jxl/flower/flower.pnm");
  size_t xsize, ysize, channels, bitdepth;
  std::vector<uint8_t> pixels;
  ASSERT_TRUE(ReadPNM(imgdata, &xsize, &ysize, &channels, &bitdepth, &pixels));
  if (img->xsize == 0) img->xsize = xsize;
  if (img->ysize == 0) img->ysize = ysize;
  ASSERT_LE(img->xsize, xsize);
  ASSERT_LE(img->ysize, ysize);
  ASSERT_EQ(3, channels);
  ASSERT_EQ(8, bitdepth);
  size_t in_bytes_per_pixel = channels;
  size_t in_stride = xsize * in_bytes_per_pixel;
  size_t x0 = (xsize - img->xsize) / 2;
  size_t y0 = (ysize - img->ysize) / 2;
  SetNumChannels(img->color_space, &img->components);
  size_t out_bytes_per_pixel = img->components;
  size_t out_stride = img->xsize * out_bytes_per_pixel;
  img->pixels.resize(img->ysize * out_stride);
  for (size_t iy = 0; iy < img->ysize; ++iy) {
    size_t y = y0 + iy;
    for (size_t ix = 0; ix < img->xsize; ++ix) {
      size_t x = x0 + ix;
      size_t idx_in = y * in_stride + x * in_bytes_per_pixel;
      size_t idx_out = iy * out_stride + ix * out_bytes_per_pixel;
      ConvertPixel(&pixels[idx_in], &img->pixels[idx_out], img->color_space,
                   img->components);
    }
  }
}

void GenerateRawData(const CompressParams& jparams, TestImage* img) {
  for (size_t c = 0; c < img->components; ++c) {
    size_t xsize = jparams.comp_width(*img, c);
    size_t ysize = jparams.comp_height(*img, c);
    size_t factor_y = jparams.max_v_sample() / jparams.v_samp(c);
    size_t factor_x = jparams.max_h_sample() / jparams.h_samp(c);
    size_t factor = factor_x * factor_y;
    std::vector<uint8_t> plane(ysize * xsize);
    size_t bytes_per_pixel = img->components;
    for (size_t y = 0; y < ysize; ++y) {
      for (size_t x = 0; x < xsize; ++x) {
        int result = 0;
        for (size_t iy = 0; iy < factor_y; ++iy) {
          size_t yy = std::min(y * factor_y + iy, img->ysize - 1);
          for (size_t ix = 0; ix < factor_x; ++ix) {
            size_t xx = std::min(x * factor_x + ix, img->xsize - 1);
            size_t pixel_ix = (yy * img->xsize + xx) * bytes_per_pixel + c;
            result += img->pixels[pixel_ix];
          }
        }
        result = static_cast<uint8_t>((result + factor / 2) / factor);
        plane[y * xsize + x] = result;
      }
    }
    img->raw_data.emplace_back(std::move(plane));
  }
}

void GenerateCoeffs(const CompressParams& jparams, TestImage* img) {
  for (size_t c = 0; c < img->components; ++c) {
    size_t xsize_blocks = jparams.comp_width(*img, c) / DCTSIZE;
    size_t ysize_blocks = jparams.comp_height(*img, c) / DCTSIZE;
    std::vector<JCOEF> plane(ysize_blocks * xsize_blocks * DCTSIZE2);
    for (size_t by = 0; by < ysize_blocks; ++by) {
      for (size_t bx = 0; bx < xsize_blocks; ++bx) {
        for (size_t k = 0; k < DCTSIZE2; ++k) {
          plane[(by * xsize_blocks + bx) * DCTSIZE2 + k] = (bx - by) / (k + 1);
        }
      }
    }
    img->coeffs.emplace_back(std::move(plane));
  }
}

bool EncodeWithJpegli(const TestImage& input, const CompressParams& jparams,
                      std::vector<uint8_t>* compressed) {
  MyClientData data;
  jpeg_compress_struct cinfo;
  ERROR_HANDLER_SETUP(return false);
  jpegli_create_compress(&cinfo);
  jpegli_mem_dest(&cinfo, &data.buffer, &data.size);
  cinfo.image_width = input.xsize;
  cinfo.image_height = input.ysize;
  cinfo.input_components = input.components;
  cinfo.in_color_space = input.color_space;
  if (jparams.xyb_mode) {
    jpegli_set_xyb_mode(&cinfo);
  }
  jpegli_set_defaults(&cinfo);
  if (jparams.override_JFIF >= 0) {
    cinfo.write_JFIF_header = jparams.override_JFIF;
  }
  if (jparams.override_Adobe >= 0) {
    cinfo.write_Adobe_marker = jparams.override_Adobe;
  }
  if (jparams.set_jpeg_colorspace) {
    jpegli_set_colorspace(&cinfo, jparams.jpeg_color_space);
  }
  if (!jparams.comp_ids.empty()) {
    for (int c = 0; c < cinfo.num_components; ++c) {
      cinfo.comp_info[c].component_id = jparams.comp_ids[c];
    }
  }
  if (!jparams.h_sampling.empty()) {
    for (int c = 0; c < cinfo.num_components; ++c) {
      cinfo.comp_info[c].h_samp_factor = jparams.h_sampling[c];
      cinfo.comp_info[c].v_samp_factor = jparams.v_sampling[c];
    }
  }
  if (!jparams.quant_indexes.empty()) {
    for (int c = 0; c < cinfo.num_components; ++c) {
      cinfo.comp_info[c].quant_tbl_no = jparams.quant_indexes[c];
    }
    for (const auto& table : jparams.quant_tables) {
      if (table.add_raw) {
        cinfo.quant_tbl_ptrs[table.slot_idx] =
            jpegli_alloc_quant_table((j_common_ptr)&cinfo);
        for (int k = 0; k < DCTSIZE2; ++k) {
          cinfo.quant_tbl_ptrs[table.slot_idx]->quantval[k] = table.quantval[k];
        }
        cinfo.quant_tbl_ptrs[table.slot_idx]->sent_table = FALSE;
      } else {
        jpegli_add_quant_table(&cinfo, table.slot_idx, &table.basic_table[0],
                               table.scale_factor, table.force_baseline);
      }
    }
  }
  if (jparams.progressive_id > 0) {
    const ScanScript& script = kTestScript[jparams.progressive_id - 1];
    cinfo.scan_info = script.scans;
    cinfo.num_scans = script.num_scans;
  } else if (jparams.progressive_level >= 0) {
    jpegli_set_progressive_level(&cinfo, jparams.progressive_level);
  }
  cinfo.restart_interval = jparams.restart_interval;
  cinfo.restart_in_rows = jparams.restart_in_rows;
  cinfo.optimize_coding = jparams.optimize_coding;
  cinfo.raw_data_in = !input.raw_data.empty();
  if (!jparams.optimize_coding && jparams.use_flat_dc_luma_code) {
    JHUFF_TBL* tbl = cinfo.dc_huff_tbl_ptrs[0];
    memset(tbl, 0, sizeof(*tbl));
    tbl->bits[4] = 15;
    for (int i = 0; i < 15; ++i) tbl->huffval[i] = i;
  }
  jpegli_set_quality(&cinfo, jparams.quality, TRUE);
  if (jparams.libjpeg_mode) {
    jpegli_enable_adaptive_quantization(&cinfo, FALSE);
    jpegli_use_standard_quant_tables(&cinfo);
    jpegli_set_progressive_level(&cinfo, 0);
  }
  if (input.coeffs.empty()) {
    jpegli_start_compress(&cinfo, TRUE);
    if (jparams.add_marker) {
      jpegli_write_marker(&cinfo, kSpecialMarker, kMarkerData,
                          sizeof(kMarkerData));
    }
  }
  if (cinfo.raw_data_in) {
    // Need to copy because jpeg API requires non-const pointers.
    std::vector<std::vector<uint8_t>> raw_data = input.raw_data;
    size_t max_lines = jparams.max_v_sample() * DCTSIZE;
    std::vector<std::vector<JSAMPROW>> rowdata(cinfo.num_components);
    std::vector<JSAMPARRAY> data(cinfo.num_components);
    for (int c = 0; c < cinfo.num_components; ++c) {
      rowdata[c].resize(jparams.v_samp(c) * DCTSIZE);
      data[c] = &rowdata[c][0];
    }
    while (cinfo.next_scanline < cinfo.image_height) {
      for (int c = 0; c < cinfo.num_components; ++c) {
        size_t cwidth = cinfo.comp_info[c].width_in_blocks * DCTSIZE;
        size_t cheight = cinfo.comp_info[c].height_in_blocks * DCTSIZE;
        size_t num_lines = jparams.v_samp(c) * DCTSIZE;
        size_t y0 = (cinfo.next_scanline / max_lines) * num_lines;
        for (size_t i = 0; i < num_lines; ++i) {
          rowdata[c][i] =
              (y0 + i < cheight ? &raw_data[c][(y0 + i) * cwidth] : nullptr);
        }
      }
      size_t num_lines = jpegli_write_raw_data(&cinfo, &data[0], max_lines);
      EXPECT_EQ(num_lines, max_lines);
    }
  } else if (!input.coeffs.empty()) {
    j_common_ptr comptr = reinterpret_cast<j_common_ptr>(&cinfo);
    jvirt_barray_ptr* coef_arrays = reinterpret_cast<jvirt_barray_ptr*>((
        *cinfo.mem->alloc_small)(
        comptr, JPOOL_IMAGE, cinfo.num_components * sizeof(jvirt_barray_ptr)));
    for (int c = 0; c < cinfo.num_components; ++c) {
      size_t xsize_blocks = jparams.comp_width(input, c) / DCTSIZE;
      size_t ysize_blocks = jparams.comp_height(input, c) / DCTSIZE;
      coef_arrays[c] = (*cinfo.mem->request_virt_barray)(
          comptr, JPOOL_IMAGE, FALSE, xsize_blocks, ysize_blocks, 1);
    }
    jpegli_write_coefficients(&cinfo, coef_arrays);
    if (jparams.add_marker) {
      jpegli_write_marker(&cinfo, kSpecialMarker, kMarkerData,
                          sizeof(kMarkerData));
    }
    for (int c = 0; c < cinfo.num_components; ++c) {
      jpeg_component_info* comp = &cinfo.comp_info[c];
      for (size_t by = 0; by < comp->height_in_blocks; ++by) {
        JBLOCKARRAY ba = (*cinfo.mem->access_virt_barray)(
            comptr, coef_arrays[c], by, 1, true);
        size_t stride = comp->width_in_blocks * sizeof(JBLOCK);
        size_t offset = by * comp->width_in_blocks * DCTSIZE2;
        memcpy(ba[0], &input.coeffs[c][offset], stride);
      }
    }
  } else {
    size_t stride = cinfo.image_width * cinfo.input_components;
    std::vector<uint8_t> row_bytes(stride);
    for (size_t y = 0; y < cinfo.image_height; ++y) {
      memcpy(&row_bytes[0], &input.pixels[y * stride], stride);
      JSAMPROW row[] = {row_bytes.data()};
      jpegli_write_scanlines(&cinfo, row, 1);
    }
  }
  jpegli_finish_compress(&cinfo);
  jpegli_destroy_compress(&cinfo);
  compressed->resize(data.size);
  std::copy_n(data.buffer, data.size, compressed->data());
  std::free(data.buffer);
  return true;
}

// Verifies that an image encoded with libjpegli can be decoded with libjpeg,
// and checks that the jpeg coding metadata matches jparams.
void DecodeWithLibjpeg(const CompressParams& jparams,
                       const std::vector<uint8_t>& compressed,
                       JpegIOMode output_mode, TestImage* output) {
  jpeg_decompress_struct cinfo = {};
  const auto try_catch_block = [&]() {
    jpeg_error_mgr jerr;
    jmp_buf env;
    cinfo.err = jpeg_std_error(&jerr);
    if (setjmp(env)) {
      FAIL();
    }
    cinfo.client_data = reinterpret_cast<void*>(&env);
    cinfo.err->error_exit = [](j_common_ptr cinfo) {
      (*cinfo->err->output_message)(cinfo);
      jmp_buf* env = reinterpret_cast<jmp_buf*>(cinfo->client_data);
      longjmp(*env, 1);
    };
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, compressed.data(), compressed.size());
    if (jparams.add_marker) {
      jpeg_save_markers(&cinfo, kSpecialMarker, 0xffff);
    }
    EXPECT_EQ(JPEG_REACHED_SOS,
              jpeg_read_header(&cinfo, /*require_image=*/TRUE));
    jxl::msan::UnpoisonMemory(
        cinfo.comp_info, cinfo.num_components * sizeof(cinfo.comp_info[0]));
    output->xsize = cinfo.image_width;
    output->ysize = cinfo.image_height;
    output->components = cinfo.num_components;
    cinfo.buffered_image = TRUE;
    cinfo.out_color_space = output->color_space;
    if (jparams.override_JFIF >= 0) {
      EXPECT_EQ(cinfo.saw_JFIF_marker, jparams.override_JFIF);
    }
    if (jparams.override_Adobe >= 0) {
      EXPECT_EQ(cinfo.saw_Adobe_marker, jparams.override_Adobe);
    }
    if (output->color_space == JCS_UNKNOWN) {
      cinfo.jpeg_color_space = JCS_UNKNOWN;
    }
    if (jparams.add_marker) {
      bool marker_found = false;
      for (jpeg_saved_marker_ptr marker = cinfo.marker_list; marker != nullptr;
           marker = marker->next) {
        jxl::msan::UnpoisonMemory(marker, sizeof(*marker));
        jxl::msan::UnpoisonMemory(marker->data, marker->data_length);
        if (marker->marker == kSpecialMarker &&
            marker->data_length == sizeof(kMarkerData) &&
            memcmp(marker->data, kMarkerData, sizeof(kMarkerData)) == 0) {
          marker_found = true;
        }
      }
      EXPECT_TRUE(marker_found);
    }
    if (!jparams.comp_ids.empty()) {
      for (int i = 0; i < cinfo.num_components; ++i) {
        EXPECT_EQ(cinfo.comp_info[i].component_id, jparams.comp_ids[i]);
      }
    }
    if (!jparams.h_sampling.empty()) {
      for (int i = 0; i < cinfo.num_components; ++i) {
        EXPECT_EQ(cinfo.comp_info[i].h_samp_factor, jparams.h_sampling[i]);
        EXPECT_EQ(cinfo.comp_info[i].v_samp_factor, jparams.v_sampling[i]);
      }
    }
    if (!jparams.quant_indexes.empty()) {
      for (int i = 0; i < cinfo.num_components; ++i) {
        EXPECT_EQ(cinfo.comp_info[i].quant_tbl_no, jparams.quant_indexes[i]);
      }
      for (const auto& table : jparams.quant_tables) {
        JQUANT_TBL* quant_table = cinfo.quant_tbl_ptrs[table.slot_idx];
        jxl::msan::UnpoisonMemory(quant_table, sizeof(*quant_table));
        ASSERT_TRUE(quant_table != nullptr);
        for (int k = 0; k < DCTSIZE2; ++k) {
          EXPECT_EQ(quant_table->quantval[k], table.quantval[k]);
        }
      }
    }
    if (output_mode == COEFFICIENTS) {
      j_common_ptr comptr = reinterpret_cast<j_common_ptr>(&cinfo);
      jvirt_barray_ptr* coef_arrays = jpeg_read_coefficients(&cinfo);
      ASSERT_TRUE(coef_arrays != nullptr);
      for (int c = 0; c < cinfo.num_components; ++c) {
        jpeg_component_info* comp = &cinfo.comp_info[c];
        std::vector<JCOEF> coeffs(comp->width_in_blocks *
                                  comp->height_in_blocks * DCTSIZE2);
        for (size_t by = 0; by < comp->height_in_blocks; ++by) {
          JBLOCKARRAY ba = (*cinfo.mem->access_virt_barray)(
              comptr, coef_arrays[c], by, 1, true);
          size_t stride = comp->width_in_blocks * sizeof(JBLOCK);
          size_t offset = by * comp->width_in_blocks * DCTSIZE2;
          memcpy(&coeffs[offset], ba[0], stride);
        }
        output->coeffs.emplace_back(std::move(coeffs));
      }
    } else {
      EXPECT_TRUE(jpeg_start_decompress(&cinfo));
      while (!jpeg_input_complete(&cinfo)) {
        EXPECT_GT(cinfo.input_scan_number, 0);
        EXPECT_TRUE(jpeg_start_output(&cinfo, cinfo.input_scan_number));
        if (jparams.progressive_id > 0) {
          ASSERT_LE(jparams.progressive_id, kNumTestScripts);
          const ScanScript& script = kTestScript[jparams.progressive_id - 1];
          ASSERT_LE(cinfo.input_scan_number, script.num_scans);
          const jpeg_scan_info& scan =
              script.scans[cinfo.input_scan_number - 1];
          ASSERT_EQ(cinfo.comps_in_scan, scan.comps_in_scan);
          for (int i = 0; i < cinfo.comps_in_scan; ++i) {
            EXPECT_EQ(cinfo.cur_comp_info[i]->component_index,
                      scan.component_index[i]);
          }
          EXPECT_EQ(cinfo.Ss, scan.Ss);
          EXPECT_EQ(cinfo.Se, scan.Se);
          EXPECT_EQ(cinfo.Ah, scan.Ah);
          EXPECT_EQ(cinfo.Al, scan.Al);
        }
        EXPECT_TRUE(jpeg_finish_output(&cinfo));
        if (jparams.restart_interval > 0) {
          EXPECT_EQ(cinfo.restart_interval, jparams.restart_interval);
        } else if (jparams.restart_in_rows > 0) {
          EXPECT_EQ(cinfo.restart_interval,
                    jparams.restart_in_rows * cinfo.MCUs_per_row);
        }
      }
      EXPECT_TRUE(jpeg_start_output(&cinfo, cinfo.input_scan_number));
      size_t stride = cinfo.image_width * cinfo.out_color_components;
      output->pixels.resize(cinfo.image_height * stride);
      for (size_t y = 0; y < cinfo.image_height; ++y) {
        JSAMPROW rows[] = {
            reinterpret_cast<JSAMPLE*>(&output->pixels[y * stride])};
        jxl::msan::UnpoisonMemory(
            rows[0],
            sizeof(JSAMPLE) * cinfo.output_components * cinfo.image_width);
        EXPECT_EQ(1, jpeg_read_scanlines(&cinfo, rows, 1));
      }
      EXPECT_TRUE(jpeg_finish_output(&cinfo));
    }
    EXPECT_TRUE(jpeg_finish_decompress(&cinfo));
  };
  try_catch_block();
  jpeg_destroy_decompress(&cinfo);
}

void VerifyOutputImage(const TestImage& input, const TestImage& output,
                       double max_rms) {
  EXPECT_EQ(output.xsize, input.xsize);
  EXPECT_EQ(output.ysize, input.ysize);
  EXPECT_EQ(output.components, input.components);
  if (!input.coeffs.empty()) {
    ASSERT_EQ(input.coeffs.size(), input.components);
    ASSERT_EQ(output.coeffs.size(), input.components);
    for (size_t c = 0; c < input.components; ++c) {
      ASSERT_EQ(output.coeffs[c].size(), input.coeffs[c].size());
      EXPECT_EQ(0, memcmp(input.coeffs[c].data(), output.coeffs[c].data(),
                          input.coeffs[c].size()));
    }
  } else {
    ASSERT_EQ(output.pixels.size(), input.pixels.size());
    const double mul = 1.0 / 255.0;
    double diff2 = 0.0;
    for (size_t i = 0; i < input.pixels.size(); ++i) {
      double sample_orig = input.pixels[i] * mul;
      double sample_output = output.pixels[i] * mul;
      double diff = sample_orig - sample_output;
      diff2 += diff * diff;
    }
    double rms = std::sqrt(diff2 / input.pixels.size()) / mul;
    printf("rms: %f\n", rms);
    EXPECT_LE(rms, max_rms);
  }
}

}  // namespace jpegli

#endif  // LIB_JPEGLI_TEST_UTILS_H_
