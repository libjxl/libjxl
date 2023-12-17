// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/enc/jpg.h"

#if JPEGXL_ENABLE_JPEG
#include <jpeglib.h>
#include <setjmp.h>
#endif
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include "lib/extras/exif.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/sanitizers.h"

namespace jxl {
namespace extras {

#if JPEGXL_ENABLE_JPEG
namespace {

constexpr unsigned char kICCSignature[12] = {
    0x49, 0x43, 0x43, 0x5F, 0x50, 0x52, 0x4F, 0x46, 0x49, 0x4C, 0x45, 0x00};
constexpr int kICCMarker = JPEG_APP0 + 2;
constexpr size_t kMaxBytesInMarker = 65533;

constexpr unsigned char kExifSignature[6] = {0x45, 0x78, 0x69,
                                             0x66, 0x00, 0x00};
constexpr int kExifMarker = JPEG_APP0 + 1;

#define ARRAY_SIZE(X) (sizeof(X) / sizeof((X)[0]))

// Popular jpeg scan scripts
// The fields of the individual scans are:
// comps_in_scan, component_index[], Ss, Se, Ah, Al
static constexpr jpeg_scan_info kScanScript1[] = {
    {1, {0}, 0, 0, 0, 0},   //
    {1, {1}, 0, 0, 0, 0},   //
    {1, {2}, 0, 0, 0, 0},   //
    {1, {0}, 1, 8, 0, 0},   //
    {1, {0}, 9, 63, 0, 0},  //
    {1, {1}, 1, 63, 0, 0},  //
    {1, {2}, 1, 63, 0, 0},  //
};
static constexpr size_t kNumScans1 = ARRAY_SIZE(kScanScript1);

static constexpr jpeg_scan_info kScanScript2[] = {
    {1, {0}, 0, 0, 0, 0},   //
    {1, {1}, 0, 0, 0, 0},   //
    {1, {2}, 0, 0, 0, 0},   //
    {1, {0}, 1, 2, 0, 1},   //
    {1, {0}, 3, 63, 0, 1},  //
    {1, {0}, 1, 63, 1, 0},  //
    {1, {1}, 1, 63, 0, 0},  //
    {1, {2}, 1, 63, 0, 0},  //
};
static constexpr size_t kNumScans2 = ARRAY_SIZE(kScanScript2);

static constexpr jpeg_scan_info kScanScript3[] = {
    {1, {0}, 0, 0, 0, 0},   //
    {1, {1}, 0, 0, 0, 0},   //
    {1, {2}, 0, 0, 0, 0},   //
    {1, {0}, 1, 63, 0, 2},  //
    {1, {0}, 1, 63, 2, 1},  //
    {1, {0}, 1, 63, 1, 0},  //
    {1, {1}, 1, 63, 0, 0},  //
    {1, {2}, 1, 63, 0, 0},  //
};
static constexpr size_t kNumScans3 = ARRAY_SIZE(kScanScript3);

static constexpr jpeg_scan_info kScanScript4[] = {
    {3, {0, 1, 2}, 0, 0, 0, 1},  //
    {1, {0}, 1, 5, 0, 2},        //
    {1, {2}, 1, 63, 0, 1},       //
    {1, {1}, 1, 63, 0, 1},       //
    {1, {0}, 6, 63, 0, 2},       //
    {1, {0}, 1, 63, 2, 1},       //
    {3, {0, 1, 2}, 0, 0, 1, 0},  //
    {1, {2}, 1, 63, 1, 0},       //
    {1, {1}, 1, 63, 1, 0},       //
    {1, {0}, 1, 63, 1, 0},       //
};
static constexpr size_t kNumScans4 = ARRAY_SIZE(kScanScript4);

static constexpr jpeg_scan_info kScanScript5[] = {
    {3, {0, 1, 2}, 0, 0, 0, 1},  //
    {1, {0}, 1, 5, 0, 2},        //
    {1, {1}, 1, 5, 0, 2},        //
    {1, {2}, 1, 5, 0, 2},        //
    {1, {1}, 6, 63, 0, 2},       //
    {1, {2}, 6, 63, 0, 2},       //
    {1, {0}, 6, 63, 0, 2},       //
    {1, {0}, 1, 63, 2, 1},       //
    {1, {1}, 1, 63, 2, 1},       //
    {1, {2}, 1, 63, 2, 1},       //
    {3, {0, 1, 2}, 0, 0, 1, 0},  //
    {1, {0}, 1, 63, 1, 0},       //
    {1, {1}, 1, 63, 1, 0},       //
    {1, {2}, 1, 63, 1, 0},       //
};
static constexpr size_t kNumScans5 = ARRAY_SIZE(kScanScript5);

// default progressive mode of jpegli
static constexpr jpeg_scan_info kScanScript6[] = {
    {3, {0, 1, 2}, 0, 0, 0, 0},  //
    {1, {0}, 1, 2, 0, 0},        //
    {1, {1}, 1, 2, 0, 0},        //
    {1, {2}, 1, 2, 0, 0},        //
    {1, {0}, 3, 63, 0, 2},       //
    {1, {1}, 3, 63, 0, 2},       //
    {1, {2}, 3, 63, 0, 2},       //
    {1, {0}, 3, 63, 2, 1},       //
    {1, {1}, 3, 63, 2, 1},       //
    {1, {2}, 3, 63, 2, 1},       //
    {1, {0}, 3, 63, 1, 0},       //
    {1, {1}, 3, 63, 1, 0},       //
    {1, {2}, 3, 63, 1, 0},       //
};
static constexpr size_t kNumScans6 = ARRAY_SIZE(kScanScript6);

// Adapt RGB scan info to grayscale jpegs.
void FilterScanComponents(const jpeg_compress_struct* cinfo,
                          jpeg_scan_info* si) {
  const int all_comps_in_scan = si->comps_in_scan;
  si->comps_in_scan = 0;
  for (int j = 0; j < all_comps_in_scan; ++j) {
    const int component = si->component_index[j];
    if (component < cinfo->input_components) {
      si->component_index[si->comps_in_scan++] = component;
    }
  }
}

Status SetJpegProgression(int progressive_id,
                          std::vector<jpeg_scan_info>* scan_infos,
                          jpeg_compress_struct* cinfo) {
  if (progressive_id < 0) {
    return true;
  }
  if (progressive_id == 0) {
    jpeg_simple_progression(cinfo);
    return true;
  }
  constexpr const jpeg_scan_info* kScanScripts[] = {kScanScript1, kScanScript2,
                                                    kScanScript3, kScanScript4,
                                                    kScanScript5, kScanScript6};
  constexpr size_t kNumScans[] = {kNumScans1, kNumScans2, kNumScans3,
                                  kNumScans4, kNumScans5, kNumScans6};
  if (progressive_id > static_cast<int>(ARRAY_SIZE(kNumScans))) {
    return JXL_FAILURE("Unknown jpeg scan script id %d", progressive_id);
  }
  const jpeg_scan_info* scan_script = kScanScripts[progressive_id - 1];
  const size_t num_scans = kNumScans[progressive_id - 1];
  // filter scan script for number of components
  for (size_t i = 0; i < num_scans; ++i) {
    jpeg_scan_info scan_info = scan_script[i];
    FilterScanComponents(cinfo, &scan_info);
    if (scan_info.comps_in_scan > 0) {
      scan_infos->emplace_back(std::move(scan_info));
    }
  }
  cinfo->scan_info = scan_infos->data();
  cinfo->num_scans = scan_infos->size();
  return true;
}

bool IsSRGBEncoding(const JxlColorEncoding& c) {
  return ((c.color_space == JXL_COLOR_SPACE_RGB ||
           c.color_space == JXL_COLOR_SPACE_GRAY) &&
          c.primaries == JXL_PRIMARIES_SRGB &&
          c.white_point == JXL_WHITE_POINT_D65 &&
          c.transfer_function == JXL_TRANSFER_FUNCTION_SRGB);
}

void WriteICCProfile(jpeg_compress_struct* const cinfo,
                     const std::vector<uint8_t>& icc) {
  constexpr size_t kMaxIccBytesInMarker =
      kMaxBytesInMarker - sizeof kICCSignature - 2;
  const int num_markers =
      static_cast<int>(DivCeil(icc.size(), kMaxIccBytesInMarker));
  size_t begin = 0;
  for (int current_marker = 0; current_marker < num_markers; ++current_marker) {
    const size_t length = std::min(kMaxIccBytesInMarker, icc.size() - begin);
    jpeg_write_m_header(
        cinfo, kICCMarker,
        static_cast<unsigned int>(length + sizeof kICCSignature + 2));
    for (const unsigned char c : kICCSignature) {
      jpeg_write_m_byte(cinfo, c);
    }
    jpeg_write_m_byte(cinfo, current_marker + 1);
    jpeg_write_m_byte(cinfo, num_markers);
    for (size_t i = 0; i < length; ++i) {
      jpeg_write_m_byte(cinfo, icc[begin]);
      ++begin;
    }
  }
}
void WriteExif(jpeg_compress_struct* const cinfo,
               const std::vector<uint8_t>& exif) {
  jpeg_write_m_header(
      cinfo, kExifMarker,
      static_cast<unsigned int>(exif.size() + sizeof kExifSignature));
  for (const unsigned char c : kExifSignature) {
    jpeg_write_m_byte(cinfo, c);
  }
  for (size_t i = 0; i < exif.size(); ++i) {
    jpeg_write_m_byte(cinfo, exif[i]);
  }
}

Status SetChromaSubsampling(const std::string& subsampling,
                            jpeg_compress_struct* const cinfo) {
  const std::pair<const char*,
                  std::pair<std::array<uint8_t, 3>, std::array<uint8_t, 3>>>
      options[] = {{"444", {{{1, 1, 1}}, {{1, 1, 1}}}},
                   {"420", {{{2, 1, 1}}, {{2, 1, 1}}}},
                   {"422", {{{2, 1, 1}}, {{1, 1, 1}}}},
                   {"440", {{{1, 1, 1}}, {{2, 1, 1}}}}};
  for (const auto& option : options) {
    if (subsampling == option.first) {
      for (size_t i = 0; i < 3; i++) {
        cinfo->comp_info[i].h_samp_factor = option.second.first[i];
        cinfo->comp_info[i].v_samp_factor = option.second.second[i];
      }
      return true;
    }
  }
  return false;
}

struct JpegParams {
  int quality = 100;
  std::string chroma_subsampling = "444";
  // Libjpeg parameters
  int progressive_id = -1;
  bool optimize_coding = true;
  bool is_xyb = false;
};

Status EncodeImageJPG(const PackedImage& image, const JxlBasicInfo& info,
                      const std::vector<uint8_t>& icc,
                      std::vector<uint8_t> exif, const JpegParams& params,
                      std::vector<uint8_t>* bytes) {
  if (params.quality > 100) {
    return JXL_FAILURE("JPEG quality should be within 0-100");
  }
  if (BITS_IN_JSAMPLE != 8 || sizeof(JSAMPLE) != 1) {
    return JXL_FAILURE("Only 8 bit JSAMPLE is supported.");
  }
  jpeg_compress_struct cinfo = {};
  jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  unsigned char* buffer = nullptr;
  unsigned long size = 0;
  jpeg_mem_dest(&cinfo, &buffer, &size);
  cinfo.image_width = image.xsize;
  cinfo.image_height = image.ysize;
  cinfo.input_components = info.num_color_channels;
  cinfo.in_color_space = info.num_color_channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
  jpeg_set_defaults(&cinfo);
  cinfo.optimize_coding = params.optimize_coding;
  if (cinfo.input_components == 3) {
    JXL_RETURN_IF_ERROR(
        SetChromaSubsampling(params.chroma_subsampling, &cinfo));
  }
  if (params.is_xyb) {
    // Tell libjpeg not to convert XYB data to YCbCr.
    jpeg_set_colorspace(&cinfo, JCS_RGB);
  }
  jpeg_set_quality(&cinfo, params.quality, TRUE);
  std::vector<jpeg_scan_info> scan_infos;
  JXL_RETURN_IF_ERROR(
      SetJpegProgression(params.progressive_id, &scan_infos, &cinfo));
  jpeg_start_compress(&cinfo, TRUE);
  if (!icc.empty()) {
    WriteICCProfile(&cinfo, icc);
  }
  if (!exif.empty()) {
    ResetExifOrientation(exif);
    WriteExif(&cinfo, exif);
  }
  if (cinfo.input_components > 3 || cinfo.input_components < 0)
    return JXL_FAILURE("invalid numbers of components");

  std::vector<uint8_t> row_bytes(image.stride);
  const uint8_t* pixels = reinterpret_cast<const uint8_t*>(image.pixels());
  if (cinfo.num_components == (int)image.format.num_channels &&
      image.format.data_type == JXL_TYPE_UINT8) {
    for (size_t y = 0; y < info.ysize; ++y) {
      memcpy(&row_bytes[0], pixels + y * image.stride, image.stride);
      JSAMPROW row[] = {row_bytes.data()};
      jpeg_write_scanlines(&cinfo, row, 1);
    }
  } else if (image.format.data_type == JXL_TYPE_UINT8) {
    for (size_t y = 0; y < info.ysize; ++y) {
      const uint8_t* image_row = pixels + y * image.stride;
      for (size_t x = 0; x < info.xsize; ++x) {
        const uint8_t* image_pixel = image_row + x * image.pixel_stride();
        memcpy(&row_bytes[x * cinfo.num_components], image_pixel,
               cinfo.num_components);
      }
      JSAMPROW row[] = {row_bytes.data()};
      jpeg_write_scanlines(&cinfo, row, 1);
    }
  } else {
    for (size_t y = 0; y < info.ysize; ++y) {
      const uint8_t* image_row = pixels + y * image.stride;
      for (size_t x = 0; x < info.xsize; ++x) {
        const uint8_t* image_pixel = image_row + x * image.pixel_stride();
        for (int c = 0; c < cinfo.num_components; ++c) {
          uint32_t val16 = (image_pixel[2 * c] << 8) + image_pixel[2 * c + 1];
          row_bytes[x * cinfo.num_components + c] = (val16 + 128) / 257;
        }
      }
      JSAMPROW row[] = {row_bytes.data()};
      jpeg_write_scanlines(&cinfo, row, 1);
    }
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  bytes->resize(size);
  // Compressed image data is initialized by libjpeg, which we are not
  // instrumenting with msan.
  msan::UnpoisonMemory(buffer, size);
  std::copy_n(buffer, size, bytes->data());
  std::free(buffer);
  return true;
}

class JPEGEncoder : public Encoder {
  std::vector<JxlPixelFormat> AcceptedFormats() const override {
    std::vector<JxlPixelFormat> formats;
    for (const uint32_t num_channels : {1, 2, 3, 4}) {
      for (JxlEndianness endianness : {JXL_BIG_ENDIAN, JXL_LITTLE_ENDIAN}) {
        formats.push_back(JxlPixelFormat{/*num_channels=*/num_channels,
                                         /*data_type=*/JXL_TYPE_UINT8,
                                         /*endianness=*/endianness,
                                         /*align=*/0});
      }
      formats.push_back(JxlPixelFormat{/*num_channels=*/num_channels,
                                       /*data_type=*/JXL_TYPE_UINT16,
                                       /*endianness=*/JXL_BIG_ENDIAN,
                                       /*align=*/0});
    }
    return formats;
  }
  Status Encode(const PackedPixelFile& ppf, EncodedImage* encoded_image,
                ThreadPool* pool = nullptr) const override {
    JXL_RETURN_IF_ERROR(VerifyBasicInfo(ppf.info));
    JpegParams params;
    for (const auto& it : options()) {
      if (it.first == "q") {
        std::istringstream is(it.second);
        JXL_RETURN_IF_ERROR(static_cast<bool>(is >> params.quality));
      } else if (it.first == "chroma_subsampling") {
        params.chroma_subsampling = it.second;
      } else if (it.first == "progressive") {
        std::istringstream is(it.second);
        JXL_RETURN_IF_ERROR(static_cast<bool>(is >> params.progressive_id));
      } else if (it.first == "optimize" && it.second == "OFF") {
        params.optimize_coding = false;
      }
    }
    params.is_xyb = (ppf.color_encoding.color_space == JXL_COLOR_SPACE_XYB);
    std::vector<uint8_t> icc;
    if (!IsSRGBEncoding(ppf.color_encoding)) {
      icc = ppf.icc;
    }
    encoded_image->bitstreams.clear();
    encoded_image->bitstreams.reserve(ppf.frames.size());
    for (const auto& frame : ppf.frames) {
      JXL_RETURN_IF_ERROR(VerifyPackedImage(frame.color, ppf.info));
      encoded_image->bitstreams.emplace_back();
      JXL_RETURN_IF_ERROR(EncodeImageJPG(frame.color, ppf.info, icc,
                                         ppf.metadata.exif, params,
                                         &encoded_image->bitstreams.back()));
    }
    return true;
  }
};

}  // namespace
#endif

std::unique_ptr<Encoder> GetJPEGEncoder() {
#if JPEGXL_ENABLE_JPEG
  return jxl::make_unique<JPEGEncoder>();
#else
  return nullptr;
#endif
}

}  // namespace extras
}  // namespace jxl
