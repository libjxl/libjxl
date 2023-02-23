// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/encode.h"

#include <cmath>
#include <initializer_list>
#include <vector>

#include "lib/jpegli/adaptive_quantization.h"
#include "lib/jpegli/bitstream.h"
#include "lib/jpegli/color_transform.h"
#include "lib/jpegli/dct.h"
#include "lib/jpegli/encode_internal.h"
#include "lib/jpegli/entropy_coding.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/memory_manager.h"
#include "lib/jpegli/quant.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/span.h"

namespace jpegli {

constexpr size_t kMaxBytesInMarker = 65533;

void CheckState(j_compress_ptr cinfo, int state) {
  if (cinfo->global_state != state) {
    JPEGLI_ERROR("Unexpected global state %d [expected %d]",
                 cinfo->global_state, state);
  }
}

void CheckState(j_compress_ptr cinfo, int state1, int state2) {
  if (cinfo->global_state != state1 && cinfo->global_state != state2) {
    JPEGLI_ERROR("Unexpected global state %d [expected %d or %d]",
                 cinfo->global_state, state1, state2);
  }
}

float LinearQualityToDistance(int scale_factor) {
  scale_factor = std::min(5000, std::max(0, scale_factor));
  int quality =
      scale_factor < 100 ? 100 - scale_factor / 2 : 5000 / scale_factor;
  return jpegli_quality_to_distance(quality);
}

// Initialize cinfo fields that are not dependent on input image. This is shared
// between jpegli_CreateCompress() and jpegli_set_defaults()
void InitializeCompressParams(j_compress_ptr cinfo) {
  cinfo->data_precision = 8;
  cinfo->num_scans = 0;
  cinfo->scan_info = nullptr;
  cinfo->raw_data_in = false;
  cinfo->arith_code = false;
  cinfo->optimize_coding = true;
  cinfo->CCIR601_sampling = false;
  cinfo->smoothing_factor = 0;
  cinfo->dct_method = JDCT_FLOAT;
  cinfo->restart_interval = 0;
  cinfo->restart_in_rows = 0;
  cinfo->write_JFIF_header = false;
  cinfo->JFIF_major_version = 1;
  cinfo->JFIF_minor_version = 1;
  cinfo->density_unit = 0;
  cinfo->X_density = 1;
  cinfo->Y_density = 1;
}

bool CheckColorSpaceComponents(int num_components, J_COLOR_SPACE colorspace) {
  switch (colorspace) {
    case JCS_GRAYSCALE:
      return num_components == 1;
    case JCS_RGB:
    case JCS_YCbCr:
    case JCS_EXT_RGB:
    case JCS_EXT_BGR:
      return num_components == 3;
    case JCS_CMYK:
    case JCS_YCCK:
    case JCS_EXT_RGBX:
    case JCS_EXT_BGRX:
    case JCS_EXT_XBGR:
    case JCS_EXT_XRGB:
    case JCS_EXT_RGBA:
    case JCS_EXT_BGRA:
    case JCS_EXT_ABGR:
    case JCS_EXT_ARGB:
      return num_components == 4;
    default:
      // Unrecognized colorspaces can have any number of channels, since no
      // color transform will be performed on them.
      return true;
  }
}

void ColorTransform(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;

  if (!CheckColorSpaceComponents(cinfo->input_components,
                                 cinfo->in_color_space)) {
    JPEGLI_ERROR("Invalid number of input components %d for colorspace %d",
                 cinfo->input_components, cinfo->in_color_space);
  }
  if (!CheckColorSpaceComponents(cinfo->num_components,
                                 cinfo->jpeg_color_space)) {
    JPEGLI_ERROR("Invalid number of components %d for colorspace %d",
                 cinfo->num_components, cinfo->jpeg_color_space);
  }

  if (cinfo->jpeg_color_space == cinfo->in_color_space) {
    if (cinfo->num_components != cinfo->input_components) {
      JPEGLI_ERROR("Input/output components mismatch:  %d vs %d",
                   cinfo->input_components, cinfo->num_components);
    }
    // No color transform requested.
    return;
  }

  if (cinfo->in_color_space == JCS_RGB && m->xyb_mode) {
    JPEGLI_ERROR("Color transform on XYB colorspace is not supported.");
  }

  if (cinfo->jpeg_color_space == JCS_GRAYSCALE) {
    if (cinfo->in_color_space == JCS_RGB) {
      for (size_t y = 0; y < cinfo->image_height; ++y) {
        RGBToYCbCr(m->input_buffer[0].Row(y), m->input_buffer[1].Row(y),
                   m->input_buffer[2].Row(y), cinfo->image_width);
      }
    } else if (cinfo->in_color_space == JCS_YCbCr ||
               cinfo->in_color_space == JCS_YCCK) {
      // Since the first luminance channel is the grayscale version of the
      // image, nothing to do here
    }
  } else if (cinfo->jpeg_color_space == JCS_YCbCr) {
    if (cinfo->in_color_space == JCS_RGB) {
      for (size_t y = 0; y < cinfo->image_height; ++y) {
        RGBToYCbCr(m->input_buffer[0].Row(y), m->input_buffer[1].Row(y),
                   m->input_buffer[2].Row(y), cinfo->image_width);
      }
    }
  } else if (cinfo->jpeg_color_space == JCS_YCCK) {
    if (cinfo->in_color_space == JCS_CMYK) {
      for (size_t y = 0; y < cinfo->image_height; ++y) {
        CMYKToYCCK(m->input_buffer[0].Row(y), m->input_buffer[1].Row(y),
                   m->input_buffer[2].Row(y), m->input_buffer[3].Row(y),
                   cinfo->image_width);
      }
    }
  } else {
    // TODO(szabadka) Support more color transforms.
    JPEGLI_ERROR("Unsupported color transform %d -> %d", cinfo->in_color_space,
                 cinfo->jpeg_color_space);
  }
}

void PadInputToBlockMultiple(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  const size_t xsize_padded = m->xsize_blocks * DCTSIZE;
  const size_t ysize_padded = m->ysize_blocks * DCTSIZE;
  for (int c = 0; c < cinfo->num_components; ++c) {
    for (size_t y = 0; y < cinfo->image_height; ++y) {
      float* row = m->input_buffer[c].Row(y);
      const float last_val = row[cinfo->image_width - 1];
      for (size_t x = cinfo->image_width; x < xsize_padded; ++x) {
        row[x] = last_val;
      }
    }
    const float* last_row = m->input_buffer[c].Row(cinfo->image_height - 1);
    for (size_t y = cinfo->image_height; y < ysize_padded; ++y) {
      m->input_buffer[c].CopyRow(y, last_row, xsize_padded);
    }
  }
}

void Downsample(RowBuffer<float>* image, size_t orig_xsize, size_t orig_ysize,
                size_t factor_x, size_t factor_y) {
  if (factor_x == 1 && factor_y == 1) {
    return;
  }
  size_t xsize = orig_xsize / factor_x;
  size_t ysize = orig_ysize / factor_y;
  for (size_t y = 0; y < ysize; y++) {
    float* row_out = image->Row(y);
    for (size_t x = 0; x < xsize; x++) {
      size_t cnt = 0;
      float sum = 0;
      for (size_t iy = 0; iy < factor_y && iy + factor_y * y < orig_ysize;
           iy++) {
        const float* row_in = image->Row(factor_y * y + iy);
        for (size_t ix = 0; ix < factor_x && ix + factor_x * x < orig_xsize;
             ix++) {
          sum += row_in[x * factor_x + ix];
          cnt++;
        }
      }
      row_out[x] = sum / cnt;
    }
  }
}

void DownsampleComponents(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  for (int c = 0; c < cinfo->num_components; c++) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    JXL_DASSERT(cinfo->max_h_samp_factor % comp->h_samp_factor == 0);
    JXL_DASSERT(cinfo->max_v_samp_factor % comp->v_samp_factor == 0);
    const int h_factor = cinfo->max_h_samp_factor / comp->h_samp_factor;
    const int v_factor = cinfo->max_v_samp_factor / comp->v_samp_factor;
    Downsample(&m->input_buffer[c], m->xsize_blocks * DCTSIZE,
               m->ysize_blocks * DCTSIZE, h_factor, v_factor);
  }
}

void ReadLine(const uint8_t* row_in, size_t xsize, size_t c,
              size_t num_components, JpegliDataType data_type,
              JpegliEndianness endianness, float* row_out) {
  if (row_in == nullptr) {
    memset(row_out, 0, xsize * sizeof(row_out[0]));
    return;
  }
  static constexpr double kMul8 = 1.0 / 255.0;
  static constexpr double kMul16 = 1.0 / 65535.0;
  const int pwidth = num_components * jpegli_bytes_per_sample(data_type);
  bool is_little_endian =
      (endianness == JPEGLI_LITTLE_ENDIAN ||
       (endianness == JPEGLI_NATIVE_ENDIAN && IsLittleEndian()));
  if (data_type == JPEGLI_TYPE_UINT8) {
    const uint8_t* p = &row_in[c];
    for (size_t x = 0; x < xsize; ++x, p += pwidth) {
      row_out[x] = p[0] * kMul8;
    }
  } else if (data_type == JPEGLI_TYPE_UINT16 && is_little_endian) {
    const uint8_t* p = &row_in[c * 2];
    for (size_t x = 0; x < xsize; ++x, p += pwidth) {
      row_out[x] = LoadLE16(p) * kMul16;
    }
  } else if (data_type == JPEGLI_TYPE_UINT16 && !is_little_endian) {
    const uint8_t* p = &row_in[c * 2];
    for (size_t x = 0; x < xsize; ++x, p += pwidth) {
      row_out[x] = LoadBE16(p) * kMul16;
    }
  } else if (data_type == JPEGLI_TYPE_FLOAT && is_little_endian) {
    const uint8_t* p = &row_in[c * 4];
    for (size_t x = 0; x < xsize; ++x, p += pwidth) {
      row_out[x] = LoadLEFloat(p);
    }
  } else if (data_type == JPEGLI_TYPE_FLOAT && !is_little_endian) {
    const uint8_t* p = &row_in[c * 4];
    for (size_t x = 0; x < xsize; ++x, p += pwidth) {
      row_out[x] = LoadBEFloat(p);
    }
  }
}

void CopyCoefficients(j_compress_ptr cinfo,
                      std::vector<std::vector<coeff_t>>* all_coeffs) {
  for (int c = 0; c < cinfo->num_components; c++) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    const size_t xsize_blocks = comp->width_in_blocks;
    const size_t ysize_blocks = comp->height_in_blocks;
    std::vector<coeff_t> coeffs(xsize_blocks * ysize_blocks * kDCTBlockSize);
    for (size_t by = 0; by < ysize_blocks; ++by) {
      JBLOCKARRAY ba = (*cinfo->mem->access_virt_barray)(
          reinterpret_cast<j_common_ptr>(cinfo),
          cinfo->master->coeff_buffers[c], by, 1, false);
      JXL_ASSERT(sizeof(coeff_t) == sizeof(JCOEF));
      memcpy(&coeffs[by * xsize_blocks * kDCTBlockSize], ba[0],
             xsize_blocks * sizeof(ba[0][0]));
    }
    all_coeffs->emplace_back(std::move(coeffs));
  }
}

struct ProgressiveScan {
  int Ss, Se, Ah, Al;
  bool interleaved;
};

void SetDefaultScanScript(j_compress_ptr cinfo) {
  int level = cinfo->master->progressive_level;
  std::vector<ProgressiveScan> progressive_mode;
  bool interleave_dc =
      (cinfo->max_h_samp_factor == 1 && cinfo->max_v_samp_factor == 1);
  if (level == 0) {
    progressive_mode.push_back({0, 63, 0, 0, true});
  } else if (level == 1) {
    progressive_mode.push_back({0, 0, 0, 0, interleave_dc});
    progressive_mode.push_back({1, 63, 0, 1, false});
    progressive_mode.push_back({1, 63, 1, 0, false});
  } else {
    progressive_mode.push_back({0, 0, 0, 0, interleave_dc});
    progressive_mode.push_back({1, 2, 0, 0, false});
    progressive_mode.push_back({3, 63, 0, 2, false});
    progressive_mode.push_back({3, 63, 2, 1, false});
    progressive_mode.push_back({3, 63, 1, 0, false});
  }

  cinfo->script_space_size = 0;
  for (const auto& scan : progressive_mode) {
    int comps = scan.interleaved ? MAX_COMPS_IN_SCAN : 1;
    cinfo->script_space_size += DivCeil(cinfo->num_components, comps);
  }
  cinfo->script_space =
      Allocate<jpeg_scan_info>(cinfo, cinfo->script_space_size);

  jpeg_scan_info* next_scan = cinfo->script_space;
  for (const auto& scan : progressive_mode) {
    int comps = scan.interleaved ? MAX_COMPS_IN_SCAN : 1;
    for (int c = 0; c < cinfo->num_components; c += comps) {
      next_scan->Ss = scan.Ss;
      next_scan->Se = scan.Se;
      next_scan->Ah = scan.Ah;
      next_scan->Al = scan.Al;
      next_scan->comps_in_scan = std::min(comps, cinfo->num_components - c);
      for (int j = 0; j < next_scan->comps_in_scan; ++j) {
        next_scan->component_index[j] = c + j;
      }
      ++next_scan;
    }
  }
  JXL_ASSERT(next_scan - cinfo->script_space == cinfo->script_space_size);
  cinfo->scan_info = cinfo->script_space;
  cinfo->num_scans = cinfo->script_space_size;
}

void ValidateScanScript(j_compress_ptr cinfo) {
  // Mask of coefficient bits defined by the scan script, for each component
  // and coefficient index.
  uint16_t comp_mask[kMaxComponents][DCTSIZE2] = {};
  static constexpr int kMaxRefinementBit = 10;

  for (int i = 0; i < cinfo->num_scans; ++i) {
    const jpeg_scan_info& si = cinfo->scan_info[i];
    if (si.comps_in_scan < 1 || si.comps_in_scan > MAX_COMPS_IN_SCAN) {
      JPEGLI_ERROR("Invalid number of components in scan %d", si.comps_in_scan);
    }
    int last_ci = -1;
    for (int j = 0; j < si.comps_in_scan; ++j) {
      int ci = si.component_index[j];
      if (ci < 0 || ci >= cinfo->num_components) {
        JPEGLI_ERROR("Invalid component index %d in scan", ci);
      } else if (ci == last_ci) {
        JPEGLI_ERROR("Duplicate component index %d in scan", ci);
      } else if (ci < last_ci) {
        JPEGLI_ERROR("Out of order component index %d in scan", ci);
      }
      last_ci = ci;
    }
    if (si.Ss < 0 || si.Se < si.Ss || si.Se >= DCTSIZE2) {
      JPEGLI_ERROR("Invalid spectral range %d .. %d in scan", si.Ss, si.Se);
    }
    if (si.Ah < 0 || si.Al < 0 || si.Al > kMaxRefinementBit) {
      JPEGLI_ERROR("Invalid refinement bits %d/%d", si.Ah, si.Al);
    }
    if (!cinfo->progressive_mode) {
      if (si.Ss != 0 || si.Se != DCTSIZE2 - 1 || si.Ah != 0 || si.Al != 0) {
        JPEGLI_ERROR("Invalid scan for sequential mode");
      }
    } else {
      if (si.Ss == 0 && si.Se != 0) {
        JPEGLI_ERROR("DC and AC together in progressive scan");
      }
    }
    if (si.Ss != 0 && si.comps_in_scan != 1) {
      JPEGLI_ERROR("Interleaved AC only scan.");
    }
    for (int j = 0; j < si.comps_in_scan; ++j) {
      int ci = si.component_index[j];
      if (si.Ss != 0 && comp_mask[ci][0] == 0) {
        JPEGLI_ERROR("AC before DC in component %d of scan", ci);
      }
      for (int k = si.Ss; k <= si.Se; ++k) {
        if (comp_mask[ci][k] == 0) {
          if (si.Ah != 0) {
            JPEGLI_ERROR("Invalid first scan refinement bit");
          }
          comp_mask[ci][k] = ((0xffff << si.Al) & 0xffff);
        } else {
          if (comp_mask[ci][k] != ((0xffff << si.Ah) & 0xffff) ||
              si.Al != si.Ah - 1) {
            JPEGLI_ERROR("Invalid refinement bit progression.");
          }
          comp_mask[ci][k] |= 1 << si.Al;
        }
      }
    }
  }
  for (int c = 0; c < cinfo->num_components; ++c) {
    for (int k = 0; k < DCTSIZE2; ++k) {
      if (comp_mask[c][k] != 0xffff) {
        JPEGLI_ERROR("Incomplete scan of component %d and frequency %d", c, k);
      }
    }
  }
}

void ProcessCompressionParams(j_compress_ptr cinfo) {
  if (cinfo->dest == nullptr) {
    JPEGLI_ERROR("Missing destination.");
  }
  if (cinfo->image_width < 1 || cinfo->image_height < 1 ||
      cinfo->input_components < 1) {
    JPEGLI_ERROR("Empty input image.");
  }
  if (cinfo->image_width > static_cast<int>(JPEG_MAX_DIMENSION) ||
      cinfo->image_height > static_cast<int>(JPEG_MAX_DIMENSION) ||
      cinfo->input_components > static_cast<int>(kMaxComponents)) {
    JPEGLI_ERROR("Input image too big.");
  }
  if (cinfo->num_components < 1 ||
      cinfo->num_components > static_cast<int>(kMaxComponents)) {
    JPEGLI_ERROR("Invalid number of components.");
  }
  if (cinfo->data_precision != kJpegPrecision) {
    JPEGLI_ERROR("Invalid data precision");
  }
  if (cinfo->arith_code) {
    JPEGLI_ERROR("Arithmetic coding is not implemented.");
  }
  if (cinfo->CCIR601_sampling) {
    JPEGLI_ERROR("CCIR601 sampling is not implemented.");
  }
  if (cinfo->restart_interval > 65535u) {
    JPEGLI_ERROR("Restart interval too big");
  }
  jpeg_comp_master* m = cinfo->master;
  m->next_marker_byte = nullptr;
  if (cinfo->scan_info != nullptr) {
    cinfo->progressive_mode =
        cinfo->scan_info->Ss != 0 || cinfo->scan_info->Se != DCTSIZE2 - 1;
    ValidateScanScript(cinfo);
  } else {
    cinfo->progressive_mode = cinfo->master->progressive_level > 0;
  }
  cinfo->max_h_samp_factor = cinfo->max_v_samp_factor = 1;
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    if (comp->component_index != c) {
      JPEGLI_ERROR("Invalid component index");
    }
    for (int j = 0; j < c; ++j) {
      if (cinfo->comp_info[j].component_id == comp->component_id) {
        JPEGLI_ERROR("Duplicate component id %d", comp->component_id);
      }
    }
    if (comp->h_samp_factor <= 0 || comp->v_samp_factor <= 0 ||
        comp->h_samp_factor > MAX_SAMP_FACTOR ||
        comp->v_samp_factor > MAX_SAMP_FACTOR) {
      JPEGLI_ERROR("Invalid sampling factor %d x %d", comp->h_samp_factor,
                   comp->v_samp_factor);
    }
    cinfo->max_h_samp_factor =
        std::max(comp->h_samp_factor, cinfo->max_h_samp_factor);
    cinfo->max_v_samp_factor =
        std::max(comp->v_samp_factor, cinfo->max_v_samp_factor);
  }
  if (cinfo->num_components == 1 &&
      (cinfo->max_h_samp_factor != 1 || cinfo->max_v_samp_factor != 1)) {
    JPEGLI_ERROR("Sampling is not supported for simgle component image.");
  }
  size_t iMCU_width = DCTSIZE * cinfo->max_h_samp_factor;
  size_t iMCU_height = DCTSIZE * cinfo->max_v_samp_factor;
  size_t total_iMCU_cols = DivCeil(cinfo->image_width, iMCU_width);
  cinfo->total_iMCU_rows = DivCeil(cinfo->image_height, iMCU_height);
  m->xsize_blocks = total_iMCU_cols * cinfo->max_h_samp_factor;
  m->ysize_blocks = cinfo->total_iMCU_rows * cinfo->max_v_samp_factor;
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    if (cinfo->max_h_samp_factor % comp->h_samp_factor != 0 ||
        cinfo->max_v_samp_factor % comp->v_samp_factor != 0) {
      JPEGLI_ERROR("Non-integral sampling ratios are not supported.");
    }
    const size_t h_factor = cinfo->max_h_samp_factor / comp->h_samp_factor;
    const size_t v_factor = cinfo->max_v_samp_factor / comp->v_samp_factor;
    comp->downsampled_width = DivCeil(cinfo->image_width, h_factor);
    comp->downsampled_height = DivCeil(cinfo->image_height, v_factor);
    comp->width_in_blocks = DivCeil(comp->downsampled_width, DCTSIZE);
    comp->height_in_blocks = DivCeil(comp->downsampled_height, DCTSIZE);
  }
}

template <typename T>
void SetSentTableFlag(T** table_ptrs, size_t num, boolean val) {
  for (size_t i = 0; i < num; ++i) {
    if (table_ptrs[i]) table_ptrs[i]->sent_table = val;
  }
}

void WriteFileHeader(j_compress_ptr cinfo) {
  (*cinfo->dest->init_destination)(cinfo);
  // SOI
  WriteOutput(cinfo, {0xFF, 0xD8});
  // APP0
  if (cinfo->write_JFIF_header) {
    EncodeAPP0(cinfo);
  }
  // APP14
  if (cinfo->write_Adobe_marker) {
    EncodeAPP14(cinfo);
  }
}

void EncodeScans(j_compress_ptr cinfo,
                 const std::vector<std::vector<coeff_t>>& coeffs) {
  if (cinfo->num_scans == 1 && cinfo->optimize_coding &&
      cinfo->restart_interval == 0 && cinfo->restart_in_rows == 0) {
    EncodeSingleScan(cinfo, coeffs);
    return;
  }
  std::vector<JPEGHuffmanCode> huffman_codes;
  if (cinfo->optimize_coding || cinfo->progressive_mode) {
    OptimizeHuffmanCodes(cinfo, coeffs, &huffman_codes);
  } else {
    CopyHuffmanCodes(cinfo, &huffman_codes);
  }
  size_t dht_index = 0;
  size_t last_restart_interval = 0;
  for (int i = 0; i < cinfo->num_scans; ++i) {
    cinfo->restart_interval = RestartIntervalForScan(cinfo, i);
    if (cinfo->restart_interval != last_restart_interval) {
      EncodeDRI(cinfo);
      last_restart_interval = cinfo->restart_interval;
    }
    size_t num_dht = cinfo->master->scan_coding_info[i].num_huffman_codes;
    if (num_dht > 0) {
      EncodeDHT(cinfo, huffman_codes.data() + dht_index, num_dht);
      dht_index += num_dht;
    }
    EncodeSOS(cinfo, i);
    if (!EncodeScan(cinfo, coeffs, i)) {
      JPEGLI_ERROR("Failed to encode scan.");
    }
  }
}

}  // namespace jpegli

void jpegli_CreateCompress(j_compress_ptr cinfo, int version,
                           size_t structsize) {
  cinfo->mem = nullptr;
  if (structsize != sizeof(*cinfo)) {
    JPEGLI_ERROR("jpegli_compress_struct has wrong size.");
  }
  jpegli::InitMemoryManager(reinterpret_cast<j_common_ptr>(cinfo));
  cinfo->progress = nullptr;
  cinfo->is_decompressor = FALSE;
  cinfo->global_state = jpegli::kEncStart;
  cinfo->dest = nullptr;
  cinfo->image_width = 0;
  cinfo->image_height = 0;
  cinfo->input_components = 0;
  cinfo->in_color_space = JCS_UNKNOWN;
  cinfo->input_gamma = 1.0f;
  cinfo->num_components = 0;
  cinfo->jpeg_color_space = JCS_UNKNOWN;
  cinfo->comp_info = nullptr;
  for (int i = 0; i < NUM_QUANT_TBLS; ++i) {
    cinfo->quant_tbl_ptrs[i] = nullptr;
  }
  for (int i = 0; i < NUM_HUFF_TBLS; ++i) {
    cinfo->dc_huff_tbl_ptrs[i] = nullptr;
    cinfo->ac_huff_tbl_ptrs[i] = nullptr;
  }
  memset(cinfo->arith_dc_L, 0, sizeof(cinfo->arith_dc_L));
  memset(cinfo->arith_dc_U, 0, sizeof(cinfo->arith_dc_U));
  memset(cinfo->arith_ac_K, 0, sizeof(cinfo->arith_ac_K));
  cinfo->write_Adobe_marker = false;
  jpegli::InitializeCompressParams(cinfo);
  cinfo->master = new jpeg_comp_master;
}

void jpegli_set_xyb_mode(j_compress_ptr cinfo) {
  CheckState(cinfo, jpegli::kEncStart);
  cinfo->master->xyb_mode = true;
}

void jpegli_set_defaults(j_compress_ptr cinfo) {
  CheckState(cinfo, jpegli::kEncStart);
  jpegli::InitializeCompressParams(cinfo);
  jpegli_default_colorspace(cinfo);
  jpegli_set_progressive_level(cinfo, jpegli::kDefaultProgressiveLevel);
  jpegli::AddStandardHuffmanTables(cinfo, /*is_dc=*/false);
  jpegli::AddStandardHuffmanTables(cinfo, /*is_dc=*/true);
}

void jpegli_default_colorspace(j_compress_ptr cinfo) {
  CheckState(cinfo, jpegli::kEncStart);
  switch (cinfo->in_color_space) {
    case JCS_GRAYSCALE:
      jpegli_set_colorspace(cinfo, JCS_GRAYSCALE);
      break;
    case JCS_RGB: {
      if (cinfo->master->xyb_mode) {
        jpegli_set_colorspace(cinfo, JCS_RGB);
      } else {
        jpegli_set_colorspace(cinfo, JCS_YCbCr);
      }
      break;
    }
    case JCS_YCbCr:
      jpegli_set_colorspace(cinfo, JCS_YCbCr);
      break;
    case JCS_CMYK:
      jpegli_set_colorspace(cinfo, JCS_CMYK);
      break;
    case JCS_YCCK:
      jpegli_set_colorspace(cinfo, JCS_YCCK);
      break;
    case JCS_UNKNOWN:
      jpegli_set_colorspace(cinfo, JCS_UNKNOWN);
      break;
    default:
      JPEGLI_ERROR("Unsupported input colorspace %d", cinfo->in_color_space);
  }
}

void jpegli_set_colorspace(j_compress_ptr cinfo, J_COLOR_SPACE colorspace) {
  CheckState(cinfo, jpegli::kEncStart);
  cinfo->jpeg_color_space = colorspace;
  switch (colorspace) {
    case JCS_GRAYSCALE:
      cinfo->num_components = 1;
      break;
    case JCS_RGB:
    case JCS_YCbCr:
      cinfo->num_components = 3;
      break;
    case JCS_CMYK:
    case JCS_YCCK:
      cinfo->num_components = 4;
      break;
    case JCS_UNKNOWN:
      cinfo->num_components =
          std::min<int>(jpegli::kMaxComponents, cinfo->input_components);
      break;
    default:
      JPEGLI_ERROR("Unsupported jpeg colorspace %d", colorspace);
  }
  // Adobe marker is only needed to distinguish CMYK and YCCK JPEGs.
  cinfo->write_Adobe_marker = (cinfo->jpeg_color_space == JCS_YCCK);
  cinfo->comp_info =
      jpegli::Allocate<jpeg_component_info>(cinfo, jpegli::kMaxComponents);
  memset(cinfo->comp_info, 0,
         jpegli::kMaxComponents * sizeof(jpeg_component_info));
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    comp->component_index = c;
    comp->component_id = c + 1;
    comp->h_samp_factor = 1;
    comp->v_samp_factor = 1;
    comp->quant_tbl_no = 0;
    comp->dc_tbl_no = 0;
    comp->ac_tbl_no = 0;
  }
  if (colorspace == JCS_RGB) {
    cinfo->comp_info[0].component_id = 'R';
    cinfo->comp_info[1].component_id = 'G';
    cinfo->comp_info[2].component_id = 'B';
    if (cinfo->master->xyb_mode) {
      // Subsample blue channel.
      cinfo->comp_info[0].h_samp_factor = cinfo->comp_info[0].v_samp_factor = 2;
      cinfo->comp_info[1].h_samp_factor = cinfo->comp_info[1].v_samp_factor = 2;
      cinfo->comp_info[2].h_samp_factor = cinfo->comp_info[2].v_samp_factor = 1;
      // Use separate quantization tables for each component
      cinfo->comp_info[1].quant_tbl_no = 1;
      cinfo->comp_info[2].quant_tbl_no = 2;
    }
  } else if (colorspace == JCS_CMYK) {
    cinfo->comp_info[0].component_id = 'C';
    cinfo->comp_info[1].component_id = 'M';
    cinfo->comp_info[2].component_id = 'Y';
    cinfo->comp_info[3].component_id = 'K';
  } else if (colorspace == JCS_YCbCr || colorspace == JCS_YCCK) {
    // Use separate quantization and Huffman tables for luma and chroma
    cinfo->comp_info[1].quant_tbl_no = 1;
    cinfo->comp_info[2].quant_tbl_no = 1;
    cinfo->comp_info[1].dc_tbl_no = cinfo->comp_info[1].ac_tbl_no = 1;
    cinfo->comp_info[2].dc_tbl_no = cinfo->comp_info[2].ac_tbl_no = 1;
  }
}

void jpegli_set_distance(j_compress_ptr cinfo, float distance) {
  CheckState(cinfo, jpegli::kEncStart);
  cinfo->master->distance = distance;
}

float jpegli_quality_to_distance(int quality) {
  return (quality >= 100  ? 0.01f
          : quality >= 30 ? 0.1f + (100 - quality) * 0.09f
                          : 53.0f / 3000.0f * quality * quality -
                                23.0f / 20.0f * quality + 25.0f);
}

void jpegli_set_quality(j_compress_ptr cinfo, int quality,
                        boolean force_baseline) {
  CheckState(cinfo, jpegli::kEncStart);
  cinfo->master->distance = jpegli_quality_to_distance(quality);
  cinfo->master->force_baseline = force_baseline;
}

void jpegli_set_linear_quality(j_compress_ptr cinfo, int scale_factor,
                               boolean force_baseline) {
  CheckState(cinfo, jpegli::kEncStart);
  cinfo->master->distance = jpegli::LinearQualityToDistance(scale_factor);
  cinfo->master->force_baseline = force_baseline;
}

int jpegli_quality_scaling(int quality) {
  quality = std::min(100, std::max(1, quality));
  return quality < 50 ? 5000 / quality : 200 - 2 * quality;
}

void jpegli_use_standard_quant_tables(j_compress_ptr cinfo) {
  CheckState(cinfo, jpegli::kEncStart);
  cinfo->master->use_std_tables = true;
}

void jpegli_add_quant_table(j_compress_ptr cinfo, int which_tbl,
                            const unsigned int* basic_table, int scale_factor,
                            boolean force_baseline) {
  CheckState(cinfo, jpegli::kEncStart);
  if (which_tbl < 0 || which_tbl > NUM_QUANT_TBLS) {
    JPEGLI_ERROR("Invalid quant table index %d", which_tbl);
  }
  if (cinfo->quant_tbl_ptrs[which_tbl] == nullptr) {
    cinfo->quant_tbl_ptrs[which_tbl] =
        jpegli_alloc_quant_table(reinterpret_cast<j_common_ptr>(cinfo));
  }
  int max_qval = force_baseline ? 255 : 32767U;
  JQUANT_TBL* quant_table = cinfo->quant_tbl_ptrs[which_tbl];
  for (int k = 0; k < DCTSIZE2; ++k) {
    int qval = (basic_table[k] * scale_factor + 50) / 100;
    qval = std::max(1, std::min(qval, max_qval));
    quant_table->quantval[k] = qval;
  }
  quant_table->sent_table = FALSE;
}

void jpegli_enable_adaptive_quantization(j_compress_ptr cinfo, boolean value) {
  CheckState(cinfo, jpegli::kEncStart);
  cinfo->master->use_adaptive_quantization = value;
}

void jpegli_simple_progression(j_compress_ptr cinfo) {
  CheckState(cinfo, jpegli::kEncStart);
  jpegli_set_progressive_level(cinfo, jpegli::kDefaultProgressiveLevel);
}

void jpegli_set_progressive_level(j_compress_ptr cinfo, int level) {
  CheckState(cinfo, jpegli::kEncStart);
  if (level < 0) {
    JPEGLI_ERROR("Invalid progressive level %d", level);
  }
  cinfo->master->progressive_level = level;
}

void jpegli_copy_critical_parameters(j_decompress_ptr srcinfo,
                                     j_compress_ptr dstinfo) {
  CheckState(dstinfo, jpegli::kEncStart);
  // Image parameters.
  dstinfo->image_width = srcinfo->image_width;
  dstinfo->image_height = srcinfo->image_height;
  dstinfo->input_components = srcinfo->num_components;
  dstinfo->in_color_space = srcinfo->jpeg_color_space;
  dstinfo->input_gamma = srcinfo->output_gamma;
  // Compression parameters.
  jpegli_set_defaults(dstinfo);
  jpegli_set_colorspace(dstinfo, srcinfo->jpeg_color_space);
  if (dstinfo->num_components != srcinfo->num_components) {
    const auto& cinfo = dstinfo;
    return JPEGLI_ERROR("Mismatch between src colorspace and components");
  }
  dstinfo->data_precision = srcinfo->data_precision;
  dstinfo->CCIR601_sampling = srcinfo->CCIR601_sampling;
  dstinfo->JFIF_major_version = srcinfo->JFIF_major_version;
  dstinfo->JFIF_minor_version = srcinfo->JFIF_minor_version;
  dstinfo->density_unit = srcinfo->density_unit;
  dstinfo->X_density = srcinfo->X_density;
  dstinfo->Y_density = srcinfo->Y_density;
  for (int c = 0; c < dstinfo->num_components; ++c) {
    jpeg_component_info* srccomp = &srcinfo->comp_info[c];
    jpeg_component_info* dstcomp = &dstinfo->comp_info[c];
    dstcomp->component_id = srccomp->component_id;
    dstcomp->h_samp_factor = srccomp->h_samp_factor;
    dstcomp->v_samp_factor = srccomp->v_samp_factor;
    dstcomp->quant_tbl_no = srccomp->quant_tbl_no;
  }
  for (int i = 0; i < NUM_QUANT_TBLS; ++i) {
    if (!srcinfo->quant_tbl_ptrs[i]) continue;
    if (dstinfo->quant_tbl_ptrs[i] == nullptr) {
      dstinfo->quant_tbl_ptrs[i] = jpegli::Allocate<JQUANT_TBL>(dstinfo, 1);
    }
    memcpy(dstinfo->quant_tbl_ptrs[i], srcinfo->quant_tbl_ptrs[i],
           sizeof(JQUANT_TBL));
    dstinfo->quant_tbl_ptrs[i]->sent_table = FALSE;
  }
}

void jpegli_suppress_tables(j_compress_ptr cinfo, boolean suppress) {
  jpegli::SetSentTableFlag(cinfo->quant_tbl_ptrs, NUM_QUANT_TBLS, suppress);
  jpegli::SetSentTableFlag(cinfo->dc_huff_tbl_ptrs, NUM_HUFF_TBLS, suppress);
  jpegli::SetSentTableFlag(cinfo->ac_huff_tbl_ptrs, NUM_HUFF_TBLS, suppress);
}

void jpegli_start_compress(j_compress_ptr cinfo, boolean write_all_tables) {
  CheckState(cinfo, jpegli::kEncStart);
  jpegli::ProcessCompressionParams(cinfo);
  jpeg_comp_master* m = cinfo->master;
  size_t stride = m->xsize_blocks * DCTSIZE;
  for (int c = 0; c < cinfo->input_components; ++c) {
    m->input_buffer[c].Allocate(m->ysize_blocks * DCTSIZE, stride);
  }
  if (write_all_tables) {
    jpegli_suppress_tables(cinfo, FALSE);
  }
  (*cinfo->mem->realize_virt_arrays)(reinterpret_cast<j_common_ptr>(cinfo));
  jpegli::WriteFileHeader(cinfo);
  cinfo->next_scanline = 0;
  cinfo->global_state = jpegli::kEncHeader;
}

void jpegli_write_coefficients(j_compress_ptr cinfo,
                               jvirt_barray_ptr* coef_arrays) {
  CheckState(cinfo, jpegli::kEncStart);
  jpegli::ProcessCompressionParams(cinfo);
  (*cinfo->mem->realize_virt_arrays)(reinterpret_cast<j_common_ptr>(cinfo));
  cinfo->master->coeff_buffers = coef_arrays;
  jpegli_suppress_tables(cinfo, FALSE);
  jpegli::WriteFileHeader(cinfo);
  cinfo->next_scanline = cinfo->image_height;
  cinfo->global_state = jpegli::kEncWriteCoeffs;
}

void jpegli_write_tables(j_compress_ptr cinfo) {
  CheckState(cinfo, jpegli::kEncStart);
  if (cinfo->dest == nullptr) {
    JPEGLI_ERROR("Missing destination.");
  }
  (*cinfo->dest->init_destination)(cinfo);
  // SOI
  jpegli::WriteOutput(cinfo, {0xFF, 0xD8});
  // DQT
  jpegli::FinalizeQuantMatrices(cinfo);
  jpegli::EncodeDQT(cinfo);
  // DHT
  std::vector<jpegli::JPEGHuffmanCode> huffman_codes;
  jpegli::CopyHuffmanCodes(cinfo, &huffman_codes);
  jpegli::EncodeDHT(cinfo, huffman_codes.data(), huffman_codes.size());
  // EOI
  jpegli::WriteOutput(cinfo, {0xFF, 0xD9});
  (*cinfo->dest->term_destination)(cinfo);
  jpegli_suppress_tables(cinfo, TRUE);
}

void jpegli_write_m_header(j_compress_ptr cinfo, int marker,
                           unsigned int datalen) {
  CheckState(cinfo, jpegli::kEncHeader, jpegli::kEncWriteCoeffs);
  jpeg_comp_master* m = cinfo->master;
  if (datalen > jpegli::kMaxBytesInMarker) {
    JPEGLI_ERROR("Invalid marker length %u", datalen);
  }
  if (marker != 0xfe && (marker < 0xe0 || marker > 0xef)) {
    JPEGLI_ERROR(
        "jpegli_write_m_header: Only APP and COM markers are supported.");
  }
  std::vector<uint8_t> marker_data(4 + datalen);
  marker_data[0] = 0xff;
  marker_data[1] = marker;
  marker_data[2] = (datalen + 2) >> 8;
  marker_data[3] = (datalen + 2) & 0xff;
  jpegli::WriteOutput(cinfo, &marker_data[0], 4);
  m->special_markers.emplace_back(std::move(marker_data));
  m->next_marker_byte = &m->special_markers.back()[4];
}

void jpegli_write_m_byte(j_compress_ptr cinfo, int val) {
  if (cinfo->master->next_marker_byte == nullptr) {
    JPEGLI_ERROR("Marker header missing.");
  }
  *cinfo->master->next_marker_byte = val;
  jpegli::WriteOutput(cinfo, cinfo->master->next_marker_byte, 1);
  cinfo->master->next_marker_byte++;
}

void jpegli_write_marker(j_compress_ptr cinfo, int marker,
                         const JOCTET* dataptr, unsigned int datalen) {
  jpegli_write_m_header(cinfo, marker, datalen);
  jpegli::WriteOutput(cinfo, dataptr, datalen);
  memcpy(cinfo->master->next_marker_byte, dataptr, datalen);
  cinfo->master->next_marker_byte = nullptr;
}

void jpegli_write_icc_profile(j_compress_ptr cinfo, const JOCTET* icc_data_ptr,
                              unsigned int icc_data_len) {
  constexpr size_t kMaxIccBytesInMarker =
      jpegli::kMaxBytesInMarker - sizeof jpegli::kICCSignature - 2;
  const int num_markers =
      static_cast<int>(jpegli::DivCeil(icc_data_len, kMaxIccBytesInMarker));
  size_t begin = 0;
  for (int current_marker = 0; current_marker < num_markers; ++current_marker) {
    const size_t length = std::min(kMaxIccBytesInMarker, icc_data_len - begin);
    jpegli_write_m_header(
        cinfo, jpegli::kICCMarker,
        static_cast<unsigned int>(length + sizeof jpegli::kICCSignature + 2));
    for (const unsigned char c : jpegli::kICCSignature) {
      jpegli_write_m_byte(cinfo, c);
    }
    jpegli_write_m_byte(cinfo, current_marker + 1);
    jpegli_write_m_byte(cinfo, num_markers);
    for (size_t i = 0; i < length; ++i) {
      jpegli_write_m_byte(cinfo, icc_data_ptr[begin]);
      ++begin;
    }
  }
  cinfo->master->next_marker_byte = nullptr;
}

void jpegli_set_input_format(j_compress_ptr cinfo, JpegliDataType data_type,
                             JpegliEndianness endianness) {
  CheckState(cinfo, jpegli::kEncHeader);
  cinfo->master->data_type = data_type;
  cinfo->master->endianness = endianness;
}

JDIMENSION jpegli_write_scanlines(j_compress_ptr cinfo, JSAMPARRAY scanlines,
                                  JDIMENSION num_lines) {
  CheckState(cinfo, jpegli::kEncHeader, jpegli::kEncReadImage);
  if (cinfo->raw_data_in) {
    JPEGLI_ERROR("jpegli_write_raw_data() must be called for raw data mode.");
  }
  cinfo->global_state = jpegli::kEncReadImage;
  jpeg_comp_master* m = cinfo->master;
  if (num_lines + cinfo->next_scanline > cinfo->image_height) {
    num_lines = cinfo->image_height - cinfo->next_scanline;
  }
  for (int c = 0; c < cinfo->input_components; ++c) {
    for (size_t i = 0; i < num_lines; ++i) {
      jpegli::ReadLine(scanlines[i], cinfo->image_width, c,
                       cinfo->num_components, m->data_type, m->endianness,
                       m->input_buffer[c].Row(cinfo->next_scanline + i));
    }
  }
  cinfo->next_scanline += num_lines;
  return num_lines;
}

JDIMENSION jpegli_write_raw_data(j_compress_ptr cinfo, JSAMPIMAGE data,
                                 JDIMENSION num_lines) {
  CheckState(cinfo, jpegli::kEncHeader, jpegli::kEncReadImage);
  if (!cinfo->raw_data_in) {
    JPEGLI_ERROR("jpegli_write_raw_data(): raw data mode was not set");
  }
  cinfo->global_state = jpegli::kEncReadImage;
  jpeg_comp_master* m = cinfo->master;
  if (cinfo->next_scanline >= cinfo->image_height) {
    return 0;
  }
  size_t iMCU_height = DCTSIZE * cinfo->max_v_samp_factor;
  if (num_lines < iMCU_height) {
    JPEGLI_ERROR("Missing input lines, minimum is %u", iMCU_height);
  }
  size_t iMCU_y = cinfo->next_scanline / iMCU_height;
  for (int c = 0; c < cinfo->num_components; ++c) {
    JSAMPARRAY plane = data[c];
    jpeg_component_info* comp = &cinfo->comp_info[c];
    size_t xsize = comp->width_in_blocks * DCTSIZE;
    size_t ysize = comp->v_samp_factor * DCTSIZE;
    size_t y0 = iMCU_y * ysize;
    for (size_t i = 0; i < ysize; ++i) {
      jpegli::ReadLine(plane[i], xsize, 0, 1, m->data_type, m->endianness,
                       m->input_buffer[c].Row(y0 + i));
    }
  }
  cinfo->next_scanline += iMCU_height;
  return iMCU_height;
}

void jpegli_finish_compress(j_compress_ptr cinfo) {
  CheckState(cinfo, jpegli::kEncReadImage, jpegli::kEncWriteCoeffs);
  if (cinfo->next_scanline < cinfo->image_height) {
    JPEGLI_ERROR("Incomplete image, expected %d rows, got %d",
                 cinfo->image_height, cinfo->next_scanline);
  }

  if (!cinfo->raw_data_in && cinfo->global_state != jpegli::kEncWriteCoeffs) {
    jpegli::ColorTransform(cinfo);
    jpegli::PadInputToBlockMultiple(cinfo);
    jpegli::DownsampleComponents(cinfo);
  }
  if (cinfo->global_state != jpegli::kEncWriteCoeffs) {
    jpegli::ComputeAdaptiveQuantField(cinfo);
  }

  // DQT
  jpegli::FinalizeQuantMatrices(cinfo);
  jpegli::EncodeDQT(cinfo);

  // SOF
  jpegli::EncodeSOF(cinfo);

  std::vector<std::vector<jpegli::coeff_t>> coeffs;
  if (cinfo->global_state == jpegli::kEncWriteCoeffs) {
    jpegli::CopyCoefficients(cinfo, &coeffs);
  } else {
    jpegli::ComputeDCTCoefficients(cinfo, &coeffs);
  }

  if (cinfo->scan_info == nullptr) {
    jpegli::SetDefaultScanScript(cinfo);
    // This should never fail since we are generating the scan script above, but
    // if there is a bug in the scan script generation code, it is better to
    // fail here than to create a corrupt JPEG file.
    jpegli::ValidateScanScript(cinfo);
  }

  jpegli::EncodeScans(cinfo, coeffs);

  // EOI
  jpegli::WriteOutput(cinfo, {0xFF, 0xD9});
  (*cinfo->dest->term_destination)(cinfo);

  // Release memory and reset global state.
  jpegli_abort_compress(cinfo);
}

void jpegli_abort_compress(j_compress_ptr cinfo) {
  jpegli_abort(reinterpret_cast<j_common_ptr>(cinfo));
}

void jpegli_destroy_compress(j_compress_ptr cinfo) {
  jpegli_destroy(reinterpret_cast<j_common_ptr>(cinfo));
}
