// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/decode.h"

#include <string.h>

#include <vector>

#include "lib/jpegli/color_quantize.h"
#include "lib/jpegli/decode_internal.h"
#include "lib/jpegli/decode_marker.h"
#include "lib/jpegli/decode_scan.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/memory_manager.h"
#include "lib/jpegli/render.h"
#include "lib/jpegli/source_manager.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/status.h"

namespace jpegli {

void InitializeImage(j_decompress_ptr cinfo) {
  cinfo->restart_interval = 0;
  cinfo->saw_JFIF_marker = FALSE;
  cinfo->JFIF_major_version = 1;
  cinfo->JFIF_minor_version = 1;
  cinfo->density_unit = 0;
  cinfo->X_density = 1;
  cinfo->Y_density = 1;
  cinfo->saw_Adobe_marker = FALSE;
  cinfo->Adobe_transform = 0;
  cinfo->CCIR601_sampling = FALSE;  // not used
  cinfo->marker_list = nullptr;
  cinfo->comp_info = nullptr;
  cinfo->input_scan_number = 0;
  cinfo->output_scanline = 0;
  cinfo->unread_marker = 0;
  cinfo->coef_bits = nullptr;
  // We set all these to zero since we don't yet support arithmetic coding.
  memset(cinfo->arith_dc_L, 0, sizeof(cinfo->arith_dc_L));
  memset(cinfo->arith_dc_U, 0, sizeof(cinfo->arith_dc_U));
  memset(cinfo->arith_ac_K, 0, sizeof(cinfo->arith_ac_K));
  // Initialize the private fields.
  cinfo->master->colormap_lut_ = nullptr;
  cinfo->master->pixels_ = nullptr;
  cinfo->master->scanlines_ = nullptr;
  cinfo->master->regenerate_inverse_colormap_ = true;
  for (int i = 0; i < kMaxComponents; ++i) {
    cinfo->master->dither_[i] = nullptr;
    cinfo->master->error_row_[i] = nullptr;
  }
  cinfo->master->output_passes_done_ = 0;
}

void InitializeDecompressParams(j_decompress_ptr cinfo) {
  cinfo->jpeg_color_space = JCS_UNKNOWN;
  cinfo->out_color_space = JCS_UNKNOWN;
  cinfo->scale_num = 1;
  cinfo->scale_denom = 1;
  cinfo->output_gamma = 0.0f;
  cinfo->buffered_image = FALSE;
  cinfo->raw_data_out = FALSE;
  cinfo->dct_method = JDCT_DEFAULT;
  cinfo->do_fancy_upsampling = TRUE;
  cinfo->do_block_smoothing = FALSE;
  cinfo->quantize_colors = FALSE;
  cinfo->dither_mode = JDITHER_FS;
  cinfo->two_pass_quantize = TRUE;
  cinfo->desired_number_of_colors = 256;
  cinfo->enable_1pass_quant = FALSE;
  cinfo->enable_external_quant = FALSE;
  cinfo->enable_2pass_quant = FALSE;
  cinfo->actual_number_of_colors = 0;
  cinfo->colormap = nullptr;
  // Initialize the private fields.
  for (int i = 0; i < 16; ++i) {
    cinfo->master->app_marker_parsers[i] = nullptr;
  }
  cinfo->master->com_marker_parser = nullptr;
}

void InitProgressMonitor(j_decompress_ptr cinfo, bool coef_only) {
  if (!cinfo->progress) return;
  jpeg_decomp_master* m = cinfo->master;
  int nc = cinfo->num_components;
  int estimated_num_scans =
      cinfo->progressive_mode ? 2 + 3 * nc : (m->is_multiscan_ ? nc : 1);
  cinfo->progress->pass_limit = cinfo->total_iMCU_rows * estimated_num_scans;
  cinfo->progress->pass_counter = 0;
  if (coef_only) {
    cinfo->progress->total_passes = 1;
  } else {
    int input_passes = !cinfo->buffered_image && m->is_multiscan_ ? 1 : 0;
    bool two_pass_quant =
        cinfo->quantize_colors && !cinfo->colormap && cinfo->two_pass_quantize;
    cinfo->progress->total_passes = input_passes + (two_pass_quant ? 2 : 1);
  }
  cinfo->progress->completed_passes = 0;
}

void InitProgressMonitorForOutput(j_decompress_ptr cinfo) {
  if (!cinfo->progress) return;
  jpeg_decomp_master* m = cinfo->master;
  int passes_per_output = cinfo->enable_2pass_quant ? 2 : 1;
  int output_passes_left = cinfo->buffered_image && !m->found_eoi_ ? 2 : 1;
  cinfo->progress->total_passes =
      m->output_passes_done_ + passes_per_output * output_passes_left;
  cinfo->progress->completed_passes = m->output_passes_done_;
}

void ProgressMonitorInputPass(j_decompress_ptr cinfo) {
  if (!cinfo->progress) return;
  cinfo->progress->pass_counter =
      ((cinfo->input_scan_number - 1) * cinfo->total_iMCU_rows +
       cinfo->input_iMCU_row);
  if (cinfo->progress->pass_counter > cinfo->progress->pass_limit) {
    cinfo->progress->pass_limit =
        cinfo->input_scan_number * cinfo->total_iMCU_rows;
  }
  (*cinfo->progress->progress_monitor)(reinterpret_cast<j_common_ptr>(cinfo));
}

void ProgressMonitorOutputPass(j_decompress_ptr cinfo) {
  if (!cinfo->progress) return;
  jpeg_decomp_master* m = cinfo->master;
  int input_passes = !cinfo->buffered_image && m->is_multiscan_ ? 1 : 0;
  cinfo->progress->pass_counter = cinfo->output_scanline;
  cinfo->progress->pass_limit = cinfo->output_height;
  cinfo->progress->completed_passes = input_passes + m->output_passes_done_;
  (*cinfo->progress->progress_monitor)(reinterpret_cast<j_common_ptr>(cinfo));
}

int ConsumeInput(j_decompress_ptr cinfo) {
  jpeg_source_mgr* src = cinfo->src;
  std::vector<uint8_t> buffer;
  const uint8_t* last_input_byte = src->next_input_byte + src->bytes_in_buffer;
  int status;
  for (;;) {
    if (cinfo->global_state == kDecProcessScan) {
      status = ProcessScan(cinfo);
    } else {
      status = ProcessMarkers(cinfo);
    }
    if (status != JPEG_SUSPENDED) {
      break;
    }
    if (buffer.size() != src->bytes_in_buffer) {
      // Save the unprocessed bytes in the input to a temporary buffer.
      buffer.assign(src->next_input_byte,
                    src->next_input_byte + src->bytes_in_buffer);
    }
    if (!(*cinfo->src->fill_input_buffer)(cinfo)) {
      return status;
    }
    if (src->bytes_in_buffer == 0) {
      JPEGLI_ERROR("Empty input.");
    }
    // Save the end of the current input so that we can restore it after the
    // input processing succeeds.
    last_input_byte = cinfo->src->next_input_byte + src->bytes_in_buffer;
    // Extend the temporary buffer with the new bytes and point the input to it.
    buffer.insert(buffer.end(), src->next_input_byte, last_input_byte);
    src->next_input_byte = buffer.data();
    src->bytes_in_buffer = buffer.size();
  }
  // Restore the input pointer in case we had to change it to a temporary
  // buffer earlier.
  src->next_input_byte = last_input_byte - src->bytes_in_buffer;
  if (status == JPEG_SCAN_COMPLETED) {
    cinfo->global_state = kDecProcessMarkers;
  } else if (status == JPEG_REACHED_SOS) {
    cinfo->global_state =
        cinfo->global_state == kDecInHeader ? kDecHeaderDone : kDecProcessScan;
  }
  return status;
}

bool IsInputReady(j_decompress_ptr cinfo) {
  if (cinfo->master->found_eoi_) {
    return true;
  }
  if (cinfo->input_scan_number > cinfo->output_scan_number) {
    return true;
  }
  if (cinfo->input_scan_number < cinfo->output_scan_number) {
    return false;
  }
  if (cinfo->input_iMCU_row == cinfo->total_iMCU_rows) {
    return true;
  }
  return cinfo->input_iMCU_row > cinfo->output_iMCU_row;
}

bool ReadOutputPass(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  if (!m->pixels_) {
    size_t stride = cinfo->out_color_components * cinfo->output_width;
    size_t num_samples = cinfo->output_height * stride;
    m->pixels_ = Allocate<uint8_t>(cinfo, num_samples, JPOOL_IMAGE);
    m->scanlines_ =
        Allocate<JSAMPROW>(cinfo, cinfo->output_height, JPOOL_IMAGE);
    for (size_t i = 0; i < cinfo->output_height; ++i) {
      m->scanlines_[i] = &m->pixels_[i * stride];
    }
  }
  size_t num_output_rows = 0;
  while (num_output_rows < cinfo->output_height) {
    if (IsInputReady(cinfo)) {
      ProgressMonitorOutputPass(cinfo);
      ProcessOutput(cinfo, &num_output_rows, m->scanlines_,
                    cinfo->output_height);
    } else if (ConsumeInput(cinfo) == JPEG_SUSPENDED) {
      return false;
    }
  }
  cinfo->output_scanline = 0;
  cinfo->output_iMCU_row = 0;
  return true;
}

boolean PrepareQuantizedOutput(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  if (cinfo->raw_data_out) {
    JPEGLI_ERROR("Color quantization is not supported in raw data mode.");
  }
  if (m->output_data_type_ != JPEGLI_TYPE_UINT8) {
    JPEGLI_ERROR("Color quantization must use 8-bit mode.");
  }
  m->quant_mode_ = cinfo->colormap ? 3 : cinfo->two_pass_quantize ? 2 : 1;
  if (m->quant_mode_ > 1 && cinfo->dither_mode == JDITHER_ORDERED) {
    JPEGLI_WARN("Changing dither mode to JDITHER_FS");
    cinfo->dither_mode = JDITHER_FS;
  }
  if (m->quant_mode_ == 1) {
    ChooseColorMap1Pass(cinfo);
  } else if (m->quant_mode_ == 2) {
    m->quant_pass_ = 0;
    if (!ReadOutputPass(cinfo)) {
      return FALSE;
    }
    ChooseColorMap2Pass(cinfo);
  }
  if (m->quant_mode_ == 2 ||
      (m->quant_mode_ == 3 && m->regenerate_inverse_colormap_)) {
    CreateInverseColorMap(cinfo);
  }
  if (cinfo->dither_mode == JDITHER_ORDERED) {
    CreateOrderedDitherTables(cinfo);
  } else if (cinfo->dither_mode == JDITHER_FS) {
    InitFSDitherState(cinfo);
  }
  m->quant_pass_ = 1;
  return TRUE;
}

}  // namespace jpegli

void jpegli_CreateDecompress(j_decompress_ptr cinfo, int version,
                             size_t structsize) {
  cinfo->mem = nullptr;
  if (structsize != sizeof(*cinfo)) {
    JPEGLI_ERROR("jpeg_decompress_struct has wrong size.");
  }
  jpegli::InitMemoryManager(reinterpret_cast<j_common_ptr>(cinfo));
  cinfo->is_decompressor = TRUE;
  cinfo->progress = nullptr;
  cinfo->src = nullptr;
  for (int i = 0; i < NUM_QUANT_TBLS; i++) {
    cinfo->quant_tbl_ptrs[i] = nullptr;
  }
  for (int i = 0; i < NUM_HUFF_TBLS; i++) {
    cinfo->dc_huff_tbl_ptrs[i] = nullptr;
    cinfo->ac_huff_tbl_ptrs[i] = nullptr;
  }
  cinfo->global_state = jpegli::kDecStart;
  cinfo->sample_range_limit = nullptr;  // not used
  cinfo->rec_outbuf_height = 1;         // output works with any buffer height
  cinfo->master = new jpeg_decomp_master;
  jpegli::InitializeDecompressParams(cinfo);
  jpegli::InitializeImage(cinfo);
}

void jpegli_destroy_decompress(j_decompress_ptr cinfo) {
  jpegli_destroy(reinterpret_cast<j_common_ptr>(cinfo));
}

void jpegli_abort_decompress(j_decompress_ptr cinfo) {
  jpegli_abort(reinterpret_cast<j_common_ptr>(cinfo));
}

void jpegli_save_markers(j_decompress_ptr cinfo, int marker_code,
                         unsigned int length_limit) {
  jpeg_decomp_master* m = cinfo->master;
  m->markers_to_save_.insert(marker_code);
}

void jpegli_set_marker_processor(j_decompress_ptr cinfo, int marker_code,
                                 jpeg_marker_parser_method routine) {
  jpeg_decomp_master* m = cinfo->master;
  if (marker_code == 0xfe) {
    m->com_marker_parser = routine;
  } else if (marker_code >= 0xe0 && marker_code <= 0xef) {
    m->app_marker_parsers[marker_code - 0xe0] = routine;
  } else {
    JPEGLI_ERROR("jpegli_set_marker_processor: invalid marker code %d",
                 marker_code);
  }
}

int jpegli_consume_input(j_decompress_ptr cinfo) {
  if (cinfo->global_state == jpegli::kDecStart) {
    (*cinfo->err->reset_error_mgr)(reinterpret_cast<j_common_ptr>(cinfo));
    (*cinfo->src->init_source)(cinfo);
    jpegli::InitializeImage(cinfo);
    cinfo->global_state = jpegli::kDecInHeader;
  }
  if (cinfo->global_state == jpegli::kDecHeaderDone) {
    return JPEG_REACHED_SOS;
  }
  if (cinfo->master->found_eoi_) {
    return JPEG_REACHED_EOI;
  }
  if (cinfo->global_state == jpegli::kDecInHeader ||
      cinfo->global_state == jpegli::kDecProcessMarkers ||
      cinfo->global_state == jpegli::kDecProcessScan) {
    return jpegli::ConsumeInput(cinfo);
  }
  JPEGLI_ERROR("Unexpected state %d", cinfo->global_state);
  return JPEG_REACHED_EOI;  // return value does not matter
}

int jpegli_read_header(j_decompress_ptr cinfo, boolean require_image) {
  if (cinfo->global_state != jpegli::kDecStart &&
      cinfo->global_state != jpegli::kDecInHeader) {
    JPEGLI_ERROR("jpegli_read_header: unexpected state %d",
                 cinfo->global_state);
  }
  for (;;) {
    int retcode = jpegli_consume_input(cinfo);
    if (retcode == JPEG_SUSPENDED) {
      return retcode;
    } else if (retcode == JPEG_REACHED_SOS) {
      break;
    } else if (retcode == JPEG_REACHED_EOI) {
      JPEGLI_ERROR("jpegli_read_header: unexpected EOI marker.");
    }
  };
  return JPEG_HEADER_OK;
}

boolean jpegli_read_icc_profile(j_decompress_ptr cinfo, JOCTET** icc_data_ptr,
                                unsigned int* icc_data_len) {
  if (cinfo->global_state == jpegli::kDecStart ||
      cinfo->global_state == jpegli::kDecInHeader) {
    JPEGLI_ERROR("jpegli_read_icc_profile: unexpected state %d",
                 cinfo->global_state);
  }
  if (icc_data_ptr == nullptr || icc_data_len == nullptr) {
    JPEGLI_ERROR("jpegli_read_icc_profile: invalid output buffer");
  }
  jpeg_decomp_master* m = cinfo->master;
  if (m->icc_profile_.empty()) {
    *icc_data_ptr = nullptr;
    *icc_data_len = 0;
    return FALSE;
  }
  *icc_data_len = m->icc_profile_.size();
  *icc_data_ptr = (JOCTET*)malloc(*icc_data_len);
  if (*icc_data_ptr == nullptr) {
    JPEGLI_ERROR("jpegli_read_icc_profile: Out of memory");
  }
  memcpy(*icc_data_ptr, m->icc_profile_.data(), *icc_data_len);
  return TRUE;
}

void jpegli_calc_output_dimensions(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  if (!m->found_sof_) {
    JPEGLI_ERROR("No SOF marker found.");
  }
  if (cinfo->raw_data_out) {
    if (cinfo->scale_num != 1 || cinfo->scale_denom != 1) {
      JPEGLI_ERROR("Output scaling is not supported in raw output mode");
    }
  }
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    m->h_factor[c] = cinfo->max_h_samp_factor / comp->h_samp_factor;
    m->v_factor[c] = cinfo->max_v_samp_factor / comp->v_samp_factor;
  }
  if (cinfo->scale_num != 1 || cinfo->scale_denom != 1) {
    int dctsize = 16;
    while (cinfo->scale_num * DCTSIZE <= cinfo->scale_denom * (dctsize - 1)) {
      --dctsize;
    }
    m->min_scaled_dct_size = dctsize;
    cinfo->output_width =
        jpegli::DivCeil(cinfo->image_width * dctsize, DCTSIZE);
    cinfo->output_height =
        jpegli::DivCeil(cinfo->image_height * dctsize, DCTSIZE);
    for (int c = 0; c < cinfo->num_components; ++c) {
      m->scaled_dct_size[c] = m->min_scaled_dct_size;
      // Prefer IDCT scaling over 2x upsampling.
      while (m->scaled_dct_size[c] < DCTSIZE && (m->v_factor[c] % 2) == 0 &&
             (m->h_factor[c] % 2) == 0) {
        m->scaled_dct_size[c] *= 2;
        m->v_factor[c] /= 2;
        m->h_factor[c] /= 2;
      }
    }
  } else {
    cinfo->output_width = cinfo->image_width;
    cinfo->output_height = cinfo->image_height;
    m->min_scaled_dct_size = DCTSIZE;
    for (int c = 0; c < cinfo->num_components; ++c) {
      m->scaled_dct_size[c] = DCTSIZE;
    }
  }
  cinfo->output_components =
      cinfo->quantize_colors ? 1 : cinfo->out_color_components;
  cinfo->rec_outbuf_height = 1;
}

boolean jpegli_has_multiple_scans(j_decompress_ptr cinfo) {
  if (cinfo->input_scan_number == 0) {
    JPEGLI_ERROR("No SOS marker found.");
  }
  return cinfo->master->is_multiscan_;
}

boolean jpegli_input_complete(j_decompress_ptr cinfo) {
  return cinfo->master->found_eoi_;
}

boolean jpegli_start_decompress(j_decompress_ptr cinfo) {
  if (cinfo->global_state == jpegli::kDecHeaderDone) {
    jpegli_calc_output_dimensions(cinfo);
    cinfo->global_state = jpegli::kDecProcessScan;
    jpegli::InitProgressMonitor(cinfo, /*coef_only=*/false);
    if (cinfo->buffered_image == TRUE) {
      cinfo->output_scan_number = 0;
      return TRUE;
    }
  } else if (!cinfo->master->is_multiscan_) {
    JPEGLI_ERROR("jpegli_start_decompress: unexpected state %d",
                 cinfo->global_state);
  }
  if (cinfo->master->is_multiscan_) {
    if (cinfo->global_state != jpegli::kDecProcessScan &&
        cinfo->global_state != jpegli::kDecProcessMarkers) {
      JPEGLI_ERROR("jpegli_start_decompress: unexpected state %d",
                   cinfo->global_state);
    }
    while (!cinfo->master->found_eoi_) {
      jpegli::ProgressMonitorInputPass(cinfo);
      int retcode = jpegli::ConsumeInput(cinfo);
      if (retcode == JPEG_SUSPENDED) {
        return FALSE;
      }
    }
  }
  cinfo->output_scan_number = cinfo->input_scan_number;
  jpegli::PrepareForOutput(cinfo);
  if (cinfo->quantize_colors) {
    return jpegli::PrepareQuantizedOutput(cinfo);
  } else {
    return TRUE;
  }
}

boolean jpegli_start_output(j_decompress_ptr cinfo, int scan_number) {
  jpeg_decomp_master* m = cinfo->master;
  if (!cinfo->buffered_image) {
    JPEGLI_ERROR("jpegli_start_output: buffered image mode was not set");
  }
  if (cinfo->global_state != jpegli::kDecProcessScan &&
      cinfo->global_state != jpegli::kDecProcessMarkers) {
    JPEGLI_ERROR("jpegli_start_output: unexpected state %d",
                 cinfo->global_state);
  }
  cinfo->output_scan_number = std::max(1, scan_number);
  if (m->found_eoi_) {
    cinfo->output_scan_number =
        std::min(cinfo->output_scan_number, cinfo->input_scan_number);
  }
  jpegli::InitProgressMonitorForOutput(cinfo);
  // TODO(szabadka): Figure out how much we can reuse.
  jpegli::PrepareForOutput(cinfo);
  if (cinfo->quantize_colors) {
    return jpegli::PrepareQuantizedOutput(cinfo);
  } else {
    return TRUE;
  }
}

boolean jpegli_finish_output(j_decompress_ptr cinfo) {
  if (!cinfo->buffered_image) {
    JPEGLI_ERROR("jpegli_finish_output: buffered image mode was not set");
  }
  if (cinfo->global_state != jpegli::kDecProcessScan &&
      cinfo->global_state != jpegli::kDecProcessMarkers) {
    JPEGLI_ERROR("jpegli_finish_output: unexpected state %d",
                 cinfo->global_state);
  }
  // Advance input to the start of the next scan, or to the end of input.
  while (cinfo->input_scan_number <= cinfo->output_scan_number &&
         !cinfo->master->found_eoi_) {
    if (jpegli::ConsumeInput(cinfo) == JPEG_SUSPENDED) {
      return FALSE;
    }
  }
  return TRUE;
}

JDIMENSION jpegli_read_scanlines(j_decompress_ptr cinfo, JSAMPARRAY scanlines,
                                 JDIMENSION max_lines) {
  jpeg_decomp_master* m = cinfo->master;
  if (cinfo->global_state != jpegli::kDecProcessScan &&
      cinfo->global_state != jpegli::kDecProcessMarkers) {
    JPEGLI_ERROR("jpegli_read_scanlines: unexpected state %d",
                 cinfo->global_state);
  }
  if (cinfo->buffered_image) {
    if (cinfo->output_scan_number == 0) {
      JPEGLI_ERROR(
          "jpegli_read_scanlines: "
          "jpegli_start_output() was not called");
    }
  } else if (m->is_multiscan_ && !m->found_eoi_) {
    JPEGLI_ERROR(
        "jpegli_read_scanlines: "
        "jpegli_start_decompress() did not finish");
  }
  if (cinfo->output_scanline + max_lines > cinfo->output_height) {
    max_lines = cinfo->output_height - cinfo->output_scanline;
  }
  jpegli::ProgressMonitorOutputPass(cinfo);
  size_t num_output_rows = 0;
  while (num_output_rows < max_lines) {
    if (jpegli::IsInputReady(cinfo)) {
      jpegli::ProcessOutput(cinfo, &num_output_rows, scanlines, max_lines);
    } else if (jpegli::ConsumeInput(cinfo) == JPEG_SUSPENDED) {
      break;
    }
  }
  return num_output_rows;
}

JDIMENSION jpegli_skip_scanlines(j_decompress_ptr cinfo, JDIMENSION num_lines) {
  // TODO(szabadka) Skip the IDCT for skipped over blocks.
  return jpegli_read_scanlines(cinfo, nullptr, num_lines);
}

void jpegli_crop_scanline(j_decompress_ptr cinfo, JDIMENSION* xoffset,
                          JDIMENSION* width) {
  jpeg_decomp_master* m = cinfo->master;
  if ((cinfo->global_state != jpegli::kDecProcessScan &&
       cinfo->global_state != jpegli::kDecProcessMarkers) ||
      cinfo->output_scanline != 0) {
    JPEGLI_ERROR("jpegli_crop_decompress: unexpected state %d",
                 cinfo->global_state);
  }
  if (cinfo->raw_data_out) {
    JPEGLI_ERROR("Output cropping is not supported in raw data mode");
  }
  if (xoffset == nullptr || width == nullptr || *width == 0 ||
      *xoffset + *width > cinfo->output_width) {
    JPEGLI_ERROR("jpegli_crop_scanline: Invalid arguments");
  }
  // TODO(szabadka) Skip the IDCT for skipped over blocks.
  size_t xend = *xoffset + *width;
  size_t iMCU_width = m->min_scaled_dct_size * cinfo->max_h_samp_factor;
  *xoffset = (*xoffset / iMCU_width) * iMCU_width;
  *width = xend - *xoffset;
  cinfo->master->xoffset_ = *xoffset;
  cinfo->output_width = *width;
}

JDIMENSION jpegli_read_raw_data(j_decompress_ptr cinfo, JSAMPIMAGE data,
                                JDIMENSION max_lines) {
  if ((cinfo->global_state != jpegli::kDecProcessScan &&
       cinfo->global_state != jpegli::kDecProcessMarkers) ||
      !cinfo->raw_data_out) {
    JPEGLI_ERROR("jpegli_read_raw_data: unexpected state %d",
                 cinfo->global_state);
  }
  size_t iMCU_height = cinfo->max_v_samp_factor * DCTSIZE;
  if (max_lines < iMCU_height) {
    JPEGLI_ERROR("jpegli_read_raw_data: output buffer too small");
  }
  jpegli::ProgressMonitorOutputPass(cinfo);
  while (!jpegli::IsInputReady(cinfo)) {
    if (jpegli::ConsumeInput(cinfo) == JPEG_SUSPENDED) {
      return 0;
    }
  }
  if (cinfo->output_iMCU_row < cinfo->total_iMCU_rows) {
    jpegli::ProcessRawOutput(cinfo, data);
    return iMCU_height;
  }
  return 0;
}

jvirt_barray_ptr* jpegli_read_coefficients(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  if (!cinfo->buffered_image && cinfo->global_state == jpegli::kDecHeaderDone) {
    jpegli::InitProgressMonitor(cinfo, /*coef_only=*/true);
    cinfo->global_state = jpegli::kDecProcessScan;
  }
  if (cinfo->global_state != jpegli::kDecProcessScan &&
      cinfo->global_state != jpegli::kDecProcessMarkers) {
    JPEGLI_ERROR("jpegli_read_coefficients: unexpected state %d",
                 cinfo->global_state);
  }
  if (!cinfo->buffered_image) {
    while (!m->found_eoi_) {
      jpegli::ProgressMonitorInputPass(cinfo);
      int retcode = jpegli::ConsumeInput(cinfo);
      if (retcode == JPEG_SUSPENDED) {
        return nullptr;
      }
    }
  }
  j_common_ptr comptr = reinterpret_cast<j_common_ptr>(cinfo);
  jvirt_barray_ptr* coef_arrays = jpegli::Allocate<jvirt_barray_ptr>(
      cinfo, cinfo->num_components, JPOOL_IMAGE);
  for (int c = 0; c < cinfo->num_components; ++c) {
    size_t xsize_blocks = cinfo->comp_info[c].width_in_blocks;
    size_t ysize_blocks = cinfo->comp_info[c].height_in_blocks;
    coef_arrays[c] = (*cinfo->mem->request_virt_barray)(
        comptr, JPOOL_IMAGE, FALSE, xsize_blocks, ysize_blocks, 1);
  }
  (*cinfo->mem->realize_virt_arrays)(comptr);
  for (int c = 0; c < cinfo->num_components; ++c) {
    jpeg_component_info* comp = &cinfo->comp_info[c];
    for (size_t by = 0; by < comp->height_in_blocks; ++by) {
      JBLOCKARRAY ba = (*cinfo->mem->access_virt_barray)(comptr, coef_arrays[c],
                                                         by, 1, true);
      size_t stride = comp->width_in_blocks * sizeof(JBLOCK);
      size_t offset = by * comp->width_in_blocks * DCTSIZE2;
      memcpy(ba[0], &m->components_[c].coeffs[offset], stride);
    }
  }
  return coef_arrays;
}

boolean jpegli_finish_decompress(j_decompress_ptr cinfo) {
  if (cinfo->global_state != jpegli::kDecProcessScan &&
      cinfo->global_state != jpegli::kDecProcessMarkers) {
    JPEGLI_ERROR("jpegli_finish_decompress: unexpected state %d",
                 cinfo->global_state);
  }
  while (!cinfo->master->found_eoi_) {
    int retcode = jpegli::ConsumeInput(cinfo);
    if (retcode == JPEG_SUSPENDED) {
      return FALSE;
    }
  }
  (*cinfo->src->term_source)(cinfo);
  jpegli_abort_decompress(cinfo);
  return TRUE;
}

boolean jpegli_resync_to_restart(j_decompress_ptr cinfo, int desired) {
  // The default resync_to_restart will just throw an error.
  JPEGLI_ERROR("Invalid restart marker found.");
  return TRUE;
}

void jpegli_new_colormap(j_decompress_ptr cinfo) {
  if (cinfo->global_state != jpegli::kDecProcessScan &&
      cinfo->global_state != jpegli::kDecProcessMarkers) {
    JPEGLI_ERROR("jpegli_new_colormap: unexpected state %d",
                 cinfo->global_state);
  }
  if (!cinfo->buffered_image) {
    JPEGLI_ERROR("jpegli_new_colormap: not in  buffered image mode");
  }
  if (!cinfo->enable_external_quant) {
    JPEGLI_ERROR("external quantization was not enabled");
  }
  if (!cinfo->quantize_colors || cinfo->colormap == nullptr) {
    JPEGLI_ERROR("jpegli_new_colormap: not in external colormap mode");
  }
  cinfo->master->regenerate_inverse_colormap_ = true;
}

void jpegli_set_output_format(j_decompress_ptr cinfo, JpegliDataType data_type,
                              JpegliEndianness endianness) {
  cinfo->master->output_data_type_ = data_type;
  cinfo->master->swap_endianness_ =
      ((endianness == JPEGLI_BIG_ENDIAN && IsLittleEndian()) ||
       (endianness == JPEGLI_LITTLE_ENDIAN && !IsLittleEndian()));
}
