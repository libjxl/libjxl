// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/* clang-format off */
#include <stdint.h>
#include <stdio.h>
#include <jpeglib.h>
#include <string.h>
/* clang-format on */

#include <vector>

#include "lib/jpegli/decode_internal.h"
#include "lib/jpegli/decode_marker.h"
#include "lib/jpegli/decode_scan.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/memory_manager.h"
#include "lib/jpegli/render.h"
#include "lib/jpegli/source_manager.h"
#include "lib/jxl/base/status.h"

namespace jpegli {

enum DecodeState {
  kStart,
  kInHeader,
  kHeaderDone,
  kProcessMarkers,
  kProcessScan,
};

void InitializeImage(j_decompress_ptr cinfo) {
  cinfo->jpeg_color_space = JCS_UNKNOWN;
  cinfo->restart_interval = 0;
  cinfo->saw_JFIF_marker = FALSE;
  cinfo->JFIF_major_version = 1;
  cinfo->JFIF_minor_version = 1;
  cinfo->density_unit = 0;
  cinfo->X_density = 1;
  cinfo->Y_density = 1;
  cinfo->saw_Adobe_marker = FALSE;
  cinfo->Adobe_transform = 0;
}

int ConsumeInput(j_decompress_ptr cinfo) {
  jpeg_source_mgr* src = cinfo->src;
  std::vector<uint8_t> buffer;
  const uint8_t* last_input_byte = src->next_input_byte + src->bytes_in_buffer;
  int status;
  for (;;) {
    if (cinfo->global_state == kProcessScan) {
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
    cinfo->global_state = kProcessMarkers;
  } else if (status == JPEG_REACHED_SOS) {
    cinfo->global_state =
        cinfo->global_state == kInHeader ? kHeaderDone : kProcessScan;
  }
  return status;
}

}  // namespace jpegli

void jpeg_CreateDecompress(j_decompress_ptr cinfo, int version,
                           size_t structsize) {
  if (structsize != sizeof(*cinfo)) {
    JPEGLI_ERROR("jpeg_decompress_struct has wrong size.");
  }
  cinfo->master = new jpeg_decomp_master;
  cinfo->mem =
      reinterpret_cast<struct jpeg_memory_mgr*>(new jpegli::MemoryManager);
  cinfo->marker_list = nullptr;
  cinfo->input_scan_number = 0;
  cinfo->quantize_colors = FALSE;
  cinfo->desired_number_of_colors = 0;
  cinfo->master->output_bit_depth_ = 8;
  cinfo->global_state = jpegli::kStart;
}

void jpeg_destroy_decompress(j_decompress_ptr cinfo) {
  auto mem = reinterpret_cast<jpegli::MemoryManager*>(cinfo->mem);
  for (void* ptr : mem->owned_ptrs) {
    free(ptr);
  }
  delete mem;
  delete cinfo->master;
}

void jpeg_abort_decompress(j_decompress_ptr cinfo) {}

void jpeg_save_markers(j_decompress_ptr cinfo, int marker_code,
                       unsigned int length_limit) {
  jpeg_decomp_master* m = cinfo->master;
  m->markers_to_save_.insert(marker_code);
}

int jpeg_consume_input(j_decompress_ptr cinfo) {
  if (cinfo->global_state == jpegli::kStart) {
    (*cinfo->src->init_source)(cinfo);
    jpegli::InitializeImage(cinfo);
    cinfo->global_state = jpegli::kInHeader;
  }
  if (cinfo->global_state == jpegli::kHeaderDone) {
    return JPEG_REACHED_SOS;
  }
  if (cinfo->master->found_eoi_) {
    return JPEG_REACHED_EOI;
  }
  if (cinfo->global_state == jpegli::kInHeader ||
      cinfo->global_state == jpegli::kProcessMarkers ||
      cinfo->global_state == jpegli::kProcessScan) {
    return jpegli::ConsumeInput(cinfo);
  }
  JPEGLI_ERROR("Unexpected state %d", cinfo->global_state);
  return JPEG_REACHED_EOI;  // return value does not matter
}

int jpeg_read_header(j_decompress_ptr cinfo, boolean require_image) {
  if (cinfo->global_state != jpegli::kStart &&
      cinfo->global_state != jpegli::kInHeader) {
    JPEGLI_ERROR("jpeg_read_header: unexpected state %d", cinfo->global_state);
  }
  for (;;) {
    int retcode = jpeg_consume_input(cinfo);
    if (retcode == JPEG_SUSPENDED) {
      return retcode;
    } else if (retcode == JPEG_REACHED_SOS) {
      break;
    } else if (retcode == JPEG_REACHED_EOI) {
      JPEGLI_ERROR("jpeg_read_header: unexpected EOI marker.");
    }
  };
  return JPEG_HEADER_OK;
}

boolean jpeg_read_icc_profile(j_decompress_ptr cinfo, JOCTET** icc_data_ptr,
                              unsigned int* icc_data_len) {
  if (cinfo->global_state == jpegli::kStart ||
      cinfo->global_state == jpegli::kInHeader) {
    JPEGLI_ERROR("jpeg_read_icc_profile: unexpected state %d",
                 cinfo->global_state);
  }
  if (icc_data_ptr == nullptr || icc_data_len == nullptr) {
    JPEGLI_ERROR("jpeg_read_icc_profile: invalid output buffer");
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
    JPEGLI_ERROR("jpeg_read_icc_profile: Out of memory");
  }
  memcpy(*icc_data_ptr, m->icc_profile_.data(), *icc_data_len);
  return TRUE;
}

void jpeg_calc_output_dimensions(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  if (!m->found_sof_) {
    JPEGLI_ERROR("No SOF marker found.");
  }
  // Resampling is not yet implemented.
  cinfo->output_width = cinfo->image_width;
  cinfo->output_height = cinfo->image_height;
  cinfo->output_components = cinfo->out_color_components;
  cinfo->rec_outbuf_height = 1;
  if (!cinfo->quantize_colors) {
    for (size_t depth = 1; depth <= 16; ++depth) {
      if (cinfo->desired_number_of_colors == (1 << depth)) {
        m->output_bit_depth_ = depth;
      }
    }
    // Signal to the application code that we did set the output bit depth.
    cinfo->desired_number_of_colors = 0;
  }
}

boolean jpeg_has_multiple_scans(j_decompress_ptr cinfo) {
  if (cinfo->input_scan_number == 0) {
    JPEGLI_ERROR("No SOS marker found.");
  }
  return cinfo->master->is_multiscan_;
}

boolean jpeg_start_decompress(j_decompress_ptr cinfo) {
  if (cinfo->global_state == jpegli::kHeaderDone) {
    jpeg_calc_output_dimensions(cinfo);
    cinfo->global_state = jpegli::kProcessScan;
  } else if (!cinfo->master->is_multiscan_) {
    JPEGLI_ERROR("jpeg_start_decompress: unexpected state %d",
                 cinfo->global_state);
  }
  if (cinfo->master->is_multiscan_) {
    if (cinfo->global_state != jpegli::kProcessScan &&
        cinfo->global_state != jpegli::kProcessMarkers) {
      JPEGLI_ERROR("jpeg_start_decompress: unexpected state %d",
                   cinfo->global_state);
    }
    while (!cinfo->master->found_eoi_) {
      int retcode = jpegli::ConsumeInput(cinfo);
      if (retcode == JPEG_SUSPENDED) {
        return FALSE;
      }
    }
  }
  jpegli::PrepareForOutput(cinfo);
  return TRUE;
}

JDIMENSION jpeg_read_scanlines(j_decompress_ptr cinfo, JSAMPARRAY scanlines,
                               JDIMENSION max_lines) {
  jpeg_decomp_master* m = cinfo->master;
  if (cinfo->global_state != jpegli::kProcessScan &&
      cinfo->global_state != jpegli::kProcessMarkers) {
    JPEGLI_ERROR("jpeg_read_scanlines: unexpected state %d",
                 cinfo->global_state);
  }
  if (m->is_multiscan_ && !m->found_eoi_) {
    JPEGLI_ERROR("jpeg_read_scanlines: jpeg_start_decompress() did not finish");
  }
  size_t num_output_rows = 0;
  while (cinfo->output_scanline < cinfo->output_height &&
         num_output_rows < max_lines) {
    if (cinfo->input_iMCU_row <= cinfo->output_iMCU_row &&
        cinfo->output_iMCU_row < cinfo->total_iMCU_rows &&
        jpegli::ConsumeInput(cinfo) == JPEG_SUSPENDED) {
      return num_output_rows;
    }
    jpegli::ProcessOutput(cinfo, &num_output_rows, scanlines, max_lines);
  }
  return num_output_rows;
}

boolean jpeg_finish_decompress(j_decompress_ptr cinfo) {
  if (cinfo->global_state != jpegli::kProcessScan &&
      cinfo->global_state != jpegli::kProcessMarkers) {
    JPEGLI_ERROR("jpeg_finish_decompress: unexpected state %d",
                 cinfo->global_state);
  }
  while (!cinfo->master->found_eoi_) {
    int retcode = jpegli::ConsumeInput(cinfo);
    if (retcode == JPEG_SUSPENDED) {
      return FALSE;
    }
  }
  return TRUE;
}
