// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/* clang-format off */
#include <stdint.h>
#include <stdio.h>
#include <jpeglib.h>
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

#define JPEGLI_STATE_READ_HEADER 1
#define JPEGLI_STATE_START_DECOMPRESS 2
#define JPEGLI_STATE_READ_SCANLINES 3

typedef jpeg_decomp_master::State State;

namespace jpegli {

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

bool ShouldStop(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  if (cinfo->global_state == JPEGLI_STATE_READ_HEADER) {
    return cinfo->input_scan_number > 0;
  } else if (cinfo->global_state == JPEGLI_STATE_START_DECOMPRESS) {
    return m->found_eoi_;
  } else if (cinfo->global_state == JPEGLI_STATE_READ_SCANLINES) {
    return m->num_output_rows_ >= m->max_lines_;
  }
  JPEGLI_ERROR("Unexpected global state");
  return false;
}

bool DoSomeWork(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  const uint8_t* data = cinfo->src->next_input_byte;
  size_t len = cinfo->src->bytes_in_buffer;
  size_t pos = 0;
  std::vector<uint8_t> buffer;
  const uint8_t* last_src_buf_start = data;
  size_t last_src_buf_len = len;

  while (!ShouldStop(cinfo)) {
    bool status = true;
    if (cinfo->global_state == JPEGLI_STATE_READ_HEADER &&
        m->state_ == State::kStart) {
      // Look for the SOI marker.
      if (len >= 2) {
        if (data[0] != 0xff || data[1] != 0xd8) {
          JPEGLI_ERROR("Did not find SOI marker.");
        }
        pos += 2;
        jpegli::AdvanceInput(cinfo, 2);
        InitializeImage(cinfo);
        m->found_soi_ = true;
        m->state_ = State::kProcessMarkers;
      } else {
        status = false;
      }
    } else if (m->state_ == State::kProcessMarkers) {
      status = jpegli::ProcessMarker(cinfo, data, len, &pos);
    } else if ((cinfo->global_state == JPEGLI_STATE_START_DECOMPRESS ||
                cinfo->global_state == JPEGLI_STATE_READ_SCANLINES) &&
               m->state_ == State::kScan) {
      status = jpegli::ProcessScan(cinfo, data, len, &pos);
    } else if (cinfo->global_state == JPEGLI_STATE_READ_SCANLINES &&
               m->state_ == State::kRender) {
      jpegli::ProcessOutput(cinfo, &m->num_output_rows_, m->scanlines_,
                            m->max_lines_);
    } else if (cinfo->global_state == JPEGLI_STATE_READ_SCANLINES &&
               m->state_ == State::kEnd) {
      break;
    } else {
      JPEGLI_ERROR("Unexpected state.");
    }
    if (!status) {
      if (buffer.empty()) {
        buffer.assign(data, data + len);
      }
      if ((*cinfo->src->fill_input_buffer)(cinfo)) {
        buffer.insert(
            buffer.end(), cinfo->src->next_input_byte,
            cinfo->src->next_input_byte + cinfo->src->bytes_in_buffer);
        data = buffer.data();
        len = buffer.size();
        last_src_buf_start = cinfo->src->next_input_byte;
        last_src_buf_len = cinfo->src->bytes_in_buffer;
        cinfo->src->next_input_byte = data + pos;
        cinfo->src->bytes_in_buffer = len - pos;
      } else {
        return false;
      }
    }
  }

  if (!buffer.empty()) {
    cinfo->src->next_input_byte =
        (last_src_buf_start + last_src_buf_len - buffer.size() + pos);
    cinfo->src->bytes_in_buffer = buffer.size() - pos;
  }
  return true;
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

int jpeg_read_header(j_decompress_ptr cinfo, boolean require_image) {
  cinfo->global_state = JPEGLI_STATE_READ_HEADER;
  bool success = jpegli::DoSomeWork(cinfo);
  return success ? JPEG_HEADER_OK : JPEG_SUSPENDED;
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
  m->output_bit_depth_ = 8;
  if (!cinfo->quantize_colors) {
    for (size_t depth = 1; depth <= 16; ++depth) {
      if (cinfo->desired_number_of_colors == (1 << depth)) {
        m->output_bit_depth_ = depth;
      }
    }
  }
}

boolean jpeg_start_decompress(j_decompress_ptr cinfo) {
  jpeg_calc_output_dimensions(cinfo);
  if (cinfo->progressive_mode) {
    cinfo->global_state = JPEGLI_STATE_START_DECOMPRESS;
    if (!jpegli::DoSomeWork(cinfo)) {
      return FALSE;
    }
  }
  jpegli::PrepareForOutput(cinfo);
  return TRUE;
}

JDIMENSION jpeg_read_scanlines(j_decompress_ptr cinfo, JSAMPARRAY scanlines,
                               JDIMENSION max_lines) {
  jpeg_decomp_master* m = cinfo->master;
  if (max_lines == 0 || m->state_ == State::kEnd) {
    return 0;
  }
  cinfo->global_state = JPEGLI_STATE_READ_SCANLINES;
  m->num_output_rows_ = 0;
  m->scanlines_ = scanlines;
  m->max_lines_ = max_lines;
  jpegli::DoSomeWork(cinfo);
  return m->num_output_rows_;
}

boolean jpeg_finish_decompress(j_decompress_ptr cinfo) { return TRUE; }
