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

typedef jpeg_decomp_master::State State;

void jpeg_CreateDecompress(j_decompress_ptr cinfo, int version,
                           size_t structsize) {
  if (structsize != sizeof(*cinfo)) {
    JPEGLI_ERROR("jpeg_decompress_struct has wrong size.");
  }
  cinfo->master = new jpeg_decomp_master;
  cinfo->mem =
      reinterpret_cast<struct jpeg_memory_mgr*>(new jpegli::MemoryManager);
  cinfo->marker_list = nullptr;
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
  jpeg_decomp_master* m = cinfo->master;
  const uint8_t* data = cinfo->src->next_input_byte;
  size_t len = cinfo->src->bytes_in_buffer;
  size_t pos = 0;
  std::vector<uint8_t> buffer;
  const uint8_t* last_src_buf_start = data;
  size_t last_src_buf_len = len;

  while (!m->found_sos_) {
    bool status = true;
    if (m->state_ == State::kStart) {
      // Look for the SOI marker.
      if (len >= 2) {
        if (data[0] != 0xff || data[1] != 0xd8) {
          JPEGLI_ERROR("Did not find SOI marker.");
        }
        pos += 2;
        jpegli::AdvanceInput(cinfo, 2);
        m->found_soi_ = true;
        m->state_ = State::kProcessMarkers;
      } else {
        status = false;
      }
    } else if (m->state_ == State::kProcessMarkers) {
      status = jpegli::ProcessMarker(cinfo, data, len, &pos);
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
        return JPEG_SUSPENDED;
      }
    }
  }

  if (!buffer.empty()) {
    cinfo->src->next_input_byte =
        (last_src_buf_start + last_src_buf_len - buffer.size() + pos);
    cinfo->src->bytes_in_buffer = buffer.size() - pos;
  }
  return JPEG_HEADER_OK;
}

boolean jpeg_start_decompress(j_decompress_ptr cinfo) {
  jpeg_decomp_master* m = cinfo->master;
  if (m->is_progressive_) {
    const uint8_t* data = cinfo->src->next_input_byte;
    size_t len = cinfo->src->bytes_in_buffer;
    size_t pos = 0;
    std::vector<uint8_t> buffer;
    const uint8_t* last_src_buf_start = data;
    size_t last_src_buf_len = len;
    while (!m->found_eoi_) {
      bool status = true;
      if (m->state_ == State::kProcessMarkers) {
        status = jpegli::ProcessMarker(cinfo, data, len, &pos);
      } else if (m->state_ == State::kScan) {
        status = jpegli::ProcessScan(cinfo, data, len, &pos);
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
          return FALSE;
        }
      }
    }
    if (!buffer.empty()) {
      cinfo->src->next_input_byte =
          (last_src_buf_start + last_src_buf_len - buffer.size() + pos);
      cinfo->src->bytes_in_buffer = buffer.size() - pos;
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
  size_t num_output_rows = 0;
  const uint8_t* data = cinfo->src->next_input_byte;
  size_t len = cinfo->src->bytes_in_buffer;
  size_t pos = 0;
  std::vector<uint8_t> buffer;
  const uint8_t* last_src_buf_start = data;
  size_t last_src_buf_len = len;

  while (num_output_rows < max_lines) {
    bool status = true;
    if (m->state_ == State::kProcessMarkers) {
      status = jpegli::ProcessMarker(cinfo, data, len, &pos);
    } else if (m->state_ == State::kScan) {
      status = jpegli::ProcessScan(cinfo, data, len, &pos);
    } else if (m->state_ == State::kRender) {
      jpegli::ProcessOutput(cinfo, &num_output_rows, scanlines, max_lines);
    } else if (m->state_ == State::kEnd) {
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
        return num_output_rows;
      }
    }
  }
  if (!buffer.empty()) {
    cinfo->src->next_input_byte =
        (last_src_buf_start + last_src_buf_len - buffer.size() + pos);
    cinfo->src->bytes_in_buffer = buffer.size() - pos;
  }
  return num_output_rows;
}

boolean jpeg_finish_decompress(j_decompress_ptr cinfo) { return TRUE; }
