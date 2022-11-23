// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/source_manager.h"

#include "lib/jpegli/memory_manager.h"

namespace jpegli {

void init_source(j_decompress_ptr cinfo) {}

void skip_input_data(j_decompress_ptr cinfo, long num_bytes) {}

boolean resync_to_restart(j_decompress_ptr cinfo, int desired) { return FALSE; }

void term_source(j_decompress_ptr cinfo) {}

boolean EmitFakeEoiMarker(j_decompress_ptr cinfo) {
  static constexpr uint8_t kFakeEoiMarker[2] = {0xff, 0xd9};
  cinfo->src->next_input_byte = kFakeEoiMarker;
  cinfo->src->bytes_in_buffer = 2;
  return TRUE;
}

}  // namespace jpegli

void jpeg_mem_src(j_decompress_ptr cinfo, const unsigned char* inbuffer,
                  unsigned long insize) {
  cinfo->src = jpegli::Allocate<jpeg_source_mgr>(cinfo, 1);
  cinfo->src->next_input_byte = inbuffer;
  cinfo->src->bytes_in_buffer = insize;
  cinfo->src->init_source = jpegli::init_source;
  cinfo->src->fill_input_buffer = jpegli::EmitFakeEoiMarker;
  cinfo->src->skip_input_data = jpegli::skip_input_data;
  cinfo->src->resync_to_restart = jpegli::resync_to_restart;
  cinfo->src->term_source = jpegli::term_source;
}
