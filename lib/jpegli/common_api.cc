// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/* clang-format off */
#include <stdint.h>
#include <stdio.h>
#include <jpeglib.h>
#include <stdlib.h>
/* clang-format on */

#include "lib/jpegli/decode_internal.h"
#include "lib/jpegli/encode_internal.h"
#include "lib/jpegli/memory_manager.h"

void jpeg_abort(j_common_ptr cinfo) {
  auto mem = reinterpret_cast<jpegli::MemoryManager*>(cinfo->mem);
  if (mem == nullptr) {
    return;
  }
  for (void* ptr : mem->owned_ptrs) {
    free(ptr);
  }
  mem->owned_ptrs.clear();
  if (cinfo->is_decompressor) {
    cinfo->global_state = jpegli::DecodeState::kStart;
  }
}

void jpeg_destroy(j_common_ptr cinfo) {
  auto mem = reinterpret_cast<jpegli::MemoryManager*>(cinfo->mem);
  if (mem == nullptr) {
    return;
  }
  for (void* ptr : mem->owned_ptrs) {
    free(ptr);
  }
  delete mem;
  cinfo->mem = nullptr;
  if (cinfo->is_decompressor) {
    cinfo->global_state = jpegli::DecodeState::kNull;
    delete reinterpret_cast<j_decompress_ptr>(cinfo)->master;
  } else {
    delete reinterpret_cast<j_compress_ptr>(cinfo)->master;
  }
}

JQUANT_TBL* jpeg_alloc_quant_table(j_common_ptr cinfo) {
  JQUANT_TBL* table = jpegli::Allocate<JQUANT_TBL>(cinfo, 1);
  table->sent_table = FALSE;
  return table;
}

JHUFF_TBL* jpeg_alloc_huff_table(j_common_ptr cinfo) {
  JHUFF_TBL* table = jpegli::Allocate<JHUFF_TBL>(cinfo, 1);
  table->sent_table = FALSE;
  return table;
}
