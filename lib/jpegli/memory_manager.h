// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_MEMORY_MANAGER_H_
#define LIB_JPEGLI_MEMORY_MANAGER_H_

/* clang-format off */
#include <stdio.h>
#include <jpeglib.h>
#include <stdlib.h>
/* clang-format on */

namespace jpegli {

void InitMemoryManager(j_common_ptr cinfo);

template <typename T>
T* Allocate(j_common_ptr cinfo, size_t len, int pool_id = JPOOL_PERMANENT) {
  void* p = (*cinfo->mem->alloc_small)(cinfo, pool_id, len * sizeof(T));
  return reinterpret_cast<T*>(p);
}

template <typename T>
T* Allocate(j_decompress_ptr cinfo, size_t len, int pool_id = JPOOL_PERMANENT) {
  return Allocate<T>(reinterpret_cast<j_common_ptr>(cinfo), len, pool_id);
}

template <typename T>
T* Allocate(j_compress_ptr cinfo, size_t len, int pool_id = JPOOL_PERMANENT) {
  return Allocate<T>(reinterpret_cast<j_common_ptr>(cinfo), len, pool_id);
}

}  // namespace jpegli

#endif  // LIB_JPEGLI_MEMORY_MANAGER_H_
