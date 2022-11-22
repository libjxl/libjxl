// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JPEGLI_MEMORY_MANAGER_H_
#define LIB_JPEGLI_MEMORY_MANAGER_H_

/* clang-format off */
#include <stdint.h>
#include <stdio.h>
#include <jpeglib.h>
/* clang-format on */

#include <vector>

namespace jpegli {

struct MemoryManager {
  struct jpeg_memory_mgr pub;
  std::vector<void*> owned_ptrs;
};

}  // namespace jpegli

#endif  // LIB_JPEGLI_MEMORY_MANAGER_H_
