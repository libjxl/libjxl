// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/compressed_icc.h>

JXL_BOOL JxlIccProfileEncode(JxlMemoryManager* memory_manager,
                             const uint8_t* icc, size_t icc_size,
                             uint8_t** compressed_icc,
                             size_t* compressed_icc_size) {
  return JXL_TRUE;
}

JXL_BOOL JxlIccProfileDecode(JxlMemoryManager* memory_manager,
                             const uint8_t* compressed_icc,
                             size_t compressed_icc_size, uint8_t** icc,
                             size_t* icc_size) {
  return JXL_TRUE;
}
