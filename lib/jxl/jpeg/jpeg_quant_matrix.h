// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Data structures that represent the contents of a jpeg file.

#ifndef LIB_JXL_JPEG_JPEG_QUANT_MATRIX_H_
#define LIB_JXL_JPEG_JPEG_QUANT_MATRIX_H_

#include <stddef.h>
#include <stdint.h>

#include "lib/jxl/jpeg/jpeg_constants.h"

namespace jxl {
namespace jpeg {

static const size_t kQFactorBits = 6;
static const size_t kQFactorLimit = 1u << kQFactorBits;

void FillQuantMatrix(bool is_chroma, uint32_t q, uint8_t dst[kDCTBlockSize]);
uint32_t FindBestMatrix(const int* src, bool is_chroma,
                        uint8_t dst[kDCTBlockSize]);

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_JPEG_QUANT_MATRIX_H_
