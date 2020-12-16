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

#ifndef LIB_JXL_JPEG_ENC_JPEG_STATE_H_
#define LIB_JXL_JPEG_ENC_JPEG_STATE_H_

#include <stddef.h>
#include <stdint.h>

#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {
namespace jpeg {

bool BrunsliSerialize(const JPEGData& jpg, uint32_t skip_sections,
                      uint8_t* data, size_t* len);

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_ENC_JPEG_STATE_H_
