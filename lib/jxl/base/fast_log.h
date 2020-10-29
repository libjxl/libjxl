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

#ifndef LIB_JXL_BASE_FAST_LOG_H_
#define LIB_JXL_BASE_FAST_LOG_H_

#include <stdint.h>
#include <string.h>

#include <cmath>

namespace jxl {

// L1 error ~9.1E-3 (see fast_log_test).
static inline float FastLog2f(float f) {
  int32_t f_bits;
  memcpy(&f_bits, &f, 4);
  int exp = ((f_bits >> 23) & 0xFF) - 126;
  uint32_t fr_bits = (f_bits & 0x807fffff) | 0x3f000000;
  float fr;
  memcpy(&fr, &fr_bits, 4);
  // TODO(veluca): improve constants.
  return exp + (-1.34752046f * fr + 3.98979143f) * fr - 2.64898502f;
}

}  // namespace jxl

#endif  // LIB_JXL_BASE_FAST_LOG_H_
