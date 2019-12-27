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

#ifndef JXL_MODULAR_MA_UTIL_H_
#define JXL_MODULAR_MA_UTIL_H_

#include <stdint.h>

#include "jxl/modular/ma/bit.h"

namespace jxl {

static inline int ilog2(uint32_t l) {
  if (l == 0) {
    return 0;
  }
  return sizeof(unsigned int) * 8 - 1 - __builtin_clz(l);
}

}  // namespace jxl

#endif  // JXL_MODULAR_MA_UTIL_H_
