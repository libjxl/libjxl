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

#include "jxl/modular/ma/chance.h"

namespace jxl {

/** Computes an approximation of log(4096 / x) / log(2) * base */
static uint32_t log4kf(int x, uint32_t base) {
  int bits = 8 * sizeof(int) - __builtin_clz(x);
  uint64_t y = ((uint64_t)x) << (32 - bits);
  uint32_t res = base * (13 - bits);
  uint32_t add = base;
  while ((add > 1) && ((y & 0x7FFFFFFF) != 0)) {
    y = (((uint64_t)y) * y + 0x40000000) >> 31;
    add >>= 1;
    if ((y >> 32) != 0) {
      res -= add;
      y >>= 1;
    }
  }
  return res;
}

Log4kTable::Log4kTable() {
  data[0] = 0;
  for (int i = 1; i <= 4096; i++) {
    data[i] = (log4kf(i, (65535UL << 16) / 12) + (1 << 15)) >> 16;
  }
}

const Log4kTable log4k;

}  // namespace jxl