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

#ifndef JXL_MODULAR_MA_CHANCE_H_
#define JXL_MODULAR_MA_CHANCE_H_

#include <stddef.h>
#include <stdint.h>

#include <cmath>

namespace jxl {

struct Log4kTable {
  uint16_t data[4097];
  Log4kTable();
};

extern const Log4kTable log4k;

class SimpleBitChance {
 protected:
  uint16_t chance;  // stored as a 12-bit number

 public:
  SimpleBitChance() { chance = 0x800; }

  uint16_t inline get_12bit() const { return chance; }
  void set_12bit(uint16_t chance) { this->chance = chance; }
  // update chances according to this function
  void inline put(bool bit) {
    if (bit)
      chance += ((0x1000 - chance) >> 5);
    else
      chance -= ((chance) >> 5);
  }
  // can also add 30 before shifting, to make sure that the really extreme
  // chances can be reached. Doesn't seem to help for compression though, and
  // it's quite a bit slower

  void estim(bool bit, uint64_t &total) const {
    total += log4k.data[bit ? chance : 4096 - chance];
  }
};

}  // namespace jxl

#endif  // JXL_MODULAR_MA_CHANCE_H_
