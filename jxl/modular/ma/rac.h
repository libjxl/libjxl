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

#ifndef JXL_MODULAR_MA_RAC_H_
#define JXL_MODULAR_MA_RAC_H_

#include <stdint.h>

#include "jxl/base/status.h"
#include "jxl/modular/config.h"

namespace jxl {

/* RAC configuration for 24-bit RAC */
class RacConfig24 {
 public:
  typedef uint_fast32_t data_t;
  //  typedef uint32_t data_t;
  static const data_t MAX_RANGE_BITS = 24;
  static const data_t MIN_RANGE_BITS = 16;
  static const data_t MIN_RANGE = (1UL << MIN_RANGE_BITS);
  static const data_t BASE_RANGE = (1UL << MAX_RANGE_BITS);

  static inline data_t chance_12bit_chance(uint16_t b12,
                                           data_t range) ATTRIBUTE_HOT {
    //        JXL_DASSERT(b12 > 0);
    JXL_DASSERT((b12 >> 12) == 0);
    if (sizeof(data_t) > 4)
      return (range * b12) >> 12;
    else
      return ((((range & 0xFFF) * b12) >> 12) + ((range >> 12) * b12));
    //    if (sizeof(data_t) > 4) return (range * b12 + 0x800) >> 12;
    //    else return ((((range & 0xFFF) * b12) >> 12) + ((range >> 12) * b12));
    // We want to compute (range * b12 + 0x800) >> 12. On 64-bit architectures
    // this is no problem
    //        if (sizeof(data_t) > 4) return (range * b12 + 0x800) >> 12;
    // Unfortunately, this can overflow the 32-bit data type on 32-bit
    // architectures, so split range in its lower and upper 12 bits, and compute
    // separately.
    //        else return ((((range & 0xFFF) * b12 + 0x800) >> 12) + ((range >>
    //        12) * b12));
    // (no worries, the compiler eliminates this branching)
  }
};

template <typename Config, typename IO>
class RacInput {
 public:
  typedef typename Config::data_t rac_t;

 private:
  rac_t range;
  rac_t low;

 protected:
  IO& io;

 private:
  rac_t read_catch_eof() {
    rac_t c = io.get_c();
    // no reason to branch here to catch end-of-stream, just return garbage
    // (0xFF I guess) if a premature EOS happens
    // if(c == io.EOS) return 0;
    return c;
  }
  void inline input() {
    if (JXL_UNLIKELY(range <= Config::MIN_RANGE)) {
      low <<= 8;
      range <<= 8;
      low |= read_catch_eof();
      if (JXL_UNLIKELY(range <= Config::MIN_RANGE)) {
        low <<= 8;
        range <<= 8;
        low |= read_catch_eof();
      }
    }
  }
  bool inline get(const rac_t chance) ATTRIBUTE_HOT {
    JXL_DASSERT(chance >= 0);
    JXL_DASSERT(chance < range);
    bool result = (low >= range - chance);
    if (result) {
      low -= range - chance;
      range = chance;
    } else {
      range -= chance;
    }
    input();
    return result;
  }

 public:
  explicit RacInput(IO& ioin) : range(Config::BASE_RANGE), low(0), io(ioin) {
    rac_t r = Config::BASE_RANGE;
    while (r > 1) {
      low <<= 8;
      low |= read_catch_eof();
      r >>= 8;
    }
  }

  bool inline read_12bit_chance(uint16_t b12) ATTRIBUTE_HOT {
    return get(Config::chance_12bit_chance(b12, range));
  }

  bool inline read_bit() { return get(range >> 1); }
};

template <typename IO>
class RacInput24 : public RacInput<RacConfig24, IO> {
 public:
  explicit RacInput24(IO& io) : RacInput<RacConfig24, IO>(io) {}
};

template <typename IO>
using RacIn = RacInput24<IO>;

}  // namespace jxl

#endif  // JXL_MODULAR_MA_RAC_H_
