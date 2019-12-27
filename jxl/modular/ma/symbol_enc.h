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

#ifndef JXL_MODULAR_MA_SYMBOL_ENC_H_
#define JXL_MODULAR_MA_SYMBOL_ENC_H_

#include <stdlib.h>

#include "jxl/base/status.h"
#include "jxl/modular/ma/symbol.h"

namespace jxl {

template <typename RAC>
void UniformSymbolCoder<RAC>::write_int(int min, int max, int val) {
  JXL_DASSERT(max >= min);
  if (min != 0) {
    max -= min;
    val -= min;
  }
  if (max == 0) return;

  // split in [0..med] [med+1..max]
  int med = max / 2;
  if (val > med) {
    rac.write_bit(true);
    write_int(med + 1, max, val);
  } else {
    rac.write_bit(false);
    write_int(0, med, val);
  }
  return;
}

template <typename SymbolCoder>
void writer(SymbolCoder& coder, int bits, int value) {
  int pos = 0;
  while (pos++ < bits) {
    coder.write(value & 1, BIT_MANT, pos);
    value >>= 1;
  }
}

template <int bits, typename SymbolCoder>
void writer(SymbolCoder& coder, int min, int max, int value) {
  JXL_DASSERT(min <= max);
  JXL_DASSERT(value >= min);
  JXL_DASSERT(value <= max);

  // avoid doing anything if the value is already known
  if (min == max) return;

  if (value == 0) {  // value is zero
    coder.write(true, BIT_ZERO);
    return;
  }

  JXL_DASSERT(min <= 0 && max >= 0);  // should always be the case, because
                                      // guess should always be in valid range

  coder.write(false, BIT_ZERO);
  int sign = (value > 0 ? 1 : 0);
  if (max > 0 && min < 0) {
    // only output sign bit if value can be both pos and neg
    coder.write(sign, BIT_SIGN);
  }
  const int a = abs(value);
  const int e = ilog2(a);
  int amax = sign ? abs(max) : abs(min);

  int emax = ilog2(amax);

  int i = 0;  // ilog2(amin);
  while (i < emax) {
    // if exponent >i is impossible, we are done
    if ((1 << (i + 1)) > amax) break;
    // if exponent i is possible, output the exponent bit
    coder.write(i == e, BIT_EXP, i);
    if (i == e) break;
    i++;
  }

  int have = (1 << e);
  int left = have - 1;
  for (int pos = e; pos > 0;) {
    int bit = 1;
    left ^= (1 << (--pos));
    int minabs1 = have | (1 << pos);
    if (minabs1 > amax) {  // 1-bit is impossible
      bit = 0;
    } else {
      bit = (a >> pos) & 1;
      coder.write(bit, BIT_MANT, pos);
    }
    have |= (bit << pos);
  }
}

template <int bits, typename SymbolCoder>
int estimate_writer(SymbolCoder& coder, int min, int max, int value) {
  JXL_DASSERT(min <= max);
  JXL_DASSERT(value >= min);
  JXL_DASSERT(value <= max);

  uint64_t total = 0;

  // avoid doing anything if the value is already known
  if (min == max) return total;

  if (value == 0) {  // value is zero
    coder.estimate(true, BIT_ZERO, 0, total);
    return total;
  }

  JXL_DASSERT(min <= 0 && max >= 0);  // should always be the case, because
                                      // guess should always be in valid range

  coder.estimate(false, BIT_ZERO, 0, total);
  int sign = (value > 0 ? 1 : 0);
  if (max > 0 && min < 0) {
    // only output sign bit if value can be both pos and neg
    coder.estimate(sign, BIT_SIGN, 0, total);
  }
  if (sign) min = 1;
  if (!sign) max = -1;
  const int a = abs(value);
  const int e = ilog2(a);
  int amin = sign ? abs(min) : abs(max);
  int amax = sign ? abs(max) : abs(min);

  int emax = ilog2(amax);
  int i = ilog2(amin);

  while (i < emax) {
    // if exponent >i is impossible, we are done
    if ((1 << (i + 1)) > amax) break;
    // if exponent i is possible, output the exponent bit
    coder.estimate(i == e, BIT_EXP, i, total);
    if (i == e) break;
    i++;
  }

  int have = (1 << e);
  int left = have - 1;
  for (int pos = e; pos > 0;) {
    int bit = 1;
    left ^= (1 << (--pos));
    int minabs1 = have | (1 << pos);
    if (minabs1 > amax) {  // 1-bit is impossible
      bit = 0;
    } else {
      bit = (a >> pos) & 1;
      coder.estimate(bit, BIT_MANT, pos, total);
    }
    have |= (bit << pos);
  }
  return total;
}

template <typename BitChance, typename RAC, int bits>
void SimpleSymbolBitCoder<BitChance, RAC, bits>::write(bool bit,
                                                       SymbolChanceBitType typ,
                                                       int i) {
  BitChance& bch = ctx.bit(typ, i);
  rac.write_12bit_chance(bch.get_12bit(), bit);
  bch.put(bit);
}

template <typename BitChance, typename RAC, int bits>
void SimpleSymbolCoder<BitChance, RAC, bits>::write_int(int min, int max,
                                                        int value) {
  SimpleSymbolBitCoder<BitChance, RAC, bits> bitCoder(ctx, rac);
  writer<bits, SimpleSymbolBitCoder<BitChance, RAC, bits> >(bitCoder, min, max,
                                                            value);
}
template <typename BitChance, typename RAC, int bits>
void SimpleSymbolCoder<BitChance, RAC, bits>::write_int(int nbits, int value) {
  JXL_DASSERT(nbits <= bits);
  SimpleSymbolBitCoder<BitChance, RAC, bits> bitCoder(ctx, rac);
  writer(bitCoder, nbits, value);
}

}  // namespace jxl

#endif  // JXL_MODULAR_MA_SYMBOL_ENC_H_
