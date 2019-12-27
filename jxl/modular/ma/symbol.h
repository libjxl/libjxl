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

#ifndef JXl_MODULAR_MA_SYMBOL_H_
#define JXl_MODULAR_MA_SYMBOL_H_

#include <vector>

#include "jxl/base/status.h"
#include "jxl/modular/config.h"
#include "jxl/modular/ma/chance.h"
#include "jxl/modular/ma/util.h"

namespace jxl {

template <typename RAC>
class UniformSymbolCoder {
 private:
  RAC &rac;

 public:
  explicit UniformSymbolCoder(RAC &racIn) : rac(racIn) {}

#ifdef HAS_ENCODER
  void write_int(int min, int max, int val);
  void write_int(int bits, int val) { write_int(0, (1 << bits) - 1, val); }
#endif
  int read_int(int min, int len) {
    JXL_DASSERT(len >= 0);
    if (len == 0) return min;

    // split in [0..med] [med+1..len]
    int med = len / 2;
    bool bit = rac.read_bit();
    if (bit) {
      return read_int(min + med + 1, len - (med + 1));
    } else {
      return read_int(min, med);
    }
  }
  int read_int(int bits) { return read_int(0, (1 << bits) - 1); }
};

typedef enum {
  BIT_ZERO,
  BIT_SIGN,
  BIT_EXP,
  BIT_MANT,
} SymbolChanceBitType;

static const uint16_t ZERO_CHANCE = 1024;
static const uint16_t SIGN_CHANCE = 2048;
static const uint16_t MANT_CHANCE = 1024;

template <typename BitChance, int bits>
class SymbolChance {
 public:
  BitChance bit_zero;
  BitChance bit_sign;
  BitChance bit_exp[bits - 1];
  BitChance bit_mant[bits];

 public:
  BitChance inline &bitZero() { return bit_zero; }

  BitChance inline &bitSign() { return bit_sign; }

  // all exp bits 0         -> int(log2(val)) == 0  [ val == 1 ]
  // exp bits up to i are 1 -> int(log2(val)) == i+1
  BitChance inline &bitExp(unsigned int i) {
    JXL_DASSERT(i >= 0 && i < (bits - 1));
    return bit_exp[i];
  }
  BitChance inline &bitMant(unsigned int i) {
    JXL_DASSERT(i >= 0 && i < bits);
    return bit_mant[i];
  }

  BitChance inline &bit(SymbolChanceBitType typ, unsigned int i = 0) {
    switch (typ) {
      default:
      case BIT_ZERO:
        return bitZero();
      case BIT_SIGN:
        return bitSign();
      case BIT_EXP:
        return bitExp(i);
      case BIT_MANT:
        return bitMant(i);
    }
  }
  SymbolChance() = default;  // don't init  (needed for fast copy constructor in
                             // CompoundSymbolChances?)

  explicit SymbolChance(uint16_t zero_chance) {
    bitZero().set_12bit(zero_chance);
    //        bitSign().set_12bit(0x800); // 50%, which is the default anyway
    // uint64_t p = zero_chance;       // implicit denominator of 0x1000
    uint64_t rp = 0x1000 - zero_chance;  // 1-p

    // assume geometric distribution: Pr(X=k) = (1-p)^k p
    // (p = zero_chance)
    // cumulative distribution function: Pr(X < k) = 1-(1-p)^k = 1-rp^k
    //                                   Pr(X >= k) = (1-p)^k = rp^k
    // bitExp(i) == true   iff   X >= 2^i && X < 2^(i+1)
    // bitExp(i) is only used if the lower bound holds
    // Pr(X >= 2^i && X < 2^(i+1) | X >= 2^i ) = (rp^2^i - rp^2^(i+1)) / rp^2^i
    // = 1 - rp^2^i

    for (int i = 0; i < bits - 1; i++) {
      if (rp < 0x100) rp = 0x100;
      if (rp > 0xf00) rp = 0xf00;
      bitExp(i).set_12bit(0x1000 - rp);
      rp = (rp * rp + 0x800) >> 12;  // square it
    }
    for (int i = 0; i < bits; i++) {
      bitMant(i).set_12bit(MANT_CHANCE);
    }
  }
};

template <typename SymbolCoder>
int reader(SymbolCoder &coder, int bits) {
  int pos = 0;
  int value = 0;
  int b = 1;
  while (pos++ < bits) {
    if (coder.read(BIT_MANT, pos)) value += b;
    b *= 2;
  }
  return value;
}

template <int bits, typename SymbolCoder>
int reader(SymbolCoder &coder, int min, int max) ATTRIBUTE_HOT;

template <int bits, typename SymbolCoder>
int reader(SymbolCoder &coder, int min, int max) {
  JXL_DASSERT(min < max);  // make sure not to call this if min==max can happen
  bool sign;
  JXL_DASSERT(min <= 0 && max >= 0);  // should always be the case, because
                                      // guess should always be in valid range

  if (coder.read(BIT_ZERO)) return 0;
  if (min < 0) {
    if (max > 0) {
      sign = coder.read(BIT_SIGN);
    } else {
      sign = false;
    }
  } else {
    sign = true;
  }

  const unsigned int amax = (sign ? max : -min);

  const unsigned int emax = sizeof(unsigned int) * 8 - 1 - __builtin_clz(amax);

  unsigned int e = 0;
  for (; e < emax; e++) {
    if (coder.read(BIT_EXP, e)) break;
  }

  int have = (1 << e);

  for (unsigned int pos = e; pos > 0;) {
    pos--;
    int minabs1 = have | (1 << pos);
    if (minabs1 > static_cast<int>(amax)) continue;  // 1-bit is impossible
    if (coder.read(BIT_MANT, pos)) have = minabs1;
  }
  return (sign ? have : -have);
}

template <typename BitChance, typename RAC, int bits>
class SimpleSymbolBitCoder {
 private:
  SymbolChance<BitChance, bits> &ctx;
  RAC &rac;

 public:
  SimpleSymbolBitCoder(SymbolChance<BitChance, bits> &ctxIn, RAC &racIn)
      : ctx(ctxIn), rac(racIn) {}

#ifdef HAS_ENCODER
  void write(bool bit, SymbolChanceBitType typ, int i = 0);
#endif

  bool read(SymbolChanceBitType typ, unsigned int i = 0) {
    BitChance &bch = ctx.bit(typ, i);
    const bool bit = rac.read_12bit_chance(bch.get_12bit());
    bch.put(bit);
    return bit;
  }
};

template <typename BitChance, typename RAC, int bits>
class SimpleSymbolCoder {
 private:
  SymbolChance<BitChance, bits> ctx;
  RAC &rac;

 public:
  explicit SimpleSymbolCoder(RAC &racIn) : ctx(ZERO_CHANCE), rac(racIn) {}

#ifdef HAS_ENCODER
  void write_int(int min, int max, int value);
  void write_int2(int min, int max, int value) {
    if (min == max) return;
    if (min > 0)
      write_int(0, max - min, value - min);
    else if (max < 0)
      write_int(min - max, 0, value - max);
    else
      write_int(min, max, value);
  }
  void write_int(int nbits, int value);
#endif

  int read_int(int min, int max) {
    SimpleSymbolBitCoder<BitChance, RAC, bits> bitCoder(ctx, rac);
    return reader<bits, SimpleSymbolBitCoder<BitChance, RAC, bits>>(bitCoder,
                                                                    min, max);
  }
  int read_int2(int min, int max) {
    if (min == max) return min;
    if (min > 0)
      return read_int(0, max - min) + min;
    else if (max < 0)
      return read_int(min - max, 0) + max;
    else
      return read_int(min, max);
  }
  int read_int(int nbits) {
    JXL_DASSERT(nbits <= bits);
    SimpleSymbolBitCoder<BitChance, RAC, bits> bitCoder(ctx, rac);
    return reader(bitCoder, nbits);
  }
};

}  // namespace jxl

#ifdef HAS_ENCODER
#include "jxl/modular/ma/symbol_enc.h"
#endif

#endif  // JXl_MODULAR_MA_SYMBOL_H_
