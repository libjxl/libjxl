// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Headers are not converted yet; only this file is checked.
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunsafe-buffer-usage"
#endif
#include "lib/jxl/dec_huffman.h"

#include <array>
#include <cstdint>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/huffman_table.h"
#ifdef __clang__
#pragma clang diagnostic pop
#endif

namespace jxl {

static const int kCodeLengthCodes = 18;
static const std::array<uint8_t, kCodeLengthCodes> kCodeLengthCodeOrder = {
    1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15,
};
static const uint8_t kDefaultCodeLength = 8;
static const uint8_t kCodeLengthRepeatCode = 16;

static bool ReadHuffmanCodeLengths(
    const std::array<uint8_t, kCodeLengthCodes>& code_length_code_lengths,
    Span<uint8_t> code_lengths, BitReader* br) {
  const int num_symbols = static_cast<int>(code_lengths.size());
  int symbol = 0;
  uint8_t prev_code_len = kDefaultCodeLength;
  int repeat = 0;
  uint8_t repeat_code_len = 0;
  int space = 32768;
  std::array<HuffmanCode, 32> table = {};

  std::array<uint16_t, 16> counts = {};
  for (int i = 0; i < kCodeLengthCodes; ++i) {
    ++counts[code_length_code_lengths[i]];
  }
  if (!BuildHuffmanTable(table.data(), 5, code_length_code_lengths.data(),
                         kCodeLengthCodes, counts.data())) {
    return false;
  }

  while (symbol < num_symbols && space > 0) {
    br->Refill();
    const HuffmanCode& code = table[br->PeekFixedBits<5>()];
    br->Consume(code.bits);
    uint8_t code_len = static_cast<uint8_t>(code.value);
    if (code_len < kCodeLengthRepeatCode) {
      repeat = 0;
      code_lengths[symbol++] = code_len;
      if (code_len != 0) {
        prev_code_len = code_len;
        space -= 32768u >> code_len;
      }
    } else {
      const int extra_bits = code_len - 14;
      int old_repeat;
      int repeat_delta;
      uint8_t new_len = 0;
      if (code_len == kCodeLengthRepeatCode) {
        new_len = prev_code_len;
      }
      if (repeat_code_len != new_len) {
        repeat = 0;
        repeat_code_len = new_len;
      }
      old_repeat = repeat;
      if (repeat > 0) {
        repeat -= 2;
        repeat <<= extra_bits;
      }
      repeat += static_cast<int>(br->ReadBits(extra_bits) + 3);
      repeat_delta = repeat - old_repeat;
      if (symbol + repeat_delta > num_symbols) {
        return false;
      }
      for (int i = 0; i < repeat_delta; ++i) {
        code_lengths[symbol + i] = repeat_code_len;
      }
      symbol += repeat_delta;
      if (repeat_code_len != 0) {
        space -= repeat_delta << (15 - repeat_code_len);
      }
    }
  }
  if (space != 0) {
    return false;
  }
  for (int i = symbol; i < num_symbols; ++i) {
    code_lengths[i] = 0;
  }
  return true;
}

static JXL_INLINE bool ReadSimpleCode(size_t alphabet_size, BitReader* br,
                                      Span<HuffmanCode> table) {
  size_t max_bits =
      (alphabet_size > 1u) ? FloorLog2Nonzero(alphabet_size - 1u) + 1 : 0;

  size_t num_symbols = br->ReadFixedBits<2>() + 1;

  std::array<uint16_t, 4> symbols = {};
  for (size_t i = 0; i < num_symbols; ++i) {
    uint16_t symbol = br->ReadBits(max_bits);
    if (symbol >= alphabet_size) {
      return false;
    }
    symbols[i] = symbol;
  }

  for (size_t i = 0; i < num_symbols - 1; ++i) {
    for (size_t j = i + 1; j < num_symbols; ++j) {
      if (symbols[i] == symbols[j]) return false;
    }
  }

  // 4 symbols have to option to encode.
  if (num_symbols == 4) num_symbols += br->ReadFixedBits<1>();

  const auto swap_symbols = [&symbols](size_t i, size_t j) {
    uint16_t t = symbols[j];
    symbols[j] = symbols[i];
    symbols[i] = t;
  };

  size_t table_size = 1;
  switch (num_symbols) {
    case 1:
      table[0] = {0, symbols[0]};
      break;
    case 2:
      if (symbols[0] > symbols[1]) swap_symbols(0, 1);
      table[0] = {1, symbols[0]};
      table[1] = {1, symbols[1]};
      table_size = 2;
      break;
    case 3:
      if (symbols[1] > symbols[2]) swap_symbols(1, 2);
      table[0] = {1, symbols[0]};
      table[2] = {1, symbols[0]};
      table[1] = {2, symbols[1]};
      table[3] = {2, symbols[2]};
      table_size = 4;
      break;
    case 4: {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
          if (symbols[i] > symbols[j]) swap_symbols(i, j);
        }
      }
      table[0] = {2, symbols[0]};
      table[2] = {2, symbols[1]};
      table[1] = {2, symbols[2]};
      table[3] = {2, symbols[3]};
      table_size = 4;
      break;
    }
    case 5: {
      if (symbols[2] > symbols[3]) swap_symbols(2, 3);
      table[0] = {1, symbols[0]};
      table[1] = {2, symbols[1]};
      table[2] = {1, symbols[0]};
      table[3] = {3, symbols[2]};
      table[4] = {1, symbols[0]};
      table[5] = {2, symbols[1]};
      table[6] = {1, symbols[0]};
      table[7] = {3, symbols[3]};
      table_size = 8;
      break;
    }
    default: {
      // Unreachable.
      return false;
    }
  }

  const size_t goal_size = 1u << kHuffmanTableBits;
  while (table_size != goal_size) {
    for (size_t i = 0; i < table_size; ++i) {
      table[table_size + i] = table[i];
    }
    table_size <<= 1;
  }

  return true;
}

bool HuffmanDecodingData::ReadFromBitStream(size_t alphabet_size,
                                            BitReader* br) {
  if (alphabet_size > (1 << PREFIX_MAX_BITS)) return false;

  /* simple_code_or_skip is used as follows:
     1 for simple code;
     0 for no skipping, 2 skips 2 code lengths, 3 skips 3 code lengths */
  uint32_t simple_code_or_skip = br->ReadFixedBits<2>();
  if (simple_code_or_skip == 1u) {
    table_.resize(1u << kHuffmanTableBits);
    return ReadSimpleCode(alphabet_size, br,
                          Span<HuffmanCode>(table_.data(), table_.size()));
  }

  std::vector<uint8_t> code_lengths(alphabet_size, 0);
  std::array<uint8_t, kCodeLengthCodes> code_length_code_lengths = {};
  int space = 32;
  int num_codes = 0;
  /* Static Huffman code for the code length code lengths */
  /* clang-format off */
  static const std::array<HuffmanCode, 16> huff = {{
      {2, 0}, {2, 4}, {2, 3}, {3, 2}, {2, 0}, {2, 4}, {2, 3}, {4, 1},
      {2, 0}, {2, 4}, {2, 3}, {3, 2}, {2, 0}, {2, 4}, {2, 3}, {4, 5},
  }};
  /* clang-format on */
  for (size_t i = simple_code_or_skip; i < kCodeLengthCodes && space > 0; ++i) {
    const int code_len_idx = kCodeLengthCodeOrder[i];
    br->Refill();
    const HuffmanCode& code = huff[br->PeekFixedBits<4>()];
    br->Consume(code.bits);
    uint8_t v = static_cast<uint8_t>(code.value);
    code_length_code_lengths[code_len_idx] = v;
    if (v != 0) {
      space -= (32u >> v);
      ++num_codes;
    }
  }
  bool ok = (num_codes == 1 || space == 0) &&
            ReadHuffmanCodeLengths(
                code_length_code_lengths,
                Span<uint8_t>(code_lengths.data(), code_lengths.size()), br);

  if (!ok) return false;
  std::array<uint16_t, 16> counts = {};
  for (size_t i = 0; i < alphabet_size; ++i) {
    ++counts[code_lengths[i]];
  }
  table_.resize(alphabet_size + 376);
  uint32_t table_size =
      BuildHuffmanTable(table_.data(), kHuffmanTableBits, code_lengths.data(),
                        alphabet_size, counts.data());
  table_.resize(table_size);
  return (table_size > 0);
}

}  // namespace jxl
