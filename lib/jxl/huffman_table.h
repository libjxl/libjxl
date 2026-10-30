// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_HUFFMAN_TABLE_H_
#define LIB_JXL_HUFFMAN_TABLE_H_

#include <stdint.h>
#include <stdlib.h>

#include "lib/jxl/base/span.h"

namespace jxl {

struct HuffmanCode {
  uint8_t bits;   /* number of bits used for this symbol */
  uint16_t value; /* symbol value or table offset */
};

/* Builds Huffman lookup table assuming code lengths are in symbol order. */
/* Returns 0 in case of error (invalid tree or memory error), otherwise
   populated size of table. The number of symbols is `code_lengths.size()`;
   `count` is the per-code-length histogram (so it must hold at least
   PREFIX_MAX_BITS + 1 entries) and is used as scratch space. */
uint32_t BuildHuffmanTable(Span<HuffmanCode> root_table, int root_bits,
                           Span<const uint8_t> code_lengths,
                           Span<uint16_t> count);

}  // namespace jxl

#endif  // LIB_JXL_HUFFMAN_TABLE_H_
