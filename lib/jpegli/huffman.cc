// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/huffman.h"

#include <limits>
#include <vector>

namespace jpegli {

// Returns the table width of the next 2nd level table, count is the histogram
// of bit lengths for the remaining symbols, len is the code length of the next
// processed symbol.
static inline int NextTableBitSize(const int* count, int len) {
  int left = 1 << (len - kJpegHuffmanRootTableBits);
  while (len < static_cast<int>(kJpegHuffmanMaxBitLength)) {
    left -= count[len];
    if (left <= 0) break;
    ++len;
    left <<= 1;
  }
  return len - kJpegHuffmanRootTableBits;
}

void BuildJpegHuffmanTable(const uint32_t* count, const uint32_t* symbols,
                           HuffmanTableEntry* lut) {
  HuffmanTableEntry code;    // current table entry
  HuffmanTableEntry* table;  // next available space in table
  int len;                   // current code length
  int idx;                   // symbol index
  int key;                   // prefix code
  int reps;                  // number of replicate key values in current table
  int low;                   // low bits for current root entry
  int table_bits;            // key length of current table
  int table_size;            // size of current table

  // Make a local copy of the input bit length histogram.
  int tmp_count[kJpegHuffmanMaxBitLength + 1] = {0};
  int total_count = 0;
  for (len = 1; len <= static_cast<int>(kJpegHuffmanMaxBitLength); ++len) {
    tmp_count[len] = count[len];
    total_count += tmp_count[len];
  }

  table = lut;
  table_bits = kJpegHuffmanRootTableBits;
  table_size = 1 << table_bits;

  // Special case code with only one value.
  if (total_count == 1) {
    code.bits = 0;
    code.value = symbols[0];
    for (key = 0; key < table_size; ++key) {
      table[key] = code;
    }
    return;
  }

  // Fill in root table.
  key = 0;
  idx = 0;
  for (len = 1; len <= kJpegHuffmanRootTableBits; ++len) {
    for (; tmp_count[len] > 0; --tmp_count[len]) {
      code.bits = len;
      code.value = symbols[idx++];
      reps = 1 << (kJpegHuffmanRootTableBits - len);
      while (reps--) {
        table[key++] = code;
      }
    }
  }

  // Fill in 2nd level tables and add pointers to root table.
  table += table_size;
  table_size = 0;
  low = 0;
  for (len = kJpegHuffmanRootTableBits + 1;
       len <= static_cast<int>(kJpegHuffmanMaxBitLength); ++len) {
    for (; tmp_count[len] > 0; --tmp_count[len]) {
      // Start a new sub-table if the previous one is full.
      if (low >= table_size) {
        table += table_size;
        table_bits = NextTableBitSize(tmp_count, len);
        table_size = 1 << table_bits;
        low = 0;
        lut[key].bits = table_bits + kJpegHuffmanRootTableBits;
        lut[key].value = (table - lut) - key;
        ++key;
      }
      code.bits = len - kJpegHuffmanRootTableBits;
      code.value = symbols[idx++];
      reps = 1 << (table_bits - code.bits);
      while (reps--) {
        table[low++] = code;
      }
    }
  }
}

// A node of a Huffman tree.
struct HuffmanTree {
  HuffmanTree(uint32_t count, int16_t left, int16_t right)
      : total_count(count), index_left(left), index_right_or_value(right) {}
  uint32_t total_count;
  int16_t index_left;
  int16_t index_right_or_value;
};

void SetDepth(const HuffmanTree& p, HuffmanTree* pool, uint8_t* depth,
              uint8_t level) {
  if (p.index_left >= 0) {
    ++level;
    SetDepth(pool[p.index_left], pool, depth, level);
    SetDepth(pool[p.index_right_or_value], pool, depth, level);
  } else {
    depth[p.index_right_or_value] = level;
  }
}

// Sort the root nodes, least popular first.
static JXL_INLINE bool Compare(const HuffmanTree& v0, const HuffmanTree& v1) {
  return v0.total_count < v1.total_count;
}

// This function will create a Huffman tree.
//
// The catch here is that the tree cannot be arbitrarily deep.
// Brotli specifies a maximum depth of 15 bits for "code trees"
// and 7 bits for "code length code trees."
//
// count_limit is the value that is to be faked as the minimum value
// and this minimum value is raised until the tree matches the
// maximum length requirement.
//
// This algorithm is not of excellent performance for very long data blocks,
// especially when population counts are longer than 2**tree_limit, but
// we are not planning to use this with extremely long blocks.
//
// See http://en.wikipedia.org/wiki/Huffman_coding
void CreateHuffmanTree(const uint32_t* data, const size_t length,
                       const int tree_limit, uint8_t* depth) {
  // For block sizes below 64 kB, we never need to do a second iteration
  // of this loop. Probably all of our block sizes will be smaller than
  // that, so this loop is mostly of academic interest. If we actually
  // would need this, we would be better off with the Katajainen algorithm.
  for (uint32_t count_limit = 1;; count_limit *= 2) {
    std::vector<HuffmanTree> tree;
    tree.reserve(2 * length + 1);

    for (size_t i = length; i != 0;) {
      --i;
      if (data[i]) {
        const uint32_t count = std::max(data[i], count_limit - 1);
        tree.emplace_back(count, -1, static_cast<int16_t>(i));
      }
    }

    const size_t n = tree.size();
    if (n == 1) {
      // Fake value; will be fixed on upper level.
      depth[tree[0].index_right_or_value] = 1;
      break;
    }

    std::stable_sort(tree.begin(), tree.end(), Compare);

    // The nodes are:
    // [0, n): the sorted leaf nodes that we start with.
    // [n]: we add a sentinel here.
    // [n + 1, 2n): new parent nodes are added here, starting from
    //              (n+1). These are naturally in ascending order.
    // [2n]: we add a sentinel at the end as well.
    // There will be (2n+1) elements at the end.
    const HuffmanTree sentinel(std::numeric_limits<uint32_t>::max(), -1, -1);
    tree.push_back(sentinel);
    tree.push_back(sentinel);

    size_t i = 0;      // Points to the next leaf node.
    size_t j = n + 1;  // Points to the next non-leaf node.
    for (size_t k = n - 1; k != 0; --k) {
      size_t left, right;
      if (tree[i].total_count <= tree[j].total_count) {
        left = i;
        ++i;
      } else {
        left = j;
        ++j;
      }
      if (tree[i].total_count <= tree[j].total_count) {
        right = i;
        ++i;
      } else {
        right = j;
        ++j;
      }

      // The sentinel node becomes the parent node.
      size_t j_end = tree.size() - 1;
      tree[j_end].total_count =
          tree[left].total_count + tree[right].total_count;
      tree[j_end].index_left = static_cast<int16_t>(left);
      tree[j_end].index_right_or_value = static_cast<int16_t>(right);

      // Add back the last sentinel node.
      tree.push_back(sentinel);
    }
    JXL_DASSERT(tree.size() == 2 * n + 1);
    SetDepth(tree[2 * n - 1], &tree[0], depth, 0);

    // We need to pack the Huffman tree in tree_limit bits.
    // If this was not successful, add fake entities to the lowest values
    // and retry.
    if (*std::max_element(&depth[0], &depth[length]) <= tree_limit) {
      break;
    }
  }
}

}  // namespace jpegli
