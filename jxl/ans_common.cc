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

#include "jxl/ans_common.h"

#include <numeric>

#include "jxl/ans_params.h"
#include "jxl/base/status.h"

namespace jxl {

std::vector<int> CreateFlatHistogram(int length, int total_count) {
  JXL_ASSERT(length > 0);
  JXL_ASSERT(length <= total_count);
  const int count = total_count / length;
  std::vector<int> result(length, count);
  const int rem_counts = total_count % length;
  for (int i = 0; i < rem_counts; ++i) {
    ++result[i];
  }
  return result;
}

// First, all trailing non-occuring symbols are removed from the distribution;
// if this leaves the distribution empty, a dummy symbol with max weight is
// added. This ensures that the resulting distribution sums to total table size.
// Then, `entry_size` is chosen to be the largest power of two so that
// `table_size` = ANS_TAB_SIZE/`entry_size` is at least as big as the
// distribution size.
// Note that each entry will only ever contain two different symbols, and
// consecutive ranges of offsets, which allows us to use a compact
// representation.
// Each entry is initialized with only the (symbol=i, offset) pairs; then
// positions for which the entry overflows (i.e. distribution[i] > entry_size)
// or is not full are computed, and put into a stack in increasing order.
// Missing symbols in the distribution are padded with 0 (because `table_size`
// >= number of symbols). The `cutoff` value for each entry is initialized to
// the number of occupied slots in that entry (i.e. `distributions[i]`). While
// the overflowing-symbol stack is not empty (which implies that the
// underflowing-symbol stack also is not), the top overfull and underfull
// positions are popped from the stack; the empty slots in the underfull entry
// are then filled with as many slots as needed from the overfull entry; such
// slots are placed after the slots in the overfull entry, and `offsets[1]` is
// computed accordingly. The formerly underfull entry is thus now neither
// underfull nor overfull, and represents exactly two symbols. The overfull
// entry might be either overfull or underfull, and is pushed into the
// corresponding stack.
void InitAliasTable(std::vector<int> distribution, int range,
                    size_t log_alpha_size, AliasTable::Entry* JXL_RESTRICT a) {
  while (!distribution.empty() && distribution.back() == 0) {
    distribution.pop_back();
  }
  // Ensure that a valid table is always returned, even for an empty
  // alphabet. Otherwise, a specially-crafted stream might crash the
  // decoder.
  if (distribution.empty()) {
    distribution.emplace_back(range);
  }
  const int table_size = 1 << log_alpha_size;
#if JXL_ENABLE_ASSERT
  int sum = std::accumulate(distribution.begin(), distribution.end(), 0);
#endif  // JXL_ENABLE_ASSERT
  JXL_ASSERT(sum == range);
  // range must be a power of two
  JXL_ASSERT((range & (range - 1)) == 0);
  JXL_ASSERT(distribution.size() <= table_size);
  JXL_ASSERT(table_size <= range);
  const int entry_size = range >> log_alpha_size;  // this is exact
  std::vector<int> underfull_posn;
  std::vector<int> overfull_posn;
  int cutoffs[ANS_MAX_ALPHA_SIZE];
  // Initialize entries.
  for (size_t i = 0; i < distribution.size(); i++) {
    cutoffs[i] = distribution[i];
    if (cutoffs[i] > entry_size) {
      overfull_posn.push_back(i);
    } else if (cutoffs[i] < entry_size) {
      underfull_posn.push_back(i);
    }
  }
  for (int i = distribution.size(); i < table_size; i++) {
    cutoffs[i] = 0;
    underfull_posn.push_back(i);
  }
  // Reassign overflow/underflow values.
  while (!overfull_posn.empty()) {
    int overfull_i = overfull_posn.back();
    overfull_posn.pop_back();
    JXL_ASSERT(!underfull_posn.empty());
    int underfull_i = underfull_posn.back();
    underfull_posn.pop_back();
    int underfull_by = entry_size - cutoffs[underfull_i];
    cutoffs[overfull_i] -= underfull_by;
    // overfull positions have their original symbols
    a[underfull_i].right_value = overfull_i;
    a[underfull_i].offsets1 = cutoffs[overfull_i];
    // Slots in the right part of entry underfull_i were taken from the end
    // of the symbols in entry overfull_i.
    if (cutoffs[overfull_i] < entry_size) {
      underfull_posn.push_back(overfull_i);
    } else if (cutoffs[overfull_i] > entry_size) {
      overfull_posn.push_back(overfull_i);
    }
  }
  for (int i = 0; i < table_size; i++) {
    // cutoffs[i] is properly initialized but the clang-analyzer doesn't infer
    // it since it is partially initialized across two for-loops.
    // NOLINTNEXTLINE(clang-analyzer-core.UndefinedBinaryOperatorResult)
    if (cutoffs[i] == entry_size) {
      a[i].right_value = i;
      a[i].offsets1 = 0;
      a[i].cutoff = 0;
    } else {
      // Note that, if cutoff is not equal to entry_size,
      // a[i].offsets1 was initialized with (overfull cutoff) -
      // (entry_size - a[i].cutoff). Thus, subtracting
      // a[i].cutoff cannot make it negative.
      a[i].offsets1 -= cutoffs[i];
      a[i].cutoff = cutoffs[i];
    }
    const size_t freq0 = i < distribution.size() ? distribution[i] : 0;
    const size_t i1 = a[i].right_value;
    const size_t freq1 = i1 < distribution.size() ? distribution[i1] : 0;
    a[i].freq0 = static_cast<uint16_t>(freq0);
    a[i].freq1_xor_freq0 = static_cast<uint16_t>(freq1 ^ freq0);
  }
}

}  // namespace jxl
