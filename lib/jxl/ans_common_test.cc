// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/ans_common.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

void VerifyAliasDistribution(const std::vector<int32_t>& distribution,
                             uint32_t log_range) {
  constexpr size_t log_alpha_size = 8;
  std::vector<AliasTable::Entry> table(1 << log_alpha_size);
  Status ok =
      InitAliasTable(distribution, log_range, log_alpha_size, table.data());
  ASSERT_TRUE(ok);
  uint32_t range = 1 << log_range;
  std::vector<std::vector<uint32_t>> offsets(distribution.size());
  for (uint32_t i = 0; i < range; i++) {
    AliasTable::Symbol s =
        AliasTable::Lookup(table.data(), i, ANS_LOG_TAB_SIZE - 8,
                           (1 << (ANS_LOG_TAB_SIZE - 8)) - 1);
    offsets[s.value].push_back(s.offset);
  }
  for (uint32_t i = 0; i < distribution.size(); i++) {
    ASSERT_EQ(static_cast<size_t>(distribution[i]), offsets[i].size());
    std::sort(offsets[i].begin(), offsets[i].end());
    for (uint32_t j = 0; j < offsets[i].size(); j++) {
      ASSERT_EQ(offsets[i][j], j);
    }
  }
}

TEST(ANSCommonTest, LookupDecodesPackedEntry) {
  // Lookup() reads a packed uint64_t out of Entry; verify every decoded
  // field maps to the right member (intra-object read must start at the
  // entry base, not a member address).
  AliasTable::Entry e = {};
  e.cutoff = 0x12;
  e.right_value = 0x34;
  e.freq0 = 0x5678;
  e.offsets1 = 0x0100;
  e.freq1_xor_freq0 = 0x9ABC;
  AliasTable::Entry table[1] = {e};
  const size_t log_entry_size = 16;
  const size_t entry_size_minus_1 = (size_t(1) << log_entry_size) - 1;
  // pos below cutoff -> symbol i, freq0
  {
    AliasTable::Symbol s =
        AliasTable::Lookup(table, /*value=*/0x05, log_entry_size,
                           entry_size_minus_1);
    EXPECT_EQ(size_t(0), s.value);
    EXPECT_EQ(size_t(0x05), s.offset);
    EXPECT_EQ(size_t(0x5678), s.freq);
  }
  // pos at/above cutoff -> right_value, offsets1 + pos, freq1
  {
    AliasTable::Symbol s =
        AliasTable::Lookup(table, /*value=*/0x20, log_entry_size,
                           entry_size_minus_1);
    EXPECT_EQ(size_t(0x34), s.value);
    EXPECT_EQ(size_t(0x0100 + 0x20), s.offset);
    EXPECT_EQ(size_t(0x9ABC ^ 0x5678), s.freq);
  }
}

TEST(ANSCommonTest, AliasDistributionSmoke) {
  VerifyAliasDistribution({ANS_TAB_SIZE / 2, ANS_TAB_SIZE / 2},
                          ANS_LOG_TAB_SIZE);
  VerifyAliasDistribution({ANS_TAB_SIZE}, ANS_LOG_TAB_SIZE);
  VerifyAliasDistribution({0, 0, 0, ANS_TAB_SIZE, 0}, ANS_LOG_TAB_SIZE);
}

}  // namespace
}  // namespace jxl
