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

#include "jxl/toc.h"

#include <random>

#include "gtest/gtest.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/span.h"
#include "jxl/common.h"

namespace jxl {
namespace {

void Roundtrip(size_t num_entries, std::mt19937* rng) {
  // Generate num_entries groups of random (byte-aligned) length
  std::vector<BitWriter> group_codes(num_entries);
  for (BitWriter& writer : group_codes) {
    const size_t max_bits = (*rng)() & 0xFFF;
    BitWriter::Allotment allotment(&writer, max_bits + kBitsPerByte);
    size_t i = 0;
    for (; i + BitWriter::kMaxBitsPerCall < max_bits;
         i += BitWriter::kMaxBitsPerCall) {
      writer.Write(BitWriter::kMaxBitsPerCall, 0);
    }
    for (; i < max_bits; i += 1) {
      writer.Write(/*n_bits=*/1, 0);
    }
    writer.ZeroPadToByte();
    AuxOut aux_out;
    ReclaimAndCharge(&writer, &allotment, 0, &aux_out);
  }

  BitWriter writer;
  AuxOut aux_out;
  JXL_CHECK(WriteGroupOffsets(group_codes, &writer, &aux_out));

  BitReader reader(writer.GetSpan());
  std::vector<uint64_t> group_offsets;
  ASSERT_TRUE(ReadGroupOffsets(num_entries, &reader, &group_offsets));
  ASSERT_EQ(num_entries + 1, group_offsets.size());
  EXPECT_TRUE(reader.Close());

  EXPECT_EQ(0, group_offsets[0]);
  size_t prefix_sum = 0;
  for (size_t i = 1; i < num_entries + 1; ++i) {
    EXPECT_TRUE(group_codes[i - 1].BitsWritten() % kBitsPerByte == 0);
    prefix_sum += group_codes[i - 1].BitsWritten() / kBitsPerByte;
    EXPECT_EQ(prefix_sum, group_offsets[i]);
  }
}

TEST(TocTest, Test) {
  std::mt19937 rng(12345);
  for (size_t num_entries = 0; num_entries < 5; ++num_entries) {
    Roundtrip(num_entries, &rng);
  }
}

}  // namespace
}  // namespace jxl
