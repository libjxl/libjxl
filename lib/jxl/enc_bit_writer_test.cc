// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_bit_writer.h"

#include <jxl/memory_manager.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

struct BitPatch {
  size_t len;
  uint64_t bits;
};

TEST(BitWriterTest, RandomSequence) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();

  auto mt = jxl::make_unique<std::mt19937>(42);
  std::uniform_int_distribution<> num_bits_dist(1, BitWriter::kMaxBitsPerCall);
  constexpr size_t kNumSequences = 1024 * 1024;
  std::vector<BitPatch> content;
  content.reserve(kNumSequences);
  size_t total_bits = 0;
  for (size_t i = 0; i < kNumSequences; ++i) {
    size_t len = num_bits_dist(*mt);
    uint64_t mask = (static_cast<uint64_t>(1) << len) - 1;
    uint64_t bits = (*mt)() & mask;
    content.emplace_back(BitPatch{len, bits});
    total_bits += len;
  }

  BitWriter writer{memory_manager};
  auto write_content = [&content, &writer]() -> Status {
    for (auto& patch : content) {
      writer.Write(patch.len, patch.bits);
    }
    writer.ZeroPadToByte();
    return true;
  };
  EXPECT_TRUE(writer.WithMaxBits(RoundUpBitsToByteMultiple(total_bits),
                                 LayerType::Header, nullptr, write_content));

  size_t num_mismatches = 0;
  BitReader reader(writer.GetSpan());
  for (auto& patch : content) {
    uint64_t bits = reader.ReadBits(patch.len);
    uint64_t expected_bits = patch.bits;
    if (bits != expected_bits) num_mismatches++;
  }
  EXPECT_TRUE(reader.JumpToByteBoundary());
  EXPECT_TRUE(reader.Close());
  EXPECT_EQ(num_mismatches, 0u);
}

}  // namespace
}  // namespace jxl
