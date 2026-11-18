// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/base/span.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

std::vector<uint8_t> Iota(size_t n) {
  std::vector<uint8_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(i);
  return v;
}

TEST(SpanTest, SubspanReturnsRequestedRange) {
  const std::vector<uint8_t> data = Iota(10);
  const Bytes all(data.data(), data.size());

  JXL_TEST_ASSIGN_OR_DIE(Bytes middle, all.subspan(2, 3));
  EXPECT_EQ(3u, middle.size());
  EXPECT_EQ(2u, middle[0]);
  EXPECT_EQ(4u, middle[2]);

  JXL_TEST_ASSIGN_OR_DIE(Bytes whole, all.subspan(0, 10));
  EXPECT_EQ(10u, whole.size());
  EXPECT_EQ(all.data(), whole.data());
}

TEST(SpanTest, SubspanSuffixRunsToEnd) {
  const std::vector<uint8_t> data = Iota(10);
  const Bytes all(data.data(), data.size());

  JXL_TEST_ASSIGN_OR_DIE(Bytes tail, all.subspan(7));
  EXPECT_EQ(3u, tail.size());
  EXPECT_EQ(7u, tail[0]);

  // A suffix starting exactly at the end is empty, not an error.
  JXL_TEST_ASSIGN_OR_DIE(Bytes end, all.subspan(10));
  EXPECT_TRUE(end.empty());
}

// An empty range at the one-past-the-end offset is well defined and must be
// accepted; only offsets *beyond* that are rejected.
TEST(SpanTest, SubspanAcceptsEmptyRangeAtEnd) {
  const std::vector<uint8_t> data = Iota(4);
  const Bytes all(data.data(), data.size());

  JXL_TEST_ASSIGN_OR_DIE(Bytes empty, all.subspan(4, 0));
  EXPECT_TRUE(empty.empty());
}

TEST(SpanTest, SubspanRejectsOutOfRange) {
  const std::vector<uint8_t> data = Iota(10);
  const Bytes all(data.data(), data.size());

  EXPECT_FALSE(all.subspan(11, 0).ok());  // offset past the end
  EXPECT_FALSE(all.subspan(11).ok());     // suffix past the end
  EXPECT_FALSE(all.subspan(0, 11).ok());  // count longer than the span
  EXPECT_FALSE(all.subspan(5, 6).ok());   // offset + count overruns
  EXPECT_FALSE(all.subspan(10, 1).ok());  // one byte past the end
}

// Callers commonly compute a count by subtracting a fixed header/trailer size,
// e.g. `chunk.subspan(8, chunk.size() - 12)`. When the span is shorter than
// that constant the subtraction wraps to a huge size_t. subspan must reject
// the wrapped value rather than construct a span over unrelated memory, so the
// bound is checked as `count <= len_ - offset` only after `offset <= len_`.
TEST(SpanTest, SubspanRejectsUnderflowedCount) {
  const std::vector<uint8_t> data = Iota(8);
  const Bytes chunk(data.data(), data.size());

  const size_t wrapped = chunk.size() - 12;  // wraps: 8 - 12
  EXPECT_GT(wrapped, chunk.size());
  EXPECT_FALSE(chunk.subspan(8, wrapped).ok());
}

TEST(SpanTest, SubspanOfEmptySpan) {
  const Bytes empty;
  EXPECT_EQ(0u, empty.size());

  JXL_TEST_ASSIGN_OR_DIE(Bytes same, empty.subspan(0, 0));
  EXPECT_TRUE(same.empty());

  EXPECT_FALSE(empty.subspan(1, 0).ok());
  EXPECT_FALSE(empty.subspan(0, 1).ok());
}

TEST(SpanTest, SubspanComposes) {
  const std::vector<uint8_t> data = Iota(16);
  const Bytes all(data.data(), data.size());

  JXL_TEST_ASSIGN_OR_DIE(Bytes body, all.subspan(4, 8));
  JXL_TEST_ASSIGN_OR_DIE(Bytes inner, body.subspan(2, 4));
  EXPECT_EQ(4u, inner.size());
  EXPECT_EQ(6u, inner[0]);

  // The narrowed span carries its own bound.
  EXPECT_FALSE(body.subspan(0, 9).ok());
}

}  // namespace
}  // namespace jxl
