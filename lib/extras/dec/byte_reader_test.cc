// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/byte_reader.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace jxl {
namespace extras {
namespace {

TEST(ByteReaderTest, ReadsAndTracksOffset) {
  const uint8_t data[] = {0x10, 0x20, 0x30};
  ByteReader reader(Span<const uint8_t>(data, sizeof(data)));

  EXPECT_EQ(0u, reader.offset());
  EXPECT_EQ(3u, reader.remaining());
  uint8_t value = 0;
  ASSERT_TRUE(reader.ReadU8(&value));
  EXPECT_EQ(0x10, value);
  EXPECT_EQ(1u, reader.offset());
  EXPECT_EQ(2u, reader.remaining());
}

TEST(ByteReaderTest, PeekDoesNotAdvance) {
  const uint8_t data[] = {0xaa, 0xbb};
  ByteReader reader(Span<const uint8_t>(data, sizeof(data)));

  uint8_t value = 0;
  ASSERT_TRUE(reader.Peek(&value));
  EXPECT_EQ(0xaa, value);
  EXPECT_EQ(0u, reader.offset());
}

TEST(ByteReaderTest, ReadReturnsBoundedSpan) {
  const uint8_t data[] = {1, 2, 3, 4};
  ByteReader reader(Span<const uint8_t>(data, sizeof(data)));

  Span<const uint8_t> out;
  ASSERT_TRUE(reader.Read(2, &out));
  ASSERT_EQ(2u, out.size());
  EXPECT_EQ(1, out[0]);
  EXPECT_EQ(2, out[1]);
  EXPECT_EQ(2u, reader.offset());
}

TEST(ByteReaderTest, OversizedReadFailsWithoutAdvancing) {
  const uint8_t data[] = {1, 2, 3, 4};
  ByteReader reader(Span<const uint8_t>(data, sizeof(data)));

  ASSERT_TRUE(reader.Skip(1));
  Span<const uint8_t> out;
  EXPECT_FALSE(reader.Read(4, &out));
  EXPECT_EQ(1u, reader.offset());
  EXPECT_EQ(3u, reader.remaining());
}

TEST(ByteReaderTest, FailedReadPreservesOutputAndCursor) {
  const uint8_t data[] = {1, 2, 3, 4};
  const uint8_t sentinel[] = {0xaa, 0xbb};
  ByteReader reader(Span<const uint8_t>(data, sizeof(data)));

  ASSERT_TRUE(reader.Skip(1));
  Span<const uint8_t> out(sentinel, sizeof(sentinel));
  EXPECT_FALSE(reader.Read(4, &out));
  ASSERT_EQ(sizeof(sentinel), out.size());
  EXPECT_EQ(&sentinel[0], out.data());
  EXPECT_EQ(0xaa, out[0]);
  EXPECT_EQ(0xbb, out[1]);
  EXPECT_EQ(1u, reader.offset());
}

TEST(ByteReaderTest, ZeroLengthReadSucceedsWithoutAdvancing) {
  const uint8_t data[] = {1};
  ByteReader reader(Span<const uint8_t>(data, sizeof(data)));

  Span<const uint8_t> out;
  ASSERT_TRUE(reader.Read(0, &out));
  EXPECT_EQ(0u, out.size());
  EXPECT_EQ(0u, reader.offset());

  ASSERT_TRUE(reader.Skip(1));
  ASSERT_TRUE(reader.Read(0, &out));
  EXPECT_EQ(0u, out.size());
  EXPECT_EQ(1u, reader.offset());
}

TEST(ByteReaderTest, SkipAndEnd) {
  const uint8_t data[] = {1, 2};
  ByteReader reader(Span<const uint8_t>(data, sizeof(data)));

  ASSERT_TRUE(reader.Skip(2));
  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(0u, reader.remaining());
  EXPECT_FALSE(reader.Skip(1));
  EXPECT_EQ(2u, reader.offset());
}

TEST(ByteReaderTest, FailedReadLeavesOutputUntouched) {
  const uint8_t data[] = {1};
  ByteReader reader(Span<const uint8_t>(data, sizeof(data)));

  uint8_t value = 0x7f;
  ASSERT_TRUE(reader.Skip(1));
  EXPECT_FALSE(reader.ReadU8(&value));
  EXPECT_EQ(0x7f, value);
}

TEST(ByteReaderTest, EmptyReader) {
  ByteReader reader{Span<const uint8_t>()};

  EXPECT_TRUE(reader.AtEnd());
  EXPECT_EQ(0u, reader.remaining());
  uint8_t value = 0x55;
  EXPECT_FALSE(reader.Peek(&value));
  EXPECT_EQ(0x55, value);
}

}  // namespace
}  // namespace extras
}  // namespace jxl
