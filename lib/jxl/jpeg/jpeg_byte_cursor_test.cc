// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/jpeg/jpeg_byte_cursor.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "lib/jxl/testing.h"

namespace jxl {
namespace jpeg {
namespace {

TEST(JpegByteCursorTest, EmptyInput) {
  JpegByteCursor cursor{Bytes()};
  uint8_t byte = 0xa5;
  uint16_t word = 0x1234;
  EXPECT_FALSE(cursor.U8(&byte));
  EXPECT_FALSE(cursor.U16BE(&word));
  EXPECT_FALSE(cursor.Skip(1));
  EXPECT_EQ(0xa5, byte);
  EXPECT_EQ(0x1234, word);
  EXPECT_EQ(0u, cursor.offset());
  EXPECT_EQ(0u, cursor.remaining());

  EXPECT_TRUE(cursor.Skip(0));
  Bytes empty;
  ASSERT_TRUE(cursor.Take(0, &empty));
  EXPECT_EQ(nullptr, empty.data());
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(0u, cursor.offset());
}

TEST(JpegByteCursorTest, ReadsConsecutiveBytesAndBigEndianWords) {
  const uint8_t data[] = {0x12, 0x34, 0xfe, 0xdc, 0x56};
  JpegByteCursor cursor{Bytes(data)};
  EXPECT_EQ(0u, cursor.offset());
  EXPECT_EQ(sizeof(data), cursor.remaining());

  uint16_t word = 0;
  ASSERT_TRUE(cursor.U16BE(&word));
  EXPECT_EQ(0x1234, word);
  EXPECT_EQ(2u, cursor.offset());
  EXPECT_EQ(3u, cursor.remaining());

  uint8_t byte = 0;
  ASSERT_TRUE(cursor.U8(&byte));
  EXPECT_EQ(0xfe, byte);
  ASSERT_TRUE(cursor.U16BE(&word));
  EXPECT_EQ(0xdc56, word);
  EXPECT_EQ(sizeof(data), cursor.offset());
  EXPECT_EQ(0u, cursor.remaining());

  EXPECT_FALSE(cursor.U8(&byte));
  EXPECT_FALSE(cursor.U16BE(&word));
  EXPECT_EQ(0xfe, byte);
  EXPECT_EQ(0xdc56, word);
  EXPECT_EQ(sizeof(data), cursor.offset());
}

TEST(JpegByteCursorTest, TruncatedWordDoesNotConsumeItsFirstByte) {
  const uint8_t data[] = {0xab};
  JpegByteCursor cursor{Bytes(data)};
  uint16_t word = 0xcdef;
  EXPECT_FALSE(cursor.U16BE(&word));
  EXPECT_EQ(0xcdef, word);
  EXPECT_EQ(0u, cursor.offset());
  EXPECT_EQ(1u, cursor.remaining());

  uint8_t byte = 0;
  ASSERT_TRUE(cursor.U8(&byte));
  EXPECT_EQ(0xab, byte);
  EXPECT_EQ(1u, cursor.offset());
  EXPECT_EQ(0u, cursor.remaining());
}

TEST(JpegByteCursorTest, SkipAllowsExactEndAndPreservesOffsetOnFailure) {
  const uint8_t data[] = {0, 1, 2, 3};
  JpegByteCursor cursor{Bytes(data)};
  ASSERT_TRUE(cursor.Skip(1));
  EXPECT_FALSE(cursor.Skip(4));
  EXPECT_EQ(1u, cursor.offset());
  EXPECT_EQ(3u, cursor.remaining());
  ASSERT_TRUE(cursor.Skip(3));
  EXPECT_EQ(sizeof(data), cursor.offset());
  EXPECT_EQ(0u, cursor.remaining());
  EXPECT_TRUE(cursor.Skip(0));
  EXPECT_FALSE(cursor.Skip(1));
  EXPECT_EQ(sizeof(data), cursor.offset());
}

TEST(JpegByteCursorTest, TakeReturnsViewsAndAdvancesPastThem) {
  const uint8_t data[] = {0x10, 0x20, 0x30, 0x40, 0x50};
  JpegByteCursor cursor{Bytes(data)};
  ASSERT_TRUE(cursor.Skip(1));
  Bytes payload;
  ASSERT_TRUE(cursor.Take(3, &payload));
  EXPECT_EQ(data + 1, payload.data());
  ASSERT_EQ(3u, payload.size());
  EXPECT_EQ(0x20, payload[0]);
  EXPECT_EQ(0x40, payload[2]);
  EXPECT_EQ(4u, cursor.offset());
  EXPECT_EQ(1u, cursor.remaining());

  uint8_t byte = 0;
  ASSERT_TRUE(cursor.U8(&byte));
  EXPECT_EQ(0x50, byte);
  EXPECT_EQ(0u, cursor.remaining());
}

TEST(JpegByteCursorTest, FailedTakeLeavesOutputAndOffsetUnchanged) {
  const uint8_t data[] = {1, 2, 3};
  const uint8_t sentinel[] = {0xa5};
  JpegByteCursor cursor{Bytes(data)};
  ASSERT_TRUE(cursor.Skip(1));
  Bytes payload(sentinel);
  EXPECT_FALSE(cursor.Take(3, &payload));
  EXPECT_EQ(sentinel, payload.data());
  EXPECT_EQ(sizeof(sentinel), payload.size());
  EXPECT_EQ(1u, cursor.offset());
  EXPECT_EQ(2u, cursor.remaining());

  ASSERT_TRUE(cursor.Take(2, &payload));
  EXPECT_EQ(data + 1, payload.data());
  EXPECT_EQ(2u, payload.size());
  EXPECT_EQ(sizeof(data), cursor.offset());
}

TEST(JpegByteCursorTest, ZeroLengthTakePreservesPositionAtBeginningAndEnd) {
  const uint8_t data[] = {1, 2};
  JpegByteCursor cursor{Bytes(data)};
  Bytes empty;
  ASSERT_TRUE(cursor.Take(0, &empty));
  EXPECT_EQ(data, empty.data());
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(0u, cursor.offset());
  EXPECT_EQ(sizeof(data), cursor.remaining());

  ASSERT_TRUE(cursor.Skip(sizeof(data)));
  ASSERT_TRUE(cursor.Take(0, &empty));
  EXPECT_EQ(data + sizeof(data), empty.data());
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(sizeof(data), cursor.offset());
  EXPECT_EQ(0u, cursor.remaining());
}

TEST(JpegByteCursorTest, RejectsCountsThatWouldWrapAnOffsetAddition) {
  const uint8_t data[] = {1, 2, 3};
  const uint8_t sentinel[] = {0xa5};
  JpegByteCursor cursor{Bytes(data)};
  ASSERT_TRUE(cursor.Skip(1));
  const size_t huge = std::numeric_limits<size_t>::max();
  EXPECT_FALSE(cursor.Skip(huge));
  Bytes payload(sentinel);
  EXPECT_FALSE(cursor.Take(huge, &payload));
  EXPECT_EQ(sentinel, payload.data());
  EXPECT_EQ(sizeof(sentinel), payload.size());
  EXPECT_EQ(1u, cursor.offset());
  EXPECT_EQ(2u, cursor.remaining());

  uint16_t word = 0;
  ASSERT_TRUE(cursor.U16BE(&word));
  EXPECT_EQ(0x0203, word);
}

}  // namespace
}  // namespace jpeg
}  // namespace jxl
