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

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_bit_writer.h"

namespace jxl {
namespace {

TEST(BitReaderTest, ExtendsWithZeroes) {
  for (size_t size = 4; size < 32; ++size) {
    std::vector<uint8_t> data(size, 0xff);

    for (size_t n_bytes = 0; n_bytes < size; n_bytes++) {
      BitReader br(Span<const uint8_t>(data.data(), n_bytes));
      // Read all the bits
      for (size_t i = 0; i < n_bytes * kBitsPerByte; i++) {
        ASSERT_EQ(br.ReadBits(1), 1) << "n_bytes=" << n_bytes << " i=" << i;
      }

      // PEEK more than the declared size - all will be zero. Cannot consume.
      for (size_t i = 0; i < BitReader::kMaxBitsPerCall; i++) {
        ASSERT_EQ(br.PeekBits(i), 0)
            << "size=" << size << "n_bytes=" << n_bytes << " i=" << i;
      }

      EXPECT_TRUE(br.Close());
    }
  }
}

struct Symbol {
  uint32_t num_bits;
  uint32_t value;
};

// Reading from output gives the same values.
TEST(BitReaderTest, TestRoundTrip) {
  ThreadPoolInternal pool(8);
  pool.Run(0, 1000, ThreadPool::SkipInit(),
           [](const int task, const int /* thread */) {
             constexpr size_t kMaxBits = 8000;
             BitWriter writer;
             BitWriter::Allotment allotment(&writer, kMaxBits);

             std::vector<Symbol> symbols;
             symbols.reserve(1000);

             std::mt19937 rng(55537 + 129 * task);
             std::uniform_int_distribution<> dist(1, 32);  // closed interval

             for (;;) {
               const uint32_t num_bits = dist(rng);
               if (writer.BitsWritten() + num_bits > kMaxBits) break;
               const uint32_t value = rng() >> (32 - num_bits);
               symbols.push_back({num_bits, value});
               writer.Write(num_bits, value);
             }

             writer.ZeroPadToByte();
             ReclaimAndCharge(&writer, &allotment, 0, nullptr);
             BitReader reader(writer.GetSpan());
             for (const Symbol& s : symbols) {
               EXPECT_EQ(s.value, reader.ReadBits(s.num_bits));
             }
             EXPECT_TRUE(reader.Close());
           });
}

// SkipBits is the same as reading that many bits.
TEST(BitReaderTest, TestSkip) {
  ThreadPoolInternal pool(8);
  pool.Run(
      0, 96, ThreadPool::SkipInit(),
      [](const int task, const int /* thread */) {
        constexpr size_t kSize = 100;

        for (size_t skip = 0; skip < 128; ++skip) {
          BitWriter writer;
          BitWriter::Allotment allotment(&writer, kSize * kBitsPerByte);
          // Start with "task" 1-bits.
          for (int i = 0; i < task; ++i) {
            writer.Write(1, 1);
          }

          // Write 0-bits that we will skip over
          for (size_t i = 0; i < skip; ++i) {
            writer.Write(1, 0);
          }

          // Write terminator bits '101'
          writer.Write(3, 5);
          EXPECT_EQ(task + skip + 3, writer.BitsWritten());
          writer.ZeroPadToByte();
          AuxOut aux_out;
          ReclaimAndCharge(&writer, &allotment, 0, &aux_out);
          EXPECT_LT(aux_out.layers[0].total_bits, kSize * 8);

          BitReader reader1(writer.GetSpan());
          BitReader reader2(writer.GetSpan());
          // Verify initial 1-bits
          for (int i = 0; i < task; ++i) {
            EXPECT_EQ(1, reader1.ReadBits(1));
            EXPECT_EQ(1, reader2.ReadBits(1));
          }

          // SkipBits or manually read "skip" bits
          reader1.SkipBits(skip);
          for (size_t i = 0; i < skip; ++i) {
            EXPECT_EQ(0, reader2.ReadBits(1)) << " skip=" << skip << " i=" << i;
          }
          EXPECT_EQ(reader1.TotalBitsConsumed(), reader2.TotalBitsConsumed());

          // Ensure both readers see the terminator bits.
          EXPECT_EQ(5, reader1.ReadBits(3));
          EXPECT_EQ(5, reader2.ReadBits(3));

          EXPECT_TRUE(reader1.Close());
          EXPECT_TRUE(reader2.Close());
        }
      });
}

// Verifies byte order and different groupings of bits.
TEST(BitReaderTest, TestOrder) {
  constexpr size_t kMaxBits = 16;

  // u(1) - bits written into LSBs of first byte
  {
    BitWriter writer;
    BitWriter::Allotment allotment(&writer, kMaxBits);
    for (size_t i = 0; i < 5; ++i) {
      writer.Write(1, 1);
    }
    for (size_t i = 0; i < 5; ++i) {
      writer.Write(1, 0);
    }
    for (size_t i = 0; i < 6; ++i) {
      writer.Write(1, 1);
    }

    writer.ZeroPadToByte();
    ReclaimAndCharge(&writer, &allotment, 0, nullptr);
    BitReader reader(writer.GetSpan());
    EXPECT_EQ(0x1F, reader.ReadFixedBits<8>());
    EXPECT_EQ(0xFC, reader.ReadFixedBits<8>());
    EXPECT_TRUE(reader.Close());
  }

  // u(8) - get bytes in the same order
  {
    BitWriter writer;
    BitWriter::Allotment allotment(&writer, kMaxBits);
    writer.Write(8, 0xF8);
    writer.Write(8, 0x3F);

    writer.ZeroPadToByte();
    ReclaimAndCharge(&writer, &allotment, 0, nullptr);
    BitReader reader(writer.GetSpan());
    EXPECT_EQ(0xF8, reader.ReadFixedBits<8>());
    EXPECT_EQ(0x3F, reader.ReadFixedBits<8>());
    EXPECT_TRUE(reader.Close());
  }

  // u(16) - little-endian bytes
  {
    BitWriter writer;
    BitWriter::Allotment allotment(&writer, kMaxBits);
    writer.Write(16, 0xF83F);

    writer.ZeroPadToByte();
    ReclaimAndCharge(&writer, &allotment, 0, nullptr);
    BitReader reader(writer.GetSpan());
    EXPECT_EQ(0x3F, reader.ReadFixedBits<8>());
    EXPECT_EQ(0xF8, reader.ReadFixedBits<8>());
    EXPECT_TRUE(reader.Close());
  }

  // Non-byte-aligned, mixed sizes
  {
    BitWriter writer;
    BitWriter::Allotment allotment(&writer, kMaxBits);
    writer.Write(1, 1);
    writer.Write(3, 6);
    writer.Write(8, 0xDB);
    writer.Write(4, 8);

    writer.ZeroPadToByte();
    ReclaimAndCharge(&writer, &allotment, 0, nullptr);
    BitReader reader(writer.GetSpan());
    EXPECT_EQ(0xBD, reader.ReadFixedBits<8>());
    EXPECT_EQ(0x8D, reader.ReadFixedBits<8>());
    EXPECT_TRUE(reader.Close());
  }
}

TEST(BitReaderTest, TotalCountersTest) {
  uint8_t buf[8] = {1, 2, 3, 4};
  BitReader reader(Span<const uint8_t>(buf, sizeof(buf)));

  EXPECT_EQ(sizeof(buf), reader.TotalBytes());
  EXPECT_EQ(0, reader.TotalBitsConsumed());
  reader.ReadFixedBits<1>();
  EXPECT_EQ(1, reader.TotalBitsConsumed());

  reader.ReadFixedBits<10>();
  EXPECT_EQ(11, reader.TotalBitsConsumed());

  reader.ReadFixedBits<4>();
  EXPECT_EQ(15, reader.TotalBitsConsumed());

  reader.ReadFixedBits<1>();
  EXPECT_EQ(16, reader.TotalBitsConsumed());

  reader.ReadFixedBits<16>();
  EXPECT_EQ(32, reader.TotalBitsConsumed());

  EXPECT_TRUE(reader.Close());
}

TEST(BitReaderTest, MoveTest) {
  uint8_t buf[8] = {1, 2, 3, 4};
  BitReader reader2;
  {
    BitReader reader1(Span<const uint8_t>(buf, sizeof(buf)));

    EXPECT_EQ(0, reader1.TotalBitsConsumed());
    reader1.ReadFixedBits<16>();
    EXPECT_EQ(16, reader1.TotalBitsConsumed());

    reader2 = std::move(reader1);
    // From this point reader1 is invalid, but can continue to access reader2
    // and we don't need to call Close() on reader1.
  }

  EXPECT_EQ(16, reader2.TotalBitsConsumed());
  EXPECT_EQ(3U, reader2.ReadFixedBits<8>());
  EXPECT_EQ(24, reader2.TotalBitsConsumed());

  EXPECT_TRUE(reader2.Close());
}

TEST(BitReaderTest, SuspendResumeTest) {
  uint8_t buf[12];
  for (size_t i = 0; i < sizeof(buf); i++) {
    buf[i] = 0x11 * (i + 3);  // values are 0x33, 0x44, 0x55, ...
  }

  {
    // Suspend before starting, no bytes read.
    BitReader reader(Span<const uint8_t>(buf, sizeof(buf)));
    EXPECT_EQ(12u, reader.Suspend());
    EXPECT_TRUE(reader.Close());
  }

  {
    // Suspend with a few bits of the first byte, this should keep those
    // remaining bits from the first byte available in the bit reader.
    BitReader reader(Span<const uint8_t>(buf, sizeof(buf)));
    reader.ReadBits(12);
    EXPECT_EQ(10, reader.Suspend());
    EXPECT_EQ(12, reader.TotalBitsConsumed());

    const std::vector<uint8_t> other_buf = {0x11, 0x22};
    reader.Resume(other_buf);
    // The TotalBitsConsumed() should not change.
    EXPECT_EQ(12, reader.TotalBitsConsumed());
    // This is reading one nibble from the 0x44 value and the first byte of the
    // second buffer (0x11).
    EXPECT_EQ(0x0114, reader.ReadBits(12));
    EXPECT_TRUE(reader.Close());
  }

  {
    // Suspend and resume near the end of the input buffer to exercise the
    // overread_bytes_ workflow.
    BitReader reader(Span<const uint8_t>(buf, sizeof(buf)));
    for (size_t i = 0; i < sizeof(buf) - 2; i++) {
      reader.ReadBits(8);
    }

    reader.ReadBits(4);
    // At this point we only have 12 bits remaining.
    EXPECT_EQ(sizeof(buf) * 8 - 12, reader.TotalBitsConsumed());
    // Only 1 byte of those 12 unused bits is returned from Suspend().
    EXPECT_EQ(1, reader.Suspend());
    EXPECT_EQ(sizeof(buf) * 8 - 12, reader.TotalBitsConsumed());

    const std::vector<uint8_t> other_buf = {0x11};
    reader.Resume(other_buf);
    // The TotalBitsConsumed() should not change.
    EXPECT_EQ(sizeof(buf) * 8 - 12, reader.TotalBitsConsumed());
    // This will read the last 4 bits (0xd) and the 0x11 in the new buffer.
    EXPECT_EQ(0x011d, reader.ReadBits(12));

    EXPECT_EQ(0, reader.Suspend());
    EXPECT_TRUE(reader.Close());
  }
}

TEST(BitReaderTest, ResumeFromEmptyReaderTest) {
  BitReader reader;
  uint8_t buf[8] = {1, 2, 3, 4};
  reader.Resume(Span<const uint8_t>(buf, sizeof(buf)));
  EXPECT_EQ(0x04030201, reader.ReadBits(32));
  EXPECT_EQ(32, reader.TotalBitsConsumed());
  EXPECT_TRUE(reader.Close());
}

TEST(BitReaderTest, CanReadWithinBoundsTest) {
  uint8_t buf[8] = {1, 2, 3, 4};
  BitReader reader(Span<const uint8_t>(buf, sizeof(buf)));

  EXPECT_TRUE(reader.CanReadWithinBounds(0));
  EXPECT_TRUE(reader.CanReadWithinBounds(1));
  EXPECT_TRUE(reader.CanReadWithinBounds(64));
  EXPECT_FALSE(reader.CanReadWithinBounds(65));

  reader.ReadBits(30);
  reader.ReadBits(30);
  // Only 4 bits remaining.
  EXPECT_TRUE(reader.CanReadWithinBounds(0));
  EXPECT_TRUE(reader.CanReadWithinBounds(4));
  EXPECT_FALSE(reader.CanReadWithinBounds(5));

  EXPECT_EQ(0, reader.Suspend());
  const std::vector<uint8_t> other_buf = {1, 2};
  reader.Resume(other_buf);
  // We should have 2 more bytes available now, so a total of 20.
  EXPECT_TRUE(reader.CanReadWithinBounds(20));
  EXPECT_FALSE(reader.CanReadWithinBounds(21));

  EXPECT_TRUE(reader.Close());
}

}  // namespace
}  // namespace jxl
