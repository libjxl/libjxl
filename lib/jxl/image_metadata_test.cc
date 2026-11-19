// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/image_metadata.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_fields.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

// Writes an ImageMetadata bundle that declares `n_extra` extra channels.
//
// The bitstream is emitted field-by-field rather than through Bundle::Write so
// that the test exercises the READ path: the point of the cap is that a file
// declaring more than the encoder limit is rejected on parse, before any
// per-channel allocation. A Bundle::Write round-trip would fail on the write
// side first and would not prove the decoder is protected.
void WriteMetadataWithExtraChannels(BitWriter* writer, uint32_t n_extra) {
  writer->Write(1, 0);  // all_default = false
  writer->Write(1, 0);  // extra_fields = false
  // BitDepth: integer samples, bits_per_sample = 8 (U32 selector 0 = Val(8))
  writer->Write(1, 0);  // floating_point_sample = false
  writer->Write(2, 0);  // bits_per_sample selector 0 -> 8
  writer->Write(1, 1);  // modular_16_bit_buffer_sufficient = true
  // num_extra_channels: U32(Val(0), Val(1), BitsOffset(4,2), BitsOffset(12,1))
  // Pick the distribution the value actually needs, matching the encoder, so
  // the under-limit cases are canonical bitstreams.
  if (n_extra == 0) {
    writer->Write(2, 0);  // Val(0)
  } else if (n_extra == 1) {
    writer->Write(2, 1);  // Val(1)
  } else if (n_extra <= 17) {
    writer->Write(2, 2);  // BitsOffset(4,2) -> 2..17
    writer->Write(4, n_extra - 2);
  } else {
    writer->Write(2, 3);  // BitsOffset(12,1) -> 1..4096
    writer->Write(12, n_extra - 1);
  }
  // One all-default ExtraChannelInfo per channel (Bundle::AllDefault -> 1 bit).
  for (uint32_t i = 0; i < n_extra; ++i) {
    writer->Write(1, 1);
  }
  // Trailing fields the reader will look for after the extra-channel list.
  // (For an over-limit count the cap rejects before these are reached, but the
  // under-limit cases must be complete or the parse fails for the wrong reason.)
  writer->Write(1, 1);  // xyb_encoded = true (the default)
  writer->Write(1, 1);  // ColorEncoding all_default = true
  writer->ZeroPadToByte();
}

// Parses the written metadata. Returns true if the bundle was accepted.
bool ParseMetadata(const PaddedBytes& bytes, ImageMetadata* metadata) {
  BitReader reader(Span<const uint8_t>(bytes.data(), bytes.size()));
  Status ok = Bundle::Read(&reader, metadata);
  // Close the reader regardless: Bundle::Read may leave it mid-stream.
  if (ok) EXPECT_TRUE(reader.Close());
  return ok;
}

TEST(ImageMetadataTest, AcceptsEncoderChannelLimit) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  for (uint32_t n : {0u, 1u, 4u, 256u}) {
    BitWriter writer{memory_manager};
    ASSERT_TRUE(writer.WithMaxBits(1 << 16, LayerType::Header, nullptr, [&] {
      WriteMetadataWithExtraChannels(&writer, n);
      return true;
    }));
    PaddedBytes bytes = std::move(writer).TakeBytes();

    ImageMetadata metadata;
    EXPECT_TRUE(ParseMetadata(bytes, &metadata))
        << "n_extra=" << n << " should be accepted";
    EXPECT_EQ(metadata.extra_channel_info.size(), n) << "n_extra=" << n;
  }
}

TEST(ImageMetadataTest, RejectsExcessiveExtraChannels) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  // 257 is the smallest value over the encoder limit (encode.cc caps at 256);
  // 4096 is the U32 BitsOffset(12,1) maximum and the value used by the PoC.
  for (uint32_t n : {257u, 1024u, 4096u}) {
    BitWriter writer{memory_manager};
    ASSERT_TRUE(writer.WithMaxBits(1 << 20, LayerType::Header, nullptr, [&] {
      WriteMetadataWithExtraChannels(&writer, n);
      return true;
    }));
    PaddedBytes bytes = std::move(writer).TakeBytes();

    // A tiny file must not be able to force one full-frame plane per channel.
    EXPECT_LT(bytes.size(), 4096u) << "n_extra=" << n;

    ImageMetadata metadata;
    EXPECT_FALSE(ParseMetadata(bytes, &metadata))
        << "n_extra=" << n << " must be rejected at parse time";
  }
}

TEST(ImageMetadataTest, ExtraChannelCapMatchesEncoder) {
  // The decoder limit must agree with the encoder limit so a valid encode
  // always round-trips. Keep this assertion next to the cap so a future change
  // to one side is caught here.
  ImageMetadata metadata;
  metadata.extra_channel_info.resize(256);
  BitWriter writer{jxl::test::MemoryManager()};
  EXPECT_TRUE(Bundle::Write(metadata, &writer, LayerType::Header, nullptr));

  metadata.extra_channel_info.resize(257);
  BitWriter writer2{jxl::test::MemoryManager()};
  EXPECT_FALSE(Bundle::Write(metadata, &writer2, LayerType::Header, nullptr));
}

}  // namespace
}  // namespace jxl
