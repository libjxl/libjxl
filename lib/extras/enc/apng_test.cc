// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/enc/apng.h"

#include <jxl/codestream_header.h>
#include <jxl/types.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "lib/extras/enc/encode.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace extras {
namespace {

PackedPixelFile MakeAnimation(uint32_t tps_numerator, uint32_t tps_denominator,
                              uint32_t duration) {
  PackedPixelFile ppf;
  ppf.info.xsize = 4;
  ppf.info.ysize = 4;
  ppf.info.bits_per_sample = 8;
  ppf.info.num_color_channels = 3;
  ppf.info.alpha_bits = 0;
  ppf.info.have_animation = JXL_TRUE;
  ppf.info.animation.tps_numerator = tps_numerator;
  ppf.info.animation.tps_denominator = tps_denominator;
  ppf.info.animation.num_loops = 0;
  JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_BIG_ENDIAN, 0};
  for (int i = 0; i < 2; ++i) {
    JXL_TEST_ASSIGN_OR_DIE(PackedFrame frame,
                           PackedFrame::Create(4, 4, format));
    memset(frame.color.pixels(), 0x40 + i, frame.color.pixels_size);
    frame.frame_info.duration = duration;
    ppf.frames.emplace_back(std::move(frame));
  }
  return ppf;
}

// Finds the first fcTL chunk in an APNG bitstream and returns its
// delay_num / delay_den (16-bit big-endian fields at offset 20 / 22 of the
// 26-byte chunk data).
void FindFctlDelay(const std::vector<uint8_t>& bytes, uint16_t* delay_num,
                   uint16_t* delay_den) {
  for (size_t p = 0; p + 8 + 26 <= bytes.size(); ++p) {
    if (memcmp(bytes.data() + p + 4, "fcTL", 4) == 0) {
      const uint8_t* d = bytes.data() + p + 8;
      *delay_num = static_cast<uint16_t>((d[20] << 8) | d[21]);
      *delay_den = static_cast<uint16_t>((d[22] << 8) | d[23]);
      return;
    }
  }
  FAIL() << "no fcTL chunk found";
}

// NTSC-style timing: 30000/1001 ticks per second, 300 ticks = 10.01 s. The
// exact fraction 300*1001/30000 reduces to 1001/100 and must be written to
// the fcTL delay fields as such. The old code stored the un-reduced uint32
// product (truncated to 16 bits) in delay_num and the full tps_numerator in
// delay_den, corrupting the frame delay to 38156/30000 s.
TEST(APNGEncoderTest, FcTLDelayFractionFitsIn16Bits) {
  if (!GetAPNGEncoder()) GTEST_SKIP() << "APNG encoder not available";
  PackedPixelFile ppf = MakeAnimation(30000, 1001, 300);
  auto encoder = GetAPNGEncoder();
  EncodedImage encoded;
  ASSERT_TRUE(encoder->Encode(ppf, &encoded, nullptr));
  ASSERT_EQ(encoded.bitstreams.size(), 1u);
  uint16_t delay_num = 0, delay_den = 0;
  FindFctlDelay(encoded.bitstreams[0], &delay_num, &delay_den);
  EXPECT_EQ(delay_num, 1001);
  EXPECT_EQ(delay_den, 100);
}

// Control cell: a timing that already fits in the 16-bit fields with gcd 1
// must be written identically with and without the fix.
TEST(APNGEncoderTest, FcTLDelayControlUnchanged) {
  if (!GetAPNGEncoder()) GTEST_SKIP() << "APNG encoder not available";
  PackedPixelFile ppf = MakeAnimation(7, 3, 2);  // 2 * 3 / 7 s -> 6/7
  auto encoder = GetAPNGEncoder();
  EncodedImage encoded;
  ASSERT_TRUE(encoder->Encode(ppf, &encoded, nullptr));
  uint16_t delay_num = 0, delay_den = 0;
  FindFctlDelay(encoded.bitstreams[0], &delay_num, &delay_den);
  EXPECT_EQ(delay_num, 6);
  EXPECT_EQ(delay_den, 7);
}

}  // namespace
}  // namespace extras
}  // namespace jxl
