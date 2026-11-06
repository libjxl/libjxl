// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_external_image.h"

#include <jxl/types.h>

#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "lib/extras/codec_in_out.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/pnm.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

TEST(ExternalImageTest, InvalidSize) {
  if (JXL_CRASH_ON_ERROR) {
    GTEST_SKIP() << "Skipping due to JXL_CRASH_ON_ERROR";
  }
  ImageMetadata im;
  im.SetAlphaBits(8);
  ImageBundle ib(jxl::test::MemoryManager(), &im);

  JxlPixelFormat format = {4, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};
  std::vector<uint8_t> buf(10 * 100 * 8);
  EXPECT_FALSE(ConvertFromExternal(
      Bytes(buf.data(), 10), /*xsize=*/10, /*ysize=*/100,
      /*c_current=*/ColorEncoding::SRGB(),
      /*bits_per_sample=*/16, format, nullptr, &ib, /*set_alpha=*/true));
  EXPECT_FALSE(ConvertFromExternal(
      Bytes(buf.data(), buf.size() - 1), /*xsize=*/10, /*ysize=*/100,
      /*c_current=*/ColorEncoding::SRGB(),
      /*bits_per_sample=*/16, format, nullptr, &ib, /*set_alpha=*/true));
  EXPECT_TRUE(ConvertFromExternal(
      Bytes(buf), /*xsize=*/10,
      /*ysize=*/100, /*c_current=*/ColorEncoding::SRGB(),
      /*bits_per_sample=*/16, format, nullptr, &ib, /*set_alpha=*/true));
}

TEST(ExternalImageTest, AlphaMissing) {
  ImageMetadata im;
  im.SetAlphaBits(0);  // No alpha
  ImageBundle ib(jxl::test::MemoryManager(), &im);

  const size_t xsize = 10;
  const size_t ysize = 20;
  const uint8_t buf[xsize * ysize * 4] = {};

  JxlPixelFormat format = {4, JXL_TYPE_UINT8, JXL_BIG_ENDIAN, 0};
  // has_alpha is true but the ImageBundle has no alpha. Alpha channel should
  // be ignored.
  EXPECT_TRUE(ConvertFromExternal(Bytes(buf, sizeof(buf)), xsize, ysize,
                                  /*c_current=*/ColorEncoding::SRGB(),
                                  /*bits_per_sample=*/8, format, nullptr, &ib));
  EXPECT_FALSE(ib.HasAlpha());
}

TEST(ExternalImageTest, AlphaPremultiplied) {
  ImageMetadata im;
  im.SetAlphaBits(8, true);

  ImageBundle ib(jxl::test::MemoryManager(), &im);
  const size_t xsize = 10;
  const size_t ysize = 20;
  const size_t size = xsize * ysize * 8;
  const uint8_t buf[size] = {};

  JxlPixelFormat format = {4, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};
  EXPECT_TRUE(BufferToImageBundle(format, xsize, ysize, buf, size, nullptr,
                                  ColorEncoding::SRGB(), &ib,
                                  /*set_alpha*/ true));
}

// Verify zero-sized external buffers are rejected instead of bypassing the
// buffer-size check and reading out of bounds in LoadFloatRow.
TEST(ExternalImageTest, ZeroSizeExternalBufferRejected) {
  if (JXL_CRASH_ON_ERROR) {
    return;
  }
  ImageMetadata im;
  ImageBundle ib(jxl::test::MemoryManager(), &im);
  const JxlPixelFormat rgb_format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  std::vector<uint8_t> one_byte(1, 0x2a);
  EXPECT_FALSE(ConvertFromExternal(Bytes(one_byte.data(), 0), /*xsize=*/1,
                                   /*ysize=*/1, ColorEncoding::SRGB(),
                                   /*bits_per_sample=*/8, rgb_format, nullptr,
                                   &ib));

  extras::PackedPixelFile ppf;
  ppf.info.xsize = 1;
  ppf.info.ysize = 1;
  ppf.info.num_color_channels = 3;
  ppf.info.bits_per_sample = 8;
  ppf.info.orientation = JXL_ORIENT_IDENTITY;
  ppf.info.uses_original_profile = JXL_TRUE;
  JxlColorEncodingSetToSRGB(&ppf.color_encoding, JXL_FALSE);

  const JxlPixelFormat gray_format = {1, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  JXL_TEST_ASSIGN_OR_DIE(extras::PackedFrame frame,
                         extras::PackedFrame::Create(1, 1, gray_format));
  frame.frame_info.layer_info.xsize = 1;
  frame.frame_info.layer_info.ysize = 1;
  frame.color.format.num_channels = 3;
  frame.color.pixels_size = 0;
  ppf.frames.emplace_back(std::move(frame));

  auto io = jxl::make_unique<CodecInOut>(jxl::test::MemoryManager());
  EXPECT_FALSE(
      extras::ConvertPackedPixelFileToCodecInOut(ppf, nullptr, io.get()));
}

// Verify mismatched frame extra-channel counts are rejected cleanly instead
// of writing out of bounds past ImageBundle::extra_channels().
TEST(ExternalImageTest, PackedFrameExtraChannelMismatchRejected) {
  if (JXL_CRASH_ON_ERROR) {
    return;
  }
  extras::PackedPixelFile ppf;
  ppf.info.xsize = 1;
  ppf.info.ysize = 1;
  ppf.info.num_color_channels = 3;
  ppf.info.bits_per_sample = 8;
  ppf.info.orientation = JXL_ORIENT_IDENTITY;
  ppf.info.uses_original_profile = JXL_TRUE;
  ppf.info.num_extra_channels = 1;
  JxlColorEncodingSetToSRGB(&ppf.color_encoding, JXL_FALSE);

  extras::PackedExtraChannel channel_info = {};
  channel_info.index = 0;
  channel_info.ec_info.type = JXL_CHANNEL_OPTIONAL;
  channel_info.ec_info.bits_per_sample = 8;
  ppf.extra_channels_info.push_back(channel_info);

  const JxlPixelFormat color_format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  const JxlPixelFormat ec_format = {1, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  JXL_TEST_ASSIGN_OR_DIE(extras::PackedFrame frame,
                         extras::PackedFrame::Create(1, 1, color_format));
  frame.frame_info.layer_info.xsize = 1;
  frame.frame_info.layer_info.ysize = 1;
  JXL_TEST_ASSIGN_OR_DIE(extras::PackedImage ec0,
                         extras::PackedImage::Create(1, 1, ec_format));
  JXL_TEST_ASSIGN_OR_DIE(extras::PackedImage ec1,
                         extras::PackedImage::Create(1, 1, ec_format));
  frame.extra_channels.emplace_back(std::move(ec0));
  frame.extra_channels.emplace_back(std::move(ec1));
  ppf.frames.emplace_back(std::move(frame));

  auto io = jxl::make_unique<CodecInOut>(jxl::test::MemoryManager());
  EXPECT_FALSE(
      extras::ConvertPackedPixelFileToCodecInOut(ppf, nullptr, io.get()));

  ppf.frames[0].extra_channels.clear();
  EXPECT_FALSE(
      extras::ConvertPackedPixelFileToCodecInOut(ppf, nullptr, io.get()));
}

// Positive control: a valid 5-channel PAM (RGB_ALPHA + extra channel) keeps
// interleaved alpha at extra_channels()[0] and places the non-alpha extra
// channel at extra_channels()[1], while planar RGB + planar alpha (as emitted
// by DecodeImageGIF) populates extra_channels()[0].
TEST(ExternalImageTest, PackedFrameInterleavedAlphaAndExtraChannelAccepted) {
  std::string pam =
      "P7\n"
      "WIDTH 2\n"
      "HEIGHT 1\n"
      "DEPTH 5\n"
      "MAXVAL 255\n"
      "TUPLTYPE RGB_ALPHA\n"
      "TUPLTYPE Optional\n"
      "ENDHDR\n";
  const uint8_t samples[10] = {
      10, 20, 30, 64,  200,
      40, 50, 60, 192, 100,
  };
  pam.append(reinterpret_cast<const char*>(samples), sizeof(samples));

  extras::PackedPixelFile ppf;
  ASSERT_TRUE(extras::DecodeImagePNM(
      Span<const uint8_t>(reinterpret_cast<const uint8_t*>(pam.data()),
                          pam.size()),
      extras::ColorHints(), &ppf));
  ASSERT_EQ(1u, ppf.extra_channels_info.size());
  ASSERT_EQ(1u, ppf.frames.size());
  ASSERT_EQ(1u, ppf.frames[0].extra_channels.size());

  auto io = jxl::make_unique<CodecInOut>(jxl::test::MemoryManager());
  ASSERT_TRUE(
      extras::ConvertPackedPixelFileToCodecInOut(ppf, nullptr, io.get()));
  ASSERT_EQ(2u, io->Main().extra_channels().size());
  ASSERT_EQ(2u, io->Main().extra_channels()[0].xsize());
  ASSERT_EQ(1u, io->Main().extra_channels()[0].ysize());
  ASSERT_EQ(2u, io->Main().extra_channels()[1].xsize());
  ASSERT_EQ(1u, io->Main().extra_channels()[1].ysize());
  EXPECT_NEAR(64.0f / 255.0f, io->Main().extra_channels()[0].Row(0)[0], 1e-4f);
  EXPECT_NEAR(200.0f / 255.0f, io->Main().extra_channels()[1].Row(0)[0], 1e-4f);

  // Planar alpha case (e.g. GIF): 3-channel RGB color + planar alpha in
  // frame.extra_channels[0] with alpha_bits == 8 and empty extra_channels_info.
  extras::PackedPixelFile gif_ppf;
  gif_ppf.info.xsize = 2;
  gif_ppf.info.ysize = 1;
  gif_ppf.info.num_color_channels = 3;
  gif_ppf.info.bits_per_sample = 8;
  gif_ppf.info.alpha_bits = 8;
  gif_ppf.info.orientation = JXL_ORIENT_IDENTITY;
  gif_ppf.info.uses_original_profile = JXL_TRUE;
  JxlColorEncodingSetToSRGB(&gif_ppf.color_encoding, JXL_FALSE);
  const JxlPixelFormat rgb_format = {3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  const JxlPixelFormat a_format = {1, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
  JXL_TEST_ASSIGN_OR_DIE(extras::PackedFrame gif_frame,
                         extras::PackedFrame::Create(2, 1, rgb_format));
  gif_frame.frame_info.layer_info.xsize = 2;
  gif_frame.frame_info.layer_info.ysize = 1;
  JXL_TEST_ASSIGN_OR_DIE(extras::PackedImage gif_alpha,
                         extras::PackedImage::Create(2, 1, a_format));
  static_cast<uint8_t*>(gif_alpha.pixels())[0] = 32;
  static_cast<uint8_t*>(gif_alpha.pixels())[1] = 224;
  gif_frame.extra_channels.emplace_back(std::move(gif_alpha));
  gif_ppf.frames.emplace_back(std::move(gif_frame));

  auto gif_io = jxl::make_unique<CodecInOut>(jxl::test::MemoryManager());
  ASSERT_TRUE(extras::ConvertPackedPixelFileToCodecInOut(gif_ppf, nullptr,
                                                         gif_io.get()));
  ASSERT_EQ(1u, gif_io->Main().extra_channels().size());
  EXPECT_NEAR(32.0f / 255.0f, gif_io->Main().extra_channels()[0].Row(0)[0],
              1e-4f);
}

}  // namespace
}  // namespace jxl
