// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/color_encoding_internal.h"

#include <jxl/cms.h>
#include <jxl/color_encoding.h>
#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/encode.h>
#include <jxl/encode_cxx.h>

#include <cstdint>
#include <cstdlib>  // rand
#include <memory>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/cms/color_encoding_cms.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

using ::jxl::cms::ColorEncoding;

TEST(ColorEncodingTest, RoundTripAll) {
  for (const test::ColorEncodingDescriptor& cdesc : test::AllEncodings()) {
    ColorEncoding c_original = test::ColorEncodingFromDescriptor(cdesc).View();
    // Verify Set(Get) yields the same white point/primaries/gamma.
    {
      ColorEncoding c;
      EXPECT_TRUE(c.SetWhitePoint(c_original.GetWhitePoint()));
      EXPECT_EQ(c_original.white_point, c.white_point);
    }
    {
      ColorEncoding c;
      PrimariesCIExy p;
      ASSERT_TRUE(c_original.GetPrimaries(p));
      EXPECT_TRUE(c.SetPrimaries(p));
      EXPECT_EQ(c_original.primaries, c.primaries);
    }
    if (c_original.tf.have_gamma) {
      ColorEncoding c;
      EXPECT_TRUE(c.tf.SetGamma(c_original.tf.GetGamma()));
      EXPECT_TRUE(c_original.tf.IsSame(c.tf));
    }
  }
}

TEST(ColorEncodingTest, CustomWhitePoint) {
  ColorEncoding c;
  // Nonsensical values
  CIExy xy_in;
  xy_in.x = 0.8;
  xy_in.y = 0.01;
  EXPECT_TRUE(c.SetWhitePoint(xy_in));
  const CIExy xy = c.GetWhitePoint();

  ColorEncoding c2;
  EXPECT_TRUE(c2.SetWhitePoint(xy));
  EXPECT_TRUE(c.SameColorSpace(c2));
}

TEST(ColorEncodingTest, CustomPrimaries) {
  ColorEncoding c;
  PrimariesCIExy xy_in;
  // Nonsensical values
  xy_in.r.x = -0.01;
  xy_in.r.y = 0.2;
  xy_in.g.x = 0.4;
  xy_in.g.y = 0.401;
  xy_in.b.x = 1.1;
  xy_in.b.y = -1.2;
  EXPECT_TRUE(c.SetPrimaries(xy_in));
  PrimariesCIExy xy;
  ASSERT_TRUE(c.GetPrimaries(xy));

  ColorEncoding c2;
  EXPECT_TRUE(c2.SetPrimaries(xy));
  EXPECT_TRUE(c.SameColorSpace(c2));
}

TEST(ColorEncodingTest, CustomGamma) {
  ColorEncoding c;
  if (!JXL_CRASH_ON_ERROR) {
    EXPECT_FALSE(c.tf.SetGamma(0.0));
    EXPECT_FALSE(c.tf.SetGamma(-1E-6));
    EXPECT_FALSE(c.tf.SetGamma(1.001));
  }

  EXPECT_TRUE(c.tf.SetGamma(1.0));
  EXPECT_FALSE(c.tf.have_gamma);
  EXPECT_TRUE(c.tf.IsLinear());

  EXPECT_TRUE(c.tf.SetGamma(0.123));
  EXPECT_TRUE(c.tf.have_gamma);
  const double gamma = c.tf.GetGamma();

  ColorEncoding c2;
  EXPECT_TRUE(c2.tf.SetGamma(gamma));
  EXPECT_TRUE(c.SameColorEncoding(c2));
  EXPECT_TRUE(c2.tf.have_gamma);
}

TEST(ColorEncodingTest, InternalExternalConversion) {
  ColorEncoding source_internal;
  ColorEncoding destination_internal;

  const auto rand_float = []() {
    return (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 0.5) +
           0.25;
  };

  for (int i = 0; i < 100; i++) {
    source_internal.color_space = static_cast<ColorSpace>(rand() % 4);
    CIExy wp;
    wp.x = rand_float();
    wp.y = rand_float();
    EXPECT_TRUE(source_internal.SetWhitePoint(wp));
    if (source_internal.HasPrimaries()) {
      PrimariesCIExy primaries;
      primaries.r.x = rand_float();
      primaries.r.y = rand_float();
      primaries.g.x = rand_float();
      primaries.g.y = rand_float();
      primaries.b.x = rand_float();
      primaries.b.y = rand_float();
      EXPECT_TRUE(source_internal.SetPrimaries(primaries));
    }
    jxl::cms::CustomTransferFunction tf;
    EXPECT_TRUE(tf.SetGamma(rand_float()));
    source_internal.tf = tf;
    source_internal.rendering_intent = static_cast<RenderingIntent>(rand() % 4);

    JxlColorEncoding external = source_internal.ToExternal();
    EXPECT_TRUE(destination_internal.FromExternal(external));

    EXPECT_EQ(source_internal.color_space, destination_internal.color_space);
    EXPECT_EQ(source_internal.white_point, destination_internal.white_point);
    CIExy src_wp = source_internal.GetWhitePoint();
    CIExy dst_wp = destination_internal.GetWhitePoint();
    EXPECT_EQ(src_wp.x, dst_wp.x);
    EXPECT_EQ(src_wp.y, dst_wp.y);
    if (source_internal.HasPrimaries()) {
      PrimariesCIExy src_p;
      ASSERT_TRUE(source_internal.GetPrimaries(src_p));
      PrimariesCIExy dst_p;
      ASSERT_TRUE(destination_internal.GetPrimaries(dst_p));
      EXPECT_EQ(src_p.r.x, dst_p.r.x);
      EXPECT_EQ(src_p.r.y, dst_p.r.y);
      EXPECT_EQ(src_p.g.x, dst_p.g.x);
      EXPECT_EQ(src_p.g.y, dst_p.g.y);
      EXPECT_EQ(src_p.b.x, dst_p.b.x);
      EXPECT_EQ(src_p.b.y, dst_p.b.y);
    }
    EXPECT_EQ(source_internal.tf.have_gamma,
              destination_internal.tf.have_gamma);
    if (source_internal.tf.have_gamma) {
      EXPECT_EQ(source_internal.tf.GetGamma(),
                destination_internal.tf.GetGamma());
    } else {
      EXPECT_EQ(source_internal.tf.GetTransferFunction(),
                destination_internal.tf.GetTransferFunction());
    }
    EXPECT_EQ(source_internal.rendering_intent,
              destination_internal.rendering_intent);
  }
}

// Decodes a JPEG XL file with the SimpleRenderPipeline (the slow pipeline used
// by tests) into sRGB.
Status DecodeWithSimplePipelineToSRGB(Span<const uint8_t> file) {
  if (file.size() >= 12 && file[0] == 0x00 && file[1] == 0x00 &&
      file[2] == 0x00 && file[3] == 0x0C && file[4] == 'J' && file[5] == 'X' &&
      file[6] == 'L' && file[7] == ' ') {
    size_t pos = 0;
    bool found_jxlc = false;
    while (pos + 8 <= file.size()) {
      const uint32_t box_size = (static_cast<uint32_t>(file[pos]) << 24) |
                                (static_cast<uint32_t>(file[pos + 1]) << 16) |
                                (static_cast<uint32_t>(file[pos + 2]) << 8) |
                                static_cast<uint32_t>(file[pos + 3]);
      if (box_size < 8 || pos + box_size > file.size()) {
        return JXL_FAILURE("Invalid container box size");
      }
      if (file[pos + 4] == 'j' && file[pos + 5] == 'x' &&
          file[pos + 6] == 'l' && file[pos + 7] == 'c') {
        file = Span<const uint8_t>(file.data() + pos + 8, box_size - 8);
        found_jxlc = true;
        break;
      }
      pos += box_size;
    }
    JXL_RETURN_IF_ERROR(found_jxlc);
  }
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  Status ret = true;
  CodecMetadata metadata;
  size_t frame_start = 0;
  {
    BitReader reader(file);
    BitReaderScopedCloser closer(reader, ret);
    JXL_RETURN_IF_ERROR(reader.ReadFixedBits<16>() == 0x0AFF);
    JXL_RETURN_IF_ERROR(ReadSizeHeader(&reader, &metadata.size));
    JXL_RETURN_IF_ERROR(ReadImageMetadata(&reader, &metadata.m));
    metadata.transform_data.nonserialized_xyb_encoded = metadata.m.xyb_encoded;
    JXL_RETURN_IF_ERROR(Bundle::Read(&reader, &metadata.transform_data));
    if (metadata.m.color_encoding.WantICC()) {
      IccBytes icc;
      JXL_RETURN_IF_ERROR(test::ReadICC(&reader, &icc));
      JXL_RETURN_IF_ERROR(
          metadata.m.color_encoding.SetICC(std::move(icc), JxlGetDefaultCms()));
    }
    JXL_RETURN_IF_ERROR(reader.JumpToByteBoundary());
    frame_start = reader.TotalBitsConsumed() / kBitsPerByte;
  }
  JXL_RETURN_IF_ERROR(ret);

  auto dec_state = jxl::make_unique<PassesDecoderState>(memory_manager);
  JXL_RETURN_IF_ERROR(
      dec_state->output_encoding_info.SetFromMetadata(metadata));
  dec_state->output_encoding_info.color_management_system = *JxlGetDefaultCms();
  dec_state->output_encoding_info.cms_set = true;
  JXL_RETURN_IF_ERROR(dec_state->output_encoding_info.MaybeSetColorEncoding(
      jxl::ColorEncoding::SRGB(/*is_gray=*/false)));
  FrameHeader frame_header(&metadata);
  ImageBundle decoded(memory_manager, &metadata.m);
  return DecodeFrame(dec_state.get(), /*pool=*/nullptr,
                     file.data() + frame_start, file.size() - frame_start,
                     &frame_header, &decoded, metadata,
                     /*use_slow_rendering_pipeline=*/true);
}

constexpr size_t kCmykXSize = 64;
constexpr size_t kCmykYSize = 2;

// Encodes a valid CMYK file (original profile) with a black extra channel.
void EncodeValidCmykWithBlackChannel(std::vector<uint8_t>* compressed) {
  const IccBytes cmyk_icc = jxl::test::ReadTestData(
      "external/Compact-ICC-Profiles/profiles/CGATS001Compat-v2-micro.icc");
  ASSERT_FALSE(cmyk_icc.empty());

  JxlEncoderPtr enc = JxlEncoderMake(nullptr);
  ASSERT_NE(nullptr, enc.get());
  JxlEncoderFrameSettings* frame_settings =
      JxlEncoderFrameSettingsCreate(enc.get(), nullptr);
  ASSERT_NE(nullptr, frame_settings);

  JxlBasicInfo basic_info;
  JxlEncoderInitBasicInfo(&basic_info);
  basic_info.xsize = kCmykXSize;
  basic_info.ysize = kCmykYSize;
  basic_info.bits_per_sample = 8;
  basic_info.num_color_channels = 3;
  basic_info.num_extra_channels = 1;
  basic_info.uses_original_profile = JXL_TRUE;
  ASSERT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetBasicInfo(enc.get(), &basic_info));

  JxlExtraChannelInfo k_info;
  JxlEncoderInitExtraChannelInfo(JXL_CHANNEL_BLACK, &k_info);
  k_info.bits_per_sample = 8;
  ASSERT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderSetExtraChannelInfo(enc.get(), 0, &k_info));
  ASSERT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetICCProfile(enc.get(), cmyk_icc.data(),
                                                     cmyk_icc.size()));

  const JxlPixelFormat cmy_format = {3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};
  const JxlPixelFormat k_format = {1, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};
  std::vector<uint8_t> cmy_pixels(kCmykXSize * kCmykYSize * 3, 200);
  std::vector<uint8_t> k_pixels(kCmykXSize * kCmykYSize, 240);
  ASSERT_EQ(JXL_ENC_SUCCESS,
            JxlEncoderAddImageFrame(frame_settings, &cmy_format,
                                    cmy_pixels.data(), cmy_pixels.size()));
  ASSERT_EQ(JXL_ENC_SUCCESS, JxlEncoderSetExtraChannelBuffer(
                                 frame_settings, &k_format, k_pixels.data(),
                                 k_pixels.size(), 0));
  JxlEncoderCloseInput(enc.get());

  compressed->assign(512, 0);
  uint8_t* next_out = compressed->data();
  size_t avail_out = compressed->size();
  JxlEncoderStatus enc_res = JXL_ENC_NEED_MORE_OUTPUT;
  while (enc_res == JXL_ENC_NEED_MORE_OUTPUT) {
    const size_t offset = static_cast<size_t>(next_out - compressed->data());
    if (avail_out == 0) {
      compressed->resize(compressed->size() * 2);
      next_out = compressed->data() + offset;
      avail_out = compressed->size() - offset;
    }
    enc_res = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
  }
  ASSERT_EQ(JXL_ENC_SUCCESS, enc_res);
  compressed->resize(static_cast<size_t>(next_out - compressed->data()));
}

// A valid CMYK file with a black extra channel decodes through the public
// JxlDecoder API (LowMemoryRenderPipeline).
TEST(ColorEncodingTest, ValidCmykWithBlackChannelDecodesWithJxlDecoder) {
  std::vector<uint8_t> compressed;
  EncodeValidCmykWithBlackChannel(&compressed);
  ASSERT_FALSE(compressed.empty());

  JxlDecoderPtr dec = JxlDecoderMake(nullptr);
  ASSERT_NE(nullptr, dec.get());
  ASSERT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO |
                                                     JXL_DEC_COLOR_ENCODING |
                                                     JXL_DEC_FULL_IMAGE));
  ASSERT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetCms(dec.get(), *JxlGetDefaultCms()));
  ASSERT_EQ(JXL_DEC_SUCCESS, JxlDecoderSetInput(dec.get(), compressed.data(),
                                                compressed.size()));
  JxlDecoderCloseInput(dec.get());
  ASSERT_EQ(JXL_DEC_BASIC_INFO, JxlDecoderProcessInput(dec.get()));
  ASSERT_EQ(JXL_DEC_COLOR_ENCODING, JxlDecoderProcessInput(dec.get()));

  JxlColorEncoding srgb;
  JxlColorEncodingSetToSRGB(&srgb, /*is_gray=*/JXL_FALSE);
  ASSERT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetPreferredColorProfile(dec.get(), &srgb));
  ASSERT_EQ(JXL_DEC_NEED_IMAGE_OUT_BUFFER, JxlDecoderProcessInput(dec.get()));

  const JxlPixelFormat format = {3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0};
  std::vector<uint8_t> out_rgb(kCmykXSize * kCmykYSize * 3, 0);
  ASSERT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetImageOutBuffer(dec.get(), &format, out_rgb.data(),
                                        out_rgb.size()));
  ASSERT_EQ(JXL_DEC_FULL_IMAGE, JxlDecoderProcessInput(dec.get()));
  ASSERT_EQ(JXL_DEC_SUCCESS, JxlDecoderProcessInput(dec.get()));
}

// SimpleRenderPipeline calls CmsStage::PrepareForThreads before
// CmsStage::SetInputSizes. The transform buffers used to be sized for zero
// pixels in that order, and ProcessRow overflowed them for a 64 pixel row.
TEST(ColorEncodingTest, CmykSimpleRenderPipelineTransformBufferOverflow) {
  std::vector<uint8_t> compressed;
  EncodeValidCmykWithBlackChannel(&compressed);
  ASSERT_FALSE(compressed.empty());
  EXPECT_TRUE(DecodeWithSimplePipelineToSRGB(Span<const uint8_t>(compressed)));
}

}  // namespace
}  // namespace jxl
