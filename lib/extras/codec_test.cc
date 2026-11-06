// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/codestream_header.h>
#include <jxl/color_encoding.h>
#include <jxl/encode.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "lib/extras/common.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/enc/encode.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {

using ::jxl::test::ThreadPoolForTests;

namespace extras {

Status PnmParseSigned(Bytes str, double* v);
Status PnmParseUnsigned(Bytes str, size_t* v);

namespace {

TEST(CodecTest, EncodersRejectSurplusFrameExtraChannel) {
  PackedPixelFile ppf;
  ppf.info.xsize = 1;
  ppf.info.ysize = 1;
  ppf.info.num_color_channels = 1;
  ppf.info.bits_per_sample = 8;
  const JxlPixelFormat gray{1, JXL_TYPE_UINT8, JXL_BIG_ENDIAN, 0};
  JXL_TEST_ASSIGN_OR_DIE(PackedFrame frame, PackedFrame::Create(1, 1, gray));
  JXL_TEST_ASSIGN_OR_DIE(PackedImage ec, PackedImage::Create(1, 1, gray));
  memset(frame.color.pixels(), 0x7f, frame.color.pixels_size);
  memset(ec.pixels(), 0x3f, ec.pixels_size);
  frame.extra_channels.emplace_back(std::move(ec));
  ppf.frames.emplace_back(std::move(frame));
  for (const char* extension : {".pgm", ".png"}) {
    auto encoder = Encoder::FromExtension(extension);
    if (!encoder) continue;
    EncodedImage encoded;
    EXPECT_FALSE(encoder->Encode(ppf, &encoded, nullptr)) << extension;
  }
}

TEST(CodecTest, PackedPixelFileValidExtraChannelCounts) {
  PackedPixelFile ppf;
  ppf.info.xsize = 1;
  ppf.info.ysize = 1;
  ppf.info.num_color_channels = 1;
  ppf.info.bits_per_sample = 8;
  PackedExtraChannel metadata{};
  metadata.ec_info.bits_per_sample = 8;
  ppf.extra_channels_info.push_back(metadata);
  const JxlPixelFormat gray{1, JXL_TYPE_UINT8, JXL_BIG_ENDIAN, 0};
  for (int i = 0; i < 2; ++i) {
    JXL_TEST_ASSIGN_OR_DIE(PackedFrame frame, PackedFrame::Create(1, 1, gray));
    JXL_TEST_ASSIGN_OR_DIE(PackedImage ec, PackedImage::Create(1, 1, gray));
    memset(frame.color.pixels(), 0x7f, frame.color.pixels_size);
    memset(ec.pixels(), 0x3f, ec.pixels_size);
    frame.extra_channels.emplace_back(std::move(ec));
    ppf.frames.emplace_back(std::move(frame));
  }
  ASSERT_TRUE(ppf.Validate());
  for (const char* extension : {".pgm", ".png"}) {
    auto encoder = Encoder::FromExtension(extension);
    if (!encoder) continue;
    EncodedImage encoded;
    ASSERT_TRUE(encoder->Encode(ppf, &encoded, nullptr)) << extension;
    ASSERT_EQ(encoded.bitstreams.size(), 2u);
    ASSERT_EQ(encoded.extra_channel_bitstreams.size(), 1u);
    ASSERT_EQ(encoded.extra_channel_bitstreams[0].size(), 2u);
    for (const auto& bytes : encoded.extra_channel_bitstreams[0]) {
      PackedPixelFile decoded;
      ASSERT_TRUE(DecodeBytes(Bytes(bytes), ColorHints(), &decoded));
      ASSERT_EQ(decoded.frames.size(), 1u);
      EXPECT_EQ(*static_cast<const uint8_t*>(decoded.frames[0].color.pixels()),
                0x3f);
    }
  }

  ppf.frames[1].extra_channels.clear();
  EXPECT_FALSE(ppf.Validate());
  for (const char* extension : {".pgm", ".png"}) {
    auto encoder = Encoder::FromExtension(extension);
    if (!encoder) continue;
    EncodedImage encoded;
    EXPECT_FALSE(encoder->Encode(ppf, &encoded, nullptr)) << extension;
  }
}

Span<const uint8_t> MakeSpan(const char* str) {
  return Bytes(reinterpret_cast<const uint8_t*>(str), strlen(str));
}

std::string ExtensionFromCodec(Codec codec, const bool is_gray,
                               const bool has_alpha,
                               const size_t bits_per_sample) {
  switch (codec) {
    case Codec::kJPG:
      return ".jpg";
    case Codec::kPGX:
      return ".pgx";
    case Codec::kPNG:
      return ".png";
    case Codec::kPNM:
      if (bits_per_sample == 32) return ".pfm";
      if (has_alpha) return ".pam";
      return is_gray ? ".pgm" : ".ppm";
    case Codec::kEXR:
      return ".exr";
    default:
      return std::string();
  }
}

void VerifySameImage(const PackedImage& im0, size_t bits_per_sample0,
                     const PackedImage& im1, size_t bits_per_sample1,
                     bool lossless = true) {
  ASSERT_EQ(im0.xsize, im1.xsize);
  ASSERT_EQ(im0.ysize, im1.ysize);
  ASSERT_EQ(im0.format.num_channels, im1.format.num_channels);
  auto get_factor = [](JxlPixelFormat f, size_t bits) -> double {
    return 1.0 / ((1u << std::min(test::GetPrecision(f.data_type), bits)) - 1);
  };
  double factor0 = get_factor(im0.format, bits_per_sample0);
  double factor1 = get_factor(im1.format, bits_per_sample1);
  const auto* pixels0 = static_cast<const uint8_t*>(im0.pixels());
  const auto* pixels1 = static_cast<const uint8_t*>(im1.pixels());
  auto rgba0 =
      test::ConvertToRGBA32(pixels0, im0.xsize, im0.ysize, im0.format, factor0);
  auto rgba1 =
      test::ConvertToRGBA32(pixels1, im1.xsize, im1.ysize, im1.format, factor1);
  double tolerance =
      lossless ? 0.5 * std::min(factor0, factor1) : 3.0f / 255.0f;
  if (bits_per_sample0 == 32 || bits_per_sample1 == 32) {
    tolerance = 0.5 * std::max(factor0, factor1);
  }
  for (size_t y = 0; y < im0.ysize; ++y) {
    for (size_t x = 0; x < im0.xsize; ++x) {
      for (size_t c = 0; c < im0.format.num_channels; ++c) {
        size_t ix = (y * im0.xsize + x) * 4 + c;
        double val0 = rgba0[ix];
        double val1 = rgba1[ix];
        ASSERT_NEAR(val1, val0, tolerance)
            << "y = " << y << " x = " << x << " c = " << c;
      }
    }
  }
}

JxlColorEncoding CreateTestColorEncoding(bool is_gray) {
  JxlColorEncoding c;
  c.color_space = is_gray ? JXL_COLOR_SPACE_GRAY : JXL_COLOR_SPACE_RGB;
  c.white_point = JXL_WHITE_POINT_D65;
  c.primaries = JXL_PRIMARIES_P3;
  c.rendering_intent = JXL_RENDERING_INTENT_RELATIVE;
  c.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
  // Roundtrip through internal color encoding to fill in primaries and white
  // point CIE xy coordinates.
  ColorEncoding c_internal;
  EXPECT_TRUE(c_internal.FromExternal(c));
  c = c_internal.ToExternal();
  return c;
}

std::vector<uint8_t> GenerateICC(JxlColorEncoding color_encoding) {
  ColorEncoding c;
  EXPECT_TRUE(c.FromExternal(color_encoding));
  EXPECT_TRUE(!c.ICC().empty());
  return c.ICC();
}

void StoreRandomValue(uint8_t* out, Rng* rng, JxlPixelFormat format,
                      size_t bits_per_sample) {
  uint64_t max_val = (1ull << bits_per_sample) - 1;
  if (format.data_type == JXL_TYPE_UINT8) {
    *out = rng->UniformU(0, max_val);
  } else if (format.data_type == JXL_TYPE_UINT16) {
    uint32_t val = rng->UniformU(0, max_val);
    if (format.endianness == JXL_BIG_ENDIAN) {
      StoreBE16(val, out);
    } else {
      StoreLE16(val, out);
    }
  } else {
    ASSERT_EQ(format.data_type, JXL_TYPE_FLOAT);
    float val = rng->UniformF(0.0, 1.0);
    uint32_t uval;
    memcpy(&uval, &val, 4);
    if (format.endianness == JXL_BIG_ENDIAN) {
      StoreBE32(uval, out);
    } else {
      StoreLE32(uval, out);
    }
  }
}

void FillPackedImage(size_t bits_per_sample, PackedImage* image) {
  JxlPixelFormat format = image->format;
  size_t bytes_per_channel = PackedImage::BitsPerChannel(format.data_type) / 8;
  uint8_t* out = static_cast<uint8_t*>(image->pixels());
  size_t stride = image->xsize * format.num_channels * bytes_per_channel;
  ASSERT_EQ(image->pixels_size, image->ysize * stride);
  Rng rng(129);
  for (size_t y = 0; y < image->ysize; ++y) {
    for (size_t x = 0; x < image->xsize; ++x) {
      for (size_t c = 0; c < format.num_channels; ++c) {
        StoreRandomValue(out, &rng, format, bits_per_sample);
        out += bytes_per_channel;
      }
    }
  }
}

struct TestImageParams {
  Codec codec;
  size_t xsize;
  size_t ysize;
  size_t bits_per_sample;
  bool is_gray;
  bool add_alpha;
  bool big_endian;
  bool add_extra_channels;

  bool ShouldTestRoundtrip() const {
    if (codec == Codec::kPNG) {
      return bits_per_sample <= 16;
    } else if (codec == Codec::kPNM) {
      // TODO(szabadka) Make PNM encoder endianness-aware.
      return ((bits_per_sample <= 16 && big_endian) ||
              (bits_per_sample == 32 && !add_alpha && !big_endian));
    } else if (codec == Codec::kPGX) {
      return ((bits_per_sample == 8 || bits_per_sample == 16) && is_gray &&
              !add_alpha);
    } else if (codec == Codec::kEXR) {
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
      // OpenEXR 2.3 has a memory leak in IlmThread_2_3::ThreadPool
      return false;
#else
      return bits_per_sample == 32 && !is_gray;
#endif
    } else if (codec == Codec::kJPG) {
      return bits_per_sample == 8 && !add_alpha;
    } else {
      return false;
    }
  }

  JxlPixelFormat PixelFormat() const {
    JxlPixelFormat format;
    format.num_channels = (is_gray ? 1 : 3) + (add_alpha ? 1 : 0);
    format.data_type = (bits_per_sample == 32 ? JXL_TYPE_FLOAT
                        : bits_per_sample > 8 ? JXL_TYPE_UINT16
                                              : JXL_TYPE_UINT8);
    format.endianness = big_endian ? JXL_BIG_ENDIAN : JXL_LITTLE_ENDIAN;
    format.align = 0;
    return format;
  }

  std::string DebugString() const {
    std::ostringstream os;
    os << "bps:" << bits_per_sample << " gr:" << is_gray << " al:" << add_alpha
       << " be: " << big_endian << " ec: " << add_extra_channels;
    return os.str();
  }
};

void CreateTestImage(const TestImageParams& params, PackedPixelFile* ppf) {
  ppf->info.xsize = params.xsize;
  ppf->info.ysize = params.ysize;
  ppf->info.bits_per_sample = params.bits_per_sample;
  ppf->info.exponent_bits_per_sample = params.bits_per_sample == 32 ? 8 : 0;
  ppf->info.num_color_channels = params.is_gray ? 1 : 3;
  ppf->info.alpha_bits = params.add_alpha ? params.bits_per_sample : 0;
  ppf->info.alpha_premultiplied = TO_JXL_BOOL(params.codec == Codec::kEXR);

  JxlColorEncoding color_encoding = CreateTestColorEncoding(params.is_gray);
  ppf->icc = GenerateICC(color_encoding);
  ppf->color_encoding = color_encoding;

  JXL_TEST_ASSIGN_OR_DIE(
      PackedFrame frame,
      PackedFrame::Create(params.xsize, params.ysize, params.PixelFormat()));
  FillPackedImage(params.bits_per_sample, &frame.color);
  if (params.add_extra_channels) {
    for (size_t i = 0; i < 7; ++i) {
      JxlPixelFormat ec_format = params.PixelFormat();
      ec_format.num_channels = 1;
      JXL_TEST_ASSIGN_OR_DIE(
          PackedImage ec,
          PackedImage::Create(params.xsize, params.ysize, ec_format));
      FillPackedImage(params.bits_per_sample, &ec);
      frame.extra_channels.emplace_back(std::move(ec));
      PackedExtraChannel pec;
      pec.ec_info.bits_per_sample = params.bits_per_sample;
      pec.ec_info.type = static_cast<JxlExtraChannelType>(i);
      ppf->extra_channels_info.emplace_back(std::move(pec));
    }
  }
  ppf->frames.emplace_back(std::move(frame));
}

// Ensures reading a newly written file leads to the same image pixels.
void TestRoundTrip(const TestImageParams& params, ThreadPool* pool) {
  if (!params.ShouldTestRoundtrip()) return;

  std::string extension = ExtensionFromCodec(
      params.codec, params.is_gray, params.add_alpha, params.bits_per_sample);
  printf("Codec %s %s\n", extension.c_str(), params.DebugString().c_str());

  if (!CanDecode(params.codec)) {
    fprintf(stderr, "Skipping test because of missing decoding support.\n");
    return;
  }
  auto encoder = Encoder::FromExtension(extension);
  if (!encoder) {
    fprintf(stderr, "Skipping test because of missing encoding support.\n");
    return;
  }

  PackedPixelFile ppf_in;
  CreateTestImage(params, &ppf_in);
  EncodedImage encoded;
  ASSERT_TRUE(encoder->Encode(ppf_in, &encoded, pool));
  ASSERT_EQ(encoded.bitstreams.size(), 1u);

  PackedPixelFile ppf_out;
  ColorHints color_hints;
  if (params.codec == Codec::kPNM || params.codec == Codec::kPGX) {
    color_hints.Add("color_space",
                    params.is_gray ? "Gra_D65_Rel_SRG" : "RGB_D65_SRG_Rel_SRG");
  }
  ASSERT_TRUE(DecodeBytes(Bytes(encoded.bitstreams[0]), color_hints, &ppf_out));
  if (params.codec == Codec::kPNG && ppf_out.icc.empty()) {
    // Decoding a PNG may drop the ICC profile if there's a valid cICP chunk.
    // Rendering intent is not preserved in this case.
    EXPECT_EQ(ppf_in.color_encoding.color_space,
              ppf_out.color_encoding.color_space);
    EXPECT_EQ(ppf_in.color_encoding.white_point,
              ppf_out.color_encoding.white_point);
    if (ppf_in.color_encoding.color_space != JXL_COLOR_SPACE_GRAY) {
      EXPECT_EQ(ppf_in.color_encoding.primaries,
                ppf_out.color_encoding.primaries);
    }
    EXPECT_EQ(ppf_in.color_encoding.transfer_function,
              ppf_out.color_encoding.transfer_function);
    EXPECT_EQ(ppf_out.color_encoding.rendering_intent,
              JXL_RENDERING_INTENT_RELATIVE);
  } else if (params.codec != Codec::kPNM && params.codec != Codec::kPGX &&
             params.codec != Codec::kEXR) {
    EXPECT_EQ(ppf_in.icc, ppf_out.icc);
  }

  ASSERT_EQ(ppf_out.frames.size(), 1u);
  const auto& frame_in = ppf_in.frames[0];
  const auto& frame_out = ppf_out.frames[0];
  VerifySameImage(frame_in.color, ppf_in.info.bits_per_sample, frame_out.color,
                  ppf_out.info.bits_per_sample,
                  /*lossless=*/params.codec != Codec::kJPG);
  ASSERT_EQ(frame_in.extra_channels.size(), frame_out.extra_channels.size());
  ASSERT_EQ(ppf_out.extra_channels_info.size(),
            frame_out.extra_channels.size());
  for (size_t i = 0; i < frame_in.extra_channels.size(); ++i) {
    VerifySameImage(frame_in.extra_channels[i], ppf_in.info.bits_per_sample,
                    frame_out.extra_channels[i], ppf_out.info.bits_per_sample,
                    /*lossless=*/true);
    EXPECT_EQ(ppf_out.extra_channels_info[i].ec_info.type,
              ppf_in.extra_channels_info[i].ec_info.type);
  }
}

TEST(CodecTest, TestRoundTrip) {
  ThreadPoolForTests pool(12);

  TestImageParams params;
  params.xsize = 7;
  params.ysize = 4;

  for (Codec codec :
       {Codec::kPNG, Codec::kPNM, Codec::kPGX, Codec::kEXR, Codec::kJPG}) {
    for (int bits_per_sample : {4, 8, 10, 12, 16, 32}) {
      for (bool is_gray : {false, true}) {
        for (bool add_alpha : {false, true}) {
          for (bool big_endian : {false, true}) {
            params.codec = codec;
            params.bits_per_sample = static_cast<size_t>(bits_per_sample);
            params.is_gray = is_gray;
            params.add_alpha = add_alpha;
            params.big_endian = big_endian;
            params.add_extra_channels = false;
            TestRoundTrip(params, pool.get());
            if (codec == Codec::kPNM && add_alpha) {
              params.add_extra_channels = true;
              TestRoundTrip(params, pool.get());
            }
          }
        }
      }
    }
  }
}

TEST(CodecTest, LosslessPNMRoundtrip) {
  ThreadPoolForTests pool(12);

  static const char* kChannels[] = {"", "g", "ga", "rgb", "rgba"};
  static const char* kExtension[] = {"", ".pgm", ".pam", ".ppm", ".pam"};
  for (size_t bit_depth = 1; bit_depth <= 16; ++bit_depth) {
    for (size_t channels = 1; channels <= 4; ++channels) {
      if (bit_depth == 1 && (channels == 2 || channels == 4)) continue;
      std::string extension(kExtension[channels]);
      std::string filename = "jxl/flower/flower_small." +
                             std::string(kChannels[channels]) + ".depth" +
                             std::to_string(bit_depth) + extension;
      const std::vector<uint8_t> orig = jxl::test::ReadTestData(filename);

      PackedPixelFile ppf;
      ColorHints color_hints;
      color_hints.Add("color_space",
                      channels < 3 ? "Gra_D65_Rel_SRG" : "RGB_D65_SRG_Rel_SRG");
      ASSERT_TRUE(DecodeBytes(Bytes(orig), color_hints, &ppf));

      EncodedImage encoded;
      auto encoder = Encoder::FromExtension(extension);
      ASSERT_TRUE(encoder.get());
      ASSERT_TRUE(encoder->Encode(ppf, &encoded, pool.get()));
      ASSERT_EQ(encoded.bitstreams.size(), 1u);
      ASSERT_EQ(orig.size(), encoded.bitstreams[0].size());
      EXPECT_EQ(0,
                memcmp(orig.data(), encoded.bitstreams[0].data(), orig.size()));
    }
  }
}

TEST(CodecTest, TestPNM) {
  size_t u = 77777;  // Initialized to wrong value.
  double d = 77.77;
  const std::string max_unsigned =
      std::to_string(std::numeric_limits<size_t>::max());

// Failing to parse invalid strings results in a crash if `JXL_CRASH_ON_ERROR`
// is defined and hence the tests fail. Therefore we only run these tests if
// `JXL_CRASH_ON_ERROR` is not defined.
#if (!JXL_CRASH_ON_ERROR)
  ASSERT_FALSE(PnmParseUnsigned(MakeSpan(""), &u));
  ASSERT_FALSE(PnmParseUnsigned(MakeSpan("+"), &u));
  ASSERT_FALSE(PnmParseUnsigned(MakeSpan("-"), &u));
  ASSERT_FALSE(PnmParseUnsigned(MakeSpan("A"), &u));
  const std::string overflowing_unsigned = max_unsigned + "0";
  ASSERT_FALSE(PnmParseUnsigned(MakeSpan(overflowing_unsigned.c_str()), &u));

  ASSERT_FALSE(PnmParseSigned(MakeSpan(""), &d));
  ASSERT_FALSE(PnmParseSigned(MakeSpan("+"), &d));
  ASSERT_FALSE(PnmParseSigned(MakeSpan("-"), &d));
  ASSERT_FALSE(PnmParseSigned(MakeSpan("A"), &d));
#endif
  ASSERT_TRUE(PnmParseUnsigned(MakeSpan("1"), &u));
  ASSERT_TRUE(u == 1);

  ASSERT_TRUE(PnmParseUnsigned(MakeSpan("32"), &u));
  ASSERT_TRUE(u == 32);

  ASSERT_TRUE(PnmParseUnsigned(MakeSpan(max_unsigned.c_str()), &u));
  ASSERT_TRUE(u == std::numeric_limits<size_t>::max());

  ASSERT_TRUE(PnmParseSigned(MakeSpan("1"), &d));
  ASSERT_TRUE(d == 1.0);
  ASSERT_TRUE(PnmParseSigned(MakeSpan("+2"), &d));
  ASSERT_TRUE(d == 2.0);
  ASSERT_TRUE(PnmParseSigned(MakeSpan("-3"), &d));
  ASSERT_TRUE(std::abs(d - -3.0) < 1E-15);
  ASSERT_TRUE(PnmParseSigned(MakeSpan("3.141592"), &d));
  ASSERT_TRUE(std::abs(d - 3.141592) < 1E-15);
  ASSERT_TRUE(PnmParseSigned(MakeSpan("-3.141592"), &d));
  ASSERT_TRUE(std::abs(d - -3.141592) < 1E-15);
}

TEST(CodecTest, PNMSampleAboveMaxVal) {
  const auto decode = [](const std::string& header,
                         const std::vector<uint8_t>& samples) -> bool {
    std::vector<uint8_t> pnm(header.begin(), header.end());
    pnm.insert(pnm.end(), samples.begin(), samples.end());
    PackedPixelFile ppf;
    return static_cast<bool>(DecodeBytes(Bytes(pnm), ColorHints(), &ppf));
  };
  const std::string kPam12 =
      "P7\nWIDTH 1\nHEIGHT 1\nDEPTH 4\nMAXVAL 4095\n"
      "TUPLTYPE RGB_ALPHA\nENDHDR\n";

  // Samples up to MaxVal are valid.
  EXPECT_TRUE(decode("P5\n2 1\n15\n", {0, 15}));
  EXPECT_TRUE(decode("P5\n2 1\n255\n", {0, 255}));
  EXPECT_TRUE(decode("P6\n1 1\n4095\n", {0x0F, 0xFF, 0, 0, 0x0F, 0xFF}));
  EXPECT_TRUE(decode("P5\n1 1\n65535\n", {0xFF, 0xFF}));
  EXPECT_TRUE(decode(kPam12, {0, 0, 0, 0, 0, 0, 0x0F, 0xFF}));

#if (!JXL_CRASH_ON_ERROR)
  // Samples above MaxVal are rejected.
  EXPECT_FALSE(decode("P5\n2 1\n15\n", {0, 16}));
  EXPECT_FALSE(decode("P6\n1 1\n4095\n", {0x0F, 0xFF, 0x10, 0x00, 0, 0}));
  EXPECT_FALSE(decode("P6\n1 1\n4095\n", {0, 0, 0, 0, 0xFF, 0xFF}));
  EXPECT_FALSE(decode(kPam12, {0, 0, 0, 0, 0, 0, 0x10, 0x00}));
  // Check every sample position of images larger than 8 bytes.
  const std::vector<uint8_t> max4(19, 15);
  std::vector<uint8_t> max12;
  for (size_t i = 0; i < 19; ++i) max12.insert(max12.end(), {0x0F, 0xFF});
  EXPECT_TRUE(decode("P5\n19 1\n15\n", max4));
  EXPECT_TRUE(decode("P5\n19 1\n4095\n", max12));
  for (size_t i = 0; i < 19; ++i) {
    std::vector<uint8_t> samples4 = max4;
    samples4[i] = 16;
    EXPECT_FALSE(decode("P5\n19 1\n15\n", samples4)) << i;
    std::vector<uint8_t> samples12 = max12;
    samples12[2 * i] = 0x10;
    samples12[2 * i + 1] = 0x00;
    EXPECT_FALSE(decode("P5\n19 1\n4095\n", samples12)) << i;
  }
#endif
}

TEST(CodecTest, FormatNegotiation) {
  const std::vector<JxlPixelFormat> accepted_formats = {
      {/*num_channels=*/4,
       /*data_type=*/JXL_TYPE_UINT16,
       /*endianness=*/JXL_NATIVE_ENDIAN,
       /*align=*/0},
      {/*num_channels=*/3,
       /*data_type=*/JXL_TYPE_UINT8,
       /*endianness=*/JXL_NATIVE_ENDIAN,
       /*align=*/0},
      {/*num_channels=*/3,
       /*data_type=*/JXL_TYPE_UINT16,
       /*endianness=*/JXL_NATIVE_ENDIAN,
       /*align=*/0},
      {/*num_channels=*/1,
       /*data_type=*/JXL_TYPE_UINT8,
       /*endianness=*/JXL_NATIVE_ENDIAN,
       /*align=*/0},
  };

  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.bits_per_sample = 12;
  info.num_color_channels = 2;

  JxlPixelFormat format;
  EXPECT_FALSE(SelectFormat(accepted_formats, info, &format));

  info.num_color_channels = 3;
  ASSERT_TRUE(SelectFormat(accepted_formats, info, &format));
  EXPECT_EQ(format.num_channels, info.num_color_channels);
  // 16 is the smallest accepted format that can accommodate the 12-bit data.
  EXPECT_EQ(format.data_type, JXL_TYPE_UINT16);
}

TEST(CodecTest, FormatNegotiationGrayscalePromotion) {
  const std::vector<JxlPixelFormat> accepted_formats = {
      {/*num_channels=*/3,
       /*data_type=*/JXL_TYPE_UINT8,
       /*endianness=*/JXL_NATIVE_ENDIAN,
       /*align=*/0},
      {/*num_channels=*/3,
       /*data_type=*/JXL_TYPE_UINT16,
       /*endianness=*/JXL_NATIVE_ENDIAN,
       /*align=*/0},
  };

  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.bits_per_sample = 8;
  info.num_color_channels = 1;

  JxlPixelFormat format;
  ASSERT_TRUE(SelectFormat(accepted_formats, info, &format));
  EXPECT_EQ(format.num_channels, 3u);
  EXPECT_EQ(format.data_type, JXL_TYPE_UINT8);
}

TEST(CodecTest, FormatNegotiationGrayAlphaPromotion) {
  const std::vector<JxlPixelFormat> accepted_formats = {
      {/*num_channels=*/3,
       /*data_type=*/JXL_TYPE_UINT8,
       /*endianness=*/JXL_NATIVE_ENDIAN,
       /*align=*/0},
  };

  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.bits_per_sample = 8;
  info.num_color_channels = 1;
  info.alpha_bits = 8;

  JxlPixelFormat format;
  ASSERT_TRUE(SelectFormat(accepted_formats, info, &format));
  EXPECT_EQ(format.num_channels, 3u);
  EXPECT_EQ(format.data_type, JXL_TYPE_UINT8);
}

#if (!JXL_CRASH_ON_ERROR)
TEST(CodecTest, FormatNegotiationGrayscaleNoMatch) {
  const std::vector<JxlPixelFormat> accepted_formats = {
      {/*num_channels=*/5,
       /*data_type=*/JXL_TYPE_UINT8,
       /*endianness=*/JXL_NATIVE_ENDIAN,
       /*align=*/0},
  };

  JxlBasicInfo info;
  JxlEncoderInitBasicInfo(&info);
  info.bits_per_sample = 8;
  info.num_color_channels = 1;

  JxlPixelFormat format;
  EXPECT_FALSE(SelectFormat(accepted_formats, info, &format));
}
#endif

TEST(CodecTest, EncodeToPNG) {
  ThreadPool* const pool = nullptr;

  std::unique_ptr<Encoder> png_encoder = Encoder::FromExtension(".png");
  if (!png_encoder) {
    fprintf(stderr, "Skipping test because of missing codec support.\n");
    return;
  }

  const std::vector<uint8_t> original_png = jxl::test::ReadTestData(
      "external/wesaturate/500px/tmshre_riaphotographs_srgb8.png");
  PackedPixelFile ppf;
  ASSERT_TRUE(extras::DecodeBytes(Bytes(original_png), ColorHints(), &ppf));

  const JxlPixelFormat& format = ppf.frames.front().color.format;
  const auto& format_matcher = [&format](const JxlPixelFormat& candidate) {
    return (candidate.num_channels == format.num_channels) &&
           (candidate.data_type == format.data_type) &&
           (candidate.endianness == format.endianness);
  };
  const auto formats = png_encoder->AcceptedFormats();
  ASSERT_TRUE(std::any_of(formats.begin(), formats.end(), format_matcher));
  EncodedImage encoded_png;
  ASSERT_TRUE(png_encoder->Encode(ppf, &encoded_png, pool));
  EXPECT_TRUE(encoded_png.icc.empty());
  ASSERT_EQ(encoded_png.bitstreams.size(), 1u);

  PackedPixelFile decoded_ppf;
  ASSERT_TRUE(extras::DecodeBytes(Bytes(encoded_png.bitstreams.front()),
                                  ColorHints(), &decoded_ppf));

  ASSERT_EQ(decoded_ppf.info.bits_per_sample, ppf.info.bits_per_sample);
  ASSERT_EQ(decoded_ppf.frames.size(), 1u);
  VerifySameImage(ppf.frames[0].color, ppf.info.bits_per_sample,
                  decoded_ppf.frames[0].color,
                  decoded_ppf.info.bits_per_sample);
}

TEST(CodecTest, ExtensionFromPath) {
  // The extension is derived from the final path component only: dots in
  // directory names must not leak into the result, the returned extension
  // never contains a path separator, and a trailing dot that carries no
  // extension is reported as none.
  std::string extension;
  size_t bits_per_sample = 0;

  extension.clear();
  ASSERT_EQ(CodecFromPath("a.png/../../b", &bits_per_sample, &extension),
            Codec::kUnknown);
  EXPECT_EQ(extension, "");

  // A valid codec extension in the final component is still detected even
  // with a dotted directory part ahead of it.
  extension.clear();
  ASSERT_EQ(CodecFromPath("a.jpg/../../b.jpg", &bits_per_sample, &extension),
            Codec::kJPG);
  EXPECT_EQ(extension, ".jpg");

  extension.clear();
  ASSERT_EQ(CodecFromPath("dir.name/file", &bits_per_sample, &extension),
            Codec::kUnknown);
  EXPECT_EQ(extension, "");

  extension.clear();
  ASSERT_EQ(CodecFromPath("out.png.", &bits_per_sample, &extension),
            Codec::kUnknown);
  EXPECT_EQ(extension, "");

  // Backslash separators are stripped as well.
  extension.clear();
  ASSERT_EQ(CodecFromPath("C:\\x\\y.png", &bits_per_sample, &extension),
            Codec::kPNG);
  EXPECT_EQ(extension, ".png");

  extension.clear();
  ASSERT_EQ(CodecFromPath("/dev/stdout", &bits_per_sample, &extension),
            Codec::kUnknown);
  EXPECT_EQ(extension, "");

  // Bare extension strings (as passed by the benchmark tooling) keep
  // working, including the .pfm bits_per_sample side effect.
  extension.clear();
  ASSERT_EQ(CodecFromPath(".pfm", &bits_per_sample, &extension), Codec::kPNM);
  EXPECT_EQ(extension, ".pfm");
  EXPECT_EQ(bits_per_sample, 32u);
}

TEST(CodecTest, DecodeTruncatedPNGLeavesMissingRowsZero) {
  std::unique_ptr<Encoder> png_encoder = Encoder::FromExtension(".png");
  if (!png_encoder || !CanDecode(Codec::kPNG)) {
    fprintf(stderr, "Skipping test because of missing codec support.\n");
    return;
  }

  constexpr size_t kXSize = 64;
  constexpr size_t kYSize = 64;
  TestImageParams params = {Codec::kPNG,
                            kXSize,
                            kYSize,
                            /*bits_per_sample=*/8,
                            /*is_gray=*/false,
                            /*add_alpha=*/false,
                            /*big_endian=*/true,
                            /*add_extra_channels=*/false};
  PackedPixelFile ppf_in;
  CreateTestImage(params, &ppf_in);
  EncodedImage encoded;
  ASSERT_TRUE(png_encoder->Encode(ppf_in, &encoded, nullptr));
  ASSERT_EQ(encoded.bitstreams.size(), 1u);
  std::vector<uint8_t> png = encoded.bitstreams[0];

  // Declare twice the height in IHDR (the chunk CRC is not checked), so the
  // pixel data ends after half of the rows.
  ASSERT_EQ(memcmp(&png[12], "IHDR", 4), 0);
  ASSERT_EQ(LoadBE32(&png[20]), kYSize);
  StoreBE32(2 * kYSize, &png[20]);

  // Fill freed heap blocks of the size of the pixel buffer, so that a buffer
  // that is not cleared holds stale bytes instead of fresh zero pages.
  for (size_t i = 0; i < 4; ++i) {
    std::vector<uint8_t> dirty(kXSize * 2 * kYSize * 3, 0xA5);
    EXPECT_EQ(dirty.back(), 0xA5);
  }

  PackedPixelFile ppf_out;
  ASSERT_TRUE(DecodeBytes(Bytes(png), ColorHints(), &ppf_out));
  ASSERT_EQ(ppf_out.frames.size(), 1u);
  const PackedImage& in = ppf_in.frames[0].color;
  const PackedImage& out = ppf_out.frames[0].color;
  ASSERT_EQ(out.xsize, kXSize);
  ASSERT_EQ(out.ysize, 2 * kYSize);
  ASSERT_EQ(out.pixels_size, 2 * in.pixels_size);

  // The rows in the file are decoded, the rows missing from it are zero.
  const auto* in_pixels = static_cast<const uint8_t*>(in.pixels());
  const auto* out_pixels = static_cast<const uint8_t*>(out.pixels());
  EXPECT_EQ(memcmp(in_pixels, out_pixels, in.pixels_size), 0);
  for (size_t i = in.pixels_size; i < out.pixels_size; ++i) {
    ASSERT_EQ(out_pixels[i], 0) << "byte " << i;
  }
}

// Frame names are arbitrary bytes from the codestream. The NumPy metadata
// writer must produce valid JSON for them: a raw '"' in a name used to
// terminate the JSON string early, which both broke parsing and allowed a
// crafted name to inject additional fields into the frame dictionary.
TEST(CodecTest, NumPyMetadataEscapesFrameName) {
  std::unique_ptr<Encoder> npy_encoder = Encoder::FromExtension(".npy");
  ASSERT_TRUE(npy_encoder);

  const JxlPixelFormat format{/*num_channels=*/1, JXL_TYPE_FLOAT,
                              JXL_LITTLE_ENDIAN, /*align=*/0};
  PackedPixelFile ppf;
  ppf.info.xsize = 2;
  ppf.info.ysize = 2;
  ppf.info.num_color_channels = 1;
  ppf.info.bits_per_sample = 32;
  ppf.info.exponent_bits_per_sample = 8;
  JXL_TEST_ASSIGN_OR_DIE(PackedFrame frame, PackedFrame::Create(2, 2, format));
  memset(frame.color.pixels(), 0, frame.color.pixels_size);
  frame.name = std::string("q\"b\\n\ne\x01", 8);
  ppf.frames.emplace_back(std::move(frame));

  EncodedImage encoded;
  ASSERT_TRUE(npy_encoder->Encode(ppf, &encoded, nullptr));
  const std::string metadata(encoded.metadata.begin(), encoded.metadata.end());
  EXPECT_NE(metadata.find("\"name\": \"q\\\"b\\\\n\\ne\\u0001\""),
            std::string::npos)
      << metadata;
  // A quote inside the name must never end the JSON string.
  EXPECT_EQ(metadata.find("\"q\"b"), std::string::npos) << metadata;
}
}  // namespace
}  // namespace extras
}  // namespace jxl
