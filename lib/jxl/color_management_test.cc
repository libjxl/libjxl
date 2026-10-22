// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>
#include <jxl/cms_interface.h>
#include <jxl/memory_manager.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/cms/color_encoding_cms.h"
#include "lib/jxl/cms/opsin_params.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/test_image.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {

std::ostream& operator<<(std::ostream& os, const CIExy& xy) {
  return os << "{x=" << xy.x << ", y=" << xy.y << "}";
}

std::ostream& operator<<(std::ostream& os, const PrimariesCIExy& primaries) {
  return os << "{r=" << primaries.r << ", g=" << primaries.g
            << ", b=" << primaries.b << "}";
}

namespace {

// Small enough to be fast. If changed, must update Generate*.
constexpr size_t kWidth = 16;

constexpr size_t kNumThreads = 1;  // only have a single row.

size_t CountDistinctColors(const extras::PackedImage& image, size_t limit) {
  std::set<std::array<uint16_t, 3>> colors;
  for (size_t y = 0; y < image.ysize; ++y) {
    for (size_t x = 0; x < image.xsize; ++x) {
      std::array<uint16_t, 3> color;
      for (size_t c = 0; c < color.size(); ++c) {
        color[c] = static_cast<uint16_t>(std::lround(
            Clamp1(image.GetPixelValue(y, x, c), 0.0f, 1.0f) * 65535.0f));
      }
      colors.insert(color);
      if (colors.size() > limit) return colors.size();
    }
  }
  return colors.size();
}

uint32_t ReadBE32(const uint8_t* bytes) {
  return (static_cast<uint32_t>(bytes[0]) << 24) |
         (static_cast<uint32_t>(bytes[1]) << 16) |
         (static_cast<uint32_t>(bytes[2]) << 8) |
         static_cast<uint32_t>(bytes[3]);
}

void WriteBE32(uint8_t* bytes, uint32_t value) {
  bytes[0] = static_cast<uint8_t>(value >> 24);
  bytes[1] = static_cast<uint8_t>(value >> 16);
  bytes[2] = static_cast<uint8_t>(value >> 8);
  bytes[3] = static_cast<uint8_t>(value);
}

void WriteBE16(uint8_t* bytes, uint16_t value) {
  bytes[0] = static_cast<uint8_t>(value >> 8);
  bytes[1] = static_cast<uint8_t>(value);
}

size_t FindIccTagRecord(const IccBytes& icc, const char* signature) {
  if (icc.size() < 132U) return 0;
  const uint32_t tag_count = ReadBE32(icc.data() + 128);
  if (tag_count > (icc.size() - 132U) / 12U) return 0;
  for (uint32_t i = 0; i < tag_count; ++i) {
    const size_t record = 132U + 12U * i;
    if (std::memcmp(icc.data() + record, signature, 4) == 0) return record;
  }
  return 0;
}

bool HasIccTag(const IccBytes& icc, const char* signature) {
  return FindIccTagRecord(icc, signature) != 0;
}

StatusOr<IccBytes> CreateDisplayP3MatrixProfile() {
  ColorEncoding encoding;
  encoding.SetColorSpace(ColorSpace::kRGB);
  JXL_RETURN_IF_ERROR(encoding.SetWhitePointType(WhitePoint::kD65));
  JXL_RETURN_IF_ERROR(encoding.SetPrimariesType(Primaries::kP3));
  encoding.Tf().SetTransferFunction(TransferFunction::kSRGB);
  encoding.SetRenderingIntent(RenderingIntent::kRelative);
  JXL_RETURN_IF_ERROR(encoding.CreateICC());

  IccBytes icc = encoding.ICC();
  const size_t cicp_record = FindIccTagRecord(icc, "cicp");
  JXL_ENSURE(cicp_record != 0);
  std::memcpy(icc.data() + cicp_record, "tst0", 4);
  // A zero profile ID is valid and avoids retaining the checksum after the
  // intentional test mutation.
  std::fill(icc.begin() + 84, icc.begin() + 100, 0);
  return icc;
}

StatusOr<IccBytes> CreateNonconformingDisplayP3Profile() {
  JXL_ASSIGN_OR_RETURN(IccBytes icc, CreateDisplayP3MatrixProfile());

  const size_t wtpt_record = FindIccTagRecord(icc, "wtpt");
  JXL_ENSURE(wtpt_record != 0);
  const size_t wtpt_offset = ReadBE32(icc.data() + wtpt_record + 4);
  const size_t wtpt_size = ReadBE32(icc.data() + wtpt_record + 8);
  JXL_ENSURE(wtpt_offset <= icc.size());
  JXL_ENSURE(wtpt_size >= 20U && wtpt_size <= icc.size() - wtpt_offset);
  JXL_ENSURE(std::memcmp(icc.data() + wtpt_offset, "XYZ ", 4) == 0);

  // ICC v4 display profiles normally store D50 here. This deliberately stores
  // D65 while retaining CHAD and D50-adapted matrix columns, matching the
  // malformed profile shape that triggered issue #4911.
  constexpr std::array<double, 3> kD65XYZ = {0.3127 / 0.3290, 1.0,
                                             (1.0 - 0.3127 - 0.3290) / 0.3290};
  for (size_t i = 0; i < kD65XYZ.size(); ++i) {
    WriteBE32(icc.data() + wtpt_offset + 8U + 4U * i,
              static_cast<uint32_t>(
                  static_cast<int32_t>(std::lround(kD65XYZ[i] * 65536.0))));
  }
  return icc;
}

StatusOr<test::TestImage> CreateNonconformingDisplayP3TestImage(
    const IccBytes& icc, size_t size, size_t bits_per_sample,
    size_t alpha_pattern) {
  test::TestImage image;
  JXL_RETURN_IF_ERROR(image.SetDimensions(size, size));
  JXL_RETURN_IF_ERROR(image.SetChannels(4));
  image.SetAllBitDepths(bits_per_sample);
  JXL_RETURN_IF_ERROR(image.SetColorEncoding("DisplayP3"));
  JXL_ASSIGN_OR_RETURN(auto frame, image.AddFrame());
  for (size_t y = 0; y < size; ++y) {
    for (size_t x = 0; x < size; ++x) {
      const double ramp = static_cast<double>(x + y * size) /
                          static_cast<double>(size * size - 1);
      JXL_RETURN_IF_ERROR(frame.SetValue(y, x, 0, 0.945 + 0.055 * ramp));
      JXL_RETURN_IF_ERROR(frame.SetValue(y, x, 1, 2.0 * ramp / 65535.0));
      JXL_RETURN_IF_ERROR(frame.SetValue(y, x, 2, 8.0 * ramp / 65535.0));
      const double alpha = alpha_pattern == 0   ? 1.0
                           : alpha_pattern == 1 ? (x + y) % 2
                                                : 0.25 + 0.75 * ramp;
      JXL_RETURN_IF_ERROR(frame.SetValue(y, x, 3, alpha));
    }
  }
  image.ppf().icc.assign(icc.begin(), icc.end());
  image.ppf().orig_icc.assign(icc.begin(), icc.end());
  image.ppf().primary_color_representation =
      extras::PackedPixelFile::kIccIsPrimary;
  return image;
}

bool ReplaceIccTag(IccBytes* target, const char* target_signature,
                   const char* new_signature, const IccBytes& source,
                   const char* source_signature) {
  const size_t target_record = FindIccTagRecord(*target, target_signature);
  const size_t source_record = FindIccTagRecord(source, source_signature);
  if (target_record == 0 || source_record == 0) return false;
  const size_t source_offset = ReadBE32(source.data() + source_record + 4);
  const size_t source_size = ReadBE32(source.data() + source_record + 8);
  if (source_offset > source.size() ||
      source_size > source.size() - source_offset) {
    return false;
  }
  const size_t target_offset = (target->size() + 3U) & ~size_t{3};
  target->resize(target_offset + source_size);
  std::copy(source.begin() + source_offset,
            source.begin() + source_offset + source_size,
            target->begin() + target_offset);
  std::memcpy(target->data() + target_record, new_signature, 4);
  WriteBE32(target->data() + target_record + 4,
            static_cast<uint32_t>(target_offset));
  WriteBE32(target->data() + target_record + 8,
            static_cast<uint32_t>(source_size));
  WriteBE32(target->data(), static_cast<uint32_t>(target->size()));
  return true;
}

bool ReplaceTrcsWithSampledCurve(IccBytes* icc,
                                 bool make_non_equivalent = true) {
  constexpr size_t kEntries = 256;
  IccBytes curve(12U + 2U * kEntries);
  std::memcpy(curve.data(), "curv", 4);
  WriteBE32(curve.data() + 8, kEntries);
  for (size_t i = 0; i < kEntries; ++i) {
    const double x = static_cast<double>(i) / (kEntries - 1);
    double y = x <= 0.04045 ? x / 12.92 : std::pow((x + 0.055) / 1.055, 2.4);
    if (make_non_equivalent && i == 25) y += 0.25;
    WriteBE16(
        curve.data() + 12U + 2U * i,
        static_cast<uint16_t>(std::lround(Clamp1(y, 0.0, 1.0) * 65535.0)));
  }

  const size_t offset = (icc->size() + 3U) & ~size_t{3};
  icc->resize(offset + curve.size());
  std::copy(curve.begin(), curve.end(), icc->begin() + offset);
  for (const char* signature : {"rTRC", "gTRC", "bTRC"}) {
    const size_t record = FindIccTagRecord(*icc, signature);
    if (record == 0) return false;
    WriteBE32(icc->data() + record + 4, static_cast<uint32_t>(offset));
    WriteBE32(icc->data() + record + 8, static_cast<uint32_t>(curve.size()));
  }
  WriteBE32(icc->data(), static_cast<uint32_t>(icc->size()));
  return true;
}

bool ModifiedTrcRemainsICC(const IccBytes& icc, size_t trc_offset,
                           uint32_t delta) {
  IccBytes modified_icc = icc;
  WriteBE32(modified_icc.data() + trc_offset + 16,
            ReadBE32(modified_icc.data() + trc_offset + 16) + delta);
  ColorEncoding modified;
  if (!modified.SetICC(std::move(modified_icc), JxlGetDefaultCms())) {
    return false;
  }
  modified.DecideIfWantICC(*JxlGetDefaultCms());
  return modified.WantICC();
}

void CheckNonconformingDisplayP3FullSizeLossy(
    const extras::PackedPixelFile& input, size_t distinct_color_limit) {
  extras::JXLCompressParams params;
  params.distance = 1.0f;
  extras::JXLDecompressParams decode_params;
  decode_params.accepted_formats.push_back(
      {4, input.frames[0].color.format.data_type,
       input.frames[0].color.format.endianness, /*align=*/0});
  extras::PackedPixelFile output;
  ASSERT_GT(test::Roundtrip(input, params, decode_params, nullptr, &output),
            0U);
  const uint32_t output_bits =
      input.frames[0].color.format.data_type == JXL_TYPE_UINT8 ? 8U : 16U;
  EXPECT_EQ(output.info.bits_per_sample, output_bits);
  EXPECT_EQ(output.primary_color_representation,
            extras::PackedPixelFile::kColorEncodingIsPrimary);
  EXPECT_GT(CountDistinctColors(output.frames[0].color, distinct_color_limit),
            distinct_color_limit);
  // Butteraugli converts both inputs through their declared color encodings,
  // so this also checks color fidelity in a common viewing space.
  EXPECT_LE(test::ButteraugliDistance(input, output), 1.5f);
}

void CheckNonconformingDisplayP3Tolerance(
    const extras::PackedPixelFile& input) {
  ASSERT_GE(input.icc.size(), 132U);
  const uint32_t tag_count = ReadBE32(input.icc.data() + 128);
  ASSERT_LE(tag_count, (input.icc.size() - 132U) / 12U);
  size_t trc_offset = 0;
  size_t trc_size = 0;
  for (uint32_t i = 0; i < tag_count; ++i) {
    const size_t record = 132U + 12U * i;
    if (std::memcmp(input.icc.data() + record, "rTRC", 4) == 0) {
      trc_offset = ReadBE32(input.icc.data() + record + 4);
      trc_size = ReadBE32(input.icc.data() + record + 8);
      break;
    }
  }
  ASSERT_GT(trc_offset, 0U);
  ASSERT_LE(trc_offset + trc_size, input.icc.size());
  ASSERT_GE(trc_size, 20U);
  ASSERT_EQ(std::memcmp(input.icc.data() + trc_offset, "para", 4), 0);
  // Keep the v4 matrix/CHAD/white-point shape but make the shared TRC clearly
  // non-equivalent. Both CMS implementations must retain the ICC profile.
  EXPECT_TRUE(ModifiedTrcRemainsICC(input.icc, trc_offset, 4096));
#if !JPEGXL_ENABLE_SKCMS
  // In the lcms path, adding 16 fixed-point LSBs is a small mutation rejected
  // by the relaxed 3E-4 functional-equivalence comparison.
  EXPECT_TRUE(ModifiedTrcRemainsICC(input.icc, trc_offset, 16));
#endif
}

struct Globals {
  // TODO(deymo): Make this a const.
  static Globals* GetInstance() {
    static Globals ret;
    return &ret;
  }

 private:
  Globals() {
    JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
    in_gray = GenerateGray();
    in_color = GenerateColor();
    JXL_TEST_ASSIGN_OR_DIE(out_gray, ImageF::Create(memory_manager, kWidth, 1));
    JXL_TEST_ASSIGN_OR_DIE(out_color,
                           ImageF::Create(memory_manager, kWidth * 3, 1));

    c_native = ColorEncoding::LinearSRGB(/*is_gray=*/false);
    c_gray = ColorEncoding::LinearSRGB(/*is_gray=*/true);
  }

  static ImageF GenerateGray() {
    JXL_TEST_ASSIGN_OR_DIE(
        ImageF gray, ImageF::Create(jxl::test::MemoryManager(), kWidth, 1));
    float* JXL_RESTRICT row = gray.Row(0);
    // Increasing left to right
    for (uint32_t x = 0; x < kWidth; ++x) {
      row[x] = x * 1.0f / (kWidth - 1);  // [0, 1]
    }
    return gray;
  }

  static ImageF GenerateColor() {
    JXL_TEST_ASSIGN_OR_DIE(
        ImageF image,
        ImageF::Create(jxl::test::MemoryManager(), kWidth * 3, 1));
    float* JXL_RESTRICT interleaved = image.Row(0);
    std::fill(interleaved, interleaved + kWidth * 3, 0.0f);

    // [0, 4): neutral
    for (int32_t x = 0; x < 4; ++x) {
      interleaved[3 * x + 0] = x * 1.0f / 3;  // [0, 1]
      interleaved[3 * x + 2] = interleaved[3 * x + 1] = interleaved[3 * x + 0];
    }

    // [4, 13): pure RGB with low/medium/high saturation
    for (int32_t c = 0; c < 3; ++c) {
      interleaved[3 * (4 + c) + c] = 0.08f + c * 0.01f;
      interleaved[3 * (7 + c) + c] = 0.75f + c * 0.01f;
      interleaved[3 * (10 + c) + c] = 1.0f;
    }

    // [13, 16): impure, not quite saturated RGB
    interleaved[3 * 13 + 0] = 0.86f;
    interleaved[3 * 13 + 2] = interleaved[3 * 13 + 1] = 0.16f;
    interleaved[3 * 14 + 1] = 0.87f;
    interleaved[3 * 14 + 2] = interleaved[3 * 14 + 0] = 0.16f;
    interleaved[3 * 15 + 2] = 0.88f;
    interleaved[3 * 15 + 1] = interleaved[3 * 15 + 0] = 0.16f;

    return image;
  }

 public:
  // ImageF so we can use VerifyRelativeError; all are interleaved RGB.
  ImageF in_gray;
  ImageF in_color;
  ImageF out_gray;
  ImageF out_color;
  ColorEncoding c_native;
  ColorEncoding c_gray;
};

class ColorManagementTest
    : public ::testing::TestWithParam<test::ColorEncodingDescriptor> {
 public:
  // "Same" pixels after converting g->c_native -> c -> g->c_native.
  static void VerifyPixelRoundTrip(const ColorEncoding& c) {
    Globals* g = Globals::GetInstance();
    const ColorEncoding& c_native = c.IsGray() ? g->c_gray : g->c_native;
    const JxlCmsInterface& cms = *JxlGetDefaultCms();
    ColorSpaceTransform xform_fwd(cms);
    ColorSpaceTransform xform_rev(cms);
    const float intensity_target =
        c.Tf().IsHLG() ? 1000 : kDefaultIntensityTarget;
    ASSERT_TRUE(
        xform_fwd.Init(c_native, c, intensity_target, kWidth, kNumThreads));
    ASSERT_TRUE(
        xform_rev.Init(c, c_native, intensity_target, kWidth, kNumThreads));

    const size_t thread = 0;
    const ImageF& in = c.IsGray() ? g->in_gray : g->in_color;
    ImageF* JXL_RESTRICT out = c.IsGray() ? &g->out_gray : &g->out_color;
    ASSERT_TRUE(
        xform_fwd.Run(thread, in.Row(0), xform_fwd.BufDst(thread), kWidth));
    ASSERT_TRUE(
        xform_rev.Run(thread, xform_fwd.BufDst(thread), out->Row(0), kWidth));

    // With lcms2, this value is lower: 5E-5
    double max_l1 = 7E-4;
    // Most are lower; reached 3E-7 with D60 AP0.
    double max_rel = 4E-7;
    if (c.IsGray()) max_rel = 2E-5;
    JXL_TEST_ASSERT_OK(VerifyRelativeError(in, *out, max_l1, max_rel, _));
  }
};
JXL_GTEST_INSTANTIATE_TEST_SUITE_P(ColorManagementTestInstantiation,
                                   ColorManagementTest,
                                   ::testing::ValuesIn(test::AllEncodings()));

// Exercises the ColorManagement interface for ALL ColorEncoding synthesizable
// via enums.
TEST_P(ColorManagementTest, VerifyAllProfiles) {
  ColorEncoding actual = ColorEncodingFromDescriptor(GetParam());
  printf("%s\n", Description(actual).c_str());

  // Can create profile.
  ASSERT_TRUE(actual.CreateICC());

  // Can set an equivalent ColorEncoding from the generated ICC profile.
  ColorEncoding expected;
  ASSERT_TRUE(expected.SetICC(IccBytes(actual.ICC()), JxlGetDefaultCms()));

  EXPECT_EQ(actual.GetRenderingIntent(), expected.GetRenderingIntent())
      << "different rendering intent: " << ToString(actual.GetRenderingIntent())
      << " instead of " << ToString(expected.GetRenderingIntent());
  EXPECT_EQ(actual.GetColorSpace(), expected.GetColorSpace())
      << "different color space: " << ToString(actual.GetColorSpace())
      << " instead of " << ToString(expected.GetColorSpace());
  EXPECT_EQ(actual.GetWhitePointType(), expected.GetWhitePointType())
      << "different white point: " << ToString(actual.GetWhitePointType())
      << " instead of " << ToString(expected.GetWhitePointType());
  EXPECT_EQ(actual.HasPrimaries(), expected.HasPrimaries());
  if (actual.HasPrimaries()) {
    EXPECT_EQ(actual.GetPrimariesType(), expected.GetPrimariesType())
        << "different primaries: " << ToString(actual.GetPrimariesType())
        << " instead of " << ToString(expected.GetPrimariesType());
  }

  static const auto tf_to_string =
      [](const jxl::cms::CustomTransferFunction& tf) {
        if (tf.have_gamma) {
          return "g" + ToString(tf.GetGamma());
        }
        return ToString(tf.transfer_function);
      };
  EXPECT_TRUE(actual.Tf().IsSame(expected.Tf()))
      << "different transfer function: " << tf_to_string(actual.Tf())
      << " instead of " << tf_to_string(expected.Tf());

  VerifyPixelRoundTrip(actual);
}

#define EXPECT_CIEXY_NEAR(A, E, T)                                       \
  {                                                                      \
    CIExy _actual = (A);                                                 \
    CIExy _expected = (E);                                               \
    double _tolerance = (T);                                             \
    EXPECT_NEAR(_actual.x, _expected.x, _tolerance) << "x is different"; \
    EXPECT_NEAR(_actual.y, _expected.y, _tolerance) << "y is different"; \
  }

#define EXPECT_PRIMARIES_NEAR(A, E, T)                                         \
  {                                                                            \
    PrimariesCIExy _actual = (A);                                              \
    PrimariesCIExy _expected = (E);                                            \
    double _tolerance = (T);                                                   \
    EXPECT_NEAR(_actual.r.x, _expected.r.x, _tolerance) << "r.x is different"; \
    EXPECT_NEAR(_actual.r.y, _expected.r.y, _tolerance) << "r.y is different"; \
    EXPECT_NEAR(_actual.g.x, _expected.g.x, _tolerance) << "g.x is different"; \
    EXPECT_NEAR(_actual.g.y, _expected.g.y, _tolerance) << "g.y is different"; \
    EXPECT_NEAR(_actual.b.x, _expected.b.x, _tolerance) << "b.x is different"; \
    EXPECT_NEAR(_actual.b.y, _expected.b.y, _tolerance) << "b.y is different"; \
  }

TEST_F(ColorManagementTest, sRGBChromaticity) {
  const ColorEncoding sRGB = ColorEncoding::SRGB();
  EXPECT_CIEXY_NEAR(sRGB.GetWhitePoint(), CIExy(0.3127, 0.3290), 1e-4);
  PrimariesCIExy srgb_primaries = {{0.64, 0.33}, {0.30, 0.60}, {0.15, 0.06}};
  PrimariesCIExy p;
  ASSERT_TRUE(sRGB.GetPrimaries(p));
  EXPECT_PRIMARIES_NEAR(p, srgb_primaries, 1e-4);
}

TEST_F(ColorManagementTest, D2700Chromaticity) {
  std::vector<uint8_t> icc_data =
      jxl::test::ReadTestData("jxl/color_management/sRGB-D2700.icc");
  IccBytes icc;
  Bytes(icc_data).AppendTo(icc);
  ColorEncoding sRGB_D2700;
  ASSERT_TRUE(sRGB_D2700.SetICC(std::move(icc), JxlGetDefaultCms()));

  EXPECT_CIEXY_NEAR(sRGB_D2700.GetWhitePoint(), CIExy(0.45986, 0.41060), 1e-4);
  // The illuminant-relative chromaticities of this profile's primaries are the
  // same as for sRGB. It is the PCS-relative chromaticities that would be
  // different.
  PrimariesCIExy srgb_primaries = {{0.64, 0.33}, {0.30, 0.60}, {0.15, 0.06}};
  PrimariesCIExy p;
  ASSERT_TRUE(sRGB_D2700.GetPrimaries(p));
  EXPECT_PRIMARIES_NEAR(p, srgb_primaries, 1e-4);
}

TEST_F(ColorManagementTest, D2700ToSRGB) {
  std::vector<uint8_t> icc_data =
      jxl::test::ReadTestData("jxl/color_management/sRGB-D2700.icc");
  IccBytes icc;
  Bytes(icc_data).AppendTo(icc);
  ColorEncoding sRGB_D2700;
  ASSERT_TRUE(sRGB_D2700.SetICC(std::move(icc), JxlGetDefaultCms()));

  ColorSpaceTransform transform(*JxlGetDefaultCms());
  ASSERT_TRUE(transform.Init(sRGB_D2700, ColorEncoding::SRGB(),
                             kDefaultIntensityTarget, 1, 1));
  Color sRGB_D2700_values{0.863, 0.737, 0.490};
  Color sRGB_values;
  ASSERT_TRUE(
      transform.Run(0, sRGB_D2700_values.data(), sRGB_values.data(), 1));
  Color sRGB_expected{0.914, 0.745, 0.601};
  EXPECT_ARRAY_NEAR(sRGB_values, sRGB_expected, 1e-3);
}

TEST_F(ColorManagementTest, NonconformingDisplayP3DoesNotCollapseWithLossyXYB) {
  JXL_TEST_ASSIGN_OR_DIE(const IccBytes expected_icc,
                         CreateNonconformingDisplayP3Profile());
  JXL_TEST_ASSIGN_OR_DIE(test::TestImage test_image,
                         CreateNonconformingDisplayP3TestImage(
                             expected_icc, 256, 16, /*alpha_pattern=*/0));
  extras::PackedPixelFile& input = test_image.ppf();
  ASSERT_EQ(input.primary_color_representation,
            extras::PackedPixelFile::kIccIsPrimary);
  EXPECT_EQ(input.icc, expected_icc);
  EXPECT_EQ(input.info.bits_per_sample, 16U);
  EXPECT_EQ(input.info.alpha_bits, 16U);

  {
    ColorEncoding parsed;
    ASSERT_TRUE(parsed.SetICC(IccBytes(input.icc), JxlGetDefaultCms()));
    ASSERT_TRUE(parsed.HasPrimaries());
    EXPECT_EQ(parsed.GetWhitePointType(), WhitePoint::kD65);
    EXPECT_EQ(parsed.GetPrimariesType(), Primaries::kP3);
    EXPECT_TRUE(parsed.Tf().IsSRGB());
  }

  ASSERT_NO_FATAL_FAILURE(CheckNonconformingDisplayP3FullSizeLossy(input, 100));
  ASSERT_NO_FATAL_FAILURE(CheckNonconformingDisplayP3Tolerance(input));
}

TEST_F(ColorManagementTest, NonconformingDisplayP3InputVariants) {
  JXL_TEST_ASSIGN_OR_DIE(const IccBytes icc,
                         CreateNonconformingDisplayP3Profile());
  struct Variant {
    size_t bits_per_sample;
    size_t alpha_pattern;
  };
  for (const Variant& variant :
       {Variant{8, 0}, Variant{10, 1}, Variant{12, 2}}) {
    SCOPED_TRACE(testing::Message()
                 << "bits=" << variant.bits_per_sample
                 << " alpha_pattern=" << variant.alpha_pattern);
    JXL_TEST_ASSIGN_OR_DIE(
        test::TestImage test_image,
        CreateNonconformingDisplayP3TestImage(icc, 64, variant.bits_per_sample,
                                              variant.alpha_pattern));
    ASSERT_NO_FATAL_FAILURE(
        CheckNonconformingDisplayP3FullSizeLossy(test_image.ppf(), 4));
  }

  JXL_TEST_ASSIGN_OR_DIE(
      test::TestImage lossless_image,
      CreateNonconformingDisplayP3TestImage(icc, 64, 16, /*alpha_pattern=*/2));
  extras::JXLCompressParams params;
  params.distance = 0.0f;
  extras::JXLDecompressParams decode_params;
  decode_params.accepted_formats.push_back(
      {4, lossless_image.ppf().frames[0].color.format.data_type,
       lossless_image.ppf().frames[0].color.format.endianness, /*align=*/0});
  extras::PackedPixelFile output;
  ASSERT_GT(test::Roundtrip(lossless_image.ppf(), params, decode_params,
                            nullptr, &output),
            0U);
  ASSERT_EQ(output.frames.size(), 1U);
  EXPECT_TRUE(test::SamePixels(lossless_image.ppf().frames[0].color,
                               output.frames[0].color));
  EXPECT_EQ(output.primary_color_representation,
            extras::PackedPixelFile::kIccIsPrimary);
}

TEST_F(ColorManagementTest, ConformingDisplayP3ProfileUsesStructuredFields) {
  JXL_TEST_ASSIGN_OR_DIE(IccBytes conforming_icc,
                         CreateDisplayP3MatrixProfile());
  ASSERT_GE(conforming_icc.size(), 128U);
  EXPECT_EQ(conforming_icc[8], 4U);
  EXPECT_EQ(std::memcmp(conforming_icc.data() + 12, "mntr", 4), 0);
  EXPECT_EQ(std::memcmp(conforming_icc.data() + 16, "RGB ", 4), 0);
  EXPECT_EQ(std::memcmp(conforming_icc.data() + 20, "XYZ ", 4), 0);
  for (const char* tag :
       {"wtpt", "chad", "rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC"}) {
    EXPECT_TRUE(HasIccTag(conforming_icc, tag));
  }
  EXPECT_FALSE(HasIccTag(conforming_icc, "A2B0"));
  ColorEncoding actual;
  ASSERT_TRUE(actual.SetICC(std::move(conforming_icc), JxlGetDefaultCms()));
  actual.DecideIfWantICC(*JxlGetDefaultCms());
  EXPECT_FALSE(actual.WantICC());
  EXPECT_EQ(actual.GetWhitePointType(), WhitePoint::kD65);
  EXPECT_EQ(actual.GetPrimariesType(), Primaries::kP3);
  EXPECT_TRUE(actual.Tf().IsSRGB());
}

TEST_F(ColorManagementTest, CustomChadMatrixProfileRemainsICC) {
  JXL_TEST_ASSIGN_OR_DIE(IccBytes icc, CreateNonconformingDisplayP3Profile());
  const size_t chad_record = FindIccTagRecord(icc, "chad");
  ASSERT_GT(chad_record, 0U);
  const size_t chad_offset = ReadBE32(icc.data() + chad_record + 4);
  const size_t chad_size = ReadBE32(icc.data() + chad_record + 8);
  ASSERT_GE(chad_size, 44U);
  ASSERT_LE(chad_offset + chad_size, icc.size());
  ASSERT_EQ(std::memcmp(icc.data() + chad_offset, "sf32", 4), 0);

  // Structured color fields can only regenerate Bradford adaptation. A valid
  // profile with a different CHAD must therefore retain its ICC profile.
  WriteBE32(icc.data() + chad_offset + 8,
            ReadBE32(icc.data() + chad_offset + 8) + 4096);

  ColorEncoding actual;
  ASSERT_TRUE(actual.SetICC(std::move(icc), JxlGetDefaultCms()));
  actual.DecideIfWantICC(*JxlGetDefaultCms());
  EXPECT_TRUE(actual.WantICC());
}

TEST_F(ColorManagementTest, SingularChadIsNotSimplified) {
  JXL_TEST_ASSIGN_OR_DIE(IccBytes icc, CreateNonconformingDisplayP3Profile());
  const size_t chad_record = FindIccTagRecord(icc, "chad");
  ASSERT_GT(chad_record, 0U);
  const size_t chad_offset = ReadBE32(icc.data() + chad_record + 4);
  const size_t chad_size = ReadBE32(icc.data() + chad_record + 8);
  ASSERT_GE(chad_size, 44U);
  ASSERT_LE(chad_offset + chad_size, icc.size());
  for (size_t i = 0; i < 9; ++i) {
    WriteBE32(icc.data() + chad_offset + 8U + 4U * i, 0);
  }

  ColorEncoding actual;
  // Rejecting a singular CHAD is also safe. If the CMS accepts the malformed
  // profile, it must retain the ICC rather than enter the compatibility path.
  if (actual.SetICC(std::move(icc), JxlGetDefaultCms())) {
    actual.DecideIfWantICC(*JxlGetDefaultCms());
    EXPECT_TRUE(actual.WantICC());
  }
}

TEST_F(ColorManagementTest, NearSingularChadIsNotSimplified) {
  JXL_TEST_ASSIGN_OR_DIE(IccBytes icc, CreateNonconformingDisplayP3Profile());
  const size_t chad_record = FindIccTagRecord(icc, "chad");
  ASSERT_GT(chad_record, 0U);
  const size_t chad_offset = ReadBE32(icc.data() + chad_record + 4);
  const size_t chad_size = ReadBE32(icc.data() + chad_record + 8);
  ASSERT_GE(chad_size, 44U);
  ASSERT_LE(chad_offset + chad_size, icc.size());
  ASSERT_EQ(std::memcmp(icc.data() + chad_offset, "sf32", 4), 0);

  // Each diagonal value is 1/65536 in ICC s15Fixed16 notation. This is a
  // nonzero, ICC-representable determinant below Inv3x3Matrix's threshold.
  constexpr uint32_t kOneFixedPointLsb = 1;
  constexpr double kDeterminant = 1.0 / (65536.0 * 65536.0 * 65536.0);
  EXPECT_GT(kDeterminant, 0.0);
  EXPECT_LT(kDeterminant, 1e-10);
  for (size_t i = 0; i < 9; ++i) {
    WriteBE32(icc.data() + chad_offset + 8U + 4U * i,
              i % 4 == 0 ? kOneFixedPointLsb : 0);
  }

  ColorEncoding actual;
  // Rejecting an unsafe CHAD is also safe. If the CMS accepts the malformed
  // profile, it must retain the ICC rather than enter the compatibility path.
  if (actual.SetICC(std::move(icc), JxlGetDefaultCms())) {
    actual.DecideIfWantICC(*JxlGetDefaultCms());
    EXPECT_TRUE(actual.WantICC());
  }
}

TEST_F(ColorManagementTest, Version2DisplayP3ProfileUsesStructuredFields) {
  JXL_TEST_ASSIGN_OR_DIE(IccBytes icc, CreateDisplayP3MatrixProfile());
  ASSERT_GE(icc.size(), 12U);

  // Change the header version to ICC v2.1. The malformed-profile compatibility
  // rule is v4-only, and valid v2 interpretation must remain unchanged.
  icc[8] = 2;
  icc[9] = 0x10;
  icc[10] = 0;
  icc[11] = 0;

  ColorEncoding actual;
  ASSERT_TRUE(actual.SetICC(std::move(icc), JxlGetDefaultCms()));
  actual.DecideIfWantICC(*JxlGetDefaultCms());
  EXPECT_FALSE(actual.WantICC());
  EXPECT_EQ(actual.GetWhitePointType(), WhitePoint::kD65);
  EXPECT_EQ(actual.GetPrimariesType(), Primaries::kP3);
  EXPECT_TRUE(actual.Tf().IsSRGB());
}

TEST_F(ColorManagementTest, LUTProfileRemainsICC) {
  ColorEncoding lut_source;
  lut_source.SetColorSpace(ColorSpace::kXYB);
  lut_source.SetRenderingIntent(RenderingIntent::kPerceptual);
  ASSERT_TRUE(lut_source.CreateICC());
  ASSERT_TRUE(HasIccTag(lut_source.ICC(), "A2B0"));

  JXL_TEST_ASSIGN_OR_DIE(IccBytes hybrid,
                         CreateNonconformingDisplayP3Profile());
  ASSERT_TRUE(ReplaceIccTag(&hybrid, "desc", "A2B0", lut_source.ICC(), "A2B0"));
  ASSERT_TRUE(HasIccTag(hybrid, "rXYZ"));
  ASSERT_TRUE(HasIccTag(hybrid, "rTRC"));
  ASSERT_TRUE(HasIccTag(hybrid, "A2B0"));

  ColorEncoding actual;
  ASSERT_TRUE(actual.SetICC(std::move(hybrid), JxlGetDefaultCms()));
  actual.DecideIfWantICC(*JxlGetDefaultCms());
  EXPECT_TRUE(actual.WantICC());
  EXPECT_NE(actual.GetWhitePointType(), WhitePoint::kD65);
}

TEST_F(ColorManagementTest, SampledTrcProfileSkipsWhitePointRecovery) {
  JXL_TEST_ASSIGN_OR_DIE(IccBytes icc, CreateNonconformingDisplayP3Profile());
  ASSERT_TRUE(ReplaceTrcsWithSampledCurve(&icc));

  ColorEncoding actual;
  ASSERT_TRUE(actual.SetICC(std::move(icc), JxlGetDefaultCms()));
  actual.DecideIfWantICC(*JxlGetDefaultCms());
  EXPECT_TRUE(actual.WantICC());
  EXPECT_NE(actual.GetWhitePointType(), WhitePoint::kD65);
}

TEST_F(ColorManagementTest, NoChadProfileSkipsCompatibilityPath) {
  JXL_TEST_ASSIGN_OR_DIE(IccBytes icc, CreateNonconformingDisplayP3Profile());
  const size_t chad_record = FindIccTagRecord(icc, "chad");
  ASSERT_GT(chad_record, 0U);
  std::memcpy(icc.data() + chad_record, "tst1", 4);
  ASSERT_TRUE(ReplaceTrcsWithSampledCurve(&icc, /*make_non_equivalent=*/false));

  ColorEncoding actual;
  ASSERT_TRUE(actual.SetICC(std::move(icc), JxlGetDefaultCms()));
  actual.DecideIfWantICC(*JxlGetDefaultCms());
  EXPECT_FALSE(actual.WantICC());
  EXPECT_EQ(actual.GetWhitePointType(), WhitePoint::kD65);
  EXPECT_EQ(actual.GetPrimariesType(), Primaries::kP3);
  EXPECT_TRUE(actual.Tf().IsSRGB());
}

TEST_F(ColorManagementTest, UnequalTrcMatrixProfileRemainsICC) {
  JXL_TEST_ASSIGN_OR_DIE(IccBytes icc, CreateNonconformingDisplayP3Profile());
  ASSERT_GE(icc.size(), 132U);
  const uint32_t tag_count = ReadBE32(icc.data() + 128);
  ASSERT_LE(tag_count, (icc.size() - 132U) / 12U);
  size_t green_trc_record = 0;
  for (uint32_t i = 0; i < tag_count; ++i) {
    const size_t record = 132U + 12U * i;
    if (std::memcmp(icc.data() + record, "gTRC", 4) == 0) {
      green_trc_record = record;
      break;
    }
  }
  ASSERT_GT(green_trc_record, 0U);
  const size_t trc_offset = ReadBE32(icc.data() + green_trc_record + 4);
  const size_t trc_size = ReadBE32(icc.data() + green_trc_record + 8);
  ASSERT_GE(trc_size, 20U);
  ASSERT_LE(trc_offset + trc_size, icc.size());

  const IccBytes green_trc(icc.begin() + trc_offset,
                           icc.begin() + trc_offset + trc_size);
  const size_t new_trc_offset = (icc.size() + 3U) & ~size_t{3};
  icc.resize(new_trc_offset + trc_size);
  std::copy(green_trc.begin(), green_trc.end(), icc.begin() + new_trc_offset);
  WriteBE32(icc.data(), static_cast<uint32_t>(icc.size()));
  WriteBE32(icc.data() + green_trc_record + 4,
            static_cast<uint32_t>(new_trc_offset));
  WriteBE32(icc.data() + new_trc_offset + 16,
            ReadBE32(icc.data() + new_trc_offset + 16) + 4096);

  ColorEncoding actual;
  ASSERT_TRUE(actual.SetICC(std::move(icc), JxlGetDefaultCms()));
  actual.DecideIfWantICC(*JxlGetDefaultCms());
  EXPECT_TRUE(actual.WantICC());
}

TEST_F(ColorManagementTest, P3HlgTo2020Hlg) {
  ColorEncoding p3_hlg;
  p3_hlg.SetColorSpace(ColorSpace::kRGB);
  ASSERT_TRUE(p3_hlg.SetWhitePointType(WhitePoint::kD65));
  ASSERT_TRUE(p3_hlg.SetPrimariesType(Primaries::kP3));
  p3_hlg.Tf().SetTransferFunction(TransferFunction::kHLG);
  ASSERT_TRUE(p3_hlg.CreateICC());

  ColorEncoding rec2020_hlg = p3_hlg;
  ASSERT_TRUE(rec2020_hlg.SetPrimariesType(Primaries::k2100));
  ASSERT_TRUE(rec2020_hlg.CreateICC());

  ColorSpaceTransform transform(*JxlGetDefaultCms());
  ASSERT_TRUE(transform.Init(p3_hlg, rec2020_hlg, 1000, 1, 1));
  Color p3_hlg_values{0., 0.75, 0.};
  Color rec2020_hlg_values;
  ASSERT_TRUE(
      transform.Run(0, p3_hlg_values.data(), rec2020_hlg_values.data(), 1));
  Color rec2020_hlg_expected{0.3973, 0.7382, 0.1183};
  EXPECT_ARRAY_NEAR(rec2020_hlg_values, rec2020_hlg_expected, 1e-4);
}

TEST_F(ColorManagementTest, HlgOotf) {
  ColorEncoding p3_hlg;
  p3_hlg.SetColorSpace(ColorSpace::kRGB);
  ASSERT_TRUE(p3_hlg.SetWhitePointType(WhitePoint::kD65));
  ASSERT_TRUE(p3_hlg.SetPrimariesType(Primaries::kP3));
  p3_hlg.Tf().SetTransferFunction(TransferFunction::kHLG);
  ASSERT_TRUE(p3_hlg.CreateICC());

  ColorSpaceTransform transform_to_1000(*JxlGetDefaultCms());
  ASSERT_TRUE(
      transform_to_1000.Init(p3_hlg, ColorEncoding::LinearSRGB(), 1000, 1, 1));
  // HDR reference white: https://www.itu.int/pub/R-REP-BT.2408-4-2021
  Color p3_hlg_values{0.75, 0.75, 0.75};
  Color linear_srgb_values;
  ASSERT_TRUE(transform_to_1000.Run(0, p3_hlg_values.data(),
                                    linear_srgb_values.data(), 1));
  // On a 1000-nit display, HDR reference white should be 203 cd/m² which is
  // 0.203 times the maximum.
  EXPECT_ARRAY_NEAR(linear_srgb_values, (Color{0.203, 0.203, 0.203}), 1e-3);

  ColorSpaceTransform transform_to_400(*JxlGetDefaultCms());
  ASSERT_TRUE(
      transform_to_400.Init(p3_hlg, ColorEncoding::LinearSRGB(), 400, 1, 1));
  ASSERT_TRUE(transform_to_400.Run(0, p3_hlg_values.data(),
                                   linear_srgb_values.data(), 1));
  // On a 400-nit display, it should be 100 cd/m².
  EXPECT_ARRAY_NEAR(linear_srgb_values, (Color{0.250, 0.250, 0.250}), 1e-3);

  p3_hlg_values[2] = 0.50;
  ASSERT_TRUE(transform_to_1000.Run(0, p3_hlg_values.data(),
                                    linear_srgb_values.data(), 1));
  EXPECT_ARRAY_NEAR(linear_srgb_values, (Color{0.201, 0.201, 0.050}), 1e-3);

  ColorSpaceTransform transform_from_400(*JxlGetDefaultCms());
  ASSERT_TRUE(
      transform_from_400.Init(ColorEncoding::LinearSRGB(), p3_hlg, 400, 1, 1));
  linear_srgb_values[0] = linear_srgb_values[1] = linear_srgb_values[2] = 0.250;
  ASSERT_TRUE(transform_from_400.Run(0, linear_srgb_values.data(),
                                     p3_hlg_values.data(), 1));
  EXPECT_ARRAY_NEAR(p3_hlg_values, (Color{0.75, 0.75, 0.75}), 1e-3);

  ColorEncoding grayscale_hlg;
  grayscale_hlg.SetColorSpace(ColorSpace::kGray);
  ASSERT_TRUE(grayscale_hlg.SetWhitePointType(WhitePoint::kD65));
  grayscale_hlg.Tf().SetTransferFunction(TransferFunction::kHLG);
  ASSERT_TRUE(grayscale_hlg.CreateICC());

  ColorSpaceTransform grayscale_transform(*JxlGetDefaultCms());
  ASSERT_TRUE(grayscale_transform.Init(
      grayscale_hlg, ColorEncoding::LinearSRGB(/*is_gray=*/true), 1000, 1, 1));
  const float grayscale_hlg_value = 0.75;
  float linear_grayscale_value;
  ASSERT_TRUE(grayscale_transform.Run(0, &grayscale_hlg_value,
                                      &linear_grayscale_value, 1));
  EXPECT_NEAR(linear_grayscale_value, 0.203, 1e-3);
}

TEST_F(ColorManagementTest, XYBProfile) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  ColorEncoding c_xyb;
  c_xyb.SetColorSpace(ColorSpace::kXYB);
  c_xyb.SetRenderingIntent(RenderingIntent::kPerceptual);
  ASSERT_TRUE(c_xyb.CreateICC());
  ColorEncoding c_native = ColorEncoding::LinearSRGB(false);

  static const size_t kGridDim = 17;
  static const size_t kNumColors = kGridDim * kGridDim * kGridDim;
  const JxlCmsInterface& cms = *JxlGetDefaultCms();
  ColorSpaceTransform xform(cms);
  ASSERT_TRUE(
      xform.Init(c_xyb, c_native, kDefaultIntensityTarget, kNumColors, 1));

  JXL_TEST_ASSIGN_OR_DIE(Image3F native,
                         Image3F::Create(memory_manager, kNumColors, 1));
  float mul = 1.0f / (kGridDim - 1);
  for (size_t ir = 0, x = 0; ir < kGridDim; ++ir) {
    for (size_t ig = 0; ig < kGridDim; ++ig) {
      for (size_t ib = 0; ib < kGridDim; ++ib, ++x) {
        native.PlaneRow(0, 0)[x] = ir * mul;
        native.PlaneRow(1, 0)[x] = ig * mul;
        native.PlaneRow(2, 0)[x] = ib * mul;
      }
    }
  }
  JXL_TEST_ASSIGN_OR_DIE(Image3F opsin,
                         Image3F::Create(memory_manager, kNumColors, 1));
  ASSERT_TRUE(CopyImageTo(native, &opsin));
  ASSERT_TRUE(ToXYB(c_native, kDefaultIntensityTarget, nullptr, nullptr, &opsin,
                    cms, nullptr));

  JXL_TEST_ASSIGN_OR_DIE(Image3F opsin2,
                         Image3F::Create(memory_manager, kNumColors, 1));
  ASSERT_TRUE(CopyImageTo(opsin, &opsin2));
  ScaleXYB(&opsin2);

  float* src = xform.BufSrc(0);
  for (size_t i = 0; i < kNumColors; ++i) {
    for (size_t c = 0; c < 3; ++c) {
      src[3 * i + c] = opsin2.PlaneRow(c, 0)[i];
    }
  }

  float* dst = xform.BufDst(0);
  ASSERT_TRUE(xform.Run(0, src, dst, kNumColors));

  JXL_TEST_ASSIGN_OR_DIE(Image3F out,
                         Image3F::Create(memory_manager, kNumColors, 1));
  for (size_t i = 0; i < kNumColors; ++i) {
    for (size_t c = 0; c < 3; ++c) {
      out.PlaneRow(c, 0)[i] = dst[3 * i + c];
    }
  }

  auto debug_print_color = [&](size_t i) {
    printf(
        "(%f, %f, %f) -> (%9.6f, %f, %f) -> (%f, %f, %f) -> "
        "(%9.6f, %9.6f, %9.6f)",
        native.PlaneRow(0, 0)[i], native.PlaneRow(1, 0)[i],
        native.PlaneRow(2, 0)[i], opsin.PlaneRow(0, 0)[i],
        opsin.PlaneRow(1, 0)[i], opsin.PlaneRow(2, 0)[i],
        opsin2.PlaneRow(0, 0)[i], opsin2.PlaneRow(1, 0)[i],
        opsin2.PlaneRow(2, 0)[i], out.PlaneRow(0, 0)[i], out.PlaneRow(1, 0)[i],
        out.PlaneRow(2, 0)[i]);
  };

  float max_err[3] = {};
  size_t max_err_i[3] = {};
  for (size_t i = 0; i < kNumColors; ++i) {
    for (size_t c = 0; c < 3; ++c) {
      // debug_print_color(i); printf("\n");
      float err = std::abs(native.PlaneRow(c, 0)[i] - out.PlaneRow(c, 0)[i]);
      if (err > max_err[c]) {
        max_err[c] = err;
        max_err_i[c] = i;
      }
    }
  }
  static float kMaxError[3] = {8.7e-4, 4.4e-4, 5.2e-4};
  printf("Maximum errors:\n");
  for (size_t c = 0; c < 3; ++c) {
    debug_print_color(max_err_i[c]);
    printf("    %f\n", max_err[c]);
    EXPECT_LT(max_err[c], kMaxError[c]);
  }
}

TEST_F(ColorManagementTest, GoldenXYBCube) {
  std::vector<int32_t> actual;
  const jxl::cms::ColorCube3D& cube = jxl::cms::UnscaledA2BCube();
  for (size_t ix = 0; ix < 2; ++ix) {
    for (size_t iy = 0; iy < 2; ++iy) {
      for (size_t ib = 0; ib < 2; ++ib) {
        const jxl::cms::ColorCube0D& out_f = cube[ix][iy][ib];
        for (int i = 0; i < 3; ++i) {
          int32_t val = static_cast<int32_t>(std::lround(65535 * out_f[i]));
          ASSERT_TRUE(val >= 0 && val <= 65535);
          actual.push_back(val);
        }
      }
    }
  }

  std::vector<int32_t> expected = {0,     3206,  0,     0,     3206,  28873,
                                   62329, 65535, 36662, 62329, 65535, 65535,
                                   3206,  0,     0,     3206,  0,     28873,
                                   65535, 62329, 36662, 65535, 62329, 65535};
  EXPECT_EQ(actual, expected);
}

}  // namespace
}  // namespace jxl
