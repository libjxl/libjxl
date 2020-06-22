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

#include "jxl/external_image.h"

#include <stdio.h>

#include <array>
#include <new>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/common.h"
#include "jxl/image_ops.h"
#include "jxl/image_test_utils.h"
#include "jxl/test_utils.h"

namespace jxl {
namespace {

// Small enough to be fast. If changed, must update Generate*.
static constexpr size_t kWidth = 8;
static constexpr size_t kHeight = 3;

struct Globals {
  Globals()
      : pool(3)  // matches kHeight below
  {
    // Linear BT.2100
    c_wide[0].SetColorSpace(ColorSpace::kRGB);
    c_wide[0].primaries = Primaries::k2100;
    c_wide[0].tf.SetTransferFunction(TransferFunction::kLinear);
    JXL_CHECK(c_wide[0].CreateICC());

    // Same, but gray
    c_wide[1] = c_wide[0];
    c_wide[1].SetColorSpace(ColorSpace::kGray);
    JXL_CHECK(c_wide[1].CreateICC());

    // Index = (channels - 1) + 8bit ? 0 : 4.
    color[1 - 1 + 0] = MakeGray();
    color[2 - 1 + 0] = MakeGray();
    alpha[2 - 1 + 0] = MakeAlpha(8);
    color[1 - 1 + 4] = MakeGray();
    color[2 - 1 + 4] = MakeGray();
    alpha[2 - 1 + 4] = MakeAlpha(16);
    color[3 - 1 + 0] = MakeColor();
    color[4 - 1 + 0] = MakeColor();
    alpha[4 - 1 + 0] = MakeAlpha(8);
    color[3 - 1 + 4] = MakeColor();
    color[4 - 1 + 4] = MakeColor();
    alpha[4 - 1 + 4] = MakeAlpha(16);
  }

 private:
  static ImageU MakeAlpha(size_t bits_per_sample) {
    const uint16_t max_alpha = bits_per_sample <= 8 ? 255 : 65535;
    ImageU alpha(kWidth, kHeight);
    RandomFillImage(&alpha, max_alpha);
    return alpha;
  }

  static Image3F MakeGray() {
    Image3F out(kWidth, kHeight);
    for (size_t c = 0; c < 3; ++c) {
      for (int32_t y = 0; y < kHeight; ++y) {
        float* JXL_RESTRICT row = out.PlaneRow(c, y);
        // Increasing left to right, top to bottom
        for (int32_t x = 0; x < kWidth; ++x) {
          row[x] = y + x * (255.0f - (kHeight - 1)) / (kWidth - 1);  // [0, 255]
        }
      }
    }
    return out;
  }

  static Image3F MakeColor() {
    Image3F out(kWidth, kHeight);
    ZeroFillImage(&out);

    // Row 0: neutral
    float* JXL_RESTRICT row0 = out.PlaneRow(0, 0);
    float* JXL_RESTRICT row1 = out.PlaneRow(1, 0);
    float* JXL_RESTRICT row2 = out.PlaneRow(2, 0);
    for (int32_t x = 0; x < kWidth; ++x) {
      row2[x] = row1[x] = row0[x] = x * 255.0f / (kWidth - 1);  // [0, 255]
    }

    // Row 1: pure RGB
    for (int32_t x = 0; x < kWidth; ++x) {
      const int32_t c = x % 3;
      float* JXL_RESTRICT row = out.PlaneRow(c, 1);
      row[x] = x * 255.0f / (kWidth - 1);  // [0, 255]
    }

    // Row 2: mixed
    for (int32_t x = 0; x < kWidth; ++x) {
      for (size_t c = 0; c < 3; ++c) {
        out.PlaneRow(c, 2)[x] = 40.0f;
      }
      const int32_t c = x % 3;
      float* JXL_RESTRICT row = out.PlaneRow(c, 2);
      row[x] = x * 255.0f / (kWidth - 1);  // [0, 255]
    }
    return out;
  }

 public:
  ThreadPoolInternal pool;
  std::array<ColorEncoding, 2> c_wide;

  // Pregenerate to reduce allocations.
  Image3F color[8];
  ImageU alpha[8];
};
static Globals* g;

class ExternalImageParametricTest
    : public ::testing::TestWithParam<test::ColorEncodingDescriptor> {
 public:
  static void SetUpTestSuite() { g = new Globals; }
  static void TearDownTestSuite() { delete g; }

  // "Same" pixels (modulo quantization) after converting to/from ExternalImage.
  static void VerifyPixelRoundTrip(const ColorEncoding& c_external,
                                   const bool add_alpha, const bool big_endian,
                                   const size_t bits_per_sample) {
    if (big_endian && (bits_per_sample == 8)) return;

    printf("sb:%zu, gr:%d al:%d be:%d\n", bits_per_sample, c_external.IsGray(),
           add_alpha, big_endian);

    const size_t channels = c_external.Channels() + add_alpha;
    const size_t idx = channels - 1 + (bits_per_sample <= 8 ? 0 : 4);
    const ColorEncoding& c_current = g->c_wide[c_external.IsGray()];

    CodecInOut io;
    if (bits_per_sample == 32) {
      io.metadata.SetFloat32Samples();
    } else {
      io.metadata.SetUintSamples(bits_per_sample);
    }
    io.metadata.color_encoding = c_current;
    io.SetFromImage(CopyImage(g->color[idx]), c_current);
    ImageBundle& ib = io.Main();
    const size_t alpha_bits = bits_per_sample <= 8 ? 8 : 16;
    if (add_alpha) {
      io.metadata.alpha_bits = alpha_bits;
      ib.SetAlpha(CopyImage(g->alpha[idx]), /*alpha_is_premultiplied=*/false);
    }

    const ImageU* alpha = io.metadata.HasAlpha() ? &ib.alpha() : nullptr;
    // Rescale values instead of clipping. This avoids truncating negative or
    // > max (out of gamut) samples - happens after chromatic adaptation of pure
    // colors (Bradford matrix has negative entries). Reduces round-trip error.
    // Codec currently are not able to store this in metadata.
    CodecIntervals temp_intervals;
    const ExternalImage external(&g->pool, ib.color(), Rect(ib), ib.c_current(),
                                 c_external, ib.HasAlpha(),
                                 ib.AlphaIsPremultiplied(), alpha, alpha_bits,
                                 bits_per_sample, big_endian, &temp_intervals);
    ASSERT_TRUE(external.IsHealthy());

    // Copy for later comparison.
    const Image3F& prev_color = CopyImage(ib.color());

    // Ensure TransformTo actually fills color.
    ZeroFillImage(const_cast<Image3F*>(&ib.color()));

    // Copy c_external pixels to ib..
    ASSERT_TRUE(external.CopyTo(&temp_intervals, &g->pool, &ib));
    ASSERT_TRUE(ib.c_current().SameColorSpace(c_external));

    // .. and transform back to c_current
    ASSERT_TRUE(ib.TransformTo(c_current, &g->pool));

    if (add_alpha) {
      JXL_CHECK(SamePixels(g->alpha[idx], ib.alpha()));
    }

    // => should be the same as prev_color.
    double max_l1, max_rel;
    const bool is2100 = c_external.tf.IsPQ() || c_external.tf.IsHLG();
    // Different gamut or chromatic adaptation => higher error.
    const bool needsCompute =
        (!c_external.IsGray() &&
         c_external.primaries != ib.c_current().primaries) ||
        c_external.white_point != ib.c_current().white_point;
#if JPEGXL_ENABLE_SKCMS
    if (true) {
#else
    if (bits_per_sample <= 12) {
#endif
      if (is2100 && needsCompute) {
        max_l1 = 8.3;  // 8 bit is just not enough for PQ + adaptation.
        max_rel = 2E-3;
      } else if (c_external.IsGray()) {
        max_l1 = 0.4;
        max_rel = is2100 ? 2E-2 : 7E-3;
      } else if (is2100) {
        max_l1 = 5E-4;
        max_rel = 2E-2;
      } else if (needsCompute) {
        max_l1 = 2.0;
        max_rel = 4E-2;
      } else {
        max_l1 = 5E-4;
        max_rel = 1.5E-2;
      }
    } else if (bits_per_sample == 16) {
      if (is2100 && needsCompute) {
        max_l1 = 3E-2;
        max_rel = 5E-4;
      } else if (c_external.IsGray()) {
        max_l1 = 7E-3;
        max_rel = 1E-4;
      } else if (is2100) {
        max_l1 = 7E-5;
        max_rel = 7E-5;
      } else if (needsCompute) {
        max_l1 = 1E-2;
        max_rel = 2E-4;
      } else {
        max_l1 = 1E-6;
        max_rel = 3E-5;
      }
    } else {
      JXL_ASSERT(bits_per_sample == 32);
      if (is2100 && needsCompute) {
        max_l1 = 4E-4;
        max_rel = 2E-5;
      } else if (c_external.IsGray()) {
        max_l1 = 4E-7;
        max_rel = 2E-5;
      } else if (is2100) {
        max_l1 = 4E-7;
        max_rel = 2E-5;
      } else if (needsCompute) {
        max_l1 = 7E-3;
        max_rel = 2E-5;
      } else {
        max_l1 = 2E-8;
#ifdef __ARM_ARCH
        // pow() implementation in arm is a bit less precise than in x86 and
        // therefore we need a bigger error margin in this case.
        max_rel = 2E-5;
#else
        max_rel = 1E-6;
#endif
      }
    }

    VerifyRelativeError(prev_color, ib.color(), max_l1, max_rel);
  }

  // Copying from CodecInOut to Image3 => same pixels (modulo clipping).
  template <typename T>
  static void TestCopyTo() {
    const size_t bits_per_sample = sizeof(T) * kBitsPerByte;
    // Largest value the converted image can have.
    const float external_max = bits_per_sample == 16 ? 65535 : 255;
    // ExternalImage assumes input ImageF are always [0, 255] regardless of
    // the output type.
    const float range = 256.0f;
    const float scale = external_max / 255;

    // First and last are out of bounds.
    const size_t xsize = 1 + 16 + 1;
    const float step = range / (xsize - 2);
    const size_t ysize = 2;

    Image3F image(xsize, ysize);
    for (size_t c = 0; c < 3; ++c) {
      float* JXL_RESTRICT row0 = image.PlaneRow(c, 0);
      row0[0] = -1.0f;
      for (size_t x = 1; x < xsize; ++x) row0[x] = x * step - 1;
      EXPECT_GE(row0[xsize - 1], range);
      float* JXL_RESTRICT row1 = image.PlaneRow(c, 1);
      row1[0] = -9.0f;
      for (size_t x = 1; x < xsize; ++x) row1[x] = (x - 1) * step;
      EXPECT_GE(row1[xsize - 1], range);
    }

    CodecInOut io;
    io.metadata.SetFloat32Samples();
    io.metadata.color_encoding = ColorEncoding::SRGB();
    io.SetFromImage(std::move(image), io.metadata.color_encoding);
    const ImageBundle& ib = io.Main();

    Image3<T> out;
    ASSERT_TRUE(
        ib.CopyTo(Rect(ib), ColorEncoding::SRGB(ib.IsGray()), &out, &g->pool));
    for (size_t c = 0; c < 3; ++c) {
      const T* row_out0 = out.PlaneRow(c, 0);
      const T* row_out1 = out.PlaneRow(c, 1);

      // No clipping/clamping for float.
      if (sizeof(T) != sizeof(float)) {
        // x = 0: clamped to min
        EXPECT_EQ(0, row_out0[0]);
        EXPECT_EQ(0, row_out1[0]);

        // x = xsize - 1: clamped to external_max
        EXPECT_EQ(external_max, row_out0[xsize - 1]);
        EXPECT_EQ(external_max, row_out1[xsize - 1]);
      }

      // Interior: same values
      const float* row_in0 = ib.color().ConstPlaneRow(c, 0);
      const float* row_in1 = ib.color().ConstPlaneRow(c, 1);
      for (size_t x = 1; x < xsize - 1; ++x) {
        EXPECT_NEAR(scale * row_in0[x], row_out0[x], 5E-5) << " 0 " << x;
        EXPECT_NEAR(scale * row_in1[x], row_out1[x], 5E-5) << " 1 " << x;
      }
    }
  }
};
INSTANTIATE_TEST_SUITE_P(ExternalImageTestInstantiation,
                         ExternalImageParametricTest,
                         ::testing::ValuesIn(test::AllEncodings()));

TEST_F(ExternalImageParametricTest, TestConvert) {
  TestCopyTo<uint8_t>();
  TestCopyTo<uint16_t>();
  TestCopyTo<float>();
}

TEST_P(ExternalImageParametricTest, RoundTrip) {
  ColorEncoding c_external = ColorEncodingFromDescriptor(GetParam());
  // To speed up the test, ignore rendering_intent (unused here)
  if (c_external.rendering_intent != RenderingIntent::kRelative) return;
  JXL_CHECK(c_external.CreateICC());

  printf("%s\n", Description(c_external).c_str());

  for (size_t bits_per_sample : {8, 16, 32}) {
    for (bool add_alpha : {false, true}) {
      for (bool big_endian : {false, true}) {
        VerifyPixelRoundTrip(c_external, add_alpha, big_endian,
                             bits_per_sample);
      }
    }
  }
}

#if !defined(JXL_CRASH_ON_ERROR)
TEST(ExternalImageTest, InvalidSize) {
  PackedImage desc(/*xsize=*/10, /*ysize=*/100,
                   /*c_current=*/ColorEncoding::SRGB(),
                   /*has_alpha=*/true, /*alpha_is_premultiplied=*/false,
                   /*bits_per_alpha=*/8, /*bits_per_sample=*/16,
                   /*big_endian=*/true, /*flipped_y=*/false);

  ImageMetadata im;
  im.alpha_bits = 8;
  ImageBundle ib(&im);

  const uint8_t buf[10 * 100 * 8] = {};
  EXPECT_FALSE(CopyTo(desc, Span<const uint8_t>(buf, 10), nullptr, &ib));
  EXPECT_FALSE(
      CopyTo(desc, Span<const uint8_t>(buf, sizeof(buf) - 1), nullptr, &ib));
  EXPECT_TRUE(
      CopyTo(desc, Span<const uint8_t>(buf, sizeof(buf)), nullptr, &ib));
}
#endif

}  // namespace
}  // namespace jxl
