// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/types.h>

#include <cstdint>
#include <sstream>
#include <utility>
#include <vector>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/dec/jxl.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/blending.h"
#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

TEST(BlendingTest, Crops) {
  const std::vector<uint8_t> compressed =
      jxl::test::ReadTestData("jxl/blending/cropped_traffic_light.jxl");
  extras::JXLDecompressParams dparams;
  dparams.accepted_formats = {{3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, 0}};
  extras::PackedPixelFile decoded;
  ASSERT_TRUE(DecodeImageJXL(compressed.data(), compressed.size(), dparams,
                             /*decoded_bytes=*/nullptr, &decoded));
  ASSERT_EQ(decoded.frames.size(), 4u);

  int i = 0;
  for (auto&& decoded_frame : decoded.frames) {
    std::ostringstream filename;
    filename << "jxl/blending/cropped_traffic_light_frame-" << i << ".png";
    const std::vector<uint8_t> compressed_frame =
        jxl::test::ReadTestData(filename.str());
    extras::PackedPixelFile decoded_frame_ppf;
    decoded_frame_ppf.info = decoded.info;
    decoded_frame_ppf.primary_color_representation =
        decoded.primary_color_representation;
    decoded_frame_ppf.color_encoding = decoded.color_encoding;
    decoded_frame_ppf.icc = decoded.icc;
    decoded_frame_ppf.extra_channels_info = decoded.extra_channels_info;
    decoded_frame_ppf.frames.emplace_back(std::move(decoded_frame));
    extras::PackedPixelFile expected_frame_ppf;
    ASSERT_TRUE(extras::DecodeBytes(Bytes(compressed_frame),
                                    extras::ColorHints(), &expected_frame_ppf));
    EXPECT_EQ(0.0f,
              test::ComputeDistance2(decoded_frame_ppf, expected_frame_ppf));
    ++i;
  }
}


// ---------------------------------------------------------------------------
// Zero-width blend regions.
//
// A zero-width region is legal: PlaneBase::Allocate returns without allocating
// when xsize_ == 0, so every Row() of the temporary in PerformBlending is null.
// The `copy` lambda used to hand that null straight to memcpy as the
// destination -- undefined behaviour even with a zero copy size, and an abort
// under UBSan. Every other blend path in the function already guards with
// `if (xsize)`; these cases pin the one that did not.
// ---------------------------------------------------------------------------
struct BlendFixture {
  std::vector<float> bg_storage;
  std::vector<float> fg_storage;
  std::vector<float> out_storage;
  std::vector<float*> bg;
  std::vector<float*> fg;
  std::vector<float*> out;
};

BlendFixture MakeFixture(size_t x0, size_t xsize) {
  BlendFixture f;
  const size_t width = x0 + xsize + 1;
  f.bg_storage.assign(3 * width, 0.25f);
  f.fg_storage.assign(3 * width, 0.75f);
  f.out_storage.assign(3 * width, 0.0f);
  for (size_t p = 0; p < 3; p++) {
    f.bg.push_back(f.bg_storage.data() + p * width);
    f.fg.push_back(f.fg_storage.data() + p * width);
    f.out.push_back(f.out_storage.data() + p * width);
  }
  return f;
}

Status Blend(size_t x0, size_t xsize, PatchBlendMode mode) {
  JxlMemoryManager* mm = jxl::test::MemoryManager();
  BlendFixture f = MakeFixture(x0, xsize);
  PatchBlending color_blending{mode, /*alpha_channel=*/0, /*clamp=*/false};
  // No extra channels, so `has_alpha` is false and the color switch takes the
  // `copy(...)` arm for kBlendAbove / kBlendBelow.
  std::vector<ExtraChannelInfo> extra_channel_info;
  return PerformBlending(mm, f.bg.data(), f.fg.data(), f.out.data(), x0, xsize,
                         color_blending, /*ec_blending=*/nullptr,
                         extra_channel_info);
}

TEST(BlendingTest, ZeroWidthBlendAboveIsSafe) {
  EXPECT_TRUE(Blend(/*x0=*/0, /*xsize=*/0, PatchBlendMode::kBlendAbove));
}

TEST(BlendingTest, ZeroWidthBlendBelowIsSafe) {
  EXPECT_TRUE(Blend(/*x0=*/0, /*xsize=*/0, PatchBlendMode::kBlendBelow));
}

TEST(BlendingTest, ZeroWidthNonZeroOffsetIsSafe) {
  // x0 != 0 exercises the `src[p] + x0` arithmetic as well as the null dest.
  EXPECT_TRUE(Blend(/*x0=*/4, /*xsize=*/0, PatchBlendMode::kBlendAbove));
}

TEST(BlendingTest, NonZeroWidthStillBlends) {
  // Control: a real blend region must keep working exactly as before.
  EXPECT_TRUE(Blend(/*x0=*/0, /*xsize=*/8, PatchBlendMode::kBlendAbove));
  EXPECT_TRUE(Blend(/*x0=*/2, /*xsize=*/4, PatchBlendMode::kBlendBelow));
}

}  // namespace
}  // namespace jxl
