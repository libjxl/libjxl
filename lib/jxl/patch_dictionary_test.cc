// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/encode.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/dec/decode.h"
#include "lib/extras/dec/jxl.h"
#include "lib/extras/enc/jxl.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_patch_dictionary.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_patch_dictionary.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace {

using ::jxl::test::ButteraugliDistance;
using ::jxl::test::GetImage;
using ::jxl::test::ReadTestData;
using ::jxl::test::Roundtrip;

TEST(PatchDictionaryTest, GrayscaleModular) {
  const std::vector<uint8_t> orig = ReadTestData("jxl/grayscale_patches.png");
  extras::PackedPixelFile ppf;
  ASSERT_TRUE(DecodeBytes(Bytes(orig), jxl::extras::ColorHints(), &ppf));

  extras::JXLCompressParams cparams = jxl::test::CompressParamsForLossless();
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PATCHES, 1);
  extras::JXLDecompressParams dparams;

  extras::PackedPixelFile ppf2;
  // Without patches: ~25k
  size_t compressed_size = Roundtrip(ppf, cparams, dparams, nullptr, &ppf2);
  EXPECT_LE(compressed_size, 8000u);
  JXL_TEST_ASSIGN_OR_DIE(ImageF image, GetImage(ppf));
  JXL_TEST_ASSIGN_OR_DIE(ImageF image2, GetImage(ppf2));
  JXL_TEST_ASSERT_OK(VerifyRelativeError(image, image2, 1e-7f, 0, _));
}

TEST(PatchDictionaryTest, GrayscaleVarDCT) {
  const std::vector<uint8_t> orig = ReadTestData("jxl/grayscale_patches.png");
  extras::PackedPixelFile ppf;
  ASSERT_TRUE(DecodeBytes(Bytes(orig), jxl::extras::ColorHints(), &ppf));

  extras::JXLCompressParams cparams;
  cparams.AddOption(JXL_ENC_FRAME_SETTING_PATCHES, 1);
  extras::JXLDecompressParams dparams;

  extras::PackedPixelFile ppf2;
  // Without patches: ~47k
  size_t compressed_size = Roundtrip(ppf, cparams, dparams, nullptr, &ppf2);
  EXPECT_LE(compressed_size, 14000u);
  // Without patches: ~1.2
  EXPECT_LE(ButteraugliDistance(ppf, ppf2), 1.1);
}

// The dictionary object is reused across frames: decoding a frame's patches
// must not keep the blend modes of a previous frame's patches.
TEST(PatchDictionaryTest, DecodeReplacesPreviousFrameBlendModes) {
  JxlMemoryManager* memory_manager = jxl::test::MemoryManager();
  ImageMetadata metadata;
  std::array<ReferenceFrame, 4> refs;
  for (ReferenceFrame& ref : refs) {
    ref.frame = jxl::make_unique<ImageBundle>(memory_manager, &metadata);
  }
  JXL_TEST_ASSIGN_OR_DIE(Image3F color, Image3F::Create(memory_manager, 1, 1));
  FillImage(1.0f, &color);
  ASSERT_TRUE(
      refs[0].frame->SetFromImage(std::move(color), ColorEncoding::SRGB()));
  refs[0].ib_is_in_xyb = true;

  PatchDictionary decoded(memory_manager);
  decoded.SetShared(&refs);
  // Frame 1 adds the reference pixel, frame 2 replaces with it.
  for (PatchBlendMode mode : {PatchBlendMode::kAdd, PatchBlendMode::kReplace}) {
    PatchDictionary pdic(memory_manager);
    PatchDictionaryEncoder::SetPositions(&pdic, {{0, 0, 0}}, {{0, 0, 0, 1, 1}},
                                         {{mode, 0, false}}, 1);
    BitWriter writer(memory_manager);
    ASSERT_TRUE(PatchDictionaryEncoder::Encode(pdic, &writer,
                                               LayerType::Dictionary, nullptr));
    writer.ZeroPadToByte();
    BitReader reader(writer.GetSpan());
    bool uses_extra_channels = false;
    ASSERT_TRUE(
        decoded.Decode(memory_manager, &reader, 1, 1, 0, &uses_extra_channels));
    ASSERT_TRUE(reader.Close());

    float pixel[3] = {0.5f, 0.5f, 0.5f};
    float* rows[3] = {&pixel[0], &pixel[1], &pixel[2]};
    ASSERT_TRUE(decoded.AddOneRow(rows, 0, 0, 1, {}));
    EXPECT_EQ(pixel[0], mode == PatchBlendMode::kAdd ? 1.5f : 1.0f);
  }
}

}  // namespace
}  // namespace jxl
