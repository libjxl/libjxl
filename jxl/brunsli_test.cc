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

#include "jxl/brunsli.h"

#include <stddef.h>

#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/enc_butteraugli_comparator.h"
#include "jxl/extras/codec.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/testdata_path.h"

namespace jxl {
namespace {

// Returns compressed size [bytes].
size_t Roundtrip(CodecInOut* io, const BrunsliEncoderOptions& enc_options,
                 const BrunsliDecoderOptions& dec_options, ThreadPool* pool,
                 CodecInOut* JXL_RESTRICT io2, float max_butteraugli_score) {
  PaddedBytes compressed;

  EXPECT_TRUE(PixelsToBrunsli(io, &compressed, enc_options, pool));

  BrunsliDecoderMeta meta;
  EXPECT_TRUE(BrunsliToPixels(jxl::Span<const uint8_t>(compressed), io2,
                              dec_options, &meta, pool));

  EXPECT_LE(ButteraugliDistance(*io, *io2, /* hf_assymetry */ 1.0f,
                                /*distmap=*/nullptr, pool),
            max_butteraugli_score);

  return compressed.size();
}

TEST(BrunsliTest, RoundtripSinglePixel) {
  ThreadPool* pool = nullptr;
  Image3F image(1, 1);
  BrunsliEncoderOptions enc_options;
  BrunsliDecoderOptions dec_options;

  image.PlaneRow(0, 0)[0] = 0.0f;
  image.PlaneRow(1, 0)[0] = 0.0f;
  image.PlaneRow(2, 0)[0] = 0.0f;

  CodecInOut io;
  io.metadata.bits_per_sample = 8;
  io.metadata.color_encoding = ColorManagement::SRGB();
  io.SetFromImage(std::move(image), io.metadata.color_encoding);
  CodecInOut io2;
  Roundtrip(&io, enc_options, dec_options, pool, &io2, 0.001f);
}

TEST(BrunsliTest, RoundtripTiny) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(16, 16);

  BrunsliEncoderOptions enc_options;
  enc_options.quant_scale = 0.0f;
  BrunsliDecoderOptions dec_options;

  CodecInOut io2;
  Roundtrip(&io, enc_options, dec_options, pool, &io2, 0.3f);
}

TEST(BrunsliTest, RoundtripSmallQ0) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  BrunsliEncoderOptions enc_options;
  enc_options.quant_scale = 0.0f;
  BrunsliDecoderOptions dec_options;

  CodecInOut io2;
  Roundtrip(&io, enc_options, dec_options, pool, &io2, 0.5f);
}

TEST(BrunsliTest, RoundtripDecOptions) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);

  BrunsliEncoderOptions enc_options;
  BrunsliDecoderOptions dec_options;
  CodecInOut io2;

  // Also try different decoding options.
  enc_options.quant_scale = 2.0f;

  dec_options.fix_dc_staircase = false;
  dec_options.gaborish = false;
  Roundtrip(&io, enc_options, dec_options, pool, &io2, 5.4f);

  dec_options.fix_dc_staircase = true;
  dec_options.gaborish = false;
  Roundtrip(&io, enc_options, dec_options, pool, &io2, 5.2f);

  dec_options.fix_dc_staircase = false;
  dec_options.gaborish = true;
  Roundtrip(&io, enc_options, dec_options, pool, &io2, 4.777f);
}

TEST(BrunsliTest, RoundtripUnalignedQ0_5) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 12, io.ysize() / 7);

  BrunsliEncoderOptions enc_options;
  enc_options.quant_scale = 0.5f;
  BrunsliDecoderOptions dec_options;

  CodecInOut io2;
  Roundtrip(&io, enc_options, dec_options, pool, &io2, 1.3579f);
}

TEST(BrunsliTest, RoundtripLarge) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("imagecompression.info/flower_foveon.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));

  BrunsliEncoderOptions enc_options;
  enc_options.quant_scale = 2.0f;
  BrunsliDecoderOptions dec_options;

  CodecInOut io2;
  Roundtrip(&io, enc_options, dec_options, pool, &io2, 8.88888);
}

}  // namespace
}  // namespace jxl
