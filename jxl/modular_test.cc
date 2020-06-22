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

#include <stdint.h>
#include <stdio.h>

#include <array>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "jxl/aux_out.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/override.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/dec_file.h"
#include "jxl/dec_params.h"
#include "jxl/enc_butteraugli_comparator.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_file.h"
#include "jxl/enc_params.h"
#include "jxl/extras/codec.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/image_test_utils.h"
#include "jxl/modular/encoding/encoding.h"
#include "jxl/test_utils.h"
#include "jxl/testdata.h"

namespace jxl {
namespace {
using test::Roundtrip;

TEST(ModularTest, RoundtripLossy) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CompressParams cparams;
  cparams.modular_group_mode = true;
  cparams.quality_pair = {90.0f, 90.0f};
  DecompressParams dparams;

  CodecInOut io_out;
  size_t compressed_size;

  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));

  compressed_size = Roundtrip(&io, cparams, dparams, pool, &io_out);
  EXPECT_LE(compressed_size, 150000);
  EXPECT_LE(ButteraugliDistance(io, io_out, cparams.hf_asymmetry, cparams.xmul,
                                /*distmap=*/nullptr, pool),
            1.5);
}

TEST(ModularTest, RoundtripLossyWP) {
  ThreadPool* pool = nullptr;
  const PaddedBytes orig =
      ReadTestData("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CompressParams cparams;
  cparams.modular_group_mode = true;
  cparams.quality_pair = {90.0f, 90.0f};
  cparams.options.predictor = {Predictor::Weighted};
  DecompressParams dparams;

  CodecInOut io_out;
  size_t compressed_size;

  CodecInOut io;
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, pool));

  compressed_size = Roundtrip(&io, cparams, dparams, pool, &io_out);
  EXPECT_LE(compressed_size, 200000);
  EXPECT_LE(ButteraugliDistance(io, io_out, cparams.hf_asymmetry, cparams.xmul,
                                /*distmap=*/nullptr, pool),
            1.5);
}

TEST(ModularTest, RoundtripExtraProperties) {
  constexpr size_t kSize = 250;
  Image image(kSize, kSize, /*maxval=*/255, 3);
  ModularOptions options;
  options.max_properties = 4;
  options.predictor = Predictor::Zero;
  std::mt19937 rng(0);
  std::uniform_int_distribution<> dist(0, 8);
  for (size_t y = 0; y < kSize; y++) {
    for (size_t x = 0; x < kSize; x++) {
      image.channel[0].plane.Row(y)[x] = image.channel[2].plane.Row(y)[x] =
          dist(rng);
    }
  }
  ZeroFillImage(&image.channel[1].plane);
  BitWriter writer;
  ASSERT_TRUE(ModularGenericCompress(image, options, &writer));
  writer.ZeroPadToByte();
  Image decoded(kSize, kSize, /*maxval=*/255, image.channel.size());
  for (size_t i = 0; i < image.channel.size(); i++) {
    const Channel& ch = image.channel[i];
    decoded.channel[i] = Channel(ch.w, ch.h, ch.hshift, ch.vshift);
  }
  Status status = true;
  {
    BitReader reader(writer.GetSpan());
    BitReaderScopedCloser closer(&reader, &status);
    ASSERT_TRUE(
        ModularGenericDecompress(&reader, decoded, /*group_id=*/0, &options));
  }
  ASSERT_TRUE(status);
  ASSERT_EQ(image.channel.size(), decoded.channel.size());
  for (size_t c = 0; c < image.channel.size(); c++) {
    for (size_t y = 0; y < image.channel[c].plane.ysize(); y++) {
      for (size_t x = 0; x < image.channel[c].plane.xsize(); x++) {
        EXPECT_EQ(image.channel[c].plane.Row(y)[x],
                  decoded.channel[c].plane.Row(y)[x])
            << "c = " << c << ", x = " << x << ",  y = " << y;
      }
    }
  }
}

}  // namespace
}  // namespace jxl
