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

#include <stddef.h>

#include <string>

#include "gtest/gtest.h"
#include "jxl/aux_out.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/override.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_encoding.h"
#include "jxl/dec_file.h"
#include "jxl/dec_params.h"
#include "jxl/enc_butteraugli_comparator.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_file.h"
#include "jxl/enc_params.h"
#include "jxl/extras/codec.h"
#include "jxl/headers.h"
#include "jxl/image_bundle.h"
#include "jxl/test_utils.h"
#include "jxl/testdata_path.h"

namespace jxl {
namespace {
using test::Roundtrip;

TEST(PreviewTest, RoundtripGivenPreview) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));
  io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);
  // Same as main image
  io.preview_frame = io.Main().Copy();
  io.metadata.m2.have_preview = true;
  ASSERT_TRUE(
      io.preview.Set(io.preview_frame.xsize(), io.preview_frame.ysize()));

  CompressParams cparams;
  cparams.butteraugli_distance = 2.0;
  cparams.speed_tier = SpeedTier::kSquirrel;
  DecompressParams dparams;
  dparams.preview = Override::kOff;

  CodecInOut io2;
  Roundtrip(&io, cparams, dparams, pool, &io2);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            2.5);
  EXPECT_EQ(0, io2.preview_frame.xsize());

  dparams.preview = Override::kOn;

  CodecInOut io3;
  Roundtrip(&io, cparams, dparams, pool, &io3);
  EXPECT_LE(ButteraugliDistance(io, io2, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            2.5);

  // Preview image also close to original
  EXPECT_NE(0, io3.preview_frame.xsize());
  EXPECT_LE(
      ButteraugliDistance(io.Main(), io3.preview_frame, cparams.hf_asymmetry,
                          /*distmap=*/nullptr, pool),
      2.5);
}

}  // namespace
}  // namespace jxl
