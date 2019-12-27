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

#include "gtest/gtest.h"
#include "jxl/color_management.h"
#include "jxl/common.h"
#include "jxl/dec_file.h"
#include "jxl/enc_butteraugli_comparator.h"
#include "jxl/enc_file.h"
#include "jxl/extras/codec.h"
#include "jxl/extras/codec_jpg.h"
#include "jxl/image_ops.h"
#include "jxl/image_test_utils.h"
#include "jxl/testdata_path.h"

namespace jxl {
namespace {

TEST(JPEGkPassesTest, RoundtripLarge) {
  ThreadPool* pool = nullptr;
  const std::string pathname =
      GetTestDataPath("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ASSERT_TRUE(SetFromFile(pathname, &io, pool));

  // encode to JPEG
  PaddedBytes encoded;
  io.jpeg_quality = 90;
  JXL_CHECK(Encode(io, Codec::kJPG, io.metadata.color_encoding,
                   io.metadata.bits_per_sample, &encoded, pool));

  // decode JPEG to pixels (using libjpeg)
  CodecInOut io2;
  JXL_CHECK(jxl::DecodeImageJPG(Span<const uint8_t>(encoded), &io2));
  const ImageBundle& ib2 = io2.Main();

  // decode JPEG to DCT coeffs
  CodecInOut io3;
  JXL_CHECK(SetFromBytes(Span<const uint8_t>(encoded), &io3, pool,
                         jxl::DecodeTarget::kQuantizedCoeffs));

  CompressParams cparams;
  DecompressParams dparams;

  PaddedBytes compressed;
  AuxOut* aux_out = nullptr;
  PassesEncoderState enc_state;

  // encode DCT coeffs as kPasses JXL
  EXPECT_TRUE(
      EncodeFile(cparams, &io3, &enc_state, &compressed, aux_out, pool));
  // decode JXL to pixels
  CodecInOut io4;
  EXPECT_TRUE(DecodeFile(dparams, compressed, &io4, aux_out, pool));
  const ImageBundle& ib4 = io4.Main();

  // TODO: investigate where the difference between libjpeg and kPasses comes
  //       from (and see if it can be reduced)
  EXPECT_LE(ButteraugliDistance(ib2, ib4, cparams.hf_asymmetry,
                                /*distmap=*/nullptr, pool),
            0.87654f);

  // decode JXL to DCT coefficients
  CodecInOut io5;
  dparams.keep_dct = true;
  EXPECT_TRUE(DecodeFile(dparams, compressed, &io5, aux_out, pool));

  // encode the result to JPEG
  PaddedBytes encoded2;
  io5.jpeg_quality = 100;  // ignored but has to be set
  JXL_CHECK(Encode(io5, Codec::kJPG, io5.metadata.color_encoding,
                   io5.metadata.bits_per_sample, &encoded2, pool));

  // decode JPEG to pixels (using libjpeg)
  CodecInOut io6;
  JXL_CHECK(jxl::DecodeImageJPG(Span<const uint8_t>(encoded2), &io6));
  const ImageBundle& ib6 = io6.Main();

  EXPECT_TRUE(SamePixels(ib2.color(), ib6.color()));
}

}  // namespace
}  // namespace jxl
