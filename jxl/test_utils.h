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

#ifndef JXL_TEST_UTILS_H_
#define JXL_TEST_UTILS_H_

// Macros and functions useful for tests.

#include "jxl/aux_out_fwd.h"
#include "jxl/base/data_parallel.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_encoding.h"
#include "jxl/dec_file.h"
#include "jxl/dec_params.h"
#include "jxl/enc_file.h"
#include "jxl/enc_params.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#ifdef JXL_DISABLE_SLOW_TESTS
#define JXL_SLOW_TEST(X) DISABLED_##X
#else
#define JXL_SLOW_TEST(X) X
#endif  // JXL_DISABLE_SLOW_TESTS

namespace jxl {
namespace test {

MATCHER_P(MatchesPrimariesAndTransferFunction, color_encoding, "") {
  return arg.primaries == color_encoding.primaries &&
         arg.tf.IsSame(color_encoding.tf);
}

MATCHER(MatchesPrimariesAndTransferFunction, "") {
  return testing::ExplainMatchResult(
      MatchesPrimariesAndTransferFunction(std::get<1>(arg)), std::get<0>(arg),
      result_listener);
}

// Returns compressed size [bytes].
size_t Roundtrip(CodecInOut* io, const CompressParams& cparams,
                 const DecompressParams& dparams, ThreadPool* pool,
                 CodecInOut* JXL_RESTRICT io2, AuxOut* aux_out = nullptr) {
  PaddedBytes compressed;

  std::vector<ColorEncoding> original_metadata_encodings;
  std::vector<ColorEncoding> original_current_encodings;
  for (const ImageBundle& ib : io->frames) {
    // Remember original encoding, will be returned by decoder.
    original_metadata_encodings.push_back(ib.metadata()->color_encoding);
    // c_current should not change during encoding.
    original_current_encodings.push_back(ib.c_current());
  }

  PassesEncoderState enc_state;
  EXPECT_TRUE(EncodeFile(cparams, io, &enc_state, &compressed, aux_out, pool));

  std::vector<ColorEncoding> metadata_encodings_1;
  for (const ImageBundle& ib1 : io->frames) {
    metadata_encodings_1.push_back(ib1.metadata()->color_encoding);
  }

  // Should still be in the same color space after encoding.
  EXPECT_THAT(metadata_encodings_1,
              testing::Pointwise(MatchesPrimariesAndTransferFunction(),
                                 original_metadata_encodings));

  EXPECT_TRUE(DecodeFile(dparams, compressed, io2, aux_out, pool));

  std::vector<ColorEncoding> metadata_encodings_2;
  std::vector<ColorEncoding> current_encodings_2;
  for (const ImageBundle& ib2 : io2->frames) {
    metadata_encodings_2.push_back(ib2.metadata()->color_encoding);
    current_encodings_2.push_back(ib2.c_current());
  }

  EXPECT_THAT(io2->frames, testing::SizeIs(io->frames.size()));
  if (!cparams.modular_group_mode) {
    // Non-modular returns linear sRGB.
    EXPECT_THAT(current_encodings_2,
                testing::Each(MatchesPrimariesAndTransferFunction(
                    ColorEncoding::LinearSRGB())));
  } else {
    // Modular returns the original color space.
    EXPECT_THAT(current_encodings_2,
                testing::Pointwise(MatchesPrimariesAndTransferFunction(),
                                   original_current_encodings));
  }

  // Decoder returns the originals passed to the encoder.
  EXPECT_THAT(metadata_encodings_2,
              testing::Pointwise(MatchesPrimariesAndTransferFunction(),
                                 original_metadata_encodings));

  return compressed.size();
}

// A POD descriptor of a ColorEncoding. Only used in tests as the return value
// of AllEncodings().
struct ColorEncodingDescriptor {
  ColorSpace color_space;
  WhitePoint white_point;
  Primaries primaries;
  TransferFunction tf;
  RenderingIntent rendering_intent;
};

static inline ColorEncoding ColorEncodingFromDescriptor(
    const ColorEncodingDescriptor& desc) {
  ColorEncoding c;
  c.SetColorSpace(desc.color_space);
  c.white_point = desc.white_point;
  c.primaries = desc.primaries;
  c.tf.SetTransferFunction(desc.tf);
  c.rendering_intent = desc.rendering_intent;
  return c;
}

// Define the operator<< for tests.
static inline ::std::ostream& operator<<(::std::ostream& os,
                                         const ColorEncodingDescriptor& c) {
  return os << "ColorEncoding/" << Description(ColorEncodingFromDescriptor(c));
}

// Returns ColorEncodingDescriptors, which are only used in tests. To obtain a
// ColorEncoding object call ColorEncodingFromDescriptor and then call
// ColorEncoding::CreateProfile() on that object to generate a profile.
std::vector<ColorEncodingDescriptor> AllEncodings() {
  std::vector<ColorEncodingDescriptor> all_encodings;
  all_encodings.reserve(300);
  ColorEncoding c;

  for (ColorSpace cs : Values<ColorSpace>()) {
    if (cs == ColorSpace::kUnknown || cs == ColorSpace::kXYB) continue;
    c.SetColorSpace(cs);

    for (WhitePoint wp : Values<WhitePoint>()) {
      if (wp == WhitePoint::kCustom) continue;
      if (c.ImplicitWhitePoint() && c.white_point != wp) continue;
      c.white_point = wp;

      for (Primaries primaries : Values<Primaries>()) {
        if (primaries == Primaries::kCustom) continue;
        if (!c.HasPrimaries()) continue;
        c.primaries = primaries;

        for (TransferFunction tf : Values<TransferFunction>()) {
          if (tf == TransferFunction::kUnknown) continue;
          if (c.tf.SetImplicit() &&
              (c.tf.IsGamma() || c.tf.GetTransferFunction() != tf)) {
            continue;
          }
          c.tf.SetTransferFunction(tf);

          for (RenderingIntent ri : Values<RenderingIntent>()) {

            ColorEncodingDescriptor cdesc;
            cdesc.color_space = cs;
            cdesc.white_point = wp;
            cdesc.primaries = primaries;
            cdesc.tf = tf;
            cdesc.rendering_intent = ri;
            all_encodings.push_back(cdesc);
          }
        }
      }
    }
  }

  return all_encodings;
}

}  // namespace test
}  // namespace jxl

#endif  // JXL_TEST_UTILS_H_
