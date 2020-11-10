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

#include <string>

#include "gtest/gtest.h"
#include "lib/extras/codec.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/dec_params.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_file.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_test_utils.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testdata.h"

namespace jxl {
namespace {

struct CompressedDCTestParams {
  explicit CompressedDCTestParams(const double butteraugli_distance,
                                  const bool fast_mode = false,
                                  const bool shrink8 = false)
      : butteraugli_distance(butteraugli_distance),
        fast_mode(fast_mode),
        shrink8(shrink8) {}
  double butteraugli_distance;
  bool fast_mode;
  bool shrink8;
};

std::ostream& operator<<(std::ostream& os, CompressedDCTestParams params) {
  auto previous_flags = os.flags();
  os << std::boolalpha;
  os << "CompressedDCTestParams{/*butteraugli_distance=*/"
     << params.butteraugli_distance << ", /*fast_mode=*/" << params.fast_mode
     << ", /*shrink8=*/" << params.shrink8 << "}";
  os.flags(previous_flags);
  return os;
}

class CompressedDCTest : public testing::TestWithParam<CompressedDCTestParams> {
 protected:
  // Returns compressed size [bytes].
  static void Roundtrip(CodecInOut* io, const CompressParams& cparams,
                        const DecompressParams& dparams, ThreadPool* pool) {
    PaddedBytes compressed;

    Image3F encoding_dc;
    Image3F encoding_dec;
    AuxOut encoding_info;
    encoding_info.testing_aux.dc = &encoding_dc;
    encoding_info.testing_aux.decoded = &encoding_dec;

    PassesEncoderState enc_state;
    EXPECT_TRUE(
        EncodeFile(cparams, io, &enc_state, &compressed, &encoding_info, pool));

    Image3F decoding_dc;
    AuxOut decoding_info;
    decoding_info.testing_aux.dc = &decoding_dc;

    EXPECT_TRUE(DecodeFile(dparams, compressed, io, &decoding_info, pool));

    // Without FMA, 1E-6 is sufficient.
    const float kErrorThreshold = 7e-6f;
    VerifyRelativeError(decoding_dc, encoding_dc, kErrorThreshold,
                        kErrorThreshold);
    VerifyRelativeError(*io->Main().color(), encoding_dec, kErrorThreshold,
                        kErrorThreshold);
  }
};

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(
    CompressedDCTestInstantiation, CompressedDCTest,
    testing::Values(CompressedDCTestParams{/*butteraugli_distance=*/1.0,
                                           /*fast_mode=*/false,
                                           /*shrink8=*/true},
                    CompressedDCTestParams{/*butteraugli_distance=*/1.0},
                    CompressedDCTestParams{/*butteraugli_distance=*/1.5},
                    CompressedDCTestParams{/*butteraugli_distance=*/2.0},
                    CompressedDCTestParams{/*butteraugli_distance=*/1.0,
                                           /*fast_mode=*/true},
                    CompressedDCTestParams{/*butteraugli_distance=*/1.5,
                                           /*fast_mode=*/true},
                    CompressedDCTestParams{/*butteraugli_distance=*/2.0,
                                           /*fast_mode=*/true}));

TEST_P(CompressedDCTest, Roundtrip) {
  const PaddedBytes orig =
      ReadTestData("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  ThreadPoolInternal pool(8);
  ASSERT_TRUE(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));

  const CompressedDCTestParams& params = GetParam();

  if (params.shrink8) {
    io.ShrinkTo(io.xsize() / 8, io.ysize() / 8);
  }

  CompressParams cparams;
  cparams.butteraugli_distance = params.butteraugli_distance;
  if (params.fast_mode) {
    cparams.speed_tier = SpeedTier::kSquirrel;
  }
  DecompressParams dparams;

  Roundtrip(&io, cparams, dparams, &pool);
}

}  // namespace
}  // namespace jxl
