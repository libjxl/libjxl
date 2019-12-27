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
#include "jxl/dec_file.h"
#include "jxl/dec_params.h"
#include "jxl/enc_file.h"
#include "jxl/enc_params.h"

#ifdef JXL_DISABLE_SLOW_TESTS
#define JXL_SLOW_TEST(X) DISABLED_##X
#else
#define JXL_SLOW_TEST(X) X
#endif  // JXL_DISABLE_SLOW_TESTS

namespace jxl {
namespace test {
// Returns compressed size [bytes].
size_t Roundtrip(CodecInOut* io, const CompressParams& cparams,
                 const DecompressParams& dparams, ThreadPool* pool,
                 CodecInOut* JXL_RESTRICT io2, AuxOut* aux_out = nullptr) {
  const ImageBundle& ib1 = io->Main();
  PaddedBytes compressed;

  // Remember original encoding, will be returned by decoder.
  const Primaries ext_pr = ib1.metadata()->color_encoding.primaries;
  const CustomTransferFunction ext_tf = ib1.metadata()->color_encoding.tf;
  // c_current should not change during encoding.
  const Primaries cur_pr = ib1.c_current().primaries;
  const CustomTransferFunction cur_tf = ib1.c_current().tf;

  PassesEncoderState enc_state;
  EXPECT_TRUE(EncodeFile(cparams, io, &enc_state, &compressed, aux_out, pool));

  // Should still be in the same color space after encoding.
  EXPECT_EQ(cur_pr, ib1.c_current().primaries);
  EXPECT_TRUE(cur_tf.IsSame(ib1.c_current().tf));

  EXPECT_TRUE(DecodeFile(dparams, compressed, io2, aux_out, pool));
  const ImageBundle& ib2 = io2->Main();

  if (!cparams.modular_group_mode) {
    // Non-modular returns linear sRGB.
    EXPECT_EQ(Primaries::kSRGB, ib2.c_current().primaries);
    EXPECT_TRUE(ib2.c_current().tf.IsLinear());
  } else {
    // Modular returns the original color space.
    EXPECT_EQ(ib1.c_current().primaries, ib2.c_current().primaries);
    EXPECT_TRUE(ib1.c_current().tf.IsSame(ib2.c_current().tf));
  }

  // Decoder returns the originals passed to the encoder.
  EXPECT_EQ(ext_pr, ib2.metadata()->color_encoding.primaries);
  EXPECT_TRUE(ext_tf.IsSame(ib2.metadata()->color_encoding.tf));

  return compressed.size();
}

}  // namespace test
}  // namespace jxl

#endif  // JXL_TEST_UTILS_H_
