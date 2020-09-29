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

#include "jxl/dct_util.h"

#include <stdint.h>

#include <string>

#include "gtest/gtest.h"
#include "jxl/base/bits.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/codec_in_out.h"
#include "jxl/dct_scales.h"
#include "jxl/dec_dct.h"
#include "jxl/enc_dct.h"
#include "jxl/enc_xyb.h"
#include "jxl/extras/codec.h"
#include "jxl/image_ops.h"
#include "jxl/image_test_utils.h"
#include "jxl/testdata.h"

namespace jxl {
namespace {

static Image3F OpsinTestImage() {
  const PaddedBytes orig =
      ReadTestData("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  JXL_CHECK(SetFromBytes(Span<const uint8_t>(orig), &io, /*pool=*/nullptr));
  ThreadPool* null_pool = nullptr;
  Image3F opsin(io.xsize(), io.ysize());
  ImageBundle unused_linear;
  (void)ToXYB(io.Main(), null_pool, &opsin, &unused_linear);
  opsin.ShrinkTo(opsin.ysize() & ~7, opsin.xsize() & ~7);
  return opsin;
}

TEST(DctUtilTest, DCTRoundtrip) {
  Image3F opsin = OpsinTestImage();
  const size_t xsize_blocks = opsin.xsize() / kBlockDim;
  const size_t ysize_blocks = opsin.ysize() / kBlockDim;

  Image3F coeffs(xsize_blocks * kDCTBlockSize, ysize_blocks);
  Image3F recon(xsize_blocks * kBlockDim, ysize_blocks * kBlockDim);

  TransposedScaledDCT(opsin, &coeffs);
  TransposedScaledIDCT(coeffs, &recon);
  VerifyRelativeError(opsin, recon, 1e-6, 1e-6);
}

}  // namespace
}  // namespace jxl
