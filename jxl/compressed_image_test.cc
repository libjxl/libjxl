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

#include <algorithm>
#include <string>
#include <utility>

#include "gtest/gtest.h"
#include "jxl/ac_strategy.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/base/thread_pool_internal.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/common.h"
#include "jxl/enc_adaptive_quantization.h"
#include "jxl/enc_butteraugli_comparator.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_params.h"
#include "jxl/enc_xyb.h"
#include "jxl/extras/codec.h"
#include "jxl/frame_header.h"
#include "jxl/gaborish.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/loop_filter.h"
#include "jxl/multiframe.h"
#include "jxl/passes_state.h"
#include "jxl/quant_weights.h"
#include "jxl/quantizer.h"
#include "jxl/testdata.h"

namespace jxl {
namespace {

// Verifies ReconOpsinImage reconstructs with low butteraugli distance.
void RunRGBRoundTrip(float distance, bool fast) {
  ThreadPoolInternal pool(4);

  const PaddedBytes orig =
      ReadTestData("wesaturate/500px/u76c0g_bliznaca_srgb8.png");
  CodecInOut io;
  JXL_CHECK(SetFromBytes(Span<const uint8_t>(orig), &io, &pool));
  // This test can only handle a single group.
  io.ShrinkTo(std::min(io.xsize(), kGroupDim), std::min(io.ysize(), kGroupDim));

  Image3F opsin(io.xsize(), io.ysize());
  ImageBundle unused_linear;
  (void)ToXYB(io.Main(), &pool, &opsin, &unused_linear);
  opsin = PadImageToMultiple(opsin, kBlockDim);
  opsin = GaborishInverse(opsin, 1.0f, &pool);

  CompressParams cparams;
  cparams.butteraugli_distance = distance;
  if (fast) {
    cparams.speed_tier = SpeedTier::kWombat;
  }

  FrameHeader frame_header;
  frame_header.color_transform = ColorTransform::kXYB;
  frame_header.animation_frame.nonserialized_have_timecode = false;
  LoopFilter loop_filter;
  loop_filter.gab = true;
  loop_filter.gab_custom = true;
  loop_filter.gab_x_weight1 = 0.11501538179658321;
  loop_filter.gab_x_weight2 = 0.089979079587015454;
  loop_filter.gab_y_weight1 = 0.11501538179658321;
  loop_filter.gab_y_weight2 = 0.089979079587015454;
  loop_filter.gab_b_weight1 = 0.11501538179658321;
  loop_filter.gab_b_weight2 = 0.089979079587015454;
  loop_filter.epf_iters = 0;

  FrameDimensions frame_dim;
  frame_dim.Set(opsin.xsize(), opsin.ysize(), /*group_size_shift=*/1,
                /*max_hshift=*/0, /*max_vshift=*/0);
  Multiframe multiframe;
  PassesEncoderState enc_state;
  JXL_CHECK(InitializePassesSharedState(frame_header, loop_filter, io.metadata,
                                        frame_dim, &multiframe,
                                        &enc_state.shared));

  enc_state.x_qm_multiplier = 1.0f;
  enc_state.shared.quantizer.SetQuant(4.0f, 4.0f,
                                      &enc_state.shared.raw_quant_field);
  enc_state.shared.ac_strategy.FillDCT8();
  enc_state.cparams = cparams;
  ZeroFillImage(&enc_state.shared.epf_sharpness);
  Image3F recon = RoundtripImage(opsin, &enc_state, &pool);

  CodecInOut io1;
  io1.metadata.bit_depth = io.metadata.bit_depth;
  io1.metadata.color_encoding = ColorEncoding::LinearSRGB();
  io1.SetFromImage(std::move(recon), io1.metadata.color_encoding);

  EXPECT_LE(ButteraugliDistance(io, io1, cparams.ba_params,
                                /*distmap=*/nullptr, &pool),
            1.2);
}

TEST(CompressedImageTest, RGBRoundTrip_1) { RunRGBRoundTrip(1.0, false); }

TEST(CompressedImageTest, RGBRoundTrip_1_fast) { RunRGBRoundTrip(1.0, true); }

TEST(CompressedImageTest, RGBRoundTrip_2) { RunRGBRoundTrip(2.0, false); }

TEST(CompressedImageTest, RGBRoundTrip_2_fast) { RunRGBRoundTrip(2.0, true); }

}  // namespace
}  // namespace jxl
