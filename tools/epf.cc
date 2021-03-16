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

#include "tools/epf.h"

#include "lib/jxl/common.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_reconstruct.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/epf.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image_ops.h"

using jxl::kSigmaBorder;
using jxl::kSigmaPadding;

namespace jpegxl {
namespace tools {

jxl::Status RunEPF(uint32_t epf_iters, const float distance,
                   const int sharpness_parameter, jxl::CodecInOut* const io,
                   jxl::ThreadPool* const pool) {
  const jxl::ColorEncoding original_color_encoding =
      io->metadata.m.color_encoding;
  jxl::Image3F opsin(io->xsize(), io->ysize());
  (void)ToXYB(io->Main(), pool, &opsin);

  JXL_CHECK(io->metadata.size.Set(opsin.xsize(), opsin.ysize()));

  opsin = PadImageToMultiple(opsin, jxl::kBlockDim);

  jxl::FrameHeader frame_header(&io->metadata);
  jxl::LoopFilter& lf = frame_header.loop_filter;
  lf.gab = false;
  lf.epf_iters = epf_iters;
  jxl::FrameDimensions frame_dim = frame_header.ToFrameDimensions();

  static constexpr float kAcQuant = 0.84f;
  const float dc_quant = jxl::InitialQuantDC(distance);
  const float ac_quant = kAcQuant / distance;
  jxl::PassesDecoderState state;
  JXL_RETURN_IF_ERROR(
      jxl::InitializePassesSharedState(frame_header, &state.shared_storage));
  state.shared_storage.ac_strategy.FillDCT8();
  state.shared_storage.raw_quant_field =
      jxl::ImageI(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  state.shared_storage.quantizer.SetQuant(
      dc_quant, ac_quant, &state.shared_storage.raw_quant_field);
  state.shared_storage.epf_sharpness =
      jxl::ImageB(frame_dim.xsize_blocks, frame_dim.ysize_blocks);
  jxl::FillImage(static_cast<uint8_t>(sharpness_parameter),
                 &state.shared_storage.epf_sharpness);

  state.filter_weights.Init(lf, frame_dim);
  ComputeSigma(jxl::Rect(state.shared_storage.epf_sharpness), &state);

  // Call with `force_fir` set to true to force to apply filters to all of the
  // input image.
  JXL_CHECK(FinalizeFrameDecoding(&io->Main(), &state, pool, /*force_fir=*/true,
                                  /*skip_blending=*/true));
  JXL_RETURN_IF_ERROR(io->TransformTo(original_color_encoding, pool));
  return true;
}

}  // namespace tools
}  // namespace jpegxl
