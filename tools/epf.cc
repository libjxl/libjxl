// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/epf.h"

#include "lib/jxl/common.h"
#include "lib/jxl/dec_cache.h"
#include "lib/jxl/dec_reconstruct.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_color_management.h"
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
  (void)ToXYB(io->Main(), pool, &opsin, jxl::GetJxlCms());

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

  JXL_RETURN_IF_ERROR(state.filter_weights.Init(lf, frame_dim));
  ComputeSigma(jxl::Rect(state.shared_storage.epf_sharpness), &state);

  // Call with `force_fir` set to true to force to apply filters to all of the
  // input image.
  JXL_CHECK(FinalizeFrameDecoding(&io->Main(), &state, pool, /*force_fir=*/true,
                                  /*skip_blending=*/true, /*move_ec=*/true));
  JXL_RETURN_IF_ERROR(
      io->TransformTo(original_color_encoding, jxl::GetJxlCms(), pool));
  return true;
}

}  // namespace tools
}  // namespace jpegxl
