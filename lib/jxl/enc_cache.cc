// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_cache.h"

#include <stddef.h>
#include <stdint.h>

#include <memory>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/compressed_dc.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/enc_aux_out.h"
#include "lib/jxl/enc_frame.h"
#include "lib/jxl/enc_group.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_quant_weights.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quantizer.h"

namespace jxl {

Status InitializePassesEncoder(const FrameHeader& frame_header,
                               const Image3F& opsin, const Rect& rect,
                               const JxlCmsInterface& cms, ThreadPool* pool,
                               PassesEncoderState* enc_state,
                               ModularFrameEncoder* modular_frame_encoder,
                               AuxOut* aux_out) {
  PassesSharedState& JXL_RESTRICT shared = enc_state->shared;

  enc_state->x_qm_multiplier = std::pow(1.25f, frame_header.x_qm_scale - 2.0f);
  enc_state->b_qm_multiplier = std::pow(1.25f, frame_header.b_qm_scale - 2.0f);

  if (enc_state->coeffs.size() < frame_header.passes.num_passes) {
    enc_state->coeffs.reserve(frame_header.passes.num_passes);
    for (size_t i = enc_state->coeffs.size();
         i < frame_header.passes.num_passes; i++) {
      // Allocate enough coefficients for each group on every row.
      JXL_ASSIGN_OR_RETURN(
          std::unique_ptr<ACImageT<int32_t>> coeffs,
          ACImageT<int32_t>::Make(kGroupDim * kGroupDim,
                                  shared.frame_dim.num_groups));
      enc_state->coeffs.emplace_back(std::move(coeffs));
    }
  }
  while (enc_state->coeffs.size() > frame_header.passes.num_passes) {
    enc_state->coeffs.pop_back();
  }

  if (enc_state->initialize_global_state) {
    float scale =
        shared.quantizer.ScaleGlobalScale(enc_state->cparams.quant_ac_rescale);
    DequantMatricesScaleDC(&shared.matrices, scale);
    shared.quantizer.RecomputeFromGlobalScale();
  }

  JXL_ASSIGN_OR_RETURN(Image3F dc,
                       Image3F::Create(shared.frame_dim.xsize_blocks,
                                       shared.frame_dim.ysize_blocks));
  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, shared.frame_dim.num_groups, ThreadPool::NoInit,
      [&](size_t group_idx, size_t _) {
        ComputeCoefficients(group_idx, enc_state, opsin, rect, &dc);
      },
      "Compute coeffs"));

  if (frame_header.flags & FrameHeader::kUseDcFrame) {
    CompressParams cparams = enc_state->cparams;
    cparams.dots = Override::kOff;
    cparams.noise = Override::kOff;
    cparams.patches = Override::kOff;
    cparams.gaborish = Override::kOff;
    cparams.epf = 0;
    cparams.resampling = 1;
    cparams.ec_resampling = 1;
    // The DC frame will have alpha=0. Don't erase its contents.
    cparams.keep_invisible = Override::kOn;
    JXL_ASSERT(cparams.progressive_dc > 0);
    cparams.progressive_dc--;
    // Use kVarDCT in max_error_mode for intermediate progressive DC,
    // and kModular for the smallest DC (first in the bitstream)
    if (cparams.progressive_dc == 0) {
      cparams.modular_mode = true;
      cparams.speed_tier =
          SpeedTier(std::max(static_cast<int>(SpeedTier::kTortoise),
                             static_cast<int>(cparams.speed_tier) - 1));
      cparams.butteraugli_distance =
          std::max(kMinButteraugliDistance,
                   enc_state->cparams.butteraugli_distance * 0.02f);
    } else {
      cparams.max_error_mode = true;
      for (size_t c = 0; c < 3; c++) {
        cparams.max_error[c] = shared.quantizer.MulDC()[c];
      }
      // Guess a distance that produces good initial results.
      cparams.butteraugli_distance =
          std::max(kMinButteraugliDistance,
                   enc_state->cparams.butteraugli_distance * 0.1f);
    }
    ImageBundle ib(&shared.metadata->m);
    // This is a lie - dc is in XYB
    // (but EncodeFrame will skip RGB->XYB conversion anyway)
    ib.SetFromImage(
        std::move(dc),
        ColorEncoding::LinearSRGB(shared.metadata->m.color_encoding.IsGray()));
    if (!ib.metadata()->extra_channel_info.empty()) {
      // Add placeholder extra channels to the patch image: dc_level frames do
      // not yet support extra channels, but the codec expects that the amount
      // of extra channels in frames matches that in the metadata of the
      // codestream.
      std::vector<ImageF> extra_channels;
      extra_channels.reserve(ib.metadata()->extra_channel_info.size());
      for (size_t i = 0; i < ib.metadata()->extra_channel_info.size(); i++) {
        JXL_ASSIGN_OR_RETURN(ImageF ch, ImageF::Create(ib.xsize(), ib.ysize()));
        extra_channels.emplace_back(std::move(ch));
        // Must initialize the image with data to not affect blending with
        // uninitialized memory.
        // TODO(lode): dc_level must copy and use the real extra channels
        // instead.
        ZeroFillImage(&extra_channels.back());
      }
      ib.SetExtraChannels(std::move(extra_channels));
    }
    auto special_frame = std::unique_ptr<BitWriter>(new BitWriter());
    FrameInfo dc_frame_info;
    dc_frame_info.frame_type = FrameType::kDCFrame;
    dc_frame_info.dc_level = frame_header.dc_level + 1;
    dc_frame_info.ib_needs_color_transform = false;
    dc_frame_info.save_before_color_transform = true;  // Implicitly true
    AuxOut dc_aux_out;
    JXL_CHECK(EncodeFrame(cparams, dc_frame_info, shared.metadata, ib, cms,
                          pool, special_frame.get(),
                          aux_out ? &dc_aux_out : nullptr));
    if (aux_out) {
      for (const auto& l : dc_aux_out.layers) {
        aux_out->layers[kLayerDC].Assimilate(l);
      }
    }
    const Span<const uint8_t> encoded = special_frame->GetSpan();
    enc_state->special_frames.emplace_back(std::move(special_frame));

    ImageBundle decoded(&shared.metadata->m);
    std::unique_ptr<PassesDecoderState> dec_state =
        jxl::make_unique<PassesDecoderState>();
    JXL_CHECK(
        dec_state->output_encoding_info.SetFromMetadata(*shared.metadata));
    const uint8_t* frame_start = encoded.data();
    size_t encoded_size = encoded.size();
    for (int i = 0; i <= cparams.progressive_dc; ++i) {
      JXL_CHECK(DecodeFrame(dec_state.get(), pool, frame_start, encoded_size,
                            /*frame_header=*/nullptr, &decoded,
                            *shared.metadata));
      frame_start += decoded.decoded_bytes();
      encoded_size -= decoded.decoded_bytes();
    }
    // TODO(lode): frame_header.dc_level should be equal to
    // dec_state.frame_header.dc_level - 1 here, since above we set
    // dc_frame_info.dc_level = frame_header.dc_level + 1, and
    // dc_frame_info.dc_level is used by EncodeFrame. However, if EncodeFrame
    // outputs multiple frames, this assumption could be wrong.
    const Image3F& dc_frame =
        dec_state->shared->dc_frames[frame_header.dc_level];
    JXL_ASSIGN_OR_RETURN(shared.dc_storage,
                         Image3F::Create(dc_frame.xsize(), dc_frame.ysize()));
    CopyImageTo(dc_frame, &shared.dc_storage);
    ZeroFillImage(&shared.quant_dc);
    shared.dc = &shared.dc_storage;
    JXL_CHECK(encoded_size == 0);
  } else {
    std::atomic<bool> has_error{false};
    auto compute_dc_coeffs = [&](int group_index, int /* thread */) {
      if (has_error) return;
      const Rect r = enc_state->shared.frame_dim.DCGroupRect(group_index);
      int modular_group_index = group_index;
      if (enc_state->streaming_mode) {
        JXL_ASSERT(group_index == 0);
        modular_group_index = enc_state->dc_group_index;
      }
      if (!modular_frame_encoder->AddVarDCTDC(
              frame_header, dc, r, modular_group_index,
              enc_state->cparams.speed_tier < SpeedTier::kFalcon, enc_state,
              /*jpeg_transcode=*/false)) {
        has_error = true;
        return;
      }
    };
    JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_dc_groups,
                                  ThreadPool::NoInit, compute_dc_coeffs,
                                  "Compute DC coeffs"));
    if (has_error) return JXL_FAILURE("Compute DC coeffs failed");
    // TODO(veluca): this is only useful in tests and if inspection is enabled.
    if (!(frame_header.flags & FrameHeader::kSkipAdaptiveDCSmoothing)) {
      JXL_RETURN_IF_ERROR(AdaptiveDCSmoothing(shared.quantizer.MulDC(),
                                              &shared.dc_storage, pool));
    }
  }
  std::atomic<bool> has_error{false};
  auto compute_ac_meta = [&](int group_index, int /* thread */) {
    const Rect r = enc_state->shared.frame_dim.DCGroupRect(group_index);
    int modular_group_index = group_index;
    if (enc_state->streaming_mode) {
      JXL_ASSERT(group_index == 0);
      modular_group_index = enc_state->dc_group_index;
    }
    if (!modular_frame_encoder->AddACMetadata(r, modular_group_index,
                                              /*jpeg_transcode=*/false,
                                              enc_state)) {
      has_error = true;
      return;
    }
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, shared.frame_dim.num_dc_groups,
                                ThreadPool::NoInit, compute_ac_meta,
                                "Compute AC Metadata"));
  if (has_error) return JXL_FAILURE("Compute AC Metadata failed");
  return true;
}

}  // namespace jxl
