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

#include "jxl/enc_cache.h"

#include <stddef.h>
#include <stdint.h>

#include <type_traits>

#include "jxl/ac_strategy.h"
#include "jxl/aux_out.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/base/span.h"
#include "jxl/color_encoding.h"
#include "jxl/common.h"
#include "jxl/compressed_dc.h"
#include "jxl/dct_util.h"
#include "jxl/dec_frame.h"
#include "jxl/enc_frame.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/passes_state.h"
#include "jxl/quantizer.h"

namespace jxl {

void InitializePassesEncoder(const Image3F& opsin, ThreadPool* pool,
                             PassesEncoderState* enc_state, AuxOut* aux_out) {
  PROFILER_FUNC;
  constexpr size_t N = kBlockDim;

  PassesSharedState& JXL_RESTRICT shared = enc_state->shared;

  const size_t xsize_blocks = shared.frame_dim.xsize_blocks;
  const size_t ysize_blocks = shared.frame_dim.ysize_blocks;

  if (shared.frame_header.color_transform == ColorTransform::kXYB) {
    enc_state->x_qm_multiplier =
        std::pow(2.0f, 0.5f * shared.frame_header.x_qm_scale - 0.5f);
  } else {
    enc_state->x_qm_multiplier = 1.0f;  // don't scale X quantization in YCbCr
  }

  if (enc_state->coeffs.size() != shared.frame_header.passes.num_passes) {
    enc_state->coeffs.resize(shared.frame_header.passes.num_passes);
    for (size_t i = 0; i < shared.frame_header.passes.num_passes; i++) {
      static_assert(std::is_same<float, ac_qcoeff_t>::value,
                    "float != ac_coeff_t");
      // Allocate enough coefficients for each group on every row.
      enc_state->coeffs[i] =
          ACImage3(kGroupDim * kGroupDim, shared.frame_dim.num_groups);
    }
  }
  Image3F dc = Image3F(xsize_blocks, ysize_blocks);

  size_t dc_stride = dc.PixelsPerRow();
  size_t opsin_stride = opsin.PixelsPerRow();

  auto compute_coeffs = [&](int group_index, int /* thread */) {
    const size_t gx = group_index % shared.frame_dim.xsize_groups;
    const size_t gy = group_index / shared.frame_dim.xsize_groups;
    size_t offset = 0;
    float* JXL_RESTRICT rows[3] = {
        enc_state->coeffs[0].PlaneRow(0, group_index),
        enc_state->coeffs[0].PlaneRow(1, group_index),
        enc_state->coeffs[0].PlaneRow(2, group_index),
    };
    for (size_t by = gy * kGroupDimInBlocks;
         by < ysize_blocks && by < (gy + 1) * kGroupDimInBlocks; ++by) {
      const float* JXL_RESTRICT opsin_rows[3] = {
          opsin.ConstPlaneRow(0, by * N),
          opsin.ConstPlaneRow(1, by * N),
          opsin.ConstPlaneRow(2, by * N),
      };
      float* JXL_RESTRICT dc_rows[3] = {
          dc.PlaneRow(0, by),
          dc.PlaneRow(1, by),
          dc.PlaneRow(2, by),
      };
      for (size_t bx = gx * kGroupDimInBlocks;
           bx < xsize_blocks && bx < (gx + 1) * kGroupDimInBlocks; ++bx) {
        AcStrategy acs = shared.ac_strategy.ConstRow(by)[bx];
        if (!acs.IsFirstBlock()) continue;
        for (size_t c = 0; c < 3; ++c) {
          acs.TransformFromPixels(opsin_rows[c] + bx * N, opsin_stride,
                                  rows[c] + offset);
          acs.DCFromLowestFrequencies(rows[c] + offset, dc_rows[c] + bx,
                                      dc_stride);
        }
        offset += kDCTBlockSize << acs.log2_covered_blocks();
      }
    }
  };

  RunOnPool(pool, 0, shared.frame_dim.num_groups, ThreadPool::SkipInit(),
            compute_coeffs, "Compute coeffs");
  if (aux_out != nullptr) {
    aux_out->InspectImage3F("compressed_image:InitializeFrameEncCache:dc", dc);
  }

  if (shared.frame_header.flags & FrameHeader::kUseDcFrame) {
    CompressParams cparams = enc_state->cparams;
    // Guess a distance that produces good initial results.
    cparams.butteraugli_distance =
        std::max(kMinButteraugliDistance,
                 enc_state->cparams.butteraugli_distance * 0.1f);
    cparams.dots = Override::kOff;
    cparams.patches = Override::kOff;
    cparams.gaborish = Override::kOff;
    cparams.adaptive_reconstruction = Override::kOff;
    cparams.max_error_mode = true;
    for (size_t c = 0; c < 3; c++) {
      cparams.max_error[c] = shared.quantizer.MulDC()[c];
    }
    FrameDimensions frame_dim;
    frame_dim.Set(enc_state->shared.frame_dim.xsize << (3 * cparams.dc_level),
                  enc_state->shared.frame_dim.ysize << (3 * cparams.dc_level));
    cparams.dc_level++;
    cparams.progressive_dc--;
    // Use kVarDCT in max_error_mode for intermediate progressive DC,
    // and kModularGroup for the smallest DC (first in the bitstream)
    if (cparams.progressive_dc == 0) {
      cparams.modular_group_mode = true;
      cparams.quality_pair.first = cparams.quality_pair.second =
          99.f - enc_state->cparams.butteraugli_distance * 2.f;
    }
    ImageMetadata metadata;
    metadata.color_encoding = ColorEncoding::LinearSRGB();
    ImageBundle ib(&metadata);
    // This is a lie - dc is in XYB
    // (but EncodeFrame will skip RGB->XYB conversion anyway)
    ib.SetFromImage(std::move(dc), ColorEncoding::LinearSRGB());
    PassesEncoderState state;
    enc_state->special_frames.emplace_back();
    JXL_CHECK(EncodeFrame(cparams, nullptr, ib, &state, pool,
                          &enc_state->special_frames.back(), nullptr,
                          enc_state->shared.multiframe));
    const Span<const uint8_t> encoded =
        enc_state->special_frames.back().GetSpan();
    BitReader br(encoded);
    ImageBundle decoded(&metadata);
    JXL_CHECK(DecodeFrame({}, encoded, nullptr, &frame_dim, shared.multiframe,
                          pool, &br, nullptr, &decoded));
    shared.dc_storage =
        CopyImage(*shared.multiframe->SavedDc(cparams.dc_level));
    JXL_CHECK(br.Close());
  } else {
    const size_t xsize_dc_groups = DivCeil(xsize_blocks, kDcGroupDimInBlocks);
    const size_t ysize_dc_groups = DivCeil(ysize_blocks, kDcGroupDimInBlocks);

    enc_state->dc_tokens =
        std::vector<std::vector<Token>>(xsize_dc_groups * ysize_dc_groups);
    // Disable extra levels when compressing very high quality images: this
    // allows us to use a finer quantization step.
    enc_state->extra_dc_levels.resize(
        xsize_dc_groups * ysize_dc_groups,
        (enc_state->cparams.butteraugli_distance < 0.5 ? 0 : 1));

    auto compute_dc_coeffs = [&](int group_index, int /* thread */) {
      TokenizeDC(group_index, dc, enc_state, aux_out);
    };
    RunOnPool(pool, 0, shared.frame_dim.num_dc_groups, ThreadPool::SkipInit(),
              compute_dc_coeffs, "Compute DC coeffs");
    // TODO(veluca): this is only useful in tests and if inspection is enabled.
    if ((shared.frame_header.flags & FrameHeader::kSkipAdaptiveDCSmoothing) ==
        0) {
      AdaptiveDCSmoothing(shared.dc_quant_field, &shared.dc_storage, pool);
    }
  }

  if (aux_out != nullptr) {
    aux_out->InspectImage3F("compressed_image:InitializeFrameEncCache:dc_dec",
                            shared.dc_storage);
  }
}

void EncCache::InitOnce() {
  PROFILER_FUNC;

  if (num_nzeroes.xsize() == 0) {
    num_nzeroes = Image3I(kGroupDimInBlocks, kGroupDimInBlocks);
  }
}

}  // namespace jxl
