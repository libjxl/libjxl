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

#ifndef JXL_ENC_CACHE_H_
#define JXL_ENC_CACHE_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "jxl/ac_strategy.h"
#include "jxl/aux_out.h"
#include "jxl/base/data_parallel.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/coeff_order.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/common.h"
#include "jxl/dct_util.h"
#include "jxl/dot_dictionary.h"
#include "jxl/enc_ans.h"
#include "jxl/enc_params.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/passes_state.h"
#include "jxl/patch_dictionary.h"
#include "jxl/quant_weights.h"
#include "jxl/quantizer.h"

namespace jxl {

// Contains encoder state.
struct PassesEncoderState {
  PassesSharedState shared;

  ImageF initial_quant_field;  // Invalid in Falcon mode.

  // Per-pass DCT coefficients for the image. One row per group.
  // Used for both quantized and non-quantized coefficients (only coeffs[0]).
  // WARNING: assumes ac_qcoeff_t == float!
  std::vector<ACImage3> coeffs;

  // Vector of tokens for each DC group.
  std::vector<std::vector<Token>> dc_tokens;

  // Raw data for special (reference+DC) frames.
  std::vector<BitWriter> special_frames;

  // Quantized DC.
  Image3S dc;

  // Number of extra DC levels, per group.
  std::vector<int> extra_dc_levels;

  // Storage for reference frames. More than one to allow for photographic and
  // non-photographic patches, as well as mixing previous frames and special
  // frames as sources.
  Image3F reference_frames[kMaxNumReferenceFrames];

  // Per-dc16-group-tokens.
  std::vector<std::vector<Token>> downsampled_dc_tokens;

  CompressParams cparams;

  struct PassData {
    std::vector<std::vector<Token>> ac_tokens;
    std::vector<uint8_t> context_map;
    EntropyEncodingData codes;
  };

  std::vector<PassData> passes;

  // Multiplier to be applied to the quant matrices of the x channel.
  float x_qm_multiplier = 1.0f;
};

// Initialize per-frame information.
void InitializePassesEncoder(const Image3F& opsin, ThreadPool* pool,
                             PassesEncoderState* passes_enc_state,
                             AuxOut* aux_out);

// Working area for ComputeCoefficients (per-group!)
struct EncCache {
  // Allocates memory when first called, shrinks images to current group size.
  void InitOnce();

  // TokenizeCoefficients
  Image3I num_nzeroes;
};

}  // namespace jxl

#endif  // JXL_ENC_CACHE_H_
