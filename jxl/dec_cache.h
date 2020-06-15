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

#ifndef JXL_DEC_CACHE_H_
#define JXL_DEC_CACHE_H_

#include <stdint.h>

#include "jxl/ac_strategy.h"
#include "jxl/coeff_order.h"
#include "jxl/common.h"
#include "jxl/dec_noise.h"
#include "jxl/image.h"
#include "jxl/passes_state.h"
#include "jxl/quant_weights.h"

namespace jxl {

// Line-based EPF only needs to keep in cache 13 lines of the image, so 256 is
// sufficient for everything to fit in the L2 cache.
constexpr size_t kApplyImageFeaturesTileDim = 256;

constexpr size_t kEpf1InputRows = 7;
constexpr size_t kEpf2InputRows = 3;

// Per-frame decoder state. All the images here should be accessed through a
// group rect (either with block units or pixel units).
struct PassesDecoderState {
  PassesSharedState shared_storage;
  // Allows avoiding copies for encoder loop.
  const PassesSharedState* JXL_RESTRICT shared = &shared_storage;

  // Storage for RNG output for noise synthesis.
  Image3F noise;

  // Pointer to previous/next frame, to be added to the current one, if any.
  // Gets updated by adding the currently decoded frame to it.
  Image3F* JXL_RESTRICT frame_storage = nullptr;

  // For ANS decoding.
  std::vector<ANSCode> code;
  std::vector<std::vector<uint8_t>> context_map;

  // Multiplier to be applied to the quant matrices of the x channel.
  float x_dm_multiplier;

  // Normalized weights for gaborish, in XYB order, each weight for Manhattan
  // distance of 0, 1 and 2 respectively.
  float gab_weights[9];

  // Decoded image, with padding.
  Image3F decoded;

  // Sigma values for EPF, if enabled.
  // Note that, for speed reasons, this is actually kInvSigmaNum / sigma.
  ImageF sigma;

  // Tile storage for ApplyImageFeatures steps. Storage1 has 2 blocks of padding
  // per side, storage2 has 1.
  std::vector<Image3F> storage1;
  std::vector<Image3F> storage2;

  void EnsureStorage(size_t num_threads) {
    for (size_t i = storage1.size(); i < num_threads; i++) {
      // We allocate twice what is needed since the last rects are larger in one
      // dimension.
      // Since we use row-based processing and cyclic addressing, we only need 7
      // rows in storage1 and 3 in storage2.
      storage1.emplace_back(kApplyImageFeaturesTileDim + 4 * kBlockDim,
                            kEpf1InputRows);
      storage2.emplace_back(kApplyImageFeaturesTileDim + 2 * kBlockDim,
                            kEpf2InputRows);
    }
  }

  // Initializes decoder-specific structures using information from *shared.
  void Init(ThreadPool* pool) {
    frame_storage = shared->multiframe->FrameStorage(
        shared->frame_dim.xsize_padded, shared->frame_dim.ysize_padded);

    if (shared->frame_header.color_transform == ColorTransform::kXYB) {
      x_dm_multiplier =
          std::pow(0.5f, 0.5f * shared->frame_header.x_qm_scale - 0.5f);
    } else {
      x_dm_multiplier = 1.0f;  // don't scale X quantization in YCbCr
    }

    if (shared->frame_header.flags & FrameHeader::kNoise) {
      noise = Image3F(shared->frame_dim.xsize_padded,
                      shared->frame_dim.ysize_padded);
      PROFILER_ZONE("GenerateNoise");
      auto generate_noise = [&](int group_index, int _) {
        RandomImage3(shared->PaddedGroupRect(group_index), &noise);
      };
      RunOnPool(pool, 0, shared->frame_dim.num_groups, ThreadPool::SkipInit(),
                generate_noise, "Generate noise");
    }

    const LoopFilter& lf = shared->image_features.loop_filter;
    if (lf.epf || lf.gab) {
      decoded = Image3F(shared->frame_dim.xsize_padded + 4 * kBlockDim,
                        shared->frame_dim.ysize_padded + 4 * kBlockDim);
#if MEMORY_SANITIZER
      // Avoid errors due to loading vectors on the outermost padding.
      ZeroFillImage(&decoded);
#endif
    }
    if (lf.epf) {
      sigma = ImageF(shared->frame_dim.xsize_blocks + 4,
                     shared->frame_dim.ysize_blocks + 4);
    }
    if (lf.gab) {
      lf.GaborishWeights(gab_weights);
    }
  }
};

// Temp images required for decoding a single group. Reduces memory allocations
// for large images because we only initialize min(#threads, #groups) instances.
struct GroupDecCache {
  void InitOnce(size_t num_passes) {
    PROFILER_FUNC;

    if (num_passes != 0 && num_nzeroes[0].xsize() == 0) {
      // Allocate enough for a whole group - partial groups on the right/bottom
      // border just use a subset. The valid size is passed via Rect.

      for (size_t i = 0; i < num_passes; i++) {
        num_nzeroes[i] = Image3I(kGroupDimInBlocks, kGroupDimInBlocks);
      }
    }
  }

  // AC decoding
  Image3I num_nzeroes[kMaxNumPasses];
};

}  // namespace jxl

#endif  // JXL_DEC_CACHE_H_
