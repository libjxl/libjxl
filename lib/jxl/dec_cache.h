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

#ifndef LIB_JXL_DEC_CACHE_H_
#define LIB_JXL_DEC_CACHE_H_

#include <stdint.h>

#include <hwy/base.h>  // HWY_ALIGN_MAX

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/common.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/dec_noise.h"
#include "lib/jxl/filters.h"
#include "lib/jxl/image.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"

namespace jxl {

// Per-frame decoder state. All the images here should be accessed through a
// group rect (either with block units or pixel units).
struct PassesDecoderState {
  PassesSharedState shared_storage;
  // Allows avoiding copies for encoder loop.
  const PassesSharedState* JXL_RESTRICT shared = &shared_storage;

  // Whether some AC groups are only partially present. This implies that we
  // need to run ApplyImageFeatures at the end, and not per-group.
  bool has_partial_ac_groups = false;

  // Storage for RNG output for noise synthesis.
  Image3F noise;

  // For ANS decoding.
  std::vector<ANSCode> code;
  std::vector<std::vector<uint8_t>> context_map;

  // Multiplier to be applied to the quant matrices of the x channel.
  float x_dm_multiplier;
  float b_dm_multiplier;

  // Padded decoded image. This image has two blocks of padding on each left and
  // and right sides and xsize() rounded up to a block size, but it has no
  // vertical padding.
  Image3F decoded;
  size_t decoded_padding = kMaxFilterPadding;

  // Seed for noise, to have different noise per-frame.
  size_t noise_seed = 0;

  // Filter application pipeline used by ApplyImageFeatures. One entry is needed
  // per thread.
  std::vector<FilterPipeline> filter_pipelines;

  // Input weights used by the filters. These are shared from multiple threads
  // but are read-only for the filter application.
  FilterWeights filter_weights;

  // Hook to do colorspace transforms to an ImageBundle.
  std::function<Status(ImageBundle*, const ColorEncoding&, ThreadPool*)>
      do_colorspace_transform = nullptr;

  void EnsureStorage(size_t num_threads) {
    // TODO(deymo): Don't request any memory if there's no need to apply any
    // filter.

    // We need one filter_storage per thread, ensure we have at least that many.
    if (filter_pipelines.size() < num_threads) {
      filter_pipelines.resize(num_threads);
    }
  }

  // Initializes decoder-specific structures using information from *shared.
  void Init(ThreadPool* pool) {
    x_dm_multiplier =
        std::pow(1 / (1.25f), shared->frame_header.x_qm_scale - 2.0f);
    b_dm_multiplier =
        std::pow(1 / (1.25f), shared->frame_header.b_qm_scale - 2.0f);

    if (shared->frame_header.flags & FrameHeader::kNoise) {
      noise = Image3F(shared->frame_dim.xsize_padded,
                      shared->frame_dim.ysize_padded);
      PROFILER_ZONE("GenerateNoise");
      auto generate_noise = [&](int group_index, int _) {
        RandomImage3(noise_seed + group_index,
                     shared->PaddedGroupRect(group_index), &noise);
      };
      RunOnPool(pool, 0, shared->frame_dim.num_groups, ThreadPool::SkipInit(),
                generate_noise, "Generate noise");
      {
        PROFILER_ZONE("High pass noise");
        // 4 * (1 - box kernel)
        WeightsSymmetric5 weights{{HWY_REP4(-3.84)}, {HWY_REP4(0.16)},
                                  {HWY_REP4(0.16)},  {HWY_REP4(0.16)},
                                  {HWY_REP4(0.16)},  {HWY_REP4(0.16)}};
        // TODO(veluca): avoid copy.
        // TODO(veluca): avoid having a full copy of the image in main memory.
        ImageF noise_tmp(noise.xsize(), noise.ysize());
        for (size_t c = 0; c < 3; c++) {
          Symmetric5(noise.Plane(c), Rect(noise), weights, pool, &noise_tmp);
          std::swap(noise.Plane(c), noise_tmp);
        }
        noise_seed += shared->frame_dim.num_groups;
      }
    }

    // decoded must be padded to a multiple of kBlockDim rows since the last
    // rows may be used by the filters even if they are outside the frame
    // dimension.
    decoded = Image3F(shared->frame_dim.xsize_padded + 2 * decoded_padding,
                      shared->frame_dim.ysize_padded);
#if MEMORY_SANITIZER
    // Avoid errors due to loading vectors on the outermost padding.
    ZeroFillImage(&decoded);
#endif
    const LoopFilter& lf = shared->frame_header.loop_filter;
    filter_weights.Init(lf, shared->frame_dim);
    for (auto& fp : filter_pipelines) {
      // De-initialize FilterPipelines.
      fp.num_filters = 0;
    }
  }
};

// Temp images required for decoding a single group. Reduces memory allocations
// for large images because we only initialize min(#threads, #groups) instances.
struct GroupDecCache {
  void InitOnce(size_t num_passes) {
    PROFILER_FUNC;

    for (size_t i = 0; i < num_passes; i++) {
      if (num_nzeroes[i].xsize() == 0) {
        // Allocate enough for a whole group - partial groups on the
        // right/bottom border just use a subset. The valid size is passed via
        // Rect.

        num_nzeroes[i] = Image3I(kGroupDimInBlocks, kGroupDimInBlocks);
      }
    }
  }

  // Scratch space used by DecGroupImpl().
  HWY_ALIGN_MAX float dec_group_block[3 * AcStrategy::kMaxCoeffArea];
  HWY_ALIGN_MAX float dec_group_local_block[AcStrategy::kMaxCoeffArea];
  // For TransformToPixels.
  HWY_ALIGN_MAX float scratch_space[2 * AcStrategy::kMaxCoeffArea];

  // AC decoding
  Image3I num_nzeroes[kMaxNumPasses];
};

static_assert(sizeof(GroupDecCache) % hwy::kMaxVectorSize == 0,
              "GroupDecCache must be aligned to vector size.");

}  // namespace jxl

#endif  // LIB_JXL_DEC_CACHE_H_
