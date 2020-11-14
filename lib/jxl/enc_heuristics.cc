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

#include "lib/jxl/enc_heuristics.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <numeric>
#include <string>

#include "lib/jxl/ar_control_field.h"
#include "lib/jxl/dot_dictionary.h"
#include "lib/jxl/enc_ac_strategy.h"
#include "lib/jxl/enc_adaptive_quantization.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_modular.h"
#include "lib/jxl/enc_noise.h"

namespace jxl {
namespace {
void FindBestBlockEntropyModel(PassesEncoderState& enc_state) {
  if (enc_state.cparams.speed_tier == SpeedTier::kFalcon) {
    return;
  }
  const ImageI& rqf = enc_state.shared.raw_quant_field;
  // No need to change context modeling for small images.
  size_t tot = rqf.xsize() * rqf.ysize();
  size_t size_for_ctx_model =
      (1 << 10) * enc_state.cparams.butteraugli_distance;
  if (tot < size_for_ctx_model) return;

  struct OccCounters {
    // count the occurrences of each qf value and each strategy type.
    OccCounters(const ImageI& rqf, const AcStrategyImage& ac_strategy) {
      for (size_t y = 0; y < rqf.ysize(); y++) {
        const int32_t* qf_row = rqf.Row(y);
        AcStrategyRow acs_row = ac_strategy.ConstRow(y);
        for (size_t x = 0; x < rqf.xsize(); x++) {
          int ord = kStrategyOrder[acs_row[x].RawStrategy()];
          int qf = qf_row[x] - 1;
          qf_counts[qf]++;
          qf_ord_counts[ord][qf]++;
          ord_counts[ord]++;
        }
      }
    }

    size_t qf_counts[256] = {};
    size_t qf_ord_counts[kNumOrders][256] = {};
    size_t ord_counts[kNumOrders] = {};
  };
  // The OccCounters struct is too big to allocate on the stack.
  std::unique_ptr<OccCounters> counters(
      new OccCounters(rqf, enc_state.shared.ac_strategy));

  // Splitting the context model according to the quantization field seems to
  // mostly benefit only large images.
  size_t size_for_qf_split = (1 << 13) * enc_state.cparams.butteraugli_distance;
  size_t num_qf_segments = tot < size_for_qf_split ? 1 : 2;
  std::vector<uint32_t>& qft = enc_state.shared.block_ctx_map.qf_thresholds;
  qft.clear();
  // Divide the quant field in up to num_qf_segments segments.
  size_t cumsum = 0;
  size_t next = 1;
  size_t last_cut = 256;
  size_t cut = tot * next / num_qf_segments;
  for (uint32_t j = 0; j < 256; j++) {
    cumsum += counters->qf_counts[j];
    if (cumsum > cut) {
      if (j != 0) {
        qft.push_back(j);
      }
      last_cut = j;
      while (cumsum > cut) {
        next++;
        cut = tot * next / num_qf_segments;
      }
    } else if (next > qft.size() + 1) {
      if (j - 1 == last_cut && j != 0) {
        qft.push_back(j);
      }
    }
  }

  // Count the occurrences of each segment.
  std::vector<size_t> counts(kNumOrders * (qft.size() + 1));
  size_t qft_pos = 0;
  for (size_t j = 0; j < 256; j++) {
    if (qft_pos < qft.size() && j == qft[qft_pos]) {
      qft_pos++;
    }
    for (size_t i = 0; i < kNumOrders; i++) {
      counts[qft_pos + i * (qft.size() + 1)] += counters->qf_ord_counts[i][j];
    }
  }

  // Repeatedly merge the lowest-count pair.
  std::vector<uint8_t> remap((qft.size() + 1) * kNumOrders);
  std::iota(remap.begin(), remap.end(), 0);
  std::vector<uint8_t> clusters(remap);
  // This is O(n^2 log n), but n <= 14.
  while (clusters.size() > 5) {
    std::sort(clusters.begin(), clusters.end(),
              [&](int a, int b) { return counts[a] > counts[b]; });
    counts[clusters[clusters.size() - 2]] += counts[clusters.back()];
    counts[clusters.back()] = 0;
    remap[clusters.back()] = clusters[clusters.size() - 2];
    clusters.pop_back();
  }
  for (size_t i = 0; i < remap.size(); i++) {
    while (remap[remap[i]] != remap[i]) {
      remap[i] = remap[remap[i]];
    }
  }
  // Relabel starting from 0.
  std::vector<uint8_t> remap_remap(remap.size(), remap.size());
  size_t num = 0;
  for (size_t i = 0; i < remap.size(); i++) {
    if (remap_remap[remap[i]] == remap.size()) {
      remap_remap[remap[i]] = num++;
    }
    remap[i] = remap_remap[remap[i]];
  }
  // Write the block context map.
  auto& ctx_map = enc_state.shared.block_ctx_map.ctx_map;
  ctx_map = remap;
  ctx_map.resize(remap.size() * 3);
  for (size_t i = remap.size(); i < remap.size() * 3; i++) {
    ctx_map[i] = remap[i % remap.size()] + num;
  }
  enc_state.shared.block_ctx_map.num_ctxs =
      *std::max_element(ctx_map.begin(), ctx_map.end()) + 1;
}
// Returns the target size based on whether bitrate or direct targetsize is
// given.
size_t TargetSize(const CompressParams& cparams,
                  const FrameDimensions& frame_dim) {
  if (cparams.target_size > 0) {
    return cparams.target_size;
  }
  if (cparams.target_bitrate > 0.0) {
    return 0.5 + cparams.target_bitrate * frame_dim.xsize * frame_dim.ysize /
                     kBitsPerByte;
  }
  return 0;
}
}  // namespace

Status DefaultEncoderHeuristics::LossyFrameHeuristics(
    PassesEncoderState* enc_state, ModularFrameEncoder* modular_frame_encoder,
    const ImageBundle* linear, Image3F* opsin, ThreadPool* pool,
    AuxOut* aux_out) {
  PROFILER_ZONE("JxlLossyFrameHeuristics uninstrumented");
  CompressParams& cparams = enc_state->cparams;
  PassesSharedState& shared = enc_state->shared;
  const FrameDimensions& frame_dim = enc_state->shared.frame_dim;
  size_t target_size = TargetSize(cparams, frame_dim);
  size_t opsin_target_size = target_size;
  if (cparams.target_size > 0 || cparams.target_bitrate > 0.0) {
    cparams.target_size = opsin_target_size;
  } else if (cparams.butteraugli_distance < 0) {
    return JXL_FAILURE("Expected non-negative distance");
  }

  // Compute an initial estimate of the quantization field.
  if (cparams.speed_tier != SpeedTier::kFalcon) {
    // Call InitialQuantField only in Hare mode or slower. Otherwise, rely
    // on simple heuristics in FindBestAcStrategy.
    if (cparams.speed_tier > SpeedTier::kHare) {
      enc_state->initial_quant_field =
          ImageF(shared.frame_dim.xsize_blocks, shared.frame_dim.ysize_blocks);
    } else {
      // Call this here, as it relies on pre-gaborish values.
      // TODO(veluca): adjust to post-gaborish values.
      // TODO(veluca): call after image features.
      enc_state->initial_quant_field = InitialQuantField(
          cparams.butteraugli_distance, *opsin, shared.frame_dim, pool, 1.0f);
    }
  }

  // Compute parameters for noise synthesis.
  if (shared.frame_header.flags & FrameHeader::kNoise) {
    PROFILER_ZONE("enc GetNoiseParam");
    // Don't start at zero amplitude since adding noise is expensive -- it
    // significantly slows down decoding, and this is unlikely to
    // completely go away even with advanced optimizations. After the
    // kNoiseModelingRampUpDistanceRange we have reached the full level,
    // i.e. noise is no longer represented by the compressed image, so we
    // can add full noise by the noise modeling itself.
    static const float kNoiseModelingRampUpDistanceRange = 0.6;
    static const float kNoiseLevelAtStartOfRampUp = 0.25;
    static const float kNoiseRampupStart = 1.0;
    // TODO(user) test and properly select quality_coef with smooth
    // filter
    float quality_coef = 1.0f;
    const float rampup = (cparams.butteraugli_distance - kNoiseRampupStart) /
                         kNoiseModelingRampUpDistanceRange;
    if (rampup < 1.0f) {
      quality_coef = kNoiseLevelAtStartOfRampUp +
                     (1.0f - kNoiseLevelAtStartOfRampUp) * rampup;
    }
    if (rampup < 0.0f) {
      quality_coef = kNoiseRampupStart;
    }
    if (!GetNoiseParameter(*opsin, &shared.image_features.noise_params,
                           quality_coef)) {
      shared.frame_header.flags &= ~FrameHeader::kNoise;
    }
  }

  // TODO(veluca): do something about animations.

  // Find and subtract splines.
  if (cparams.speed_tier <= SpeedTier::kSquirrel) {
    shared.image_features.splines = FindSplines(*opsin);
    JXL_RETURN_IF_ERROR(
        shared.image_features.splines.SubtractFrom(opsin, shared.cmap));
  }

  // Find and subtract patches/dots.
  if (ApplyOverride(cparams.patches,
                    cparams.speed_tier <= SpeedTier::kSquirrel)) {
    FindBestPatchDictionary(*opsin, enc_state, pool, aux_out);
    shared.image_features.patches.SubtractFrom(opsin);
  }

  // Apply inverse-gaborish.
  if (shared.frame_header.loop_filter.gab) {
    *opsin = GaborishInverse(*opsin, 0.9908511000000001f, pool);
  }

  FindBestDequantMatrices(cparams, *opsin, modular_frame_encoder,
                          &enc_state->shared.matrices);

  // For speeds up to Wombat, we only compute the color correlation map
  // once we know the transform type and the quantization map.
  if (cparams.speed_tier <= SpeedTier::kSquirrel) {
    FindBestColorCorrelationMap(
        *opsin, enc_state->shared.matrices,
        /*ac_strategy=*/nullptr, /*raw_quant_field=*/nullptr,
        /*quantizer=*/nullptr, pool, &enc_state->shared.cmap,
        /*fast=*/false);
  }

  // Choose block sizes.
  FindBestAcStrategy(*opsin, enc_state, pool, aux_out);

  // Choose amount of post-processing smoothing.
  FindBestArControlField(*opsin, enc_state, pool);

  // Refine quantization levels.
  FindBestQuantizer(linear, *opsin, enc_state, pool, aux_out);

  // Compute a non-default CfL map if we are at Hare speed, or slower.
  if (cparams.speed_tier <= SpeedTier::kHare) {
    FindBestColorCorrelationMap(
        *opsin, enc_state->shared.matrices, &enc_state->shared.ac_strategy,
        &enc_state->shared.raw_quant_field, &enc_state->shared.quantizer, pool,
        &enc_state->shared.cmap,
        /*fast=*/cparams.speed_tier >= SpeedTier::kWombat);
  }

  // Choose a context model that depends on the amount of quantization for AC.
  if (cparams.speed_tier != SpeedTier::kFalcon) {
    FindBestBlockEntropyModel(*enc_state);
  }
  return true;
}

}  // namespace jxl
