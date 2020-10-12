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

#include "jxl/enc_frame.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "jxl/ac_context.h"
#include "jxl/ac_strategy.h"
#include "jxl/ans_params.h"
#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/override.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/coeff_order.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/common.h"
#include "jxl/compressed_dc.h"
#include "jxl/dct_util.h"
#include "jxl/enc_adaptive_quantization.h"
#include "jxl/enc_ans.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_group.h"
#include "jxl/enc_modular.h"
#include "jxl/enc_noise.h"
#include "jxl/enc_params.h"
#include "jxl/enc_xyb.h"
#include "jxl/entropy_coder.h"
#include "jxl/fields.h"
#include "jxl/frame_header.h"
#include "jxl/gaborish.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/image_ops.h"
#include "jxl/loop_filter.h"
#include "jxl/multiframe.h"
#include "jxl/patch_dictionary.h"
#include "jxl/quant_weights.h"
#include "jxl/quantizer.h"
#include "jxl/splines.h"
#include "jxl/toc.h"

namespace jxl {
namespace {

void ClusterGroups(PassesEncoderState* enc_state) {
  // This only considers pass 0 for now.
  std::vector<uint8_t> context_map;
  EntropyEncodingData codes;
  auto& ac = enc_state->passes[0].ac_tokens;
  size_t limit = std::ceil(std::sqrt(ac.size()));
  if (limit == 1) return;
  size_t num_contexts = enc_state->shared.block_ctx_map.NumACContexts();
  std::vector<float> costs(ac.size());
  HistogramParams params;
  params.uint_method = HistogramParams::HybridUintMethod::kNone;
  params.lz77_method = HistogramParams::LZ77Method::kNone;
  params.ans_histogram_strategy =
      HistogramParams::ANSHistogramStrategy::kApproximate;
  size_t max = 0;
  float total_cost = 0;
  auto token_cost = [&](std::vector<std::vector<Token>>& tokens, size_t num_ctx,
                        bool estimate = true) {
    // TODO(veluca): not estimating is very expensive.
    BitWriter writer;
    size_t c = BuildAndEncodeHistograms(
        params, num_ctx, tokens, &codes, &context_map,
        estimate ? nullptr : &writer, 0, /*aux_out=*/0);
    if (estimate) return c;
    for (size_t i = 0; i < tokens.size(); i++) {
      WriteTokens(tokens[i], codes, context_map, &writer, 0, nullptr);
    }
    return writer.BitsWritten();
  };
  for (size_t i = 0; i < ac.size(); i++) {
    std::vector<std::vector<Token>> tokens{ac[i]};
    costs[i] =
        token_cost(tokens, enc_state->shared.block_ctx_map.NumACContexts());
    if (costs[i] > costs[max]) {
      max = i;
    }
    total_cost += costs[i];
  }
  auto dist = [&](int i, int j) {
    std::vector<std::vector<Token>> tokens{ac[i], ac[j]};
    return token_cost(tokens, num_contexts) - costs[i] - costs[j];
  };
  std::vector<size_t> out{max};
  std::vector<size_t> old_map(ac.size());
  std::vector<float> dists(ac.size());
  size_t farthest = 0;
  for (size_t i = 0; i < ac.size(); i++) {
    if (i == max) continue;
    dists[i] = dist(max, i);
    if (dists[i] > dists[farthest]) {
      farthest = i;
    }
  }

  while (dists[farthest] > 0 && out.size() < limit) {
    out.push_back(farthest);
    dists[farthest] = 0;
    enc_state->histogram_idx[farthest] = out.size() - 1;
    for (size_t i = 0; i < ac.size(); i++) {
      float d = dist(out.back(), i);
      if (d < dists[i]) {
        dists[i] = d;
        old_map[i] = enc_state->histogram_idx[i];
        enc_state->histogram_idx[i] = out.size() - 1;
      }
      if (dists[i] > dists[farthest]) {
        farthest = i;
      }
    }
  }

  std::vector<size_t> remap(out.size());
  std::iota(remap.begin(), remap.end(), 0);
  for (size_t i = 0; i < enc_state->histogram_idx.size(); i++) {
    enc_state->histogram_idx[i] = remap[enc_state->histogram_idx[i]];
  }
  auto remap_cost = [&](std::vector<size_t> remap) {
    std::vector<size_t> re_remap(remap.size(), remap.size());
    size_t r = 0;
    for (size_t i = 0; i < remap.size(); i++) {
      if (re_remap[remap[i]] == remap.size()) {
        re_remap[remap[i]] = r++;
      }
      remap[i] = re_remap[remap[i]];
    }
    auto tokens = ac;
    size_t max_hist = 0;
    for (size_t i = 0; i < tokens.size(); i++) {
      for (size_t j = 0; j < tokens[i].size(); j++) {
        size_t hist = remap[enc_state->histogram_idx[i]];
        tokens[i][j].context += hist * num_contexts;
        max_hist = std::max(hist + 1, max_hist);
      }
    }
    return token_cost(tokens, max_hist * num_contexts, /*estimate=*/false);
  };

  for (size_t src = 0; src < out.size(); src++) {
    float cost = remap_cost(remap);
    size_t best = src;
    for (size_t j = src + 1; j < out.size(); j++) {
      if (remap[src] == remap[j]) continue;
      auto remap_c = remap;
      std::replace(remap_c.begin(), remap_c.end(), remap[src], remap[j]);
      float c = remap_cost(remap_c);
      if (c < cost) {
        best = j;
        cost = c;
      }
    }
    if (src != best) {
      std::replace(remap.begin(), remap.end(), remap[src], remap[best]);
    }
  }
  std::vector<size_t> re_remap(remap.size(), remap.size());
  size_t r = 0;
  for (size_t i = 0; i < remap.size(); i++) {
    if (re_remap[remap[i]] == remap.size()) {
      re_remap[remap[i]] = r++;
    }
    remap[i] = re_remap[remap[i]];
  }

  enc_state->shared.num_histograms =
      *std::max_element(remap.begin(), remap.end()) + 1;
  for (size_t i = 0; i < enc_state->histogram_idx.size(); i++) {
    enc_state->histogram_idx[i] = remap[enc_state->histogram_idx[i]];
  }
  for (size_t i = 0; i < ac.size(); i++) {
    for (size_t j = 0; j < ac[i].size(); j++) {
      ac[i][j].context += enc_state->histogram_idx[i] * num_contexts;
    }
  }
}

uint64_t FrameFlagsFromParams(const CompressParams& cparams) {
  uint64_t flags = 0;

  const float dist = cparams.butteraugli_distance;

  // We don't add noise at low butteraugli distances because the original
  // noise is stored within the compressed image and adding noise makes things
  // worse.
  if (ApplyOverride(cparams.noise, dist >= kMinButteraugliForNoise)) {
    flags |= FrameHeader::kNoise;
  }

  if (cparams.progressive_dc) {
    flags |= FrameHeader::kUseDcFrame;
  }

  return flags;
}

Status LoopFilterFromParams(const CompressParams& cparams,
                            LoopFilter* JXL_RESTRICT loop_filter) {
  // Gaborish defaults to enabled in Hare or slower.
  loop_filter->gab =
      ApplyOverride(cparams.gaborish, cparams.speed_tier <= SpeedTier::kHare);

  // By default we only use epf_iters=2 when EPF is enabled.
  loop_filter->epf_iters =
      ApplyOverride(cparams.adaptive_reconstruction,
                    cparams.butteraugli_distance >=
                        kMinButteraugliForAdaptiveReconstruction)
          ? 2
          : 0;

  return true;
}

Status MakeFrameHeader(const CompressParams& cparams,
                       const AnimationFrame* animation_frame_or_null,
                       const ImageBundle& ib, Multiframe* multiframe,
                       FrameHeader* JXL_RESTRICT frame_header,
                       LoopFilter* JXL_RESTRICT loop_filter) {
  frame_header->nonserialized_xyb_image =
      (cparams.color_transform == ColorTransform::kXYB);
  frame_header->has_animation = animation_frame_or_null != nullptr;
  if (frame_header->has_animation) {
    frame_header->animation_frame = *animation_frame_or_null;
  }

  multiframe->InitPasses(&frame_header->passes);

  if (cparams.modular_group_mode) {
    frame_header->encoding = FrameEncoding::kModular;
    frame_header->group_size_shift = cparams.modular_group_size_shift;
  }

  if (ib.IsJPEG()) {
    // we are transcoding a JPEG, so we don't get to choose
    frame_header->encoding = FrameEncoding::kVarDCT;
    frame_header->color_transform = ib.color_transform;
    frame_header->chroma_subsampling = ib.chroma_subsampling;
  } else {
    frame_header->color_transform = cparams.color_transform;
    if (cparams.chroma_subsampling.MaxHShift() != 0 ||
        cparams.chroma_subsampling.MaxVShift() != 0) {
      // TODO(veluca): properly pad the input image to support this.
      return JXL_FAILURE(
          "Chroma subsampling is not supported when not recompressing JPEGs");
    }
    frame_header->chroma_subsampling = cparams.chroma_subsampling;
  }

  if (frame_header->IsLossy()) {
    frame_header->flags = FrameFlagsFromParams(cparams);

    JXL_RETURN_IF_ERROR(LoopFilterFromParams(cparams, loop_filter));
  }

  if (cparams.dc_level) {
    frame_header->dc_level = cparams.dc_level;
  }
  frame_header->save_as_reference = cparams.save_as_reference;
  if (cparams.save_as_reference == 0 && cparams.dc_level == 0) {
    frame_header->is_displayed = true;
  }

  return true;
}

}  // namespace

class LossyFrameEncoder {
 public:
  LossyFrameEncoder(const CompressParams& cparams,
                    const FrameHeader& frame_header,
                    const LoopFilter& loop_filter,
                    const ImageMetadata& image_metadata,
                    const FrameDimensions& frame_dim,
                    PassesEncoderState* JXL_RESTRICT enc_state,
                    Multiframe* multiframe, ThreadPool* pool, AuxOut* aux_out)
      : enc_state_(enc_state), pool_(pool), aux_out_(aux_out) {
    JXL_CHECK(InitializePassesSharedState(
        frame_header, loop_filter, image_metadata, frame_dim, multiframe,
        &enc_state_->shared, /*encoder=*/true));
    enc_state_->cparams = cparams;
    enc_state_->passes.clear();
  }

  Status ComputeEncodingData(const ImageBundle* linear,
                             Image3F* JXL_RESTRICT opsin, ThreadPool* pool,
                             ModularFrameEncoder* modular_frame_encoder,
                             BitWriter* JXL_RESTRICT writer,
                             FrameHeader* frame_header) {
    PROFILER_ZONE("ComputeEncodingData uninstrumented");
    JXL_ASSERT((opsin->xsize() % kBlockDim) == 0 &&
               (opsin->ysize() % kBlockDim) == 0);
    PassesSharedState& shared = enc_state_->shared;

    if (!enc_state_->cparams.max_error_mode) {
      float x_qm_scale_steps[3] = {0.65f, 1.25f, 9.0f};
      shared.frame_header.x_qm_scale = 0;
      for (float x_qm_scale_step : x_qm_scale_steps) {
        if (enc_state_->cparams.butteraugli_distance > x_qm_scale_step) {
          shared.frame_header.x_qm_scale++;
        }
      }
    }

    JXL_RETURN_IF_ERROR(enc_state_->heuristics->LossyFrameHeuristics(
        enc_state_, linear, opsin, pool_, aux_out_));

    InitializePassesEncoder(*opsin, pool_, enc_state_, modular_frame_encoder,
                            aux_out_);

    enc_state_->passes.resize(shared.multiframe->GetNumPasses());
    for (PassesEncoderState::PassData& pass : enc_state_->passes) {
      pass.ac_tokens.resize(shared.frame_dim.num_groups);
    }

    ComputeAllCoeffOrders(shared.frame_dim);
    shared.num_histograms = 1;

    const auto tokenize_group_init = [&](const size_t num_threads) {
      group_caches_.resize(num_threads);
      return true;
    };
    const auto tokenize_group = [&](const int group_index, const int thread) {
      // Tokenize coefficients.
      const Rect rect = shared.BlockGroupRect(group_index);
      for (size_t idx_pass = 0; idx_pass < enc_state_->passes.size();
           idx_pass++) {
        const ac_qcoeff_t* JXL_RESTRICT ac_rows[3] = {
            enc_state_->coeffs[idx_pass].ConstPlaneRow(0, group_index),
            enc_state_->coeffs[idx_pass].ConstPlaneRow(1, group_index),
            enc_state_->coeffs[idx_pass].ConstPlaneRow(2, group_index),
        };
        // Ensure group cache is initialized.
        group_caches_[thread].InitOnce();
        TokenizeCoefficients(
            &shared.coeff_orders[idx_pass * kCoeffOrderSize], rect, ac_rows,
            shared.ac_strategy, frame_header->chroma_subsampling,
            &group_caches_[thread].num_nzeroes,
            &enc_state_->passes[idx_pass].ac_tokens[group_index],
            enc_state_->shared.quant_dc, enc_state_->shared.raw_quant_field,
            enc_state_->shared.block_ctx_map);
      }
    };
    RunOnPool(pool_, 0, shared.frame_dim.num_groups, tokenize_group_init,
              tokenize_group, "TokenizeGroup");

    if (!shared.frame_header.animation_frame.is_last &&
        !shared.frame_header.animation_frame.have_crop) {
      RoundtripImage(*opsin, enc_state_, pool_,
                     /*save_decompressed=*/true);
    }

    if (aux_out_ && aux_out_->testing_aux.dc) {
      *aux_out_->testing_aux.dc = CopyImage(shared.dc_storage);
    }
    if (aux_out_ && aux_out_->testing_aux.decoded) {
      *aux_out_->testing_aux.decoded =
          RoundtripImage(*opsin, enc_state_, pool_,
                         /*save_decompressed=*/false);
    }

    *frame_header = shared.frame_header;
    return true;
  }

  Status ComputeJPEGTranscodingData(const brunsli::JPEGData& jpeg_data,
                                    ModularFrameEncoder* modular_frame_encoder,
                                    FrameHeader* frame_header) {
    PROFILER_ZONE("ComputeJPEGTranscodingData uninstrumented");
    PassesSharedState& shared = enc_state_->shared;

    FrameDimensions frame_dim;
    frame_dim.Set(jpeg_data.width, jpeg_data.height,
                  frame_header->group_size_shift,
                  frame_header->chroma_subsampling.MaxHShift(),
                  frame_header->chroma_subsampling.MaxVShift());

    const size_t xsize = frame_dim.xsize_padded;
    const size_t ysize = frame_dim.ysize_padded;
    const size_t xsize_blocks = frame_dim.xsize_blocks;
    const size_t ysize_blocks = frame_dim.ysize_blocks;

    shared.frame_header.x_qm_scale = 0;

    // no-op chroma from luma
    shared.cmap = ColorCorrelationMap(xsize, ysize, false);
    shared.ac_strategy.FillDCT8();
    FillImage(uint8_t(0), &shared.epf_sharpness);

    enc_state_->coeffs.clear();
    enc_state_->coeffs.emplace_back(
        ACImage3(kGroupDim * kGroupDim, frame_dim.num_groups));

    // convert JPEG quantization table to a Quantizer object
    float dcquantization[3];
    std::vector<QuantEncoding> qe(DequantMatrices::kNum,
                                  QuantEncoding::Library(0));

    int jpeg_c_map[3] = {1, 0, 2};
    if (jpeg_data.components.size() != 3) {
      jpeg_c_map[0] = jpeg_c_map[2] = 0;
    }

    std::vector<int> qt(192);
    for (size_t c = 0; c < 3; c++) {
      size_t jpeg_c = jpeg_c_map[c];
      const int* quant =
          jpeg_data.quant[jpeg_data.components[jpeg_c].quant_idx].values.data();

      dcquantization[c] = 8.0f / quant[0];
      for (size_t y = 0; y < 8; y++) {
        for (size_t x = 0; x < 8; x++) {
          // JPEG XL transposes the DCT, JPEG doesn't.
          qt[c * 64 + 8 * x + y] = quant[8 * y + x];
        }
      }
    }
    shared.matrices.SetCustomDC(dcquantization);
    float dcquantization_r[3] = {1.0f / dcquantization[0],
                                 1.0f / dcquantization[1],
                                 1.0f / dcquantization[2]};

    qe[AcStrategy::Type::DCT] = QuantEncoding::RAW(qt);
    shared.matrices.SetCustom(qe);
    // Ensure that InvGlobalScale() is 1.
    shared.quantizer = Quantizer(&shared.matrices, 1, kGlobalScaleDenom);
    // Recompute MulDC() and InvMulDC().
    shared.quantizer.RecomputeFromGlobalScale();

    // Per-block dequant scaling should be 1.
    FillImage(static_cast<int>(shared.quantizer.InvGlobalScale()),
              &shared.raw_quant_field);

    std::vector<int32_t> scaled_qtable(192);
    for (size_t c = 0; c < 3; c++) {
      for (size_t i = 0; i < 64; i++) {
        scaled_qtable[64 * c + i] =
            (1 << kCFLFixedPointPrecision) * qt[64 + i] / qt[64 * c + i];
      }
    }

    auto jpeg_row = [&](size_t c, size_t y) {
      return jpeg_data.components[jpeg_c_map[c]].coeffs.data() +
             jpeg_data.components[jpeg_c_map[c]].width_in_blocks *
                 kDCTBlockSize * y;
    };

    Image3F dc = Image3F(xsize_blocks, ysize_blocks);
    bool DCzero =
        (shared.frame_header.color_transform == ColorTransform::kYCbCr);
    // Compute chroma-from-luma for AC (doesn't seem to be useful for DC)
    if (frame_header->chroma_subsampling.Is444() &&
        enc_state_->cparams.force_cfl_jpeg_recompression &&
        jpeg_data.components.size() == 3) {
      for (size_t c : {0, 2}) {
        ImageSB* map = (c == 0 ? &shared.cmap.ytox_map : &shared.cmap.ytob_map);
        const float kScale = kDefaultColorFactor;
        const int kOffset = 127;
        const float kBase =
            c == 0 ? shared.cmap.YtoXRatio(0) : shared.cmap.YtoBRatio(0);
        const float kZeroThresh =
            kScale * kZeroBiasDefault[c] *
            0.9999f;  // just epsilon less for better rounding

        auto process_row = [&](int task, int thread) {
          size_t ty = task;
          int8_t* JXL_RESTRICT row_out = map->Row(ty);
          for (size_t tx = 0; tx < map->xsize(); ++tx) {
            const size_t y0 = ty * kColorTileDimInBlocks;
            const size_t x0 = tx * kColorTileDimInBlocks;
            const size_t y1 = std::min(frame_dim.ysize_blocks,
                                       (ty + 1) * kColorTileDimInBlocks);
            const size_t x1 = std::min(frame_dim.xsize_blocks,
                                       (tx + 1) * kColorTileDimInBlocks);
            int32_t d_num_zeros[257] = {0};
            // TODO(veluca): this needs SIMD + fixed point adaptation, and/or
            // conversion to the new CfL algorithm.
            for (size_t y = y0; y < y1; ++y) {
              const int16_t* JXL_RESTRICT row_m = jpeg_row(1, y);
              const int16_t* JXL_RESTRICT row_s = jpeg_row(c, y);
              for (size_t x = x0; x < x1; ++x) {
                for (int coeffpos = 1; coeffpos < kDCTBlockSize; coeffpos++) {
                  const float scaled_m =
                      row_m[x * kDCTBlockSize + coeffpos] *
                      scaled_qtable[64 * c + coeffpos] *
                      (1.0f / (1 << kCFLFixedPointPrecision));
                  const float scaled_s =
                      kScale * row_s[x * kDCTBlockSize + coeffpos] +
                      (kOffset - kBase * kScale) * scaled_m;
                  if (std::abs(scaled_m) > 1e-8f) {
                    float from, to;
                    if (scaled_m > 0) {
                      from = (scaled_s - kZeroThresh) / scaled_m;
                      to = (scaled_s + kZeroThresh) / scaled_m;
                    } else {
                      from = (scaled_s + kZeroThresh) / scaled_m;
                      to = (scaled_s - kZeroThresh) / scaled_m;
                    }
                    if (from < 0.0f) {
                      from = 0.0f;
                    }
                    if (to > 255.0f) {
                      to = 255.0f;
                    }
                    // Instead of clamping the both values
                    // we just check that range is sane.
                    if (from <= to) {
                      d_num_zeros[static_cast<int>(std::ceil(from))]++;
                      d_num_zeros[static_cast<int>(std::floor(to + 1))]--;
                    }
                  }
                }
              }
            }
            int best = 0;
            int32_t best_sum = 0;
            FindIndexOfSumMaximum(d_num_zeros, 256, &best, &best_sum);
            int32_t offset_sum = 0;
            for (size_t i = 0; i < 256; ++i) {
              if (i <= kOffset) {
                offset_sum += d_num_zeros[i];
              }
            }
            row_out[tx] = 0;
            if (best_sum > offset_sum + 1) {
              row_out[tx] = best - kOffset;
            }
          }
        };

        RunOnPool(pool_, 0, map->ysize(), ThreadPool::SkipInit(), process_row,
                  "FindCorrelation");
      }
    }
    if (!frame_header->chroma_subsampling.Is444()) {
      ZeroFillImage(&dc);
      ZeroFillImage(&enc_state_->coeffs[0]);
    }
    // JPEG DC is from -1024 to 1023.
    std::vector<size_t> dc_counts[3] = {};
    dc_counts[0].resize(2048);
    dc_counts[1].resize(2048);
    dc_counts[2].resize(2048);
    size_t total_dc[3] = {};
    for (size_t c : {1, 0, 2}) {
      if (jpeg_data.components.size() == 1 && c != 1) {
        ZeroFillImage(const_cast<ImageF*>(&enc_state_->coeffs[0].Plane(c)));
        ZeroFillImage(const_cast<ImageF*>(&dc.Plane(c)));
        // Ensure no division by 0.
        dc_counts[c][1024] = 1;
        total_dc[c] = 1;
        continue;
      }
      size_t hshift = frame_header->chroma_subsampling.HShift(c);
      size_t vshift = frame_header->chroma_subsampling.VShift(c);
      ImageSB& map = (c == 0 ? shared.cmap.ytox_map : shared.cmap.ytob_map);
      for (size_t group_index = 0; group_index < frame_dim.num_groups;
           group_index++) {
        const size_t gx = group_index % frame_dim.xsize_groups;
        const size_t gy = group_index / frame_dim.xsize_groups;
        size_t offset = 0;
        float* JXL_RESTRICT ac = enc_state_->coeffs[0].PlaneRow(c, group_index);
        for (size_t by = gy * kGroupDimInBlocks;
             by < ysize_blocks && by < (gy + 1) * kGroupDimInBlocks; ++by) {
          if ((by >> vshift) << vshift != by) continue;
          const int16_t* JXL_RESTRICT inputjpeg = jpeg_row(c, by >> vshift);
          const int16_t* JXL_RESTRICT inputjpegY = jpeg_row(1, by);
          float* JXL_RESTRICT fdc = dc.PlaneRow(c, by >> vshift);
          const int8_t* JXL_RESTRICT cm =
              map.ConstRow(by / kColorTileDimInBlocks);
          for (size_t bx = gx * kGroupDimInBlocks;
               bx < xsize_blocks && bx < (gx + 1) * kGroupDimInBlocks; ++bx) {
            if ((bx >> hshift) << hshift != bx) continue;
            size_t base = (bx >> hshift) * kDCTBlockSize;
            int idc;
            if (DCzero) {
              idc = inputjpeg[base];
            } else {
              idc = (inputjpeg[base] * qt[c * 64] + 1024.f) / qt[c * 64];
            }
            dc_counts[c][idc + 1024]++;
            total_dc[c]++;
            fdc[bx >> hshift] = idc * dcquantization_r[c];
            if (c == 1 || !enc_state_->cparams.force_cfl_jpeg_recompression ||
                !frame_header->chroma_subsampling.Is444()) {
              for (size_t y = 0; y < 8; y++) {
                for (size_t x = 0; x < 8; x++) {
                  ac[offset + y * 8 + x] = inputjpeg[base + x * 8 + y];
                }
              }
            } else {
              const int32_t scale =
                  shared.cmap.RatioJPEG(cm[bx / kColorTileDimInBlocks]);

              for (size_t y = 0; y < 8; y++) {
                for (size_t x = 0; x < 8; x++) {
                  int Y = inputjpegY[kDCTBlockSize * bx + x * 8 + y];
                  int QChroma = inputjpeg[kDCTBlockSize * bx + x * 8 + y];
                  // Fixed-point multiply of CfL scale with quant table ratio
                  // first, and Y value second.
                  int coeff_scale = (scale * scaled_qtable[64 * c + y * 8 + x] +
                                     (1 << (kCFLFixedPointPrecision - 1))) >>
                                    kCFLFixedPointPrecision;
                  int cfl_factor = (Y * coeff_scale +
                                    (1 << (kCFLFixedPointPrecision - 1))) >>
                                   kCFLFixedPointPrecision;
                  int QCR = QChroma - cfl_factor;
                  ac[offset + y * 8 + x] = QCR;
                }
              }
            }
            offset += 64;
          }
        }
      }
    }

    auto& dct = enc_state_->shared.block_ctx_map.dc_thresholds;
    auto& num_dc_ctxs = enc_state_->shared.block_ctx_map.num_dc_ctxs;
    enc_state_->shared.block_ctx_map.num_dc_ctxs = 1;
    for (size_t i = 0; i < 3; i++) {
      dct[i].clear();
      int num_thresholds = (CeilLog2Nonzero(total_dc[i]) - 10) / 2;
      // up to 3 buckets per channel:
      // dark/medium/bright, yellow/unsat/blue, green/unsat/red
      num_thresholds = std::min(std::max(num_thresholds, 0), 2);
      size_t cumsum = 0;
      size_t cut = total_dc[i] / (num_thresholds + 1);
      for (int j = 0; j < 2048; j++) {
        cumsum += dc_counts[i][j];
        if (cumsum > cut) {
          dct[i].push_back(j - 1025);
          cut = total_dc[i] * (dct[i].size() + 1) / (num_thresholds + 1);
        }
      }
      num_dc_ctxs *= dct[i].size() + 1;
    }

    auto& ctx_map = enc_state_->shared.block_ctx_map.ctx_map;
    ctx_map.clear();
    ctx_map.resize(3 * kNumOrders * num_dc_ctxs, 0);

    int lbuckets = (dct[1].size() + 1);
    for (size_t i = 0; i < num_dc_ctxs; i++) {
      // up to 9 contexts for luma
      ctx_map[i] = i / lbuckets;
      // up to 3 contexts for chroma
      ctx_map[kNumOrders * num_dc_ctxs + i] =
          num_dc_ctxs / lbuckets + (i % lbuckets);
      ctx_map[2 * kNumOrders * num_dc_ctxs + i] =
          num_dc_ctxs / lbuckets + (i % lbuckets);
    }
    enc_state_->shared.block_ctx_map.num_ctxs =
        *std::max_element(ctx_map.begin(), ctx_map.end()) + 1;

    enc_state_->histogram_idx.resize(shared.frame_dim.num_groups);

    // disable DC frame for now
    shared.frame_header.UpdateFlag(false, FrameHeader::kUseDcFrame);
    auto compute_dc_coeffs = [&](int group_index, int /* thread */) {
      modular_frame_encoder->AddVarDCTDC(dc, group_index, /*nl_dc=*/false,
                                         enc_state_);
      modular_frame_encoder->AddACMetadata(group_index, /*jpeg_transcode=*/true,
                                           enc_state_);
    };
    RunOnPool(pool_, 0, shared.frame_dim.num_dc_groups, ThreadPool::SkipInit(),
              compute_dc_coeffs, "Compute DC coeffs");

    // Must happen before WriteFrameHeader!
    shared.frame_header.UpdateFlag(true, FrameHeader::kSkipAdaptiveDCSmoothing);

    enc_state_->passes.resize(shared.multiframe->GetNumPasses());
    for (PassesEncoderState::PassData& pass : enc_state_->passes) {
      pass.ac_tokens.resize(shared.frame_dim.num_groups);
    }

    JXL_CHECK(enc_state_->passes.size() ==
              1);  // skipping coeff splitting so need to have only one pass

    ComputeAllCoeffOrders(frame_dim);
    shared.num_histograms = 1;

    const auto tokenize_group_init = [&](const size_t num_threads) {
      group_caches_.resize(num_threads);
      return true;
    };
    const auto tokenize_group = [&](const int group_index, const int thread) {
      // Tokenize coefficients.
      const Rect rect = shared.BlockGroupRect(group_index);
      for (size_t idx_pass = 0; idx_pass < enc_state_->passes.size();
           idx_pass++) {
        const ac_qcoeff_t* JXL_RESTRICT ac_rows[3] = {
            enc_state_->coeffs[idx_pass].ConstPlaneRow(0, group_index),
            enc_state_->coeffs[idx_pass].ConstPlaneRow(1, group_index),
            enc_state_->coeffs[idx_pass].ConstPlaneRow(2, group_index),
        };
        // Ensure group cache is initialized.
        group_caches_[thread].InitOnce();
        TokenizeCoefficients(
            &shared.coeff_orders[idx_pass * kCoeffOrderSize], rect, ac_rows,
            shared.ac_strategy, frame_header->chroma_subsampling,
            &group_caches_[thread].num_nzeroes,
            &enc_state_->passes[idx_pass].ac_tokens[group_index],
            enc_state_->shared.quant_dc, enc_state_->shared.raw_quant_field,
            enc_state_->shared.block_ctx_map);
      }
    };
    RunOnPool(pool_, 0, shared.frame_dim.num_groups, tokenize_group_init,
              tokenize_group, "TokenizeGroup");
    *frame_header = shared.frame_header;
    return true;
  }

  Status EncodeGlobalDCInfo(const FrameHeader& frame_header,
                            BitWriter* writer) const {
    // Encode quantizer DC and global scale.
    JXL_RETURN_IF_ERROR(
        enc_state_->shared.quantizer.Encode(writer, kLayerQuant, aux_out_));
    EncodeBlockCtxMap(enc_state_->shared.block_ctx_map, writer, aux_out_);
    enc_state_->shared.cmap.EncodeDC(writer, kLayerDC, aux_out_);
    return true;
  }

  Status EncodeGlobalACInfo(BitWriter* writer,
                            ModularFrameEncoder* modular_frame_encoder) {
    JXL_RETURN_IF_ERROR(enc_state_->shared.matrices.Encode(
        writer, kLayerDequantTables, aux_out_, modular_frame_encoder));
    if (enc_state_->cparams.speed_tier <= SpeedTier::kTortoise) {
      ClusterGroups(enc_state_);
    }
    size_t num_histo_bits =
        CeilLog2Nonzero(enc_state_->shared.frame_dim.num_groups);
    if (num_histo_bits != 0) {
      BitWriter::Allotment allotment(writer, num_histo_bits);
      writer->Write(num_histo_bits, enc_state_->shared.num_histograms - 1);
      ReclaimAndCharge(writer, &allotment, kLayerAC, aux_out_);
    }

    for (size_t i = 0; i < enc_state_->shared.multiframe->GetNumPasses(); i++) {
      // Encode coefficient orders.
      uint32_t used_orders = ComputeUsedOrders(
          enc_state_->cparams.speed_tier, enc_state_->shared.ac_strategy,
          Rect(enc_state_->shared.raw_quant_field));
      size_t order_bits = 0;
      JXL_RETURN_IF_ERROR(
          U32Coder::CanEncode(kOrderEnc, used_orders, &order_bits));
      BitWriter::Allotment allotment(writer, order_bits);
      JXL_CHECK(U32Coder::Write(kOrderEnc, used_orders, writer));
      ReclaimAndCharge(writer, &allotment, kLayerOrder, aux_out_);
      EncodeCoeffOrders(used_orders,
                        &enc_state_->shared.coeff_orders[i * kCoeffOrderSize],
                        writer, kLayerOrder, aux_out_);

      // Encode histograms.
      HistogramParams hist_params(
          enc_state_->cparams.speed_tier,
          enc_state_->shared.block_ctx_map.NumACContexts());
      if (enc_state_->cparams.speed_tier > SpeedTier::kTortoise) {
        hist_params.lz77_method = HistogramParams::LZ77Method::kNone;
      }
      BuildAndEncodeHistograms(
          hist_params,
          enc_state_->shared.num_histograms *
              enc_state_->shared.block_ctx_map.NumACContexts(),
          enc_state_->passes[i].ac_tokens, &enc_state_->passes[i].codes,
          &enc_state_->passes[i].context_map, writer, kLayerAC, aux_out_);
    }

    return true;
  }

  Status EncodeACGroup(size_t pass, size_t group_index, BitWriter* group_code,
                       AuxOut* local_aux_out) {
    return EncodeGroupTokenizedCoefficients(
        group_index, pass, enc_state_->histogram_idx[group_index], *enc_state_,
        group_code, local_aux_out);
  }

  PassesEncoderState* State() { return enc_state_; }

 private:
  void ComputeAllCoeffOrders(const FrameDimensions& frame_dim) {
    PROFILER_FUNC;
    for (size_t i = 0; i < enc_state_->shared.multiframe->GetNumPasses(); i++) {
      uint32_t used_orders = 0;
      // No coefficient reordering in Falcon mode.
      if (enc_state_->cparams.speed_tier != SpeedTier::kFalcon) {
        used_orders = ComputeUsedOrders(
            enc_state_->cparams.speed_tier, enc_state_->shared.ac_strategy,
            Rect(enc_state_->shared.raw_quant_field));
      }
      ComputeCoeffOrder(enc_state_->cparams.speed_tier, enc_state_->coeffs[i],
                        enc_state_->shared.ac_strategy, frame_dim, used_orders,
                        &enc_state_->shared.coeff_orders[i * kCoeffOrderSize]);
    }
  }

  template <typename V, typename R>
  static inline void FindIndexOfSumMaximum(const V* array, const size_t len,
                                           R* idx, V* sum) {
    JXL_ASSERT(len > 0);
    V maxval = 0;
    V val = 0;
    R maxidx = 0;
    for (size_t i = 0; i < len; ++i) {
      val += array[i];
      if (val > maxval) {
        maxval = val;
        maxidx = i;
      }
    }
    *idx = maxidx;
    *sum = maxval;
  }

  PassesEncoderState* JXL_RESTRICT enc_state_;
  ThreadPool* pool_;
  AuxOut* aux_out_;
  std::vector<EncCache> group_caches_;
};

Status EncodeFrame(const CompressParams& cparams_orig,
                   const AnimationFrame* animation_frame_or_null,
                   const ImageBundle& ib, PassesEncoderState* passes_enc_state,
                   ThreadPool* pool, BitWriter* writer, AuxOut* aux_out,
                   Multiframe* multiframe) {
  ib.VerifyMetadata();
  CompressParams cparams = cparams_orig;
  if (cparams.dc_level + cparams.progressive_dc > 3) {
    return JXL_FAILURE("Too many levels of progressive DC");
  }

  if (cparams.butteraugli_distance != 0 &&
      cparams.butteraugli_distance < kMinButteraugliDistance) {
    return JXL_FAILURE("Butteraugli distance is too low");
  }

  if (ib.IsJPEG()) {
    cparams.gaborish = Override::kOff;
    cparams.adaptive_reconstruction = Override::kOff;
  }

  const size_t xsize = ib.xsize();
  const size_t ysize = ib.ysize();
  if (xsize == 0 || ysize == 0) return JXL_FAILURE("Empty image");

  FrameHeader frame_header;
  LoopFilter loop_filter;
  JXL_RETURN_IF_ERROR(MakeFrameHeader(cparams, animation_frame_or_null, ib,
                                      multiframe, &frame_header, &loop_filter));
  // Check that if the codestream header says xyb_encoded, the color_transform
  // matches the requirement. This is checked from the cparams here, even though
  // optimally we'd be able to check this against what has actually been written
  // in the main codestream header, but since ib is a const object and the data
  // written to the main codestream header is (in modified form) in ib, the
  // encoder cannot indicate this fact in the ib's metadata.
  if (cparams_orig.color_transform == ColorTransform::kXYB) {
    if (frame_header.color_transform != ColorTransform::kXYB) {
      return JXL_FAILURE(
          "The color transform of frames must be xyb if the codestream is xyb "
          "encoded");
    }
  } else {
    if (frame_header.color_transform == ColorTransform::kXYB) {
      return JXL_FAILURE(
          "The color transform of frames cannot be xyb if the codestream is "
          "not xyb encoded");
    }
  }

  FrameDimensions frame_dim;
  frame_dim.Set(ib.xsize(), ib.ysize(), frame_header.group_size_shift,
                frame_header.chroma_subsampling.MaxHShift(),
                frame_header.chroma_subsampling.MaxVShift());

  const size_t num_groups = frame_dim.num_groups;

  multiframe->StartFrame(frame_header);

  Image3F opsin;
  const ColorEncoding& c = ColorEncoding::LinearSRGB(ib.IsGray());
  ImageMetadata metadata;
  metadata.color_encoding = c;
  ImageBundle linear_storage(&metadata);
  const ImageBundle* JXL_RESTRICT linear = &ib;

  std::vector<AuxOut> aux_outs;
  // LossyFrameEncoder stores a reference to a std::function<Status(size_t)>
  // so we need to keep the std::function<Status(size_t)> being referenced
  // alive while lossy_frame_encoder is used. We could make resize_aux_outs a
  // lambda type by making LossyFrameEncoder a template instead, but this is
  // simpler.
  const std::function<Status(size_t)> resize_aux_outs =
      [&aux_outs, aux_out](size_t num_threads) -> Status {
    if (aux_out != nullptr) {
      size_t old_size = aux_outs.size();
      for (size_t i = num_threads; i < old_size; i++) {
        aux_out->Assimilate(aux_outs[i]);
      }
      aux_outs.resize(num_threads);
      // Each thread needs these INPUTS. Don't copy the entire AuxOut
      // because it may contain stats which would be Assimilated multiple
      // times below.
      for (size_t i = old_size; i < aux_outs.size(); i++) {
        aux_outs[i].testing_aux = aux_out->testing_aux;
        aux_outs[i].dump_image = aux_out->dump_image;
        aux_outs[i].debug_prefix = aux_out->debug_prefix;
      }
    }
    return true;
  };

  LossyFrameEncoder lossy_frame_encoder(cparams, frame_header, loop_filter,
                                        metadata, frame_dim, passes_enc_state,
                                        multiframe, pool, aux_out);
  ModularFrameEncoder modular_frame_encoder(frame_dim, frame_header, cparams);

  // Used by SetCustom.
  passes_enc_state->shared.matrices.SetModularFrameEncoder(
      &modular_frame_encoder);

  if (ib.IsJPEG()) {
    JXL_RETURN_IF_ERROR(lossy_frame_encoder.ComputeJPEGTranscodingData(
        *ib.jpeg_data, &modular_frame_encoder, &frame_header));
  } else {
    // Avoid a copy in PadImageToMultiple by allocating a large enough image
    // to begin with.
    opsin =
        Image3F(RoundUpToBlockDim(ib.xsize()), RoundUpToBlockDim(ib.ysize()));
    opsin.ShrinkTo(ib.xsize(), ib.ysize());

    if (frame_header.color_transform == ColorTransform::kXYB &&
        cparams.dc_level == 0 && cparams.save_as_reference == 0) {
      linear = ToXYB(ib, pool, &opsin, &linear_storage);

      // We only need linear sRGB in slow VarDCT modes.
      if (cparams.speed_tier > SpeedTier::kKitten ||
          frame_header.encoding == FrameEncoding::kModular) {
        linear = nullptr;
        linear_storage = ImageBundle();  // free the memory
      }
    } else {  // RGB or YCbCr: don't do anything (forward YCbCr is not
              // implemented, this is only used when the input is already in
              // YCbCr)
              // If encoding a special DC or reference frame, don't do anything:
              // input is already in XYB.
      CopyImageTo(ib.color(), &opsin);
    }
    if (aux_out != nullptr) {
      JXL_RETURN_IF_ERROR(
          aux_out->InspectImage3F("enc_frame:OpsinDynamicsImage", opsin));
    }
    if (frame_header.encoding == FrameEncoding::kVarDCT) {
      PadImageToBlockMultipleInPlace(&opsin);
      JXL_RETURN_IF_ERROR(lossy_frame_encoder.ComputeEncodingData(
          linear, &opsin, pool, &modular_frame_encoder, writer, &frame_header));
    }
  }
  // needs to happen *AFTER* VarDCT-ComputeEncodingData.
  JXL_RETURN_IF_ERROR(modular_frame_encoder.ComputeEncodingData(
      frame_header, ib, &opsin, lossy_frame_encoder.State(), pool, aux_out,
      /* do_color=*/frame_header.encoding == FrameEncoding::kModular));

  writer->AppendByteAligned(lossy_frame_encoder.State()->special_frames);
  frame_header.UpdateFlag(
      lossy_frame_encoder.State()->shared.image_features.patches.HasAny(),
      FrameHeader::kPatches);
  frame_header.UpdateFlag(
      lossy_frame_encoder.State()->shared.image_features.splines.HasAny(),
      FrameHeader::kSplines);
  JXL_RETURN_IF_ERROR(WriteFrameHeader(frame_header, writer, aux_out));
  if (frame_header.IsLossy()) {
    JXL_RETURN_IF_ERROR(WriteLoopFilter(loop_filter, writer, aux_out));
  }

  const size_t num_passes = multiframe->GetNumPasses();

  // DC global info + DC groups + AC global info + AC groups *
  // num_passes.
  const bool has_ac_global = true;
  std::vector<BitWriter> group_codes(NumTocEntries(frame_dim.num_groups,
                                                   frame_dim.num_dc_groups,
                                                   num_passes, has_ac_global));
  const size_t global_ac_index = frame_dim.num_dc_groups + 1;
  const bool is_small_image = frame_dim.num_groups == 1 && num_passes == 1;
  const auto get_output = [&](const size_t index) {
    return &group_codes[is_small_image ? 0 : index];
  };
  auto ac_group_code = [&](size_t pass, size_t group) {
    return get_output(AcGroupIndex(pass, group, frame_dim.num_groups,
                                   frame_dim.num_dc_groups, has_ac_global));
  };

  if (frame_header.flags & FrameHeader::kPatches) {
    lossy_frame_encoder.State()->shared.image_features.patches.Encode(
        get_output(0), kLayerDictionary, aux_out);
  }

  if (frame_header.flags & FrameHeader::kSplines) {
    lossy_frame_encoder.State()->shared.image_features.splines.Encode(
        get_output(0), kLayerSplines, aux_out);
  }

  if (frame_header.flags & FrameHeader::kNoise) {
    EncodeNoise(lossy_frame_encoder.State()->shared.image_features.noise_params,
                get_output(0), kLayerNoise, aux_out);
  }

  JXL_RETURN_IF_ERROR(lossy_frame_encoder.State()->shared.matrices.EncodeDC(
      get_output(0), kLayerDequantTables, aux_out));
  if (frame_header.IsLossy()) {
    // encoding == kVarDCT
    JXL_RETURN_IF_ERROR(
        lossy_frame_encoder.EncodeGlobalDCInfo(frame_header, get_output(0)));
  }
  JXL_RETURN_IF_ERROR(
      modular_frame_encoder.EncodeGlobalInfo(get_output(0), aux_out));
  JXL_RETURN_IF_ERROR(modular_frame_encoder.EncodeStream(
      get_output(0), aux_out, kLayerModularGlobal, ModularStreamId::Global()));

  const auto process_dc_group = [&](const int group_index, const int thread) {
    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;
    BitWriter* output = get_output(group_index + 1);
    if (frame_header.IsLossy() &&
        !(frame_header.flags & FrameHeader::kUseDcFrame)) {
      BitWriter::Allotment allotment(output, 2);
      output->Write(2, modular_frame_encoder.extra_dc_precision[group_index]);
      ReclaimAndCharge(output, &allotment, kLayerDC, my_aux_out);
      JXL_CHECK(modular_frame_encoder.EncodeStream(
          output, my_aux_out, kLayerDC,
          ModularStreamId::VarDCTDC(group_index)));
    }
    JXL_CHECK(modular_frame_encoder.EncodeStream(
        output, my_aux_out, kLayerModularDcGroup,
        ModularStreamId::ModularDC(group_index)));
    if (frame_header.IsLossy()) {
      const Rect& rect =
          lossy_frame_encoder.State()->shared.DCGroupRect(group_index);
      size_t nb_bits = CeilLog2Nonzero(rect.xsize() * rect.ysize());
      BitWriter::Allotment allotment(output, nb_bits);
      output->Write(nb_bits,
                    modular_frame_encoder.ac_metadata_size[group_index] - 1);
      ReclaimAndCharge(output, &allotment, kLayerControlFields, my_aux_out);
      JXL_CHECK(modular_frame_encoder.EncodeStream(
          output, my_aux_out, kLayerControlFields,
          ModularStreamId::ACMetadata(group_index)));
    }
  };
  RunOnPool(pool, 0, frame_dim.num_dc_groups, resize_aux_outs, process_dc_group,
            "EncodeDCGroup");

  if (frame_header.encoding == FrameEncoding::kVarDCT) {
    JXL_RETURN_IF_ERROR(lossy_frame_encoder.EncodeGlobalACInfo(
        get_output(global_ac_index), &modular_frame_encoder));
  }

  std::atomic<int> num_errors{0};
  const auto process_group = [&](const int group_index, const int thread) {
    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;

    for (size_t i = 0; i < multiframe->GetNumPasses(); i++) {
      if (frame_header.encoding == FrameEncoding::kVarDCT) {
        if (!lossy_frame_encoder.EncodeACGroup(
                i, group_index, ac_group_code(i, group_index), my_aux_out)) {
          num_errors.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      }
      // Write all modular encoded data (color?, alpha, depth, extra channels)
      if (!modular_frame_encoder.EncodeStream(
              ac_group_code(i, group_index), my_aux_out, kLayerModularAcGroup,
              ModularStreamId::ModularAC(group_index, i))) {
        num_errors.fetch_add(1, std::memory_order_relaxed);
        return;
      }
    }
  };
  RunOnPool(pool, 0, num_groups, resize_aux_outs, process_group,
            "EncodeGroupCoefficients");

  // Resizing aux_outs to 0 also Assimilates the array.
  static_cast<void>(resize_aux_outs(0));
  JXL_RETURN_IF_ERROR(num_errors.load(std::memory_order_relaxed) == 0);

  for (BitWriter& bw : group_codes) {
    bw.ZeroPadToByte();  // end of group.
  }

  JXL_RETURN_IF_ERROR(WriteGroupOffsets(group_codes, nullptr, writer, aux_out));
  writer->AppendByteAligned(group_codes);
  writer->ZeroPadToByte();  // end of frame.

  return true;
}

}  // namespace jxl
