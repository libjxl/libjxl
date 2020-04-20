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

#include <array>
#include <atomic>
#include <hwy/interface.h>
#include <vector>

#include "jxl/ac_context.h"
#include "jxl/ac_strategy.h"
#include "jxl/ans_params.h"
#include "jxl/ar_control_field.h"
#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/bits.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/override.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/brunsli.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/coeff_order.h"
#include "jxl/color_encoding.h"
#include "jxl/color_management.h"
#include "jxl/common.h"
#include "jxl/compressed_dc.h"
#include "jxl/dct_util.h"
#include "jxl/dot_dictionary.h"
#include "jxl/enc_ac_strategy.h"
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

  loop_filter->epf = ApplyOverride(
      cparams.adaptive_reconstruction,
      cparams.butteraugli_distance >= kMinButteraugliForAdaptiveReconstruction);

  return true;
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

// Also modifies the block, e.g. by removing patches.
Status JxlLossyFrameHeuristics(PassesEncoderState* enc_state,
                               const ImageBundle* linear, Image3F* opsin,
                               ThreadPool* pool, AuxOut* aux_out) {
  CompressParams& cparams = enc_state->cparams;
  const FrameDimensions& frame_dim = enc_state->shared.frame_dim;
  size_t target_size = TargetSize(cparams, frame_dim);
  size_t opsin_target_size = target_size;
  if (cparams.target_size > 0 || cparams.target_bitrate > 0.0) {
    cparams.target_size = opsin_target_size;
  } else if (cparams.butteraugli_distance < 0) {
    return JXL_FAILURE("Expected non-negative distance");
  }

  PROFILER_ZONE("JxlLossyFrameHeuristics uninstrumented");

  FindBestDequantMatrices(cparams, *opsin, &enc_state->shared.matrices);

  // Non-default cmap is on only for Hare or slower.
  auto find_best_cmap =
      ChooseFindBestColorCorrelationMap(hwy::SupportedTargets());
  if (cparams.speed_tier <= SpeedTier::kHare) {
    find_best_cmap(*opsin, enc_state->shared.matrices,
                   /*ac_strategy=*/nullptr, pool, &enc_state->shared.cmap);
  }

  ChooseFindBestAcStrategy(hwy::SupportedTargets())(*opsin, enc_state, pool,
                                                    aux_out);

  // Cmap is updated for different block sizes only for Wombat or slower.
  if (cparams.speed_tier <= SpeedTier::kWombat) {
    find_best_cmap(*opsin, enc_state->shared.matrices,
                   &enc_state->shared.ac_strategy, pool,
                   &enc_state->shared.cmap);
  }

  FindBestArControlField(*opsin, enc_state, pool);

  FindBestQuantizer(linear, *opsin, enc_state, pool, aux_out);
  return true;
}

Status MakeFrameHeader(const CompressParams& cparams,
                       const AnimationFrame* animation_frame_or_null,
                       const ImageBundle& ib, Multiframe* multiframe,
                       FrameHeader* JXL_RESTRICT frame_header,
                       LoopFilter* JXL_RESTRICT loop_filter) {
  frame_header->has_animation = animation_frame_or_null != nullptr;
  if (frame_header->has_animation) {
    frame_header->animation_frame = *animation_frame_or_null;
  }

  multiframe->InitPasses(&frame_header->passes);

  if (cparams.modular_group_mode) {
    frame_header->encoding = FrameEncoding::kModularGroup;
  }

  if (ib.IsJPEG()) {
    // we are transcoding a JPEG, so we don't get to choose
    frame_header->encoding = FrameEncoding::kVarDCT;
    frame_header->color_transform = ib.color_transform;
    frame_header->chroma_subsampling = ib.chroma_subsampling;
    JXL_ASSERT(frame_header->chroma_subsampling ==
                   YCbCrChromaSubsampling::k444 ||
               cparams.brunsli_group_mode);
  } else {
    frame_header->color_transform = cparams.color_transform;
    frame_header->chroma_subsampling = cparams.chroma_subsampling;
  }

  if (cparams.brunsli_group_mode) {
    frame_header->encoding = FrameEncoding::kJpegGroup;
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

  frame_header->has_alpha = ib.HasAlpha();
  frame_header->alpha_is_premultiplied = ib.AlphaIsPremultiplied();
  if (!frame_header->IsDisplayed()) {
    frame_header->has_alpha = false;
    frame_header->alpha_is_premultiplied = false;
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
                    Multiframe* multiframe, ThreadPool* pool,
                    const std::function<Status(size_t)>& pool_init,
                    AuxOut* aux_out, std::vector<AuxOut>* aux_outs)
      : enc_state_(enc_state),
        pool_(pool),
        pool_init_(pool_init),
        aux_out_(aux_out),
        aux_outs_(aux_outs) {
    JXL_CHECK(InitializePassesSharedState(
        frame_header, loop_filter, image_metadata, frame_dim, multiframe,
        &enc_state_->shared, /*encoder=*/true));
    enc_state_->cparams = cparams;
    enc_state_->passes.clear();
  }

  Status ComputeEncodingData(const ImageBundle* linear,
                             Image3F* JXL_RESTRICT opsin, ThreadPool* pool,
                             BitWriter* JXL_RESTRICT writer,
                             FrameHeader* frame_header) {
    PROFILER_ZONE("ComputeEncodingData uninstrumented");
    JXL_ASSERT((opsin->xsize() % kBlockDim) == 0 &&
               (opsin->ysize() % kBlockDim) == 0);
    PassesSharedState& shared = enc_state_->shared;

    if (!enc_state_->cparams.max_error_mode) {
      float x_qm_scale_steps[3] = {1.2f, 2.4f, 4.8f};
      shared.frame_header.x_qm_scale = 0;
      for (float x_qm_scale_step : x_qm_scale_steps) {
        if (enc_state_->cparams.butteraugli_distance > x_qm_scale_step) {
          shared.frame_header.x_qm_scale++;
        }
      }
    }

    if (enc_state_->cparams.speed_tier != SpeedTier::kFalcon) {
      // Call InitialQuantField only in Hare mode or slower. Otherwise, rely
      // on simple heuristics in FindBestAcStrategy.
      if (enc_state_->cparams.speed_tier > SpeedTier::kHare) {
        enc_state_->initial_quant_field = ImageF(shared.frame_dim.xsize_blocks,
                                                 shared.frame_dim.ysize_blocks);
      } else {
        // Call this here, as it relies on pre-gaborish values.
        // TODO(veluca): adjust to post-gaborish values.
        // TODO(veluca): call after image features.
        enc_state_->initial_quant_field =
            InitialQuantField(enc_state_->cparams.butteraugli_distance, *opsin,
                              shared.frame_dim, pool, 1.0f);
      }
    }

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
      const float rampup =
          (enc_state_->cparams.butteraugli_distance - kNoiseRampupStart) /
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

    shared.multiframe->DecorrelateOpsin(opsin);

    if (enc_state_->cparams.speed_tier <= SpeedTier::kSquirrel) {
      shared.image_features.splines = FindSplines(*opsin);
      JXL_RETURN_IF_ERROR(shared.image_features.splines.SubtractFrom(
          opsin, enc_state_->shared.cmap));
    }

    if (ApplyOverride(enc_state_->cparams.patches,
                      enc_state_->cparams.speed_tier <= SpeedTier::kSquirrel)) {
      FindBestPatchDictionary(*opsin, enc_state_, pool, aux_out_);
      enc_state_->shared.image_features.patches.SubtractFrom(opsin);
    }

    if (shared.image_features.loop_filter.gab) {
      *opsin = GaborishInverse(*opsin, 0.9908511000000001f, pool);
    }

    JXL_RETURN_IF_ERROR(
        JxlLossyFrameHeuristics(enc_state_, linear, opsin, pool_, aux_out_));

    InitializePassesEncoder(*opsin, pool_, enc_state_, aux_out_);

    enc_state_->passes.resize(shared.multiframe->GetNumPasses());
    for (PassesEncoderState::PassData& pass : enc_state_->passes) {
      pass.ac_tokens.resize(shared.frame_dim.num_groups);
    }

    {
      PROFILER_ZONE("PixelsToGroupCoefficients");
      const auto compute_group_cache_init = [&](const size_t num_threads) {
        group_caches_.resize(num_threads);
        return pool_init_(num_threads);
      };
      const auto compute_coef =
          ChooseComputeCoefficients(hwy::SupportedTargets());
      const auto compute_group_cache = [&](const int group_index,
                                           const int thread) {
        // Compute coefficients and coefficient split.
        AuxOut* my_aux_out = aux_out_ ? &(*aux_outs_)[thread] : nullptr;
        compute_coef(group_index, enc_state_, my_aux_out);
      };
      RunOnPool(pool_, 0, shared.frame_dim.num_groups, compute_group_cache_init,
                compute_group_cache, "PixelsToGroupCoefficients");
    }

    shared.num_histograms = 1;

    ComputeAllCoeffOrders(shared.frame_dim);

    const auto tokenize_group_init = [&](const size_t num_threads) {
      group_caches_.resize(num_threads);
      return true;
    };
    auto tokenize_coeffs = ChooseTokenizeCoefficients(hwy::SupportedTargets());
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
        tokenize_coeffs(&shared.coeff_orders[idx_pass * kCoeffOrderSize], rect,
                        ac_rows, shared.ac_strategy,
                        &group_caches_[thread].num_nzeroes,
                        &enc_state_->passes[idx_pass].ac_tokens[group_index]);
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

  Status ComputeJPEGTranscodingData(const Image3F& opsin_orig,
                                    const std::vector<int32_t>& quant_table,
                                    FrameHeader* frame_header) {
    constexpr size_t N = kBlockDim;
    PROFILER_ZONE("ComputeJPEGTranscodingData uninstrumented");
    PassesSharedState& shared = enc_state_->shared;

    FrameDimensions frame_dim;
    frame_dim.Set(opsin_orig.xsize(), opsin_orig.ysize());

    const size_t xsize = opsin_orig.xsize();
    const size_t ysize = opsin_orig.ysize();
    const size_t xsize_blocks = xsize / N;
    const size_t ysize_blocks = ysize / N;

    shared.frame_header.x_qm_scale = 0;

    // no-op chroma from luma
    shared.cmap = ColorCorrelationMap(xsize, ysize, false);
    shared.ac_strategy.FillDCT8();
    FillImage(uint8_t(0), &shared.epf_sharpness);

    enc_state_->coeffs.clear();
    enc_state_->coeffs.emplace_back(
        ACImage3(kGroupDim * kGroupDim, frame_dim.num_groups));

    // convert JPEG quantization table to a Quantizer object
    float dcquantization[3] = {8.0f / quant_table[0],
                               8.0f / quant_table[1 * 64],
                               8.0f / quant_table[2 * 64]};
    shared.matrices.SetCustomDC(dcquantization);
    std::vector<QuantEncoding> qe(DequantMatrices::kNum,
                                  QuantEncoding::Library(0));
    std::vector<int> qt(192);
    for (size_t c = 0; c < 3; c++) {
      for (size_t y = 0; y < 8; y++) {
        for (size_t x = 0; x < 8; x++) {
          // JPEG XL transposes the DCT, JPEG doesn't.
          qt[c * 64 + 8 * x + y] = quant_table[c * 64 + 8 * y + x];
        }
      }
    }
    qe[AcStrategy::Type::DCT] = QuantEncoding::RAW(qt);
    shared.matrices.SetCustom(qe);
    // Recompute MulDC() and InvMulDC().
    shared.quantizer.RecomputeFromGlobalScale();

    // Per-block dequant scaling should be 1.
    FillImage(static_cast<int>(shared.quantizer.InvGlobalScale()),
              &shared.raw_quant_field);

    Image3F dc = Image3F(xsize_blocks, ysize_blocks);
    enc_state_->dc = Image3S(xsize_blocks, ysize_blocks);
    intptr_t onerow = opsin_orig.Plane(0).PixelsPerRow();
    bool DCzero =
        (shared.frame_header.color_transform == ColorTransform::kYCbCr);
    // Compute chroma-from-luma for AC (doesn't seem to be useful for DC)
    if (frame_header->chroma_subsampling == YCbCrChromaSubsampling::k444) {
      for (size_t c : {0, 2}) {
        ImageB* map = (c == 0 ? &shared.cmap.ytox_map : &shared.cmap.ytob_map);
        const float kScale = kDefaultColorFactor;
        const int kOffset = kColorOffset;
        const float kBase = c == 0 ? shared.cmap.YtoXRatio(kOffset)
                                   : shared.cmap.YtoBRatio(kOffset);
        const float kZeroThresh =
            kScale * kZeroBiasDefault[c] *
            0.9999f;  // just epsilon less for better rounding

        auto process_row = [&](int task, int thread) {
          size_t ty = task;
          uint8_t* JXL_RESTRICT row_out = map->Row(ty);
          for (size_t tx = 0; tx < map->xsize(); ++tx) {
            const size_t y0 = ty * kColorTileDimInBlocks * kBlockDim;
            const size_t x0 = tx * kColorTileDimInBlocks * kBlockDim;
            const size_t y1 =
                std::min(opsin_orig.ysize(),
                         (ty + 1) * kColorTileDimInBlocks * kBlockDim);
            const size_t x1 =
                std::min(opsin_orig.xsize(),
                         (tx + 1) * kColorTileDimInBlocks * kBlockDim);
            int32_t d_num_zeros[257] = {0};
            for (size_t y = y0; y < y1; ++y) {
              const float* JXL_RESTRICT row_m = opsin_orig.PlaneRow(1, y);
              const float* JXL_RESTRICT row_s = opsin_orig.PlaneRow(c, y);
              for (size_t x = x0; x < x1; ++x) {
                int coeffpos = (x % kBlockDim) + kBlockDim * (y % kBlockDim);
                if (coeffpos == 0) continue;
                const float scaled_m = row_m[x] * quant_table[64 + coeffpos] /
                                       quant_table[64 * c + coeffpos];
                const float scaled_s =
                    kScale * row_s[x] + (kOffset - kBase * kScale) * scaled_m;
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
            int best = 0;
            int32_t best_sum = 0;
            FindIndexOfSumMaximum(d_num_zeros, 256, &best, &best_sum);
            int32_t offset_sum = 0;
            for (size_t i = 0; i < 256; ++i) {
              if (i <= kOffset) {
                offset_sum += d_num_zeros[i];
              }
            }
            if (best_sum > offset_sum + 1) {
              row_out[tx] = best;
            } else {
              row_out[tx] = kOffset;
            }
          }
        };

        RunOnPool(pool_, 0, map->ysize(), ThreadPool::SkipInit(), process_row,
                  "FindCorrelation");
      }
    }

    for (size_t c : {1, 0, 2}) {
      ImageB& map = (c == 0 ? shared.cmap.ytox_map : shared.cmap.ytob_map);
      for (size_t group_index = 0; group_index < frame_dim.num_groups;
           group_index++) {
        const size_t gx = group_index % frame_dim.xsize_groups;
        const size_t gy = group_index / frame_dim.xsize_groups;
        size_t offset = 0;
        float* JXL_RESTRICT ac = enc_state_->coeffs[0].PlaneRow(c, group_index);
        for (size_t by = gy * kGroupDimInBlocks;
             by < ysize_blocks && by < (gy + 1) * kGroupDimInBlocks; ++by) {
          const float* JXL_RESTRICT inputjpeg = opsin_orig.PlaneRow(c, by * 8);
          int16_t* JXL_RESTRICT idc = enc_state_->dc.PlaneRow(c, by);
          const float* JXL_RESTRICT inputjpegY = opsin_orig.PlaneRow(1, by * 8);
          float* JXL_RESTRICT fdc = dc.PlaneRow(c, by);
          const uint8_t* JXL_RESTRICT cm =
              map.ConstRow(by / kColorTileDimInBlocks);
          for (size_t bx = gx * kGroupDimInBlocks;
               bx < xsize_blocks && bx < (gx + 1) * kGroupDimInBlocks; ++bx) {
            if (DCzero) {
              idc[bx] = inputjpeg[bx * 8];
            } else {
              idc[bx] = (inputjpeg[bx * 8] * quant_table[c * 64] + 1024.f) /
                        quant_table[c * 64];
            }
            fdc[bx] = idc[bx] / dcquantization[c];
            if (c == 1) {
              for (int i = 0; i < 64; i++) {
                ac[offset + i] = inputjpeg[8 * bx + (i % 8) * onerow + (i / 8)];
              }
            } else {
              const float scale =
                  (c == 0
                       ? shared.cmap.YtoXRatio(cm[bx / kColorTileDimInBlocks])
                       : shared.cmap.YtoBRatio(cm[bx / kColorTileDimInBlocks]));

              for (int i = 0; i < 64; i++) {
                float Y = inputjpegY[8 * bx + (i % 8) * onerow + (i / 8)] *
                          quant_table[64 + i];
                float QChroma = inputjpeg[8 * bx + (i % 8) * onerow + (i / 8)];
                // Apply it like this to keep it reversible
                int QCR = QChroma -
                          static_cast<int>(Y * scale / quant_table[64 * c + i]);
                ac[offset + i] = QCR;
              }
            }
            offset += 64;
          }
        }
      }
    }

    // disable DC frame for now
    shared.frame_header.UpdateFlag(false, FrameHeader::kUseDcFrame);
    const size_t xsize_dc_groups = DivCeil(xsize_blocks, kDcGroupDimInBlocks);
    const size_t ysize_dc_groups = DivCeil(ysize_blocks, kDcGroupDimInBlocks);
    enc_state_->dc_tokens =
        std::vector<std::vector<Token>>(xsize_dc_groups * ysize_dc_groups);
    enc_state_->extra_dc_levels.resize(xsize_dc_groups * ysize_dc_groups, 0);
    const auto tokenize_dc = ChooseTokenizeDC(hwy::SupportedTargets());
    auto compute_dc_coeffs = [&](int group_index, int /* thread */) {
      tokenize_dc(group_index, dc, enc_state_, aux_out_);
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

    // Number of histograms and coefficient orders, per pass (always 1 for
    // now). Encoded as shared.num_histograms - 1.
    shared.num_histograms = 1;
    JXL_ASSERT(shared.num_histograms <= shared.frame_dim.num_groups);

    ComputeAllCoeffOrders(frame_dim);

    const auto tokenize_group_init = [&](const size_t num_threads) {
      group_caches_.resize(num_threads);
      return true;
    };
    auto tokenize_coeffs = ChooseTokenizeCoefficients(hwy::SupportedTargets());
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
        tokenize_coeffs(&shared.coeff_orders[idx_pass * kCoeffOrderSize], rect,
                        ac_rows, shared.ac_strategy,
                        &group_caches_[thread].num_nzeroes,
                        &enc_state_->passes[idx_pass].ac_tokens[group_index]);
      }
    };
    RunOnPool(pool_, 0, shared.frame_dim.num_groups, tokenize_group_init,
              tokenize_group, "TokenizeGroup");
    *frame_header = shared.frame_header;
    return true;
  }

  Status EncodeGlobalDCInfo(const FrameHeader& frame_header,
                            BitWriter* writer) const {
    JXL_RETURN_IF_ERROR(enc_state_->shared.matrices.EncodeDC(
        writer, kLayerDequantTables, aux_out_));

    // Encode quantizer DC and global scale.
    JXL_RETURN_IF_ERROR(
        enc_state_->shared.quantizer.Encode(writer, kLayerQuant, aux_out_));
    enc_state_->shared.cmap.EncodeDC(writer, kLayerDC, aux_out_);
    return true;
  }

  Status EncodeDCGroup(size_t group_index, BitWriter* group_code,
                       AuxOut* local_aux_out) {
    return jxl::EncodeDCGroup(*enc_state_, group_index, group_code,
                              local_aux_out);
  }

  Status EncodeGlobalACInfo(BitWriter* writer) {
    JXL_RETURN_IF_ERROR(enc_state_->shared.matrices.Encode(
        writer, kLayerDequantTables, aux_out_));
    size_t num_histo_bits =
        enc_state_->shared.frame_dim.num_groups == 1
            ? 0
            : CeilLog2Nonzero(enc_state_->shared.frame_dim.num_groups - 1);
    if (num_histo_bits != 0) {
      BitWriter::Allotment allotment(writer, num_histo_bits);
      writer->Write(num_histo_bits, enc_state_->shared.num_histograms - 1);
      ReclaimAndCharge(writer, &allotment, kLayerAC, aux_out_);
    }
    for (size_t i = 0; i < enc_state_->shared.multiframe->GetNumPasses(); i++) {
      // Encode coefficient orders.
      for (size_t histo = 0; histo < enc_state_->shared.num_histograms;
           histo++) {
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
      }

      // Encode histograms.
      HistogramParams hist_params(enc_state_->cparams.speed_tier, kNumContexts);
      for (size_t histo = 0; histo < enc_state_->shared.num_histograms;
           histo++) {
        // Same histogram encoding even in fast mode - the cost of
        // clustering is now lower (one per frame, not group), and it avoids
        // huge size penalties for small images.
        BuildAndEncodeHistograms(
            hist_params, kNumContexts, enc_state_->passes[i].ac_tokens,
            &enc_state_->passes[i].codes, &enc_state_->passes[i].context_map,
            writer, kLayerAC, aux_out_);
      }
    }

    return true;
  }

  Status EncodeACGroup(size_t pass, size_t group_index, BitWriter* group_code,
                       AuxOut* local_aux_out) {
    return EncodeGroupTokenizedCoefficients(group_index, pass, *enc_state_,
                                            group_code, local_aux_out);
  }

  PassesEncoderState* State() { return enc_state_; }

 private:
  void ComputeAllCoeffOrders(const FrameDimensions& frame_dim) {
    PROFILER_FUNC;
    for (size_t histo = 0; histo < enc_state_->shared.num_histograms; histo++) {
      for (size_t i = 0; i < enc_state_->shared.multiframe->GetNumPasses();
           i++) {
        uint32_t used_orders = 0;
        // No coefficient reordering in Falcon mode.
        if (enc_state_->cparams.speed_tier != SpeedTier::kFalcon) {
          used_orders = ComputeUsedOrders(
              enc_state_->cparams.speed_tier, enc_state_->shared.ac_strategy,
              Rect(enc_state_->shared.raw_quant_field));
        }
        ComputeCoeffOrder(
            enc_state_->coeffs[i], enc_state_->shared.ac_strategy, frame_dim,
            used_orders, &enc_state_->shared.coeff_orders[i * kCoeffOrderSize]);
      }
    }
  }

  PassesEncoderState* JXL_RESTRICT enc_state_;
  ThreadPool* pool_;
  const std::function<Status(size_t)>& pool_init_;
  AuxOut* aux_out_;
  std::vector<AuxOut>* aux_outs_;
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

  FrameDimensions frame_dim;
  frame_dim.Set(ib.xsize(), ib.ysize());

  if (ib.IsJPEG()) {
    cparams.gaborish = Override::kOff;
    cparams.adaptive_reconstruction = Override::kOff;

    if (ib.chroma_subsampling != YCbCrChromaSubsampling::k444 &&
        !cparams.brunsli_group_mode) {
      JXL_DEBUG_V(2, "Subsampled JPEG, using kJpegGroup\n");
      cparams.brunsli_group_mode = true;
    }
  }

  const size_t xsize = ib.xsize();
  const size_t ysize = ib.ysize();
  const size_t num_groups = frame_dim.num_groups;
  if (xsize == 0 || ysize == 0) return JXL_FAILURE("Empty image");

  FrameHeader frame_header;
  LoopFilter loop_filter;
  JXL_RETURN_IF_ERROR(MakeFrameHeader(cparams, animation_frame_or_null, ib,
                                      multiframe, &frame_header, &loop_filter));

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
      }
    }
    return true;
  };

  LossyFrameEncoder lossy_frame_encoder(
      cparams, frame_header, loop_filter, metadata, frame_dim, passes_enc_state,
      multiframe, pool, resize_aux_outs, aux_out, &aux_outs);
  BrunsliFrameEncoder jpeg_frame_encoder(frame_dim, pool);
  ModularFrameEncoder modular_frame_encoder;

  if (cparams.brunsli_group_mode) {
    if (!ib.IsJPEG()) {
      return JXL_FAILURE("JpegGroup mode requires JPEG quant table");
    }
    JXL_RETURN_IF_ERROR(jpeg_frame_encoder.ReadSourceImage(
        &ib, ib.jpeg_quant_table, frame_header.chroma_subsampling));
  } else if (ib.IsJPEG()) {
    JXL_RETURN_IF_ERROR(lossy_frame_encoder.ComputeJPEGTranscodingData(
        ib.color(), ib.jpeg_quant_table, &frame_header));
  } else {
    // Avoid a copy in PadImageToMultiple by allocating a large enough image
    // to begin with.
    opsin =
        Image3F(RoundUpToBlockDim(ib.xsize()), RoundUpToBlockDim(ib.ysize()));
    opsin.ShrinkTo(ib.xsize(), ib.ysize());

    if (frame_header.color_transform == ColorTransform::kXYB &&
        cparams.dc_level == 0 && cparams.save_as_reference == 0) {
      linear = (*ChooseToXYB(hwy::SupportedTargets()))(ib, pool, &opsin,
                                                       &linear_storage);

      // We only need linear sRGB in slow VarDCT modes.
      if (cparams.speed_tier > SpeedTier::kKitten ||
          frame_header.encoding == FrameEncoding::kModularGroup) {
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
          linear, &opsin, pool, writer, &frame_header));
    }
  }
  JXL_RETURN_IF_ERROR(modular_frame_encoder.ComputeEncodingData(
      cparams, frame_header, ib, &opsin, lossy_frame_encoder.State(),
      /* encode_color= */ frame_header.encoding ==
          FrameEncoding::kModularGroup));

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
  const bool has_ac_global = !frame_header.IsJpeg();
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

  if (frame_header.IsJpeg()) {
    // Heavy-lifting happens here.
    JXL_RETURN_IF_ERROR(jpeg_frame_encoder.DoEncode());
    JXL_RETURN_IF_ERROR(jpeg_frame_encoder.SerializeHeader(get_output(0)));
  }
  if (frame_header.IsLossy()) {
    // encoding == kVarDCT
    JXL_RETURN_IF_ERROR(
        lossy_frame_encoder.EncodeGlobalDCInfo(frame_header, get_output(0)));
    // Lossless has no DC info.
  }
  JXL_RETURN_IF_ERROR(
      modular_frame_encoder.EncodeGlobalInfo(get_output(0), aux_out));

  const auto process_dc_group = [&](const int group_index, const int thread) {
    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;
    BitWriter* output = get_output(group_index + 1);
    const size_t gx = group_index % frame_dim.xsize_dc_groups;
    const size_t gy = group_index / frame_dim.xsize_dc_groups;
    const Rect rect(gx * kDcGroupDim, gy * kDcGroupDim, kDcGroupDim,
                    kDcGroupDim);
    // minShift==3 because kDcGroupDim>>3 == kGroupDim
    // maxShift==1000 is infinity
    JXL_CHECK(modular_frame_encoder.EncodeGroup(rect, output, my_aux_out, 3,
                                                1000, kLayerModularDcGroup));
    if (frame_header.IsJpeg()) {
      JXL_CHECK(
          jpeg_frame_encoder.SerializeDcGroup(group_index, output, my_aux_out));
    } else if (frame_header.IsLossy()) {
      JXL_CHECK(
          lossy_frame_encoder.EncodeDCGroup(group_index, output, my_aux_out));
    }
  };
  RunOnPool(pool, 0, frame_dim.num_dc_groups, resize_aux_outs, process_dc_group,
            "EncodeDCGroup");

  const bool use_lossy_encoder =
      (frame_header.encoding != FrameEncoding::kModularGroup &&
       frame_header.encoding != FrameEncoding::kJpegGroup);

  if (use_lossy_encoder) {
    JXL_RETURN_IF_ERROR(
        lossy_frame_encoder.EncodeGlobalACInfo(get_output(global_ac_index)));
  }

  std::atomic<int> num_errors{0};
  const auto process_group = [&](const int group_index, const int thread) {
    AuxOut* my_aux_out = aux_out ? &aux_outs[thread] : nullptr;

    const size_t gx = group_index % frame_dim.xsize_groups;
    const size_t gy = group_index / frame_dim.xsize_groups;
    // For modular, don't limit to image dimensions here (is done in
    // EncodeGroup)
    const Rect mrect(gx * kGroupDim, gy * kGroupDim, kGroupDim, kGroupDim);
    int maxShift = 2;
    int minShift = 0;
    for (size_t i = 0; i < multiframe->GetNumPasses(); i++) {
      // Write all modular encoded data (color?, alpha, depth, extra channels)
      for (uint32_t j = 0; j < frame_header.passes.num_downsample; ++j) {
        if (i <= frame_header.passes.last_pass[j]) {
          if (frame_header.passes.downsample[j] == 8) minShift = 3;
          if (frame_header.passes.downsample[j] == 4) minShift = 2;
          if (frame_header.passes.downsample[j] == 2) minShift = 1;
          if (frame_header.passes.downsample[j] == 1) minShift = 0;
        }
      }
      //      printf("Encoding shifts %i..%i\n",minShift,maxShift);
      if (!modular_frame_encoder.EncodeGroup(
              mrect, ac_group_code(i, group_index), my_aux_out, minShift,
              maxShift, kLayerModularAcGroup)) {
        num_errors.fetch_add(1, std::memory_order_relaxed);
        return;
      }
      maxShift = minShift - 1;
      minShift = 0;
      if (frame_header.encoding == FrameEncoding::kJpegGroup) {
        if (multiframe->GetNumPasses() != 1) {
          num_errors.fetch_add(1, std::memory_order_relaxed);
          return;
        }
        if (!jpeg_frame_encoder.SerializeAcGroup(
                group_index, ac_group_code(i, group_index), my_aux_out)) {
          num_errors.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      } else if (frame_header.encoding == FrameEncoding::kVarDCT) {
        if (!lossy_frame_encoder.EncodeACGroup(
                i, group_index, ac_group_code(i, group_index), my_aux_out)) {
          num_errors.fetch_add(1, std::memory_order_relaxed);
          return;
        }
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
