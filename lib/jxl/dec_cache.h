// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_CACHE_H_
#define LIB_JXL_DEC_CACHE_H_

#include <stdint.h>

#include <atomic>
#include <hwy/base.h>  // HWY_ALIGN_MAX

#include "jxl/decode.h"
#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/coeff_order.h"
#include "lib/jxl/common.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/dec_group_border.h"
#include "lib/jxl/dec_noise.h"
#include "lib/jxl/image.h"
#include "lib/jxl/passes_state.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/render_pipeline/render_pipeline.h"
#include "lib/jxl/render_pipeline/stage_upsampling.h"
#include "lib/jxl/sanitizers.h"

namespace jxl {

constexpr size_t kSigmaBorder = 1;
constexpr size_t kSigmaPadding = 2;

struct PixelCallback {
  PixelCallback() = default;
  PixelCallback(JxlImageOutInitCallback init, JxlImageOutRunCallback run,
                JxlImageOutDestroyCallback destroy, void* init_opaque)
      : init(init), run(run), destroy(destroy), init_opaque(init_opaque) {
#if JXL_ENABLE_ASSERT
    const bool has_init = init != nullptr;
    const bool has_run = run != nullptr;
    const bool has_destroy = destroy != nullptr;
    JXL_ASSERT(has_init == has_run && has_run == has_destroy);
#endif
  }

  bool IsPresent() const { return run != nullptr; }

  void* Init(size_t num_threads, size_t num_pixels) const {
    return init(init_opaque, num_threads, num_pixels);
  }

  JxlImageOutInitCallback init = nullptr;
  JxlImageOutRunCallback run = nullptr;
  JxlImageOutDestroyCallback destroy = nullptr;
  void* init_opaque = nullptr;
};

// Per-frame decoder state. All the images here should be accessed through a
// group rect (either with block units or pixel units).
struct PassesDecoderState {
  PassesSharedState shared_storage;
  // Allows avoiding copies for encoder loop.
  const PassesSharedState* JXL_RESTRICT shared = &shared_storage;

  // 8x upsampling stage for DC.
  std::unique_ptr<RenderPipelineStage> upsampler8x;

  // For ANS decoding.
  std::vector<ANSCode> code;
  std::vector<std::vector<uint8_t>> context_map;

  // Multiplier to be applied to the quant matrices of the x channel.
  float x_dm_multiplier;
  float b_dm_multiplier;

  // Sigma values for EPF.
  ImageF sigma;

  // RGB8 output buffer. If not nullptr, image data will be written to this
  // buffer instead of being written to the output ImageBundle. The image data
  // is assumed to have the stride given by `rgb_stride`, hence row `i` starts
  // at position `i * rgb_stride`.
  uint8_t* rgb_output;
  size_t rgb_stride = 0;

  size_t output_channels;

  // Whether to use int16 float-XYB-to-uint8-srgb conversion.
  bool fast_xyb_srgb8_conversion;

  // If true, rgb_output or callback output is RGBA using 4 instead of 3 bytes
  // per pixel.
  bool rgb_output_is_rgba;
  // If true, the RGBA output will be unpremultiplied before writing to the
  // output callback (the output buffer case is handled in ConvertToExternal).
  bool unpremul_alpha;

  bool swap_endianness;
  Orientation undo_orientation;

  // Callback for line-by-line output.
  PixelCallback pixel_callback;

  // Buffer of upsampling * kApplyImageFeaturesTileDim ones.
  std::vector<float> opaque_alpha;
  // One row per thread
  std::vector<std::vector<float>> pixel_callback_rows;

  // Used for seeding noise.
  size_t visible_frame_index = 0;
  size_t nonvisible_frame_index = 0;

  // Keep track of the transform types used.
  std::atomic<uint32_t> used_acs{0};

  // Storage for coefficients if in "accumulate" mode.
  std::unique_ptr<ACImage> coefficients = make_unique<ACImageT<int32_t>>(0, 0);

  // Rendering pipeline.
  std::unique_ptr<RenderPipeline> render_pipeline;

  // Storage for the current frame if it can be referenced by future frames.
  ImageBundle frame_storage_for_referencing;

  struct PipelineOptions {
    bool use_slow_render_pipeline;
    bool coalescing;
    bool render_spotcolors;
  };

  Status PreparePipeline(ImageBundle* decoded, PipelineOptions options);

  // Information for colour conversions.
  OutputEncodingInfo output_encoding_info;

  // Initializes decoder-specific structures using information from *shared.
  Status Init() {
    x_dm_multiplier =
        std::pow(1 / (1.25f), shared->frame_header.x_qm_scale - 2.0f);
    b_dm_multiplier =
        std::pow(1 / (1.25f), shared->frame_header.b_qm_scale - 2.0f);

    rgb_output = nullptr;
    rgb_output_is_rgba = false;
    output_channels = 3;
    unpremul_alpha = false;
    swap_endianness = false;
    undo_orientation = Orientation::kIdentity;
    fast_xyb_srgb8_conversion = false;
    pixel_callback = PixelCallback();
    used_acs = 0;

    upsampler8x = GetUpsamplingStage(shared->metadata->transform_data, 0, 3);
    if (shared->frame_header.loop_filter.epf_iters > 0) {
      sigma = ImageF(shared->frame_dim.xsize_blocks + 2 * kSigmaPadding,
                     shared->frame_dim.ysize_blocks + 2 * kSigmaPadding);
    }
    return true;
  }

  // Initialize the decoder state after all of DC is decoded.
  Status InitForAC(ThreadPool* pool) {
    shared_storage.coeff_order_size = 0;
    for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
      if (((1 << o) & used_acs) == 0) continue;
      uint8_t ord = kStrategyOrder[o];
      shared_storage.coeff_order_size =
          std::max(kCoeffOrderOffset[3 * (ord + 1)] * kDCTBlockSize,
                   shared_storage.coeff_order_size);
    }
    size_t sz = shared_storage.frame_header.passes.num_passes *
                shared_storage.coeff_order_size;
    if (sz > shared_storage.coeff_orders.size()) {
      shared_storage.coeff_orders.resize(sz);
    }
    return true;
  }

  // Fills the `state->filter_weights.sigma` image with the precomputed sigma
  // values in the area inside `block_rect`. Accesses the AC strategy, quant
  // field and epf_sharpness fields in the corresponding positions.
  void ComputeSigma(const Rect& block_rect, PassesDecoderState* state);
};

// Temp images required for decoding a single group. Reduces memory allocations
// for large images because we only initialize min(#threads, #groups) instances.
struct GroupDecCache {
  void InitOnce(size_t num_passes, size_t used_acs) {
    PROFILER_FUNC;

    for (size_t i = 0; i < num_passes; i++) {
      if (num_nzeroes[i].xsize() == 0) {
        // Allocate enough for a whole group - partial groups on the
        // right/bottom border just use a subset. The valid size is passed via
        // Rect.

        num_nzeroes[i] = Image3I(kGroupDimInBlocks, kGroupDimInBlocks);
      }
    }
    size_t max_block_area = 0;

    for (uint8_t o = 0; o < AcStrategy::kNumValidStrategies; ++o) {
      AcStrategy acs = AcStrategy::FromRawStrategy(o);
      if ((used_acs & (1 << o)) == 0) continue;
      size_t area =
          acs.covered_blocks_x() * acs.covered_blocks_y() * kDCTBlockSize;
      max_block_area = std::max(area, max_block_area);
    }

    if (max_block_area > max_block_area_) {
      max_block_area_ = max_block_area;
      // We need 3x float blocks for dequantized coefficients and 1x for scratch
      // space for transforms.
      float_memory_ = hwy::AllocateAligned<float>(max_block_area_ * 4);
      // We need 3x int32 or int16 blocks for quantized coefficients.
      int32_memory_ = hwy::AllocateAligned<int32_t>(max_block_area_ * 3);
      int16_memory_ = hwy::AllocateAligned<int16_t>(max_block_area_ * 3);
    }

    dec_group_block = float_memory_.get();
    scratch_space = dec_group_block + max_block_area_ * 3;
    dec_group_qblock = int32_memory_.get();
    dec_group_qblock16 = int16_memory_.get();
  }

  void InitDCBufferOnce() {
    if (dc_buffer.xsize() == 0) {
      dc_buffer = ImageF(kGroupDimInBlocks + kRenderPipelineXOffset * 2,
                         kGroupDimInBlocks + 4);
    }
  }

  // Scratch space used by DecGroupImpl().
  float* dec_group_block;
  int32_t* dec_group_qblock;
  int16_t* dec_group_qblock16;

  // For TransformToPixels.
  float* scratch_space;
  // Note that scratch_space is never used at the same time as dec_group_qblock.
  // Moreover, only one of dec_group_qblock16 is ever used.
  // TODO(veluca): figure out if we can save allocations.

  // AC decoding
  Image3I num_nzeroes[kMaxNumPasses];

  // Buffer for DC upsampling.
  ImageF dc_buffer;

 private:
  hwy::AlignedFreeUniquePtr<float[]> float_memory_;
  hwy::AlignedFreeUniquePtr<int32_t[]> int32_memory_;
  hwy::AlignedFreeUniquePtr<int16_t[]> int16_memory_;
  size_t max_block_area_ = 0;
};

}  // namespace jxl

#endif  // LIB_JXL_DEC_CACHE_H_
