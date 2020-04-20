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

#ifndef JXL_PASSES_STATE_H_
#define JXL_PASSES_STATE_H_

#include "jxl/ac_strategy.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/common.h"
#include "jxl/dot_dictionary.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/image_bundle.h"
#include "jxl/loop_filter.h"
#include "jxl/multiframe.h"
#include "jxl/noise.h"
#include "jxl/patch_dictionary.h"
#include "jxl/quant_weights.h"
#include "jxl/quantizer.h"
#include "jxl/splines.h"

// Structures that hold the (en/de)coder state for a JPEG XL kVarDCT
// (en/de)coder.

namespace jxl {

struct ImageFeatures {
  LoopFilter loop_filter;
  NoiseParams noise_params;
  PatchDictionary patches;
  Splines splines;
};

// State common to both encoder and decoder.
// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct PassesSharedState {
  // Headers and metadata.
  FrameHeader frame_header;
  ImageMetadata metadata;

  FrameDimensions frame_dim;

  // Control fields and parameters.
  AcStrategyImage ac_strategy;

  // Dequant matrices + quantizer.
  DequantMatrices matrices;
  Quantizer quantizer{&matrices};
  ImageI raw_quant_field;

  // Per-block side information for EPF detail preservation.
  ImageB epf_sharpness;

  ColorCorrelationMap cmap;

  OpsinParams opsin_params;

  ImageFeatures image_features;

  // Memory area for storing coefficient orders.
  std::vector<coeff_order_t> coeff_orders =
      std::vector<coeff_order_t>(kMaxNumPasses * kCoeffOrderSize);

  // Decoder-side DC and DC quantization information.
  Image3F dc_storage;
  const Image3F* JXL_RESTRICT dc = &dc_storage;

  // Only useful if adaptive DC smoothing is enabled.
  Image3F dc_quant_field;

  Multiframe* JXL_RESTRICT multiframe = nullptr;

  // Number of histograms and coefficient orders, per pass (always 1 for
  // now). Encoded as num_histograms_ - 1.
  size_t num_histograms = 0;

  bool IsGrayscale() const { return metadata.color_encoding.IsGray(); }

  Rect GroupRect(size_t group_index) const {
    const size_t gx = group_index % frame_dim.xsize_groups;
    const size_t gy = group_index / frame_dim.xsize_groups;
    const Rect rect(gx * kGroupDim, gy * kGroupDim, kGroupDim, kGroupDim,
                    frame_dim.xsize, frame_dim.ysize);
    return rect;
  }

  Rect PaddedGroupRect(size_t group_index) const {
    const size_t gx = group_index % frame_dim.xsize_groups;
    const size_t gy = group_index / frame_dim.xsize_groups;
    const Rect rect(gx * kGroupDim, gy * kGroupDim, kGroupDim, kGroupDim,
                    frame_dim.xsize_padded, frame_dim.ysize_padded);
    return rect;
  }

  Rect BlockGroupRect(size_t group_index) const {
    const size_t gx = group_index % frame_dim.xsize_groups;
    const size_t gy = group_index / frame_dim.xsize_groups;
    const Rect rect(gx * kGroupDimInBlocks, gy * kGroupDimInBlocks,
                    kGroupDimInBlocks, kGroupDimInBlocks,
                    frame_dim.xsize_blocks, frame_dim.ysize_blocks);
    return rect;
  }

  Rect DCGroupRect(size_t group_index) const {
    const size_t gx = group_index % frame_dim.xsize_dc_groups;
    const size_t gy = group_index / frame_dim.xsize_dc_groups;
    const Rect rect(gx * kDcGroupDimInBlocks, gy * kDcGroupDimInBlocks,
                    kDcGroupDimInBlocks, kDcGroupDimInBlocks,
                    frame_dim.xsize_blocks, frame_dim.ysize_blocks);
    return rect;
  }
};

// Initialized the state information that is shared between encoder and decoder.
Status InitializePassesSharedState(const FrameHeader& frame_header,
                                   const LoopFilter& loop_filter,
                                   const ImageMetadata& image_metadata,
                                   const FrameDimensions& frame_dim,
                                   Multiframe* JXL_RESTRICT multiframe,
                                   PassesSharedState* JXL_RESTRICT shared,
                                   bool encoder = false);

}  // namespace jxl

#endif  // JXL_PASSES_STATE_H_
