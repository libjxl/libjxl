// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_ENCODING_ENC_ENCODING_H_
#define LIB_JXL_MODULAR_ENCODING_ENC_ENCODING_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/base/status.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;
struct GroupHeader;

Tree PredefinedTree(ModularOptions::TreeKind tree_kind, size_t total_pixels,
                    int bitdepth, int prevprop);

StatusOr<Tree> LearnTree(
    const Image *images, const ModularOptions *opts, uint32_t start,
    uint32_t stop,
    const std::vector<ModularMultiplierInfo> &multiplier_info = {});

// Default single-image compress.
Status ModularGenericCompress(const Image &image, const ModularOptions &opts,
                              BitWriter &writer, AuxOut *aux_out = nullptr,
                              LayerType layer = static_cast<LayerType>(0),
                              size_t group_id = 0);

// Computes the image a decoder would get from `tree` with all residuals zero
// (as in a stream encoded with ModularOptions::zero_tokens and no transforms):
// the channels of `image` must have their sizes set; their samples are
// overwritten. `group_id` is the stream ID (tree property 1).
// The weighted predictor header of a stream encoded with `options`.
void SetWPHeader(const ModularOptions& options, weighted::Header* header);

Status EvaluateTreeWithZeroResiduals(
    const Tree& tree, size_t group_id, Image* image,
    const weighted::Header& wp_header = weighted::Header());

// The same with the residuals of `pattern` (as in ModularOptions::
// residual_patterns: they start after the meta channels; null: all zero) for
// the channels [first_channel, num_channels) of `image`, and the smallest and
// largest sample values seen. Samples of the other channels are kept.
Status EvaluateTreeWithResiduals(
    const Tree& tree, size_t group_id, const ResidualPattern* pattern,
    size_t first_channel, size_t num_channels, Image* image, int64_t* min_value,
    int64_t* max_value, const weighted::Header& wp_header = weighted::Header());

// For encoding with a given tree.
Status ModularCompress(const Image &image, const ModularOptions &opts,
                       size_t group_id, const Tree &tree, GroupHeader &header,
                       std::vector<Token> &tokens, size_t *width);
}  // namespace jxl

#endif  // LIB_JXL_MODULAR_ENCODING_ENC_ENCODING_H_
