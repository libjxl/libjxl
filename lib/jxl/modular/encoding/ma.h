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

#ifndef LIB_JXL_MODULAR_ENCODING_MA_H_
#define LIB_JXL_MODULAR_ENCODING_MA_H_

#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/modular/options.h"

namespace jxl {

// inner nodes
struct PropertyDecisionNode {
  PropertyVal splitval;
  int16_t property;  // -1: leaf node, lchild points to leaf node
  uint32_t lchild;
  uint32_t rchild;
  Predictor predictor;
  int64_t predictor_offset;
  uint32_t multiplier;

  PropertyDecisionNode(int p, int split_val, int lchild, int rchild,
                       Predictor predictor, int64_t predictor_offset,
                       uint32_t multiplier)
      : splitval(split_val),
        property(p),
        lchild(lchild),
        rchild(rchild),
        predictor(predictor),
        predictor_offset(predictor_offset),
        multiplier(multiplier) {}
  PropertyDecisionNode()
      : splitval(0),
        property(-1),
        lchild(0),
        rchild(0),
        predictor(Predictor::Zero),
        predictor_offset(0),
        multiplier(1){};
  static PropertyDecisionNode Leaf(Predictor predictor, int64_t offset = 0,
                                   uint32_t multiplier = 1) {
    return PropertyDecisionNode(-1, 0, 0, 0, predictor, offset, multiplier);
  }
  static PropertyDecisionNode Split(int p, int split_val, int lchild,
                                    int rchild = -1) {
    if (rchild == -1) rchild = lchild + 1;
    return PropertyDecisionNode(p, split_val, lchild, rchild, Predictor::Zero,
                                0, 1);
  }
};

using Tree = std::vector<PropertyDecisionNode>;

constexpr size_t kNumTreeContexts = 6;

void TokenizeTree(const Tree &tree, std::vector<Token> *tokens,
                  Tree *decoder_tree);

Status DecodeTree(BitReader *br, ANSSymbolReader *reader,
                  const std::vector<uint8_t> &context_map, Tree *tree);

void ChooseAndQuantizeProperties(
    size_t max_properties, size_t max_property_values,
    const std::vector<std::vector<int>> &residuals, bool force_wp_only,
    std::vector<std::vector<int>> *props,
    std::vector<std::vector<int>> *compact_properties,
    std::vector<size_t> *props_to_use);

void ComputeBestTree(const std::vector<std::vector<int>> &residuals,
                     const std::vector<std::vector<int>> &props,
                     const std::vector<Predictor> &predictors,
                     const std::vector<std::vector<int>> compact_properties,
                     const std::vector<size_t> &props_to_use, float threshold,
                     size_t max_properties,
                     const std::vector<ModularMultiplierInfo> &mul_info,
                     StaticPropRange static_prop_range,
                     float fast_decode_multiplier, Tree *tree);

}  // namespace jxl
#endif  // LIB_JXL_MODULAR_ENCODING_MA_H_
