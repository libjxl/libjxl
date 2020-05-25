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

#ifndef JXL_MODULAR_ENCODING_MA_H_
#define JXL_MODULAR_ENCODING_MA_H_

#include "jxl/entropy_coder.h"
#include "jxl/modular/options.h"

namespace jxl {

// inner nodes
struct PropertyDecisionNode {
  PropertyVal splitval;
  int16_t property;  // -1: leaf node, childID points to leaf node
  // 0..nb_properties-1 : childID refers to left branch  (in inner_node)
  //                      childID+1 refers to right branch
  uint16_t childID;
  Predictor predictor;
  int64_t predictor_offset;

  explicit PropertyDecisionNode(int p = -1, int split_val = 0, int child_id = 0,
                                Predictor predictor = Predictor::Gradient,
                                int64_t predictor_offset = 0)
      : splitval(split_val),
        property(p),
        childID(child_id),
        predictor(predictor),
        predictor_offset(predictor_offset) {}
};

class Tree : public std::vector<PropertyDecisionNode> {};

constexpr size_t kNumTreeContexts = 4;

void TokenizeTree(const Tree &tree, const HybridUintConfig &uint_config,
                  size_t base_ctx, std::vector<Token> *tokens,
                  Tree *decoder_tree);

Status DecodeTree(BitReader *br, ANSSymbolReader *reader,
                  const std::vector<uint8_t> &context_map, size_t base_ctx,
                  Tree *tree, int max_property);

void ChooseAndQuantizeProperties(
    size_t max_properties, size_t max_property_values,
    const HybridUintConfig &uint_config, int64_t offset,
    const std::vector<std::vector<int>> &residuals,
    std::vector<std::vector<int>> *props,
    std::vector<std::vector<int>> *compact_properties,
    std::vector<size_t> *props_to_use);

void ComputeBestTree(const std::vector<std::vector<int>> &residuals,
                     const std::vector<std::vector<int>> &props,
                     const std::vector<Predictor> &predictors,
                     const HybridUintConfig &uint_config, int64_t base_offset,
                     const std::vector<std::vector<int>> compact_properties,
                     const std::vector<size_t> &props_to_use, float threshold,
                     size_t max_properties, Tree *tree);
}  // namespace jxl
#endif
