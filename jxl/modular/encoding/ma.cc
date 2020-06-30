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

#include "jxl/modular/encoding/ma.h"

#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "jxl/enc_ans.h"
#include "jxl/modular/encoding/context_predict.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/modular/encoding/ma.cc"
#include <hwy/foreach_target.h>

// SIMD code.
#include <hwy/before_namespace-inl.h>

#include "jxl/fast_log-inl.h"
namespace jxl {
#include <hwy/begin_target-inl.h>

const HWY_FULL(float) df;
const HWY_FULL(int32_t) di;
size_t Padded(size_t x) { return RoundUpTo(x, Lanes(df)); }

float EstimateBits(const int32_t counts[ANS_MAX_ALPHA_SIZE],
                   size_t num_symbols) {
  const auto inv_total =
      Set(df, 1.0f / std::accumulate(counts, counts + num_symbols, 0));
  const auto zero = Zero(df);
  auto bits_lanes = Zero(df);
  for (size_t i = 0; i < num_symbols; i += Lanes(df)) {
    const auto counts_v = ConvertTo(df, LoadU(di, &counts[i]));
    bits_lanes -=
        IfThenElse(counts_v == zero, zero,
                   counts_v * FastLog2f_18bits(df, counts_v * inv_total));
  }
  return GetLane(SumOfLanes(bits_lanes));
}

float EstimateTotalBits(int64_t offset, const std::vector<int> &residuals,
                        const std::vector<size_t> &indices, size_t begin,
                        size_t end) {
  float ans = 0;
  int32_t dist[ANS_MAX_ALPHA_SIZE] = {};
  size_t num_symbols = 0;
  for (size_t i = begin; i < end; i++) {
    uint32_t tok, nbits, bits;
    HybridUintConfig().Encode(PackSigned(residuals[indices[i]] - offset), &tok,
                              &bits, &nbits);
    dist[tok]++;
    ans += nbits;
    num_symbols = num_symbols > tok + 1 ? num_symbols : tok + 1;
  }
  return ans + EstimateBits(dist, num_symbols);
}

float EstimateTotalBitsAndOffset(const std::vector<int> &residuals,
                                 const std::vector<size_t> &indices,
                                 size_t begin, size_t end, int64_t *offset) {
  JXL_ASSERT(begin < end);
  int64_t sum = 0;
  // Minimize sum-of-abs-values.
  for (size_t j = begin; j < end; j++) {
    sum += residuals[indices[j]];
  }
  int64_t tot = end - begin;
  *offset = sum > 0 ? (sum + tot / 2) / tot : (sum - tot / 2) / tot;
  return EstimateTotalBits(*offset, residuals, indices, begin, end);
}

// Compute the entropy obtained by splitting up along each property.
void EstimateEntropy(
    int64_t offset, const std::vector<std::vector<int>> &residuals,
    const std::vector<std::vector<int>> &all_props,
    const std::vector<std::vector<int>> &compact_properties,
    std::vector<std::pair<float, size_t>> *props_with_entropy) {
  std::vector<uint32_t> tokens;
  tokens.reserve(residuals[0].size());
  for (int v : residuals[0]) {
    uint32_t tok, nbits, bits;
    HybridUintConfig().Encode(PackSigned(v - offset), &tok, &bits, &nbits);
    tokens.push_back(tok);
  }
  const size_t num_symbols =
      *std::max_element(tokens.begin(), tokens.end()) + 1;

  int32_t dist[ANS_MAX_ALPHA_SIZE] = {};
  std::vector<size_t> indices(tokens.size());
  std::vector<int> prop_counts;
  // Force usage of static properties.
  for (size_t i = 0; i < kNumStaticProperties; i++) {
    props_with_entropy->emplace_back(0, i);
  }
  for (size_t i = kNumStaticProperties; i < all_props.size(); i++) {
    const auto &props = all_props[i];

    // Counting sort.
    prop_counts.clear();
    prop_counts.resize(compact_properties[i].size() + 1);
    for (size_t j = 0; j < props.size(); j++) {
      prop_counts[props[j] + 1]++;
    }
    for (size_t j = 0; j < prop_counts.size() - 1; j++) {
      prop_counts[j + 1] += prop_counts[j];
    }
    for (size_t j = 0; j < props.size(); j++) {
      indices[prop_counts[props[j]]++] = j;
    }

    size_t previous_position = 0;
    size_t current_position = 0;
    float entropy = 0;
    for (; current_position < props.size(); current_position++) {
      previous_position = current_position;
      while (current_position < props.size() &&
             props[indices[current_position]] ==
                 props[indices[previous_position]]) {
        dist[tokens[indices[current_position]]]++;
        current_position++;
      }
      entropy += EstimateBits(dist, num_symbols);
      while (previous_position < current_position) {
        dist[tokens[indices[previous_position]]]--;
        previous_position++;
      }
    }
    props_with_entropy->emplace_back(entropy, i);
  }
}

void MakeSplitNode(size_t pos, int property, int splitval, Predictor lpred,
                   int64_t loff, Predictor rpred, int64_t roff, Tree *tree) {
  // Note that the tree splits on *strictly greater*.
  (*tree)[pos].childID = tree->size();
  (*tree)[pos].splitval = splitval;
  (*tree)[pos].property = property;
  tree->emplace_back();
  tree->back().predictor = rpred;
  tree->back().predictor_offset = roff;
  tree->emplace_back();
  tree->back().predictor = lpred;
  tree->back().predictor_offset = loff;
}

void FindBestSplit(const std::vector<std::vector<int>> &residuals,
                   const std::vector<std::vector<int>> &props,
                   const std::vector<Predictor> predictors,
                   const std::vector<std::vector<int>> &compact_properties,
                   std::vector<size_t> *indices, size_t pos, size_t begin,
                   size_t end, const std::vector<size_t> &props_to_use,
                   float base_bits, float threshold, Tree *tree) {
  if (begin == end) return;

  size_t split_prop = 0;
  int split_val = 0;
  size_t split_pos = begin;
  float best_split_l = std::numeric_limits<float>::max();
  float best_split_r = std::numeric_limits<float>::max();

  Predictor lpred = predictors[0];
  Predictor rpred = predictors[0];

  std::vector<std::vector<uint32_t>> tokens(residuals.size());
  for (auto &v : tokens) {
    v.reserve(end - begin);
  }
  std::vector<std::vector<uint32_t>> extra_bits(residuals.size());
  for (auto &v : extra_bits) {
    v.reserve(end - begin);
  }

  // Compute the tokens corresponding to the residuals.
  size_t max_symbols = 0;
  for (size_t pred = 0; pred < residuals.size(); pred++) {
    for (size_t i = begin; i < end; i++) {
      uint32_t tok, nbits, bits;
      HybridUintConfig().Encode(PackSigned(residuals[pred][(*indices)[i]]),
                                &tok, &bits, &nbits);
      tokens[pred].push_back(tok);
      extra_bits[pred].push_back(nbits);
      max_symbols = max_symbols > tok + 1 ? max_symbols : tok + 1;
    }
  }
  max_symbols = Padded(max_symbols);
  std::vector<int32_t> counts(max_symbols * residuals.size());
  std::vector<int32_t> tot_extra_bits(residuals.size());
  JXL_ASSERT(begin <= end);
  JXL_ASSERT(end <= indices->size());
  for (size_t pred = 0; pred < tokens.size(); pred++) {
    for (size_t i = 0; i < tokens[pred].size(); i++) {
      counts[pred * max_symbols + tokens[pred][i]]++;
      tot_extra_bits[pred] += extra_bits[pred][i];
    }
  }

  std::vector<int> prop_value_used_count;
  std::vector<int> prop_count_increase;
  std::vector<size_t> extra_bits_increase;
  // For each property, compute which of its values are used, and what
  // tokens correspond to those usages. Then, iterate through the values,
  // and compute the entropy of each side of the split (of the form `prop >
  // threshold`). Finally, find the split that minimizes the cost.
  struct CostInfo {
    float cost = std::numeric_limits<float>::max();
    Predictor pred;  // will be uninitialized in some cases, but never used.
  };
  std::vector<CostInfo> costs_l;
  std::vector<CostInfo> costs_r;
  constexpr float kChangePredPenalty = 4;
  for (size_t prop = 0; prop < props.size(); prop++) {
    costs_l.clear();
    costs_r.clear();
    costs_l.resize(end - begin);
    costs_r.resize(end - begin);
    if (prop_value_used_count.size() < compact_properties[prop].size()) {
      prop_value_used_count.resize(compact_properties[prop].size());
      prop_count_increase.resize(compact_properties[prop].size() * max_symbols *
                                 residuals.size());
      extra_bits_increase.resize(compact_properties[prop].size() *
                                 residuals.size());
    }

    size_t first_used = compact_properties[prop].size();
    size_t last_used = 0;

    for (size_t i = begin; i < end; i++) {
      size_t p = props[prop][(*indices)[i]];
      prop_value_used_count[p]++;
      for (size_t pred = 0; pred < residuals.size(); pred++) {
        size_t sym = tokens[pred][i - begin];
        prop_count_increase[p * max_symbols * residuals.size() +
                            max_symbols * pred + sym]++;
        extra_bits_increase[p * residuals.size() + pred] +=
            extra_bits[pred][i - begin];
      }
      last_used = std::max(last_used, p);
      first_used = std::min(first_used, p);
    }

    // For all predictors, compute the right and left costs of each split.
    for (size_t pred = 0; pred < residuals.size(); pred++) {
      int32_t counts_above[ANS_MAX_ALPHA_SIZE];
      memcpy(counts_above, counts.data() + pred * max_symbols,
             max_symbols * sizeof *counts_above);
      int32_t counts_below[ANS_MAX_ALPHA_SIZE];
      memset(counts_below, 0, max_symbols * sizeof *counts_below);
      size_t extra_bits_below = 0;
      // Exclude last used: this ensures neither counts_above nor counts_below
      // is empty.
      size_t split = begin;
      for (size_t i = first_used; i < last_used; i++) {
        if (!prop_value_used_count[i]) continue;
        split += prop_value_used_count[i];
        extra_bits_below += extra_bits_increase[i * residuals.size() + pred];
        for (size_t sym = 0; sym < max_symbols; sym++) {
          counts_above[sym] -=
              prop_count_increase[i * max_symbols * residuals.size() +
                                  max_symbols * pred + sym];
          counts_below[sym] +=
              prop_count_increase[i * max_symbols * residuals.size() +
                                  max_symbols * pred + sym];
        }
        float rcost = EstimateBits(counts_above, max_symbols) +
                      tot_extra_bits[pred] - extra_bits_below;
        float lcost =
            EstimateBits(counts_below, max_symbols) + extra_bits_below;
        if (predictors[pred] != (*tree)[pos].predictor) {
          rcost += kChangePredPenalty;
          lcost += kChangePredPenalty;
        }
        if (rcost < costs_r[split - begin].cost) {
          costs_r[split - begin].cost = rcost;
          costs_r[split - begin].pred = predictors[pred];
        }
        if (lcost < costs_l[split - begin].cost) {
          costs_l[split - begin].cost = lcost;
          costs_l[split - begin].pred = predictors[pred];
        }
      }
    }
    // Iterate through the possible splits and find the one with minimum sum of
    // costs of the two sides.
    size_t split = begin;
    for (size_t i = first_used; i < last_used; i++) {
      if (!prop_value_used_count[i]) continue;
      split += prop_value_used_count[i];
      float rcost = costs_r[split - begin].cost;
      float lcost = costs_l[split - begin].cost;
      if (lcost + rcost < best_split_l + best_split_r) {
        split_prop = prop;
        split_val = i;
        split_pos = split;
        best_split_l = lcost;
        lpred = costs_l[split - begin].pred;
        best_split_r = rcost;
        rpred = costs_r[split - begin].pred;
      }
    }
    // Clear prop_count_increase, extra_bits_increase and prop_value_used_count
    // arrays.
    for (size_t pred = 0; pred < residuals.size(); pred++) {
      for (size_t i = begin; i < end; i++) {
        size_t p = props[prop][(*indices)[i]];
        size_t sym = tokens[pred][i - begin];
        prop_count_increase[p * max_symbols * residuals.size() +
                            max_symbols * pred + sym] = 0;
        prop_value_used_count[p] = 0;
        extra_bits_increase[p * residuals.size() + pred] = 0;
      }
    }
  }

  if (best_split_l + best_split_r + threshold < base_bits) {
    // Split node and try to split children.
    MakeSplitNode(pos, props_to_use[split_prop],
                  compact_properties[split_prop][split_val], lpred, 0, rpred, 0,
                  tree);
    // "Sort" according to winning property
    std::nth_element(indices->begin() + begin, indices->begin() + split_pos,
                     indices->begin() + end, [&](size_t a, size_t b) {
                       return props[split_prop][a] < props[split_prop][b];
                     });
    FindBestSplit(residuals, props, predictors, compact_properties, indices,
                  (*tree)[pos].childID + 1, begin, split_pos, props_to_use,
                  best_split_l, threshold, tree);
    FindBestSplit(residuals, props, predictors, compact_properties, indices,
                  (*tree)[pos].childID, split_pos, end, props_to_use,
                  best_split_r, threshold, tree);
  } else {
    // try to pick an offset for the leaves.
    size_t pred = 0;
    for (size_t i = 0; i < predictors.size(); i++) {
      if (predictors[i] == (*tree)[pos].predictor) {
        pred = i;
        break;
      }
    }
    int64_t o;
    float c =
        EstimateTotalBitsAndOffset(residuals[pred], *indices, begin, end, &o);
    // Cost estimate of encoding the offset. Huge constant penalty to avoid
    // significant increases in tree size.
    c += 200.0f + FloorLog2Nonzero(PackSigned(o) + 1);
    if (c < base_bits) {
      (*tree)[pos].predictor_offset = o;
    }
  }
}

#include <hwy/end_target-inl.h>
}  // namespace jxl
#include <hwy/after_namespace-inl.h>

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(EstimateEntropy)    // Local function.
HWY_EXPORT(EstimateTotalBits)  // Local function.
HWY_EXPORT(FindBestSplit)      // Local function.

void ChooseAndQuantizeProperties(
    size_t max_properties, size_t max_property_values, int64_t offset,
    const std::vector<std::vector<int>> &residuals,
    std::vector<std::vector<int>> *props,
    std::vector<std::vector<int>> *compact_properties,
    std::vector<size_t> *props_to_use) {
  // Remap all properties so that there are no holes nor negative numbers.
  std::unordered_map<int, int> remap;
  std::unordered_set<int> is_present;

  std::vector<int> remap_v;
  std::vector<int> is_present_v;

  // Threshold to switch to using a hash table for property remapping.
  static constexpr size_t kVectorMaxRange = 4096;

  for (size_t i = 0; i < props->size(); i++) {
    PropertyVal min = std::numeric_limits<PropertyVal>::max();
    PropertyVal max = std::numeric_limits<PropertyVal>::min();
    for (PropertyVal x : (*props)[i]) {
      min = std::min(min, x);
      max = std::max(max, x);
    }
    if (max - min + 1 < kVectorMaxRange) {
      is_present_v.clear();
      is_present_v.resize(max - min + 1);
      remap_v.resize(max - min + 1);
      for (size_t j = 0; j < (*props)[i].size(); j++) {
        size_t idx = (*props)[i][j] - min;
        if (!is_present_v[idx]) {
          (*compact_properties)[i].push_back((*props)[i][j]);
        }
        is_present_v[idx] = 1;
      }
      std::sort((*compact_properties)[i].begin(),
                (*compact_properties)[i].end());
      for (size_t j = 0; j < (*compact_properties)[i].size(); j++) {
        remap_v[(*compact_properties)[i][j] - min] = j;
      }
      for (size_t j = 0; j < (*props)[i].size(); j++) {
        (*props)[i][j] = remap_v[(*props)[i][j] - min];
      }
    } else {
      is_present.clear();
      for (size_t j = 0; j < (*props)[i].size(); j++) {
        is_present.insert((*props)[i][j]);
      }
      (*compact_properties)[i].assign(is_present.begin(), is_present.end());
      std::sort((*compact_properties)[i].begin(),
                (*compact_properties)[i].end());
      for (size_t j = 0; j < (*compact_properties)[i].size(); j++) {
        remap[(*compact_properties)[i][j]] = j;
      }
      for (size_t j = 0; j < (*props)[i].size(); j++) {
        (*props)[i][j] = remap.at((*props)[i][j]);
      }
    }
  }

  std::vector<std::pair<float, size_t>> props_with_entropy;
  HWY_DYNAMIC_DISPATCH(EstimateEntropy)
  (offset, residuals, *props, *compact_properties, &props_with_entropy);
  std::sort(props_with_entropy.begin(), props_with_entropy.end());

  // Limit the search to the properties with the smallest resulting entropy
  // (including static properties).
  max_properties = std::min(max_properties + kNumStaticProperties,
                            props_with_entropy.size());
  props_to_use->resize(max_properties);
  for (size_t i = 0; i < max_properties; i++) {
    (*props_to_use)[i] = props_with_entropy[i].second;
  }
  // Remove other properties from the data.
  size_t num_property_values = 0;
  std::sort(props_to_use->begin(), props_to_use->end());
  for (size_t i = 0; i < props_to_use->size(); i++) {
    if (i == (*props_to_use)[i]) continue;
    (*props)[i] = std::move((*props)[(*props_to_use)[i]]);
    (*compact_properties)[i] =
        std::move((*compact_properties)[(*props_to_use)[i]]);
    num_property_values += (*compact_properties)[i].size();
  }
  props->resize(props_to_use->size());
  compact_properties->resize(props_to_use->size());

  if (num_property_values > max_property_values) {
    // Quantize properties.
    // Note that tree uses *strictly greater* comparison nodes, so when merging
    // together a sequence of consecutive values we should keep the largest one.
    // TODO(veluca): find a smarter way to do the quantization, taking into
    // account the actual distribution of symbols.
    size_t thres = residuals[0].size() / 256;
    std::vector<int> counts;
    std::vector<int> remap;
    std::vector<int> new_cp;
    for (size_t i = 0; i < max_properties; i++) {
      counts.clear();
      counts.resize((*compact_properties)[i].size());
      remap.resize((*compact_properties)[i].size());
      new_cp.clear();
      for (int v : (*props)[i]) {
        counts[v]++;
      }
      size_t running_count = 0;
      size_t remapped = 0;
      for (size_t j = 0; j < counts.size(); j++) {
        remap[j] = remapped;
        running_count += counts[j];
        if (running_count > thres) {
          remapped++;
          running_count = 0;
        }
      }
      if (running_count != 0) remapped++;
      new_cp.resize(remapped, std::numeric_limits<int>::min());
      for (size_t j = 0; j < counts.size(); j++) {
        new_cp[remap[j]] =
            std::max(new_cp[remap[j]], (*compact_properties)[i][j]);
      }
      (*compact_properties)[i] = new_cp;
      for (size_t j = 0; j < (*props)[i].size(); j++) {
        (*props)[i][j] = remap[(*props)[i][j]];
      }
    }
  }
}

void ComputeBestTree(const std::vector<std::vector<int>> &residuals,
                     const std::vector<std::vector<int>> &props,
                     const std::vector<Predictor> &predictors,
                     int64_t base_offset,
                     const std::vector<std::vector<int>> compact_properties,
                     const std::vector<size_t> &props_to_use, float threshold,
                     size_t max_properties, Tree *tree) {
  // TODO(veluca): take into account that different contexts can have different
  // uint configs.
  //
  // Initialize tree.
  tree->emplace_back();
  tree->back().predictor = predictors[0];
  tree->back().predictor_offset = base_offset;

  std::vector<size_t> indices(residuals[0].size());
  std::iota(indices.begin(), indices.end(), 0);
  float base_bits = HWY_DYNAMIC_DISPATCH(EstimateTotalBits)(
      base_offset, residuals[0], indices, 0, indices.size());
  HWY_DYNAMIC_DISPATCH(FindBestSplit)
  (residuals, props, predictors, compact_properties, &indices, 0, 0,
   indices.size(), props_to_use, base_bits, threshold, tree);
  size_t leaves = 0;
  for (size_t i = 0; i < tree->size(); i++) {
    if ((*tree)[i].property < 0) {
      (*tree)[i].childID = leaves++;
    }
  }
}

namespace {
constexpr size_t kSplitValContext = 0;
constexpr size_t kPropertyContext = 1;
constexpr size_t kPredictorContext = 2;
constexpr size_t kOffsetContext = 3;
}  // namespace

static constexpr size_t kMaxTreeSize = 1 << 26;

// TODO(veluca): very simple encoding scheme. This should be improved.
void TokenizeTree(const Tree &tree, std::vector<Token> *tokens,
                  Tree *decoder_tree) {
  JXL_ASSERT(tree.size() <= kMaxTreeSize);
  std::queue<int> q;
  q.push(0);
  size_t leaf_id = 0;
  decoder_tree->clear();
  while (!q.empty()) {
    int cur = q.front();
    q.pop();
    JXL_ASSERT(tree[cur].property >= -1);
    tokens->emplace_back(kPropertyContext, tree[cur].property + 1);
    if (tree[cur].property == -1) {
      tokens->emplace_back(kPredictorContext,
                           static_cast<int>(tree[cur].predictor));
      tokens->emplace_back(kOffsetContext,
                           PackSigned(tree[cur].predictor_offset));
      JXL_ASSERT(tree[cur].predictor < Predictor::Best);
      decoder_tree->push_back(PropertyDecisionNode(
          -1, 0, leaf_id++, tree[cur].predictor, tree[cur].predictor_offset));
      continue;
    }
    decoder_tree->push_back(
        PropertyDecisionNode(tree[cur].property, tree[cur].splitval,
                             decoder_tree->size() + q.size() + 1));
    q.push(tree[cur].childID);
    q.push(tree[cur].childID + 1);
    tokens->emplace_back(kSplitValContext, PackSigned(tree[cur].splitval));
  }
}

Status ValidateTree(
    const Tree &tree,
    const std::vector<std::pair<pixel_type, pixel_type>> &prop_bounds,
    size_t root) {
  if (tree[root].property == -1) return true;
  size_t p = tree[root].property;
  int val = tree[root].splitval;
  if (prop_bounds[p].first > val) return JXL_FAILURE("Invalid tree");
  if (prop_bounds[p].second < val) return JXL_FAILURE("Invalid tree");
  auto new_bounds = prop_bounds;
  new_bounds[p].first = val + 1;
  JXL_RETURN_IF_ERROR(ValidateTree(tree, new_bounds, tree[root].childID));
  new_bounds[p] = prop_bounds[p];
  new_bounds[p].second = val;
  return ValidateTree(tree, new_bounds, tree[root].childID + 1);
}

Status DecodeTree(BitReader *br, ANSSymbolReader *reader,
                  const std::vector<uint8_t> &context_map, Tree *tree,
                  int max_property) {
  size_t leaf_id = 0;
  size_t to_decode = 1;
  tree->clear();
  while (to_decode > 0) {
    if (tree->size() > kMaxTreeSize) {
      return JXL_FAILURE("Tree is too large");
    }
    to_decode--;
    int property =
        reader->ReadHybridUint(kPropertyContext, br, context_map) - 1;
    if (property < -1 || property > max_property) {
      return JXL_FAILURE("Invalid tree property value");
    }
    if (property == -1) {
      size_t predictor =
          reader->ReadHybridUint(kPredictorContext, br, context_map);
      if (predictor >= kNumModularPredictors) {
        return JXL_FAILURE("Invalid predictor");
      }
      int64_t predictor_offset =
          UnpackSigned(reader->ReadHybridUint(kOffsetContext, br, context_map));
      tree->push_back(PropertyDecisionNode(-1, 0, leaf_id++,
                                           static_cast<Predictor>(predictor),
                                           predictor_offset));
      continue;
    }
    int splitval =
        UnpackSigned(reader->ReadHybridUint(kSplitValContext, br, context_map));
    tree->push_back(
        PropertyDecisionNode(property, splitval, tree->size() + to_decode + 1));
    to_decode += 2;
  }
  std::vector<std::pair<pixel_type, pixel_type>> prop_bounds;
  prop_bounds.resize(max_property + 1,
                     {std::numeric_limits<pixel_type>::min(),
                      std::numeric_limits<pixel_type>::max()});
  return ValidateTree(*tree, prop_bounds, 0);
}

}  // namespace jxl
#endif  // HWY_ONCE
