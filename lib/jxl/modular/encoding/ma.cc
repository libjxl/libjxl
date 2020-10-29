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

#include "lib/jxl/modular/encoding/ma.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/modular/encoding/ma.cc"
#include <hwy/foreach_target.h>
// ^ must come before highway.h and any *-inl.h.

#include <hwy/highway.h>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include "lib/jxl/enc_ans.h"
#include "lib/jxl/fast_log-inl.h"
#include "lib/jxl/modular/encoding/context_predict.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

const HWY_FULL(float) df;
const HWY_FULL(int32_t) di;
size_t Padded(size_t x) { return RoundUpTo(x, Lanes(df)); }

float EstimateBits(const int32_t *counts, int32_t *rounded_counts,
                   size_t num_symbols) {
  // Try to approximate the effect of rounding up nonzero probabilities.
  int32_t total = std::accumulate(counts, counts + num_symbols, 0);
  const auto min = Set(di, (total + ANS_TAB_SIZE - 1) >> ANS_LOG_TAB_SIZE);
  const auto zero_i = Zero(di);
  for (size_t i = 0; i < num_symbols; i += Lanes(df)) {
    auto counts_v = LoadU(di, &counts[i]);
    counts_v = IfThenElse(counts_v == zero_i, zero_i,
                          IfThenElse(counts_v < min, min, counts_v));
    StoreU(counts_v, di, &rounded_counts[i]);
  }
  // Compute entropy of the "rounded" probabilities.
  const auto zero = Zero(df);
  const size_t total_scalar =
      std::accumulate(rounded_counts, rounded_counts + num_symbols, 0);
  const auto inv_total = Set(df, 1.0f / total_scalar);
  auto bits_lanes = Zero(df);
  auto total_v = Set(di, total_scalar);
  for (size_t i = 0; i < num_symbols; i += Lanes(df)) {
    const auto counts_v = ConvertTo(df, LoadU(di, &counts[i]));
    const auto round_counts_v = LoadU(di, &rounded_counts[i]);
    const auto probs = ConvertTo(df, round_counts_v) * inv_total;
    const auto nbps = IfThenElse(round_counts_v == total_v, BitCast(di, zero),
                                 BitCast(di, FastLog2f_18bits(df, probs)));
    bits_lanes -=
        IfThenElse(counts_v == zero, zero, counts_v * BitCast(df, nbps));
  }
  return GetLane(SumOfLanes(bits_lanes));
}

float EstimateTotalBits(int64_t offset, const std::vector<int> &residuals,
                        const std::vector<size_t> &indices, size_t begin,
                        size_t end) {
  float ans = 0;
  std::vector<int32_t> dist;
  size_t num_symbols = 0;
  for (size_t i = begin; i < end; i++) {
    uint32_t tok, nbits, bits;
    HybridUintConfig(4, 1, 2).Encode(PackSigned(residuals[indices[i]] - offset),
                                     &tok, &bits, &nbits);
    if (tok >= dist.size()) {
      dist.resize(Padded(tok + 1));
    }
    dist[tok]++;
    ans += nbits;
    num_symbols = num_symbols > tok + 1 ? num_symbols : tok + 1;
  }
  std::vector<int32_t> rounded_dist(dist.size());
  return ans + EstimateBits(dist.data(), rounded_dist.data(), num_symbols);
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
  uint32_t num_symbols = 0;
  for (int v : residuals[0]) {
    uint32_t tok, nbits, bits;
    HybridUintConfig(4, 1, 2).Encode(PackSigned(v - offset), &tok, &bits,
                                     &nbits);
    tokens.push_back(tok);
    num_symbols = std::max(tok + 1, num_symbols);
  }

  std::vector<int32_t> dist(Padded(num_symbols));
  std::vector<int32_t> rounded_dist(Padded(num_symbols));
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
        size_t tok = tokens[indices[current_position]];
        dist[tok]++;
        current_position++;
      }
      entropy += EstimateBits(dist.data(), rounded_dist.data(), num_symbols);
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
  (*tree)[pos].lchild = tree->size();
  (*tree)[pos].rchild = tree->size() + 1;
  (*tree)[pos].splitval = splitval;
  (*tree)[pos].property = property;
  tree->emplace_back();
  tree->back().property = -1;
  tree->back().predictor = rpred;
  tree->back().predictor_offset = roff;
  tree->back().multiplier = 1;
  tree->emplace_back();
  tree->back().property = -1;
  tree->back().predictor = lpred;
  tree->back().predictor_offset = loff;
  tree->back().multiplier = 1;
}

enum class IntersectionType { kNone, kPartial, kInside };
IntersectionType BoxIntersects(StaticPropRange needle, StaticPropRange haystack,
                               uint32_t &partial_axis, uint32_t &partial_val) {
  bool partial = false;
  for (size_t i = 0; i < kNumStaticProperties; i++) {
    if (haystack[i][0] >= needle[i][1]) {
      return IntersectionType::kNone;
    }
    if (haystack[i][1] <= needle[i][0]) {
      return IntersectionType::kNone;
    }
    if (haystack[i][0] <= needle[i][0] && haystack[i][1] >= needle[i][1]) {
      continue;
    }
    partial = true;
    partial_axis = i;
    if (haystack[i][0] > needle[i][0] && haystack[i][0] < needle[i][1]) {
      partial_val = haystack[i][0] - 1;
    } else {
      JXL_DASSERT(haystack[i][1] > needle[i][0] &&
                  haystack[i][1] < needle[i][1]);
      partial_val = haystack[i][1] - 1;
    }
  }
  return partial ? IntersectionType::kPartial : IntersectionType::kInside;
}

void FindBestSplit(const std::vector<std::vector<int>> &residuals,
                   const std::vector<std::vector<int>> &props,
                   const std::vector<Predictor> predictors,
                   const std::vector<std::vector<int>> &compact_properties,
                   std::vector<size_t> *indices,
                   const std::vector<size_t> &props_to_use, float threshold,
                   const std::vector<ModularMultiplierInfo> &mul_info,
                   StaticPropRange initial_static_prop_range,
                   float fast_decode_multiplier, Tree *tree) {
  struct NodeInfo {
    size_t pos;
    size_t begin;
    size_t end;
    uint64_t used_properties;
    StaticPropRange static_prop_range;
  };
  std::vector<NodeInfo> nodes;
  nodes.push_back(
      NodeInfo{0, 0, indices->size(), 0, initial_static_prop_range});
  // TODO(veluca): consider parallelizing the search (processing multiple nodes
  // at a time).
  while (!nodes.empty()) {
    size_t pos = nodes.back().pos;
    size_t begin = nodes.back().begin;
    size_t end = nodes.back().end;
    uint64_t used_properties = nodes.back().used_properties;
    StaticPropRange static_prop_range = nodes.back().static_prop_range;
    nodes.pop_back();
    if (begin == end) continue;

    int wp_prop = props_to_use.size();
    for (size_t i = 0; i < props_to_use.size(); i++) {
      if (props_to_use[i] == kNumNonrefProperties - weighted::kNumProperties) {
        wp_prop = i;
      }
    }

    struct SplitInfo {
      size_t prop = 0;
      int val = 0;
      size_t pos = 0;
      float lcost = std::numeric_limits<float>::max();
      float rcost = std::numeric_limits<float>::max();
      Predictor lpred = Predictor::Zero;
      Predictor rpred = Predictor::Zero;
      float Cost() { return lcost + rcost; }
    };

    SplitInfo best_split_static_constant;
    SplitInfo best_split_static;
    SplitInfo best_split_nonstatic;
    SplitInfo best_split_nowp;

    JXL_ASSERT(begin <= end);
    JXL_ASSERT(end <= indices->size());

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
        HybridUintConfig(4, 1, 2).Encode(
            PackSigned(residuals[pred][(*indices)[i]]), &tok, &bits, &nbits);
        tokens[pred].push_back(tok);
        extra_bits[pred].push_back(nbits);
        max_symbols = max_symbols > tok + 1 ? max_symbols : tok + 1;
      }
    }
    max_symbols = Padded(max_symbols);
    std::vector<int32_t> rounded_counts(max_symbols);
    std::vector<int32_t> counts(max_symbols * residuals.size());
    std::vector<int32_t> tot_extra_bits(residuals.size());
    for (size_t pred = 0; pred < tokens.size(); pred++) {
      for (size_t i = 0; i < tokens[pred].size(); i++) {
        counts[pred * max_symbols + tokens[pred][i]]++;
        tot_extra_bits[pred] += extra_bits[pred][i];
      }
    }

    float base_bits;
    {
      size_t pred = 0;
      for (size_t i = 0; i < predictors.size(); i++) {
        if (predictors[i] == (*tree)[pos].predictor) {
          pred = i;
        }
      }
      base_bits = EstimateBits(counts.data() + pred * max_symbols,
                               rounded_counts.data(), max_symbols) +
                  tot_extra_bits[pred];
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
      float extra_cost = 0;
      float Cost() const { return cost + extra_cost; }
      Predictor pred;  // will be uninitialized in some cases, but never used.
    };
    std::vector<CostInfo> costs_l;
    std::vector<CostInfo> costs_r;
    // The lower the threshold, the higher the expected noisiness of the
    // estimate. Thus, discourage changing predictors.
    float change_pred_penalty = 800.0f / (100.0f + threshold);
    for (size_t prop = 0; prop < props.size() && base_bits > threshold;
         prop++) {
      costs_l.clear();
      costs_r.clear();
      costs_l.resize(end - begin);
      costs_r.resize(end - begin);
      if (prop_value_used_count.size() < compact_properties[prop].size()) {
        prop_value_used_count.resize(compact_properties[prop].size());
        prop_count_increase.resize(compact_properties[prop].size() *
                                   max_symbols * residuals.size());
        extra_bits_increase.resize(compact_properties[prop].size() *
                                   residuals.size());
      }

      size_t first_used = compact_properties[prop].size();
      size_t last_used = 0;

      // TODO(veluca): consider finding multiple splits along a single property
      // at the same time, possibly with a bottom-up approach.
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

      std::vector<int32_t> counts_above(max_symbols), counts_below(max_symbols);

      // For all predictors, compute the right and left costs of each split.
      for (size_t pred = 0; pred < residuals.size(); pred++) {
        memcpy(counts_above.data(), counts.data() + pred * max_symbols,
               max_symbols * sizeof counts_above[0]);
        memset(counts_below.data(), 0, max_symbols * sizeof counts_below[0]);
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
          float rcost = EstimateBits(counts_above.data(), rounded_counts.data(),
                                     max_symbols) +
                        tot_extra_bits[pred] - extra_bits_below;
          float lcost = EstimateBits(counts_below.data(), rounded_counts.data(),
                                     max_symbols) +
                        extra_bits_below;
          float penalty = 0;
          if (predictors[pred] != (*tree)[pos].predictor) {
            penalty = change_pred_penalty;
          }
          if (rcost + penalty < costs_r[split - begin].Cost()) {
            costs_r[split - begin].cost = rcost;
            costs_r[split - begin].extra_cost = penalty;
            costs_r[split - begin].pred = predictors[pred];
          }
          if (lcost + penalty < costs_l[split - begin].Cost()) {
            costs_l[split - begin].cost = lcost;
            costs_l[split - begin].extra_cost = penalty;
            ;
            costs_l[split - begin].pred = predictors[pred];
          }
        }
      }
      // Iterate through the possible splits and find the one with minimum sum
      // of costs of the two sides.
      size_t split = begin;
      for (size_t i = first_used; i < last_used; i++) {
        if (!prop_value_used_count[i]) continue;
        split += prop_value_used_count[i];
        float rcost = costs_r[split - begin].cost;
        float lcost = costs_l[split - begin].cost;
        // WP was not used + we would use the WP property or predictor
        bool uses_wp = prop == wp_prop ||
                       costs_l[split - begin].pred == Predictor::Weighted ||
                       costs_r[split - begin].pred == Predictor::Weighted;
        bool used_wp = (used_properties & (1LU << wp_prop)) != 0 ||
                       (*tree)[pos].predictor == Predictor::Weighted;
        bool adds_wp = uses_wp && !used_wp;
        bool zero_entropy_side = rcost == 0 || lcost == 0;

        SplitInfo &best =
            prop < kNumStaticProperties
                ? (zero_entropy_side ? best_split_static_constant
                                     : best_split_static)
                : (adds_wp ? best_split_nonstatic : best_split_nowp);
        if (lcost + rcost < best.Cost()) {
          best.prop = prop;
          best.val = i;
          best.pos = split;
          best.lcost = lcost;
          best.lpred = costs_l[split - begin].pred;
          best.rcost = rcost;
          best.rpred = costs_r[split - begin].pred;
        }
      }
      // Clear prop_count_increase, extra_bits_increase and
      // prop_value_used_count arrays.
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

    SplitInfo *best = &best_split_nonstatic;
    // Try to avoid introducing WP.
    if (best_split_nowp.Cost() + threshold < base_bits &&
        best_split_nowp.Cost() <= fast_decode_multiplier * best->Cost()) {
      best = &best_split_nowp;
    }
    // Split along static props if possible and not significantly more
    // expensive.
    if (best_split_static.Cost() + threshold < base_bits &&
        best_split_static.Cost() <= fast_decode_multiplier * best->Cost()) {
      best = &best_split_static;
    }
    // Split along static props to create constant nodes if possible.
    if (best_split_static_constant.Cost() + threshold < base_bits) {
      best = &best_split_static_constant;
    }
    SplitInfo forced_split;
    // The multiplier ranges cut halfway through the current ranges of static
    // properties. We do this even if the current node is not a leaf, to
    // minimize the number of nodes in the resulting tree.
    for (size_t i = 0; i < mul_info.size(); i++) {
      uint32_t axis, val;
      IntersectionType t =
          BoxIntersects(static_prop_range, mul_info[i].range, axis, val);
      if (t == IntersectionType::kNone) continue;
      if (t == IntersectionType::kInside) {
        (*tree)[pos].multiplier = mul_info[i].multiplier;
        break;
      }
      if (t == IntersectionType::kPartial) {
        forced_split.val = val;
        forced_split.prop = axis;
        forced_split.lcost = forced_split.rcost = base_bits / 2 - threshold;
        best = &forced_split;
        best->pos = begin;
        JXL_ASSERT(best->prop == props_to_use[best->prop]);
        for (size_t x = begin; x < end; x++) {
          if (props[best->prop][(*indices)[x]] <= best->val) {
            best->pos++;
          }
        }
        break;
      }
    }

    if (best->Cost() + threshold < base_bits) {
      // Split node and try to split children.
      MakeSplitNode(pos, props_to_use[best->prop],
                    best->val < compact_properties[best->prop].size()
                        ? compact_properties[best->prop][best->val]
                        : best->val,
                    best->lpred, 0, best->rpred, 0, tree);
      // "Sort" according to winning property
      std::nth_element(indices->begin() + begin, indices->begin() + best->pos,
                       indices->begin() + end, [&](size_t a, size_t b) {
                         return props[best->prop][a] < props[best->prop][b];
                       });
      uint32_t p = props_to_use[best->prop];
      if (p >= kNumStaticProperties) {
        used_properties |= 1 << best->prop;
      }
      auto new_sp_range = static_prop_range;
      if (p < kNumStaticProperties) {
        JXL_ASSERT(best->val + 1 <= new_sp_range[p][1]);
        new_sp_range[p][1] = best->val + 1;
        JXL_ASSERT(new_sp_range[p][0] < new_sp_range[p][1]);
      }
      nodes.push_back(NodeInfo{(*tree)[pos].rchild, begin, best->pos,
                               used_properties, new_sp_range});
      new_sp_range = static_prop_range;
      if (p < kNumStaticProperties) {
        JXL_ASSERT(new_sp_range[p][0] <= best->val + 1);
        new_sp_range[p][0] = best->val + 1;
        JXL_ASSERT(new_sp_range[p][0] < new_sp_range[p][1]);
      }
      nodes.push_back(NodeInfo{(*tree)[pos].lchild, best->pos, end,
                               used_properties, new_sp_range});
    } else if ((*tree)[pos].multiplier == 1) {
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
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(EstimateEntropy);  // Local function.
HWY_EXPORT(FindBestSplit);    // Local function.

void ChooseAndQuantizeProperties(
    size_t max_properties, size_t max_property_values,
    const std::vector<std::vector<int>> &residuals, bool force_wp_only,
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
    if (force_wp_only && i >= kNumStaticProperties &&
        i != kNumNonrefProperties - weighted::kNumProperties) {
      continue;
    }
    PropertyVal min = std::numeric_limits<PropertyVal>::max();
    PropertyVal max = std::numeric_limits<PropertyVal>::min();
    for (PropertyVal x : (*props)[i]) {
      min = std::min(min, x);
      max = std::max(max, x);
    }
    if (i < kNumStaticProperties) {
      (*compact_properties)[i].resize(max + 1);
      std::iota((*compact_properties)[i].begin(),
                (*compact_properties)[i].end(), 0);
      continue;
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

  if (force_wp_only) {
    props_to_use->resize(kNumStaticProperties);
    std::iota(props_to_use->begin(), props_to_use->end(), 0);
    props_to_use->back() = kNumNonrefProperties - weighted::kNumProperties;
  } else if (max_properties + kNumStaticProperties >= props->size()) {
    props_to_use->resize(props->size());
    std::iota(props_to_use->begin(), props_to_use->end(), 0);
  } else {
    std::vector<std::pair<float, size_t>> props_with_entropy;
    HWY_DYNAMIC_DISPATCH(EstimateEntropy)
    (0, residuals, *props, *compact_properties, &props_with_entropy);
    std::sort(props_with_entropy.begin(), props_with_entropy.end());

    // Limit the search to the properties with the smallest resulting entropy
    // (including static properties).
    max_properties = std::min(max_properties + kNumStaticProperties,
                              props_with_entropy.size());
    props_to_use->resize(max_properties);
    for (size_t i = 0; i < max_properties; i++) {
      (*props_to_use)[i] = props_with_entropy[i].second;
    }
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
      if (i < kNumStaticProperties) continue;
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
                     const std::vector<std::vector<int>> compact_properties,
                     const std::vector<size_t> &props_to_use, float threshold,
                     size_t max_properties,
                     const std::vector<ModularMultiplierInfo> &mul_info,
                     StaticPropRange static_prop_range,
                     float fast_decode_multiplier, Tree *tree) {
  // TODO(veluca): take into account that different contexts can have different
  // uint configs.
  //
  // Initialize tree.
  tree->emplace_back();
  tree->back().property = -1;
  tree->back().predictor = predictors[0];
  tree->back().predictor_offset = 0;
  tree->back().multiplier = 1;
  JXL_ASSERT(props.size() < 64);

  std::vector<size_t> indices(residuals[0].size());
  std::iota(indices.begin(), indices.end(), 0);
  HWY_DYNAMIC_DISPATCH(FindBestSplit)
  (residuals, props, predictors, compact_properties, &indices, props_to_use,
   threshold, mul_info, static_prop_range, fast_decode_multiplier, tree);
}

namespace {
constexpr size_t kSplitValContext = 0;
constexpr size_t kPropertyContext = 1;
constexpr size_t kPredictorContext = 2;
constexpr size_t kOffsetContext = 3;
constexpr size_t kMultiplierLogContext = 4;
constexpr size_t kMultiplierBitsContext = 5;
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
      uint32_t mul_log = Num0BitsBelowLS1Bit_Nonzero(tree[cur].multiplier);
      uint32_t mul_bits = (tree[cur].multiplier >> mul_log) - 1;
      tokens->emplace_back(kMultiplierLogContext, mul_log);
      tokens->emplace_back(kMultiplierBitsContext, mul_bits);
      JXL_ASSERT(tree[cur].predictor < Predictor::Best);
      decoder_tree->emplace_back(-1, 0, leaf_id++, 0, tree[cur].predictor,
                                 tree[cur].predictor_offset,
                                 tree[cur].multiplier);
      continue;
    }
    decoder_tree->emplace_back(tree[cur].property, tree[cur].splitval,
                               decoder_tree->size() + q.size() + 1,
                               decoder_tree->size() + q.size() + 2,
                               Predictor::Zero, 0, 1);
    q.push(tree[cur].lchild);
    q.push(tree[cur].rchild);
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
  JXL_RETURN_IF_ERROR(ValidateTree(tree, new_bounds, tree[root].lchild));
  new_bounds[p] = prop_bounds[p];
  new_bounds[p].second = val;
  return ValidateTree(tree, new_bounds, tree[root].rchild);
}

Status DecodeTree(BitReader *br, ANSSymbolReader *reader,
                  const std::vector<uint8_t> &context_map, Tree *tree) {
  size_t leaf_id = 0;
  size_t to_decode = 1;
  tree->clear();
  while (to_decode > 0) {
    JXL_RETURN_IF_ERROR(br->AllReadsWithinBounds());
    if (tree->size() > kMaxTreeSize) {
      return JXL_FAILURE("Tree is too large");
    }
    to_decode--;
    int property =
        reader->ReadHybridUint(kPropertyContext, br, context_map) - 1;
    if (property < -1 || property >= 256) {
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
      uint32_t mul_log =
          reader->ReadHybridUint(kMultiplierLogContext, br, context_map);
      if (mul_log >= 31) {
        return JXL_FAILURE("Invalid multiplier logarithm");
      }
      uint32_t mul_bits =
          reader->ReadHybridUint(kMultiplierBitsContext, br, context_map);
      if (mul_bits + 1 >= 1 << (31 - mul_log)) {
        return JXL_FAILURE("Invalid multiplier");
      }
      uint32_t multiplier = (mul_bits + 1U) << mul_log;
      tree->emplace_back(-1, 0, leaf_id++, 0, static_cast<Predictor>(predictor),
                         predictor_offset, multiplier);
      continue;
    }
    int splitval =
        UnpackSigned(reader->ReadHybridUint(kSplitValContext, br, context_map));
    tree->emplace_back(property, splitval, tree->size() + to_decode + 1,
                       tree->size() + to_decode + 2, Predictor::Zero, 0, 1);
    to_decode += 2;
  }
  std::vector<std::pair<pixel_type, pixel_type>> prop_bounds;
  prop_bounds.resize(256, {std::numeric_limits<pixel_type>::min(),
                           std::numeric_limits<pixel_type>::max()});
  return ValidateTree(*tree, prop_bounds, 0);
}

}  // namespace jxl
#endif  // HWY_ONCE
