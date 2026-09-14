// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/encoding/enc_ma.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <queue>
#include <vector>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/encoding/ma_common.h"
#include "lib/jxl/modular/modular_image.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/modular/encoding/enc_ma.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/fast_math-inl.h"
#include "lib/jxl/base/random.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/pack_signed.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Eq;
using hwy::HWY_NAMESPACE::IfThenElse;
using hwy::HWY_NAMESPACE::Lt;
using hwy::HWY_NAMESPACE::Max;

const HWY_FULL(float) df;
const HWY_FULL(int32_t) di;
// Pad array size for use in EstimateBits.
size_t Padded(size_t x) {
  JXL_DASSERT(Lanes(df) == Lanes(di));
  return RoundUpTo(x, Lanes(df));
}

// Compute entropy of the histogram, taking into account the minimum probability
// for symbols with non-zero counts.
float EstimateBits(const int32_t *counts, size_t num_symbols) {
  JXL_DASSERT(num_symbols == Padded(num_symbols));
  auto total_v = Zero(di);
  for (size_t i = 0; i < num_symbols; i += Lanes(di)) {
    const auto counts_iv = LoadU(di, &counts[i]);
    total_v = Add(total_v, counts_iv);
  }
  total_v = SumOfLanes(di, total_v);

  const auto minprob = Set(df, 1.0f / ANS_TAB_SIZE);
  const auto inv_total = Set(df, 1.0f / GetLane(total_v));
  auto bits_lanes = Zero(df);
  for (size_t i = 0; i < num_symbols; i += Lanes(df)) {
    const auto counts_iv = LoadU(di, &counts[i]);
    const auto counts_fv = ConvertTo(df, counts_iv);
    const auto probs = Mul(counts_fv, inv_total);
    const auto mprobs = Max(probs, minprob);
    const auto nbps = IfThenZeroElse(Eq(counts_iv, total_v),
                                     BitCast(di, FastLog2f(df, mprobs)));
    bits_lanes = Sub(bits_lanes, Mul(counts_fv, BitCast(df, nbps)));
  }
  return GetLane(SumOfLanes(df, bits_lanes));
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

template<bool S>
void SplitTreeSamples(TreeSamples &tree_samples, size_t begin, size_t pos,
                      size_t end, size_t prop, uint32_t val) {
  size_t begin_pos = begin;
  size_t end_pos = pos;
  do {
    while (begin_pos < pos &&
           tree_samples.Property<S>(prop, begin_pos) <= val) {
      ++begin_pos;
    }
    while (end_pos < end && tree_samples.Property<S>(prop, end_pos) > val) {
      ++end_pos;
    }
    if (begin_pos < pos && end_pos < end) {
      tree_samples.Swap(begin_pos, end_pos);
    }
    ++begin_pos;
    ++end_pos;
  } while (begin_pos < pos && end_pos < end);
}

template <bool S>
void CollectExtraBitsIncrease(TreeSamples &tree_samples,
                              const std::vector<ResidualToken> &rtokens,
                              std::vector<int> &count_increase,
                              std::vector<size_t> &extra_bits_increase,
                              size_t begin, size_t end, size_t prop_idx,
                              size_t max_symbols) {
  for (size_t i2 = begin; i2 < end; i2++) {
    const ResidualToken &rt = rtokens[i2];
    size_t cnt = tree_samples.Count(i2);
    size_t p = tree_samples.Property<S>(prop_idx, i2);
    size_t sym = rt.tok;
    size_t ebi = rt.nbits * cnt;
    count_increase[p * max_symbols + sym] += cnt;
    extra_bits_increase[p] += ebi;
  }
}

void FindBestSplit(TreeSamples &tree_samples, float threshold,
                   const std::vector<ModularMultiplierInfo> &mul_info,
                   StaticPropRange initial_static_prop_range,
                   float fast_decode_multiplier, Tree *tree) {
  struct NodeInfo {
    size_t pos;
    size_t begin;
    size_t end;
    StaticPropRange static_prop_range;
  };
  std::vector<NodeInfo> nodes;
  nodes.push_back(NodeInfo{0, 0, tree_samples.NumDistinctSamples(),
                           initial_static_prop_range});

  size_t num_predictors = tree_samples.NumPredictors();
  size_t num_properties = tree_samples.NumProperties();

  // TODO(veluca): consider parallelizing the search (processing multiple nodes
  // at a time).
  while (!nodes.empty()) {
    size_t pos = nodes.back().pos;
    size_t begin = nodes.back().begin;
    size_t end = nodes.back().end;

    StaticPropRange static_prop_range = nodes.back().static_prop_range;
    nodes.pop_back();
    if (begin == end) continue;

    struct SplitInfo {
      size_t prop = 0;
      uint32_t val = 0;
      size_t pos = 0;
      float lcost = std::numeric_limits<float>::max();
      float rcost = std::numeric_limits<float>::max();
      Predictor lpred = Predictor::Zero;
      Predictor rpred = Predictor::Zero;
      float Cost() const { return lcost + rcost; }
    };

    SplitInfo best_split_static_constant;
    SplitInfo best_split_static;
    SplitInfo best_split_nonstatic;
    SplitInfo best_split_nowp;

    JXL_DASSERT(begin <= end);
    JXL_DASSERT(end <= tree_samples.NumDistinctSamples());

    // Compute the maximum token in the range.
    size_t max_symbols = 0;
    for (size_t pred = 0; pred < num_predictors; pred++) {
      for (size_t i = begin; i < end; i++) {
        uint32_t tok = tree_samples.Token(pred, i);
        max_symbols = max_symbols > tok + 1 ? max_symbols : tok + 1;
      }
    }
    max_symbols = Padded(max_symbols);
    std::vector<int32_t> counts(max_symbols * num_predictors);
    std::vector<uint32_t> tot_extra_bits(num_predictors);
    for (size_t pred = 0; pred < num_predictors; pred++) {
      size_t extra_bits = 0;
      const std::vector<ResidualToken>& rtokens = tree_samples.RTokens(pred);
      for (size_t i = begin; i < end; i++) {
        const ResidualToken& rt = rtokens[i];
        size_t count = tree_samples.Count(i);
        size_t eb = rt.nbits * count;
        counts[pred * max_symbols + rt.tok] += count;
        extra_bits += eb;
      }
      tot_extra_bits[pred] = extra_bits;
    }

    float base_bits;
    {
      size_t pred = tree_samples.PredictorIndex((*tree)[pos].predictor);
      base_bits =
          EstimateBits(counts.data() + pred * max_symbols, max_symbols) +
          tot_extra_bits[pred];
    }

    SplitInfo *best = &best_split_nonstatic;

    SplitInfo forced_split;
    // The multiplier ranges cut halfway through the current ranges of static
    // properties. We do this even if the current node is not a leaf, to
    // minimize the number of nodes in the resulting tree.
    for (const auto &mmi : mul_info) {
      uint32_t axis;
      uint32_t val;
      IntersectionType t =
          BoxIntersects(static_prop_range, mmi.range, axis, val);
      if (t == IntersectionType::kNone) continue;
      if (t == IntersectionType::kInside) {
        (*tree)[pos].multiplier = mmi.multiplier;
        break;
      }
      if (t == IntersectionType::kPartial) {
        JXL_DASSERT(axis < kNumStaticProperties);
        forced_split.val = tree_samples.QuantizeStaticProperty(axis, val);
        forced_split.prop = axis;
        forced_split.lcost = forced_split.rcost = base_bits / 2 - threshold;
        forced_split.lpred = forced_split.rpred = (*tree)[pos].predictor;
        best = &forced_split;
        best->pos = begin;
        JXL_DASSERT(best->prop == tree_samples.PropertyFromIndex(best->prop));
        if (best->prop < tree_samples.NumStaticProps()) {
        for (size_t x = begin; x < end; x++) {
          if (tree_samples.Property<true>(best->prop, x) <= best->val) {
            best->pos++;
          }
        }
      } else {
        size_t prop = best->prop - tree_samples.NumStaticProps();
        for (size_t x = begin; x < end; x++) {
          if (tree_samples.Property<false>(prop, x) <= best->val) {
            best->pos++;
          }
        }
      }
        break;
      }
    }

    if (best != &forced_split) {
      std::vector<int> prop_value_used_count;
      std::vector<int> count_increase;
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

      std::vector<int32_t> counts_above(max_symbols);
      std::vector<int32_t> counts_below(max_symbols);

      // The lower the threshold, the higher the expected noisiness of the
      // estimate. Thus, discourage changing predictors.
      float change_pred_penalty = 800.0f / (100.0f + threshold);
      for (size_t prop = 0; prop < num_properties && base_bits > threshold;
           prop++) {
        costs_l.clear();
        costs_r.clear();
        size_t prop_size = tree_samples.NumPropertyValues(prop);
        if (extra_bits_increase.size() < prop_size) {
          count_increase.resize(prop_size * max_symbols);
          extra_bits_increase.resize(prop_size);
        }
        // Clear prop_value_used_count (which cannot be cleared "on the go")
        prop_value_used_count.clear();
        prop_value_used_count.resize(prop_size);

        size_t first_used = prop_size;
        size_t last_used = 0;

        // TODO(veluca): consider finding multiple splits along a single
        // property at the same time, possibly with a bottom-up approach.
        if (prop < tree_samples.NumStaticProps()) {
          for (size_t i = begin; i < end; i++) {
            size_t p = tree_samples.Property<true>(prop, i);
            prop_value_used_count[p]++;
            last_used = std::max(last_used, p);
            first_used = std::min(first_used, p);
          }
        } else {
          size_t prop_idx = prop - tree_samples.NumStaticProps();
          for (size_t i = begin; i < end; i++) {
            size_t p = tree_samples.Property<false>(prop_idx, i);
            prop_value_used_count[p]++;
            last_used = std::max(last_used, p);
            first_used = std::min(first_used, p);
          }
        }
        costs_l.resize(last_used - first_used);
        costs_r.resize(last_used - first_used);
        // For all predictors, compute the right and left costs of each split.
        for (size_t pred = 0; pred < num_predictors; pred++) {
          // Compute cost and histogram increments for each property value.
          const std::vector<ResidualToken> &rtokens =
              tree_samples.RTokens(pred);
          if (prop < tree_samples.NumStaticProps()) {
            CollectExtraBitsIncrease<true>(tree_samples, rtokens,
                                           count_increase, extra_bits_increase,
                                           begin, end, prop, max_symbols);
          } else {
            CollectExtraBitsIncrease<false>(
                tree_samples, rtokens, count_increase, extra_bits_increase,
                begin, end, prop - tree_samples.NumStaticProps(), max_symbols);
          }
          memcpy(counts_above.data(), counts.data() + pred * max_symbols,
                 max_symbols * sizeof counts_above[0]);
          memset(counts_below.data(), 0, max_symbols * sizeof counts_below[0]);
          size_t extra_bits_below = 0;
          // Exclude last used: this ensures neither counts_above nor
          // counts_below is empty.
          for (size_t i = first_used; i < last_used; i++) {
            if (!prop_value_used_count[i]) continue;
            extra_bits_below += extra_bits_increase[i];
            // The increase for this property value has been used, and will not
            // be used again: clear it. Also below.
            extra_bits_increase[i] = 0;
            for (size_t sym = 0; sym < max_symbols; sym++) {
              counts_above[sym] -= count_increase[i * max_symbols + sym];
              counts_below[sym] += count_increase[i * max_symbols + sym];
              count_increase[i * max_symbols + sym] = 0;
            }
            float rcost = EstimateBits(counts_above.data(), max_symbols) +
                          tot_extra_bits[pred] - extra_bits_below;
            float lcost = EstimateBits(counts_below.data(), max_symbols) +
                          extra_bits_below;
            JXL_DASSERT(extra_bits_below <= tot_extra_bits[pred]);
            float penalty = 0;
            // Never discourage moving away from the Weighted predictor.
            if (tree_samples.PredictorFromIndex(pred) !=
                    (*tree)[pos].predictor &&
                (*tree)[pos].predictor != Predictor::Weighted) {
              penalty = change_pred_penalty;
            }
            // If everything else is equal, disfavour Weighted (slower) and
            // favour Zero (faster if it's the only predictor used in a
            // group+channel combination)
            if (tree_samples.PredictorFromIndex(pred) == Predictor::Weighted) {
              penalty += 1e-8;
            }
            if (tree_samples.PredictorFromIndex(pred) == Predictor::Zero) {
              penalty -= 1e-8;
            }
            if (rcost + penalty < costs_r[i - first_used].Cost()) {
              costs_r[i - first_used].cost = rcost;
              costs_r[i - first_used].extra_cost = penalty;
              costs_r[i - first_used].pred =
                  tree_samples.PredictorFromIndex(pred);
            }
            if (lcost + penalty < costs_l[i - first_used].Cost()) {
              costs_l[i - first_used].cost = lcost;
              costs_l[i - first_used].extra_cost = penalty;
              costs_l[i - first_used].pred =
                  tree_samples.PredictorFromIndex(pred);
            }
          }
        }
        // Iterate through the possible splits and find the one with minimum sum
        // of costs of the two sides.
        size_t split = begin;
        for (size_t i = first_used; i < last_used; i++) {
          if (!prop_value_used_count[i]) continue;
          split += prop_value_used_count[i];
          float rcost = costs_r[i - first_used].cost;
          float lcost = costs_l[i - first_used].cost;

          bool uses_wp = tree_samples.PropertyFromIndex(prop) == kWPProp ||
                         costs_l[i - first_used].pred == Predictor::Weighted ||
                         costs_r[i - first_used].pred == Predictor::Weighted;
          bool zero_entropy_side = rcost == 0 || lcost == 0;

          SplitInfo &best_ref =
              tree_samples.PropertyFromIndex(prop) < kNumStaticProperties
                  ? (zero_entropy_side ? best_split_static_constant
                                       : best_split_static)
                  : (uses_wp ? best_split_nonstatic : best_split_nowp);
          if (lcost + rcost < best_ref.Cost()) {
            best_ref.prop = prop;
            best_ref.val = i;
            best_ref.pos = split;
            best_ref.lcost = lcost;
            best_ref.lpred = costs_l[i - first_used].pred;
            best_ref.rcost = rcost;
            best_ref.rpred = costs_r[i - first_used].pred;
          }
        }
        // Clear extra_bits_increase and cost_increase for last_used.
        extra_bits_increase[last_used] = 0;
        for (size_t sym = 0; sym < max_symbols; sym++) {
          count_increase[last_used * max_symbols + sym] = 0;
        }
      }

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
    }

    if (best->Cost() + threshold < base_bits) {
      uint32_t p = tree_samples.PropertyFromIndex(best->prop);
      pixel_type dequant =
          tree_samples.UnquantizeProperty(best->prop, best->val);
      // Split node and try to split children.
      MakeSplitNode(pos, p, dequant, best->lpred, 0, best->rpred, 0, tree);
      // "Sort" according to winning property
      if (best->prop < tree_samples.NumStaticProps()) {
        SplitTreeSamples<true>(tree_samples, begin, best->pos, end, best->prop,
                               best->val);
      } else {
        SplitTreeSamples<false>(tree_samples, begin, best->pos, end,
                                best->prop - tree_samples.NumStaticProps(),
                                best->val);
      }
      auto new_sp_range = static_prop_range;
      if (p < kNumStaticProperties) {
        JXL_DASSERT(static_cast<uint32_t>(dequant + 1) <= new_sp_range[p][1]);
        new_sp_range[p][1] = dequant + 1;
        JXL_DASSERT(new_sp_range[p][0] < new_sp_range[p][1]);
      }
      nodes.push_back(
          NodeInfo{(*tree)[pos].rchild, begin, best->pos, new_sp_range});
      new_sp_range = static_prop_range;
      if (p < kNumStaticProperties) {
        JXL_DASSERT(new_sp_range[p][0] <= static_cast<uint32_t>(dequant + 1));
        new_sp_range[p][0] = dequant + 1;
        JXL_DASSERT(new_sp_range[p][0] < new_sp_range[p][1]);
      }
      nodes.push_back(
          NodeInfo{(*tree)[pos].lchild, best->pos, end, new_sp_range});
    }
  }
}

struct DP1DResult {
  size_t prop_dim = 0;
  float cost = std::numeric_limits<float>::max();
  std::vector<int32_t> cutoffs;
  std::vector<size_t> segment_preds;
};

DP1DResult Run1DDPOnSamples(TreeSamples& tree_samples, size_t prop_dim,
                           const std::vector<uint32_t>* sample_subset,
                           float nb_repeats) {
  DP1DResult res;
  res.prop_dim = prop_dim;
  size_t total_samples = sample_subset ? sample_subset->size()
                                       : tree_samples.NumDistinctSamples();
  if (total_samples == 0) return res;

  const size_t num_predictors = tree_samples.NumPredictors();
  const size_t prop_idx = prop_dim + tree_samples.NumStaticProps();

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  int32_t min_val = std::numeric_limits<int32_t>::max();
  int32_t max_val = std::numeric_limits<int32_t>::min();
  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    int32_t p = tree_samples.Property<false>(prop_dim, s);
    min_val = std::min(min_val, p);
    max_val = std::max(max_val, p);
  }

  std::vector<size_t> max_symbols(num_predictors, 0);
  for (size_t pred = 0; pred < num_predictors; pred++) {
    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      uint32_t tok = tree_samples.Token(pred, s);
      max_symbols[pred] = std::max(max_symbols[pred], static_cast<size_t>(tok + 1));
    }
    max_symbols[pred] = Padded(max_symbols[pred]);
  }

  // If all samples have the same property value: single leaf.
  if (min_val >= max_val) {
    std::vector<int32_t> hist(max_symbols[0], 0);
    float best_leaf_cost = std::numeric_limits<float>::max();
    size_t best_leaf_pred = 0;
    for (size_t pred = 0; pred < num_predictors; pred++) {
      hist.assign(max_symbols[pred], 0);
      int64_t extra = 0;
      for (size_t i = 0; i < total_samples; i++) {
        size_t s = get_sample_idx(i);
        size_t cnt = tree_samples.Count(s);
        hist[tree_samples.Token(pred, s)] += cnt;
        extra += tree_samples.RTokens(pred)[s].nbits * cnt;
      }
      float bits = EstimateBits(hist.data(), max_symbols[pred]) + extra;
      if (bits < best_leaf_cost) {
        best_leaf_cost = bits;
        best_leaf_pred = pred;
      }
    }
    res.cost = best_leaf_cost;
    res.segment_preds.push_back(best_leaf_pred);
    return res;
  }

  int32_t R = max_val - min_val + 1;
  std::vector<std::vector<std::vector<int32_t>>> freq(
      num_predictors, std::vector<std::vector<int32_t>>(R));
  for (size_t pred = 0; pred < num_predictors; pred++) {
    for (int32_t v = 0; v < R; v++) {
      freq[pred][v].resize(max_symbols[pred], 0);
    }
  }

  std::vector<std::vector<int64_t>> pref_extra_bits(
      num_predictors, std::vector<int64_t>(R, 0));
  std::vector<int64_t> sample_counts(R, 0);
  std::vector<uint8_t> exist(R, 0);

  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    int32_t val = tree_samples.Property<false>(prop_dim, s) - min_val;
    size_t cnt = tree_samples.Count(s);
    sample_counts[val] += cnt;
    exist[val] = 1;
    for (size_t pred = 0; pred < num_predictors; pred++) {
      uint32_t tok = tree_samples.Token(pred, s);
      freq[pred][val][tok] += cnt;
      pref_extra_bits[pred][val] += tree_samples.RTokens(pred)[s].nbits * cnt;
    }
  }

  for (size_t pred = 0; pred < num_predictors; pred++) {
    for (int32_t v = 1; v < R; v++) {
      pref_extra_bits[pred][v] += pref_extra_bits[pred][v - 1];
    }
  }

  std::vector<int64_t> pref_counts(R);
  pref_counts[0] = sample_counts[0];
  for (int32_t v = 1; v < R; v++) {
    pref_counts[v] = pref_counts[v - 1] + sample_counts[v];
  }

  std::vector<int32_t> prev_exist(R, -1);
  int32_t last_exist = -1;
  for (int32_t v = 0; v < R; v++) {
    if (exist[v]) {
      prev_exist[v] = last_exist;
      last_exist = v;
    } else {
      prev_exist[v] = last_exist;
    }
  }

  auto split_penalty = [&](int32_t cutoff_idx) -> float {
    int32_t unquant = tree_samples.UnquantizeProperty(prop_idx, cutoff_idx + min_val);
    return (110.0f + 3.0f * FastLog2f(std::abs(unquant) + 1.0f)) * (nb_repeats / 0.3f);
  };

  std::vector<float> dp(R, std::numeric_limits<float>::max());
  std::vector<int32_t> opt_split(R, -1);
  std::vector<size_t> pred_split(R, 0);

  std::vector<std::vector<int32_t>> residual_hist(num_predictors);
  for (size_t pred = 0; pred < num_predictors; pred++) {
    residual_hist[pred].resize(max_symbols[pred], 0);
  }

  for (int32_t i = 0; i < R; i++) {
    if (!exist[i]) continue;
    for (size_t pred = 0; pred < num_predictors; pred++) {
      std::fill(residual_hist[pred].begin(), residual_hist[pred].end(), 0);
    }

    for (int32_t j = i; j >= 0; j--) {
      for (size_t pred = 0; pred < num_predictors; pred++) {
        const auto& f = freq[pred][j];
        for (size_t k = 0; k < max_symbols[pred]; k++) {
          residual_hist[pred][k] += f[k];
        }
      }

      int64_t cur_cnt = pref_counts[i] - (j > 0 ? pref_counts[j - 1] : 0);
      if (cur_cnt == 0) continue;

      float best_int_cost = std::numeric_limits<float>::max();
      size_t best_int_pred = 0;
      for (size_t pred = 0; pred < num_predictors; pred++) {
        float bits = EstimateBits(residual_hist[pred].data(), max_symbols[pred]);
        int64_t extra = pref_extra_bits[pred][i] - (j > 0 ? pref_extra_bits[pred][j - 1] : 0);
        float total = bits + extra;
        if (total < best_int_cost) {
          best_int_cost = total;
          best_int_pred = pred;
        }
      }

      if (j == 0) {
        if (best_int_cost < dp[i]) {
          dp[i] = best_int_cost;
          opt_split[i] = -1;
          pred_split[i] = best_int_pred;
        }
      } else {
        int32_t prev_cut = prev_exist[j - 1];
        if (prev_cut != -1 && dp[prev_cut] != std::numeric_limits<float>::max()) {
          float cand = dp[prev_cut] + best_int_cost + split_penalty(prev_cut);
          if (cand < dp[i]) {
            dp[i] = cand;
            opt_split[i] = prev_cut;
            pred_split[i] = best_int_pred;
          }
        }
      }
    }
  }

  if (last_exist == -1 || dp[last_exist] == std::numeric_limits<float>::max()) {
    return res;
  }

  res.cost = dp[last_exist];
  int32_t curr = last_exist;
  while (curr != -1) {
    res.segment_preds.push_back(pred_split[curr]);
    int32_t prev = opt_split[curr];
    if (prev != -1) {
      res.cutoffs.push_back(prev + min_val);
    }
    curr = prev;
  }
  std::reverse(res.cutoffs.begin(), res.cutoffs.end());
  std::reverse(res.segment_preds.begin(), res.segment_preds.end());
  return res;
}

void BuildTreeFrom1D(size_t prop_idx,
                     const std::vector<int32_t>& cutoffs,
                     const std::vector<size_t>& segment_preds,
                     TreeSamples& tree_samples, Tree* tree,
                     size_t root_pos = 0,
                     std::vector<size_t>* leaf_positions = nullptr) {
  if (cutoffs.empty()) {
    (*tree)[root_pos] = PropertyDecisionNode::Leaf(
        tree_samples.PredictorFromIndex(segment_preds.empty() ? 0 : segment_preds[0]));
    if (leaf_positions && !leaf_positions->empty()) {
      (*leaf_positions)[0] = root_pos;
    }
    return;
  }
  int32_t property = tree_samples.PropertyFromIndex(prop_idx);
  struct NodeInfo {
    size_t begin, end, pos;
  };
  std::queue<NodeInfo> q;
  q.push(NodeInfo{0, cutoffs.size(), root_pos});

  while (!q.empty()) {
    NodeInfo info = q.front();
    q.pop();
    if (info.begin == info.end) {
      if (leaf_positions && info.begin < leaf_positions->size()) {
        (*leaf_positions)[info.begin] = info.pos;
      }
      continue;
    }
    uint32_t split = (info.begin + info.end) / 2;
    int32_t cutoff = tree_samples.UnquantizeProperty(prop_idx, cutoffs[split]);
    uint32_t lchild = tree->size();
    uint32_t rchild = tree->size() + 1;
    (*tree)[info.pos] = PropertyDecisionNode::Split(property, cutoff, lchild, rchild);

    // Left child: strictly greater than cutoff (> cutoff) -> segments [split + 1, end]
    tree->push_back(PropertyDecisionNode::Leaf(
        tree_samples.PredictorFromIndex(segment_preds[split + 1])));
    q.push(NodeInfo{split + 1, info.end, lchild});

    // Right child: less than or equal to cutoff (<= cutoff) -> segments [begin, split]
    tree->push_back(PropertyDecisionNode::Leaf(
        tree_samples.PredictorFromIndex(segment_preds[split])));
    q.push(NodeInfo{info.begin, split, rchild});
  }
}

void FindBestTree1dDP(TreeSamples& tree_samples, float nb_repeats, Tree* tree) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  if (num_props == 0 || tree_samples.NumDistinctSamples() == 0) return;

  DP1DResult best_res;
  for (size_t dim = 0; dim < num_props; dim++) {
    DP1DResult r = Run1DDPOnSamples(tree_samples, dim, nullptr, nb_repeats);
    if (r.cost < best_res.cost) {
      best_res = std::move(r);
    }
  }

  if (best_res.segment_preds.empty()) {
    (*tree)[0] = PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0));
    return;
  }

  BuildTreeFrom1D(best_res.prop_dim + tree_samples.NumStaticProps(),
                  best_res.cutoffs, best_res.segment_preds, tree_samples, tree, 0);
}

void FindBestTree2PropDP(TreeSamples& tree_samples, float nb_repeats, Tree* tree) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  if (num_props == 0 || tree_samples.NumDistinctSamples() == 0) return;
  if (num_props == 1) {
    FindBestTree1dDP(tree_samples, nb_repeats, tree);
    return;
  }

  std::vector<DP1DResult> primary_dps(num_props);
  for (size_t dim = 0; dim < num_props; dim++) {
    primary_dps[dim] = Run1DDPOnSamples(tree_samples, dim, nullptr, nb_repeats);
  }

  const size_t total_samples = tree_samples.NumDistinctSamples();

  struct BestPairRefinement {
    size_t pA = 0;
    size_t pB = 1;
    float cost = std::numeric_limits<float>::max();
    DP1DResult dpA;
    std::vector<DP1DResult> refinementsB;
  };

  BestPairRefinement best_pair;

  std::vector<size_t> ranked_props(num_props);
  for (size_t i = 0; i < num_props; i++) ranked_props[i] = i;
  std::sort(ranked_props.begin(), ranked_props.end(), [&](size_t a, size_t b) {
    return primary_dps[a].cost < primary_dps[b].cost;
  });
  size_t K = std::min<size_t>(num_props, 4);

  for (size_t rA = 0; rA < K; rA++) {
    size_t pA = ranked_props[rA];
    const auto& dpA = primary_dps[pA];
    if (dpA.segment_preds.empty()) continue;

    const size_t num_segs = dpA.cutoffs.size() + 1;
    std::vector<std::vector<uint32_t>> seg_samples(num_segs);

    for (size_t i = 0; i < total_samples; i++) {
      int32_t v = tree_samples.Property<false>(pA, i);
      size_t seg = std::upper_bound(dpA.cutoffs.begin(), dpA.cutoffs.end(), v) -
                   dpA.cutoffs.begin();
      seg_samples[seg].push_back(i);
    }

    std::vector<float> seg_base_cost(num_segs, 0.0f);
    for (size_t s = 0; s < num_segs; s++) {
      if (seg_samples[s].empty()) continue;
      size_t pred = dpA.segment_preds[s];
      size_t max_sym = 0;
      for (uint32_t idx : seg_samples[s]) {
        max_sym = std::max(max_sym, static_cast<size_t>(tree_samples.Token(pred, idx) + 1));
      }
      max_sym = Padded(max_sym);
      std::vector<int32_t> hist(max_sym, 0);
      int64_t extra = 0;
      for (uint32_t idx : seg_samples[s]) {
        size_t cnt = tree_samples.Count(idx);
        hist[tree_samples.Token(pred, idx)] += cnt;
        extra += tree_samples.RTokens(pred)[idx].nbits * cnt;
      }
      seg_base_cost[s] = EstimateBits(hist.data(), max_sym) + extra;
    }

    float cutoffsA_penalty = 0.0f;
    for (int32_t c : dpA.cutoffs) {
      int32_t unquant = tree_samples.UnquantizeProperty(
          pA + tree_samples.NumStaticProps(), c);
      cutoffsA_penalty += (110.0f + 3.0f * FastLog2f(std::abs(unquant) + 1.0f)) * (nb_repeats / 0.3f);
    }

    for (size_t rB = 0; rB < K; rB++) {
      size_t pB = ranked_props[rB];
      if (pA == pB) continue;

      float pair_cost = cutoffsA_penalty;
      std::vector<DP1DResult> refs(num_segs);

      for (size_t s = 0; s < num_segs; s++) {
        if (seg_samples[s].size() < 16) {
          pair_cost += seg_base_cost[s];
          refs[s].cost = seg_base_cost[s];
          refs[s].segment_preds = {dpA.segment_preds[s]};
        } else {
          DP1DResult refB = Run1DDPOnSamples(tree_samples, pB, &seg_samples[s], nb_repeats);
          if (refB.cost < seg_base_cost[s] && !refB.cutoffs.empty()) {
            pair_cost += refB.cost;
            refs[s] = std::move(refB);
          } else {
            pair_cost += seg_base_cost[s];
            refs[s].cost = seg_base_cost[s];
            refs[s].segment_preds = {dpA.segment_preds[s]};
          }
        }
      }

      if (pair_cost < best_pair.cost) {
        best_pair.cost = pair_cost;
        best_pair.pA = pA;
        best_pair.pB = pB;
        best_pair.dpA = dpA;
        best_pair.refinementsB = std::move(refs);
      }
    }
  }

  if (best_pair.cost == std::numeric_limits<float>::max()) {
    FindBestTree1dDP(tree_samples, nb_repeats, tree);
    return;
  }

  std::vector<size_t> seg_leaf_pos(best_pair.refinementsB.size(), static_cast<size_t>(-1));
  BuildTreeFrom1D(best_pair.pA + tree_samples.NumStaticProps(),
                  best_pair.dpA.cutoffs, best_pair.dpA.segment_preds,
                  tree_samples, tree, 0, &seg_leaf_pos);

  for (size_t s = 0; s < best_pair.refinementsB.size(); s++) {
    const auto& ref = best_pair.refinementsB[s];
    if (!ref.cutoffs.empty() && seg_leaf_pos[s] != static_cast<size_t>(-1)) {
      BuildTreeFrom1D(best_pair.pB + tree_samples.NumStaticProps(),
                      ref.cutoffs, ref.segment_preds, tree_samples, tree,
                      seg_leaf_pos[s]);
    }
  }
}

void FindBestTreeJoint2dDP(TreeSamples& tree_samples, float nb_repeats, Tree* tree) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  if (num_props <= 1 || tree_samples.NumDistinctSamples() == 0) {
    FindBestTree1dDP(tree_samples, nb_repeats, tree);
    return;
  }

  std::vector<DP1DResult> dps(num_props);
  std::vector<size_t> ranked_props(num_props);
  for (size_t i = 0; i < num_props; i++) {
    dps[i] = Run1DDPOnSamples(tree_samples, i, nullptr, nb_repeats);
    ranked_props[i] = i;
  }
  std::sort(ranked_props.begin(), ranked_props.end(), [&](size_t a, size_t b) {
    return dps[a].cost < dps[b].cost;
  });

  const size_t total_samples = tree_samples.NumDistinctSamples();
  const size_t num_predictors = tree_samples.NumPredictors();

  struct NestedDPResult {
    float cost = std::numeric_limits<float>::max();
    size_t pA = 0;
    size_t pB = 1;
    std::vector<int32_t> cutoffsA;
    std::vector<DP1DResult> segment_refinements;
  };

  NestedDPResult best_joint;

  std::vector<std::pair<size_t, size_t>> pairs_to_try;
  pairs_to_try.push_back({ranked_props[0], ranked_props[1]});
  pairs_to_try.push_back({ranked_props[1], ranked_props[0]});
  if (num_props > 2) {
    pairs_to_try.push_back({ranked_props[0], ranked_props[2]});
  }

  for (const auto& pair : pairs_to_try) {
    size_t pA = pair.first;
    size_t pB = pair.second;

    std::vector<int32_t> vals;
    vals.reserve(total_samples);
    for (size_t i = 0; i < total_samples; i++) {
      vals.push_back(tree_samples.Property<false>(pA, i));
    }
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

    if (vals.size() <= 1) continue;

    std::vector<int32_t> cand_cutoffsA;
    const size_t max_cand_cuts = 15;
    if (vals.size() <= max_cand_cuts + 1) {
      for (size_t i = 0; i + 1 < vals.size(); i++) {
        cand_cutoffsA.push_back(vals[i]);
      }
    } else {
      for (size_t i = 1; i <= max_cand_cuts; i++) {
        size_t idx = (i * (vals.size() - 1)) / (max_cand_cuts + 1);
        cand_cutoffsA.push_back(vals[idx]);
      }
      std::sort(cand_cutoffsA.begin(), cand_cutoffsA.end());
      cand_cutoffsA.erase(std::unique(cand_cutoffsA.begin(), cand_cutoffsA.end()),
                          cand_cutoffsA.end());
    }

    const size_t num_bins = cand_cutoffsA.size() + 1;
    std::vector<std::vector<uint32_t>> bin_samples(num_bins);
    for (size_t i = 0; i < total_samples; i++) {
      int32_t v = tree_samples.Property<false>(pA, i);
      size_t b = std::upper_bound(cand_cutoffsA.begin(), cand_cutoffsA.end(), v) -
                 cand_cutoffsA.begin();
      bin_samples[b].push_back(i);
    }

    auto split_penaltyA = [&](int32_t cutoff_val) -> float {
      int32_t unquant = tree_samples.UnquantizeProperty(
          pA + tree_samples.NumStaticProps(), cutoff_val);
      return (110.0f + 3.0f * FastLog2f(std::abs(unquant) + 1.0f)) * (nb_repeats / 0.3f);
    };

    std::vector<float> dp(num_bins, std::numeric_limits<float>::max());
    std::vector<int32_t> opt_prev(num_bins, -1);
    std::vector<DP1DResult> opt_ref(num_bins);

    for (size_t i = 0; i < num_bins; i++) {
      std::vector<uint32_t> interval_samples;
      for (int32_t j = static_cast<int32_t>(i); j >= 0; j--) {
        const auto& b_samp = bin_samples[j];
        interval_samples.insert(interval_samples.end(), b_samp.begin(), b_samp.end());

        float interval_cost = 0.0f;
        DP1DResult refB;

        if (interval_samples.empty()) {
          interval_cost = 0.0f;
          refB.cost = 0.0f;
          refB.segment_preds = {0};
        } else {
          float best_leaf_cost = std::numeric_limits<float>::max();
          size_t best_leaf_pred = 0;
          for (size_t pred = 0; pred < num_predictors; pred++) {
            size_t max_sym = 0;
            for (uint32_t idx : interval_samples) {
              max_sym = std::max(max_sym, static_cast<size_t>(tree_samples.Token(pred, idx) + 1));
            }
            max_sym = Padded(max_sym);
            std::vector<int32_t> hist(max_sym, 0);
            int64_t extra = 0;
            for (uint32_t idx : interval_samples) {
              size_t cnt = tree_samples.Count(idx);
              hist[tree_samples.Token(pred, idx)] += cnt;
              extra += tree_samples.RTokens(pred)[idx].nbits * cnt;
            }
            float bits = EstimateBits(hist.data(), max_sym) + extra;
            if (bits < best_leaf_cost) {
              best_leaf_cost = bits;
              best_leaf_pred = pred;
            }
          }

          if (interval_samples.size() < 32) {
            interval_cost = best_leaf_cost;
            refB.cost = best_leaf_cost;
            refB.segment_preds = {best_leaf_pred};
          } else {
            refB = Run1DDPOnSamples(tree_samples, pB, &interval_samples, nb_repeats);
            if (refB.cost < best_leaf_cost && !refB.cutoffs.empty()) {
              interval_cost = refB.cost;
            } else {
              interval_cost = best_leaf_cost;
              refB.cost = best_leaf_cost;
              refB.cutoffs.clear();
              refB.segment_preds = {best_leaf_pred};
            }
          }
        }

        if (j == 0) {
          if (interval_cost < dp[i]) {
            dp[i] = interval_cost;
            opt_prev[i] = -1;
            opt_ref[i] = std::move(refB);
          }
        } else {
          int32_t p_idx = j - 1;
          if (dp[p_idx] != std::numeric_limits<float>::max()) {
            float cand = dp[p_idx] + interval_cost + split_penaltyA(cand_cutoffsA[p_idx]);
            if (cand < dp[i]) {
              dp[i] = cand;
              opt_prev[i] = p_idx;
              opt_ref[i] = std::move(refB);
            }
          }
        }
      }
    }

    if (dp[num_bins - 1] < best_joint.cost) {
      best_joint.cost = dp[num_bins - 1];
      best_joint.pA = pA;
      best_joint.pB = pB;
      best_joint.cutoffsA.clear();
      best_joint.segment_refinements.clear();

      int32_t curr = static_cast<int32_t>(num_bins - 1);
      while (curr != -1) {
        best_joint.segment_refinements.push_back(std::move(opt_ref[curr]));
        int32_t prev = opt_prev[curr];
        if (prev != -1) {
          best_joint.cutoffsA.push_back(cand_cutoffsA[prev]);
        }
        curr = prev;
      }
      std::reverse(best_joint.cutoffsA.begin(), best_joint.cutoffsA.end());
      std::reverse(best_joint.segment_refinements.begin(),
                   best_joint.segment_refinements.end());
    }
  }

  if (best_joint.cost == std::numeric_limits<float>::max()) {
    FindBestTree1dDP(tree_samples, nb_repeats, tree);
    return;
  }

  std::vector<size_t> pA_dummy_preds(best_joint.segment_refinements.size(), 0);
  for (size_t s = 0; s < best_joint.segment_refinements.size(); s++) {
    if (!best_joint.segment_refinements[s].segment_preds.empty()) {
      pA_dummy_preds[s] = best_joint.segment_refinements[s].segment_preds[0];
    }
  }

  std::vector<size_t> seg_leaf_pos(best_joint.segment_refinements.size(), static_cast<size_t>(-1));
  BuildTreeFrom1D(best_joint.pA + tree_samples.NumStaticProps(),
                  best_joint.cutoffsA, pA_dummy_preds,
                  tree_samples, tree, 0, &seg_leaf_pos);

  for (size_t s = 0; s < best_joint.segment_refinements.size(); s++) {
    const auto& ref = best_joint.segment_refinements[s];
    if (!ref.cutoffs.empty() && seg_leaf_pos[s] != static_cast<size_t>(-1)) {
      BuildTreeFrom1D(best_joint.pB + tree_samples.NumStaticProps(),
                      ref.cutoffs, ref.segment_preds, tree_samples, tree,
                      seg_leaf_pos[s]);
    }
  }
}

void FindBestTreeGrid2dDP(TreeSamples& tree_samples, float nb_repeats, Tree* tree) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  if (num_props <= 1 || tree_samples.NumDistinctSamples() == 0) {
    FindBestTree1dDP(tree_samples, nb_repeats, tree);
    return;
  }

  std::vector<DP1DResult> dps(num_props);
  for (size_t i = 0; i < num_props; i++) {
    dps[i] = Run1DDPOnSamples(tree_samples, i, nullptr, nb_repeats);
  }

  size_t pA = 0;
  size_t pB = 1;
  float best_c = std::numeric_limits<float>::max();
  for (size_t i = 0; i < num_props; i++) {
    if (dps[i].cost < best_c) {
      best_c = dps[i].cost;
      pA = i;
    }
  }
  best_c = std::numeric_limits<float>::max();
  for (size_t i = 0; i < num_props; i++) {
    if (i != pA && dps[i].cost < best_c) {
      best_c = dps[i].cost;
      pB = i;
    }
  }

  std::vector<int32_t> cutoffsA = dps[pA].cutoffs;
  if (cutoffsA.size() > 3) cutoffsA.resize(3);
  std::vector<int32_t> cutoffsB = dps[pB].cutoffs;
  if (cutoffsB.size() > 3) cutoffsB.resize(3);

  const size_t total_samples = tree_samples.NumDistinctSamples();
  const size_t num_predictors = tree_samples.NumPredictors();
  const size_t rows = cutoffsA.size() + 1;
  const size_t cols = cutoffsB.size() + 1;

  std::vector<std::vector<std::vector<uint32_t>>> cell_samples(
      rows, std::vector<std::vector<uint32_t>>(cols));
  for (size_t i = 0; i < total_samples; i++) {
    int32_t va = tree_samples.Property<false>(pA, i);
    int32_t vb = tree_samples.Property<false>(pB, i);
    size_t r = std::upper_bound(cutoffsA.begin(), cutoffsA.end(), va) - cutoffsA.begin();
    size_t c = std::upper_bound(cutoffsB.begin(), cutoffsB.end(), vb) - cutoffsB.begin();
    cell_samples[r][c].push_back(i);
  }

  std::vector<std::vector<size_t>> cell_preds(rows, std::vector<size_t>(cols, 0));
  for (size_t r = 0; r < rows; r++) {
    for (size_t c = 0; c < cols; c++) {
      if (cell_samples[r][c].empty()) continue;
      size_t best_pred = 0;
      float best_cost = std::numeric_limits<float>::max();
      for (size_t pred = 0; pred < num_predictors; pred++) {
        size_t max_sym = 0;
        for (uint32_t idx : cell_samples[r][c]) {
          max_sym = std::max(max_sym, static_cast<size_t>(tree_samples.Token(pred, idx) + 1));
        }
        max_sym = Padded(max_sym);
        std::vector<int32_t> hist(max_sym, 0);
        int64_t extra = 0;
        for (uint32_t idx : cell_samples[r][c]) {
          size_t cnt = tree_samples.Count(idx);
          hist[tree_samples.Token(pred, idx)] += cnt;
          extra += tree_samples.RTokens(pred)[idx].nbits * cnt;
        }
        float bits = EstimateBits(hist.data(), max_sym) + extra;
        if (bits < best_cost) {
          best_cost = bits;
          best_pred = pred;
        }
      }
      cell_preds[r][c] = best_pred;
    }
  }

  std::vector<size_t> dummy_preds(rows, 0);
  std::vector<size_t> row_leaf_pos(rows, static_cast<size_t>(-1));
  BuildTreeFrom1D(pA + tree_samples.NumStaticProps(), cutoffsA, dummy_preds,
                  tree_samples, tree, 0, &row_leaf_pos);

  for (size_t r = 0; r < rows; r++) {
    if (row_leaf_pos[r] != static_cast<size_t>(-1)) {
      BuildTreeFrom1D(pB + tree_samples.NumStaticProps(), cutoffsB, cell_preds[r],
                      tree_samples, tree, row_leaf_pos[r]);
    }
  }
}

void FindBestTreeDispatch(
    TreeSamples &tree_samples, float threshold,
    const std::vector<ModularMultiplierInfo> &mul_info,
    StaticPropRange static_prop_range, float fast_decode_multiplier, Tree *tree,
    float nb_repeats, ModularOptions::TreeLearningMode tree_learning_mode) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  if (num_props == 0 || tree_learning_mode == ModularOptions::TreeLearningMode::kGreedy) {
    FindBestSplit(tree_samples, threshold, mul_info, static_prop_range,
                  fast_decode_multiplier, tree);
    return;
  }
  if (tree_learning_mode == ModularOptions::TreeLearningMode::k1dDP) {
    FindBestTree1dDP(tree_samples, nb_repeats, tree);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::k2PropertyDP) {
    FindBestTree2PropDP(tree_samples, nb_repeats, tree);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kJoint2dDP) {
    FindBestTreeJoint2dDP(tree_samples, nb_repeats, tree);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid2dDP) {
    FindBestTreeGrid2dDP(tree_samples, nb_repeats, tree);
  } else {
    FindBestSplit(tree_samples, threshold, mul_info, static_prop_range,
                  fast_decode_multiplier, tree);
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(FindBestTreeDispatch);  // Local function.

Status ComputeBestTree(TreeSamples &tree_samples, float threshold,
                       const std::vector<ModularMultiplierInfo> &mul_info,
                       StaticPropRange static_prop_range,
                       float fast_decode_multiplier, Tree *tree,
                       float nb_repeats,
                       ModularOptions::TreeLearningMode tree_learning_mode) {
  // Initialize tree.
  tree->emplace_back();
  tree->back().property = -1;
  tree->back().predictor = tree_samples.PredictorFromIndex(0);
  tree->back().predictor_offset = 0;
  tree->back().multiplier = 1;
  JXL_ENSURE(tree_samples.NumProperties() < 64);

  JXL_ENSURE(tree_samples.NumDistinctSamples() <=
             std::numeric_limits<uint32_t>::max());

  const char* env_mode = getenv("JXL_TREE_LEARNING_MODE");
  if (env_mode != nullptr) {
    if (strcmp(env_mode, "1d") == 0 || strcmp(env_mode, "dp1") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::k1dDP;
    } else if (strcmp(env_mode, "2prop") == 0 || strcmp(env_mode, "dp2") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::k2PropertyDP;
    } else if (strcmp(env_mode, "joint") == 0 || strcmp(env_mode, "nested") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kJoint2dDP;
    } else if (strcmp(env_mode, "grid") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGrid2dDP;
    } else if (strcmp(env_mode, "greedy") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGreedy;
    }
  }

  HWY_DYNAMIC_DISPATCH(FindBestTreeDispatch)
  (tree_samples, threshold, mul_info, static_prop_range, fast_decode_multiplier,
   tree, nb_repeats, tree_learning_mode);
  return true;
}

#if JXL_CXX_LANG < JXL_CXX_17
constexpr int32_t TreeSamples::kPropertyRange;
constexpr uint32_t TreeSamples::kDedupEntryUnused;
#endif

Status TreeSamples::SetPredictor(Predictor predictor,
                                 ModularOptions::TreeMode wp_tree_mode) {
  if (wp_tree_mode == ModularOptions::TreeMode::kWPOnly) {
    predictors = {Predictor::Weighted};
    residuals.resize(1);
    return true;
  }
  if (wp_tree_mode == ModularOptions::TreeMode::kNoWP &&
      predictor == Predictor::Weighted) {
    return JXL_FAILURE("Invalid predictor settings");
  }
  if (predictor == Predictor::Variable) {
    for (size_t i = 0; i < kNumModularPredictors; i++) {
      predictors.push_back(static_cast<Predictor>(i));
    }
    std::swap(predictors[0], predictors[static_cast<int>(Predictor::Weighted)]);
    std::swap(predictors[1], predictors[static_cast<int>(Predictor::Gradient)]);
  } else if (predictor == Predictor::Best) {
    predictors = {Predictor::Weighted, Predictor::Gradient};
  } else {
    predictors = {predictor};
  }
  if (wp_tree_mode == ModularOptions::TreeMode::kNoWP) {
    predictors.erase(
        std::remove(predictors.begin(), predictors.end(), Predictor::Weighted),
        predictors.end());
  }
  residuals.resize(predictors.size());
  return true;
}

Status TreeSamples::SetProperties(const std::vector<uint32_t> &properties,
                                  ModularOptions::TreeMode wp_tree_mode) {
  props_to_use = properties;
  if (wp_tree_mode == ModularOptions::TreeMode::kWPOnly) {
    props_to_use = {static_cast<uint32_t>(kWPProp)};
  }
  if (wp_tree_mode == ModularOptions::TreeMode::kGradientOnly) {
    props_to_use = {static_cast<uint32_t>(kGradientProp)};
  }
  if (wp_tree_mode == ModularOptions::TreeMode::kNoWP) {
    props_to_use.erase(
        std::remove(props_to_use.begin(), props_to_use.end(), kWPProp),
        props_to_use.end());
  }
  if (props_to_use.empty()) {
    return JXL_FAILURE("Invalid property set configuration");
  }
  num_static_props = 0;
  // Check that if static properties present, then those are at the beginning.
  for (size_t i = 0; i < props_to_use.size(); ++i) {
    uint32_t prop = props_to_use[i];
    if (prop < kNumStaticProperties) {
      JXL_DASSERT(i == prop);
      num_static_props++;
    }
  }
  props.resize(props_to_use.size() - num_static_props);
  return true;
}

void TreeSamples::InitTable(size_t log_size) {
  size_t size = 1ULL << log_size;
  if (dedup_table_.size() == size) return;
  dedup_table_.resize(size, kDedupEntryUnused);
  for (size_t i = 0; i < NumDistinctSamples(); i++) {
    if (sample_counts[i] != std::numeric_limits<uint16_t>::max()) {
      AddToTable(i);
    }
  }
}

bool TreeSamples::AddToTableAndMerge(size_t a) {
  size_t pos1 = Hash1(a);
  size_t pos2 = Hash2(a);
  if (dedup_table_[pos1] != kDedupEntryUnused &&
      IsSameSample(a, dedup_table_[pos1])) {
    JXL_DASSERT(sample_counts[a] == 1);
    sample_counts[dedup_table_[pos1]]++;
    // Remove from hash table samples that are saturated.
    if (sample_counts[dedup_table_[pos1]] ==
        std::numeric_limits<uint16_t>::max()) {
      dedup_table_[pos1] = kDedupEntryUnused;
    }
    return true;
  }
  if (dedup_table_[pos2] != kDedupEntryUnused &&
      IsSameSample(a, dedup_table_[pos2])) {
    JXL_DASSERT(sample_counts[a] == 1);
    sample_counts[dedup_table_[pos2]]++;
    // Remove from hash table samples that are saturated.
    if (sample_counts[dedup_table_[pos2]] ==
        std::numeric_limits<uint16_t>::max()) {
      dedup_table_[pos2] = kDedupEntryUnused;
    }
    return true;
  }
  AddToTable(a);
  return false;
}

void TreeSamples::AddToTable(size_t a) {
  size_t pos1 = Hash1(a);
  size_t pos2 = Hash2(a);
  if (dedup_table_[pos1] == kDedupEntryUnused) {
    dedup_table_[pos1] = a;
  } else if (dedup_table_[pos2] == kDedupEntryUnused) {
    dedup_table_[pos2] = a;
  }
}

void TreeSamples::PrepareForSamples(size_t extra_num_samples) {
  for (auto &res : residuals) {
    res.reserve(res.size() + extra_num_samples);
  }
  for (size_t i = 0; i < num_static_props; ++i) {
    static_props[i].reserve(static_props[i].size() + extra_num_samples);
  }
  for (auto &p : props) {
    p.reserve(p.size() + extra_num_samples);
  }
  size_t total_num_samples = extra_num_samples + sample_counts.size();
  size_t next_size = CeilLog2Nonzero(total_num_samples * 3 / 2);
  InitTable(next_size);
}

size_t TreeSamples::Hash1(size_t a) const {
  constexpr uint64_t constant = 0x1e35a7bd;
  uint64_t h = constant;
  for (const auto &r : residuals) {
    h = h * constant + r[a].tok;
    h = h * constant + r[a].nbits;
  }
  for (size_t i = 0; i < num_static_props; ++i) {
    h = h * constant + static_props[i][a];
  }
  for (const auto &p : props) {
    h = h * constant + p[a];
  }
  return (h >> 16) & (dedup_table_.size() - 1);
}
size_t TreeSamples::Hash2(size_t a) const {
  constexpr uint64_t constant = 0x1e35a7bd1e35a7bd;
  uint64_t h = constant;
  for (size_t i = 0; i < num_static_props; ++i) {
    h = h * constant ^ static_props[i][a];
  }
  for (const auto &p : props) {
    h = h * constant ^ p[a];
  }
  for (const auto &r : residuals) {
    h = h * constant ^ r[a].tok;
    h = h * constant ^ r[a].nbits;
  }
  return (h >> 16) & (dedup_table_.size() - 1);
}

bool TreeSamples::IsSameSample(size_t a, size_t b) const {
  bool ret = true;
  for (const auto &r : residuals) {
    if (r[a].tok != r[b].tok) {
      ret = false;
    }
    if (r[a].nbits != r[b].nbits) {
      ret = false;
    }
  }
  for (size_t i = 0; i < num_static_props; ++i) {
    if (static_props[i][a] != static_props[i][b]) {
      ret = false;
    }
  }
  for (const auto &p : props) {
    if (p[a] != p[b]) {
      ret = false;
    }
  }
  return ret;
}

void TreeSamples::AddSample(pixel_type_w pixel, const Properties &properties,
                            const pixel_type_w *predictions) {
  for (size_t i = 0; i < predictors.size(); i++) {
    pixel_type v = pixel - predictions[static_cast<int>(predictors[i])];
    uint32_t tok, nbits, bits;
    HybridUintConfig(4, 1, 2).Encode(PackSigned(v), &tok, &nbits, &bits);
    JXL_DASSERT(tok < 256);
    JXL_DASSERT(nbits < 256);
    ResidualToken token = {static_cast<uint8_t>(tok),
                           static_cast<uint8_t>(nbits)};
    residuals[i].push_back(token);
  }
  for (size_t i = 0; i < num_static_props; ++i) {
    static_props[i].push_back(QuantizeStaticProperty(i, properties[i]));
  }
  for (size_t i = num_static_props; i < props_to_use.size(); i++) {
    props[i - num_static_props].push_back(QuantizeProperty(i, properties[props_to_use[i]]));
  }
  sample_counts.push_back(1);
  num_samples++;
  if (AddToTableAndMerge(sample_counts.size() - 1)) {
    for (auto &r : residuals) r.pop_back();
    for (size_t i = 0; i < num_static_props; ++i) static_props[i].pop_back();
    for (auto &p : props) p.pop_back();
    sample_counts.pop_back();
  }
}

void TreeSamples::Swap(size_t a, size_t b) {
  if (a == b) return;
  for (auto &r : residuals) {
    std::swap(r[a], r[b]);
  }
  for (size_t i = 0; i < num_static_props; ++i) {
    std::swap(static_props[i][a], static_props[i][b]);
  }
  for (auto &p : props) {
    std::swap(p[a], p[b]);
  }
  std::swap(sample_counts[a], sample_counts[b]);
}

namespace {
std::vector<int32_t> QuantizeHistogram(const std::vector<uint32_t> &histogram,
                                       size_t num_chunks) {
  if (histogram.empty() || num_chunks == 0) return {};
  uint64_t sum = std::accumulate(histogram.begin(), histogram.end(), 0LU);
  if (sum == 0) return {};
  // TODO(veluca): selecting distinct quantiles is likely not the best
  // way to go about this.
  std::vector<int32_t> thresholds;
  uint64_t cumsum = 0;
  uint64_t threshold = 1;
  for (size_t i = 0; i < histogram.size(); i++) {
    cumsum += histogram[i];
    if (cumsum * num_chunks >= threshold * sum) {
      thresholds.push_back(i);
      while (cumsum * num_chunks >= threshold * sum) threshold++;
    }
  }
  JXL_DASSERT(thresholds.size() <= num_chunks);
  // last value collects all histogram and is not really a threshold
  thresholds.pop_back();
  return thresholds;
}

std::vector<int32_t> QuantizeSamples(const std::vector<int32_t> &samples,
                                     size_t num_chunks) {
  if (samples.empty()) return {};
  int min = *std::min_element(samples.begin(), samples.end());
  constexpr int kRange = 512;
  min = jxl::Clamp1(min, -kRange, kRange);
  std::vector<uint32_t> counts(2 * kRange + 1);
  for (int s : samples) {
    uint32_t sample_offset = jxl::Clamp1(s, -kRange, kRange) - min;
    counts[sample_offset]++;
  }
  std::vector<int32_t> thresholds = QuantizeHistogram(counts, num_chunks);
  for (auto &v : thresholds) v += min;
  return thresholds;
}

// `to[i]` is assigned value `v` conforming `from[v] <= i && from[v-1] > i`.
// This is because the decision node in the tree splits on (property) > i,
// hence everything that is not > of a threshold should be clustered
// together.
template <typename T>
void QuantMap(const std::vector<int32_t> &from, std::vector<T> &to,
              size_t num_pegs, int bias) {
  to.resize(num_pegs);
  size_t mapped = 0;
  for (size_t i = 0; i < num_pegs; i++) {
    while (mapped < from.size() && static_cast<int>(i) - bias > from[mapped]) {
      mapped++;
    }
    JXL_DASSERT(static_cast<T>(mapped) == mapped);
    to[i] = mapped;
  }
}
}  // namespace

void TreeSamples::PreQuantizeProperties(
    const StaticPropRange &range,
    const std::vector<ModularMultiplierInfo> &multiplier_info,
    const std::vector<uint32_t> &group_pixel_count,
    const std::vector<uint32_t> &channel_pixel_count,
    std::vector<pixel_type> &pixel_samples,
    std::vector<pixel_type> &diff_samples, size_t max_property_values) {
  // If we have forced splits because of multipliers, choose channel and group
  // thresholds accordingly.
  std::vector<int32_t> group_multiplier_thresholds;
  std::vector<int32_t> channel_multiplier_thresholds;
  for (const auto &v : multiplier_info) {
    if (v.range[0][0] != range[0][0]) {
      channel_multiplier_thresholds.push_back(v.range[0][0] - 1);
    }
    if (v.range[0][1] != range[0][1]) {
      channel_multiplier_thresholds.push_back(v.range[0][1] - 1);
    }
    if (v.range[1][0] != range[1][0]) {
      group_multiplier_thresholds.push_back(v.range[1][0] - 1);
    }
    if (v.range[1][1] != range[1][1]) {
      group_multiplier_thresholds.push_back(v.range[1][1] - 1);
    }
  }
  std::sort(channel_multiplier_thresholds.begin(),
            channel_multiplier_thresholds.end());
  channel_multiplier_thresholds.resize(
      std::unique(channel_multiplier_thresholds.begin(),
                  channel_multiplier_thresholds.end()) -
      channel_multiplier_thresholds.begin());
  std::sort(group_multiplier_thresholds.begin(),
            group_multiplier_thresholds.end());
  group_multiplier_thresholds.resize(
      std::unique(group_multiplier_thresholds.begin(),
                  group_multiplier_thresholds.end()) -
      group_multiplier_thresholds.begin());

  compact_properties.resize(props_to_use.size());
  auto quantize_channel = [&]() {
    if (!channel_multiplier_thresholds.empty()) {
      return channel_multiplier_thresholds;
    }
    return QuantizeHistogram(channel_pixel_count, max_property_values);
  };
  auto quantize_group_id = [&]() {
    if (!group_multiplier_thresholds.empty()) {
      return group_multiplier_thresholds;
    }
    return QuantizeHistogram(group_pixel_count, max_property_values);
  };
  auto quantize_coordinate = [&]() {
    std::vector<int32_t> quantized;
    quantized.reserve(max_property_values - 1);
    for (size_t i = 0; i + 1 < max_property_values; i++) {
      quantized.push_back((i + 1) * 256 / max_property_values - 1);
    }
    return quantized;
  };
  std::vector<int32_t> abs_pixel_thresholds;
  std::vector<int32_t> pixel_thresholds;
  auto quantize_pixel_property = [&]() {
    if (pixel_thresholds.empty()) {
      pixel_thresholds = QuantizeSamples(pixel_samples, max_property_values);
    }
    return pixel_thresholds;
  };
  auto quantize_abs_pixel_property = [&]() {
    if (abs_pixel_thresholds.empty()) {
      quantize_pixel_property();  // Compute the non-abs thresholds.
      for (auto &v : pixel_samples) v = std::abs(v);
      abs_pixel_thresholds =
          QuantizeSamples(pixel_samples, max_property_values);
    }
    return abs_pixel_thresholds;
  };
  std::vector<int32_t> abs_diff_thresholds;
  std::vector<int32_t> diff_thresholds;
  auto quantize_diff_property = [&]() {
    if (diff_thresholds.empty()) {
      diff_thresholds = QuantizeSamples(diff_samples, max_property_values);
    }
    return diff_thresholds;
  };
  auto quantize_abs_diff_property = [&]() {
    if (abs_diff_thresholds.empty()) {
      quantize_diff_property();  // Compute the non-abs thresholds.
      for (auto &v : diff_samples) v = std::abs(v);
      abs_diff_thresholds = QuantizeSamples(diff_samples, max_property_values);
    }
    return abs_diff_thresholds;
  };
  auto quantize_wp = [&]() {
    if (max_property_values < 32) {
      return std::vector<int32_t>{-127, -63, -31, -15, -7, -3, -1, 0,
                                  1,    3,   7,   15,  31, 63, 127};
    }
    if (max_property_values < 64) {
      return std::vector<int32_t>{-255, -191, -127, -95, -63, -47, -31, -23,
                                  -15,  -11,  -7,   -5,  -3,  -1,  0,   1,
                                  3,    5,    7,    11,  15,  23,  31,  47,
                                  63,   95,   127,  191, 255};
    }
    return std::vector<int32_t>{
        -255, -223, -191, -159, -127, -111, -95, -79, -63, -55, -47,
        -39,  -31,  -27,  -23,  -19,  -15,  -13, -11, -9,  -7,  -6,
        -5,   -4,   -3,   -2,   -1,   0,    1,   2,   3,   4,   5,
        6,    7,    9,    11,   13,   15,   19,  23,  27,  31,  39,
        47,   55,   63,   79,   95,   111,  127, 159, 191, 223, 255};
  };

  property_mapping.resize(props_to_use.size() - num_static_props);
  for (size_t i = 0; i < props_to_use.size(); i++) {
    if (props_to_use[i] == 0) {
      compact_properties[i] = quantize_channel();
    } else if (props_to_use[i] == 1) {
      compact_properties[i] = quantize_group_id();
    } else if (props_to_use[i] == 2 || props_to_use[i] == 3) {
      compact_properties[i] = quantize_coordinate();
    } else if (props_to_use[i] == 6 || props_to_use[i] == 7 ||
               props_to_use[i] == 8 ||
               (props_to_use[i] >= kNumNonrefProperties &&
                (props_to_use[i] - kNumNonrefProperties) % 4 == 1)) {
      compact_properties[i] = quantize_pixel_property();
    } else if (props_to_use[i] == 4 || props_to_use[i] == 5 ||
               (props_to_use[i] >= kNumNonrefProperties &&
                (props_to_use[i] - kNumNonrefProperties) % 4 == 0)) {
      compact_properties[i] = quantize_abs_pixel_property();
    } else if (props_to_use[i] >= kNumNonrefProperties &&
               (props_to_use[i] - kNumNonrefProperties) % 4 == 2) {
      compact_properties[i] = quantize_abs_diff_property();
    } else if (props_to_use[i] == kWPProp) {
      compact_properties[i] = quantize_wp();
    } else {
      compact_properties[i] = quantize_diff_property();
    }
    if (i < num_static_props) {
      QuantMap(compact_properties[i], static_property_mapping[i],
               kPropertyRange * 2 + 1, kPropertyRange);
    } else {
      QuantMap(compact_properties[i], property_mapping[i - num_static_props],
               kPropertyRange * 2 + 1, kPropertyRange);
    }
  }
}

void CollectPixelSamples(const Image &image, const ModularOptions &options,
                         uint32_t group_id,
                         std::vector<uint32_t> &group_pixel_count,
                         std::vector<uint32_t> &channel_pixel_count,
                         std::vector<pixel_type> &pixel_samples,
                         std::vector<pixel_type> &diff_samples) {
  if (options.nb_repeats == 0) return;
  if (group_pixel_count.size() <= group_id) {
    group_pixel_count.resize(group_id + 1);
  }
  if (channel_pixel_count.size() < image.channel.size()) {
    channel_pixel_count.resize(image.channel.size());
  }
  Rng rng(group_id);
  // Sample 10% of the final number of samples for property quantization.
  float fraction = std::min(options.nb_repeats * 0.1, 0.99);
  Rng::GeometricDistribution dist = Rng::MakeGeometric(fraction);
  size_t total_pixels = 0;
  std::vector<size_t> channel_ids;
  for (size_t i = 0; i < image.channel.size(); i++) {
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    if (image.channel[i].w <= 1 || image.channel[i].h == 0) {
      continue;  // skip empty or width-1 channels.
    }
    channel_ids.push_back(i);
    group_pixel_count[group_id] += image.channel[i].w * image.channel[i].h;
    channel_pixel_count[i] += image.channel[i].w * image.channel[i].h;
    total_pixels += image.channel[i].w * image.channel[i].h;
  }
  if (channel_ids.empty()) return;
  pixel_samples.reserve(pixel_samples.size() + fraction * total_pixels);
  diff_samples.reserve(diff_samples.size() + fraction * total_pixels);
  size_t i = 0;
  size_t y = 0;
  size_t x = 0;
  auto advance = [&](size_t amount) {
    x += amount;
    // Detect row overflow (rare).
    while (x >= image.channel[channel_ids[i]].w) {
      x -= image.channel[channel_ids[i]].w;
      y++;
      // Detect end-of-channel (even rarer).
      if (y == image.channel[channel_ids[i]].h) {
        i++;
        y = 0;
        if (i >= channel_ids.size()) {
          return;
        }
      }
    }
  };
  advance(rng.Geometric(dist));
  for (; i < channel_ids.size(); advance(rng.Geometric(dist) + 1)) {
    const pixel_type *row = image.channel[channel_ids[i]].Row(y);
    pixel_samples.push_back(row[x]);
    size_t xp = x == 0 ? 1 : x - 1;
    diff_samples.push_back(static_cast<int64_t>(row[x]) - row[xp]);
  }
}

// TODO(veluca): very simple encoding scheme. This should be improved.
Status TokenizeTree(const Tree &tree, std::vector<Token> *tokens,
                    Tree *decoder_tree) {
  JXL_ENSURE(tree.size() <= kMaxTreeSize);
  std::queue<int> q;
  q.push(0);
  size_t leaf_id = 0;
  decoder_tree->clear();
  while (!q.empty()) {
    int cur = q.front();
    q.pop();
    JXL_ENSURE(tree[cur].property >= -1);
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
      JXL_ENSURE(tree[cur].predictor < Predictor::Best);
      decoder_tree->emplace_back(
          -1, 0, static_cast<int>(leaf_id), 0, tree[cur].predictor,
          tree[cur].predictor_offset, tree[cur].multiplier);
      leaf_id++;
      continue;
    }
    decoder_tree->emplace_back(
        tree[cur].property, tree[cur].splitval,
        static_cast<int>(decoder_tree->size() + q.size() + 1),
        static_cast<int>(decoder_tree->size() + q.size() + 2), Predictor::Zero,
        0, 1);
    q.push(tree[cur].lchild);
    q.push(tree[cur].rchild);
    tokens->emplace_back(kSplitValContext, PackSigned(tree[cur].splitval));
  }
  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
