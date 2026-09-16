// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/encoding/enc_ma.h"

#include <algorithm>
#include <array>
#include <bitset>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <numeric>
#include <queue>
#include <set>
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

void FindBestSplit(TreeSamples &tree_samples, float scale,
                   const std::vector<ModularMultiplierInfo> &mul_info,
                   StaticPropRange initial_static_prop_range,
                   float fast_decode_multiplier, Tree *tree,
                   float base_node_cost = 92.0f, float log_node_cost = 1.2f) {
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
  float threshold = base_node_cost * scale;

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
      float val_penalty = 0.0f;
      Predictor lpred = Predictor::Zero;
      Predictor rpred = Predictor::Zero;
      float Cost() const { return lcost + rcost + val_penalty; }
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

          pixel_type dequant = tree_samples.UnquantizeProperty(prop, i);
          float val_penalty =
              (log_node_cost * FastLog2f(std::abs(dequant) + 1.0f)) * scale;

          bool uses_wp = tree_samples.PropertyFromIndex(prop) == kWPProp ||
                         costs_l[i - first_used].pred == Predictor::Weighted ||
                         costs_r[i - first_used].pred == Predictor::Weighted;
          bool zero_entropy_side = rcost == 0 || lcost == 0;

          SplitInfo &best_ref =
              tree_samples.PropertyFromIndex(prop) < kNumStaticProperties
                  ? (zero_entropy_side ? best_split_static_constant
                                       : best_split_static)
                  : (uses_wp ? best_split_nonstatic : best_split_nowp);
          if (lcost + rcost + val_penalty < best_ref.Cost()) {
            best_ref.prop = prop;
            best_ref.val = i;
            best_ref.pos = split;
            best_ref.lcost = lcost;
            best_ref.lpred = costs_l[i - first_used].pred;
            best_ref.rcost = rcost;
            best_ref.rpred = costs_r[i - first_used].pred;
            best_ref.val_penalty = val_penalty;
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

struct GreedyCutRecord {
  int32_t cutoff_val;
  float gain;
};

struct DP1DResult {
  size_t prop_dim = 0;
  float cost = std::numeric_limits<float>::max();
  std::vector<int32_t> cutoffs;
  std::vector<size_t> segment_preds;
  std::vector<GreedyCutRecord> cut_history;
};
struct Scratch1D {
  std::vector<int32_t> last_exist_le;
  std::vector<int64_t> pref_counts;
  std::vector<int64_t> pref_extra;
  std::vector<int32_t> pref_freq;
  std::vector<float> dp;
  std::vector<int32_t> opt_split;
  std::vector<size_t> pred_split;
  std::vector<int32_t> residual_hist;
  std::vector<int32_t> right_hist;
  std::vector<float> memo_cost;
  std::vector<size_t> memo_pred;

  void Init(size_t N, size_t P, size_t S) {
    if (last_exist_le.size() < N) last_exist_le.resize(N);
    if (pref_counts.size() < N) pref_counts.resize(N);
    if (pref_extra.size() < N * P) pref_extra.resize(N * P);
    if (pref_freq.size() < N * P * S) pref_freq.resize(N * P * S);
    if (dp.size() < N) dp.resize(N);
    if (opt_split.size() < N) opt_split.resize(N);
    if (pred_split.size() < N) pred_split.resize(N);
    if (residual_hist.size() < P * S) residual_hist.resize(P * S);
    if (right_hist.size() < P * S) right_hist.resize(P * S);
    if (memo_cost.size() < N * N) memo_cost.resize(N * N);
    if (memo_pred.size() < N * N) memo_pred.resize(N * N);
  }
};

struct DP1DProfilingData {
  std::atomic<uint64_t> sample_accum_calls{0};
  std::atomic<uint64_t> monge_dp_calls{0};
  std::atomic<uint64_t> total_samples{0};
  std::atomic<uint64_t> total_N{0};
  std::atomic<uint64_t> time_distinct_ns{0};
  std::atomic<uint64_t> time_accum_ns{0};
  std::atomic<uint64_t> time_prefix_sums_ns{0};
  std::atomic<uint64_t> time_actual_dp_ns{0};
  std::atomic<uint64_t> time_eval_interval_ns{0};
  std::atomic<uint64_t> num_eval_intervals{0};
  std::atomic<uint64_t> num_memo_hits{0};
  std::atomic<uint64_t> greedy_calls{0};
  std::atomic<uint64_t> time_greedy_ns{0};
  std::atomic<uint64_t> time_max_symbols_ns{0};
  std::atomic<uint64_t> time_alloc_ns{0};
  std::atomic<uint64_t> time_scatter_ns{0};
  std::atomic<uint64_t> time_multi_prop_ns{0};

  void Reset() {
    sample_accum_calls = 0;
    monge_dp_calls = 0;
    total_samples = 0;
    total_N = 0;
    time_distinct_ns = 0;
    time_accum_ns = 0;
    time_max_symbols_ns = 0;
    time_alloc_ns = 0;
    time_scatter_ns = 0;
    time_multi_prop_ns = 0;
    time_prefix_sums_ns = 0;
    time_actual_dp_ns = 0;
    time_eval_interval_ns = 0;
    num_eval_intervals = 0;
    num_memo_hits = 0;
    greedy_calls = 0;
    time_greedy_ns = 0;
  }
};
static DP1DProfilingData g_dp1d_prof;

inline void Print1DDPProfilingReport() {
  uint64_t sample_calls = g_dp1d_prof.sample_accum_calls.load();
  uint64_t monge_calls = g_dp1d_prof.monge_dp_calls.load();
  uint64_t greedy_calls = g_dp1d_prof.greedy_calls.load();
  if (sample_calls == 0 && greedy_calls == 0 && monge_calls == 0) return;
  double t_accum_ms = g_dp1d_prof.time_accum_ns.load() * 1e-6;
  double t_distinct_ms = g_dp1d_prof.time_distinct_ns.load() * 1e-6;
  double t_max_sym_ms = g_dp1d_prof.time_max_symbols_ns.load() * 1e-6;
  double t_alloc_ms = g_dp1d_prof.time_alloc_ns.load() * 1e-6;
  double t_scatter_ms = g_dp1d_prof.time_scatter_ns.load() * 1e-6;
  double t_multi_ms = g_dp1d_prof.time_multi_prop_ns.load() * 1e-6;
  double t_prefix_ms = g_dp1d_prof.time_prefix_sums_ns.load() * 1e-6;
  double t_dp_ms = g_dp1d_prof.time_actual_dp_ns.load() * 1e-6;
  double t_eval_ms = g_dp1d_prof.time_eval_interval_ns.load() * 1e-6;
  double t_greedy_ms = g_dp1d_prof.time_greedy_ns.load() * 1e-6;
  double t_total_1d_ms = t_accum_ms + t_distinct_ms + t_multi_ms + t_prefix_ms + t_dp_ms + t_greedy_ms;
  uint64_t evals = g_dp1d_prof.num_eval_intervals.load();
  uint64_t hits = g_dp1d_prof.num_memo_hits.load();
  uint64_t tot_N = g_dp1d_prof.total_N.load();
  uint64_t tot_samples = g_dp1d_prof.total_samples.load();

  fprintf(stderr, "\n==================== 1D DP / GREEDY TIME BREAKDOWN ====================\n");
  if (sample_calls > 0) {
    fprintf(stderr, "Segment sample accum calls: %lu (avg N: %.1f, avg samples: %.1f)\n",
            sample_calls, (double)tot_N / sample_calls, (double)tot_samples / sample_calls);
  }
  if (greedy_calls > 0) {
    fprintf(stderr, "Calls to Greedy 1D:        %lu (total: %8.3f ms, avg: %.3f ms/call)\n",
            greedy_calls, t_greedy_ms, t_greedy_ms / greedy_calls);
  }
  if (monge_calls > 0) {
    fprintf(stderr, "Calls to Monge 1D DP:      %lu (total: %8.3f ms, avg: %.3f ms/call)\n",
            monge_calls, t_dp_ms, t_dp_ms / monge_calls);
  }
  fprintf(stderr, "------------------------------------------------------------------------\n");
  fprintf(stderr, "1. Sample-to-table accum:  %8.3f ms (%5.1f%%)\n",
          t_distinct_ms + t_accum_ms + t_multi_ms,
          100.0 * (t_distinct_ms + t_accum_ms + t_multi_ms) / t_total_1d_ms);
  fprintf(stderr, "   - Multi-prop screen:    %8.3f ms\n", t_multi_ms);
  fprintf(stderr, "   - Distinct vals & LUT:  %8.3f ms\n", t_distinct_ms);
  fprintf(stderr, "   - Per-segment accum:    %8.3f ms\n", t_accum_ms);
  fprintf(stderr, "     * Max symbols scan:   %8.3f ms\n", t_max_sym_ms);
  fprintf(stderr, "     * Vector allocations: %8.3f ms\n", t_alloc_ms);
  fprintf(stderr, "     * Histogram scatter:  %8.3f ms\n", t_scatter_ms);
  fprintf(stderr, "2. Computing prefix sums:  %8.3f ms (%5.1f%%)\n",
          t_prefix_ms, 100.0 * t_prefix_ms / t_total_1d_ms);
  fprintf(stderr, "3. Actual Monge DP loop:   %8.3f ms (%5.1f%%)\n",
          t_dp_ms, 100.0 * t_dp_ms / t_total_1d_ms);
  fprintf(stderr, "   - eval_interval(j, i):  %8.3f ms (%5.1f%% of DP)\n",
          t_eval_ms, 100.0 * t_eval_ms / (t_dp_ms + 1e-9));
  fprintf(stderr, "   - Deque search & overhead: %5.3f ms\n",
          std::max(0.0, t_dp_ms - t_eval_ms));
  fprintf(stderr, "Total 1D time:             %8.3f ms\n", t_total_1d_ms);
  fprintf(stderr, "Interval evaluations:      %lu evaluated, %lu memo hits (%.1f%% hit rate)\n",
          evals, hits, 100.0 * hits / (evals + hits + 1e-9));
  fprintf(stderr, "========================================================================\n\n");
}

DP1DResult Solve1DDPFromTables(
    TreeSamples& tree_samples, size_t prop_dim,
    const std::vector<int32_t>& valsB,
    const int64_t* interval_counts,
    const int64_t* interval_extra,
    const int32_t* interval_freq,
    const std::vector<size_t>& max_symbols,
    size_t num_predictors,
    size_t max_symbols_stride,
    float scale, float base_node_cost = 92.0f,
    float log_node_cost = 1.2f,
    Scratch1D* scratch = nullptr) {
  DP1DResult res;
  res.prop_dim = prop_dim;
  const size_t prop_idx = prop_dim + tree_samples.NumStaticProps();
  const size_t N = valsB.size();
  const size_t P = num_predictors;
  const size_t S = max_symbols_stride;
  if (N == 0) return res;
  g_dp1d_prof.monge_dp_calls.fetch_add(1, std::memory_order_relaxed);

  Scratch1D local_scratch;
  if (!scratch) {
    scratch = &local_scratch;
  }
  scratch->Init(N, P, S);

  int32_t min_exist = -1;
  int32_t max_exist = -1;
  for (size_t v = 0; v < N; v++) {
    if (interval_counts[v] > 0) {
      if (min_exist == -1) min_exist = static_cast<int32_t>(v);
      max_exist = static_cast<int32_t>(v);
    }
  }

  if (min_exist == -1) {
    res.cost = 0.0f;
    res.segment_preds = {0};
    return res;
  }

  if (min_exist >= max_exist) {
    float best_leaf_cost = std::numeric_limits<float>::max();
    size_t best_leaf_pred = 0;
    for (size_t pred = 0; pred < P; pred++) {
      const int32_t* hist = &interval_freq[(min_exist * P + pred) * S];
      int64_t extra = interval_extra[min_exist * P + pred];
      float bits = EstimateBits(hist, max_symbols[pred]) + extra;
      if (bits < best_leaf_cost) {
        best_leaf_cost = bits;
        best_leaf_pred = pred;
      }
    }
    res.cost = best_leaf_cost;
    res.segment_preds = {best_leaf_pred};
    return res;
  }

  auto t0_pref = std::chrono::high_resolution_clock::now();
  int64_t* pref_counts = scratch->pref_counts.data();
  pref_counts[0] = interval_counts[0];
  for (size_t v = 1; v < N; v++) {
    pref_counts[v] = pref_counts[v - 1] + interval_counts[v];
  }

  int64_t* pref_extra = scratch->pref_extra.data();
  for (size_t pred = 0; pred < P; pred++) {
    pref_extra[pred] = interval_extra[pred];
    for (size_t v = 1; v < N; v++) {
      pref_extra[v * P + pred] =
          pref_extra[(v - 1) * P + pred] + interval_extra[v * P + pred];
    }
  }

  int32_t* pref_freq = scratch->pref_freq.data();
  for (size_t pred = 0; pred < P; pred++) {
    for (size_t k = 0; k < max_symbols[pred]; k++) {
      pref_freq[pred * S + k] = interval_freq[pred * S + k];
    }
    for (size_t v = 1; v < N; v++) {
      for (size_t k = 0; k < max_symbols[pred]; k++) {
        pref_freq[(v * P + pred) * S + k] =
            pref_freq[((v - 1) * P + pred) * S + k] +
            interval_freq[(v * P + pred) * S + k];
      }
    }
  }
  auto t1_pref = std::chrono::high_resolution_clock::now();
  g_dp1d_prof.time_prefix_sums_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1_pref - t0_pref).count(),
      std::memory_order_relaxed);

  auto split_penalty = [&](int32_t cutoff_idx) -> float {
    int32_t unquant = tree_samples.UnquantizeProperty(prop_idx, valsB[cutoff_idx]);
    return (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
  };

  auto t0_dp = std::chrono::high_resolution_clock::now();
  float* memo_cost = scratch->memo_cost.data();
  size_t* memo_pred = scratch->memo_pred.data();
  std::fill(memo_cost, memo_cost + N * N, -1.0f);

  auto eval_interval = [&](int32_t j, int32_t i, size_t* chosen_pred = nullptr) -> float {
    size_t memo_idx = j * N + i;
    if (memo_cost[memo_idx] >= 0.0f) {
      g_dp1d_prof.num_memo_hits.fetch_add(1, std::memory_order_relaxed);
      if (chosen_pred) *chosen_pred = memo_pred[memo_idx];
      return memo_cost[memo_idx];
    }
    g_dp1d_prof.num_eval_intervals.fetch_add(1, std::memory_order_relaxed);
    auto t0_eval = std::chrono::high_resolution_clock::now();
    int64_t cur_cnt = pref_counts[i] - (j > 0 ? pref_counts[j - 1] : 0);
    if (cur_cnt == 0) {
      memo_cost[memo_idx] = 0.0f;
      memo_pred[memo_idx] = 0;
      if (chosen_pred) *chosen_pred = 0;
      return 0.0f;
    }

    float best_int_cost = std::numeric_limits<float>::max();
    size_t best_int_pred = 0;
    int32_t* rh = scratch->residual_hist.data();

    for (size_t pred = 0; pred < P; pred++) {
      const int32_t* pi = &pref_freq[(i * P + pred) * S];
      const int32_t* pj = (j > 0 ? &pref_freq[((j - 1) * P + pred) * S] : nullptr);
      if (pj) {
        for (size_t k = 0; k < max_symbols[pred]; k++) {
          rh[k] = pi[k] - pj[k];
        }
      } else {
        for (size_t k = 0; k < max_symbols[pred]; k++) {
          rh[k] = pi[k];
        }
      }
      float bits = EstimateBits(rh, max_symbols[pred]);
      int64_t extra = pref_extra[i * P + pred] -
                      (j > 0 ? pref_extra[(j - 1) * P + pred] : 0);
      float total = bits + extra;
      if (total < best_int_cost) {
        best_int_cost = total;
        best_int_pred = pred;
      }
    }

    memo_cost[memo_idx] = best_int_cost;
    memo_pred[memo_idx] = best_int_pred;
    if (chosen_pred) *chosen_pred = best_int_pred;
    auto t1_eval = std::chrono::high_resolution_clock::now();
    g_dp1d_prof.time_eval_interval_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_eval - t0_eval).count(),
        std::memory_order_relaxed);
    return best_int_cost;
  };

  float* dp = scratch->dp.data();
  std::fill(dp, dp + N, std::numeric_limits<float>::max());
  int32_t* opt_split = scratch->opt_split.data();
  std::fill(opt_split, opt_split + N, -1);
  size_t* pred_split = scratch->pred_split.data();

  auto val = [&](int32_t j, int32_t t) -> float {
    if (j > t) return std::numeric_limits<float>::max();
    float int_cost = eval_interval(j, t);
    if (j == 0) return int_cost;
    int32_t p_idx = j - 1;
    return dp[p_idx] + int_cost + split_penalty(p_idx);
  };

  struct Candidate {
    int32_t j;
    int32_t start_i;
  };
  std::deque<Candidate> dq;
  dq.push_back(Candidate{0, 0});

  for (int32_t i = 0; i < static_cast<int32_t>(N); i++) {
    while (dq.size() >= 2 && dq[1].start_i <= i) {
      dq.pop_front();
    }
    int32_t best_j = dq.front().j;
    size_t chosen_p = 0;
    float int_c = eval_interval(best_j, i, &chosen_p);
    dp[i] = (best_j == 0 ? int_c : dp[best_j - 1] + int_c + split_penalty(best_j - 1));
    opt_split[i] = (best_j == 0 ? -1 : best_j - 1);
    pred_split[i] = chosen_p;

    int32_t j_new = i + 1;
    if (j_new < static_cast<int32_t>(N)) {
      while (!dq.empty()) {
        int32_t j_back = dq.back().j;
        int32_t start_back = dq.back().start_i;
        if (start_back >= j_new) {
          if (val(j_new, start_back) <= val(j_back, start_back)) {
            dq.pop_back();
            continue;
          }
        }
        int32_t low = std::max(start_back + 1, j_new);
        if (low >= static_cast<int32_t>(N)) {
          break;
        }
        if (val(j_new, N - 1) > val(j_back, N - 1)) {
          break;
        }
        int32_t l = low;
        int32_t r = N - 1;
        int32_t t_star = N;
        while (l <= r) {
          int32_t mid = l + (r - l) / 2;
          if (val(j_new, mid) <= val(j_back, mid)) {
            t_star = mid;
            r = mid - 1;
          } else {
            l = mid + 1;
          }
        }
        if (t_star < static_cast<int32_t>(N)) {
          dq.push_back(Candidate{j_new, t_star});
        }
        break;
      }
      if (dq.empty()) {
        dq.push_back(Candidate{j_new, j_new});
      }
    }
  }

  if (max_exist == -1 || dp[max_exist] == std::numeric_limits<float>::max()) {
    auto t1_dp = std::chrono::high_resolution_clock::now();
    g_dp1d_prof.time_actual_dp_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_dp - t0_dp).count(),
        std::memory_order_relaxed);
    return res;
  }

  res.cost = dp[max_exist];
  int32_t curr = max_exist;
  while (curr != -1) {
    res.segment_preds.push_back(pred_split[curr]);
    int32_t prev = opt_split[curr];
    if (prev != -1) {
      res.cutoffs.push_back(valsB[prev]);
    }
    curr = prev;
  }
  std::reverse(res.cutoffs.begin(), res.cutoffs.end());
  std::reverse(res.segment_preds.begin(), res.segment_preds.end());
  auto t1_dp = std::chrono::high_resolution_clock::now();
  g_dp1d_prof.time_actual_dp_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1_dp - t0_dp).count(),
      std::memory_order_relaxed);
  return res;
}

DP1DResult Solve1DGreedyFromTables(
    TreeSamples& tree_samples, size_t prop_dim,
    const std::vector<int32_t>& valsB,
    const int64_t* interval_counts,
    const int64_t* interval_extra,
    const int32_t* interval_freq,
    const std::vector<size_t>& max_symbols,
    size_t num_predictors,
    size_t max_symbols_stride,
    float scale, float base_node_cost = 92.0f,
    float log_node_cost = 1.2f,
    size_t max_cuts = static_cast<size_t>(-1),
    Scratch1D* scratch = nullptr) {
  DP1DResult res;
  res.prop_dim = prop_dim;
  const size_t prop_idx = prop_dim + tree_samples.NumStaticProps();
  const size_t N = valsB.size();
  const size_t P = num_predictors;
  const size_t S = max_symbols_stride;
  if (N == 0) return res;

  auto t0_greedy = std::chrono::high_resolution_clock::now();
  Scratch1D local_scratch;
  if (!scratch) {
    scratch = &local_scratch;
  }
  scratch->Init(N, P, S);

  int32_t min_exist = -1;
  int32_t max_exist = -1;
  for (size_t v = 0; v < N; v++) {
    if (interval_counts[v] > 0) {
      if (min_exist == -1) min_exist = static_cast<int32_t>(v);
      max_exist = static_cast<int32_t>(v);
    }
  }

  if (min_exist == -1) {
    res.cost = 0.0f;
    res.segment_preds = {0};
    auto t1_greedy = std::chrono::high_resolution_clock::now();
    g_dp1d_prof.greedy_calls.fetch_add(1, std::memory_order_relaxed);
    g_dp1d_prof.time_greedy_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_greedy - t0_greedy).count(),
        std::memory_order_relaxed);
    return res;
  }

  if (min_exist >= max_exist) {
    float best_leaf_cost = std::numeric_limits<float>::max();
    size_t best_leaf_pred = 0;
    for (size_t pred = 0; pred < P; pred++) {
      const int32_t* hist = &interval_freq[(min_exist * P + pred) * S];
      int64_t extra = interval_extra[min_exist * P + pred];
      float bits = EstimateBits(hist, max_symbols[pred]) + extra;
      if (bits < best_leaf_cost) {
        best_leaf_cost = bits;
        best_leaf_pred = pred;
      }
    }
    res.cost = best_leaf_cost;
    res.segment_preds = {best_leaf_pred};
    auto t1_greedy = std::chrono::high_resolution_clock::now();
    g_dp1d_prof.greedy_calls.fetch_add(1, std::memory_order_relaxed);
    g_dp1d_prof.time_greedy_ns.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1_greedy - t0_greedy).count(),
        std::memory_order_relaxed);
    return res;
  }

  int64_t* pref_counts = scratch->pref_counts.data();
  pref_counts[0] = interval_counts[0];
  for (size_t v = 1; v < N; v++) {
    pref_counts[v] = pref_counts[v - 1] + interval_counts[v];
  }

  int64_t* pref_extra = scratch->pref_extra.data();
  for (size_t pred = 0; pred < P; pred++) {
    pref_extra[pred] = interval_extra[pred];
    for (size_t v = 1; v < N; v++) {
      pref_extra[v * P + pred] =
          pref_extra[(v - 1) * P + pred] + interval_extra[v * P + pred];
    }
  }

  int32_t* pref_freq = scratch->pref_freq.data();
  for (size_t pred = 0; pred < P; pred++) {
    for (size_t k = 0; k < max_symbols[pred]; k++) {
      pref_freq[pred * S + k] = interval_freq[pred * S + k];
    }
    for (size_t v = 1; v < N; v++) {
      for (size_t k = 0; k < max_symbols[pred]; k++) {
        pref_freq[(v * P + pred) * S + k] =
            pref_freq[((v - 1) * P + pred) * S + k] +
            interval_freq[(v * P + pred) * S + k];
      }
    }
  }

  auto split_penalty = [&](int32_t cutoff_idx) -> float {
    int32_t unquant = tree_samples.UnquantizeProperty(prop_idx, valsB[cutoff_idx]);
    return (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
  };

  float* memo_cost = scratch->memo_cost.data();
  size_t* memo_pred = scratch->memo_pred.data();
  std::fill(memo_cost, memo_cost + N * N, -1.0f);

  auto eval_interval = [&](int32_t j, int32_t i, size_t* chosen_pred = nullptr) -> float {
    size_t memo_idx = j * N + i;
    if (memo_cost[memo_idx] >= 0.0f) {
      if (chosen_pred) *chosen_pred = memo_pred[memo_idx];
      return memo_cost[memo_idx];
    }
    int64_t cnt = pref_counts[i] - (j > 0 ? pref_counts[j - 1] : 0);
    if (cnt == 0) {
      memo_cost[memo_idx] = 0.0f;
      memo_pred[memo_idx] = 0;
      if (chosen_pred) *chosen_pred = 0;
      return 0.0f;
    }
    float best_int_cost = std::numeric_limits<float>::max();
    size_t best_int_pred = 0;
    int32_t* rh = scratch->residual_hist.data();
    for (size_t pred = 0; pred < P; pred++) {
      const int32_t* pi = &pref_freq[(i * P + pred) * S];
      const int32_t* pj = (j > 0 ? &pref_freq[((j - 1) * P + pred) * S] : nullptr);
      if (pj) {
        for (size_t k = 0; k < max_symbols[pred]; k++) {
          rh[k] = pi[k] - pj[k];
        }
      } else {
        for (size_t k = 0; k < max_symbols[pred]; k++) {
          rh[k] = pi[k];
        }
      }
      float bits = EstimateBits(rh, max_symbols[pred]);
      int64_t extra = pref_extra[i * P + pred] -
                      (j > 0 ? pref_extra[(j - 1) * P + pred] : 0);
      float total = bits + extra;
      if (total < best_int_cost) {
        best_int_cost = total;
        best_int_pred = pred;
      }
    }
    memo_cost[memo_idx] = best_int_cost;
    memo_pred[memo_idx] = best_int_pred;
    if (chosen_pred) *chosen_pred = best_int_pred;
    return best_int_cost;
  };

  struct IntervalSegment {
    int32_t L;
    int32_t R;
    int32_t best_cut = -1;
    float best_cut_cost = std::numeric_limits<float>::max();
    float gain = 0.0f;
  };

  auto find_best_segment_split = [&](IntervalSegment& seg) {
    seg.best_cut = -1;
    seg.gain = 0.0f;
    if (seg.L >= seg.R) return;
    float base_cost = eval_interval(seg.L, seg.R);
    int64_t count_L = (seg.L > 0 ? pref_counts[seg.L - 1] : 0);
    int64_t count_R = pref_counts[seg.R];
    if (count_R <= count_L) return;

    float min_split_cost = base_cost;
    int32_t best_c = -1;
    for (int32_t c = seg.L; c < seg.R; c++) {
      int64_t count_c = pref_counts[c];
      if (count_c == count_L || count_c == count_R) continue;
      float cost = eval_interval(seg.L, c) + eval_interval(c + 1, seg.R) + split_penalty(c);
      if (cost < min_split_cost) {
        min_split_cost = cost;
        best_c = c;
      }
    }
    if (best_c != -1 && min_split_cost < base_cost) {
      seg.best_cut = best_c;
      seg.best_cut_cost = min_split_cost;
      seg.gain = base_cost - min_split_cost;
    }
  };

  std::vector<IntervalSegment> segments;
  IntervalSegment root_seg{min_exist, max_exist};
  find_best_segment_split(root_seg);
  segments.push_back(root_seg);

  while (segments.size() <= max_cuts) {
    size_t best_seg_idx = segments.size();
    float max_gain = 0.0f;
    for (size_t i = 0; i < segments.size(); i++) {
      if (segments[i].best_cut != -1 && segments[i].gain > max_gain) {
        max_gain = segments[i].gain;
        best_seg_idx = i;
      }
    }
    if (best_seg_idx >= segments.size() || max_gain <= 0.0f) {
      break;
    }

    IntervalSegment seg = segments[best_seg_idx];
    int32_t c = seg.best_cut;
    res.cut_history.push_back({valsB[c], max_gain});

    IntervalSegment left_seg{seg.L, c};
    IntervalSegment right_seg{c + 1, seg.R};
    find_best_segment_split(left_seg);
    find_best_segment_split(right_seg);

    segments[best_seg_idx] = left_seg;
    segments.push_back(right_seg);
  }

  std::sort(segments.begin(), segments.end(), [](const IntervalSegment& a, const IntervalSegment& b) {
    return a.L < b.L;
  });

  float total_cost = 0.0f;
  for (size_t i = 0; i < segments.size(); i++) {
    size_t chosen_pred = 0;
    float leaf_c = eval_interval(segments[i].L, segments[i].R, &chosen_pred);
    total_cost += leaf_c;
    res.segment_preds.push_back(chosen_pred);
    if (i + 1 < segments.size()) {
      int32_t cut_idx = segments[i].R;
      res.cutoffs.push_back(valsB[cut_idx]);
      total_cost += split_penalty(cut_idx);
    }
  }
  res.cost = total_cost;

  auto t1_greedy = std::chrono::high_resolution_clock::now();
  g_dp1d_prof.greedy_calls.fetch_add(1, std::memory_order_relaxed);
  g_dp1d_prof.time_greedy_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1_greedy - t0_greedy).count(),
      std::memory_order_relaxed);
  return res;
}

DP1DResult Run1DDPOnSamples(TreeSamples& tree_samples, size_t prop_dim,
                           const std::vector<uint32_t>* sample_subset,
                           float scale, float base_node_cost = 92.0f,
                           float log_node_cost = 1.2f,
                           bool use_greedy = false,
                           const std::vector<size_t>* max_symbols_override = nullptr,
                           size_t max_symbols_stride_override = 0) {
  DP1DResult res;
  res.prop_dim = prop_dim;
  size_t total_samples = sample_subset ? sample_subset->size()
                                       : tree_samples.NumDistinctSamples();
  if (total_samples == 0) return res;

  auto t0_samples = std::chrono::high_resolution_clock::now();

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  const size_t num_predictors = tree_samples.NumPredictors();
  const size_t P = num_predictors;

  std::bitset<256> seen;
  for (size_t i = 0; i < total_samples; i++) {
    seen.set(tree_samples.Property<false>(prop_dim, get_sample_idx(i)));
  }

  std::vector<int32_t> vals;
  vals.reserve(seen.count());
  std::array<int16_t, 256> lut;
  lut.fill(-1);
  for (size_t v = 0; v < 256; v++) {
    if (seen.test(v)) {
      lut[v] = static_cast<int16_t>(vals.size());
      vals.push_back(static_cast<int32_t>(v));
    }
  }

  auto t1_distinct = std::chrono::high_resolution_clock::now();

  const size_t N = vals.size();
  if (N == 0) return res;

  std::vector<size_t> max_symbols(P, 0);
  size_t max_symbols_stride = 0;
  if (max_symbols_override != nullptr && !max_symbols_override->empty() && max_symbols_stride_override > 0) {
    max_symbols = *max_symbols_override;
    max_symbols_stride = max_symbols_stride_override;
  } else {
    for (size_t pred = 0; pred < P; pred++) {
      for (size_t i = 0; i < total_samples; i++) {
        size_t s = get_sample_idx(i);
        uint32_t tok = tree_samples.Token(pred, s);
        max_symbols[pred] = std::max(max_symbols[pred], static_cast<size_t>(tok + 1));
      }
      max_symbols[pred] = Padded(max_symbols[pred]);
      max_symbols_stride = std::max(max_symbols_stride, max_symbols[pred]);
    }
  }
  const size_t S = max_symbols_stride;

  if (N == 1) {
    float best_leaf_cost = std::numeric_limits<float>::max();
    size_t best_leaf_pred = 0;
    std::vector<int32_t> hist(S, 0);
    for (size_t pred = 0; pred < P; pred++) {
      std::fill(hist.begin(), hist.end(), 0);
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
    res.segment_preds = {best_leaf_pred};
    return res;
  }

  auto get_u = [&](uint8_t v) -> size_t {
    return static_cast<size_t>(lut[v]);
  };

  auto t_post_max_sym = std::chrono::high_resolution_clock::now();

  std::vector<int64_t> interval_counts(N, 0);
  std::vector<int64_t> interval_extra(N * P, 0);
  std::vector<int32_t> interval_freq(N * P * S, 0);

  auto t_post_alloc = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    size_t u = get_u(tree_samples.Property<false>(prop_dim, s));
    size_t cnt = tree_samples.Count(s);
    interval_counts[u] += cnt;
    for (size_t pred = 0; pred < P; pred++) {
      interval_extra[u * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
      interval_freq[(u * P + pred) * S + tree_samples.Token(pred, s)] += cnt;
    }
  }

  auto t2_accum = std::chrono::high_resolution_clock::now();

  g_dp1d_prof.sample_accum_calls.fetch_add(1, std::memory_order_relaxed);
  g_dp1d_prof.total_samples.fetch_add(total_samples, std::memory_order_relaxed);
  g_dp1d_prof.total_N.fetch_add(N, std::memory_order_relaxed);
  g_dp1d_prof.time_distinct_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1_distinct - t0_samples).count(),
      std::memory_order_relaxed);
  g_dp1d_prof.time_accum_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2_accum - t1_distinct).count(),
      std::memory_order_relaxed);
  g_dp1d_prof.time_max_symbols_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_post_max_sym - t1_distinct).count(),
      std::memory_order_relaxed);
  g_dp1d_prof.time_alloc_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_post_alloc - t_post_max_sym).count(),
      std::memory_order_relaxed);
  g_dp1d_prof.time_scatter_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t2_accum - t_post_alloc).count(),
      std::memory_order_relaxed);

  if (use_greedy) {
    return Solve1DGreedyFromTables(
        tree_samples, prop_dim, vals,
        interval_counts.data(), interval_extra.data(), interval_freq.data(),
        max_symbols, P, S, scale, base_node_cost, log_node_cost);
  }

  return Solve1DDPFromTables(
      tree_samples, prop_dim, vals,
      interval_counts.data(), interval_extra.data(), interval_freq.data(),
      max_symbols, P, S, scale, base_node_cost, log_node_cost);
}

float Solve1DCostFromTables(
    TreeSamples& tree_samples, size_t prop_dim,
    const std::vector<int32_t>& valsB,
    const int64_t* interval_counts,
    const int64_t* interval_extra,
    const int32_t* interval_freq,
    const std::vector<size_t>& max_symbols,
    size_t num_predictors,
    size_t max_symbols_stride,
    float scale, float base_node_cost = 92.0f,
    float log_node_cost = 1.2f,
    Scratch1D* scratch = nullptr) {
  const char* env_greedy = getenv("JXL_INNER_DP_GREEDY");
  if (env_greedy && (strcmp(env_greedy, "0") == 0 || strcmp(env_greedy, "false") == 0)) {
    return Solve1DDPFromTables(
        tree_samples, prop_dim, valsB,
        interval_counts, interval_extra, interval_freq,
        max_symbols, num_predictors, max_symbols_stride,
        scale, base_node_cost, log_node_cost, scratch).cost;
  }
  return Solve1DGreedyFromTables(
      tree_samples, prop_dim, valsB,
      interval_counts, interval_extra, interval_freq,
      max_symbols, num_predictors, max_symbols_stride,
      scale, base_node_cost, log_node_cost, static_cast<size_t>(-1), scratch).cost;
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

struct MultiPropertyTables {
  std::vector<std::vector<int32_t>> vals;
  std::vector<std::array<int16_t, 256>> luts;
  std::vector<std::vector<int64_t>> interval_counts;
  std::vector<std::vector<int64_t>> interval_extra;
  std::vector<std::vector<int32_t>> interval_freq;
  std::vector<size_t> max_symbols;
  size_t S = 0;
};

inline MultiPropertyTables BuildMultiPropertyTables(
    TreeSamples& tree_samples,
    const std::vector<uint32_t>* sample_subset) {
  MultiPropertyTables mpt;
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  const size_t P = tree_samples.NumPredictors();
  if (num_props == 0 || total_samples == 0 || P == 0) return mpt;

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  auto t0_multi = std::chrono::high_resolution_clock::now();

  mpt.max_symbols.assign(P, 0);
  for (size_t pred = 0; pred < P; pred++) {
    size_t max_sym = 0;
    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      max_sym = std::max(max_sym, static_cast<size_t>(tree_samples.Token(pred, s) + 1));
    }
    mpt.max_symbols[pred] = Padded(max_sym);
    mpt.S = std::max(mpt.S, mpt.max_symbols[pred]);
  }
  const size_t S = mpt.S;

  mpt.vals.resize(num_props);
  mpt.luts.resize(num_props);
  mpt.interval_counts.resize(num_props);
  mpt.interval_extra.resize(num_props);
  mpt.interval_freq.resize(num_props);

  std::vector<std::bitset<256>> seen(num_props);
  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    for (size_t p = 0; p < num_props; p++) {
      seen[p].set(tree_samples.Property<false>(p, s));
    }
  }

  for (size_t p = 0; p < num_props; p++) {
    mpt.vals[p].reserve(seen[p].count());
    mpt.luts[p].fill(-1);
    for (size_t v = 0; v < 256; v++) {
      if (seen[p].test(v)) {
        mpt.luts[p][v] = static_cast<int16_t>(mpt.vals[p].size());
        mpt.vals[p].push_back(static_cast<int32_t>(v));
      }
    }
    size_t N_p = mpt.vals[p].size();
    mpt.interval_counts[p].assign(N_p, 0);
    mpt.interval_extra[p].assign(N_p * P, 0);
    mpt.interval_freq[p].assign(N_p * P * S, 0);
  }

  std::vector<uint32_t> cur_tok(P);
  std::vector<int64_t> cur_extra(P);

  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    size_t cnt = tree_samples.Count(s);
    for (size_t pred = 0; pred < P; pred++) {
      cur_tok[pred] = tree_samples.Token(pred, s);
      cur_extra[pred] = tree_samples.RTokens(pred)[s].nbits * cnt;
    }
    for (size_t p = 0; p < num_props; p++) {
      uint8_t val = tree_samples.Property<false>(p, s);
      size_t u = static_cast<size_t>(mpt.luts[p][val]);
      mpt.interval_counts[p][u] += cnt;
      int64_t* extra_p = &mpt.interval_extra[p][u * P];
      int32_t* freq_p = &mpt.interval_freq[p][(u * P) * S];
      for (size_t pred = 0; pred < P; pred++) {
        extra_p[pred] += cur_extra[pred];
        freq_p[pred * S + cur_tok[pred]] += cnt;
      }
    }
  }

  auto t1_multi = std::chrono::high_resolution_clock::now();
  g_dp1d_prof.time_multi_prop_ns.fetch_add(
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1_multi - t0_multi).count(),
      std::memory_order_relaxed);

  return mpt;
}

struct Initial1DPropertyScreening {
  std::vector<DP1DResult> primary_dps;
  std::vector<std::vector<int32_t>> coarse_cuts;
  size_t best_1d_dim = 0;
  float best_1d_cost = std::numeric_limits<float>::max();
  MultiPropertyTables mpt;
};

inline Initial1DPropertyScreening ScreenAllProperties1D(
    TreeSamples& tree_samples,
    const std::vector<uint32_t>* sample_subset,
    float scale, float base_node_cost = 92.0f, float log_node_cost = 1.2f,
    size_t coarse_max_cuts = 4, bool refine_with_dp = true) {
  Initial1DPropertyScreening res;
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  if (num_props == 0) return res;

  MultiPropertyTables mpt = BuildMultiPropertyTables(tree_samples, sample_subset);
  res.primary_dps.resize(num_props);
  res.coarse_cuts.resize(num_props);

  size_t greedy_cuts = std::max<size_t>(coarse_max_cuts, 63);
  for (size_t p = 0; p < num_props; p++) {
    res.primary_dps[p] = Solve1DGreedyFromTables(
        tree_samples, p, mpt.vals[p],
        mpt.interval_counts[p].data(), mpt.interval_extra[p].data(), mpt.interval_freq[p].data(),
        mpt.max_symbols, tree_samples.NumPredictors(), mpt.S,
        scale, base_node_cost, log_node_cost, greedy_cuts);
    res.coarse_cuts[p].clear();
    size_t n_cuts = std::min(coarse_max_cuts, res.primary_dps[p].cut_history.size());
    for (size_t i = 0; i < n_cuts; i++) {
      res.coarse_cuts[p].push_back(res.primary_dps[p].cut_history[i].cutoff_val);
    }
    std::sort(res.coarse_cuts[p].begin(), res.coarse_cuts[p].end());
  }

  std::vector<size_t> ranked(num_props);
  for (size_t i = 0; i < num_props; i++) ranked[i] = i;
  std::sort(ranked.begin(), ranked.end(), [&](size_t a, size_t b) {
    return res.primary_dps[a].cost < res.primary_dps[b].cost;
  });

  if (refine_with_dp) {
    size_t to_refine = std::min<size_t>(num_props, 3);
    for (size_t i = 0; i < to_refine; i++) {
      size_t p = ranked[i];
      res.primary_dps[p] = Solve1DDPFromTables(
          tree_samples, p, mpt.vals[p],
          mpt.interval_counts[p].data(),
          mpt.interval_extra[p].data(),
          mpt.interval_freq[p].data(),
          mpt.max_symbols, tree_samples.NumPredictors(), mpt.S,
          scale, base_node_cost, log_node_cost);
    }
  }

  res.best_1d_dim = ranked[0];
  res.best_1d_cost = res.primary_dps[ranked[0]].cost;
  if (refine_with_dp) {
    size_t to_refine = std::min<size_t>(num_props, 3);
    for (size_t i = 1; i < to_refine; i++) {
      if (res.primary_dps[ranked[i]].cost < res.best_1d_cost) {
        res.best_1d_cost = res.primary_dps[ranked[i]].cost;
        res.best_1d_dim = ranked[i];
      }
    }
  }
  res.mpt = std::move(mpt);
  return res;
}

void FindBestTree1dDP(TreeSamples& tree_samples, float scale, Tree* tree,
                      const std::vector<uint32_t>* sample_subset = nullptr,
                      float base_node_cost = 92.0f, float log_node_cost = 1.2f) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props == 0 || total_samples == 0) {
    (*tree)[0] = PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0));
    return;
  }

  auto screen = ScreenAllProperties1D(tree_samples, sample_subset, scale, base_node_cost, log_node_cost);
  const auto& best_res = screen.primary_dps[screen.best_1d_dim];

  if (best_res.segment_preds.empty()) {
    (*tree)[0] = PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0));
    return;
  }

  BuildTreeFrom1D(best_res.prop_dim + tree_samples.NumStaticProps(),
                  best_res.cutoffs, best_res.segment_preds, tree_samples, tree, 0);
}

struct ScoredPair {
  size_t pA = 0;
  size_t pB = 0;
  float cost = std::numeric_limits<float>::max();
};

struct ScoredTriple {
  size_t pA = 0;
  size_t pB = 0;
  size_t pC = 0;
  float cost = std::numeric_limits<float>::max();
};

inline std::vector<int32_t> SelectCoarseCutoffs(const std::vector<int32_t>& cutoffs, size_t max_cuts = 4) {
  if (cutoffs.empty()) return {};
  if (cutoffs.size() <= max_cuts) return cutoffs;
  std::vector<int32_t> res;
  res.reserve(max_cuts);
  for (size_t i = 0; i < max_cuts; i++) {
    size_t idx = i * (cutoffs.size() - 1) / (max_cuts - 1);
    res.push_back(cutoffs[idx]);
  }
  res.erase(std::unique(res.begin(), res.end()), res.end());
  return res;
}

inline std::vector<int32_t> GetCoarseQuantileCuts(
    TreeSamples& tree_samples, size_t prop,
    const std::vector<uint32_t>* sample_subset,
    size_t max_cuts = 4) {
  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (total_samples == 0) return {};
  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  int32_t min_v = std::numeric_limits<int32_t>::max();
  int32_t max_v = std::numeric_limits<int32_t>::min();
  for (size_t i = 0; i < total_samples; i++) {
    int32_t v = tree_samples.Property<false>(prop, get_sample_idx(i));
    if (v < min_v) min_v = v;
    if (v > max_v) max_v = v;
  }
  if (min_v >= max_v) return {};

  uint64_t range = static_cast<uint64_t>(max_v) - min_v;
  std::vector<int32_t> unique_vals;
  if (range <= 8192) {
    size_t words = (range + 64) / 64;
    std::vector<uint64_t> seen(words, 0);
    for (size_t i = 0; i < total_samples; i++) {
      uint32_t offset = static_cast<uint32_t>(tree_samples.Property<false>(prop, get_sample_idx(i)) - min_v);
      seen[offset / 64] |= (1ULL << (offset % 64));
    }
    for (size_t w = 0; w < words; w++) {
      uint64_t mask = seen[w];
      while (mask != 0) {
        int bit = __builtin_ctzll(mask);
        unique_vals.push_back(min_v + static_cast<int32_t>(w * 64 + bit));
        mask &= mask - 1;
      }
    }
  } else {
    std::vector<int32_t> s_vals;
    size_t step = std::max<size_t>(1, total_samples / 256);
    for (size_t i = 0; i < total_samples; i += step) {
      s_vals.push_back(tree_samples.Property<false>(prop, get_sample_idx(i)));
    }
    std::sort(s_vals.begin(), s_vals.end());
    s_vals.erase(std::unique(s_vals.begin(), s_vals.end()), s_vals.end());
    unique_vals = std::move(s_vals);
  }

  size_t prop_idx = prop + tree_samples.NumStaticProps();
  size_t max_valid_cutoff = tree_samples.NumPropertyValues(prop_idx) > 1 ? tree_samples.NumPropertyValues(prop_idx) - 1 : 0;
  while (!unique_vals.empty() && static_cast<size_t>(unique_vals.back()) >= max_valid_cutoff) {
    unique_vals.pop_back();
  }
  if (unique_vals.empty()) return {};

  if (unique_vals.size() <= max_cuts) {
    return unique_vals;
  }
  std::vector<int32_t> res;
  res.reserve(max_cuts);
  for (size_t k = 0; k < max_cuts; k++) {
    size_t idx = k * (unique_vals.size() - 1) / (max_cuts - 1);
    res.push_back(unique_vals[idx]);
  }
  res.erase(std::unique(res.begin(), res.end()), res.end());
  return res;
}

std::vector<ScoredPair> FastCoarseGridPairScreening(
    TreeSamples& tree_samples,
    const std::vector<uint32_t>* sample_subset,
    const std::vector<std::vector<int32_t>>& coarse_cuts,
    float scale, float base_node_cost, float log_node_cost,
    size_t max_pairs_to_return = 12,
    std::vector<float>* out_prop_scores = nullptr) {
  const size_t num_props = coarse_cuts.size();
  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props < 2 || total_samples == 0) return {};

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  std::vector<float> cut_penalties(num_props, 0.0f);
  for (size_t p = 0; p < num_props; p++) {
    for (int32_t c : coarse_cuts[p]) {
      int32_t unquant = tree_samples.UnquantizeProperty(p + tree_samples.NumStaticProps(), c);
      cut_penalties[p] += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
    }
  }

  std::vector<std::vector<uint8_t>> sample_bins(num_props, std::vector<uint8_t>(total_samples));
  for (size_t p = 0; p < num_props; p++) {
    const auto& cuts = coarse_cuts[p];
    if (cuts.empty()) {
      std::fill(sample_bins[p].begin(), sample_bins[p].end(), 0);
      continue;
    }
    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      int32_t v = tree_samples.Property<false>(p, s);
      size_t b = 0;
      while (b < cuts.size() && v > cuts[b]) {
        b++;
      }
      sample_bins[p][i] = static_cast<uint8_t>(b);
    }
  }

  std::vector<size_t> candidate_preds;
  for (size_t pred = 0; pred < tree_samples.NumPredictors(); pred++) {
    candidate_preds.push_back(pred);
  }
  if (candidate_preds.empty()) candidate_preds.push_back(0);

  const size_t kPaddedSyms = Padded(32);
  const size_t max_cells = 5 * 5;
  const size_t max_cell_preds = max_cells * candidate_preds.size();
  std::vector<int32_t> flat_cell_hists(max_cell_preds * kPaddedSyms);
  std::vector<int64_t> flat_cell_extra(max_cell_preds);
  std::vector<size_t> flat_cell_sample_count(max_cells);

  if (out_prop_scores) {
    out_prop_scores->assign(num_props, 0.0f);
    for (size_t p = 0; p < num_props; p++) {
      const size_t num_bins = coarse_cuts[p].size() + 1;
      const size_t num_bin_preds = num_bins * candidate_preds.size();
      std::fill(flat_cell_sample_count.begin(), flat_cell_sample_count.begin() + num_bins, 0);
      std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_bin_preds, 0);
      std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_bin_preds * kPaddedSyms, 0);

      for (size_t i = 0; i < total_samples; i++) {
        size_t s = get_sample_idx(i);
        size_t b = sample_bins[p][i];
        flat_cell_sample_count[b]++;
        size_t cnt = tree_samples.Count(s);
        for (size_t pi = 0; pi < candidate_preds.size(); pi++) {
          size_t pred = candidate_preds[pi];
          uint32_t tok = tree_samples.Token(pred, s);
          size_t b_pi = b * candidate_preds.size() + pi;
          if (tok < kPaddedSyms) {
            flat_cell_hists[b_pi * kPaddedSyms + tok] += cnt;
          } else {
            flat_cell_hists[b_pi * kPaddedSyms + kPaddedSyms - 1] += cnt;
          }
          flat_cell_extra[b_pi] += tree_samples.RTokens(pred)[s].nbits * cnt;
        }
      }

      float prop_entropy = 0.0f;
      for (size_t b = 0; b < num_bins; b++) {
        if (flat_cell_sample_count[b] == 0) continue;
        float best_bits = std::numeric_limits<float>::max();
        for (size_t pi = 0; pi < candidate_preds.size(); pi++) {
          size_t b_pi = b * candidate_preds.size() + pi;
          float bits = EstimateBits(&flat_cell_hists[b_pi * kPaddedSyms], kPaddedSyms) + flat_cell_extra[b_pi];
          if (bits < best_bits) best_bits = bits;
        }
        prop_entropy += best_bits;
      }
      (*out_prop_scores)[p] = prop_entropy + cut_penalties[p];
    }
  }

  std::vector<ScoredPair> scored;
  scored.reserve(num_props * (num_props - 1) / 2);

  for (size_t pA = 0; pA < num_props; pA++) {
    const size_t num_binsA = coarse_cuts[pA].size() + 1;
    for (size_t pB = pA + 1; pB < num_props; pB++) {
      const size_t num_binsB = coarse_cuts[pB].size() + 1;
      const size_t num_cells = num_binsA * num_binsB;
      const size_t num_cell_preds = num_cells * candidate_preds.size();

      std::fill(flat_cell_sample_count.begin(), flat_cell_sample_count.begin() + num_cells, 0);
      std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_cell_preds, 0);
      std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_cell_preds * kPaddedSyms, 0);

      for (size_t i = 0; i < total_samples; i++) {
        size_t s = get_sample_idx(i);
        size_t bA = sample_bins[pA][i];
        size_t bB = sample_bins[pB][i];
        size_t cell = bA * num_binsB + bB;
        flat_cell_sample_count[cell]++;
        size_t cnt = tree_samples.Count(s);

        for (size_t pi = 0; pi < candidate_preds.size(); pi++) {
          size_t pred = candidate_preds[pi];
          uint32_t tok = tree_samples.Token(pred, s);
          size_t cell_pi = cell * candidate_preds.size() + pi;
          if (tok < kPaddedSyms) {
            flat_cell_hists[cell_pi * kPaddedSyms + tok] += cnt;
          } else {
            flat_cell_hists[cell_pi * kPaddedSyms + kPaddedSyms - 1] += cnt;
          }
          flat_cell_extra[cell_pi] += tree_samples.RTokens(pred)[s].nbits * cnt;
        }
      }

      float pair_entropy = 0.0f;
      for (size_t c = 0; c < num_cells; c++) {
        if (flat_cell_sample_count[c] == 0) continue;
        float best_cell_bits = std::numeric_limits<float>::max();
        for (size_t pi = 0; pi < candidate_preds.size(); pi++) {
          size_t cell_pi = c * candidate_preds.size() + pi;
          float bits = EstimateBits(&flat_cell_hists[cell_pi * kPaddedSyms], kPaddedSyms) + flat_cell_extra[cell_pi];
          if (bits < best_cell_bits) {
            best_cell_bits = bits;
          }
        }
        pair_entropy += best_cell_bits;
      }

      float total_pair_cost = pair_entropy + cut_penalties[pA] + cut_penalties[pB];
      scored.push_back(ScoredPair{pA, pB, total_pair_cost});
    }
  }

  std::sort(scored.begin(), scored.end(), [](const ScoredPair& a, const ScoredPair& b) {
    return a.cost < b.cost;
  });

  if (scored.size() > max_pairs_to_return) {
    scored.resize(max_pairs_to_return);
  }
  return scored;
}

inline std::vector<ScoredPair> FastCoarseGridPairScreening(
    TreeSamples& tree_samples,
    const std::vector<uint32_t>* sample_subset,
    const std::vector<DP1DResult>& primary_dps,
    float scale, float base_node_cost, float log_node_cost,
    size_t max_pairs_to_return = 12) {
  const size_t num_props = primary_dps.size();
  std::vector<std::vector<int32_t>> coarse_cuts(num_props);
  for (size_t p = 0; p < num_props; p++) {
    coarse_cuts[p] = SelectCoarseCutoffs(primary_dps[p].cutoffs, 4);
  }
  return FastCoarseGridPairScreening(tree_samples, sample_subset, coarse_cuts,
                                     scale, base_node_cost, log_node_cost, max_pairs_to_return);
}

struct EvaluatedPairTree {
  size_t pA = 0;
  size_t pB = 0;
  float cost = std::numeric_limits<float>::max();
  DP1DResult dpA;
  std::vector<DP1DResult> refinementsB;
};

struct ScratchPairEvaluation {
  std::vector<int64_t> seg_counts;
  std::vector<int64_t> seg_extra;
  std::vector<int32_t> seg_freq;
  std::vector<int64_t> seg_sample_count;
  std::vector<int32_t> leaf_hist;
  std::vector<int32_t> sub_vals;
  std::vector<int64_t> sub_counts;
  std::vector<int64_t> sub_extra;
  std::vector<int32_t> sub_freq;
  Scratch1D scratch1D;

  void Init(size_t num_segs, size_t N_B, size_t P, size_t S) {
    size_t sz_counts = num_segs * N_B;
    size_t sz_extra = num_segs * N_B * P;
    size_t sz_freq = num_segs * N_B * P * S;
    if (seg_counts.size() < sz_counts) seg_counts.resize(sz_counts);
    if (seg_extra.size() < sz_extra) seg_extra.resize(sz_extra);
    if (seg_freq.size() < sz_freq) seg_freq.resize(sz_freq);
    if (seg_sample_count.size() < num_segs) seg_sample_count.resize(num_segs);
    if (leaf_hist.size() < S) leaf_hist.resize(S);

    std::fill(seg_counts.begin(), seg_counts.begin() + sz_counts, 0);
    std::fill(seg_extra.begin(), seg_extra.begin() + sz_extra, 0);
    std::fill(seg_freq.begin(), seg_freq.begin() + sz_freq, 0);
    std::fill(seg_sample_count.begin(), seg_sample_count.begin() + num_segs, 0);
  }
};

EvaluatedPairTree Evaluate2PropRecursiveTree(
    TreeSamples& tree_samples,
    size_t pA, size_t pB,
    const DP1DResult& dpA,
    const std::vector<uint32_t>* sample_subset,
    float scale, float base_node_cost, float log_node_cost,
    bool use_greedy_refinement = false,
    const MultiPropertyTables* mpt = nullptr,
    ScratchPairEvaluation* scratch = nullptr) {
  EvaluatedPairTree res;
  res.pA = pA;
  res.pB = pB;
  res.dpA = dpA;

  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  if (dpA.cutoffs.empty()) {
    res.cost = dpA.cost;
    return res;
  }

  const size_t num_cuts = dpA.cutoffs.size();
  const size_t num_segs = num_cuts + 1;
  const size_t P = tree_samples.NumPredictors();

  if (mpt != nullptr && pB < mpt->vals.size() && pB < mpt->luts.size()) {
    const auto& valsB = mpt->vals[pB];
    const auto& lutB = mpt->luts[pB];
    const size_t N_B = valsB.size();
    const auto& max_symbols = mpt->max_symbols;
    const size_t S = mpt->S;

    if (N_B == 0) {
      res.cost = dpA.cost;
      return res;
    }

    ScratchPairEvaluation local_scratch;
    if (!scratch) scratch = &local_scratch;
    scratch->Init(num_segs, N_B, P, S);

    int64_t* seg_counts = scratch->seg_counts.data();
    int64_t* seg_extra = scratch->seg_extra.data();
    int32_t* seg_freq = scratch->seg_freq.data();
    int64_t* seg_sample_count = scratch->seg_sample_count.data();

    // Single streaming pass over all samples
    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      int32_t vA = tree_samples.Property<false>(pA, s);
      size_t seg = 0;
      while (seg < num_cuts && vA > dpA.cutoffs[seg]) {
        seg++;
      }
      seg_sample_count[seg]++;

      uint8_t vB = tree_samples.Property<false>(pB, s);
      int16_t uB_raw = lutB[vB];
      if (uB_raw < 0) continue;
      size_t uB = static_cast<size_t>(uB_raw);
      size_t cnt = tree_samples.Count(s);

      size_t seg_offset = seg * N_B + uB;
      seg_counts[seg_offset] += cnt;
      for (size_t pred = 0; pred < P; pred++) {
        uint32_t tok = tree_samples.Token(pred, s);
        int64_t ext = tree_samples.RTokens(pred)[s].nbits * cnt;
        seg_extra[seg_offset * P + pred] += ext;
        seg_freq[(seg_offset * P + pred) * S + tok] += cnt;
      }
    }

    float cutoffsA_penalty = 0.0f;
    for (int32_t c : dpA.cutoffs) {
      int32_t unquant = tree_samples.UnquantizeProperty(pA + tree_samples.NumStaticProps(), c);
      cutoffsA_penalty += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
    }

    float pair_cost = cutoffsA_penalty;
    std::vector<DP1DResult> refs(num_segs);
    int32_t* leaf_hist = scratch->leaf_hist.data();

    for (size_t seg = 0; seg < num_segs; seg++) {
      size_t base_pred = (seg < dpA.segment_preds.size()) ? dpA.segment_preds[seg] : 0;
      if (seg_sample_count[seg] == 0) {
        refs[seg].cost = 0.0f;
        refs[seg].segment_preds = {base_pred};
        continue;
      }

      std::fill(leaf_hist, leaf_hist + S, 0);
      int64_t leaf_extra = 0;
      for (size_t uB = 0; uB < N_B; uB++) {
        size_t seg_offset = seg * N_B + uB;
        leaf_extra += seg_extra[seg_offset * P + base_pred];
        const int32_t* f = &seg_freq[(seg_offset * P + base_pred) * S];
        for (size_t k = 0; k < max_symbols[base_pred]; k++) {
          leaf_hist[k] += f[k];
        }
      }
      float seg_base_cost = EstimateBits(leaf_hist, max_symbols[base_pred]) + leaf_extra;

      if (seg_sample_count[seg] < 32) {
        pair_cost += seg_base_cost;
        refs[seg].cost = seg_base_cost;
        refs[seg].segment_preds = {base_pred};
        continue;
      }

      size_t sub_N = 0;
      for (size_t uB = 0; uB < N_B; uB++) {
        if (seg_counts[seg * N_B + uB] > 0) sub_N++;
      }
      if (sub_N <= 1) {
        pair_cost += seg_base_cost;
        refs[seg].cost = seg_base_cost;
        refs[seg].segment_preds = {base_pred};
        continue;
      }

      scratch->sub_vals.resize(sub_N);
      scratch->sub_counts.resize(sub_N);
      scratch->sub_extra.resize(sub_N * P);
      scratch->sub_freq.resize(sub_N * P * S);

      int32_t* sub_vals = scratch->sub_vals.data();
      int64_t* sub_counts = scratch->sub_counts.data();
      int64_t* sub_extra = scratch->sub_extra.data();
      int32_t* sub_freq = scratch->sub_freq.data();

      size_t dst = 0;
      for (size_t uB = 0; uB < N_B; uB++) {
        size_t seg_offset = seg * N_B + uB;
        if (seg_counts[seg_offset] > 0) {
          sub_vals[dst] = valsB[uB];
          sub_counts[dst] = seg_counts[seg_offset];
          memcpy(&sub_extra[dst * P], &seg_extra[seg_offset * P], P * sizeof(int64_t));
          memcpy(&sub_freq[(dst * P) * S], &seg_freq[seg_offset * P * S], P * S * sizeof(int32_t));
          dst++;
        }
      }

      DP1DResult refB;
      if (use_greedy_refinement) {
        refB = Solve1DGreedyFromTables(
            tree_samples, pB, scratch->sub_vals,
            sub_counts, sub_extra, sub_freq,
            max_symbols, P, S, scale, base_node_cost, log_node_cost,
            static_cast<size_t>(-1), &scratch->scratch1D);
      } else {
        refB = Solve1DDPFromTables(
            tree_samples, pB, scratch->sub_vals,
            sub_counts, sub_extra, sub_freq,
            max_symbols, P, S, scale, base_node_cost, log_node_cost,
            &scratch->scratch1D);
      }

      if (refB.cost < seg_base_cost && !refB.cutoffs.empty()) {
        pair_cost += refB.cost;
        refs[seg] = std::move(refB);
      } else {
        pair_cost += seg_base_cost;
        refs[seg].cost = seg_base_cost;
        refs[seg].segment_preds = {base_pred};
      }
    }

    res.cost = pair_cost;
    res.refinementsB = std::move(refs);
    return res;
  }

  std::vector<std::vector<uint32_t>> seg_samples(num_segs);
  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    int32_t v = tree_samples.Property<false>(pA, s);
    size_t seg = std::upper_bound(dpA.cutoffs.begin(), dpA.cutoffs.end(), v) - dpA.cutoffs.begin();
    seg_samples[seg].push_back(s);
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
    int32_t unquant = tree_samples.UnquantizeProperty(pA + tree_samples.NumStaticProps(), c);
    cutoffsA_penalty += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
  }

  float pair_cost = cutoffsA_penalty;
  std::vector<DP1DResult> refs(num_segs);

  for (size_t s = 0; s < num_segs; s++) {
    if (seg_samples[s].size() < 32) {
      pair_cost += seg_base_cost[s];
      refs[s].cost = seg_base_cost[s];
      refs[s].segment_preds = {dpA.segment_preds[s]};
    } else {
      DP1DResult refB = Run1DDPOnSamples(tree_samples, pB, &seg_samples[s], scale,
                                         base_node_cost, log_node_cost, use_greedy_refinement);
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

  res.cost = pair_cost;
  res.refinementsB = std::move(refs);
  return res;
}

void FindBestTree2PropDP(TreeSamples& tree_samples, float scale, Tree* tree,
                         const std::vector<uint32_t>* sample_subset = nullptr,
                         float base_node_cost = 92.0f, float log_node_cost = 1.2f) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props == 0 || total_samples == 0) {
    (*tree)[0] = PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0));
    return;
  }
  if (num_props == 1) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  auto screen = ScreenAllProperties1D(tree_samples, sample_subset, scale, base_node_cost, log_node_cost, 4);
  auto primary_dps = std::move(screen.primary_dps);
  const auto& coarse_cuts = screen.coarse_cuts;

  // Determine candidate pairs to test using coarse-grid screening
  size_t max_pairs = 4;
  if (num_props > 12) max_pairs = 20;
  else if (num_props > 8) max_pairs = 12;
  else if (num_props > 5) max_pairs = 8;
  else if (num_props > 4) max_pairs = 6;

  const char* env_max_p = getenv("JXL_DP_MAX_PAIRS");
  if (env_max_p != nullptr) {
    max_pairs = atoi(env_max_p);
  }

  std::vector<ScoredPair> top_pairs = FastCoarseGridPairScreening(
      tree_samples, sample_subset, coarse_cuts, scale, base_node_cost, log_node_cost,
      max_pairs);

  std::vector<size_t> ranked_1d(num_props);
  for (size_t i = 0; i < num_props; i++) ranked_1d[i] = i;
  std::sort(ranked_1d.begin(), ranked_1d.end(), [&](size_t a, size_t b) {
    return primary_dps[a].cost < primary_dps[b].cost;
  });
  if (num_props >= 2) {
    top_pairs.push_back(ScoredPair{ranked_1d[0], ranked_1d[1], 0.0f});
  }

  size_t best_1d_dim = ranked_1d[0];
  float best_1d_cost = primary_dps[best_1d_dim].cost;

  EvaluatedPairTree best_pair;
  best_pair.cost = best_1d_cost;

  ScratchPairEvaluation scratch_pair;
  std::set<std::pair<size_t, size_t>> evaluated;
  for (const auto& sp : top_pairs) {
    if (evaluated.insert({sp.pA, sp.pB}).second) {
      auto resAB = Evaluate2PropRecursiveTree(tree_samples, sp.pA, sp.pB, primary_dps[sp.pA],
                                             sample_subset, scale, base_node_cost, log_node_cost,
                                             /*use_greedy_refinement=*/true, &screen.mpt, &scratch_pair);
      if (resAB.cost < best_pair.cost) best_pair = std::move(resAB);
    }
    if (evaluated.insert({sp.pB, sp.pA}).second) {
      auto resBA = Evaluate2PropRecursiveTree(tree_samples, sp.pB, sp.pA, primary_dps[sp.pB],
                                             sample_subset, scale, base_node_cost, log_node_cost,
                                             /*use_greedy_refinement=*/true, &screen.mpt, &scratch_pair);
      if (resBA.cost < best_pair.cost) best_pair = std::move(resBA);
    }
  }

  if (!best_pair.refinementsB.empty() && best_pair.cost < best_1d_cost) {
    best_pair = Evaluate2PropRecursiveTree(tree_samples, best_pair.pA, best_pair.pB,
                                           primary_dps[best_pair.pA], sample_subset, scale,
                                           base_node_cost, log_node_cost,
                                           /*use_greedy_refinement=*/false, &screen.mpt, &scratch_pair);
  }

  if (best_pair.refinementsB.empty() || best_pair.cost >= best_1d_cost) {
    const auto& b1d = primary_dps[best_1d_dim];
    if (b1d.segment_preds.empty()) {
      (*tree)[0] = PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0));
      return;
    }
    BuildTreeFrom1D(b1d.prop_dim + tree_samples.NumStaticProps(),
                    b1d.cutoffs, b1d.segment_preds, tree_samples, tree, 0);
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

std::vector<ScoredTriple> FastCoarseGridTripleScreening(
    TreeSamples& tree_samples,
    const std::vector<uint32_t>* sample_subset,
    const std::vector<std::vector<int32_t>>& coarse_cuts,
    float scale, float base_node_cost, float log_node_cost,
    size_t max_triples_to_return = 12) {
  const size_t num_props = coarse_cuts.size();
  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props < 3 || total_samples == 0) return {};

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  std::vector<float> cut_penalties(num_props, 0.0f);
  for (size_t p = 0; p < num_props; p++) {
    for (int32_t c : coarse_cuts[p]) {
      int32_t unquant = tree_samples.UnquantizeProperty(p + tree_samples.NumStaticProps(), c);
      cut_penalties[p] += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
    }
  }

  size_t stride = 1;
  const char* env_stride = getenv("JXL_SCREENING_STRIDE");
  if (env_stride != nullptr) {
    stride = std::max<size_t>(1, atoi(env_stride));
  }

  size_t num_screened_samples = (total_samples + stride - 1) / stride;
  std::vector<size_t> screened_indices(num_screened_samples);
  for (size_t si = 0, i = 0; i < total_samples; i += stride, si++) {
    screened_indices[si] = get_sample_idx(i);
  }

  std::vector<std::vector<uint8_t>> sample_bins(num_props, std::vector<uint8_t>(num_screened_samples));
  for (size_t p = 0; p < num_props; p++) {
    const auto& cuts = coarse_cuts[p];
    if (cuts.empty()) {
      std::fill(sample_bins[p].begin(), sample_bins[p].end(), 0);
      continue;
    }
    for (size_t si = 0; si < num_screened_samples; si++) {
      size_t s = screened_indices[si];
      int32_t v = tree_samples.Property<false>(p, s);
      size_t b = 0;
      while (b < cuts.size() && v > cuts[b]) {
        b++;
      }
      sample_bins[p][si] = static_cast<uint8_t>(b);
    }
  }

  std::vector<size_t> candidate_preds;
  for (size_t pred = 0; pred < tree_samples.NumPredictors(); pred++) {
    candidate_preds.push_back(pred);
  }
  if (candidate_preds.empty()) candidate_preds.push_back(0);

  const size_t kPaddedSyms = Padded(32);
  std::vector<ScoredTriple> scored;

  struct TripleKey {
    size_t a, b, c;
  };

  std::vector<TripleKey> candidate_triples;
  for (size_t pA = 0; pA < num_props; pA++) {
    for (size_t pB = pA + 1; pB < num_props; pB++) {
      for (size_t pC = pB + 1; pC < num_props; pC++) {
        candidate_triples.push_back(TripleKey{pA, pB, pC});
      }
    }
  }

  std::vector<int32_t> flat_cell_hists;
  std::vector<int64_t> flat_cell_extra;
  std::vector<size_t> cell_sample_count;

  for (const auto& trip : candidate_triples) {
    size_t qA = trip.a, qB = trip.b, qC = trip.c;
    size_t num_binsA = coarse_cuts[qA].size() + 1;
    size_t num_binsB = coarse_cuts[qB].size() + 1;
    size_t num_binsC = coarse_cuts[qC].size() + 1;
    size_t num_cells = num_binsA * num_binsB * num_binsC;
    size_t num_entries = num_cells * candidate_preds.size();

    if (flat_cell_hists.size() < num_entries * kPaddedSyms) {
      flat_cell_hists.resize(num_entries * kPaddedSyms);
    }
    if (flat_cell_extra.size() < num_entries) {
      flat_cell_extra.resize(num_entries);
    }
    if (cell_sample_count.size() < num_cells) {
      cell_sample_count.resize(num_cells);
    }

    std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * kPaddedSyms, 0);
    std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
    std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

    for (size_t si = 0; si < num_screened_samples; si++) {
      size_t s = screened_indices[si];
      size_t bA = sample_bins[qA][si];
      size_t bB = sample_bins[qB][si];
      size_t bC = sample_bins[qC][si];
      size_t cell = (bA * num_binsB + bB) * num_binsC + bC;
      cell_sample_count[cell]++;
      size_t cnt = tree_samples.Count(s);

      for (size_t pidx = 0; pidx < candidate_preds.size(); pidx++) {
        size_t pred = candidate_preds[pidx];
        uint32_t tok = tree_samples.Token(pred, s);
        size_t cell_pi = cell * candidate_preds.size() + pidx;
        size_t offset = cell_pi * kPaddedSyms;
        if (tok < kPaddedSyms) {
          flat_cell_hists[offset + tok] += cnt;
        } else {
          flat_cell_hists[offset + kPaddedSyms - 1] += cnt;
        }
        flat_cell_extra[cell_pi] += tree_samples.RTokens(pred)[s].nbits * cnt;
      }
    }

    float triple_entropy = 0.0f;
    for (size_t c = 0; c < num_cells; c++) {
      if (cell_sample_count[c] == 0) continue;
      float best_cell_bits = std::numeric_limits<float>::max();
      for (size_t pidx = 0; pidx < candidate_preds.size(); pidx++) {
        size_t cell_pi = c * candidate_preds.size() + pidx;
        float bits = EstimateBits(&flat_cell_hists[cell_pi * kPaddedSyms], kPaddedSyms) +
                     flat_cell_extra[cell_pi];
        if (bits < best_cell_bits) best_cell_bits = bits;
      }
      triple_entropy += best_cell_bits;
    }

    float total_triple_cost = triple_entropy * stride + cut_penalties[qA] + cut_penalties[qB] + cut_penalties[qC];
    scored.push_back(ScoredTriple{qA, qB, qC, total_triple_cost});
  }

  std::sort(scored.begin(), scored.end(), [](const ScoredTriple& a, const ScoredTriple& b) {
    return a.cost < b.cost;
  });

  if (scored.size() > max_triples_to_return) {
    scored.resize(max_triples_to_return);
  }
  return scored;
}


struct NestedDPResult {
  float cost = std::numeric_limits<float>::max();
  size_t pA = 0;
  size_t pB = 1;
  std::vector<int32_t> cutoffsA;
  std::vector<DP1DResult> segment_refinements;
};

struct Scratch2D {
  std::vector<int64_t> interval_counts;
  std::vector<int64_t> interval_extra;
  std::vector<int32_t> interval_freq;
  std::vector<int32_t> leaf_hist;
  std::vector<int64_t> leaf_extra;
  std::vector<float> memo_cost;
  std::vector<float> dp;
  std::vector<int32_t> opt_prev;
  Scratch1D scratch1D;

  void Init(size_t num_bins, size_t N_B, size_t P, size_t S) {
    if (interval_counts.size() < N_B) interval_counts.resize(N_B);
    if (interval_extra.size() < N_B * P) interval_extra.resize(N_B * P);
    if (interval_freq.size() < N_B * P * S) interval_freq.resize(N_B * P * S);
    if (leaf_hist.size() < P * S) leaf_hist.resize(P * S);
    if (leaf_extra.size() < P) leaf_extra.resize(P);
    if (memo_cost.size() < num_bins * num_bins) memo_cost.resize(num_bins * num_bins);
    if (dp.size() < num_bins) dp.resize(num_bins);
    if (opt_prev.size() < num_bins) opt_prev.resize(num_bins);
    scratch1D.Init(N_B, P, S);
  }
};

NestedDPResult SolveNested2dDPFromPrefixTable(
    TreeSamples& tree_samples, size_t pA, size_t pB,
    const std::vector<int32_t>& cand_cutoffsA,
    const std::vector<int32_t>& valsB,
    const int64_t* table_counts,
    const int64_t* table_extra,
    const int32_t* table_freq,
    const std::vector<size_t>& max_symbols,
    size_t num_predictors,
    size_t max_symbols_stride,
    float scale, float base_node_cost = 92.0f,
    float log_node_cost = 1.2f,
    bool cost_only = false,
    Scratch2D* scratch = nullptr,
    bool use_greedy_inner = true) {
  NestedDPResult best_res;
  best_res.pA = pA;
  best_res.pB = pB;
  const size_t num_bins = cand_cutoffsA.size() + 1;
  const size_t N_B = valsB.size();
  const size_t P = num_predictors;
  const size_t S = max_symbols_stride;
  if (num_bins <= 1 || N_B == 0) {
    return best_res;
  }

  Scratch2D local_scratch;
  if (!scratch) {
    scratch = &local_scratch;
  }
  scratch->Init(num_bins, N_B, P, S);

  auto split_penaltyA = [&](int32_t cutoff_val) -> float {
    int32_t unquant = tree_samples.UnquantizeProperty(
        pA + tree_samples.NumStaticProps(), cutoff_val);
    return (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
  };

  float* memo_cost = scratch->memo_cost.data();
  std::fill(memo_cost, memo_cost + num_bins * num_bins, -1.0f);

  int64_t* interval_counts = scratch->interval_counts.data();
  int64_t* interval_extra = scratch->interval_extra.data();
  int32_t* interval_freq = scratch->interval_freq.data();
  int32_t* leaf_hist = scratch->leaf_hist.data();
  int64_t* leaf_extra = scratch->leaf_extra.data();

  auto eval_interval = [&](int32_t j, int32_t i) -> float {
    size_t memo_idx = j * num_bins + i;
    if (memo_cost[memo_idx] >= 0.0f) {
      return memo_cost[memo_idx];
    }

    std::fill(leaf_hist, leaf_hist + P * S, 0);
    std::fill(leaf_extra, leaf_extra + P, 0);
    int64_t total_interval_count = 0;

    for (size_t uB = 0; uB < N_B; uB++) {
      size_t idx_curr = i * N_B + uB;
      size_t idx_prev = (j > 0 ? (j - 1) * N_B + uB : 0);
      int64_t cnt = table_counts[idx_curr] - (j > 0 ? table_counts[idx_prev] : 0);
      interval_counts[uB] = cnt;
      total_interval_count += cnt;
      if (cnt == 0) {
        continue;
      }
      for (size_t pred = 0; pred < P; pred++) {
        int64_t ext = table_extra[idx_curr * P + pred] -
                      (j > 0 ? table_extra[idx_prev * P + pred] : 0);
        interval_extra[uB * P + pred] = ext;
        leaf_extra[pred] += ext;
        for (size_t k = 0; k < max_symbols[pred]; k++) {
          int32_t f = table_freq[(idx_curr * P + pred) * S + k] -
                      (j > 0 ? table_freq[(idx_prev * P + pred) * S + k] : 0);
          interval_freq[(uB * P + pred) * S + k] = f;
          leaf_hist[pred * S + k] += f;
        }
      }
    }

    if (total_interval_count == 0) {
      memo_cost[memo_idx] = 0.0f;
      return 0.0f;
    }

    float best_leaf_cost = std::numeric_limits<float>::max();
    for (size_t pred = 0; pred < P; pred++) {
      float bits = EstimateBits(&leaf_hist[pred * S], max_symbols[pred]) + leaf_extra[pred];
      if (bits < best_leaf_cost) {
        best_leaf_cost = bits;
      }
    }

    float best_cost = best_leaf_cost;
    if (best_leaf_cost > base_node_cost * scale) {
      float cost1D;
      if (use_greedy_inner) {
        cost1D = Solve1DGreedyFromTables(
            tree_samples, pB, valsB,
            interval_counts, interval_extra, interval_freq,
            max_symbols, P, S, scale, base_node_cost, log_node_cost,
            /*max_cuts=*/valsB.size(), &scratch->scratch1D).cost;
      } else {
        cost1D = Solve1DDPFromTables(
            tree_samples, pB, valsB,
            interval_counts, interval_extra, interval_freq,
            max_symbols, P, S, scale, base_node_cost, log_node_cost,
            &scratch->scratch1D).cost;
      }
      if (cost1D < best_cost) {
        best_cost = cost1D;
      }
    }
    memo_cost[memo_idx] = best_cost;
    return best_cost;
  };

  float* dp = scratch->dp.data();
  std::fill(dp, dp + num_bins, std::numeric_limits<float>::max());
  int32_t* opt_prev = scratch->opt_prev.data();
  std::fill(opt_prev, opt_prev + num_bins, -1);

  auto val = [&](int32_t j, int32_t t) -> float {
    float int_cost = eval_interval(j, t);
    if (j == 0) {
      return int_cost;
    }
    int32_t p_idx = j - 1;
    return dp[p_idx] + int_cost + split_penaltyA(cand_cutoffsA[p_idx]);
  };

  bool use_monge = true;
  const char* env_monge = getenv("JXL_DP_MONGE");
  if (env_monge != nullptr &&
      (strcmp(env_monge, "0") == 0 || strcmp(env_monge, "false") == 0 ||
       strcmp(env_monge, "no") == 0)) {
    use_monge = false;
  }

  if (!use_monge) {
    for (int32_t i = 0; i < static_cast<int32_t>(num_bins); i++) {
      for (int32_t j = i; j >= 0; j--) {
        float cost = eval_interval(j, i);
        float cand = (j == 0 ? cost
                             : dp[j - 1] + cost +
                                   split_penaltyA(cand_cutoffsA[j - 1]));
        if (cand < dp[i]) {
          dp[i] = cand;
          opt_prev[i] = (j == 0 ? -1 : j - 1);
        }
      }
    }
  } else {
    struct Candidate {
      int32_t j;
      int32_t start_i;
    };
    std::deque<Candidate> dq;
    dq.push_back(Candidate{0, 0});

    for (int32_t i = 0; i < static_cast<int32_t>(num_bins); i++) {
      while (dq.size() >= 2 && dq[1].start_i <= i) {
        dq.pop_front();
      }
      int32_t best_j = dq.front().j;
      float cost = eval_interval(best_j, i);
      dp[i] = (best_j == 0 ? cost
                           : dp[best_j - 1] + cost +
                                 split_penaltyA(cand_cutoffsA[best_j - 1]));
      opt_prev[i] = (best_j == 0 ? -1 : best_j - 1);

      int32_t j_new = i + 1;
      if (j_new < static_cast<int32_t>(num_bins)) {
        while (!dq.empty()) {
          int32_t j_back = dq.back().j;
          int32_t start_back = dq.back().start_i;
          if (start_back >= j_new) {
            if (val(j_new, start_back) <= val(j_back, start_back)) {
              dq.pop_back();
              continue;
            }
          }
          int32_t low = std::max(start_back + 1, j_new);
          if (low >= static_cast<int32_t>(num_bins)) {
            break;
          }
          if (val(j_new, num_bins - 1) > val(j_back, num_bins - 1)) {
            break;
          }
          int32_t l = low;
          int32_t r = num_bins - 1;
          int32_t t_star = num_bins;
          while (l <= r) {
            int32_t mid = (l + r) / 2;
            if (val(j_new, mid) <= val(j_back, mid)) {
              t_star = mid;
              r = mid - 1;
            } else {
              l = mid + 1;
            }
          }
          if (t_star < static_cast<int32_t>(num_bins)) {
            dq.push_back(Candidate{j_new, t_star});
          }
          break;
        }
        if (dq.empty()) {
          dq.push_back(Candidate{j_new, j_new});
        }
      }
    }
  }

  best_res.cost = dp[num_bins - 1];
  if (cost_only) {
    return best_res;
  }

  best_res.cutoffsA.clear();
  best_res.segment_refinements.clear();

  int32_t curr = static_cast<int32_t>(num_bins - 1);
  while (curr != -1) {
    int32_t prev = opt_prev[curr];
    int32_t j = prev + 1;
    int32_t i = curr;

    std::fill(leaf_hist, leaf_hist + P * S, 0);
    std::fill(leaf_extra, leaf_extra + P, 0);
    int64_t total_interval_count = 0;

    for (size_t uB = 0; uB < N_B; uB++) {
      size_t idx_curr = i * N_B + uB;
      size_t idx_prev = (j > 0 ? (j - 1) * N_B + uB : 0);
      int64_t cnt = table_counts[idx_curr] - (j > 0 ? table_counts[idx_prev] : 0);
      interval_counts[uB] = cnt;
      total_interval_count += cnt;
      if (cnt == 0) continue;
      for (size_t pred = 0; pred < P; pred++) {
        int64_t ext = table_extra[idx_curr * P + pred] -
                      (j > 0 ? table_extra[idx_prev * P + pred] : 0);
        interval_extra[uB * P + pred] = ext;
        leaf_extra[pred] += ext;
        for (size_t k = 0; k < max_symbols[pred]; k++) {
          int32_t f = table_freq[(idx_curr * P + pred) * S + k] -
                      (j > 0 ? table_freq[(idx_prev * P + pred) * S + k] : 0);
          interval_freq[(uB * P + pred) * S + k] = f;
          leaf_hist[pred * S + k] += f;
        }
      }
    }

    if (total_interval_count == 0) {
      DP1DResult empty_res;
      empty_res.cost = 0.0f;
      empty_res.segment_preds = {0};
      best_res.segment_refinements.push_back(std::move(empty_res));
    } else {
      float best_leaf_cost = std::numeric_limits<float>::max();
      size_t best_leaf_pred = 0;
      for (size_t pred = 0; pred < P; pred++) {
        float bits = EstimateBits(&leaf_hist[pred * S], max_symbols[pred]) + leaf_extra[pred];
        if (bits < best_leaf_cost) {
          best_leaf_cost = bits;
          best_leaf_pred = pred;
        }
      }

      DP1DResult refB = Solve1DDPFromTables(
          tree_samples, pB, valsB,
          interval_counts, interval_extra, interval_freq,
          max_symbols, P, S, scale, base_node_cost, log_node_cost);

      if (refB.cost < best_leaf_cost && !refB.cutoffs.empty()) {
        best_res.segment_refinements.push_back(std::move(refB));
      } else {
        DP1DResult leaf_res;
        leaf_res.cost = best_leaf_cost;
        leaf_res.cutoffs.clear();
        leaf_res.segment_preds = {best_leaf_pred};
        best_res.segment_refinements.push_back(std::move(leaf_res));
      }
    }

    if (prev != -1) {
      best_res.cutoffsA.push_back(cand_cutoffsA[prev]);
    }
    curr = prev;
  }
  std::reverse(best_res.cutoffsA.begin(), best_res.cutoffsA.end());
  std::reverse(best_res.segment_refinements.begin(),
               best_res.segment_refinements.end());

  return best_res;
}

NestedDPResult RunNested2dDP(
    TreeSamples& tree_samples, size_t pA, size_t pB,
    const std::vector<uint32_t>* sample_subset,
    float scale, float base_node_cost = 92.0f, float log_node_cost = 1.2f,
    const std::vector<int32_t>* cand_cutoffs_override = nullptr,
    const std::vector<size_t>* max_symbols_override = nullptr,
    size_t max_symbols_stride_override = 0) {
  NestedDPResult best_res;
  best_res.pA = pA;
  best_res.pB = pB;
  size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (total_samples == 0) {
    best_res.cost = 0.0f;
    return best_res;
  }

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  const size_t num_predictors = tree_samples.NumPredictors();

  DP1DResult dp1A = Run1DDPOnSamples(tree_samples, pA, sample_subset, scale, base_node_cost, log_node_cost,
                                     /*use_greedy=*/false, max_symbols_override, max_symbols_stride_override);
  DP1DResult dp1B = Run1DDPOnSamples(tree_samples, pB, sample_subset, scale, base_node_cost, log_node_cost,
                                     /*use_greedy=*/false, max_symbols_override, max_symbols_stride_override);

  // Set default fallback to best 1D
  if (dp1B.cost <= dp1A.cost) {
    best_res.cost = dp1B.cost;
    best_res.cutoffsA.clear();
    best_res.segment_refinements = {dp1B};
  } else {
    best_res.cost = dp1A.cost;
    best_res.cutoffsA = dp1A.cutoffs;
    best_res.segment_refinements.resize(dp1A.cutoffs.size() + 1);
    for (size_t s = 0; s <= dp1A.cutoffs.size(); s++) {
      best_res.segment_refinements[s].cost = 0;
      best_res.segment_refinements[s].segment_preds = {dp1A.segment_preds[s]};
    }
  }

  std::bitset<256> seenA;
  for (size_t i = 0; i < total_samples; i++) {
    seenA.set(tree_samples.Property<false>(pA, get_sample_idx(i)));
  }
  std::vector<int32_t> vals;
  vals.reserve(seenA.count());
  for (size_t v = 0; v < 256; v++) {
    if (seenA.test(v)) vals.push_back(static_cast<int32_t>(v));
  }

  if (vals.size() <= 1) {
    return best_res;
  }

  std::vector<int32_t> cand_cutoffsA;
  if (cand_cutoffs_override != nullptr && !cand_cutoffs_override->empty()) {
    for (int32_t c : *cand_cutoffs_override) {
      if (c >= vals.front() && c < vals.back()) {
        cand_cutoffsA.push_back(c);
      }
    }
    std::sort(cand_cutoffsA.begin(), cand_cutoffsA.end());
    cand_cutoffsA.erase(std::unique(cand_cutoffsA.begin(), cand_cutoffsA.end()), cand_cutoffsA.end());
  } else {
    cand_cutoffsA.reserve(vals.size() - 1);
    for (size_t i = 0; i + 1 < vals.size(); i++) {
      cand_cutoffsA.push_back(vals[i]);
    }
  }

  if (cand_cutoffsA.empty()) {
    return best_res;
  }

  const size_t num_bins = cand_cutoffsA.size() + 1;
  std::vector<uint32_t> sorted_samples(total_samples);
  std::array<size_t, 257> count_bin = {0};
  for (size_t i = 0; i < total_samples; i++) {
    count_bin[tree_samples.Property<false>(pA, get_sample_idx(i)) + 1]++;
  }
  for (size_t v = 1; v < 256; v++) {
    count_bin[v] += count_bin[v - 1];
  }
  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    uint8_t v = tree_samples.Property<false>(pA, s);
    sorted_samples[count_bin[v]++] = s;
  }

  std::vector<size_t> bin_start(num_bins + 1, 0);
  {
    size_t cur_bin = 0;
    for (size_t i = 0; i < total_samples; i++) {
      int32_t v = tree_samples.Property<false>(pA, sorted_samples[i]);
      while (cur_bin < cand_cutoffsA.size() && v > cand_cutoffsA[cur_bin]) {
        cur_bin++;
        bin_start[cur_bin] = i;
      }
    }
    for (size_t b = cur_bin + 1; b <= num_bins; b++) {
      bin_start[b] = total_samples;
    }
  }

  // Collect unique values of pB
  std::bitset<256> seenB;
  for (size_t i = 0; i < total_samples; i++) {
    seenB.set(tree_samples.Property<false>(pB, get_sample_idx(i)));
  }
  std::vector<int32_t> valsB;
  valsB.reserve(seenB.count());
  std::array<int16_t, 256> lut_B;
  lut_B.fill(-1);
  for (size_t v = 0; v < 256; v++) {
    if (seenB.test(v)) {
      lut_B[v] = static_cast<int16_t>(valsB.size());
      valsB.push_back(static_cast<int32_t>(v));
    }
  }

  if (valsB.size() <= 1) {
    return best_res;
  }

  const size_t num_binsB = valsB.size();
  std::vector<size_t> max_symbols(num_predictors, 0);
  size_t max_symbols_stride = 0;
  if (max_symbols_override != nullptr && !max_symbols_override->empty() && max_symbols_stride_override > 0) {
    max_symbols = *max_symbols_override;
    max_symbols_stride = max_symbols_stride_override;
  } else {
    for (size_t pred = 0; pred < num_predictors; pred++) {
      for (size_t i = 0; i < total_samples; i++) {
        size_t s = get_sample_idx(i);
        uint32_t tok = tree_samples.Token(pred, s);
        max_symbols[pred] = std::max(max_symbols[pred], static_cast<size_t>(tok + 1));
      }
      max_symbols[pred] = Padded(max_symbols[pred]);
      max_symbols_stride = std::max(max_symbols_stride, max_symbols[pred]);
    }
  }

  auto get_uB = [&](uint8_t v) -> size_t {
    return static_cast<size_t>(lut_B[v]);
  };

  const size_t N_A = num_bins;
  const size_t N_B = num_binsB;
  const size_t P = num_predictors;
  const size_t S = max_symbols_stride;

  std::vector<int64_t> table_counts(N_A * N_B, 0);
  std::vector<int64_t> table_extra(N_A * N_B * P, 0);
  std::vector<int32_t> table_freq(N_A * N_B * P * S, 0);

  for (size_t bA = 0; bA < N_A; bA++) {
    size_t start_i = bin_start[bA];
    size_t end_i = bin_start[bA + 1];
    for (size_t idx = start_i; idx < end_i; idx++) {
      size_t s = sorted_samples[idx];
      size_t uB = get_uB(tree_samples.Property<false>(pB, s));
      size_t cnt = tree_samples.Count(s);
      table_counts[bA * N_B + uB] += cnt;
      for (size_t pred = 0; pred < P; pred++) {
        table_extra[(bA * N_B + uB) * P + pred] +=
            tree_samples.RTokens(pred)[s].nbits * cnt;
        table_freq[((bA * N_B + uB) * P + pred) * S + tree_samples.Token(pred, s)] += cnt;
      }
    }
  }

  // Prefix sums along bA dimension
  for (size_t bA = 1; bA < N_A; bA++) {
    for (size_t uB = 0; uB < N_B; uB++) {
      table_counts[bA * N_B + uB] += table_counts[(bA - 1) * N_B + uB];
      for (size_t pred = 0; pred < P; pred++) {
        table_extra[(bA * N_B + uB) * P + pred] +=
            table_extra[((bA - 1) * N_B + uB) * P + pred];
        for (size_t k = 0; k < S; k++) {
          table_freq[((bA * N_B + uB) * P + pred) * S + k] +=
              table_freq[(((bA - 1) * N_B + uB) * P + pred) * S + k];
        }
      }
    }
  }
  NestedDPResult joint = SolveNested2dDPFromPrefixTable(
      tree_samples, pA, pB, cand_cutoffsA, valsB,
      table_counts.data(), table_extra.data(), table_freq.data(),
      max_symbols, P, S, scale, base_node_cost, log_node_cost);
  if (joint.cost < best_res.cost) {
    best_res = std::move(joint);
  }
  return best_res;
}

struct EvaluatedTripleTree {
  size_t pA = 0;
  size_t pB = 0;
  size_t pC = 0;
  float cost = std::numeric_limits<float>::max();
  DP1DResult dpA;
  std::vector<DP1DResult> refinementsB;
  std::vector<std::vector<DP1DResult>> refinementsC;
};

EvaluatedTripleTree Evaluate3PropRecursiveTree(
    TreeSamples& tree_samples,
    size_t pA, size_t pB, size_t pC,
    const DP1DResult& dpA,
    const std::vector<uint32_t>* sample_subset,
    float scale, float base_node_cost, float log_node_cost) {
  EvaluatedTripleTree res;
  res.pA = pA;
  res.pB = pB;
  res.pC = pC;
  res.dpA = dpA;

  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  if (dpA.cutoffs.empty()) {
    res.cost = dpA.cost;
    return res;
  }

  size_t num_segsA = dpA.cutoffs.size() + 1;
  std::vector<std::vector<uint32_t>> seg_samplesA(num_segsA);
  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    int32_t v = tree_samples.Property<false>(pA, s);
    size_t seg = 0;
    while (seg < dpA.cutoffs.size() && v > dpA.cutoffs[seg]) {
      seg++;
    }
    seg_samplesA[seg].push_back(s);
  }

  float cutoffsA_penalty = 0.0f;
  for (int32_t c : dpA.cutoffs) {
    int32_t unquant = tree_samples.UnquantizeProperty(pA + tree_samples.NumStaticProps(), c);
    cutoffsA_penalty += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
  }

  float total_tree_cost = cutoffsA_penalty;
  std::vector<DP1DResult> refsB(num_segsA);
  std::vector<std::vector<DP1DResult>> refsC(num_segsA);

  for (size_t sA = 0; sA < num_segsA; sA++) {
    if (seg_samplesA[sA].empty()) continue;

    size_t predA = dpA.segment_preds[sA];
    size_t max_sym = 0;
    for (uint32_t idx : seg_samplesA[sA]) {
      max_sym = std::max(max_sym, static_cast<size_t>(tree_samples.Token(predA, idx) + 1));
    }
    max_sym = Padded(max_sym);
    std::vector<int32_t> hist(max_sym, 0);
    int64_t extra = 0;
    for (uint32_t idx : seg_samplesA[sA]) {
      size_t cnt = tree_samples.Count(idx);
      hist[tree_samples.Token(predA, idx)] += cnt;
      extra += tree_samples.RTokens(predA)[idx].nbits * cnt;
    }
    float segA_base_cost = EstimateBits(hist.data(), max_sym) + extra;

    if (seg_samplesA[sA].size() < 32) {
      total_tree_cost += segA_base_cost;
      refsB[sA].cost = segA_base_cost;
      refsB[sA].segment_preds = {predA};
      continue;
    }

    DP1DResult refB = Run1DDPOnSamples(tree_samples, pB, &seg_samplesA[sA], scale,
                                       base_node_cost, log_node_cost);
    if (refB.cost >= segA_base_cost || refB.cutoffs.empty()) {
      total_tree_cost += segA_base_cost;
      refsB[sA].cost = segA_base_cost;
      refsB[sA].segment_preds = {predA};
      continue;
    }

    size_t num_subsegsB = refB.cutoffs.size() + 1;
    std::vector<std::vector<uint32_t>> subseg_samplesB(num_subsegsB);
    for (uint32_t idx : seg_samplesA[sA]) {
      int32_t v = tree_samples.Property<false>(pB, idx);
      size_t subseg = std::upper_bound(refB.cutoffs.begin(), refB.cutoffs.end(), v) - refB.cutoffs.begin();
      subseg_samplesB[subseg].push_back(idx);
    }

    float cutoffsB_penalty = 0.0f;
    for (int32_t c : refB.cutoffs) {
      int32_t unquant = tree_samples.UnquantizeProperty(pB + tree_samples.NumStaticProps(), c);
      cutoffsB_penalty += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
    }

    float segB_total_cost = cutoffsB_penalty;
    std::vector<DP1DResult> subrefsC(num_subsegsB);

    for (size_t sB = 0; sB < num_subsegsB; sB++) {
      if (subseg_samplesB[sB].empty()) continue;

      size_t predB = refB.segment_preds[sB];
      size_t sub_max_sym = 0;
      for (uint32_t idx : subseg_samplesB[sB]) {
        sub_max_sym = std::max(sub_max_sym, static_cast<size_t>(tree_samples.Token(predB, idx) + 1));
      }
      sub_max_sym = Padded(sub_max_sym);
      std::vector<int32_t> sub_hist(sub_max_sym, 0);
      int64_t sub_extra = 0;
      for (uint32_t idx : subseg_samplesB[sB]) {
        size_t cnt = tree_samples.Count(idx);
        sub_hist[tree_samples.Token(predB, idx)] += cnt;
        sub_extra += tree_samples.RTokens(predB)[idx].nbits * cnt;
      }
      float subsegB_base_cost = EstimateBits(sub_hist.data(), sub_max_sym) + sub_extra;

      if (subseg_samplesB[sB].size() < 32) {
        segB_total_cost += subsegB_base_cost;
        subrefsC[sB].cost = subsegB_base_cost;
        subrefsC[sB].segment_preds = {predB};
        continue;
      }

      DP1DResult refC = Run1DDPOnSamples(tree_samples, pC, &subseg_samplesB[sB], scale,
                                         base_node_cost, log_node_cost);
      if (refC.cost < subsegB_base_cost && !refC.cutoffs.empty()) {
        segB_total_cost += refC.cost;
        subrefsC[sB] = std::move(refC);
      } else {
        segB_total_cost += subsegB_base_cost;
        subrefsC[sB].cost = subsegB_base_cost;
        subrefsC[sB].segment_preds = {predB};
      }
    }

    total_tree_cost += segB_total_cost;
    refsB[sA] = std::move(refB);
    refsC[sA] = std::move(subrefsC);
  }

  res.cost = total_tree_cost;
  res.refinementsB = std::move(refsB);
  res.refinementsC = std::move(refsC);
  return res;
}

EvaluatedTripleTree RunSinglyNested3PropDP(
    TreeSamples& tree_samples, size_t pA, size_t pB, size_t pC,
    const std::vector<uint32_t>* sample_subset, float scale,
    float base_node_cost, float log_node_cost) {
  EvaluatedTripleTree res;
  res.pA = pA; res.pB = pB; res.pC = pC;
  size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (total_samples == 0) return res;

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  NestedDPResult nestedAB = RunNested2dDP(tree_samples, pA, pB, sample_subset, scale, base_node_cost, log_node_cost);

  res.dpA.prop_dim = pA;
  res.dpA.cutoffs = nestedAB.cutoffsA;
  size_t num_segsA = nestedAB.cutoffsA.size() + 1;
  res.dpA.segment_preds.resize(num_segsA, 0);
  for (size_t sA = 0; sA < num_segsA; sA++) {
    if (sA < nestedAB.segment_refinements.size() && !nestedAB.segment_refinements[sA].segment_preds.empty()) {
      res.dpA.segment_preds[sA] = nestedAB.segment_refinements[sA].segment_preds[0];
    }
  }

  res.refinementsB = nestedAB.segment_refinements;
  if (res.refinementsB.size() < num_segsA) {
    res.refinementsB.resize(num_segsA);
  }
  res.refinementsC.resize(num_segsA);

  std::vector<std::vector<uint32_t>> seg_samplesA(num_segsA);
  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    int32_t v = tree_samples.Property<false>(pA, s);
    size_t seg = std::upper_bound(nestedAB.cutoffsA.begin(), nestedAB.cutoffsA.end(), v) - nestedAB.cutoffsA.begin();
    seg_samplesA[seg].push_back(s);
  }

  float cutoffsA_penalty = 0.0f;
  for (int32_t c : nestedAB.cutoffsA) {
    int32_t unquant = tree_samples.UnquantizeProperty(pA + tree_samples.NumStaticProps(), c);
    cutoffsA_penalty += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
  }
  float total_cost = cutoffsA_penalty;

  for (size_t sA = 0; sA < num_segsA; sA++) {
    const auto& refB = res.refinementsB[sA];
    size_t num_segsB = refB.cutoffs.size() + 1;
    res.refinementsC[sA].resize(num_segsB);

    float cutoffsB_penalty = 0.0f;
    for (int32_t c : refB.cutoffs) {
      int32_t unquant = tree_samples.UnquantizeProperty(pB + tree_samples.NumStaticProps(), c);
      cutoffsB_penalty += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
    }
    total_cost += cutoffsB_penalty;

    std::vector<std::vector<uint32_t>> seg_samplesB(num_segsB);
    for (uint32_t s : seg_samplesA[sA]) {
      int32_t v = tree_samples.Property<false>(pB, s);
      size_t seg = std::upper_bound(refB.cutoffs.begin(), refB.cutoffs.end(), v) - refB.cutoffs.begin();
      seg_samplesB[seg].push_back(s);
    }

    for (size_t sB = 0; sB < num_segsB; sB++) {
      if (seg_samplesB[sB].empty()) {
        res.refinementsC[sA][sB].prop_dim = pC;
        res.refinementsC[sA][sB].segment_preds = {0};
        continue;
      }
      size_t predB = sB < refB.segment_preds.size() ? refB.segment_preds[sB] : 0;
      size_t max_sym = 0;
      for (uint32_t idx : seg_samplesB[sB]) {
        max_sym = std::max(max_sym, static_cast<size_t>(tree_samples.Token(predB, idx) + 1));
      }
      max_sym = Padded(max_sym);
      std::vector<int32_t> hist(max_sym, 0);
      int64_t extra = 0;
      for (uint32_t idx : seg_samplesB[sB]) {
        size_t cnt = tree_samples.Count(idx);
        hist[tree_samples.Token(predB, idx)] += cnt;
        extra += tree_samples.RTokens(predB)[idx].nbits * cnt;
      }
      float base_leaf_cost = EstimateBits(hist.data(), max_sym) + extra;

      DP1DResult refC = Run1DDPOnSamples(tree_samples, pC, &seg_samplesB[sB], scale, base_node_cost, log_node_cost);
      if (refC.cost < base_leaf_cost && !refC.cutoffs.empty()) {
        total_cost += refC.cost;
        res.refinementsC[sA][sB] = std::move(refC);
      } else {
        total_cost += base_leaf_cost;
        res.refinementsC[sA][sB].cost = base_leaf_cost;
        res.refinementsC[sA][sB].segment_preds = {predB};
      }
    }
  }

  res.cost = total_cost;
  return res;
}

EvaluatedTripleTree RunStagedNested3PropDP(
    TreeSamples& tree_samples, size_t pA, size_t pB, size_t pC,
    const std::vector<uint32_t>* sample_subset, float scale,
    float base_node_cost, float log_node_cost) {
  EvaluatedTripleTree res;
  res.pA = pA; res.pB = pB; res.pC = pC;
  size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (total_samples == 0) return res;

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  auto t0 = std::chrono::steady_clock::now();
  NestedDPResult nestedAB = RunNested2dDP(tree_samples, pA, pB, sample_subset, scale, base_node_cost, log_node_cost);
  auto t1 = std::chrono::steady_clock::now();

  res.dpA.prop_dim = pA;
  res.dpA.cutoffs = nestedAB.cutoffsA;
  size_t num_segsA = nestedAB.cutoffsA.size() + 1;
  res.dpA.segment_preds.resize(num_segsA, 0);

  res.refinementsB.resize(num_segsA);
  res.refinementsC.resize(num_segsA);

  std::vector<std::vector<uint32_t>> seg_samplesA(num_segsA);
  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    int32_t v = tree_samples.Property<false>(pA, s);
    size_t seg = 0;
    while (seg < nestedAB.cutoffsA.size() && v > nestedAB.cutoffsA[seg]) {
      seg++;
    }
    seg_samplesA[seg].push_back(s);
  }

  float cutoffsA_penalty = 0.0f;
  for (int32_t c : nestedAB.cutoffsA) {
    int32_t unquant = tree_samples.UnquantizeProperty(pA + tree_samples.NumStaticProps(), c);
    cutoffsA_penalty += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
  }
  float total_cost = cutoffsA_penalty;

  double sub_time_ms = 0.0;
  for (size_t sA = 0; sA < num_segsA; sA++) {
    if (seg_samplesA[sA].empty()) {
      res.refinementsB[sA].prop_dim = pB;
      res.refinementsB[sA].segment_preds = {0};
      res.refinementsC[sA].resize(1);
      res.refinementsC[sA][0].segment_preds = {0};
      continue;
    }

    auto t_sub0 = std::chrono::steady_clock::now();
    NestedDPResult subBC = RunNested2dDP(tree_samples, pB, pC, &seg_samplesA[sA], scale, base_node_cost, log_node_cost);
    auto t_sub1 = std::chrono::steady_clock::now();
    sub_time_ms += std::chrono::duration<double, std::milli>(t_sub1 - t_sub0).count();

    total_cost += subBC.cost;

    res.refinementsB[sA].prop_dim = pB;
    res.refinementsB[sA].cutoffs = subBC.cutoffsA;
    size_t num_segsB = subBC.cutoffsA.size() + 1;
    res.refinementsB[sA].segment_preds.resize(num_segsB, 0);

    res.refinementsC[sA] = std::move(subBC.segment_refinements);
    if (res.refinementsC[sA].size() < num_segsB) {
      res.refinementsC[sA].resize(num_segsB);
    }
    for (size_t sB = 0; sB < num_segsB; sB++) {
      if (!res.refinementsC[sA][sB].segment_preds.empty()) {
        res.refinementsB[sA].segment_preds[sB] = res.refinementsC[sA][sB].segment_preds[0];
      }
    }
    if (!res.refinementsB[sA].segment_preds.empty()) {
      res.dpA.segment_preds[sA] = res.refinementsB[sA].segment_preds[0];
    }
  }

  res.cost = total_cost;
  if (getenv("JXL_LOG_STAGED_TIMING")) {
    double ab_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr, "STAGED_TIMING: perms=(%zu,%zu,%zu) ab=%.2f ms, sub=%.2f ms, num_segsA=%zu\n",
            pA, pB, pC, ab_time_ms, sub_time_ms, num_segsA);
  }
  return res;
}

EvaluatedTripleTree RunDoublyNested3PropDP(
    TreeSamples& tree_samples, size_t pA, size_t pB, size_t pC,
    const std::vector<uint32_t>* sample_subset, float scale,
    float base_node_cost, float log_node_cost,
    const MultiPropertyTables* mpt = nullptr) {
  size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (total_samples == 0) return EvaluatedTripleTree{};

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  MultiPropertyTables mpt_local;
  if (!mpt) {
    mpt_local = BuildMultiPropertyTables(tree_samples, sample_subset);
    mpt = &mpt_local;
  }

  const auto& valsA = mpt->vals[pA];
  const auto& valsB = mpt->vals[pB];
  const auto& valsC = mpt->vals[pC];
  const auto& lut_B = mpt->luts[pB];
  const auto& lut_C = mpt->luts[pC];
  const auto& max_symbols = mpt->max_symbols;
  const size_t num_predictors = tree_samples.NumPredictors();
  const size_t P = num_predictors;
  const size_t S = mpt->S;

  if (valsA.size() <= 1 || valsB.size() <= 1 || valsC.size() <= 1) {
    return RunStagedNested3PropDP(tree_samples, pA, pB, pC, sample_subset, scale, base_node_cost, log_node_cost);
  }

  std::vector<int32_t> cand_cutoffsA(valsA.begin(), valsA.end() - 1);
  std::vector<int32_t> cand_cutoffsB(valsB.begin(), valsB.end() - 1);

  const size_t num_bins = cand_cutoffsA.size() + 1;
  const size_t num_binsB = valsB.size();
  const size_t num_binsC = valsC.size();

  // Counting sort on property A
  std::vector<uint32_t> sorted_samples(total_samples);
  std::array<size_t, 257> count_bin = {0};
  for (size_t i = 0; i < total_samples; i++) {
    count_bin[tree_samples.Property<false>(pA, get_sample_idx(i)) + 1]++;
  }
  for (size_t v = 1; v < 256; v++) {
    count_bin[v] += count_bin[v - 1];
  }
  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    uint8_t v = tree_samples.Property<false>(pA, s);
    sorted_samples[count_bin[v]++] = s;
  }

  std::vector<size_t> bin_start(num_bins + 1, 0);
  {
    size_t cur_bin = 0;
    for (size_t i = 0; i < total_samples; i++) {
      int32_t v = tree_samples.Property<false>(pA, sorted_samples[i]);
      while (cur_bin < cand_cutoffsA.size() && v > cand_cutoffsA[cur_bin]) {
        cur_bin++;
        bin_start[cur_bin] = i;
      }
    }
    for (size_t b = cur_bin + 1; b <= num_bins; b++) {
      bin_start[b] = total_samples;
    }
  }

  std::vector<uint16_t> sample_uB(total_samples);
  std::vector<uint16_t> sample_uC(total_samples);
  for (size_t i = 0; i < total_samples; i++) {
    size_t s = sorted_samples[i];
    sample_uB[i] = static_cast<uint16_t>(lut_B[tree_samples.Property<false>(pB, s)]);
    sample_uC[i] = static_cast<uint16_t>(lut_C[tree_samples.Property<false>(pC, s)]);
  }

  const size_t N_B = num_binsB;
  const size_t N_C = num_binsC;

  std::vector<int64_t> pref_A_counts(num_bins, 0);
  std::vector<int64_t> pref_A_extra(num_bins * P, 0);
  std::vector<int32_t> pref_A_freq(num_bins * P * S, 0);

  for (size_t b = 0; b < num_bins; b++) {
    size_t s_start = bin_start[b];
    size_t s_end = bin_start[b + 1];
    for (size_t idx = s_start; idx < s_end; idx++) {
      size_t s = sorted_samples[idx];
      size_t cnt = tree_samples.Count(s);
      pref_A_counts[b] += cnt;
      for (size_t pred = 0; pred < P; pred++) {
        pref_A_extra[b * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
        pref_A_freq[(b * P + pred) * S + tree_samples.Token(pred, s)] += cnt;
      }
    }
    if (b > 0) {
      pref_A_counts[b] += pref_A_counts[b - 1];
      for (size_t pred = 0; pred < P; pred++) {
        pref_A_extra[b * P + pred] += pref_A_extra[(b - 1) * P + pred];
        for (size_t k = 0; k < S; k++) {
          pref_A_freq[(b * P + pred) * S + k] += pref_A_freq[((b - 1) * P + pred) * S + k];
        }
      }
    }
  }

  std::vector<int32_t> leaf_hist_buf(P * S, 0);
  auto get_leaf_costA = [&](int32_t j, int32_t i, size_t* best_pred = nullptr) -> float {
    int64_t cur_cnt = pref_A_counts[i] - (j > 0 ? pref_A_counts[j - 1] : 0);
    if (cur_cnt == 0) {
      if (best_pred) *best_pred = 0;
      return 0.0f;
    }
    float best_cost = std::numeric_limits<float>::max();
    size_t best_p = 0;
    for (size_t pred = 0; pred < P; pred++) {
      int32_t* h = &leaf_hist_buf[pred * S];
      const int32_t* pi = &pref_A_freq[(i * P + pred) * S];
      const int32_t* pj = (j > 0 ? &pref_A_freq[((j - 1) * P + pred) * S] : nullptr);
      if (pj) {
        for (size_t k = 0; k < max_symbols[pred]; k++) {
          h[k] = pi[k] - pj[k];
        }
      } else {
        for (size_t k = 0; k < max_symbols[pred]; k++) {
          h[k] = pi[k];
        }
      }
      int64_t ext = pref_A_extra[i * P + pred] -
                    (j > 0 ? pref_A_extra[(j - 1) * P + pred] : 0);
      float bits = EstimateBits(h, max_symbols[pred]) + ext;
      if (bits < best_cost) {
        best_cost = bits;
        best_p = pred;
      }
    }
    if (best_pred) *best_pred = best_p;
    return best_cost;
  };

  auto split_penaltyA = [&](int32_t cutoff_val) -> float {
    int32_t unquant = tree_samples.UnquantizeProperty(
        pA + tree_samples.NumStaticProps(), cutoff_val);
    return (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
  };

  auto split_penaltyB = [&](int32_t cutoff_val) -> float {
    int32_t unquant = tree_samples.UnquantizeProperty(
        pB + tree_samples.NumStaticProps(), cutoff_val);
    return (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
  };

  std::vector<float> memo_cost(num_bins * num_bins, -1.0f);
  Scratch2D scratch2D;

  std::vector<int64_t> table_counts;
  std::vector<int64_t> table_extra;
  std::vector<int32_t> table_freq;
  std::vector<int64_t> pref_B_counts;
  std::vector<int64_t> pref_B_extra;
  std::vector<int32_t> pref_B_freq;
  std::vector<float> memo_marginal_B;
  std::vector<int32_t> marginal_hist_buf(P * S, 0);

  std::vector<int32_t> map_B(N_B, -1);
  std::vector<int32_t> map_C(N_C, -1);
  std::vector<int32_t> active_B; active_B.reserve(N_B);
  std::vector<int32_t> active_C; active_C.reserve(N_C);
  std::vector<bool> seen_B(N_B, false);
  std::vector<bool> seen_C(N_C, false);
  std::vector<int32_t> cand_cutoffsB_compact;
  std::vector<int32_t> valsC_compact;

  auto solve_compact_BC = [&](size_t start_idx, size_t end_idx, bool cost_only) -> NestedDPResult {
    NestedDPResult res_sub;
    res_sub.pA = pB;
    res_sub.pB = pC;
    if (start_idx >= end_idx) {
      res_sub.cost = 0.0f;
      return res_sub;
    }

    for (size_t idx = start_idx; idx < end_idx; idx++) {
      seen_B[sample_uB[idx]] = true;
      seen_C[sample_uC[idx]] = true;
    }
    active_B.clear();
    for (size_t u = 0; u < N_B; u++) {
      if (seen_B[u]) {
        map_B[u] = static_cast<int32_t>(active_B.size());
        active_B.push_back(static_cast<int32_t>(u));
      }
    }
    active_C.clear();
    for (size_t u = 0; u < N_C; u++) {
      if (seen_C[u]) {
        map_C[u] = static_cast<int32_t>(active_C.size());
        active_C.push_back(static_cast<int32_t>(u));
      }
    }

    size_t M_B = active_B.size();
    size_t M_C = active_C.size();

    if (M_B <= 1) {
      valsC_compact.resize(M_C);
      for (size_t m = 0; m < M_C; m++) {
        valsC_compact[m] = valsC[active_C[m]];
      }
      scratch2D.Init(1, M_C, P, S);
      std::fill(scratch2D.interval_counts.begin(), scratch2D.interval_counts.begin() + M_C, 0);
      std::fill(scratch2D.interval_extra.begin(), scratch2D.interval_extra.begin() + M_C * P, 0);
      std::fill(scratch2D.interval_freq.begin(), scratch2D.interval_freq.begin() + M_C * P * S, 0);
      std::fill(scratch2D.leaf_hist.begin(), scratch2D.leaf_hist.begin() + P * S, 0);
      std::fill(scratch2D.leaf_extra.begin(), scratch2D.leaf_extra.begin() + P, 0);

      for (size_t idx = start_idx; idx < end_idx; idx++) {
        size_t s = sorted_samples[idx];
        size_t mC = map_C[sample_uC[idx]];
        size_t cnt = tree_samples.Count(s);
        scratch2D.interval_counts[mC] += cnt;
        for (size_t pred = 0; pred < P; pred++) {
          int64_t ext = tree_samples.RTokens(pred)[s].nbits * cnt;
          scratch2D.interval_extra[mC * P + pred] += ext;
          scratch2D.leaf_extra[pred] += ext;
          uint32_t tok = tree_samples.Token(pred, s);
          scratch2D.interval_freq[(mC * P + pred) * S + tok] += cnt;
          scratch2D.leaf_hist[pred * S + tok] += cnt;
        }
      }

      float best_leaf_cost = std::numeric_limits<float>::max();
      size_t best_leaf_pred = 0;
      for (size_t pred = 0; pred < P; pred++) {
        float bits = EstimateBits(&scratch2D.leaf_hist[pred * S], max_symbols[pred]) + scratch2D.leaf_extra[pred];
        if (bits < best_leaf_cost) {
          best_leaf_cost = bits;
          best_leaf_pred = pred;
        }
      }

      if (cost_only) {
        float cost1D = Solve1DGreedyFromTables(
            tree_samples, pC, valsC_compact,
            scratch2D.interval_counts.data(), scratch2D.interval_extra.data(), scratch2D.interval_freq.data(),
            max_symbols, P, S, scale, base_node_cost, log_node_cost,
            /*max_cuts=*/valsC_compact.size(), &scratch2D.scratch1D).cost;
        res_sub.cost = std::min(best_leaf_cost, cost1D);
      } else {
        DP1DResult refC = Solve1DDPFromTables(
            tree_samples, pC, valsC_compact,
            scratch2D.interval_counts.data(), scratch2D.interval_extra.data(), scratch2D.interval_freq.data(),
            max_symbols, P, S, scale, base_node_cost, log_node_cost);
        if (refC.cost < best_leaf_cost && !refC.cutoffs.empty()) {
          res_sub.cost = refC.cost;
          res_sub.segment_refinements = {std::move(refC)};
        } else {
          res_sub.cost = best_leaf_cost;
          DP1DResult leaf_res;
          leaf_res.cost = best_leaf_cost;
          leaf_res.segment_preds = {best_leaf_pred};
          res_sub.segment_refinements = {std::move(leaf_res)};
        }
      }
    } else {
      cand_cutoffsB_compact.resize(M_B - 1);
      for (size_t m = 0; m + 1 < M_B; m++) {
        cand_cutoffsB_compact[m] = valsB[active_B[m]];
      }
      valsC_compact.resize(M_C);
      for (size_t m = 0; m < M_C; m++) {
        valsC_compact[m] = valsC[active_C[m]];
      }

      if (table_counts.size() < M_B * M_C) table_counts.resize(M_B * M_C);
      if (table_extra.size() < M_B * M_C * P) table_extra.resize(M_B * M_C * P);
      if (table_freq.size() < M_B * M_C * P * S) table_freq.resize(M_B * M_C * P * S);

      std::fill(table_counts.begin(), table_counts.begin() + M_B * M_C, 0);
      std::fill(table_extra.begin(), table_extra.begin() + M_B * M_C * P, 0);
      std::fill(table_freq.begin(), table_freq.begin() + M_B * M_C * P * S, 0);

      for (size_t idx = start_idx; idx < end_idx; idx++) {
        size_t s = sorted_samples[idx];
        size_t mB = map_B[sample_uB[idx]];
        size_t mC = map_C[sample_uC[idx]];
        size_t cnt = tree_samples.Count(s);
        table_counts[mB * M_C + mC] += cnt;
        for (size_t pred = 0; pred < P; pred++) {
          table_extra[(mB * M_C + mC) * P + pred] +=
              tree_samples.RTokens(pred)[s].nbits * cnt;
          table_freq[((mB * M_C + mC) * P + pred) * S + tree_samples.Token(pred, s)] += cnt;
        }
      }

      for (size_t mB = 1; mB < M_B; mB++) {
        for (size_t mC = 0; mC < M_C; mC++) {
          table_counts[mB * M_C + mC] += table_counts[(mB - 1) * M_C + mC];
          for (size_t pred = 0; pred < P; pred++) {
            table_extra[(mB * M_C + mC) * P + pred] +=
                table_extra[((mB - 1) * M_C + mC) * P + pred];
            for (size_t k = 0; k < S; k++) {
              table_freq[((mB * M_C + mC) * P + pred) * S + k] +=
                  table_freq[(((mB - 1) * M_C + mC) * P + pred) * S + k];
            }
          }
        }
      }

      if (cost_only) {
        scratch2D.Init(M_B, M_C, P, S);

        if (pref_B_counts.size() < M_B) pref_B_counts.resize(M_B);
        if (pref_B_extra.size() < M_B * P) pref_B_extra.resize(M_B * P);
        if (pref_B_freq.size() < M_B * P * S) pref_B_freq.resize(M_B * P * S);
        if (memo_marginal_B.size() < M_B * M_B) memo_marginal_B.resize(M_B * M_B);
        std::fill(memo_marginal_B.begin(), memo_marginal_B.begin() + M_B * M_B, -1.0f);

        for (size_t mB = 0; mB < M_B; mB++) {
          int64_t cnt = 0;
          for (size_t mC = 0; mC < M_C; mC++) {
            cnt += table_counts[mB * M_C + mC];
          }
          pref_B_counts[mB] = cnt;
          for (size_t pred = 0; pred < P; pred++) {
            int64_t ext = 0;
            for (size_t mC = 0; mC < M_C; mC++) {
              ext += table_extra[(mB * M_C + mC) * P + pred];
            }
            pref_B_extra[mB * P + pred] = ext;
            for (size_t k = 0; k < max_symbols[pred]; k++) {
              int32_t f = 0;
              for (size_t mC = 0; mC < M_C; mC++) {
                f += table_freq[((mB * M_C + mC) * P + pred) * S + k];
              }
              pref_B_freq[(mB * P + pred) * S + k] = f;
            }
          }
        }

        auto eval_marginal_B = [&](int32_t j, int32_t i) -> float {
          size_t memo_idx = j * M_B + i;
          if (memo_marginal_B[memo_idx] >= 0.0f) {
            return memo_marginal_B[memo_idx];
          }
          int64_t cnt = pref_B_counts[i] - (j > 0 ? pref_B_counts[j - 1] : 0);
          if (cnt == 0) {
            memo_marginal_B[memo_idx] = 0.0f;
            return 0.0f;
          }
          float best_cost = std::numeric_limits<float>::max();
          int32_t* rh = marginal_hist_buf.data();
          for (size_t pred = 0; pred < P; pred++) {
            const int32_t* pi = &pref_B_freq[(i * P + pred) * S];
            const int32_t* pj = (j > 0 ? &pref_B_freq[((j - 1) * P + pred) * S] : nullptr);
            if (pj) {
              for (size_t k = 0; k < max_symbols[pred]; k++) rh[k] = pi[k] - pj[k];
            } else {
              for (size_t k = 0; k < max_symbols[pred]; k++) rh[k] = pi[k];
            }
            int64_t extra = pref_B_extra[i * P + pred] - (j > 0 ? pref_B_extra[(j - 1) * P + pred] : 0);
            float bits = EstimateBits(rh, max_symbols[pred]) + extra;
            if (bits < best_cost) best_cost = bits;
          }
          memo_marginal_B[memo_idx] = best_cost;
          return best_cost;
        };

        auto recursive_2d_greedy = [&](auto& self, int32_t L_B, int32_t R_B) -> float {
          int64_t total_C_count = 0;
          int64_t* interval_counts_C = scratch2D.interval_counts.data();
          int64_t* interval_extra_C = scratch2D.interval_extra.data();
          int32_t* interval_freq_C = scratch2D.interval_freq.data();
          int32_t* leaf_hist_C = scratch2D.leaf_hist.data();
          int64_t* leaf_extra_C = scratch2D.leaf_extra.data();

          std::fill(leaf_hist_C, leaf_hist_C + P * S, 0);
          std::fill(leaf_extra_C, leaf_extra_C + P, 0);

          for (size_t mC = 0; mC < M_C; mC++) {
            size_t idx_R = R_B * M_C + mC;
            size_t idx_L = (L_B > 0 ? (L_B - 1) * M_C + mC : 0);
            int64_t cnt = table_counts[idx_R] - (L_B > 0 ? table_counts[idx_L] : 0);
            interval_counts_C[mC] = cnt;
            total_C_count += cnt;
            if (cnt == 0) {
              for (size_t pred = 0; pred < P; pred++) {
                interval_extra_C[mC * P + pred] = 0;
                for (size_t k = 0; k < max_symbols[pred]; k++) {
                  interval_freq_C[(mC * P + pred) * S + k] = 0;
                }
              }
              continue;
            }
            for (size_t pred = 0; pred < P; pred++) {
              int64_t ext = table_extra[idx_R * P + pred] -
                  (L_B > 0 ? table_extra[idx_L * P + pred] : 0);
              interval_extra_C[mC * P + pred] = ext;
              leaf_extra_C[pred] += ext;
              for (size_t k = 0; k < max_symbols[pred]; k++) {
                int32_t f = table_freq[(idx_R * P + pred) * S + k] -
                    (L_B > 0 ? table_freq[(idx_L * P + pred) * S + k] : 0);
                interval_freq_C[(mC * P + pred) * S + k] = f;
                leaf_hist_C[pred * S + k] += f;
              }
            }
          }

          if (total_C_count == 0) return 0.0f;

          float best_leaf_cost = std::numeric_limits<float>::max();
          for (size_t pred = 0; pred < P; pred++) {
            float bits = EstimateBits(&leaf_hist_C[pred * S], max_symbols[pred]) + leaf_extra_C[pred];
            if (bits < best_leaf_cost) best_leaf_cost = bits;
          }

          float cost_leaf = best_leaf_cost;
          if (best_leaf_cost > base_node_cost * scale) {
            float cost_greedy_C = Solve1DGreedyFromTables(
                tree_samples, pC, valsC_compact,
                interval_counts_C, interval_extra_C, interval_freq_C,
                max_symbols, P, S, scale, base_node_cost, log_node_cost,
                /*max_cuts=*/valsC_compact.size(), &scratch2D.scratch1D).cost;
            if (cost_greedy_C < cost_leaf) cost_leaf = cost_greedy_C;
          }

          if (L_B >= R_B) return cost_leaf;

          float base_marginal = eval_marginal_B(L_B, R_B);
          int64_t count_L = (L_B > 0 ? pref_B_counts[L_B - 1] : 0);
          int64_t count_R = pref_B_counts[R_B];
          int32_t best_c = -1;
          float min_split_marginal = base_marginal;

          for (int32_t c = L_B; c < R_B; c++) {
            int64_t count_c = pref_B_counts[c];
            if (count_c == count_L || count_c == count_R) continue;
            float penalty_B = split_penaltyB(cand_cutoffsB_compact[c]);
            float cost = eval_marginal_B(L_B, c) + eval_marginal_B(c + 1, R_B) + penalty_B;
            if (cost < min_split_marginal) {
              min_split_marginal = cost;
              best_c = c;
            }
          }

          if (best_c == -1) {
            return cost_leaf;
          }

          float penalty = split_penaltyB(cand_cutoffsB_compact[best_c]);
          float cost_left = self(self, L_B, best_c);
          float cost_right = self(self, best_c + 1, R_B);
          float cost_split = cost_left + cost_right + penalty;

          if (cost_leaf <= cost_split) {
            return cost_leaf;
          }
          return cost_split;
        };

        res_sub.cost = recursive_2d_greedy(recursive_2d_greedy, 0, static_cast<int32_t>(M_B - 1));
      } else {
        // Winning segment of pA: Solve full nested-2D DP on (B, C)
        res_sub = SolveNested2dDPFromPrefixTable(
            tree_samples, pB, pC, cand_cutoffsB_compact, valsC_compact,
            table_counts.data(), table_extra.data(), table_freq.data(),
            max_symbols, P, S, scale, base_node_cost, log_node_cost,
            /*cost_only=*/false, &scratch2D, /*use_greedy_inner=*/true);
      }
    }

    for (int32_t uB : active_B) { seen_B[uB] = false; map_B[uB] = -1; }
    for (int32_t uC : active_C) { seen_C[uC] = false; map_C[uC] = -1; }

    return res_sub;
  };

  auto eval_outer_cost = [&](int32_t j, int32_t i) -> float {
    size_t memo_idx = j * num_bins + i;
    if (memo_cost[memo_idx] >= 0.0f) {
      return memo_cost[memo_idx];
    }
    size_t start_idx = bin_start[j];
    size_t end_idx = bin_start[i + 1];
    if (start_idx >= end_idx) {
      memo_cost[memo_idx] = 0.0f;
      return 0.0f;
    }
    size_t K = end_idx - start_idx;
    float leaf_cost = get_leaf_costA(j, i);
    float min_split_cost = base_node_cost * scale;
    if (leaf_cost <= min_split_cost || K <= 16) {
      memo_cost[memo_idx] = leaf_cost;
      return leaf_cost;
    }
    float cost = solve_compact_BC(start_idx, end_idx, /*cost_only=*/true).cost;
    cost = std::min(cost, leaf_cost);
    memo_cost[memo_idx] = cost;
    return cost;
  };

  std::vector<float> dp(num_bins, std::numeric_limits<float>::max());
  std::vector<int32_t> opt_prev(num_bins, -1);

  auto val = [&](int32_t j, int32_t t) -> float {
    if (j > t) return std::numeric_limits<float>::max();
    float prev_penalty = (j == 0 ? 0.0f : dp[j - 1] + split_penaltyA(cand_cutoffsA[j - 1]));
    float int_cost = eval_outer_cost(j, t);
    return prev_penalty + int_cost;
  };

  struct Candidate {
    int32_t j;
    int32_t start_i;
  };
  std::deque<Candidate> dq;
  dq.push_back(Candidate{0, 0});

  for (int32_t i = 0; i < static_cast<int32_t>(num_bins); i++) {
    while (dq.size() >= 2 && dq[1].start_i <= i) {
      dq.pop_front();
    }
    int32_t best_j = dq.front().j;
    float cost = eval_outer_cost(best_j, i);
    dp[i] = (best_j == 0 ? cost
                         : dp[best_j - 1] + cost +
                               split_penaltyA(cand_cutoffsA[best_j - 1]));
    opt_prev[i] = (best_j == 0 ? -1 : best_j - 1);

    int32_t j_new = i + 1;
    if (j_new < static_cast<int32_t>(num_bins)) {
      while (!dq.empty()) {
        int32_t j_back = dq.back().j;
        int32_t start_back = dq.back().start_i;
        if (start_back >= j_new) {
          if (val(j_new, start_back) <= val(j_back, start_back)) {
            dq.pop_back();
            continue;
          }
        }
        int32_t low = std::max(start_back + 1, j_new);
        if (low >= static_cast<int32_t>(num_bins)) {
          break;
        }
        if (val(j_new, num_bins - 1) > val(j_back, num_bins - 1)) {
          break;
        }
        int32_t l = low;
        int32_t r = num_bins - 1;
        int32_t t_star = num_bins;
        while (l <= r) {
          int32_t mid = (l + r) / 2;
          if (val(j_new, mid) <= val(j_back, mid)) {
            t_star = mid;
            r = mid - 1;
          } else {
            l = mid + 1;
          }
        }
        if (t_star < static_cast<int32_t>(num_bins)) {
          dq.push_back(Candidate{j_new, t_star});
        }
        break;
      }
      if (dq.empty()) {
        dq.push_back(Candidate{j_new, j_new});
      }
    }
  }

  EvaluatedTripleTree res;
  res.pA = pA; res.pB = pB; res.pC = pC;
  res.cost = dp[num_bins - 1];

  std::vector<int32_t> chosen_cutsA;
  std::vector<NestedDPResult> chosen_subBC;
  int32_t curr = static_cast<int32_t>(num_bins - 1);
  while (curr != -1) {
    int32_t prev = opt_prev[curr];
    int32_t j = prev + 1;
    int32_t i = curr;

    size_t start_idx = bin_start[j];
    size_t end_idx = bin_start[i + 1];
    size_t K = end_idx - start_idx;
    size_t best_p = 0;
    float leaf_cost = get_leaf_costA(j, i, &best_p);
    float min_split_cost = base_node_cost * scale;

    NestedDPResult subBC;
    if (leaf_cost <= min_split_cost || K <= 16) {
      subBC.pA = pB;
      subBC.pB = pC;
      subBC.cost = leaf_cost;
      DP1DResult leaf_res;
      leaf_res.cost = leaf_cost;
      leaf_res.segment_preds = {best_p};
      subBC.segment_refinements = {std::move(leaf_res)};
    } else {
      subBC = solve_compact_BC(start_idx, end_idx, /*cost_only=*/false);
      if (leaf_cost < subBC.cost) {
        subBC.cost = leaf_cost;
        subBC.cutoffsA.clear();
        DP1DResult leaf_res;
        leaf_res.cost = leaf_cost;
        leaf_res.segment_preds = {best_p};
        subBC.segment_refinements = {std::move(leaf_res)};
      }
    }
    chosen_subBC.push_back(std::move(subBC));

    if (prev != -1) {
      chosen_cutsA.push_back(cand_cutoffsA[prev]);
    }
    curr = prev;
  }
  std::reverse(chosen_cutsA.begin(), chosen_cutsA.end());
  std::reverse(chosen_subBC.begin(), chosen_subBC.end());

  res.dpA.prop_dim = pA;
  res.dpA.cutoffs = std::move(chosen_cutsA);
  size_t num_segsA = chosen_subBC.size();
  res.dpA.segment_preds.resize(num_segsA, 0);
  res.refinementsB.resize(num_segsA);
  res.refinementsC.resize(num_segsA);

  for (size_t sA = 0; sA < num_segsA; sA++) {
    const auto& sub = chosen_subBC[sA];
    res.refinementsB[sA].prop_dim = pB;
    res.refinementsB[sA].cutoffs = sub.cutoffsA;
    size_t num_segsB = sub.cutoffsA.size() + 1;
    res.refinementsB[sA].segment_preds.resize(num_segsB, 0);
    res.refinementsC[sA] = sub.segment_refinements;
    if (res.refinementsC[sA].size() < num_segsB) {
      res.refinementsC[sA].resize(num_segsB);
    }
    for (size_t sB = 0; sB < num_segsB; sB++) {
      if (!res.refinementsC[sA][sB].segment_preds.empty()) {
        res.refinementsB[sA].segment_preds[sB] = res.refinementsC[sA][sB].segment_preds[0];
      }
    }
    if (!res.refinementsB[sA].segment_preds.empty()) {
      res.dpA.segment_preds[sA] = res.refinementsB[sA].segment_preds[0];
    }
  }

  return res;
}

void BuildTreeFrom3Prop(
    TreeSamples& tree_samples,
    const EvaluatedTripleTree& triple_tree,
    Tree* tree) {
  std::vector<size_t> seg_leaf_pos(triple_tree.refinementsB.size(), static_cast<size_t>(-1));
  BuildTreeFrom1D(triple_tree.pA + tree_samples.NumStaticProps(),
                  triple_tree.dpA.cutoffs, triple_tree.dpA.segment_preds,
                  tree_samples, tree, 0, &seg_leaf_pos);

  for (size_t sA = 0; sA < triple_tree.refinementsB.size(); sA++) {
    const auto& refB = triple_tree.refinementsB[sA];
    if (seg_leaf_pos[sA] == static_cast<size_t>(-1)) continue;

    if (refB.cutoffs.empty()) {
      if (sA < triple_tree.refinementsC.size() &&
          !triple_tree.refinementsC[sA].empty() &&
          !triple_tree.refinementsC[sA][0].cutoffs.empty()) {
        const auto& refC = triple_tree.refinementsC[sA][0];
        BuildTreeFrom1D(triple_tree.pC + tree_samples.NumStaticProps(),
                        refC.cutoffs, refC.segment_preds, tree_samples, tree,
                        seg_leaf_pos[sA]);
      }
      continue;
    }

    size_t posB = seg_leaf_pos[sA];
    std::vector<size_t> subseg_leaf_pos(refB.cutoffs.size() + 1, static_cast<size_t>(-1));
    BuildTreeFrom1D(triple_tree.pB + tree_samples.NumStaticProps(),
                    refB.cutoffs, refB.segment_preds, tree_samples, tree,
                    posB, &subseg_leaf_pos);

    if (sA < triple_tree.refinementsC.size()) {
      for (size_t sB = 0; sB < triple_tree.refinementsC[sA].size(); sB++) {
        const auto& refC = triple_tree.refinementsC[sA][sB];
        if (!refC.cutoffs.empty() && sB < subseg_leaf_pos.size() &&
            subseg_leaf_pos[sB] != static_cast<size_t>(-1)) {
          BuildTreeFrom1D(triple_tree.pC + tree_samples.NumStaticProps(),
                          refC.cutoffs, refC.segment_preds, tree_samples, tree,
                          subseg_leaf_pos[sB]);
        }
      }
    }
  }
}

void FindBestTree3PropDP(TreeSamples& tree_samples, float scale, Tree* tree,
                         const std::vector<uint32_t>* sample_subset = nullptr,
                         float base_node_cost = 92.0f, float log_node_cost = 1.2f,
                         ModularOptions::TreeLearningMode mode = ModularOptions::TreeLearningMode::k3PropertyDP) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props == 0 || total_samples == 0) {
    (*tree)[0] = PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0));
    return;
  }
  if (num_props == 1) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }
  if (num_props == 2) {
    FindBestTree2PropDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  // 1D screening via prefix greedy to get <= 4 coarse cutoffs per property.
  // No initial 1D DP.
  auto screen = ScreenAllProperties1D(tree_samples, sample_subset, scale, base_node_cost, log_node_cost, 4, /*refine_with_dp=*/false);
  auto primary_dps = std::move(screen.primary_dps);
  const auto& coarse_cuts = screen.coarse_cuts;

  size_t best_1d_dim = screen.best_1d_dim;
  float best_1d_cost = screen.best_1d_cost;

  size_t max_triples = 6;
  const char* env_max_t = getenv("JXL_DP_MAX_TRIPLES");
  if (env_max_t != nullptr) {
    max_triples = atoi(env_max_t);
  }

  // Screen all candidate triples unconditionally on the coarse grid (<= 5x5x5 cells).
  // Each triple is evaluated once, picking the best predictor per cell.
  // No pair screening, no 2D tree baseline.
  std::vector<ScoredTriple> top_triples = FastCoarseGridTripleScreening(
      tree_samples, sample_subset, coarse_cuts, scale, base_node_cost, log_node_cost, max_triples);

  if (top_triples.empty()) {
    const auto& b1d = primary_dps[best_1d_dim];
    if (b1d.segment_preds.empty()) {
      (*tree)[0] = PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0));
      return;
    }
    BuildTreeFrom1D(b1d.prop_dim + tree_samples.NumStaticProps(),
                    b1d.cutoffs, b1d.segment_preds, tree_samples, tree, 0);
    return;
  }

  const char* env_n3 = getenv("JXL_NESTED3_MODE");
  if (env_n3 != nullptr) {
    if (strcmp(env_n3, "single") == 0 || strcmp(env_n3, "1") == 0) {
      mode = ModularOptions::TreeLearningMode::k3PropNestedSingle;
    } else if (strcmp(env_n3, "staged") == 0 || strcmp(env_n3, "2") == 0) {
      mode = ModularOptions::TreeLearningMode::k3PropNestedStaged;
    } else if (strcmp(env_n3, "double") == 0 || strcmp(env_n3, "3") == 0) {
      mode = ModularOptions::TreeLearningMode::k3PropNestedDouble;
    } else if (strcmp(env_n3, "rec") == 0 || strcmp(env_n3, "0") == 0) {
      mode = ModularOptions::TreeLearningMode::k3PropertyDP;
    }
  }

  EvaluatedTripleTree best_triple;
  best_triple.cost = best_1d_cost;

  if (mode == ModularOptions::TreeLearningMode::k3PropertyDP) {
    for (const auto& st : top_triples) {
      auto resABC = Evaluate3PropRecursiveTree(tree_samples, st.pA, st.pB, st.pC, primary_dps[st.pA],
                                              sample_subset, scale, base_node_cost, log_node_cost);
      if (resABC.cost < best_triple.cost) best_triple = std::move(resABC);

      auto resBAC = Evaluate3PropRecursiveTree(tree_samples, st.pB, st.pA, st.pC, primary_dps[st.pB],
                                              sample_subset, scale, base_node_cost, log_node_cost);
      if (resBAC.cost < best_triple.cost) best_triple = std::move(resBAC);

      auto resCAB = Evaluate3PropRecursiveTree(tree_samples, st.pC, st.pA, st.pB, primary_dps[st.pC],
                                              sample_subset, scale, base_node_cost, log_node_cost);
      if (resCAB.cost < best_triple.cost) best_triple = std::move(resCAB);
    }
  } else {
    // For nested 3-property modes (single, staged, double):
    // Directly run on the top-ranking triple from coarse screening!
    std::vector<size_t> sorted_props = {top_triples[0].pA, top_triples[0].pB, top_triples[0].pC};
    std::sort(sorted_props.begin(), sorted_props.end(), [&](size_t x, size_t y) {
      return primary_dps[x].cost < primary_dps[y].cost;
    });
    size_t targetA = sorted_props[0];
    size_t targetB = sorted_props[1];
    size_t targetC = sorted_props[2];
    if (getenv("JXL_DEBUG_TRIPLES")) {
      for (size_t i = 0; i < std::min<size_t>(top_triples.size(), 5); i++) {
        fprintf(stderr, "TOP_TRIPLE[%zu]: raw=(%zu, %zu, %zu) sorted_by_1d=(%zu, %zu, %zu) score=%f\n",
                i, top_triples[i].pA, top_triples[i].pB, top_triples[i].pC,
                targetA, targetB, targetC, top_triples[i].cost);
      }
    }

    if (mode == ModularOptions::TreeLearningMode::k3PropNestedSingle) {
      auto r1 = RunSinglyNested3PropDP(tree_samples, targetA, targetB, targetC, sample_subset, scale, base_node_cost, log_node_cost);
      if (r1.cost < best_triple.cost) best_triple = std::move(r1);
      auto r2 = RunSinglyNested3PropDP(tree_samples, targetB, targetA, targetC, sample_subset, scale, base_node_cost, log_node_cost);
      if (r2.cost < best_triple.cost) best_triple = std::move(r2);
    } else if (mode == ModularOptions::TreeLearningMode::k3PropNestedStaged) {
      size_t perm[6][3] = {
        {targetA, targetB, targetC},
        {targetA, targetC, targetB},
        {targetB, targetA, targetC},
        {targetB, targetC, targetA},
        {targetC, targetA, targetB},
        {targetC, targetB, targetA},
      };

      const char* env_all = getenv("JXL_STAGED_ALL_PERMS");
      if (env_all != nullptr) {
        for (int p = 0; p < 6; p++) {
          auto r = RunStagedNested3PropDP(tree_samples, perm[p][0], perm[p][1], perm[p][2],
                                          sample_subset, scale, base_node_cost, log_node_cost);
          if (r.cost < best_triple.cost) {
            best_triple = std::move(r);
          }
        }
      } else {
        // Fast screening to pick the best permutation:
        // Evaluate 1D greedy on the intervals of the root to choose the child ordering!
        size_t best_perm_idx = 0;
        float best_pair_cost = std::numeric_limits<float>::max();
        ScratchPairEvaluation scratch_pair;
        for (size_t p = 0; p < 6; p++) {
          size_t pA = perm[p][0];
          size_t pB = perm[p][1];
          auto pair_res = Evaluate2PropRecursiveTree(
              tree_samples, pA, pB, primary_dps[pA], sample_subset, scale,
              base_node_cost, log_node_cost, /*use_greedy_refinement=*/true,
              &screen.mpt, &scratch_pair);
          if (pair_res.cost < best_pair_cost) {
            best_pair_cost = pair_res.cost;
            best_perm_idx = p;
          }
        }

        auto r = RunStagedNested3PropDP(
            tree_samples, perm[best_perm_idx][0], perm[best_perm_idx][1], perm[best_perm_idx][2],
            sample_subset, scale, base_node_cost, log_node_cost);
        if (r.cost < best_triple.cost) {
          best_triple = std::move(r);
        }
      }
    } else if (mode == ModularOptions::TreeLearningMode::k3PropNestedDouble) {
      auto r1 = RunDoublyNested3PropDP(tree_samples, targetA, targetB, targetC, sample_subset, scale, base_node_cost, log_node_cost, &screen.mpt);
      if (r1.cost < best_triple.cost) best_triple = std::move(r1);
      auto r2 = RunDoublyNested3PropDP(tree_samples, targetB, targetA, targetC, sample_subset, scale, base_node_cost, log_node_cost, &screen.mpt);
      if (r2.cost < best_triple.cost) best_triple = std::move(r2);
    }
  }

  // If the 3-property tree beat the 1D greedy baseline, build it!
  if (!best_triple.refinementsC.empty() && best_triple.cost < best_1d_cost) {
    BuildTreeFrom3Prop(tree_samples, best_triple, tree);
    return;
  }

  // Otherwise fall back to best 1D greedy tree!
  const auto& b1d = primary_dps[best_1d_dim];
  if (b1d.segment_preds.empty()) {
    (*tree)[0] = PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0));
    return;
  }
  BuildTreeFrom1D(b1d.prop_dim + tree_samples.NumStaticProps(),
                  b1d.cutoffs, b1d.segment_preds, tree_samples, tree, 0);
}

void FindBestTreeJoint2dDP(TreeSamples& tree_samples, float scale, Tree* tree,
                           const std::vector<uint32_t>* sample_subset = nullptr,
                           float base_node_cost = 92.0f, float log_node_cost = 1.2f) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props == 0 || total_samples == 0) {
    (*tree)[0] = PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0));
    return;
  }
  if (num_props == 1) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  auto screen = ScreenAllProperties1D(tree_samples, sample_subset, scale, base_node_cost, log_node_cost, 4);
  auto primary_dps = std::move(screen.primary_dps);
  const auto& coarse_cuts = screen.coarse_cuts;
  size_t best_1d_dim = screen.best_1d_dim;
  float best_1d_cost = screen.best_1d_cost;

  size_t max_pairs = 4;
  if (num_props > 12) max_pairs = 20;
  else if (num_props > 8) max_pairs = 12;
  else if (num_props > 5) max_pairs = 8;
  else if (num_props > 4) max_pairs = 6;

  const char* env_pairs = getenv("JXL_NESTED_NUM_PAIRS");
  if (env_pairs != nullptr) {
    max_pairs = atoi(env_pairs);
  }

  std::vector<ScoredPair> top_pairs = FastCoarseGridPairScreening(
      tree_samples, sample_subset, coarse_cuts, scale, base_node_cost, log_node_cost, max_pairs);

  NestedDPResult best_joint;
  best_joint.cost = best_1d_cost;

  EvaluatedPairTree best_rec_pair;
  best_rec_pair.cost = best_1d_cost;
  ScratchPairEvaluation scratch_pair;
  for (const auto& sp : top_pairs) {
    auto resAB = Evaluate2PropRecursiveTree(tree_samples, sp.pA, sp.pB, primary_dps[sp.pA],
                                           sample_subset, scale, base_node_cost, log_node_cost,
                                           /*use_greedy_refinement=*/true, &screen.mpt, &scratch_pair);
    if (resAB.cost < best_rec_pair.cost) best_rec_pair = std::move(resAB);

    auto resBA = Evaluate2PropRecursiveTree(tree_samples, sp.pB, sp.pA, primary_dps[sp.pB],
                                           sample_subset, scale, base_node_cost, log_node_cost,
                                           /*use_greedy_refinement=*/true, &screen.mpt, &scratch_pair);
    if (resBA.cost < best_rec_pair.cost) best_rec_pair = std::move(resBA);
  }

  std::vector<std::pair<size_t, size_t>> pairs_to_try;
  auto add_pair = [&](size_t a, size_t b) {
    if (a == b) return;
    for (const auto& p : pairs_to_try) {
      if (p.first == a && p.second == b) return;
    }
    pairs_to_try.push_back({a, b});
  };

  if (!best_rec_pair.refinementsB.empty()) {
    add_pair(best_rec_pair.pA, best_rec_pair.pB);
    if (num_props > 8) {
      add_pair(best_rec_pair.pB, best_rec_pair.pA);
    }
  } else if (!top_pairs.empty()) {
    add_pair(top_pairs[0].pA, top_pairs[0].pB);
    add_pair(top_pairs[0].pB, top_pairs[0].pA);
  }

  for (const auto& pair : pairs_to_try) {
    NestedDPResult joint = RunNested2dDP(tree_samples, pair.first, pair.second,
                                         sample_subset, scale, base_node_cost, log_node_cost,
                                         nullptr, &screen.mpt.max_symbols, screen.mpt.S);
    if (joint.cost < best_joint.cost) {
      best_joint = std::move(joint);
    }
  }

  if (getenv("JXL_LOG_WINNING_PAIRS") && !best_joint.segment_refinements.empty() && best_joint.cost < best_1d_cost) {
    uint32_t globA = tree_samples.PropertyFromIndex(best_joint.pA + tree_samples.NumStaticProps());
    uint32_t globB = tree_samples.PropertyFromIndex(best_joint.pB + tree_samples.NumStaticProps());
    uint32_t top0 = tree_samples.PropertyFromIndex(best_1d_dim + tree_samples.NumStaticProps());
    fprintf(stderr, "WINNING_PAIR: glob=(%u, %u) top1d=%u\n", globA, globB, top0);
  }

  // If no joint DP beat best 1D DP, fall back to best 1D DP!
  if (best_joint.segment_refinements.empty() || best_joint.cost >= best_1d_cost) {
    const auto& b1d = primary_dps[best_1d_dim];
    if (b1d.segment_preds.empty()) {
      (*tree)[0] = PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0));
      return;
    }
    BuildTreeFrom1D(b1d.prop_dim + tree_samples.NumStaticProps(),
                    b1d.cutoffs, b1d.segment_preds, tree_samples, tree, 0);
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

struct BoxLeafInfo {
  float cost = 0.0f;
  size_t best_pred = 0;
  size_t total_count = 0;
};

inline BoxLeafInfo GetBoxLeafInfo2D(
    uint16_t a0, uint16_t a1, uint16_t b0, uint16_t b1,
    size_t num_binsB, size_t P, size_t S,
    const std::vector<size_t>& max_symbols,
    const int32_t* flat_cell_hists,
    const int64_t* flat_cell_extra,
    const size_t* cell_sample_count,
    const float* single_cell_bits,
    const uint8_t* single_cell_pred) {
  size_t total_cnt = 0;
  size_t non_empty_cells = 0;
  size_t single_cell_idx = 0;

  for (size_t a = a0; a < a1; a++) {
    for (size_t b = b0; b < b1; b++) {
      size_t cell = a * num_binsB + b;
      size_t cnt = cell_sample_count[cell];
      if (cnt > 0) {
        total_cnt += cnt;
        non_empty_cells++;
        single_cell_idx = cell;
      }
    }
  }

  if (total_cnt == 0) {
    return {0.0f, 0, 0};
  }

  if (non_empty_cells == 1) {
    return {single_cell_bits[single_cell_idx], single_cell_pred[single_cell_idx], total_cnt};
  }

  std::vector<int32_t> comb_h(P * S, 0);
  std::vector<int64_t> comb_e(P, 0);

  for (size_t a = a0; a < a1; a++) {
    for (size_t b = b0; b < b1; b++) {
      size_t cell = a * num_binsB + b;
      if (cell_sample_count[cell] == 0) continue;
      for (size_t p = 0; p < P; p++) {
        comb_e[p] += flat_cell_extra[cell * P + p];
        for (size_t s = 0; s < max_symbols[p]; s++) {
          comb_h[p * S + s] += flat_cell_hists[(cell * P + p) * S + s];
        }
      }
    }
  }

  float best_bits = std::numeric_limits<float>::max();
  size_t best_p = 0;
  for (size_t p = 0; p < P; p++) {
    float bits = EstimateBits(&comb_h[p * S], max_symbols[p]) + comb_e[p];
    if (bits < best_bits) {
      best_bits = bits;
      best_p = p;
    }
  }
  return {best_bits, best_p, total_cnt};
}

struct GridBox2D {
  uint16_t a0, a1;
  uint16_t b0, b1;
};

inline void BuildTreeGreedy2D(
    GridBox2D box,
    size_t propA_idx, const std::vector<int32_t>& cutoffsA, const std::vector<float>& cut_penaltiesA,
    size_t propB_idx, const std::vector<int32_t>& cutoffsB, const std::vector<float>& cut_penaltiesB,
    size_t num_binsB, size_t P, size_t S,
    const std::vector<size_t>& max_symbols,
    const int32_t* flat_cell_hists,
    const int64_t* flat_cell_extra,
    const size_t* cell_sample_count,
    const float* single_cell_bits,
    const uint8_t* single_cell_pred,
    TreeSamples& tree_samples, Tree* tree, size_t node_pos) {

  BoxLeafInfo leaf = GetBoxLeafInfo2D(
      box.a0, box.a1, box.b0, box.b1, num_binsB, P, S, max_symbols,
      flat_cell_hists, flat_cell_extra, cell_sample_count,
      single_cell_bits, single_cell_pred);

  if (leaf.total_count == 0) {
    (*tree)[node_pos] = PropertyDecisionNode::Leaf(Predictor::Zero);
    return;
  }

  float best_net_gain = 0.0f;
  int best_dim = -1; // 0 for A, 1 for B
  uint16_t best_cut = 0;

  // Try cutoffs of A
  for (uint16_t a = box.a0 + 1; a < box.a1; a++) {
    BoxLeafInfo left = GetBoxLeafInfo2D(
        a, box.a1, box.b0, box.b1, num_binsB, P, S, max_symbols,
        flat_cell_hists, flat_cell_extra, cell_sample_count,
        single_cell_bits, single_cell_pred);
    if (left.total_count == 0) continue;

    BoxLeafInfo right = GetBoxLeafInfo2D(
        box.a0, a, box.b0, box.b1, num_binsB, P, S, max_symbols,
        flat_cell_hists, flat_cell_extra, cell_sample_count,
        single_cell_bits, single_cell_pred);
    if (right.total_count == 0) continue;

    float pen = (a - 1 < cut_penaltiesA.size()) ? cut_penaltiesA[a - 1] : 0.0f;
    float net_gain = leaf.cost - (left.cost + right.cost + pen);
    if (net_gain > best_net_gain) {
      best_net_gain = net_gain;
      best_dim = 0;
      best_cut = a;
    }
  }

  // Try cutoffs of B
  for (uint16_t b = box.b0 + 1; b < box.b1; b++) {
    BoxLeafInfo left = GetBoxLeafInfo2D(
        box.a0, box.a1, b, box.b1, num_binsB, P, S, max_symbols,
        flat_cell_hists, flat_cell_extra, cell_sample_count,
        single_cell_bits, single_cell_pred);
    if (left.total_count == 0) continue;

    BoxLeafInfo right = GetBoxLeafInfo2D(
        box.a0, box.a1, box.b0, b, num_binsB, P, S, max_symbols,
        flat_cell_hists, flat_cell_extra, cell_sample_count,
        single_cell_bits, single_cell_pred);
    if (right.total_count == 0) continue;

    float pen = (b - 1 < cut_penaltiesB.size()) ? cut_penaltiesB[b - 1] : 0.0f;
    float net_gain = leaf.cost - (left.cost + right.cost + pen);
    if (net_gain > best_net_gain) {
      best_net_gain = net_gain;
      best_dim = 1;
      best_cut = b;
    }
  }

  if (best_dim == -1 || best_net_gain <= 0.0f) {
    (*tree)[node_pos] = PropertyDecisionNode::Leaf(
        tree_samples.PredictorFromIndex(leaf.best_pred));
    return;
  }

  size_t prop_idx = (best_dim == 0) ? propA_idx : propB_idx;
  const auto& cutoffs = (best_dim == 0) ? cutoffsA : cutoffsB;
  size_t cut_idx = best_cut - 1;
  int32_t raw_cut = cutoffs[cut_idx];
  int32_t unquant = tree_samples.UnquantizeProperty(prop_idx, raw_cut);
  int32_t property = tree_samples.PropertyFromIndex(prop_idx);

  uint32_t lchild = tree->size();
  uint32_t rchild = tree->size() + 1;
  (*tree)[node_pos] = PropertyDecisionNode::Split(property, unquant, lchild, rchild);
  tree->push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
  tree->push_back(PropertyDecisionNode::Leaf(Predictor::Zero));

  if (best_dim == 0) {
    BuildTreeGreedy2D({best_cut, box.a1, box.b0, box.b1},
                     propA_idx, cutoffsA, cut_penaltiesA,
                     propB_idx, cutoffsB, cut_penaltiesB,
                     num_binsB, P, S, max_symbols,
                     flat_cell_hists, flat_cell_extra, cell_sample_count,
                     single_cell_bits, single_cell_pred,
                     tree_samples, tree, lchild);
    BuildTreeGreedy2D({box.a0, best_cut, box.b0, box.b1},
                     propA_idx, cutoffsA, cut_penaltiesA,
                     propB_idx, cutoffsB, cut_penaltiesB,
                     num_binsB, P, S, max_symbols,
                     flat_cell_hists, flat_cell_extra, cell_sample_count,
                     single_cell_bits, single_cell_pred,
                     tree_samples, tree, rchild);
  } else {
    BuildTreeGreedy2D({box.a0, box.a1, best_cut, box.b1},
                     propA_idx, cutoffsA, cut_penaltiesA,
                     propB_idx, cutoffsB, cut_penaltiesB,
                     num_binsB, P, S, max_symbols,
                     flat_cell_hists, flat_cell_extra, cell_sample_count,
                     single_cell_bits, single_cell_pred,
                     tree_samples, tree, lchild);
    BuildTreeGreedy2D({box.a0, box.a1, box.b0, best_cut},
                     propA_idx, cutoffsA, cut_penaltiesA,
                     propB_idx, cutoffsB, cut_penaltiesB,
                     num_binsB, P, S, max_symbols,
                     flat_cell_hists, flat_cell_extra, cell_sample_count,
                     single_cell_bits, single_cell_pred,
                     tree_samples, tree, rchild);
  }
}

void FindBestTreeGrid2dDP(TreeSamples& tree_samples, float scale, Tree* tree,
                          const std::vector<uint32_t>* sample_subset = nullptr,
                          float base_node_cost = 92.0f, float log_node_cost = 1.2f,
                          size_t default_cuts = 7) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props <= 1 || total_samples == 0) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  size_t max_grid_cuts = default_cuts;
  const char* env_cuts = getenv("JXL_GRID_CUTS");
  if (env_cuts) max_grid_cuts = std::max<size_t>(1, std::min<size_t>(15, atoi(env_cuts)));

  auto screen = ScreenAllProperties1D(tree_samples, sample_subset, scale, base_node_cost, log_node_cost, max_grid_cuts, /*refine_with_dp=*/false);
  const auto& coarse_cuts = screen.coarse_cuts;

  std::vector<size_t> valid_props;
  for (size_t p = 0; p < num_props; p++) {
    if (!coarse_cuts[p].empty()) valid_props.push_back(p);
  }
  if (valid_props.size() <= 1) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  std::vector<std::vector<float>> cut_penalties(num_props);
  for (size_t p = 0; p < num_props; p++) {
    for (int32_t c : coarse_cuts[p]) {
      int32_t unquant = tree_samples.UnquantizeProperty(p + tree_samples.NumStaticProps(), c);
      cut_penalties[p].push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
    }
  }

  std::vector<std::vector<uint8_t>> sample_bins(num_props, std::vector<uint8_t>(total_samples));
  for (size_t p : valid_props) {
    const auto& cuts = coarse_cuts[p];
    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      int32_t v = tree_samples.Property<false>(p, s);
      size_t b = 0;
      while (b < cuts.size() && v > cuts[b]) {
        b++;
      }
      sample_bins[p][i] = static_cast<uint8_t>(b);
    }
  }

  const size_t P = tree_samples.NumPredictors();
  const size_t S = screen.mpt.S;
  const auto& max_symbols = screen.mpt.max_symbols;

  size_t best_pA = valid_props[0], best_pB = valid_props[1];
  float best_grid_cost = std::numeric_limits<float>::max();

  std::vector<int32_t> flat_cell_hists;
  std::vector<int64_t> flat_cell_extra;
  std::vector<size_t> cell_sample_count;

  for (size_t idxA = 0; idxA < valid_props.size(); idxA++) {
    size_t pA = valid_props[idxA];
    size_t num_binsA = coarse_cuts[pA].size() + 1;
    for (size_t idxB = idxA + 1; idxB < valid_props.size(); idxB++) {
      size_t pB = valid_props[idxB];
      size_t num_binsB = coarse_cuts[pB].size() + 1;
      size_t num_cells = num_binsA * num_binsB;
      size_t num_entries = num_cells * P;

      if (flat_cell_hists.size() < num_entries * S) flat_cell_hists.resize(num_entries * S);
      if (flat_cell_extra.size() < num_entries) flat_cell_extra.resize(num_entries);
      if (cell_sample_count.size() < num_cells) cell_sample_count.resize(num_cells);

      std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * S, 0);
      std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
      std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

      for (size_t i = 0; i < total_samples; i++) {
        size_t s = get_sample_idx(i);
        size_t bA = sample_bins[pA][i];
        size_t bB = sample_bins[pB][i];
        size_t cell = bA * num_binsB + bB;
        cell_sample_count[cell]++;
        size_t cnt = tree_samples.Count(s);
        for (size_t pred = 0; pred < P; pred++) {
          uint32_t tok = tree_samples.Token(pred, s);
          size_t offset = (cell * P + pred) * S;
          flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
          flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
        }
      }

      float grid_entropy = 0.0f;
      for (size_t c = 0; c < num_cells; c++) {
        if (cell_sample_count[c] == 0) continue;
        float best_cell_bits = std::numeric_limits<float>::max();
        for (size_t pred = 0; pred < P; pred++) {
          float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                       flat_cell_extra[c * P + pred];
          if (bits < best_cell_bits) best_cell_bits = bits;
        }
        grid_entropy += best_cell_bits;
      }

      float penA = 0.0f; for (float pen : cut_penalties[pA]) penA += pen;
      float penB = 0.0f; for (float pen : cut_penalties[pB]) penB += pen;
      float cost = grid_entropy + penA + penB;

      if (cost < best_grid_cost) {
        best_grid_cost = cost;
        best_pA = pA;
        best_pB = pB;
      }
    }
  }

  // Populate winning 2D grid
  size_t pA = best_pA;
  size_t pB = best_pB;
  size_t num_binsA = coarse_cuts[pA].size() + 1;
  size_t num_binsB = coarse_cuts[pB].size() + 1;
  size_t num_cells = num_binsA * num_binsB;
  size_t num_entries = num_cells * P;

  std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * S, 0);
  std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
  std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    size_t bA = sample_bins[pA][i];
    size_t bB = sample_bins[pB][i];
    size_t cell = bA * num_binsB + bB;
    cell_sample_count[cell]++;
    size_t cnt = tree_samples.Count(s);
    for (size_t pred = 0; pred < P; pred++) {
      uint32_t tok = tree_samples.Token(pred, s);
      size_t offset = (cell * P + pred) * S;
      flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
      flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
    }
  }

  std::vector<float> single_cell_bits(num_cells, 0.0f);
  std::vector<uint8_t> single_cell_pred(num_cells, 0);
  for (size_t c = 0; c < num_cells; c++) {
    if (cell_sample_count[c] == 0) continue;
    float best_bits = std::numeric_limits<float>::max();
    size_t best_p = 0;
    for (size_t pred = 0; pred < P; pred++) {
      float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                   flat_cell_extra[c * P + pred];
      if (bits < best_bits) {
        best_bits = bits;
        best_p = pred;
      }
    }
    single_cell_bits[c] = best_bits;
    single_cell_pred[c] = static_cast<uint8_t>(best_p);
  }

  tree->clear();
  tree->emplace_back();
  (*tree)[0] = PropertyDecisionNode::Leaf(Predictor::Zero);

  BuildTreeGreedy2D({0, static_cast<uint16_t>(num_binsA), 0, static_cast<uint16_t>(num_binsB)},
                    pA + tree_samples.NumStaticProps(), coarse_cuts[pA], cut_penalties[pA],
                    pB + tree_samples.NumStaticProps(), coarse_cuts[pB], cut_penalties[pB],
                    num_binsB, P, S, max_symbols,
                    flat_cell_hists.data(), flat_cell_extra.data(), cell_sample_count.data(),
                    single_cell_bits.data(), single_cell_pred.data(),
                    tree_samples, tree, 0);
}

inline BoxLeafInfo GetBoxLeafInfo3D(
    uint16_t a0, uint16_t a1, uint16_t b0, uint16_t b1, uint16_t c0, uint16_t c1,
    size_t num_binsB, size_t num_binsC, size_t P, size_t S,
    const std::vector<size_t>& max_symbols,
    const int32_t* flat_cell_hists,
    const int64_t* flat_cell_extra,
    const size_t* cell_sample_count,
    const float* single_cell_bits,
    const uint8_t* single_cell_pred) {
  size_t total_cnt = 0;
  size_t non_empty_cells = 0;
  size_t single_cell_idx = 0;

  for (size_t a = a0; a < a1; a++) {
    for (size_t b = b0; b < b1; b++) {
      for (size_t c = c0; c < c1; c++) {
        size_t cell = (a * num_binsB + b) * num_binsC + c;
        size_t cnt = cell_sample_count[cell];
        if (cnt > 0) {
          total_cnt += cnt;
          non_empty_cells++;
          single_cell_idx = cell;
        }
      }
    }
  }

  if (total_cnt == 0) {
    return {0.0f, 0, 0};
  }

  if (non_empty_cells == 1) {
    return {single_cell_bits[single_cell_idx], single_cell_pred[single_cell_idx], total_cnt};
  }

  std::vector<int32_t> comb_h(P * S, 0);
  std::vector<int64_t> comb_e(P, 0);

  for (size_t a = a0; a < a1; a++) {
    for (size_t b = b0; b < b1; b++) {
      for (size_t c = c0; c < c1; c++) {
        size_t cell = (a * num_binsB + b) * num_binsC + c;
        if (cell_sample_count[cell] == 0) continue;
        for (size_t p = 0; p < P; p++) {
          comb_e[p] += flat_cell_extra[cell * P + p];
          for (size_t s = 0; s < max_symbols[p]; s++) {
            comb_h[p * S + s] += flat_cell_hists[(cell * P + p) * S + s];
          }
        }
      }
    }
  }

  float best_bits = std::numeric_limits<float>::max();
  size_t best_p = 0;
  for (size_t p = 0; p < P; p++) {
    float bits = EstimateBits(&comb_h[p * S], max_symbols[p]) + comb_e[p];
    if (bits < best_bits) {
      best_bits = bits;
      best_p = p;
    }
  }
  return {best_bits, best_p, total_cnt};
}

struct GridBox3D {
  uint16_t a0, a1;
  uint16_t b0, b1;
  uint16_t c0, c1;
};

inline void BuildTreeGreedy3D(
    GridBox3D box,
    size_t propA_idx, const std::vector<int32_t>& cutoffsA, const std::vector<float>& cut_penaltiesA,
    size_t propB_idx, const std::vector<int32_t>& cutoffsB, const std::vector<float>& cut_penaltiesB,
    size_t propC_idx, const std::vector<int32_t>& cutoffsC, const std::vector<float>& cut_penaltiesC,
    size_t num_binsB, size_t num_binsC, size_t P, size_t S,
    const std::vector<size_t>& max_symbols,
    const int32_t* flat_cell_hists,
    const int64_t* flat_cell_extra,
    const size_t* cell_sample_count,
    const float* single_cell_bits,
    const uint8_t* single_cell_pred,
    TreeSamples& tree_samples, Tree* tree, size_t node_pos) {

  BoxLeafInfo leaf = GetBoxLeafInfo3D(
      box.a0, box.a1, box.b0, box.b1, box.c0, box.c1,
      num_binsB, num_binsC, P, S, max_symbols,
      flat_cell_hists, flat_cell_extra, cell_sample_count,
      single_cell_bits, single_cell_pred);

  if (leaf.total_count == 0) {
    (*tree)[node_pos] = PropertyDecisionNode::Leaf(Predictor::Zero);
    return;
  }

  float best_net_gain = 0.0f;
  int best_dim = -1; // 0 for A, 1 for B, 2 for C
  uint16_t best_cut = 0;

  // Try cutoffs of A
  for (uint16_t a = box.a0 + 1; a < box.a1; a++) {
    BoxLeafInfo left = GetBoxLeafInfo3D(
        a, box.a1, box.b0, box.b1, box.c0, box.c1,
        num_binsB, num_binsC, P, S, max_symbols,
        flat_cell_hists, flat_cell_extra, cell_sample_count,
        single_cell_bits, single_cell_pred);
    if (left.total_count == 0) continue;

    BoxLeafInfo right = GetBoxLeafInfo3D(
        box.a0, a, box.b0, box.b1, box.c0, box.c1,
        num_binsB, num_binsC, P, S, max_symbols,
        flat_cell_hists, flat_cell_extra, cell_sample_count,
        single_cell_bits, single_cell_pred);
    if (right.total_count == 0) continue;

    float pen = (a - 1 < cut_penaltiesA.size()) ? cut_penaltiesA[a - 1] : 0.0f;
    float net_gain = leaf.cost - (left.cost + right.cost + pen);
    if (net_gain > best_net_gain) {
      best_net_gain = net_gain;
      best_dim = 0;
      best_cut = a;
    }
  }

  // Try cutoffs of B
  for (uint16_t b = box.b0 + 1; b < box.b1; b++) {
    BoxLeafInfo left = GetBoxLeafInfo3D(
        box.a0, box.a1, b, box.b1, box.c0, box.c1,
        num_binsB, num_binsC, P, S, max_symbols,
        flat_cell_hists, flat_cell_extra, cell_sample_count,
        single_cell_bits, single_cell_pred);
    if (left.total_count == 0) continue;

    BoxLeafInfo right = GetBoxLeafInfo3D(
        box.a0, box.a1, box.b0, b, box.c0, box.c1,
        num_binsB, num_binsC, P, S, max_symbols,
        flat_cell_hists, flat_cell_extra, cell_sample_count,
        single_cell_bits, single_cell_pred);
    if (right.total_count == 0) continue;

    float pen = (b - 1 < cut_penaltiesB.size()) ? cut_penaltiesB[b - 1] : 0.0f;
    float net_gain = leaf.cost - (left.cost + right.cost + pen);
    if (net_gain > best_net_gain) {
      best_net_gain = net_gain;
      best_dim = 1;
      best_cut = b;
    }
  }

  // Try cutoffs of C
  for (uint16_t c = box.c0 + 1; c < box.c1; c++) {
    BoxLeafInfo left = GetBoxLeafInfo3D(
        box.a0, box.a1, box.b0, box.b1, c, box.c1,
        num_binsB, num_binsC, P, S, max_symbols,
        flat_cell_hists, flat_cell_extra, cell_sample_count,
        single_cell_bits, single_cell_pred);
    if (left.total_count == 0) continue;

    BoxLeafInfo right = GetBoxLeafInfo3D(
        box.a0, box.a1, box.b0, box.b1, box.c0, c,
        num_binsB, num_binsC, P, S, max_symbols,
        flat_cell_hists, flat_cell_extra, cell_sample_count,
        single_cell_bits, single_cell_pred);
    if (right.total_count == 0) continue;

    float pen = (c - 1 < cut_penaltiesC.size()) ? cut_penaltiesC[c - 1] : 0.0f;
    float net_gain = leaf.cost - (left.cost + right.cost + pen);
    if (net_gain > best_net_gain) {
      best_net_gain = net_gain;
      best_dim = 2;
      best_cut = c;
    }
  }

  if (best_dim == -1 || best_net_gain <= 0.0f) {
    (*tree)[node_pos] = PropertyDecisionNode::Leaf(
        tree_samples.PredictorFromIndex(leaf.best_pred));
    return;
  }

  size_t prop_idx = (best_dim == 0) ? propA_idx :
                    (best_dim == 1) ? propB_idx : propC_idx;
  const auto& cutoffs = (best_dim == 0) ? cutoffsA :
                        (best_dim == 1) ? cutoffsB : cutoffsC;
  size_t cut_idx = best_cut - 1;
  int32_t raw_cut = cutoffs[cut_idx];
  int32_t unquant = tree_samples.UnquantizeProperty(prop_idx, raw_cut);
  int32_t property = tree_samples.PropertyFromIndex(prop_idx);

  uint32_t lchild = tree->size();
  uint32_t rchild = tree->size() + 1;
  (*tree)[node_pos] = PropertyDecisionNode::Split(property, unquant, lchild, rchild);
  tree->push_back(PropertyDecisionNode::Leaf(Predictor::Zero));
  tree->push_back(PropertyDecisionNode::Leaf(Predictor::Zero));

  if (best_dim == 0) {
    BuildTreeGreedy3D({best_cut, box.a1, box.b0, box.b1, box.c0, box.c1},
                     propA_idx, cutoffsA, cut_penaltiesA,
                     propB_idx, cutoffsB, cut_penaltiesB,
                     propC_idx, cutoffsC, cut_penaltiesC,
                     num_binsB, num_binsC, P, S, max_symbols,
                     flat_cell_hists, flat_cell_extra, cell_sample_count,
                     single_cell_bits, single_cell_pred,
                     tree_samples, tree, lchild);
    BuildTreeGreedy3D({box.a0, best_cut, box.b0, box.b1, box.c0, box.c1},
                     propA_idx, cutoffsA, cut_penaltiesA,
                     propB_idx, cutoffsB, cut_penaltiesB,
                     propC_idx, cutoffsC, cut_penaltiesC,
                     num_binsB, num_binsC, P, S, max_symbols,
                     flat_cell_hists, flat_cell_extra, cell_sample_count,
                     single_cell_bits, single_cell_pred,
                     tree_samples, tree, rchild);
  } else if (best_dim == 1) {
    BuildTreeGreedy3D({box.a0, box.a1, best_cut, box.b1, box.c0, box.c1},
                     propA_idx, cutoffsA, cut_penaltiesA,
                     propB_idx, cutoffsB, cut_penaltiesB,
                     propC_idx, cutoffsC, cut_penaltiesC,
                     num_binsB, num_binsC, P, S, max_symbols,
                     flat_cell_hists, flat_cell_extra, cell_sample_count,
                     single_cell_bits, single_cell_pred,
                     tree_samples, tree, lchild);
    BuildTreeGreedy3D({box.a0, box.a1, box.b0, best_cut, box.c0, box.c1},
                     propA_idx, cutoffsA, cut_penaltiesA,
                     propB_idx, cutoffsB, cut_penaltiesB,
                     propC_idx, cutoffsC, cut_penaltiesC,
                     num_binsB, num_binsC, P, S, max_symbols,
                     flat_cell_hists, flat_cell_extra, cell_sample_count,
                     single_cell_bits, single_cell_pred,
                     tree_samples, tree, rchild);
  } else {
    BuildTreeGreedy3D({box.a0, box.a1, box.b0, box.b1, best_cut, box.c1},
                     propA_idx, cutoffsA, cut_penaltiesA,
                     propB_idx, cutoffsB, cut_penaltiesB,
                     propC_idx, cutoffsC, cut_penaltiesC,
                     num_binsB, num_binsC, P, S, max_symbols,
                     flat_cell_hists, flat_cell_extra, cell_sample_count,
                     single_cell_bits, single_cell_pred,
                     tree_samples, tree, lchild);
    BuildTreeGreedy3D({box.a0, box.a1, box.b0, box.b1, box.c0, best_cut},
                     propA_idx, cutoffsA, cut_penaltiesA,
                     propB_idx, cutoffsB, cut_penaltiesB,
                     propC_idx, cutoffsC, cut_penaltiesC,
                     num_binsB, num_binsC, P, S, max_symbols,
                     flat_cell_hists, flat_cell_extra, cell_sample_count,
                     single_cell_bits, single_cell_pred,
                     tree_samples, tree, rchild);
  }
}

void FindBestTreeGrid3dDP(TreeSamples& tree_samples, float scale, Tree* tree,
                          const std::vector<uint32_t>* sample_subset = nullptr,
                          float base_node_cost = 92.0f, float log_node_cost = 1.2f,
                          size_t default_cuts = 7) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props < 3 || total_samples == 0) {
    FindBestTreeGrid2dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost, default_cuts);
    return;
  }

  size_t max_grid_cuts = default_cuts;
  const char* env_cuts = getenv("JXL_GRID_CUTS");
  if (env_cuts) max_grid_cuts = std::max<size_t>(1, std::min<size_t>(15, atoi(env_cuts)));

  auto screen = ScreenAllProperties1D(tree_samples, sample_subset, scale, base_node_cost, log_node_cost, max_grid_cuts, /*refine_with_dp=*/false);
  const auto& coarse_cuts = screen.coarse_cuts;

  std::vector<size_t> valid_props;
  for (size_t p = 0; p < num_props; p++) {
    if (!coarse_cuts[p].empty()) valid_props.push_back(p);
  }
  if (valid_props.size() < 3) {
    FindBestTreeGrid2dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  // Sort valid props by 1D cost, take top 8
  std::sort(valid_props.begin(), valid_props.end(), [&](size_t a, size_t b) {
    return screen.primary_dps[a].cost < screen.primary_dps[b].cost;
  });
  if (valid_props.size() > 8) {
    valid_props.resize(8);
  }

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  std::vector<std::vector<float>> cut_penalties(num_props);
  for (size_t p = 0; p < num_props; p++) {
    for (int32_t c : coarse_cuts[p]) {
      int32_t unquant = tree_samples.UnquantizeProperty(p + tree_samples.NumStaticProps(), c);
      cut_penalties[p].push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
    }
  }

  std::vector<std::vector<uint8_t>> sample_bins(num_props, std::vector<uint8_t>(total_samples));
  for (size_t p : valid_props) {
    const auto& cuts = coarse_cuts[p];
    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      int32_t v = tree_samples.Property<false>(p, s);
      size_t b = 0;
      while (b < cuts.size() && v > cuts[b]) {
        b++;
      }
      sample_bins[p][i] = static_cast<uint8_t>(b);
    }
  }

  const size_t P = tree_samples.NumPredictors();
  const size_t S = screen.mpt.S;
  const auto& max_symbols = screen.mpt.max_symbols;

  size_t best_pA = valid_props[0], best_pB = valid_props[1], best_pC = valid_props[2];
  float best_grid_cost = std::numeric_limits<float>::max();

  std::vector<int32_t> flat_cell_hists;
  std::vector<int64_t> flat_cell_extra;
  std::vector<size_t> cell_sample_count;

  for (size_t idxA = 0; idxA < valid_props.size(); idxA++) {
    size_t pA = valid_props[idxA];
    size_t num_binsA = coarse_cuts[pA].size() + 1;
    for (size_t idxB = idxA + 1; idxB < valid_props.size(); idxB++) {
      size_t pB = valid_props[idxB];
      size_t num_binsB = coarse_cuts[pB].size() + 1;
      for (size_t idxC = idxB + 1; idxC < valid_props.size(); idxC++) {
        size_t pC = valid_props[idxC];
        size_t num_binsC = coarse_cuts[pC].size() + 1;
        size_t num_cells = num_binsA * num_binsB * num_binsC;
        size_t num_entries = num_cells * P;

        if (flat_cell_hists.size() < num_entries * S) flat_cell_hists.resize(num_entries * S);
        if (flat_cell_extra.size() < num_entries) flat_cell_extra.resize(num_entries);
        if (cell_sample_count.size() < num_cells) cell_sample_count.resize(num_cells);

        std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * S, 0);
        std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
        std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

        for (size_t i = 0; i < total_samples; i++) {
          size_t s = get_sample_idx(i);
          size_t bA = sample_bins[pA][i];
          size_t bB = sample_bins[pB][i];
          size_t bC = sample_bins[pC][i];
          size_t cell = (bA * num_binsB + bB) * num_binsC + bC;
          cell_sample_count[cell]++;
          size_t cnt = tree_samples.Count(s);
          for (size_t pred = 0; pred < P; pred++) {
            uint32_t tok = tree_samples.Token(pred, s);
            size_t offset = (cell * P + pred) * S;
            flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
            flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
          }
        }

        float grid_entropy = 0.0f;
        for (size_t c = 0; c < num_cells; c++) {
          if (cell_sample_count[c] == 0) continue;
          float best_cell_bits = std::numeric_limits<float>::max();
          for (size_t pred = 0; pred < P; pred++) {
            float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                         flat_cell_extra[c * P + pred];
            if (bits < best_cell_bits) best_cell_bits = bits;
          }
          grid_entropy += best_cell_bits;
        }

        float penA = 0.0f; for (float pen : cut_penalties[pA]) penA += pen;
        float penB = 0.0f; for (float pen : cut_penalties[pB]) penB += pen;
        float penC = 0.0f; for (float pen : cut_penalties[pC]) penC += pen;
        float cost = grid_entropy + penA + penB + penC;

        if (cost < best_grid_cost) {
          best_grid_cost = cost;
          best_pA = pA;
          best_pB = pB;
          best_pC = pC;
        }
      }
    }
  }

  // Populate winning 3D grid
  size_t pA = best_pA, pB = best_pB, pC = best_pC;
  size_t num_binsA = coarse_cuts[pA].size() + 1;
  size_t num_binsB = coarse_cuts[pB].size() + 1;
  size_t num_binsC = coarse_cuts[pC].size() + 1;
  size_t num_cells = num_binsA * num_binsB * num_binsC;
  size_t num_entries = num_cells * P;

  std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * S, 0);
  std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
  std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    size_t bA = sample_bins[pA][i];
    size_t bB = sample_bins[pB][i];
    size_t bC = sample_bins[pC][i];
    size_t cell = (bA * num_binsB + bB) * num_binsC + bC;
    cell_sample_count[cell]++;
    size_t cnt = tree_samples.Count(s);
    for (size_t pred = 0; pred < P; pred++) {
      uint32_t tok = tree_samples.Token(pred, s);
      size_t offset = (cell * P + pred) * S;
      flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
      flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
    }
  }

  std::vector<float> single_cell_bits(num_cells, 0.0f);
  std::vector<uint8_t> single_cell_pred(num_cells, 0);
  for (size_t c = 0; c < num_cells; c++) {
    if (cell_sample_count[c] == 0) continue;
    float best_bits = std::numeric_limits<float>::max();
    size_t best_p = 0;
    for (size_t pred = 0; pred < P; pred++) {
      float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                   flat_cell_extra[c * P + pred];
      if (bits < best_bits) {
        best_bits = bits;
        best_p = pred;
      }
    }
    single_cell_bits[c] = best_bits;
    single_cell_pred[c] = static_cast<uint8_t>(best_p);
  }

  tree->clear();
  tree->emplace_back();
  (*tree)[0] = PropertyDecisionNode::Leaf(Predictor::Zero);

  BuildTreeGreedy3D({0, static_cast<uint16_t>(num_binsA),
                     0, static_cast<uint16_t>(num_binsB),
                     0, static_cast<uint16_t>(num_binsC)},
                    pA + tree_samples.NumStaticProps(), coarse_cuts[pA], cut_penalties[pA],
                    pB + tree_samples.NumStaticProps(), coarse_cuts[pB], cut_penalties[pB],
                    pC + tree_samples.NumStaticProps(), coarse_cuts[pC], cut_penalties[pC],
                    num_binsB, num_binsC, P, S, max_symbols,
                    flat_cell_hists.data(), flat_cell_extra.data(), cell_sample_count.data(),
                    single_cell_bits.data(), single_cell_pred.data(),
                    tree_samples, tree, 0);
}

struct AllocatedGridCuts {
  std::vector<size_t> active_props;
  std::vector<std::vector<int32_t>> cutoffs;
  std::vector<size_t> num_bins;
  size_t total_cells = 1;
};

inline AllocatedGridCuts AllocateTupleDynamicGridCuts(
    const std::vector<size_t>& props,
    const std::vector<DP1DResult>& primary_dps,
    size_t max_cells = 2048,
    bool allow_zero_cuts = false) {
  AllocatedGridCuts res;
  size_t D = props.size();
  if (D == 0) return res;

  if (D == 1) {
    size_t p = props[0];
    size_t K = primary_dps[p].cut_history.size();
    size_t cuts = std::min(K, max_cells > 0 ? max_cells - 1 : 0);
    res.active_props = props;
    res.num_bins = {cuts + 1};
    res.total_cells = cuts + 1;
    res.cutoffs.resize(1);
    for (size_t i = 0; i < cuts; i++) {
      res.cutoffs[0].push_back(primary_dps[p].cut_history[i].cutoff_val);
    }
    std::sort(res.cutoffs[0].begin(), res.cutoffs[0].end());
    return res;
  }

  if (D == 2) {
    size_t pA = props[0], pB = props[1];
    size_t KA = primary_dps[pA].cut_history.size();
    size_t KB = primary_dps[pB].cut_history.size();

    std::vector<float> fA(KA + 1, 0.0f);
    for (size_t i = 0; i < KA; i++) fA[i + 1] = fA[i] + primary_dps[pA].cut_history[i].gain;
    std::vector<float> fB(KB + 1, 0.0f);
    for (size_t i = 0; i < KB; i++) fB[i + 1] = fB[i] + primary_dps[pB].cut_history[i].gain;

    size_t min_a = allow_zero_cuts ? 0 : std::min<size_t>(1, KA);
    size_t min_b = allow_zero_cuts ? 0 : std::min<size_t>(1, KB);

    float best_gain = -1.0f;
    size_t best_a = min_a, best_b = min_b;

    size_t max_a = std::min(KA, max_cells / (min_b + 1) > 0 ? (max_cells / (min_b + 1)) - 1 : 0);
    if (max_a >= min_a) {
      size_t b = std::min(KB, max_cells / (min_a + 1) > 0 ? (max_cells / (min_a + 1)) - 1 : 0);
      for (size_t a = min_a; a <= max_a; a++) {
        size_t na = a + 1;
        while (b > min_b && na * (b + 1) > max_cells) {
          b--;
        }
        if (na * (b + 1) > max_cells) continue;
        float gain = fA[a] + fB[b];
        if (gain > best_gain) {
          best_gain = gain;
          best_a = a;
          best_b = b;
        }
      }
    }

    res.active_props = props;
    res.num_bins = {best_a + 1, best_b + 1};
    res.total_cells = (best_a + 1) * (best_b + 1);
    res.cutoffs.resize(2);
    for (size_t i = 0; i < best_a; i++) {
      res.cutoffs[0].push_back(primary_dps[pA].cut_history[i].cutoff_val);
    }
    std::sort(res.cutoffs[0].begin(), res.cutoffs[0].end());
    for (size_t i = 0; i < best_b; i++) {
      res.cutoffs[1].push_back(primary_dps[pB].cut_history[i].cutoff_val);
    }
    std::sort(res.cutoffs[1].begin(), res.cutoffs[1].end());
    return res;
  }

  // D == 3: Nested two-pointer
  size_t pA = props[0], pB = props[1], pC = props[2];
  size_t KA = primary_dps[pA].cut_history.size();
  size_t KB = primary_dps[pB].cut_history.size();
  size_t KC = primary_dps[pC].cut_history.size();

  std::vector<float> fA(KA + 1, 0.0f);
  for (size_t i = 0; i < KA; i++) fA[i + 1] = fA[i] + primary_dps[pA].cut_history[i].gain;
  std::vector<float> fB(KB + 1, 0.0f);
  for (size_t i = 0; i < KB; i++) fB[i + 1] = fB[i] + primary_dps[pB].cut_history[i].gain;
  std::vector<float> fC(KC + 1, 0.0f);
  for (size_t i = 0; i < KC; i++) fC[i + 1] = fC[i] + primary_dps[pC].cut_history[i].gain;

  size_t min_a = allow_zero_cuts ? 0 : std::min<size_t>(1, KA);
  size_t min_b = allow_zero_cuts ? 0 : std::min<size_t>(1, KB);
  size_t min_c = allow_zero_cuts ? 0 : std::min<size_t>(1, KC);

  float best_gain = -1.0f;
  size_t best_a = min_a, best_b = min_b, best_c = min_c;

  size_t min_bc_cells = (min_b + 1) * (min_c + 1);
  size_t max_a = std::min(KA, max_cells / min_bc_cells > 0 ? (max_cells / min_bc_cells) - 1 : 0);

  if (max_a >= min_a) {
    for (size_t a = min_a; a <= max_a; a++) {
      size_t na = a + 1;
      size_t M_prime = max_cells / na;
      if (M_prime < min_bc_cells) continue;

      size_t max_b = std::min(KB, M_prime / (min_c + 1) > 0 ? (M_prime / (min_c + 1)) - 1 : 0);
      if (max_b < min_b) continue;

      // Inner two-pointer over (b, c) with budget M_prime
      size_t c = std::min(KC, M_prime / (min_b + 1) > 0 ? (M_prime / (min_b + 1)) - 1 : 0);

      for (size_t b = min_b; b <= max_b; b++) {
        size_t nb = b + 1;
        while (c > min_c && nb * (c + 1) > M_prime) {
          c--;
        }
        if (nb * (c + 1) > M_prime) continue;

        float gain = fA[a] + fB[b] + fC[c];
        if (gain > best_gain) {
          best_gain = gain;
          best_a = a;
          best_b = b;
          best_c = c;
        }
      }
    }
  }

  res.active_props = props;
  res.num_bins = {best_a + 1, best_b + 1, best_c + 1};
  res.total_cells = (best_a + 1) * (best_b + 1) * (best_c + 1);
  res.cutoffs.resize(3);
  for (size_t i = 0; i < best_a; i++) {
    res.cutoffs[0].push_back(primary_dps[pA].cut_history[i].cutoff_val);
  }
  std::sort(res.cutoffs[0].begin(), res.cutoffs[0].end());
  for (size_t i = 0; i < best_b; i++) {
    res.cutoffs[1].push_back(primary_dps[pB].cut_history[i].cutoff_val);
  }
  std::sort(res.cutoffs[1].begin(), res.cutoffs[1].end());
  for (size_t i = 0; i < best_c; i++) {
    res.cutoffs[2].push_back(primary_dps[pC].cut_history[i].cutoff_val);
  }
  std::sort(res.cutoffs[2].begin(), res.cutoffs[2].end());
  return res;
}

struct ScoredCandidateGrid {
  AllocatedGridCuts alloc;
  float theoretical_gain;
};

inline std::vector<ScoredCandidateGrid> FindTopCandidateGrids(
    const std::vector<size_t>& valid_props,
    const std::vector<DP1DResult>& primary_dps,
    size_t max_cells = 2048,
    size_t max_dims = 3,
    size_t top_k = 1) {
  std::vector<ScoredCandidateGrid> candidates;
  if (valid_props.empty()) return candidates;

  if (valid_props.size() == 1) {
    AllocatedGridCuts alloc = AllocateTupleDynamicGridCuts(valid_props, primary_dps, max_cells, true);
    float gain = 0.0f;
    for (size_t c = 0; c < alloc.cutoffs[0].size(); c++) {
      gain += primary_dps[valid_props[0]].cut_history[c].gain;
    }
    candidates.push_back({std::move(alloc), gain});
    return candidates;
  }

  if (valid_props.size() == 2 && max_dims >= 2) {
    AllocatedGridCuts alloc = AllocateTupleDynamicGridCuts(valid_props, primary_dps, max_cells, true);
    float gain = 0.0f;
    for (size_t d = 0; d < alloc.active_props.size(); d++) {
      size_t p = alloc.active_props[d];
      for (size_t c = 0; c < alloc.cutoffs[d].size(); c++) {
        gain += primary_dps[p].cut_history[c].gain;
      }
    }
    candidates.push_back({std::move(alloc), gain});
    return candidates;
  }

  // Screen candidate triples and pairs among top properties
  size_t T = std::min<size_t>(valid_props.size(), 8);

  for (size_t i = 0; i < T; i++) {
    for (size_t j = i + 1; j < T; j++) {
      if (max_dims >= 3) {
        for (size_t k = j + 1; k < T; k++) {
          AllocatedGridCuts alloc = AllocateTupleDynamicGridCuts(
              {valid_props[i], valid_props[j], valid_props[k]},
              primary_dps, max_cells, /*allow_zero_cuts=*/true);

          float gain = 0.0f;
          for (size_t d = 0; d < 3; d++) {
            size_t p = alloc.active_props[d];
            size_t ncuts = alloc.cutoffs[d].size();
            for (size_t c = 0; c < ncuts; c++) {
              gain += primary_dps[p].cut_history[c].gain;
            }
          }
          candidates.push_back({std::move(alloc), gain});
        }
      }
      // Also evaluate 2-property tuple
      if (max_dims >= 2) {
        AllocatedGridCuts alloc2 = AllocateTupleDynamicGridCuts(
            {valid_props[i], valid_props[j]},
            primary_dps, max_cells, /*allow_zero_cuts=*/true);
        float gain2 = 0.0f;
        for (size_t d = 0; d < 2; d++) {
          size_t p = alloc2.active_props[d];
          size_t ncuts = alloc2.cutoffs[d].size();
          for (size_t c = 0; c < ncuts; c++) {
            gain2 += primary_dps[p].cut_history[c].gain;
          }
        }
        candidates.push_back({std::move(alloc2), gain2});
      }
    }
  }

  // Filter out any active properties that ended up with 0 cutoffs, and deduplicate
  std::vector<ScoredCandidateGrid> unique_cands;
  for (auto& cand : candidates) {
    AllocatedGridCuts filtered_res;
    for (size_t d = 0; d < cand.alloc.active_props.size(); d++) {
      if (!cand.alloc.cutoffs[d].empty()) {
        filtered_res.active_props.push_back(cand.alloc.active_props[d]);
        filtered_res.num_bins.push_back(cand.alloc.num_bins[d]);
        filtered_res.cutoffs.push_back(std::move(cand.alloc.cutoffs[d]));
      }
    }
    filtered_res.total_cells = 1;
    for (size_t b : filtered_res.num_bins) filtered_res.total_cells *= b;
    if (filtered_res.active_props.empty()) continue;

    bool duplicate = false;
    for (const auto& u : unique_cands) {
      if (u.alloc.active_props == filtered_res.active_props &&
          u.alloc.cutoffs == filtered_res.cutoffs) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate) {
      cand.alloc = std::move(filtered_res);
      unique_cands.push_back(std::move(cand));
    }
  }

  // Sort candidates by descending theoretical gain
  std::sort(unique_cands.begin(), unique_cands.end(),
            [](const ScoredCandidateGrid& a, const ScoredCandidateGrid& b) {
              return a.theoretical_gain > b.theoretical_gain;
            });

  if (unique_cands.size() > top_k) {
    unique_cands.resize(top_k);
  }

  return unique_cands;
}

inline AllocatedGridCuts AllocateGlobalDynamicGridCuts(
    const std::vector<size_t>& valid_props,
    const std::vector<DP1DResult>& primary_dps,
    size_t max_cells = 2048,
    size_t max_dims = 3) {
  auto top_grids = FindTopCandidateGrids(valid_props, primary_dps, max_cells, max_dims, 1);
  if (top_grids.empty()) return AllocatedGridCuts();
  return std::move(top_grids[0].alloc);
}

void FindBestTreeGrid2dDynDP(TreeSamples& tree_samples, float scale, Tree* tree,
                             const std::vector<uint32_t>* sample_subset = nullptr,
                             float base_node_cost = 92.0f, float log_node_cost = 1.2f,
                             size_t default_max_cells = 2048) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props < 2 || total_samples == 0) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  size_t max_grid_cells = default_max_cells;
  const char* env_cells = getenv("JXL_GRID_CELLS");
  if (env_cells) max_grid_cells = std::max<size_t>(16, std::min<size_t>(16384, atoi(env_cells)));

  auto screen = ScreenAllProperties1D(tree_samples, sample_subset, scale, base_node_cost, log_node_cost,
                                     /*coarse_max_cuts=*/7, /*refine_with_dp=*/false);
  const auto& coarse_cuts = screen.coarse_cuts;

  std::vector<size_t> valid_props;
  for (size_t p = 0; p < num_props; p++) {
    if (!screen.primary_dps[p].cut_history.empty()) valid_props.push_back(p);
  }
  if (valid_props.size() < 2) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  std::vector<std::vector<uint16_t>> sample_bins(num_props, std::vector<uint16_t>(total_samples));
  for (size_t p : valid_props) {
    const auto& cuts = coarse_cuts[p];
    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      int32_t v = tree_samples.Property<false>(p, s);
      size_t b = 0;
      while (b < cuts.size() && v > cuts[b]) b++;
      sample_bins[p][i] = static_cast<uint16_t>(b);
    }
  }

  const size_t P = tree_samples.NumPredictors();
  const size_t S = screen.mpt.S;
  const auto& max_symbols = screen.mpt.max_symbols;

  struct CandidatePair {
    size_t pA, pB;
    float coarse_cost;
  };
  std::vector<CandidatePair> candidates;

  std::vector<int32_t> flat_cell_hists;
  std::vector<int64_t> flat_cell_extra;
  std::vector<size_t> cell_sample_count;

  for (size_t idxA = 0; idxA < valid_props.size(); idxA++) {
    size_t pA = valid_props[idxA];
    size_t num_binsA = coarse_cuts[pA].size() + 1;
    for (size_t idxB = idxA + 1; idxB < valid_props.size(); idxB++) {
      size_t pB = valid_props[idxB];
      size_t num_binsB = coarse_cuts[pB].size() + 1;
      size_t num_cells = num_binsA * num_binsB;
      size_t num_entries = num_cells * P;

      if (flat_cell_hists.size() < num_entries * S) flat_cell_hists.resize(num_entries * S);
      if (flat_cell_extra.size() < num_entries) flat_cell_extra.resize(num_entries);
      if (cell_sample_count.size() < num_cells) cell_sample_count.resize(num_cells);

      std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * S, 0);
      std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
      std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

      for (size_t i = 0; i < total_samples; i++) {
        size_t bA = sample_bins[pA][i];
        size_t bB = sample_bins[pB][i];
        size_t cell = bA * num_binsB + bB;
        cell_sample_count[cell]++;
        size_t s = get_sample_idx(i);
        size_t cnt = tree_samples.Count(s);
        for (size_t pred = 0; pred < P; pred++) {
          uint32_t tok = tree_samples.Token(pred, s);
          size_t offset = (cell * P + pred) * S;
          flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
          flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
        }
      }

      float grid_entropy = 0.0f;
      for (size_t c = 0; c < num_cells; c++) {
        if (cell_sample_count[c] == 0) continue;
        float best_cell_bits = std::numeric_limits<float>::max();
        for (size_t pred = 0; pred < P; pred++) {
          float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                       flat_cell_extra[c * P + pred];
          if (bits < best_cell_bits) best_cell_bits = bits;
        }
        grid_entropy += best_cell_bits;
      }

      auto sum_pen = [&](size_t prop, const std::vector<int32_t>& cuts) {
        float sum = 0.0f;
        for (int32_t c : cuts) {
          int32_t unquant = tree_samples.UnquantizeProperty(prop + tree_samples.NumStaticProps(), c);
          sum += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
        }
        return sum;
      };

      float cost = grid_entropy + sum_pen(pA, coarse_cuts[pA]) + sum_pen(pB, coarse_cuts[pB]);
      candidates.push_back({pA, pB, cost});
    }
  }

  std::sort(candidates.begin(), candidates.end(), [](const CandidatePair& a, const CandidatePair& b) {
    return a.coarse_cost < b.coarse_cost;
  });
  size_t to_evaluate = std::min<size_t>(candidates.size(), 8);

  size_t best_pA = candidates[0].pA, best_pB = candidates[0].pB;
  AllocatedGridCuts best_alloc = AllocateTupleDynamicGridCuts({best_pA, best_pB}, screen.primary_dps, max_grid_cells);
  float best_dyn_cost = std::numeric_limits<float>::max();

  for (size_t cand_idx = 0; cand_idx < to_evaluate; cand_idx++) {
    size_t pA = candidates[cand_idx].pA;
    size_t pB = candidates[cand_idx].pB;
    AllocatedGridCuts alloc = AllocateTupleDynamicGridCuts({pA, pB}, screen.primary_dps, max_grid_cells);
    const auto& cutsA = alloc.cutoffs[0];
    const auto& cutsB = alloc.cutoffs[1];
    size_t num_binsA = cutsA.size() + 1;
    size_t num_binsB = cutsB.size() + 1;
    size_t num_cells = num_binsA * num_binsB;
    size_t num_entries = num_cells * P;

    if (flat_cell_hists.size() < num_entries * S) flat_cell_hists.resize(num_entries * S);
    if (flat_cell_extra.size() < num_entries) flat_cell_extra.resize(num_entries);
    if (cell_sample_count.size() < num_cells) cell_sample_count.resize(num_cells);

    std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * S, 0);
    std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
    std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      int32_t vA = tree_samples.Property<false>(pA, s);
      int32_t vB = tree_samples.Property<false>(pB, s);
      size_t bA = 0; while (bA < cutsA.size() && vA > cutsA[bA]) bA++;
      size_t bB = 0; while (bB < cutsB.size() && vB > cutsB[bB]) bB++;
      size_t cell = bA * num_binsB + bB;
      cell_sample_count[cell]++;
      size_t cnt = tree_samples.Count(s);
      for (size_t pred = 0; pred < P; pred++) {
        uint32_t tok = tree_samples.Token(pred, s);
        size_t offset = (cell * P + pred) * S;
        flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
        flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
      }
    }

    float grid_entropy = 0.0f;
    for (size_t c = 0; c < num_cells; c++) {
      if (cell_sample_count[c] == 0) continue;
      float best_cell_bits = std::numeric_limits<float>::max();
      for (size_t pred = 0; pred < P; pred++) {
        float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                     flat_cell_extra[c * P + pred];
        if (bits < best_cell_bits) best_cell_bits = bits;
      }
      grid_entropy += best_cell_bits;
    }

    auto sum_pen = [&](size_t prop, const std::vector<int32_t>& cuts) {
      float sum = 0.0f;
      for (int32_t c : cuts) {
        int32_t unquant = tree_samples.UnquantizeProperty(prop + tree_samples.NumStaticProps(), c);
        sum += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
      }
      return sum;
    };

    float cost = grid_entropy + sum_pen(pA, cutsA) + sum_pen(pB, cutsB);
    if (cost < best_dyn_cost) {
      best_dyn_cost = cost;
      best_pA = pA;
      best_pB = pB;
      best_alloc = std::move(alloc);
    }
  }

  size_t pA = best_pA, pB = best_pB;
  const auto& cutsA = best_alloc.cutoffs[0];
  const auto& cutsB = best_alloc.cutoffs[1];
  size_t num_binsA = cutsA.size() + 1;
  size_t num_binsB = cutsB.size() + 1;
  size_t num_cells = num_binsA * num_binsB;
  size_t num_entries = num_cells * P;

  if (flat_cell_hists.size() < num_entries * S) flat_cell_hists.resize(num_entries * S);
  if (flat_cell_extra.size() < num_entries) flat_cell_extra.resize(num_entries);
  if (cell_sample_count.size() < num_cells) cell_sample_count.resize(num_cells);

  std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * S, 0);
  std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
  std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    int32_t vA = tree_samples.Property<false>(pA, s);
    int32_t vB = tree_samples.Property<false>(pB, s);
    size_t bA = 0; while (bA < cutsA.size() && vA > cutsA[bA]) bA++;
    size_t bB = 0; while (bB < cutsB.size() && vB > cutsB[bB]) bB++;
    size_t cell = bA * num_binsB + bB;
    cell_sample_count[cell]++;
    size_t cnt = tree_samples.Count(s);
    for (size_t pred = 0; pred < P; pred++) {
      uint32_t tok = tree_samples.Token(pred, s);
      size_t offset = (cell * P + pred) * S;
      flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
      flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
    }
  }

  std::vector<float> single_cell_bits(num_cells, 0.0f);
  std::vector<uint8_t> single_cell_pred(num_cells, 0);
  for (size_t c = 0; c < num_cells; c++) {
    if (cell_sample_count[c] == 0) continue;
    float best_bits = std::numeric_limits<float>::max();
    size_t best_p = 0;
    for (size_t pred = 0; pred < P; pred++) {
      float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                   flat_cell_extra[c * P + pred];
      if (bits < best_bits) {
        best_bits = bits;
        best_p = pred;
      }
    }
    single_cell_bits[c] = best_bits;
    single_cell_pred[c] = static_cast<uint8_t>(best_p);
  }

  std::vector<float> cut_penaltiesA, cut_penaltiesB;
  for (int32_t c : cutsA) {
    int32_t unquant = tree_samples.UnquantizeProperty(pA + tree_samples.NumStaticProps(), c);
    cut_penaltiesA.push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
  }
  for (int32_t c : cutsB) {
    int32_t unquant = tree_samples.UnquantizeProperty(pB + tree_samples.NumStaticProps(), c);
    cut_penaltiesB.push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
  }

  tree->clear();
  tree->emplace_back();
  (*tree)[0] = PropertyDecisionNode::Leaf(Predictor::Zero);

  BuildTreeGreedy2D({0, static_cast<uint16_t>(num_binsA), 0, static_cast<uint16_t>(num_binsB)},
                    pA + tree_samples.NumStaticProps(), cutsA, cut_penaltiesA,
                    pB + tree_samples.NumStaticProps(), cutsB, cut_penaltiesB,
                    num_binsB, P, S, max_symbols,
                    flat_cell_hists.data(), flat_cell_extra.data(), cell_sample_count.data(),
                    single_cell_bits.data(), single_cell_pred.data(),
                    tree_samples, tree, 0);
}

void FindBestTreeGrid3dDynDP(TreeSamples& tree_samples, float scale, Tree* tree,
                             const std::vector<uint32_t>* sample_subset = nullptr,
                             float base_node_cost = 92.0f, float log_node_cost = 1.2f,
                             size_t default_max_cells = 2048) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props < 3 || total_samples == 0) {
    FindBestTreeGrid2dDynDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost, default_max_cells);
    return;
  }

  size_t max_grid_cells = default_max_cells;
  const char* env_cells = getenv("JXL_GRID_CELLS");
  if (env_cells) max_grid_cells = std::max<size_t>(16, std::min<size_t>(16384, atoi(env_cells)));

  auto screen = ScreenAllProperties1D(tree_samples, sample_subset, scale, base_node_cost, log_node_cost,
                                     /*coarse_max_cuts=*/7, /*refine_with_dp=*/false);
  const auto& coarse_cuts = screen.coarse_cuts;

  std::vector<size_t> valid_props;
  for (size_t p = 0; p < num_props; p++) {
    if (!screen.primary_dps[p].cut_history.empty()) valid_props.push_back(p);
  }
  if (valid_props.size() < 3) {
    FindBestTreeGrid2dDynDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost, max_grid_cells);
    return;
  }

  std::sort(valid_props.begin(), valid_props.end(), [&](size_t a, size_t b) {
    return screen.primary_dps[a].cost < screen.primary_dps[b].cost;
  });
  if (valid_props.size() > 8) {
    valid_props.resize(8);
  }

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  std::vector<std::vector<uint16_t>> sample_bins(num_props, std::vector<uint16_t>(total_samples));
  for (size_t p : valid_props) {
    const auto& cuts = coarse_cuts[p];
    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      int32_t v = tree_samples.Property<false>(p, s);
      size_t b = 0;
      while (b < cuts.size() && v > cuts[b]) b++;
      sample_bins[p][i] = static_cast<uint16_t>(b);
    }
  }

  const size_t P = tree_samples.NumPredictors();
  const size_t S = screen.mpt.S;
  const auto& max_symbols = screen.mpt.max_symbols;

  struct CandidateTriple {
    size_t pA, pB, pC;
    float coarse_cost;
  };
  std::vector<CandidateTriple> candidates;

  std::vector<int32_t> flat_cell_hists;
  std::vector<int64_t> flat_cell_extra;
  std::vector<size_t> cell_sample_count;

  for (size_t idxA = 0; idxA < valid_props.size(); idxA++) {
    size_t pA = valid_props[idxA];
    size_t num_binsA = coarse_cuts[pA].size() + 1;
    for (size_t idxB = idxA + 1; idxB < valid_props.size(); idxB++) {
      size_t pB = valid_props[idxB];
      size_t num_binsB = coarse_cuts[pB].size() + 1;
      for (size_t idxC = idxB + 1; idxC < valid_props.size(); idxC++) {
        size_t pC = valid_props[idxC];
        size_t num_binsC = coarse_cuts[pC].size() + 1;
        size_t num_cells = num_binsA * num_binsB * num_binsC;
        size_t num_entries = num_cells * P;

        if (flat_cell_hists.size() < num_entries * S) flat_cell_hists.resize(num_entries * S);
        if (flat_cell_extra.size() < num_entries) flat_cell_extra.resize(num_entries);
        if (cell_sample_count.size() < num_cells) cell_sample_count.resize(num_cells);

        std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * S, 0);
        std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
        std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

        for (size_t i = 0; i < total_samples; i++) {
          size_t bA = sample_bins[pA][i];
          size_t bB = sample_bins[pB][i];
          size_t bC = sample_bins[pC][i];
          size_t cell = (bA * num_binsB + bB) * num_binsC + bC;
          cell_sample_count[cell]++;
          size_t s = get_sample_idx(i);
          size_t cnt = tree_samples.Count(s);
          for (size_t pred = 0; pred < P; pred++) {
            uint32_t tok = tree_samples.Token(pred, s);
            size_t offset = (cell * P + pred) * S;
            flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
            flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
          }
        }

        float grid_entropy = 0.0f;
        for (size_t c = 0; c < num_cells; c++) {
          if (cell_sample_count[c] == 0) continue;
          float best_cell_bits = std::numeric_limits<float>::max();
          for (size_t pred = 0; pred < P; pred++) {
            float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                         flat_cell_extra[c * P + pred];
            if (bits < best_cell_bits) best_cell_bits = bits;
          }
          grid_entropy += best_cell_bits;
        }

        auto sum_pen = [&](size_t prop, const std::vector<int32_t>& cuts) {
          float sum = 0.0f;
          for (int32_t c : cuts) {
            int32_t unquant = tree_samples.UnquantizeProperty(prop + tree_samples.NumStaticProps(), c);
            sum += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
          }
          return sum;
        };

        float cost = grid_entropy + sum_pen(pA, coarse_cuts[pA]) + sum_pen(pB, coarse_cuts[pB]) + sum_pen(pC, coarse_cuts[pC]);
        candidates.push_back({pA, pB, pC, cost});
      }
    }
  }

  std::sort(candidates.begin(), candidates.end(), [](const CandidateTriple& a, const CandidateTriple& b) {
    return a.coarse_cost < b.coarse_cost;
  });
  size_t to_evaluate = std::min<size_t>(candidates.size(), 8);

  size_t best_pA = candidates[0].pA, best_pB = candidates[0].pB, best_pC = candidates[0].pC;
  AllocatedGridCuts best_alloc = AllocateTupleDynamicGridCuts({best_pA, best_pB, best_pC}, screen.primary_dps, max_grid_cells);
  float best_dyn_cost = std::numeric_limits<float>::max();

  for (size_t cand_idx = 0; cand_idx < to_evaluate; cand_idx++) {
    size_t pA = candidates[cand_idx].pA;
    size_t pB = candidates[cand_idx].pB;
    size_t pC = candidates[cand_idx].pC;
    AllocatedGridCuts alloc = AllocateTupleDynamicGridCuts({pA, pB, pC}, screen.primary_dps, max_grid_cells);

    const auto& cutsA = alloc.cutoffs[0];
    const auto& cutsB = alloc.cutoffs[1];
    const auto& cutsC = alloc.cutoffs[2];
    size_t num_binsA = cutsA.size() + 1;
    size_t num_binsB = cutsB.size() + 1;
    size_t num_binsC = cutsC.size() + 1;
    size_t num_cells = num_binsA * num_binsB * num_binsC;
    size_t num_entries = num_cells * P;

    if (flat_cell_hists.size() < num_entries * S) flat_cell_hists.resize(num_entries * S);
    if (flat_cell_extra.size() < num_entries) flat_cell_extra.resize(num_entries);
    if (cell_sample_count.size() < num_cells) cell_sample_count.resize(num_cells);

    std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * S, 0);
    std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
    std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      int32_t vA = tree_samples.Property<false>(pA, s);
      int32_t vB = tree_samples.Property<false>(pB, s);
      int32_t vC = tree_samples.Property<false>(pC, s);
      size_t bA = 0; while (bA < cutsA.size() && vA > cutsA[bA]) bA++;
      size_t bB = 0; while (bB < cutsB.size() && vB > cutsB[bB]) bB++;
      size_t bC = 0; while (bC < cutsC.size() && vC > cutsC[bC]) bC++;
      size_t cell = (bA * num_binsB + bB) * num_binsC + bC;
      cell_sample_count[cell]++;
      size_t cnt = tree_samples.Count(s);
      for (size_t pred = 0; pred < P; pred++) {
        uint32_t tok = tree_samples.Token(pred, s);
        size_t offset = (cell * P + pred) * S;
        flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
        flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
      }
    }

    float grid_entropy = 0.0f;
    for (size_t c = 0; c < num_cells; c++) {
      if (cell_sample_count[c] == 0) continue;
      float best_cell_bits = std::numeric_limits<float>::max();
      for (size_t pred = 0; pred < P; pred++) {
        float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                     flat_cell_extra[c * P + pred];
        if (bits < best_cell_bits) best_cell_bits = bits;
      }
      grid_entropy += best_cell_bits;
    }

    auto sum_pen = [&](size_t prop, const std::vector<int32_t>& cuts) {
      float sum = 0.0f;
      for (int32_t c : cuts) {
        int32_t unquant = tree_samples.UnquantizeProperty(prop + tree_samples.NumStaticProps(), c);
        sum += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
      }
      return sum;
    };

    float cost = grid_entropy + sum_pen(pA, cutsA) + sum_pen(pB, cutsB) + sum_pen(pC, cutsC);
    if (cost < best_dyn_cost) {
      best_dyn_cost = cost;
      best_pA = pA;
      best_pB = pB;
      best_pC = pC;
      best_alloc = std::move(alloc);
    }
  }

  size_t pA = best_pA, pB = best_pB, pC = best_pC;
  const auto& cutsA = best_alloc.cutoffs[0];
  const auto& cutsB = best_alloc.cutoffs[1];
  const auto& cutsC = best_alloc.cutoffs[2];
  size_t num_binsA = cutsA.size() + 1;
  size_t num_binsB = cutsB.size() + 1;
  size_t num_binsC = cutsC.size() + 1;
  size_t num_cells = num_binsA * num_binsB * num_binsC;
  size_t num_entries = num_cells * P;

  if (flat_cell_hists.size() < num_entries * S) flat_cell_hists.resize(num_entries * S);
  if (flat_cell_extra.size() < num_entries) flat_cell_extra.resize(num_entries);
  if (cell_sample_count.size() < num_cells) cell_sample_count.resize(num_cells);

  std::fill(flat_cell_hists.begin(), flat_cell_hists.begin() + num_entries * S, 0);
  std::fill(flat_cell_extra.begin(), flat_cell_extra.begin() + num_entries, 0);
  std::fill(cell_sample_count.begin(), cell_sample_count.begin() + num_cells, 0);

  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    int32_t vA = tree_samples.Property<false>(pA, s);
    int32_t vB = tree_samples.Property<false>(pB, s);
    int32_t vC = tree_samples.Property<false>(pC, s);
    size_t bA = 0; while (bA < cutsA.size() && vA > cutsA[bA]) bA++;
    size_t bB = 0; while (bB < cutsB.size() && vB > cutsB[bB]) bB++;
    size_t bC = 0; while (bC < cutsC.size() && vC > cutsC[bC]) bC++;
    size_t cell = (bA * num_binsB + bB) * num_binsC + bC;
    cell_sample_count[cell]++;
    size_t cnt = tree_samples.Count(s);
    for (size_t pred = 0; pred < P; pred++) {
      uint32_t tok = tree_samples.Token(pred, s);
      size_t offset = (cell * P + pred) * S;
      flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
      flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
    }
  }

  std::vector<float> single_cell_bits(num_cells, 0.0f);
  std::vector<uint8_t> single_cell_pred(num_cells, 0);
  for (size_t c = 0; c < num_cells; c++) {
    if (cell_sample_count[c] == 0) continue;
    float best_bits = std::numeric_limits<float>::max();
    size_t best_p = 0;
    for (size_t pred = 0; pred < P; pred++) {
      float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                   flat_cell_extra[c * P + pred];
      if (bits < best_bits) {
        best_bits = bits;
        best_p = pred;
      }
    }
    single_cell_bits[c] = best_bits;
    single_cell_pred[c] = static_cast<uint8_t>(best_p);
  }

  std::vector<float> cut_penaltiesA, cut_penaltiesB, cut_penaltiesC;
  for (int32_t c : cutsA) {
    int32_t unquant = tree_samples.UnquantizeProperty(pA + tree_samples.NumStaticProps(), c);
    cut_penaltiesA.push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
  }
  for (int32_t c : cutsB) {
    int32_t unquant = tree_samples.UnquantizeProperty(pB + tree_samples.NumStaticProps(), c);
    cut_penaltiesB.push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
  }
  for (int32_t c : cutsC) {
    int32_t unquant = tree_samples.UnquantizeProperty(pC + tree_samples.NumStaticProps(), c);
    cut_penaltiesC.push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
  }

  tree->clear();
  tree->emplace_back();
  (*tree)[0] = PropertyDecisionNode::Leaf(Predictor::Zero);

  BuildTreeGreedy3D({0, static_cast<uint16_t>(num_binsA),
                     0, static_cast<uint16_t>(num_binsB),
                     0, static_cast<uint16_t>(num_binsC)},
                    pA + tree_samples.NumStaticProps(), cutsA, cut_penaltiesA,
                    pB + tree_samples.NumStaticProps(), cutsB, cut_penaltiesB,
                    pC + tree_samples.NumStaticProps(), cutsC, cut_penaltiesC,
                    num_binsB, num_binsC, P, S, max_symbols,
                    flat_cell_hists.data(), flat_cell_extra.data(), cell_sample_count.data(),
                    single_cell_bits.data(), single_cell_pred.data(),
                    tree_samples, tree, 0);
}

void FindBestTreeGridDynDP(TreeSamples& tree_samples, float scale, Tree* tree,
                           const std::vector<uint32_t>* sample_subset = nullptr,
                           float base_node_cost = 92.0f, float log_node_cost = 1.2f,
                           size_t default_max_cells = 2048,
                           size_t default_max_dims = 3,
                           size_t default_top_candidates = 1) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  const size_t total_samples = sample_subset ? sample_subset->size() : tree_samples.NumDistinctSamples();
  if (num_props < 2 || total_samples == 0) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  size_t max_grid_cells = default_max_cells;
  const char* env_cells = getenv("JXL_GRID_CELLS");
  if (env_cells) max_grid_cells = std::max<size_t>(16, std::min<size_t>(16384, atoi(env_cells)));

  size_t max_dims = default_max_dims;
  const char* env_dims = getenv("JXL_GRID_DIMS");
  if (env_dims) max_dims = std::max<size_t>(1, std::min<size_t>(3, atoi(env_dims)));

  size_t top_candidates = default_top_candidates;
  const char* env_top = getenv("JXL_GRID_TOP_K");
  if (!env_top) env_top = getenv("JXL_GRID_TOP_TRIPLES");
  if (!env_top) env_top = getenv("JXL_GRID_TOP_CANDIDATES");
  if (env_top) top_candidates = std::max<size_t>(1, std::min<size_t>(100, atoi(env_top)));

  auto screen = ScreenAllProperties1D(tree_samples, sample_subset, scale, base_node_cost, log_node_cost,
                                     /*coarse_max_cuts=*/63, /*refine_with_dp=*/false);

  std::vector<size_t> valid_props;
  for (size_t p = 0; p < num_props; p++) {
    if (!screen.primary_dps[p].cut_history.empty()) valid_props.push_back(p);
  }
  if (valid_props.empty()) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  std::sort(valid_props.begin(), valid_props.end(), [&](size_t a, size_t b) {
    return screen.primary_dps[a].cost < screen.primary_dps[b].cost;
  });

  auto candidates = FindTopCandidateGrids(valid_props, screen.primary_dps, max_grid_cells, max_dims, top_candidates);
  if (candidates.empty()) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  auto get_sample_idx = [&](size_t i) -> size_t {
    return sample_subset ? (*sample_subset)[i] : i;
  };

  const size_t P = tree_samples.NumPredictors();
  const size_t S = screen.mpt.S;
  const auto& max_symbols = screen.mpt.max_symbols;

  size_t best_cand_idx = 0;
  if (candidates.size() > 1) {
    float best_cand_cost = std::numeric_limits<float>::max();
    std::vector<int32_t> eval_flat_cell_hists;
    std::vector<int64_t> eval_flat_cell_extra;
    std::vector<size_t> eval_cell_sample_count;

    for (size_t cand_idx = 0; cand_idx < candidates.size(); cand_idx++) {
      const auto& cand_alloc = candidates[cand_idx].alloc;
      if (cand_alloc.active_props.size() < 2) continue;

      size_t num_cells = cand_alloc.total_cells;
      size_t num_entries = num_cells * P;
      if (eval_flat_cell_hists.size() < num_entries * S) eval_flat_cell_hists.resize(num_entries * S);
      if (eval_flat_cell_extra.size() < num_entries) eval_flat_cell_extra.resize(num_entries);
      if (eval_cell_sample_count.size() < num_cells) eval_cell_sample_count.resize(num_cells);

      std::fill(eval_flat_cell_hists.begin(), eval_flat_cell_hists.begin() + num_entries * S, 0);
      std::fill(eval_flat_cell_extra.begin(), eval_flat_cell_extra.begin() + num_entries, 0);
      std::fill(eval_cell_sample_count.begin(), eval_cell_sample_count.begin() + num_cells, 0);

      float split_penalty = 0.0f;
      for (size_t d = 0; d < cand_alloc.active_props.size(); d++) {
        size_t p = cand_alloc.active_props[d];
        for (int32_t c : cand_alloc.cutoffs[d]) {
          int32_t unquant = tree_samples.UnquantizeProperty(p + tree_samples.NumStaticProps(), c);
          split_penalty += (base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale;
        }
      }

      if (cand_alloc.active_props.size() == 2) {
        size_t pA = cand_alloc.active_props[0];
        size_t pB = cand_alloc.active_props[1];
        const auto& cutsA = cand_alloc.cutoffs[0];
        const auto& cutsB = cand_alloc.cutoffs[1];
        size_t num_binsB = cutsB.size() + 1;

        for (size_t i = 0; i < total_samples; i++) {
          size_t s = get_sample_idx(i);
          int32_t vA = tree_samples.Property<false>(pA, s);
          int32_t vB = tree_samples.Property<false>(pB, s);
          size_t bA = 0; while (bA < cutsA.size() && vA > cutsA[bA]) bA++;
          size_t bB = 0; while (bB < cutsB.size() && vB > cutsB[bB]) bB++;
          size_t cell = bA * num_binsB + bB;
          eval_cell_sample_count[cell]++;
          size_t cnt = tree_samples.Count(s);
          for (size_t pred = 0; pred < P; pred++) {
            uint32_t tok = tree_samples.Token(pred, s);
            size_t offset = (cell * P + pred) * S;
            eval_flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
            eval_flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
          }
        }
      } else if (cand_alloc.active_props.size() == 3) {
        size_t pA = cand_alloc.active_props[0];
        size_t pB = cand_alloc.active_props[1];
        size_t pC = cand_alloc.active_props[2];
        const auto& cutsA = cand_alloc.cutoffs[0];
        const auto& cutsB = cand_alloc.cutoffs[1];
        const auto& cutsC = cand_alloc.cutoffs[2];
        size_t num_binsB = cutsB.size() + 1;
        size_t num_binsC = cutsC.size() + 1;

        for (size_t i = 0; i < total_samples; i++) {
          size_t s = get_sample_idx(i);
          int32_t vA = tree_samples.Property<false>(pA, s);
          int32_t vB = tree_samples.Property<false>(pB, s);
          int32_t vC = tree_samples.Property<false>(pC, s);
          size_t bA = 0; while (bA < cutsA.size() && vA > cutsA[bA]) bA++;
          size_t bB = 0; while (bB < cutsB.size() && vB > cutsB[bB]) bB++;
          size_t bC = 0; while (bC < cutsC.size() && vC > cutsC[bC]) bC++;
          size_t cell = (bA * num_binsB + bB) * num_binsC + bC;
          eval_cell_sample_count[cell]++;
          size_t cnt = tree_samples.Count(s);
          for (size_t pred = 0; pred < P; pred++) {
            uint32_t tok = tree_samples.Token(pred, s);
            size_t offset = (cell * P + pred) * S;
            eval_flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
            eval_flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
          }
        }
      }

      float grid_entropy = 0.0f;
      for (size_t c = 0; c < num_cells; c++) {
        if (eval_cell_sample_count[c] == 0) continue;
        float best_cell_bits = std::numeric_limits<float>::max();
        for (size_t pred = 0; pred < P; pred++) {
          float bits = EstimateBits(&eval_flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                       eval_flat_cell_extra[c * P + pred];
          if (bits < best_cell_bits) best_cell_bits = bits;
        }
        grid_entropy += best_cell_bits;
      }

      float total_cand_cost = grid_entropy + split_penalty;
      if (total_cand_cost < best_cand_cost) {
        best_cand_cost = total_cand_cost;
        best_cand_idx = cand_idx;
      }
    }
  }

  AllocatedGridCuts alloc = std::move(candidates[best_cand_idx].alloc);

  if (getenv("JXL_PRINT_GRID_WINNERS")) {
    fprintf(stderr, "GRID_WINNER: dims=%zu, props=[", alloc.active_props.size());
    for (size_t i = 0; i < alloc.active_props.size(); i++) {
      size_t global_p = tree_samples.PropertyFromIndex(tree_samples.NumStaticProps() + alloc.active_props[i]);
      fprintf(stderr, "%zu%s", global_p, i + 1 < alloc.active_props.size() ? ", " : "");
    }
    fprintf(stderr, "], cuts=[");
    for (size_t i = 0; i < alloc.cutoffs.size(); i++) {
      fprintf(stderr, "%zu%s", alloc.cutoffs[i].size(), i + 1 < alloc.cutoffs.size() ? ", " : "");
    }
    fprintf(stderr, "], total_cells=%zu\n", alloc.total_cells);
  }

  if (alloc.active_props.size() == 1) {
    FindBestTree1dDP(tree_samples, scale, tree, sample_subset, base_node_cost, log_node_cost);
    return;
  }

  if (alloc.active_props.size() == 2) {
    size_t pA = alloc.active_props[0];
    size_t pB = alloc.active_props[1];
    const auto& cutsA = alloc.cutoffs[0];
    const auto& cutsB = alloc.cutoffs[1];
    size_t num_binsA = cutsA.size() + 1;
    size_t num_binsB = cutsB.size() + 1;
    size_t num_cells = num_binsA * num_binsB;
    size_t num_entries = num_cells * P;

    std::vector<int32_t> flat_cell_hists(num_entries * S, 0);
    std::vector<int64_t> flat_cell_extra(num_entries, 0);
    std::vector<size_t> cell_sample_count(num_cells, 0);

    for (size_t i = 0; i < total_samples; i++) {
      size_t s = get_sample_idx(i);
      int32_t vA = tree_samples.Property<false>(pA, s);
      int32_t vB = tree_samples.Property<false>(pB, s);
      size_t bA = 0; while (bA < cutsA.size() && vA > cutsA[bA]) bA++;
      size_t bB = 0; while (bB < cutsB.size() && vB > cutsB[bB]) bB++;
      size_t cell = bA * num_binsB + bB;
      cell_sample_count[cell]++;
      size_t cnt = tree_samples.Count(s);
      for (size_t pred = 0; pred < P; pred++) {
        uint32_t tok = tree_samples.Token(pred, s);
        size_t offset = (cell * P + pred) * S;
        flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
        flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
      }
    }

    std::vector<float> single_cell_bits(num_cells, 0.0f);
    std::vector<uint8_t> single_cell_pred(num_cells, 0);
    for (size_t c = 0; c < num_cells; c++) {
      if (cell_sample_count[c] == 0) continue;
      float best_bits = std::numeric_limits<float>::max();
      size_t best_p = 0;
      for (size_t pred = 0; pred < P; pred++) {
        float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                     flat_cell_extra[c * P + pred];
        if (bits < best_bits) {
          best_bits = bits;
          best_p = pred;
        }
      }
      single_cell_bits[c] = best_bits;
      single_cell_pred[c] = static_cast<uint8_t>(best_p);
    }

    std::vector<float> cut_penaltiesA, cut_penaltiesB;
    for (int32_t c : cutsA) {
      int32_t unquant = tree_samples.UnquantizeProperty(pA + tree_samples.NumStaticProps(), c);
      cut_penaltiesA.push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
    }
    for (int32_t c : cutsB) {
      int32_t unquant = tree_samples.UnquantizeProperty(pB + tree_samples.NumStaticProps(), c);
      cut_penaltiesB.push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
    }

    tree->clear();
    tree->emplace_back();
    (*tree)[0] = PropertyDecisionNode::Leaf(Predictor::Zero);

    BuildTreeGreedy2D({0, static_cast<uint16_t>(num_binsA), 0, static_cast<uint16_t>(num_binsB)},
                      pA + tree_samples.NumStaticProps(), cutsA, cut_penaltiesA,
                      pB + tree_samples.NumStaticProps(), cutsB, cut_penaltiesB,
                      num_binsB, P, S, max_symbols,
                      flat_cell_hists.data(), flat_cell_extra.data(), cell_sample_count.data(),
                      single_cell_bits.data(), single_cell_pred.data(),
                      tree_samples, tree, 0);
    return;
  }

  // 3 active properties
  size_t pA = alloc.active_props[0];
  size_t pB = alloc.active_props[1];
  size_t pC = alloc.active_props[2];
  const auto& cutsA = alloc.cutoffs[0];
  const auto& cutsB = alloc.cutoffs[1];
  const auto& cutsC = alloc.cutoffs[2];
  size_t num_binsA = cutsA.size() + 1;
  size_t num_binsB = cutsB.size() + 1;
  size_t num_binsC = cutsC.size() + 1;
  size_t num_cells = num_binsA * num_binsB * num_binsC;
  size_t num_entries = num_cells * P;

  std::vector<int32_t> flat_cell_hists(num_entries * S, 0);
  std::vector<int64_t> flat_cell_extra(num_entries, 0);
  std::vector<size_t> cell_sample_count(num_cells, 0);

  for (size_t i = 0; i < total_samples; i++) {
    size_t s = get_sample_idx(i);
    int32_t vA = tree_samples.Property<false>(pA, s);
    int32_t vB = tree_samples.Property<false>(pB, s);
    int32_t vC = tree_samples.Property<false>(pC, s);
    size_t bA = 0; while (bA < cutsA.size() && vA > cutsA[bA]) bA++;
    size_t bB = 0; while (bB < cutsB.size() && vB > cutsB[bB]) bB++;
    size_t bC = 0; while (bC < cutsC.size() && vC > cutsC[bC]) bC++;
    size_t cell = (bA * num_binsB + bB) * num_binsC + bC;
    cell_sample_count[cell]++;
    size_t cnt = tree_samples.Count(s);
    for (size_t pred = 0; pred < P; pred++) {
      uint32_t tok = tree_samples.Token(pred, s);
      size_t offset = (cell * P + pred) * S;
      flat_cell_hists[offset + (tok < S ? tok : S - 1)] += cnt;
      flat_cell_extra[cell * P + pred] += tree_samples.RTokens(pred)[s].nbits * cnt;
    }
  }

  std::vector<float> single_cell_bits(num_cells, 0.0f);
  std::vector<uint8_t> single_cell_pred(num_cells, 0);
  for (size_t c = 0; c < num_cells; c++) {
    if (cell_sample_count[c] == 0) continue;
    float best_bits = std::numeric_limits<float>::max();
    size_t best_p = 0;
    for (size_t pred = 0; pred < P; pred++) {
      float bits = EstimateBits(&flat_cell_hists[(c * P + pred) * S], max_symbols[pred]) +
                   flat_cell_extra[c * P + pred];
      if (bits < best_bits) {
        best_bits = bits;
        best_p = pred;
      }
    }
    single_cell_bits[c] = best_bits;
    single_cell_pred[c] = static_cast<uint8_t>(best_p);
  }

  std::vector<float> cut_penaltiesA, cut_penaltiesB, cut_penaltiesC;
  for (int32_t c : cutsA) {
    int32_t unquant = tree_samples.UnquantizeProperty(pA + tree_samples.NumStaticProps(), c);
    cut_penaltiesA.push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
  }
  for (int32_t c : cutsB) {
    int32_t unquant = tree_samples.UnquantizeProperty(pB + tree_samples.NumStaticProps(), c);
    cut_penaltiesB.push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
  }
  for (int32_t c : cutsC) {
    int32_t unquant = tree_samples.UnquantizeProperty(pC + tree_samples.NumStaticProps(), c);
    cut_penaltiesC.push_back((base_node_cost + log_node_cost * FastLog2f(std::abs(unquant) + 1.0f)) * scale);
  }

  tree->clear();
  tree->emplace_back();
  (*tree)[0] = PropertyDecisionNode::Leaf(Predictor::Zero);

  BuildTreeGreedy3D({0, static_cast<uint16_t>(num_binsA),
                     0, static_cast<uint16_t>(num_binsB),
                     0, static_cast<uint16_t>(num_binsC)},
                    pA + tree_samples.NumStaticProps(), cutsA, cut_penaltiesA,
                    pB + tree_samples.NumStaticProps(), cutsB, cut_penaltiesB,
                    pC + tree_samples.NumStaticProps(), cutsC, cut_penaltiesC,
                    num_binsB, num_binsC, P, S, max_symbols,
                    flat_cell_hists.data(), flat_cell_extra.data(), cell_sample_count.data(),
                    single_cell_bits.data(), single_cell_pred.data(),
                    tree_samples, tree, 0);
}

void MergeChannelTrees(const std::vector<Tree>& chan_trees, size_t begin, size_t end,
                       size_t min_c, Tree* tree) {
  if (end == begin + 1) {
    size_t sz = tree->size();
    tree->insert(tree->end(), chan_trees[begin].begin(), chan_trees[begin].end());
    for (size_t i = sz; i < tree->size(); i++) {
      if ((*tree)[i].property >= 0) {
        (*tree)[i].lchild += sz;
        (*tree)[i].rchild += sz;
      }
    }
    return;
  }
  size_t mid = (begin + end) / 2;
  size_t splitval = min_c + mid - 1;
  size_t cur = tree->size();
  tree->emplace_back(0 /* channel */, static_cast<int>(splitval), 0, 0,
                     Predictor::Zero, 0, 1);
  (*tree)[cur].lchild = tree->size();
  MergeChannelTrees(chan_trees, mid, end, min_c, tree);
  (*tree)[cur].rchild = tree->size();
  MergeChannelTrees(chan_trees, begin, mid, min_c, tree);
}

void FindBestTreeDispatch(
    TreeSamples &tree_samples, float scale,
    const std::vector<ModularMultiplierInfo> &mul_info,
    StaticPropRange static_prop_range, float fast_decode_multiplier, Tree *tree,
    float nb_repeats, ModularOptions::TreeLearningMode tree_learning_mode,
    float base_node_cost = 92.0f, float log_node_cost = 1.2f) {
  const size_t num_props = tree_samples.NumProperties() - tree_samples.NumStaticProps();
  if (num_props == 0 ||
      tree_learning_mode == ModularOptions::TreeLearningMode::kGreedy ||
      tree_learning_mode == ModularOptions::TreeLearningMode::kMainGreedy) {
    FindBestSplit(tree_samples, scale, mul_info, static_prop_range,
                  fast_decode_multiplier, tree, base_node_cost, log_node_cost);
    return;
  }

  size_t min_c = static_prop_range[0][0];
  size_t max_c = static_prop_range[0][1];
  size_t num_chans = max_c > min_c ? max_c - min_c : 1;

  if (num_chans > 4) {
    FindBestSplit(tree_samples, scale, mul_info, static_prop_range,
                  fast_decode_multiplier, tree, base_node_cost, log_node_cost);
    return;
  }

  if (num_chans > 1 && tree_samples.NumStaticProps() > 0 &&
      tree_learning_mode != ModularOptions::TreeLearningMode::kGreedy &&
      tree_learning_mode != ModularOptions::TreeLearningMode::kMainGreedy) {
    std::vector<std::vector<uint32_t>> chan_samples(num_chans);
    std::vector<uint32_t> chan_q(num_chans);
    for (size_t c = 0; c < num_chans; c++) {
      chan_q[c] = tree_samples.QuantizeStaticProperty(0, min_c + c);
    }
    const size_t total_samples = tree_samples.NumDistinctSamples();
    for (size_t i = 0; i < total_samples; i++) {
      uint32_t q = tree_samples.Property<true>(0, i);
      for (size_t c = 0; c < num_chans; c++) {
        if (q == chan_q[c]) {
          chan_samples[c].push_back(i);
          break;
        }
      }
    }

    std::vector<Tree> chan_trees(num_chans);
    for (size_t c = 0; c < num_chans; c++) {
      chan_trees[c].push_back(PropertyDecisionNode::Leaf(tree_samples.PredictorFromIndex(0)));
      if (tree_learning_mode == ModularOptions::TreeLearningMode::k1dDP) {
        FindBestTree1dDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                         base_node_cost, log_node_cost);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::k2PropertyDP) {
        FindBestTree2PropDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                            base_node_cost, log_node_cost);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kJoint2dDP) {
        FindBestTreeJoint2dDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                              base_node_cost, log_node_cost);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::k3PropertyDP ||
                 tree_learning_mode == ModularOptions::TreeLearningMode::k3PropNestedSingle ||
                 tree_learning_mode == ModularOptions::TreeLearningMode::k3PropNestedStaged ||
                 tree_learning_mode == ModularOptions::TreeLearningMode::k3PropNestedDouble) {
        FindBestTree3PropDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                            base_node_cost, log_node_cost, tree_learning_mode);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid2dDP ||
                 tree_learning_mode == ModularOptions::TreeLearningMode::kGrid2dDP_15) {
        size_t default_cuts = (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid2dDP_15) ? 15 : 7;
        FindBestTreeGrid2dDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                             base_node_cost, log_node_cost, default_cuts);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid3dDP ||
                 tree_learning_mode == ModularOptions::TreeLearningMode::kGrid3dDP_15) {
        size_t default_cuts = (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid3dDP_15) ? 15 : 7;
        FindBestTreeGrid3dDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                             base_node_cost, log_node_cost, default_cuts);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid2dDynDP) {
        FindBestTreeGrid2dDynDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                                base_node_cost, log_node_cost);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid3dDynDP) {
        FindBestTreeGrid3dDynDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                                base_node_cost, log_node_cost);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDynDP) {
        FindBestTreeGridDynDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                              base_node_cost, log_node_cost);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDyn2DP) {
        FindBestTreeGridDynDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                              base_node_cost, log_node_cost, 2048, 2, 1);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDynTop5DP) {
        FindBestTreeGridDynDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                              base_node_cost, log_node_cost, 2048, 3, 5);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDynTop10DP) {
        FindBestTreeGridDynDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                              base_node_cost, log_node_cost, 2048, 3, 10);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDynTop20DP) {
        FindBestTreeGridDynDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                              base_node_cost, log_node_cost, 2048, 3, 20);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDyn2Top5DP) {
        FindBestTreeGridDynDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                              base_node_cost, log_node_cost, 2048, 2, 5);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDyn2Top10DP) {
        FindBestTreeGridDynDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                              base_node_cost, log_node_cost, 2048, 2, 10);
      } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDyn2Top20DP) {
        FindBestTreeGridDynDP(tree_samples, scale, &chan_trees[c], &chan_samples[c],
                              base_node_cost, log_node_cost, 2048, 2, 20);
      }
    }
    tree->clear();
    MergeChannelTrees(chan_trees, 0, num_chans, min_c, tree);
    if (getenv("JXL_PROFILE_1D_DP")) {
      Print1DDPProfilingReport();
    }
    return;
  }

  if (tree_learning_mode == ModularOptions::TreeLearningMode::k1dDP) {
    FindBestTree1dDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::k2PropertyDP) {
    FindBestTree2PropDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kJoint2dDP) {
    FindBestTreeJoint2dDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::k3PropertyDP ||
             tree_learning_mode == ModularOptions::TreeLearningMode::k3PropNestedSingle ||
             tree_learning_mode == ModularOptions::TreeLearningMode::k3PropNestedStaged ||
             tree_learning_mode == ModularOptions::TreeLearningMode::k3PropNestedDouble) {
    FindBestTree3PropDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost, tree_learning_mode);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid2dDP ||
             tree_learning_mode == ModularOptions::TreeLearningMode::kGrid2dDP_15) {
    size_t default_cuts = (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid2dDP_15) ? 15 : 7;
    FindBestTreeGrid2dDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost, default_cuts);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid3dDP ||
             tree_learning_mode == ModularOptions::TreeLearningMode::kGrid3dDP_15) {
    size_t default_cuts = (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid3dDP_15) ? 15 : 7;
    FindBestTreeGrid3dDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost, default_cuts);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid2dDynDP) {
    FindBestTreeGrid2dDynDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGrid3dDynDP) {
    FindBestTreeGrid3dDynDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDynDP) {
    FindBestTreeGridDynDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDyn2DP) {
    FindBestTreeGridDynDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost, 2048, 2, 1);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDynTop5DP) {
    FindBestTreeGridDynDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost, 2048, 3, 5);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDynTop10DP) {
    FindBestTreeGridDynDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost, 2048, 3, 10);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDynTop20DP) {
    FindBestTreeGridDynDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost, 2048, 3, 20);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDyn2Top5DP) {
    FindBestTreeGridDynDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost, 2048, 2, 5);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDyn2Top10DP) {
    FindBestTreeGridDynDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost, 2048, 2, 10);
  } else if (tree_learning_mode == ModularOptions::TreeLearningMode::kGridDyn2Top20DP) {
    FindBestTreeGridDynDP(tree_samples, scale, tree, nullptr, base_node_cost, log_node_cost, 2048, 2, 20);
  } else {
    FindBestSplit(tree_samples, scale, mul_info, static_prop_range,
                  fast_decode_multiplier, tree, base_node_cost, log_node_cost);
  }
  if (getenv("JXL_PROFILE_1D_DP")) {
    Print1DDPProfilingReport();
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(FindBestTreeDispatch);  // Local function.

Status ComputeBestTree(TreeSamples &tree_samples, float scale,
                       const std::vector<ModularMultiplierInfo> &mul_info,
                       StaticPropRange static_prop_range,
                       float fast_decode_multiplier, Tree *tree,
                       float nb_repeats,
                       ModularOptions::TreeLearningMode tree_learning_mode,
                       float base_node_cost,
                       float log_node_cost) {
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
    } else if (strcmp(env_mode, "3prop") == 0 || strcmp(env_mode, "dp3") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::k3PropertyDP;
    } else if (strcmp(env_mode, "n3single") == 0 || strcmp(env_mode, "nested3_single") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::k3PropNestedSingle;
    } else if (strcmp(env_mode, "n3staged") == 0 || strcmp(env_mode, "nested3_staged") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::k3PropNestedStaged;
    } else if (strcmp(env_mode, "n3double") == 0 || strcmp(env_mode, "nested3_double") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::k3PropNestedDouble;
    } else if (strcmp(env_mode, "grid") == 0 || strcmp(env_mode, "grid2d") == 0 || strcmp(env_mode, "grid2d_7") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGrid2dDP;
    } else if (strcmp(env_mode, "grid2d_15") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGrid2dDP_15;
    } else if (strcmp(env_mode, "grid3d") == 0 || strcmp(env_mode, "grid3d_7") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGrid3dDP;
    } else if (strcmp(env_mode, "grid3d_15") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGrid3dDP_15;
    } else if (strcmp(env_mode, "grid2d_dyn") == 0 || strcmp(env_mode, "grid2d_d") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGrid2dDynDP;
    } else if (strcmp(env_mode, "grid3d_dyn") == 0 || strcmp(env_mode, "grid3d_d") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGrid3dDynDP;
    } else if (strcmp(env_mode, "griddyn") == 0 || strcmp(env_mode, "grid_dyn") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGridDynDP;
    } else if (strcmp(env_mode, "griddyn2") == 0 || strcmp(env_mode, "griddyn_2d") == 0 || strcmp(env_mode, "grid2dyn") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGridDyn2DP;
    } else if (strcmp(env_mode, "griddyn_top5") == 0 || strcmp(env_mode, "griddyn5") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGridDynTop5DP;
    } else if (strcmp(env_mode, "griddyn_top10") == 0 || strcmp(env_mode, "griddyn10") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGridDynTop10DP;
    } else if (strcmp(env_mode, "griddyn_top20") == 0 || strcmp(env_mode, "griddyn20") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGridDynTop20DP;
    } else if (strcmp(env_mode, "griddyn2_top5") == 0 || strcmp(env_mode, "griddyn2_5") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGridDyn2Top5DP;
    } else if (strcmp(env_mode, "griddyn2_top10") == 0 || strcmp(env_mode, "griddyn2_10") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGridDyn2Top10DP;
    } else if (strcmp(env_mode, "griddyn2_top20") == 0 || strcmp(env_mode, "griddyn2_20") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGridDyn2Top20DP;
    } else if (strcmp(env_mode, "greedy") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kGreedy;
    } else if (strcmp(env_mode, "main") == 0 || strcmp(env_mode, "orig") == 0) {
      tree_learning_mode = ModularOptions::TreeLearningMode::kMainGreedy;
    }
  }

  if (tree_learning_mode == ModularOptions::TreeLearningMode::kMainGreedy) {
    base_node_cost = 96.0f;
    log_node_cost = 0.0f;
  }

  const char* env_cbase = getenv("JXL_NODE_BASE_COST");
  if (env_cbase != nullptr) {
    base_node_cost = strtof(env_cbase, nullptr);
  }
  const char* env_clog = getenv("JXL_NODE_LOG_COST");
  if (env_clog != nullptr) {
    log_node_cost = strtof(env_clog, nullptr);
  }

  HWY_DYNAMIC_DISPATCH(FindBestTreeDispatch)
  (tree_samples, scale, mul_info, static_prop_range, fast_decode_multiplier,
   tree, nb_repeats, tree_learning_mode, base_node_cost, log_node_cost);
  if (getenv("JXL_DEBUG_TREE")) {
    fprintf(stderr, "Tree mode=%d, scale=%f, base_cost=%f, log_cost=%f, size=%zu, props=%zu, static=%zu\n",
            (int)tree_learning_mode, scale, base_node_cost, log_node_cost, tree->size(),
            tree_samples.NumProperties(), tree_samples.NumStaticProps());
    for (size_t i = 0; i < tree->size(); i++) {
      fprintf(stderr, "  node %zu: prop=%d, val=%d, l=%d, r=%d, pred=%d\n",
              i, (*tree)[i].property, (*tree)[i].splitval, (*tree)[i].lchild, (*tree)[i].rchild, (int)(*tree)[i].predictor);
    }
  }
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

Status TreeSamples::SetProperties(
    const std::vector<uint32_t> &properties,
    ModularOptions::TreeMode wp_tree_mode,
    ModularOptions::TreeLearningMode tree_learning_mode) {
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
    if (tree_learning_mode != ModularOptions::TreeLearningMode::kMainGreedy) {
      bool has_nonstatic = false;
      for (uint32_t p : props_to_use) {
        if (p >= kNumStaticProperties) has_nonstatic = true;
      }
      if (!has_nonstatic) {
        props_to_use.push_back(static_cast<uint32_t>(kGradientProp));
      }
    }
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
