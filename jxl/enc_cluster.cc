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

#include "jxl/enc_cluster.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <tuple>

#include "jxl/base/fast_log.h"
#include "jxl/base/profiler.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "jxl/enc_cluster.cc"
#include <hwy/foreach_target.h>

#include "jxl/fast_log-inl.h"

namespace jxl {

#include <hwy/begin_target-inl.h>

template <class V>
HWY_ATTR V Entropy(V count, V total) {
  const HWY_FULL(float) d;
  const auto zero = Set(d, 0.0f);
  const auto safe_div = Set(d, 1.0f);
  const auto nonzero_count = IfThenElse(count == zero, safe_div, count);
  return count * FastLog2f_18bits(total / nonzero_count);
}

HWY_ATTR void HistogramEntropy(const Histogram& a) {
  a.entropy_ = 0.0f;
  if (a.total_count_ == 0) return;

  const HWY_FULL(float) df;
  const HWY_FULL(int32_t) di;

  const auto tot = Set(df, a.total_count_);
  auto entropy_lanes = Zero(df);

  for (size_t i = 0; i < ANS_MAX_ALPHA_SIZE; i += di.N) {
    const auto counts = LoadU(di, &a.data_[i]);
    entropy_lanes += Entropy(ConvertTo(df, counts), tot);
  }
  a.entropy_ += GetLane(SumOfLanes(entropy_lanes));
}

HWY_ATTR float HistogramDistance(const Histogram& a, const Histogram& b) {
  if (a.total_count_ == 0 || b.total_count_ == 0) return 0;

  const HWY_FULL(float) df;
  const HWY_FULL(int32_t) di;

  const auto tot = Set(df, a.total_count_ + b.total_count_);
  auto distance_lanes = Zero(df);

  for (size_t i = 0; i < ANS_MAX_ALPHA_SIZE; i += di.N) {
    const auto a_counts = LoadU(di, &a.data_[i]);
    const auto b_counts = LoadU(di, &b.data_[i]);
    const auto counts = ConvertTo(df, a_counts + b_counts);
    distance_lanes += Entropy(counts, tot);
  }
  const float total_distance = GetLane(SumOfLanes(distance_lanes));
  return total_distance - a.entropy_ - b.entropy_;
}

// First step of a k-means clustering with a fancy distance metric.
void FastClusterHistograms(const std::vector<Histogram>& in,
                           const size_t num_contexts, size_t max_histograms,
                           float min_distance, std::vector<Histogram>* out,
                           std::vector<uint32_t>* histogram_symbols) {
  PROFILER_FUNC;
  size_t largest_idx = 0;
  for (size_t i = 0; i < num_contexts; i++) {
    HistogramEntropy(in[i]);
    if (in[i].total_count_ > in[largest_idx].total_count_) {
      largest_idx = i;
    }
  }
  out->clear();
  out->reserve(max_histograms);
  std::vector<float> dists(num_contexts, std::numeric_limits<float>::max());
  histogram_symbols->clear();
  histogram_symbols->resize(num_contexts, max_histograms);

  while (out->size() < max_histograms && out->size() < num_contexts) {
    (*histogram_symbols)[largest_idx] = out->size();
    out->push_back(in[largest_idx]);
    largest_idx = 0;
    for (size_t i = 0; i < num_contexts; i++) {
      dists[i] = std::min(HistogramDistance(in[i], out->back()), dists[i]);
      // Avoid repeating histograms
      if ((*histogram_symbols)[i] != max_histograms) continue;
      if (dists[i] > dists[largest_idx]) largest_idx = i;
    }
    if (dists[largest_idx] < min_distance) break;
  }

  for (size_t i = 0; i < num_contexts; i++) {
    if ((*histogram_symbols)[i] != max_histograms) continue;
    size_t best = 0;
    float best_dist = HistogramDistance(in[i], (*out)[best]);
    for (size_t j = 1; j < out->size(); j++) {
      float dist = HistogramDistance(in[i], (*out)[j]);
      if (dist < best_dist) {
        best = j;
        best_dist = dist;
      }
    }
    (*out)[best].AddHistogram(in[i]);
    HistogramEntropy((*out)[best]);
    (*histogram_symbols)[i] = best;
  }
}

#include <hwy/end_target-inl.h>

#if HWY_ONCE
HWY_EXPORT(FastClusterHistograms)

namespace {
inline double CrossEntropy(const ANSHistBin* counts, const size_t counts_len,
                           const ANSHistBin* codes, const size_t codes_len) {
  double sum = 0.0f;
  uint32_t total_count = 0;
  uint32_t total_codes = 0;
  for (size_t i = 0; i < codes_len; ++i) {
    if (codes[i] > 0) {
      if (i < counts_len && counts[i] > 0) {
        sum -= counts[i] * FastLog2f(codes[i]);
        total_count += counts[i];
      }
      total_codes += codes[i];
    }
  }
  if (total_codes > 0) {
    sum += total_count * FastLog2f(total_codes);
  }
  return sum;
}

inline double ShannonEntropy(const ANSHistBin* data, const size_t data_size) {
  return CrossEntropy(data, data_size, data, data_size);
}
}  // namespace

double Histogram::ShannonEntropy() const {
  return jxl::ShannonEntropy(data_, ANS_MAX_ALPHA_SIZE);
}

namespace {
// -----------------------------------------------------------------------------
// Histogram refinement

// Reorder histograms in *out so that the new symbols in *symbols come in
// increasing order.
void HistogramReindex(std::vector<Histogram>* out,
                      std::vector<uint32_t>* symbols) {
  std::vector<Histogram> tmp(*out);
  std::map<int, int> new_index;
  int next_index = 0;
  for (uint32_t symbol : *symbols) {
    if (new_index.find(symbol) == new_index.end()) {
      new_index[symbol] = next_index;
      (*out)[next_index] = tmp[symbol];
      ++next_index;
    }
  }
  out->resize(next_index);
  for (uint32_t& symbol : *symbols) {
    symbol = new_index[symbol];
  }
}

// Clusters together all the histograms.
// Note that most of the speedup from FastestClusterHistograms actually comes
// from having a single context during ANS encoding.
void FastestClusterHistograms(const std::vector<Histogram>& in,
                              const size_t num_contexts, size_t max_histograms,
                              std::vector<Histogram>* out,
                              std::vector<uint32_t>* histogram_symbols) {
  PROFILER_FUNC;
  histogram_symbols->resize(num_contexts, 0);
  out->resize(1);
  (*out)[0] = in[0];
  for (size_t i = 1; i < num_contexts; i++) {
    (*out)[0].AddHistogram(in[i]);
  }
}

}  // namespace

// Clusters similar histograms in 'in' together, the selected histograms are
// placed in 'out', and for each index in 'in', *histogram_symbols will
// indicate which of the 'out' histograms is the best approximation.
void ClusterHistograms(const HistogramParams params,
                       const std::vector<Histogram>& in,
                       const size_t num_contexts, size_t max_histograms,
                       std::vector<Histogram>* out,
                       std::vector<uint32_t>* histogram_symbols) {
  constexpr float kMinDistanceForDistinctFast = 64.0f;
  constexpr float kMinDistanceForDistinctBest = 16.0f;
  auto fast_cluster = ChooseFastClusterHistograms(hwy::SupportedTargets());
  if (params.clustering == HistogramParams::ClusteringType::kFastest) {
    // No reindexing needed.
    return FastestClusterHistograms(in, num_contexts, max_histograms, out,
                                    histogram_symbols);
  } else if (params.clustering == HistogramParams::ClusteringType::kFast) {
    fast_cluster(in, num_contexts, max_histograms, kMinDistanceForDistinctFast,
                 out, histogram_symbols);
  } else {
    PROFILER_FUNC;
    fast_cluster(in, num_contexts, max_histograms, kMinDistanceForDistinctBest,
                 out, histogram_symbols);
    std::vector<uint8_t> alphabet_size(out->size());
    for (size_t i = 0; i < out->size(); i++) {
      for (size_t j = 0; j < ANS_MAX_ALPHA_SIZE; j++) {
        if ((*out)[i].data_[j] != 0) alphabet_size[i] = j + 1;
      }
      (*out)[i].entropy_ = ANSPopulationCost((*out)[i].data_, alphabet_size[i]);
    }
    uint32_t next_version = 2;
    std::vector<uint32_t> version(out->size(), 1);
    std::vector<uint32_t> renumbering(out->size());
    std::iota(renumbering.begin(), renumbering.end(), 0);

    // Try to pair up clusters if doing so reduces the total cost.

    struct HistogramPair {
      // validity of a pair: p.version == max(version[i], version[j])
      float cost;
      uint32_t first;
      uint32_t second;
      uint32_t version;
      // We use > because priority queues sort in *decreasing* order, but we
      // want lower cost elements to appear first.
      bool operator<(const HistogramPair& other) const {
        return std::make_tuple(cost, first, second, version) >
               std::make_tuple(other.cost, other.first, other.second,
                               other.version);
      }
    };

    // Create list of all pairs by increasing merging cost.
    std::priority_queue<HistogramPair> pairs_to_merge;
    for (uint32_t i = 0; i < out->size(); i++) {
      for (uint32_t j = i + 1; j < out->size(); j++) {
        Histogram histo;
        histo.AddHistogram((*out)[i]);
        histo.AddHistogram((*out)[j]);
        float cost =
            ANSPopulationCost(histo.data_,
                              std::max(alphabet_size[i], alphabet_size[j])) -
            (*out)[i].entropy_ - (*out)[j].entropy_;
        // Avoid enqueueing pairs that are not advantageous to merge.
        if (cost >= 0) continue;
        pairs_to_merge.push(
            HistogramPair{cost, i, j, std::max(version[i], version[j])});
      }
    }

    // Merge the best pair to merge, add new pairs that get formed as a
    // consequence.
    while (!pairs_to_merge.empty()) {
      uint32_t first = pairs_to_merge.top().first;
      uint32_t second = pairs_to_merge.top().second;
      uint32_t ver = pairs_to_merge.top().version;
      pairs_to_merge.pop();
      if (ver != std::max(version[first], version[second]) ||
          version[first] == 0 || version[second] == 0) {
        continue;
      }
      (*out)[first].AddHistogram((*out)[second]);
      alphabet_size[first] =
          std::max(alphabet_size[first], alphabet_size[second]);
      (*out)[first].entropy_ =
          ANSPopulationCost((*out)[first].data_, alphabet_size[first]);
      for (size_t i = 0; i < renumbering.size(); i++) {
        if (renumbering[i] == second) {
          renumbering[i] = first;
        }
      }
      version[second] = 0;
      version[first] = next_version++;
      for (uint32_t j = 0; j < out->size(); j++) {
        if (j == first) continue;
        if (version[j] == 0) continue;
        Histogram histo;
        histo.AddHistogram((*out)[first]);
        histo.AddHistogram((*out)[j]);
        float cost =
            ANSPopulationCost(
                histo.data_, std::max(alphabet_size[first], alphabet_size[j])) -
            (*out)[first].entropy_ - (*out)[j].entropy_;
        // Avoid enqueueing pairs that are not advantageous to merge.
        if (cost >= 0) continue;
        pairs_to_merge.push(
            HistogramPair{cost, std::min(first, j), std::max(first, j),
                          std::max(version[first], version[j])});
      }
    }
    std::vector<uint32_t> reverse_renumbering(out->size(), -1);
    size_t num_alive = 0;
    for (size_t i = 0; i < out->size(); i++) {
      if (version[i] == 0) continue;
      (*out)[num_alive++] = (*out)[i];
      reverse_renumbering[i] = num_alive - 1;
    }
    out->resize(num_alive);
    for (size_t i = 0; i < histogram_symbols->size(); i++) {
      (*histogram_symbols)[i] =
          reverse_renumbering[renumbering[(*histogram_symbols)[i]]];
    }
  }

  // Convert the context map to a canonical form.
  HistogramReindex(out, histogram_symbols);
}

#endif  // HWY_ONCE

}  // namespace jxl
