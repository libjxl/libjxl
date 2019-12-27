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

#include <stdint.h>

#include <limits>
#include <map>
#include <memory>

#include "jxl/base/fast_log.h"
#include "jxl/base/profiler.h"

namespace jxl {

namespace {
inline double CrossEntropy(const ANSHistBin* counts, const size_t counts_len,
                           const ANSHistBin* codes, const size_t codes_len) {
  // TODO(veluca): use FastLog2f?
  double sum = 0.0f;
  uint32_t total_count = 0;
  uint32_t total_codes = 0;
  for (size_t i = 0; i < codes_len; ++i) {
    if (codes[i] > 0) {
      if (i < counts_len && counts[i] > 0) {
        sum -= counts[i] * std::log2(codes[i]);
        total_count += counts[i];
      }
      total_codes += codes[i];
    }
  }
  if (total_codes > 0) {
    sum += total_count * std::log2(total_codes);
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

struct HistogramPair {
  uint32_t idx1;
  uint32_t idx2;
  float cost_combo;
  float cost_diff;
};

inline bool operator<(const HistogramPair& p1, const HistogramPair& p2) {
  if (p1.cost_diff != p2.cost_diff) {
    return p1.cost_diff > p2.cost_diff;
  }
  return std::abs(static_cast<int32_t>(p1.idx1) -
                  static_cast<int32_t>(p1.idx2)) >
         std::abs(static_cast<int32_t>(p2.idx1) -
                  static_cast<int32_t>(p2.idx2));
}

// Returns entropy reduction of the context map when we combine two clusters.
inline float ClusterCostDiff(int size_a, int size_b) {
  int size_c = size_a + size_b;
  return size_a * FastLog2(size_a) + size_b * FastLog2(size_b) -
         size_c * FastLog2(size_c);
}

// Computes the bit cost reduction by combining out[idx1] and out[idx2] and if
// it is below a threshold, stores the pair (idx1, idx2) in the *pairs queue.
void CompareAndPushToQueue(const Histogram* out, const int* cluster_size,
                           const float* bit_cost, int idx1, int idx2,
                           std::vector<HistogramPair>* pairs) {
  if (idx1 == idx2) {
    return;
  }
  if (idx2 < idx1) {
    uint32_t t = idx2;
    idx2 = idx1;
    idx1 = t;
  }
  bool store_pair = false;
  HistogramPair p;
  p.idx1 = idx1;
  p.idx2 = idx2;
  p.cost_diff = 0.5f * ClusterCostDiff(cluster_size[idx1], cluster_size[idx2]);
  p.cost_diff -= bit_cost[idx1];
  p.cost_diff -= bit_cost[idx2];

  if (out[idx1].total_count_ == 0) {
    p.cost_combo = bit_cost[idx2];
    store_pair = true;
  } else if (out[idx2].total_count_ == 0) {
    p.cost_combo = bit_cost[idx1];
    store_pair = true;
  } else {
    const float threshold = pairs->empty()
                                ? std::numeric_limits<float>::max()
                                : std::max(0.0f, (*pairs)[0].cost_diff);
    Histogram combo = out[idx1];
    combo.AddHistogram(out[idx2]);
    float cost_combo = combo.PopulationCost();
    if (cost_combo + p.cost_diff < threshold) {
      p.cost_combo = cost_combo;
      store_pair = true;
    }
  }
  if (store_pair) {
    p.cost_diff += p.cost_combo;
    if (!pairs->empty() && (pairs->front() < p)) {
      // Replace the top of the queue if needed.
      pairs->push_back(pairs->front());
      pairs->front() = p;
    } else {
      pairs->push_back(p);
    }
  }
}

int HistogramCombine(Histogram* out, int* cluster_size, float* bit_cost,
                     uint32_t* symbols, size_t symbols_size,
                     size_t max_clusters) {
  float cost_diff_threshold = 0.0f;
  size_t min_cluster_size = 1;

  // Uniquify the list of symbols after merging empty clusters.
  std::vector<uint32_t> clusters;
  clusters.reserve(symbols_size);
  int64_t sum_of_totals = 0;
  int first_zero_pop_count_symbol = -1;
  for (size_t i = 0; i < symbols_size; ++i) {
    if (out[symbols[i]].total_count_ == 0) {
      // Merge the zero pop count histograms into one.
      if (first_zero_pop_count_symbol == -1) {
        first_zero_pop_count_symbol = symbols[i];
        clusters.push_back(symbols[i]);
      } else {
        symbols[i] = first_zero_pop_count_symbol;
      }
    } else {
      // Insert all histograms with non-zero pop counts.
      clusters.push_back(symbols[i]);
      sum_of_totals += out[symbols[i]].total_count_;
    }
  }
  if (sum_of_totals < 160) {
    // Use a single histogram if there are only a few samples.
    // This helps with small images (like 64x64 size) where the
    // context map is more expensive than the related savings.
    // TODO: Estimate the the actual difference in bitcost to
    // make the final decision of this strategy and clustering.
    *cluster_size = 1;
    Histogram combo = out[symbols[0]];
    for (size_t i = 1; i < symbols_size; ++i) {
      combo.AddHistogram(out[symbols[i]]);
    }
    out[symbols[0]] = combo;
    for (size_t i = 1; i < symbols_size; ++i) {
      symbols[i] = symbols[0];
    }
    return 1;
  }
  std::sort(clusters.begin(), clusters.end());
  clusters.resize(std::unique(clusters.begin(), clusters.end()) -
                  clusters.begin());

  // We maintain a priority queue of histogram pairs, ordered by the bit cost
  // reduction. For efficiency, only the front of the queue matters, the rest of
  // it is unordered.
  std::vector<HistogramPair> pairs;
  for (size_t idx1 = 0; idx1 < clusters.size(); ++idx1) {
    for (size_t idx2 = idx1 + 1; idx2 < clusters.size(); ++idx2) {
      CompareAndPushToQueue(out, cluster_size, bit_cost, clusters[idx1],
                            clusters[idx2], &pairs);
    }
  }

  while (clusters.size() > min_cluster_size) {
    if (pairs[0].cost_diff >= cost_diff_threshold) {
      cost_diff_threshold = std::numeric_limits<float>::max();
      min_cluster_size = max_clusters;
      continue;
    }

    // Take the best pair from the top of queue.
    uint32_t best_idx1 = pairs[0].idx1;
    uint32_t best_idx2 = pairs[0].idx2;
    out[best_idx1].AddHistogram(out[best_idx2]);
    bit_cost[best_idx1] = pairs[0].cost_combo;
    cluster_size[best_idx1] += cluster_size[best_idx2];
    for (size_t i = 0; i < symbols_size; ++i) {
      if (symbols[i] == best_idx2) {
        symbols[i] = best_idx1;
      }
    }
    for (auto cluster = clusters.begin(); cluster != clusters.end();
         ++cluster) {
      if (*cluster >= best_idx2) {
        clusters.erase(cluster);
        break;
      }
    }

    // Remove pairs intersecting the just combined best pair.
    auto copy_to = pairs.begin();
    for (size_t i = 0; i < pairs.size(); ++i) {
      HistogramPair& p = pairs[i];
      if (p.idx1 == best_idx1 || p.idx2 == best_idx1 || p.idx1 == best_idx2 ||
          p.idx2 == best_idx2) {
        // Remove invalid pair from the queue.
        continue;
      }
      if (pairs.front() < p) {
        // Replace the top of the queue if needed.
        auto front = pairs.front();
        pairs.front() = p;
        *copy_to = front;
      } else {
        *copy_to = p;
      }
      ++copy_to;
    }
    pairs.resize(copy_to - pairs.begin());

    // Push new pairs formed with the combined histogram to the queue.
    for (size_t i = 0; i < clusters.size(); ++i) {
      CompareAndPushToQueue(out, cluster_size, bit_cost, best_idx1, clusters[i],
                            &pairs);
    }
  }
  return clusters.size();
}

// -----------------------------------------------------------------------------
// Histogram refinement

// What is the bit cost of moving histogram from cur_symbol to candidate.
float HistogramBitCostDistance(const Histogram& histogram,
                               const Histogram& candidate,
                               const float candidate_bit_cost) {
  if (histogram.total_count_ == 0) {
    return 0.0;
  }
  Histogram tmp = histogram;
  tmp.AddHistogram(candidate);
  return tmp.PopulationCost() - candidate_bit_cost;
}

// Find the best 'out' histogram for each of the 'in' histograms.
// Note: we assume that out[]->bit_cost_ is already up-to-date.
void HistogramRemap(const Histogram* in, int in_size, Histogram* out,
                    float* bit_cost, uint32_t* symbols) {
  // Uniquify the list of symbols.
  std::vector<int> all_symbols(symbols, symbols + in_size);
  std::sort(all_symbols.begin(), all_symbols.end());
  all_symbols.resize(std::unique(all_symbols.begin(), all_symbols.end()) -
                     all_symbols.begin());

  for (int i = 0; i < in_size; ++i) {
    int best_out = i == 0 ? symbols[0] : symbols[i - 1];
    float best_bits =
        HistogramBitCostDistance(in[i], out[best_out], bit_cost[best_out]);
    for (auto k : all_symbols) {
      const float cur_bits =
          HistogramBitCostDistance(in[i], out[k], bit_cost[k]);
      if (cur_bits < best_bits) {
        best_bits = cur_bits;
        best_out = k;
      }
    }
    symbols[i] = best_out;
  }

  // Recompute each out based on raw and symbols.
  for (auto k : all_symbols) {
    out[k].Clear();
  }
  for (int i = 0; i < in_size; ++i) {
    out[symbols[i]].AddHistogram(in[i]);
  }
}

// Reorder histograms in *out so that the new symbols in *symbols come in
// increasing order.
void HistogramReindex(std::vector<Histogram>* out,
                      std::vector<uint32_t>* symbols) {
  std::vector<Histogram> tmp(*out);
  std::map<int, int> new_index;
  int next_index = 0;
  for (size_t i = 0; i < symbols->size(); ++i) {
    if (new_index.find((*symbols)[i]) == new_index.end()) {
      new_index[(*symbols)[i]] = next_index;
      (*out)[next_index] = tmp[(*symbols)[i]];
      ++next_index;
    }
  }
  out->resize(next_index);
  for (size_t i = 0; i < symbols->size(); ++i) {
    (*symbols)[i] = new_index[(*symbols)[i]];
  }
}

float Entropy(size_t count, float total) {
  if (count == 0) return 0;
  // TODO(veluca): FastLog2f?
  return count * std::log2(total / count);
}

void HistogramEntropy(const Histogram& a) {
  a.entropy_ = 0;
  if (a.total_count_ == 0) return;
  float tot = a.total_count_;
  for (size_t i = 0; i < ANS_MAX_ALPHA_SIZE; i++) {
    a.entropy_ += Entropy(a.data_[i], tot);
  }
}

float HistogramDistance(const Histogram& a, const Histogram& b) {
  if (a.total_count_ == 0 || b.total_count_ == 0) return 0;
  float tot = a.total_count_ + b.total_count_;
  float total_distance = -a.entropy_ - b.entropy_;
  for (size_t i = 0; i < ANS_MAX_ALPHA_SIZE; i++) {
    size_t va = a.data_[i];
    size_t vb = b.data_[i];
    total_distance += Entropy(va + vb, tot);
  }
  return total_distance;
}

// First step of a k-means clustering with a fancy distance metric.
void FastClusterHistograms(const std::vector<Histogram>& in,
                           const size_t num_contexts, size_t max_histograms,
                           std::vector<Histogram>* out,
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

  constexpr float kMinDistanceForDistinct = 64.0f;
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
    if (dists[largest_idx] < kMinDistanceForDistinct) break;
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
  // Convert the context map to a canonical form.
  HistogramReindex(out, histogram_symbols);
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
  return;
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
  if (params.clustering == HistogramParams::ClusteringType::kFastest) {
    return FastestClusterHistograms(in, num_contexts, max_histograms, out,
                                    histogram_symbols);
  }
  if (params.clustering == HistogramParams::ClusteringType::kFast) {
    return FastClusterHistograms(in, num_contexts, max_histograms, out,
                                 histogram_symbols);
  }
  PROFILER_FUNC;

  const int in_size = num_contexts;
  std::vector<int> cluster_size(in_size, 1);
  std::vector<float> bit_cost(in_size);
  out->resize(in_size);
  histogram_symbols->resize(in_size);

  for (int i = 0; i < in_size; ++i) {
    (*out)[i] = in[i];
    bit_cost[i] = in[i].PopulationCost();
    (*histogram_symbols)[i] = i;
  }

  // Collapse similar histograms within a block type.
  if (num_contexts > 1) {
    HistogramCombine(&(*out)[0], &cluster_size[0], &bit_cost[0],
                     &(*histogram_symbols)[0], num_contexts, max_histograms);
  }

  static const int kMaxClustersForHistogramRemap = 255;

  int num_clusters = 0;
  {
    PROFILER_ZONE("CLUSTER final");
    // There are no longer "block groups", so need a final round of clustering.
    num_clusters =
        HistogramCombine(&(*out)[0], &cluster_size[0], &bit_cost[0],
                         &(*histogram_symbols)[0], in_size, max_histograms);
    // Find the optimal map from original histograms to the final ones.
    if (num_clusters >= 2 && num_clusters <= kMaxClustersForHistogramRemap) {
      HistogramRemap(&in[0], in_size, &(*out)[0], &bit_cost[0],
                     &(*histogram_symbols)[0]);
    }
  }

  // Convert the context map to a canonical form.
  HistogramReindex(out, histogram_symbols);
}

}  // namespace jxl
