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

// Functions for clustering similar histograms together.

#ifndef JXL_ENC_CLUSTER_H_
#define JXL_ENC_CLUSTER_H_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <vector>

#include "jxl/ans_params.h"
#include "jxl/enc_ans.h"

namespace jxl {

struct Histogram {
  Histogram() { total_count_ = 0; }
  void Clear() {
    memset(data_, 0, ANS_MAX_ALPHA_SIZE * sizeof(data_[0]));
    total_count_ = 0;
  }
  void Add(size_t symbol) {
    JXL_DASSERT(symbol < ANS_MAX_ALPHA_SIZE);
    ++data_[symbol];
    ++total_count_;
  }
  void AddHistogram(const Histogram& other) {
    for (size_t i = 0; i < ANS_MAX_ALPHA_SIZE; ++i) {
      data_[i] += other.data_[i];
    }
    total_count_ += other.total_count_;
  }
  float PopulationCost() const {
    return ANSPopulationCost(data_, ANS_MAX_ALPHA_SIZE);
  }
  double ShannonEntropy() const;

  ANSHistBin data_[ANS_MAX_ALPHA_SIZE] = {};
  size_t total_count_;
  mutable float entropy_;  // WARNING: not kept up-to-date.
};

void ClusterHistograms(HistogramParams params, const std::vector<Histogram>& in,
                       size_t num_contexts, size_t max_histograms,
                       std::vector<Histogram>* out,
                       std::vector<uint32_t>* histogram_symbols);
}  // namespace jxl

#endif  // JXL_ENC_CLUSTER_H_
