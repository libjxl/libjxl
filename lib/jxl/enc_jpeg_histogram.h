// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_JPEG_HISTOGRAM_H_
#define LIB_JXL_ENC_JPEG_HISTOGRAM_H_

#include "lib/jxl/ac_context.h"
#include "lib/jxl/enc_jpeg_opt_data.h"

namespace jxl {
// Compact histogram for efficient incremental updates.
// Used for AC coefficient histograms as they are large and sparse.
struct CompactHistogram {
  std::vector<uint32_t> counts;
  std::vector<uint32_t> used_ids;
  std::vector<uint32_t> pos_in_used;

  CompactHistogram() = default;
  explicit CompactHistogram(size_t size)
      : counts(size, 0), used_ids(), pos_in_used(size, kInvalidCompactH) {}

  CompactHistogram(const CompactHistogram& other)
      : counts(other.counts.size(), 0),
        used_ids(other.used_ids),
        pos_in_used(other.pos_in_used.size(), kInvalidCompactH) {
    for (uint32_t i = 0; i < used_ids.size(); ++i) {
      uint32_t id = used_ids[i];
      counts[id] = other.counts[id];
      pos_in_used[id] = i;
    }
  }

  CompactHistogram& operator=(const CompactHistogram& other) {
    if (this == &other) return *this;
    if (counts.size() != other.counts.size()) {
      counts.assign(other.counts.size(), 0);
      pos_in_used.assign(other.pos_in_used.size(), kInvalidCompactH);
      used_ids.clear();
    } else {
      Clear();
    }
    used_ids = other.used_ids;
    for (uint32_t i = 0; i < used_ids.size(); ++i) {
      uint32_t id = used_ids[i];
      counts[id] = other.counts[id];
      pos_in_used[id] = i;
    }
    return *this;
  }

  CompactHistogram(CompactHistogram&&) = default;
  CompactHistogram& operator=(CompactHistogram&&) = default;

  bool empty() const { return used_ids.empty(); }

  uint32_t at(uint32_t id) const { return counts[id]; }
  uint32_t Get(uint32_t id) const { return counts[id]; }

  void Add(uint32_t id, uint32_t value = 1) {
    uint32_t& freq = counts[id];
    if (freq == 0) {
      pos_in_used[id] = static_cast<uint32_t>(used_ids.size());
      used_ids.push_back(id);
    }
    freq += value;
  }

  void Subtract(uint32_t id, uint32_t value = 1) {
    uint32_t& freq = counts[id];
    JXL_DASSERT(freq >= value);
    if (freq < value) return;
    freq -= value;
    if (freq != 0) return;
    uint32_t pos = pos_in_used[id];
    JXL_DASSERT(pos != kInvalidCompactH);
    if (pos == kInvalidCompactH) return;
    uint32_t last = used_ids.back();
    used_ids[pos] = last;
    pos_in_used[last] = pos;
    used_ids.pop_back();
    pos_in_used[id] = kInvalidCompactH;
  }

  void AddFrom(const CompactHistogram& other) {
    for (uint32_t id : other.used_ids) Add(id, other.counts[id]);
  }

  void Clear() {
    for (uint32_t id : used_ids) {
      counts[id] = 0;
      pos_in_used[id] = kInvalidCompactH;
    }
    used_ids.clear();
  }

  void swap(CompactHistogram& other) {
    counts.swap(other.counts);
    used_ids.swap(other.used_ids);
    pos_in_used.swap(other.pos_in_used);
  }
};

// Dense histogram with a `CompactHistogram`-like interface for small fixed
// alphabets. Unlike `CompactHistogram`, `empty()` and `Clear()` scan or touch
// the full storage, which is acceptable for these bounded-size tables.
template <size_t Size>
struct DenseHistogram {
  using Array = std::array<uint32_t, Size>;
  using iterator = typename Array::iterator;
  using const_iterator = typename Array::const_iterator;

  DenseHistogram() = default;
  DenseHistogram(const DenseHistogram&) = default;
  DenseHistogram& operator=(const DenseHistogram&) = default;
  DenseHistogram(DenseHistogram&&) = default;
  DenseHistogram& operator=(DenseHistogram&&) = default;

  std::array<uint32_t, Size> counts = {};

  bool empty() const {
    for (uint32_t freq : counts) {
      if (freq != 0) return false;
    }
    return true;
  }

  constexpr size_t size() const { return counts.size(); }

  uint32_t Get(uint32_t id) const { return counts[id]; }

  void Add(uint32_t id, uint32_t value = 1) { counts[id] += value; }

  void Subtract(uint32_t id, uint32_t value = 1) {
    uint32_t& freq = counts[id];
    JXL_DASSERT(freq >= value);
    if (freq < value) return;
    freq -= value;
  }

  void AddFrom(const DenseHistogram& other) {
    for (size_t i = 0; i < Size; ++i) counts[i] += other.counts[i];
  }

  void Clear() { counts.fill(0); }

  void fill(uint32_t value) { counts.fill(value); }

  uint32_t* data() { return counts.data(); }
  const uint32_t* data() const { return counts.data(); }

  uint32_t& operator[](size_t idx) { return counts[idx]; }
  const uint32_t& operator[](size_t idx) const { return counts[idx]; }

  iterator begin() { return counts.begin(); }
  iterator end() { return counts.end(); }
  const_iterator begin() const { return counts.begin(); }
  const_iterator end() const { return counts.end(); }

  void swap(DenseHistogram& other) { counts.swap(other.counts); }
};

using DenseNHistogram = DenseHistogram<kZeroDensityContextCount>;
using DenseNHistogramSet = std::vector<DenseNHistogram>;
using DenseNZPredHistogram = DenseHistogram<kJPEGNonZeroBuckets>;
using DenseNZPredHistogramSet = std::vector<DenseNZPredHistogram>;
using DenseNZHistogram = DenseHistogram<kNZHistogramsSize>;
using DenseNZHistogramSet = std::vector<DenseNZHistogram>;
using CompactHistogramSet = std::vector<CompactHistogram>;

JXL_INLINE uint32_t NZHistogramIndex(uint32_t pb, uint32_t nz_count) {
  JXL_DASSERT(pb < kJPEGNonZeroBuckets);
  JXL_DASSERT(nz_count < kJPEGNonZeroRange);
  return pb * kJPEGNonZeroRange + nz_count;
}

}  // namespace jxl

#endif  // LIB_JXL_ENC_JPEG_HISTOGRAM_H_
