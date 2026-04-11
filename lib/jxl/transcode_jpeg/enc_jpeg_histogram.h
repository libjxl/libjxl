// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Histogram containers used by JPEG lossless recompression.
//
// The threshold optimizer, clustering pass, and refinement pass all manipulate
// per-context histograms of AC symbols, zero-density contexts, and nonzero
// counts. This file provides the small set of histogram types they share.
//
// `CompactHistogram`
//   Runtime-sized AC histogram with dense counts plus a touched-bitset for
//   sparse iteration over nonzero bins and intersections.
//
// `DenseHistogram<Size>`
//   Fixed-size histogram with the same small update API, used for bounded
//   alphabets such as `zdc`, nz predictor buckets, and `(pb, nz_count)` bins.
//
// The common `Get/Add/Subtract/AddFrom/Clear` interface lets clustering and
// refinement code update dense and sparse histograms in the same style.

#ifndef LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_HISTOGRAM_H_
#define LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_HISTOGRAM_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/transcode_jpeg/enc_jpeg_opt_data.h"

namespace jxl {
// Runtime-sized histogram for AC symbols. It keeps direct dense counts for
// cheap point updates plus a touched-bitset for sparse iteration over nonzero
// bins and intersections.
struct CompactHistogram {
  using Word = uint64_t;
  std::vector<uint32_t> counts;
  std::vector<Word> touched_words;
  uint32_t nonzero_count = 0;

  CompactHistogram() = default;
  explicit CompactHistogram(size_t size)
      : counts(size, 0), touched_words((size + 63) / 64, 0) {}

  CompactHistogram(const CompactHistogram& other)
      : counts(other.counts.size(), 0),
        touched_words(other.touched_words),
        nonzero_count(other.nonzero_count) {
    other.ForEachNonZero(
        [&](uint32_t id, uint32_t freq) { counts[id] = freq; });
  }

  CompactHistogram& operator=(const CompactHistogram& other) {
    if (this == &other) return *this;
    if (counts.size() != other.counts.size()) {
      counts.assign(other.counts.size(), 0);
      touched_words.assign(other.touched_words.size(), 0);
    } else {
      Clear();
    }
    touched_words = other.touched_words;
    nonzero_count = other.nonzero_count;
    other.ForEachNonZero(
        [&](uint32_t id, uint32_t freq) { counts[id] = freq; });
    return *this;
  }
  CompactHistogram(CompactHistogram&&) = default;
  CompactHistogram& operator=(CompactHistogram&&) = default;

  bool empty() const { return nonzero_count == 0; }
  size_t nonzeros() const { return nonzero_count; }

  uint32_t Get(uint32_t id) const {
    JXL_DASSERT(id < counts.size());
    return counts[id];
  }

  template <typename Func>
  void ForEachNonZero(Func&& fn) const {
    for (size_t word_idx = 0; word_idx < touched_words.size(); ++word_idx) {
      Word word = touched_words[word_idx];
      while (word != 0) {
        uint32_t bit = static_cast<uint32_t>(Num0BitsBelowLS1Bit_Nonzero(word));
        uint32_t id = static_cast<uint32_t>(word_idx * 64 + bit);
        fn(id, counts[id]);
        word &= word - 1;
      }
    }
  }

  template <typename Func>
  void ForEachIntersection(const CompactHistogram& other, Func&& fn) const {
    JXL_DASSERT(touched_words.size() == other.touched_words.size());
    const size_t word_count =
        std::min(touched_words.size(), other.touched_words.size());
    for (size_t word_idx = 0; word_idx < word_count; ++word_idx) {
      Word word = touched_words[word_idx] & other.touched_words[word_idx];
      while (word != 0) {
        uint32_t bit = static_cast<uint32_t>(Num0BitsBelowLS1Bit_Nonzero(word));
        uint32_t id = static_cast<uint32_t>(word_idx * 64 + bit);
        fn(id, counts[id], other.counts[id]);
        word &= word - 1;
      }
    }
  }

  void Add(uint32_t id, uint32_t value = 1) {
    JXL_DASSERT(id < counts.size());
    if (value == 0) return;
    uint32_t& freq = counts[id];
    if (freq == 0) {
      touched_words[id / 64] |= Word{1} << (id & 63);
      ++nonzero_count;
    }
    freq += value;
  }

  void Subtract(uint32_t id, uint32_t value = 1) {
    JXL_DASSERT(id < counts.size());
    uint32_t& freq = counts[id];
    JXL_DASSERT(freq >= value);
    if (freq < value) return;
    freq -= value;
    if (freq != 0) return;
    touched_words[id / 64] &= ~(Word{1} << (id & 63));
    JXL_DASSERT(nonzero_count > 0);
    --nonzero_count;
  }

  void AddHistogram(const CompactHistogram& other) {
    other.ForEachNonZero([&](uint32_t id, uint32_t freq) { Add(id, freq); });
  }

  void SubtractHistogram(const CompactHistogram& other) {
    other.ForEachNonZero(
        [&](uint32_t id, uint32_t freq) { Subtract(id, freq); });
  }

  void Clear() {
    for (size_t word_idx = 0; word_idx < touched_words.size(); ++word_idx) {
      Word word = touched_words[word_idx];
      while (word != 0) {
        uint32_t bit = static_cast<uint32_t>(Num0BitsBelowLS1Bit_Nonzero(word));
        counts[word_idx * 64 + bit] = 0;
        word &= word - 1;
      }
      touched_words[word_idx] = 0;
    }
    nonzero_count = 0;
  }

  void swap(CompactHistogram& other) {
    counts.swap(other.counts);
    touched_words.swap(other.touched_words);
    std::swap(nonzero_count, other.nonzero_count);
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

  void AddHistogram(const DenseHistogram& other) {
    for (size_t i = 0; i < Size; ++i) counts[i] += other.counts[i];
  }

  void SubtractHistogram(const DenseHistogram& other) {
    for (size_t i = 0; i < Size; ++i) {
      JXL_DASSERT(counts[i] >= other.counts[i]);
      if (counts[i] < other.counts[i]) continue;
      counts[i] -= other.counts[i];
    }
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

// Concrete histogram aliases used by clustering and refinement:
// `N` histograms track zero-density-context totals, `NZPred` tracks predictor
// bucket totals, `NZ` tracks `(predictor bucket, nonzero count)` bins, and
// `CompactHistogramSet` stores the touched-bitset AC-symbol histograms per
// cluster.
using DenseNHistogram = DenseHistogram<kZeroDensityContextCount>;
using DenseNHistogramSet = std::vector<DenseNHistogram>;
using DenseNZPredHistogram = DenseHistogram<kJPEGNonZeroBuckets>;
using DenseNZPredHistogramSet = std::vector<DenseNZPredHistogram>;
using DenseNZHistogram = DenseHistogram<kNZHistogramsSize>;
using DenseNZHistogramSet = std::vector<DenseNZHistogram>;
using CompactHistogramSet = std::vector<CompactHistogram>;

// Flattens the 2D `(predictor bucket, nonzero count)` space into the linear
// storage layout used by `DenseNZHistogram`.
JXL_INLINE uint32_t NZHistogramIndex(uint32_t pb, uint32_t nz_count) {
  JXL_DASSERT(pb < kJPEGNonZeroBuckets);
  JXL_DASSERT(nz_count < kJPEGNonZeroRange);
  return pb * kJPEGNonZeroRange + nz_count;
}

}  // namespace jxl

#endif  // LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_HISTOGRAM_H_
