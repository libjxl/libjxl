// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_JPEG_INTERNAL_H_
#define LIB_JXL_ENC_JPEG_INTERNAL_H_

#include <array>
#include <vector>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {

///////////////
// Constants //
///////////////

// ---------- `f(n) = n*log2(n)` lookup, fixed-point ----------
// Scale is choosen to keep `f(freq) * kFScale` in `int64_t`.
// Overflow bound: max number of blocks `N = 2^26` for max JPEG image,
// max 4 AC positions in bin (see `kCoeffFreqContext`), then max bin `freq` is
// 2^28 and max `f(freq) = 2^28 * 28`, fixed point `f(freq) * kFScale = 2^53 *
// 28` fits in `int64_t` (and there is still room for clustering that can
// increase `n` three-fold).
constexpr int64_t kFScale = 1LL << 25;

// Partitioning limits are defined by the JPEG XL standard I.2.2.
// Max number of DC cells in 3D DC space.
constexpr int kMaxCells = 64;
// Each DC threshold count is written with 4 bits.
constexpr uint32_t kMaxIntervals = 16;
// Max number of clusters in context map.
constexpr int kMaxClusters = 16;

// JPEG DCT coefficients range: [-1024, 1024)
constexpr int kDCTRange = 2048;
constexpr int kDCTOff = 1024;  // shift into [0, 2048)
// Number of AC positions in a block.
constexpr int kNumPos = 63;
// Number of channels.
constexpr int kNumCh = 3;

// Target number of rarefied DC thresholds to reduce size of cost matrix
// for Knuth DP optimization of Dc axis partitioning.
constexpr uint32_t kMTarget = 256;

// `nz` is number of nonzero AC coefficients in a block.
// Number of contexts for prediction of `nz`, one less than `kNonZeroBuckets`.
constexpr uint32_t kJPEGNonZeroBuckets = 36;
// Number of possible values for `nz` [0, 64).
constexpr uint32_t kJPEGNonZeroRange = 64;
// Number of bins in all histograms for `nz` per context.
constexpr uint32_t kNZHistogramsSize = kJPEGNonZeroBuckets * kJPEGNonZeroRange;

// Invalid position in `pos_in_used`.
constexpr uint32_t kInvalidCompactH = std::numeric_limits<uint32_t>::max();

// Vector of DC thresholds for a channel.
using Thresholds = std::vector<int16_t>;
// AC coefficient entry in the event stream.
using ACEntry = uint32_t;
// Context map.
using ContextMap = std::vector<uint8_t>;
// Factorizations of DC thresholds into number of intervals per channel.
using Factorizations = std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>;

JXL_INLINE int CountrZero64(uint64_t x) {
#if JXL_COMPILER_MSVC
  unsigned long idx;
  return _BitScanForward64(&idx, x) ? static_cast<int>(idx) : 64;
#else
  return x ? __builtin_ctzll(x) : 64;
#endif
}

JXL_INLINE double bit_cost(int64_t cost) {
  return static_cast<double>(cost) / kFScale;
}

// Set of DC thresholds for each channel.
struct ThresholdSet {
  std::array<Thresholds, kNumCh> T;
  Thresholds& TY() { return T[0]; }
  const Thresholds& TY() const { return T[0]; }
  Thresholds& TCb() { return T[1]; }
  const Thresholds& TCb() const { return T[1]; }
  Thresholds& TCr() { return T[2]; }
  const Thresholds& TCr() const { return T[2]; }
};

struct JPEGOptData {
  // Histogram of AC coefficients per channel, position and value.
  using ACCounts =
      std::array<std::array<std::array<uint32_t, kDCTRange>, kNumPos>, kNumCh>;

  // Fixed-point entropy table: `f(n) = n*log2(n)*kFScale`.
  std::vector<int64_t> ftab;

  uint32_t channels;
  // Number of 8x8 blocks in image for each component.
  uint32_t num_blocks[kNumCh];
  // Block grid dimensions per component.
  uint32_t block_grid_w[kNumCh];
  uint32_t block_grid_h[kNumCh];
  // Max width and height of the block grid.
  uint32_t w_max;
  uint32_t h_max;
  // Vertical and horizontal subsampling factors.
  uint32_t ss_y[kNumCh];
  uint32_t ss_x[kNumCh];
  // DC values active in image for each component.
  std::vector<int16_t> DC_vals[kNumCh];
  // Counts of each DC value.
  uint32_t DC_cnt[kNumCh][kDCTRange];
  // Index of each DC value in `DC_vals`.
  uint16_t DC_idx_LUT[kNumCh][kDCTRange];

  // Number of `(zdc,ai)` bins active in image - `CompactHistogram` size.
  uint32_t num_zdcai;
  // Map from `(zdc,ai)` to compact index.
  std::vector<uint32_t> compact_map_h;
  // Map from compact index to `(zdc,ai)`.
  std::vector<uint32_t> dense_to_zdcai;

  // Run-length-encoded AC data, sorted by `(bin, dc0, dc1, dc2)`.
  // 32-bit packed format; see `GenerateRLEStream` for layout details.
  std::vector<ACEntry> AC_stream;

  // AC events of consequitive blocks per component.
  std::vector<uint32_t> block_bins[kNumCh];
  // Indices into `block_bins`, separating consequent blocks data,
  // size `block_grid_h[c] * block_grid_w[c]`.
  std::vector<uint32_t> block_offsets[kNumCh];
  // Block nonzero number and nonzero prediction context,
  // size `block_grid_h[c] * block_grid_w[c]`.
  std::vector<uint8_t> block_nonzeros[kNumCh];
  std::vector<uint8_t> block_nz_pred_bucket[kNumCh];
  // DC indices of blocks of components (DC of the block component),
  // size `block_grid_h[c] * block_grid_w[c]`.
  std::vector<uint16_t> block_DC_idx[kNumCh];
  // Coordinates of component blocks, sorted by DC index,
  // size `block_grid_h[c] * block_grid_w[c]`.
  // `y` in 16 MSB, `x` in 16 LSB.
  std::vector<uint32_t> DC_sorted_blocks[kNumCh];
  // Indices into `dc_sorted_blocks, separating different DC indices,
  // size - number of active DC values `M_comp`.
  std::vector<uint32_t> DC_block_offsets[kNumCh];

  Status BuildFromJPEG(const jpeg::JPEGData& jpeg_data, ThreadPool* pool);

  int64_t NZFTab(uint32_t n) const;
  uint32_t CompactHBin(uint32_t zdc_ai) const;
  Thresholds InitThresh(int axis, uint32_t target_intervals) const;
  Factorizations MaximalFactorizations() const;

 private:
  void InitFTab(size_t max_n);
  StatusOr<std::unique_ptr<ACCounts>> CountDCAC(const jpeg::JPEGData& jpeg_data,
                                                ThreadPool* pool);
  Status BuildBlockOptData(const jpeg::JPEGData& jpeg_data, ThreadPool* pool,
                           const ACCounts& ac_cnt);
  Status FinalizeSpatialIndexing(ThreadPool* pool);
  Status GenerateRLEStream(ThreadPool* pool);
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_JPEG_INTERNAL_H_
