// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Precomputed data model for JPEG lossless recompression.
//
// The JPEG transcode pipeline first converts `jpeg::JPEGData` into a compact
// representation that the threshold optimizer, clustering pass, and refinement
// pass can all share. This file defines that representation and the constants
// used throughout the subsystem.
//
// `JPEGOptData`
//   Owns precomputed DC statistics, compacted AC-symbol ids, per-block side
//   data, and the run-length-encoded AC stream used by the hot optimization
//   passes.
//
// The main build steps are:
//   - `CountDCAC`: collect raw DC and AC occurrence counts from the JPEG input
//   - `BuildBlockOptData`: build per-block DC / nz / AC-bin side tables
//   - `FinalizeSpatialIndexing`: sort blocks by DC value and build rank ranges
//   - `BuildACStream`: emit the packed AC stream consumed by stream walks
//
// All later optimization stages operate on `JPEGOptData` rather than the
// original JPEG structures.

#ifndef LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_OPT_DATA_H_
#define LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_OPT_DATA_H_

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "lib/jxl/ac_context.h"
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

// Invalid entry in sparse-to-dense AC-symbol maps.
constexpr uint32_t kInvalidCompactH = std::numeric_limits<uint32_t>::max();

// Fixed `HybridUintConfig(4, 2, 0)` used by the JPEG-transcode optimizer.
// The maximum token for AC values in `[-1024, 1023]` is 43, so the token
// alphabet has 44 entries.
constexpr uint32_t kACTokenCount = 44;
// Number of AC symbols per histogram: one token alphabet per `zdc`.
constexpr uint32_t kACSymbolCount = kZeroDensityContextCount * kACTokenCount;
// Number of distinct `(channel, zdc, token)` bins stored in `block_bins`.
constexpr uint32_t kACBinCount = kNumCh * kACSymbolCount;

// AC coefficient entry in the packed event stream.
using ACEntry = uint32_t;
// Compact per-coefficient bin id stored in `block_bins`.
using ACBin = uint16_t;
static_assert(kACBinCount <=
                  static_cast<size_t>(std::numeric_limits<ACBin>::max()) + 1,
              "JPEG transcode AC bins must fit in ACBin");
// Vector of DC thresholds for a channel.
using Thresholds = std::vector<int16_t>;
// Context map.
using ContextMap = std::vector<uint8_t>;
// One `(a, b, c)` factorization of DC intervals across Y/Cb/Cr.
using Factorization = std::array<uint32_t, kNumCh>;
// Factorizations of DC thresholds into number of intervals per channel.
using Factorizations = std::vector<Factorization>;

JXL_INLINE double bit_cost(int64_t cost) {
  return static_cast<double>(cost) / kFScale;
}

JXL_INLINE uint32_t MakeJpegTranscodeACSymbol(uint32_t zdc, uint32_t token) {
  JXL_DASSERT(zdc < kZeroDensityContextCount);
  JXL_DASSERT(token < kACTokenCount);
  return zdc * kACTokenCount + token;
}

JXL_INLINE uint32_t JpegTranscodeACSymbolZDC(uint32_t symbol) {
  JXL_DASSERT(symbol < kACSymbolCount);
  return symbol / kACTokenCount;
}

JXL_INLINE uint32_t JpegTranscodeACSymbolToken(uint32_t symbol) {
  JXL_DASSERT(symbol < kACSymbolCount);
  return symbol % kACTokenCount;
}

JXL_INLINE ACBin MakeJpegTranscodeACBin(uint32_t channel, uint32_t zdc,
                                        uint32_t token) {
  JXL_DASSERT(channel < kNumCh);
  return static_cast<ACBin>(channel * kACSymbolCount +
                            MakeJpegTranscodeACSymbol(zdc, token));
}

JXL_INLINE uint32_t JpegTranscodeACBinChannel(uint32_t bin) {
  return bin / kACSymbolCount;
}

JXL_INLINE uint32_t JpegTranscodeACBinSymbol(uint32_t bin) {
  return bin % kACSymbolCount;
}

JXL_INLINE uint32_t JpegTranscodeACBinContext(uint32_t bin) {
  return bin / kACTokenCount;
}

JXL_INLINE uint32_t JpegTranscodeACBinZDC(uint32_t bin) {
  return JpegTranscodeACSymbolZDC(JpegTranscodeACBinSymbol(bin));
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
  // Vertical and horizontal subsampling shifts. The actual factor is
  // `1 << vshift[c]` / `1 << hshift[c]`.
  uint8_t vshift[kNumCh];
  uint8_t hshift[kNumCh];
  // DC values active in image for each component.
  std::vector<int16_t> DC_vals[kNumCh];
  // Counts of each DC value.
  uint32_t DC_cnt[kNumCh][kDCTRange];
  // Index of each DC value in `DC_vals`.
  uint16_t DC_idx_LUT[kNumCh][kDCTRange];

  // Number of `(zdc, token)` bins active in image - `CompactHistogram` size.
  uint32_t num_zdctok;
  // Map from `(zdc, token)` to compact index.
  std::vector<uint32_t> compact_map_h;
  // Map from compact index to `(zdc, token)`.
  std::vector<uint32_t> dense_to_zdctok;

  // Run-length-encoded AC data, sorted by `(bin, dc0, dc1, dc2)`.
  // 32-bit packed format; see `BuildACStream` for layout details.
  std::vector<ACEntry> AC_stream;

  // AC events of consequitive blocks per component.
  std::vector<ACBin> block_bins[kNumCh];
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
  uint32_t CompactHBin(uint32_t zdc_token) const;

 private:
  void InitFTab(size_t max_n);
  StatusOr<std::unique_ptr<ACCounts>> CountDCAC(const jpeg::JPEGData& jpeg_data,
                                                ThreadPool* pool);
  Status BuildBlockOptData(const jpeg::JPEGData& jpeg_data, ThreadPool* pool,
                           const ACCounts& ac_cnt);
  Status FinalizeSpatialIndexing(ThreadPool* pool);
};

// Computes initial DC thresholds for one axis via a 1D surrogate entropy cost.
// Uses the same Knuth DP backend as the full AC-driven optimisation path.
Thresholds InitThresh(const JPEGOptData& d, uint32_t axis,
                      uint32_t target_intervals);

// Enumerates maximal factorizations `(a, b, c)` of the DC threshold count.
// Only maximal factorizations are returned: those where no factor can increase
// without violating `kMaxCells` or `kMaxIntervals` or exceeding DC cardinality.
// For grayscale images (`d.channels == 1`) the Y-axis factor is capped at 15
// because one cluster slot is consumed by the empty Cb+Cr context required by
// the JPEG XL context map format.
Factorizations MaximalFactorizations(const JPEGOptData& d);

// Rescales a block coordinate between component grids using floor semantics.
// The actual per-channel scale is `1 << shift`, so the mapping is a shift.
// Block coordinates stay small enough that the temporary left shift is safe.
JXL_INLINE uint32_t RescaleFloorPow2(uint32_t v, uint32_t src_shift,
                                     uint32_t dst_shift) {
  return (v << src_shift) >> dst_shift;
}

// Rescales a block coordinate between component grids using ceil semantics.
// Used when projecting a block footprint onto another subsampled grid.
// JPEG lossless transcoding only uses 0/1 shifts here, so ceil division by the
// destination scale is implemented by biasing with `dst_shift` before shifting.
JXL_INLINE uint32_t RescaleCeilPow2(uint32_t v, uint32_t src_shift,
                                    uint32_t dst_shift) {
  JXL_DASSERT(src_shift <= 1 && dst_shift <= 1);
  return ((v << src_shift) + dst_shift) >> dst_shift;
}

// Maps block `(y, x)` in `src_channel` to the block in `dst_channel` whose
// top-left anchor is reached by projecting that source block into the common
// plane and flooring back into the destination grid.
JXL_INLINE uint32_t MapTopLeftBlockIndex(const JPEGOptData& d,
                                         uint32_t src_channel, uint32_t y,
                                         uint32_t x, uint32_t dst_channel) {
  return RescaleFloorPow2(y, d.vshift[src_channel], d.vshift[dst_channel]) *
             d.block_grid_w[dst_channel] +
         RescaleFloorPow2(x, d.hshift[src_channel], d.hshift[dst_channel]);
}

}  // namespace jxl

#endif  // LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_OPT_DATA_H_
