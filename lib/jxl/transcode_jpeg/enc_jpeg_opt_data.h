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
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/enc_jpeg_frame.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jxl {

///////////////
// Constants //
///////////////

// ---------- `f(n) = n*log2(n)` lookup, fixed-point ----------
// Scale is chosen to keep `f(freq) * kFScale` in `int64_t`.
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

// Fixed `HybridUintConfig(4, 2, 0)` token alphabet used by the JPEG-transcode
// optimizer at efforts 8 and 9. The maximum token for AC values in
// `[-1024, 1023]` is 43, so the token alphabet has 44 entries.
constexpr uint32_t kACTokenCount = 44;
// Worst-case number of AC symbols and `(channel, zdc, value)` bins stored in
// `block_bins`.
constexpr uint32_t kMaxACSymbolCount = kZeroDensityContextCount * kDCTRange;
constexpr uint32_t kMaxACBinCount = kNumCh * kMaxACSymbolCount;

// Layout conventions used by the JPEG-transcode optimizer:
// - raw symbol `(zdc, ai)` = `zdc * kDCTRange + ai`
// - token symbol `(zdc, token)` = `zdc * kACTokenCount + token`
// - raw bin `(channel, zdc, ai)` =
//   `channel * kMaxACSymbolCount + zdc * kDCTRange + ai`
// - `czdc = (channel, zdc)` = `channel * kZeroDensityContextCount + zdc`

// AC histogram model optimized by the JPEG-transcode search.
enum class JPEGTranscodeACModel : uint8_t {
  // Cluster/refine on `(zdc, HybridUint(4,2,0) token)`.
  kToken420,
  // Cluster/refine on raw `(zdc, ai)` values.
  kRawAI,
};

// Sparse-to-dense symbol map for one AC histogram model.
struct CompactACHistogramData {
  // Number of active `(zdc, value)` symbols in dense histogram space.
  uint32_t num_zdcvalue = 0;
  // Sparse map: model symbol -> dense histogram id, or `kInvalidCompactH`.
  std::vector<uint32_t> compact_map_h;
  // Inverse map: dense histogram id -> original model symbol.
  std::vector<uint32_t> dense_to_zdcvalue;

  // Inserts `symbol` on first use and keeps both maps in sync.
  void AddUniqueSymbol(uint32_t symbol) {
    if (compact_map_h[symbol] == kInvalidCompactH) {
      compact_map_h[symbol] = num_zdcvalue++;
      dense_to_zdcvalue.push_back(symbol);
    }
  }
};

// AC coefficient entry in the packed event stream.
using ACEntry = uint32_t;
// Compact per-coefficient bin id stored in `block_bins`.
using ACBin = uint32_t;
static_assert(kMaxACBinCount <=
                  static_cast<size_t>(std::numeric_limits<ACBin>::max()) + 1,
              "JPEG transcode AC bins must fit in ACBin");

struct CompactACEvent {
  // Dense `hist_h` id for the selected AC histogram model.
  uint32_t hist_bin;
  // Accompanying zero-density context for `hist_N`.
  uint32_t zdc;
};

struct SignallingHistSymbol {
  // Zero-density context of one clustered AC histogram symbol.
  uint32_t zdc;
  // Signalling token used when estimating histogram-header overhead.
  uint32_t token;
};
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

  // AC histogram model chosen for this optimizer instance and the matching
  // compact map used by clustering / refinement over the shared raw stream.
  JPEGTranscodeACModel ac_hist_model = JPEGTranscodeACModel::kToken420;
  CompactACHistogramData ac_histogram;

  // Optimizer CfL context with explicit JPEG component mapping.
  // Non-null after `BuildFromJPEG` starts; used by CountDCAC and
  // BuildBlockOptData.
  const JpegCflContext* cfl_ = nullptr;
  // JPEG component -> JXL plane mapping. For true grayscale, component 0 maps
  // to plane 1 to match the encoder's grayscale fast path.
  uint8_t jpeg_to_plane[kNumCh] = {};

  // Run-length-encoded AC data, sorted by the selected histogram model and
  // then by raw `(channel, zdc, ai)` bin. 32-bit packed format; see
  // `BuildACStream` for layout details.
  std::vector<ACEntry> AC_stream;

  // AC events of consecutive blocks per component.
  std::vector<ACBin> block_bins[kNumCh];
  // Indices into `block_bins`, separating consecutive blocks data,
  // size `num_blocks[c] + 1`.
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
  // Indices into `dc_sorted_blocks`, separating different DC indices,
  // size - size `M + 1`, where `M` is the number of distinct DC values.
  std::vector<uint32_t> DC_block_offsets[kNumCh];

  static uint32_t MakeRawSymbol(uint32_t zdc, uint32_t ai) {
    JXL_DASSERT(zdc < kZeroDensityContextCount);
    JXL_DASSERT(ai < kDCTRange);
    return zdc * kDCTRange + ai;
  }
  static uint32_t RawSymbolZDC(uint32_t symbol) {
    JXL_DASSERT(symbol < kMaxACSymbolCount);
    return symbol / kDCTRange;
  }
  static uint32_t RawSymbolAI(uint32_t symbol) {
    JXL_DASSERT(symbol < kMaxACSymbolCount);
    return symbol % kDCTRange;
  }
  static ACBin MakeACBin(uint32_t channel, uint32_t zdc, uint32_t ai) {
    JXL_DASSERT(channel < kNumCh);
    return static_cast<ACBin>(channel * kMaxACSymbolCount +
                              MakeRawSymbol(zdc, ai));
  }
  static uint32_t ACBinChannel(ACBin bin) { return bin / kMaxACSymbolCount; }
  static uint32_t ACBinRawSymbol(ACBin bin) { return bin % kMaxACSymbolCount; }
  static uint32_t ACBinCZDC(ACBin bin) { return bin / kDCTRange; }
  static uint32_t ACBinZDC(ACBin bin) {
    return ACBinRawSymbol(bin) / kDCTRange;
  }
  static uint32_t ACBinAI(ACBin bin) { return bin % kDCTRange; }

  static uint32_t Token420FromAI(uint32_t ai);
  static uint32_t Token420SymbolFromRawSymbol(uint32_t raw_symbol) {
    JXL_DASSERT(raw_symbol < kMaxACSymbolCount);
    return RawSymbolZDC(raw_symbol) * kACTokenCount +
           Token420FromAI(RawSymbolAI(raw_symbol));
  }
  static uint32_t ACBinToken420HistKey(ACBin bin) {
    return ACBinCZDC(bin) * kACTokenCount + Token420FromAI(ACBinAI(bin));
  }
  uint32_t ACHistogramSymbol(ACBin bin) const {
    const uint32_t raw_symbol = ACBinRawSymbol(bin);
    return ac_hist_model == JPEGTranscodeACModel::kRawAI
               ? raw_symbol
               : Token420SymbolFromRawSymbol(raw_symbol);
  }
  uint32_t ACHistogramKey(ACBin bin) const {
    return ac_hist_model == JPEGTranscodeACModel::kRawAI
               ? bin
               : ACBinToken420HistKey(bin);
  }

  const CompactACHistogramData& ACHistogram() const { return ac_histogram; }
  uint32_t ACHistogramSize() const { return ac_histogram.num_zdcvalue; }
  CompactACEvent FromBin(ACBin bin) const {
    uint32_t raw_symbol = ACBinRawSymbol(bin);
    uint32_t zdc = raw_symbol / kDCTRange;
    uint32_t symbol =
        ac_hist_model == JPEGTranscodeACModel::kRawAI
            ? raw_symbol
            : zdc * kACTokenCount + Token420FromAI(raw_symbol % kDCTRange);
    JXL_DASSERT(symbol < ac_histogram.compact_map_h.size());
    if (symbol >= ac_histogram.compact_map_h.size()) {
      return {kInvalidCompactH, zdc};
    }
    uint32_t hist_bin = ac_histogram.compact_map_h[symbol];
    JXL_DASSERT(hist_bin != kInvalidCompactH);
    return {hist_bin, zdc};
  }
  SignallingHistSymbol SignallingHistSymbolFromSymbol(uint32_t symbol) const {
    return ac_hist_model == JPEGTranscodeACModel::kRawAI
               ? SignallingHistSymbol{symbol / kDCTRange,
                                      Token420FromAI(symbol % kDCTRange)}
               : SignallingHistSymbol{symbol / kACTokenCount,
                                      symbol % kACTokenCount};
  }

  // Apply CfL to a single transformed-target AC coefficient.
  // Returns the raw coefficient unchanged when CfL is disabled or when
  // processing the predictor source component.
  // `c` is a JPEG component index.
  // `coeff` is the raw AC value, `coeffpos` is the coefficient's natural-order
  // index within the 8x8 block (1..63 for AC).
  // `block_by`, `block_bx` are the block's position in the component grid.
  // `source_q` points to the 64-coefficient block of the corresponding source
  // component block (Y/G), only used for transformed targets.
  // The result is clamped to [-1024, 1023] to fit in the optimizer's
  // `kDCTRange`-sized histogram tables.
  JXL_INLINE int ApplyCfL(uint32_t c, int16_t coeff, uint32_t coeffpos,
                          uint32_t block_by, uint32_t block_bx,
                          const int16_t* source_q) const {
    JXL_DASSERT(cfl_ != nullptr);
    uint32_t plane = jpeg_to_plane[c];
    if (!cfl_->enabled || plane == 1) {
      return coeff;
    }
    JXL_DASSERT(plane == 0 || plane == 2);
    uint32_t target = plane >> 1;
    int32_t scale = ColorCorrelation::RatioJPEG(cfl_->cfl_map[target]->ConstRow(
        block_by / kColorTileDimInBlocks)[block_bx / kColorTileDimInBlocks]);
    int32_t coeff_scale = (scale * cfl_->scaled_qtable[target][coeffpos] +
                           (1 << (kCFLFixedPointPrecision - 1))) >>
                          kCFLFixedPointPrecision;
    int32_t Y = source_q[coeffpos];
    int32_t cfl_factor =
        (Y * coeff_scale + (1 << (kCFLFixedPointPrecision - 1))) >>
        kCFLFixedPointPrecision;
    int32_t residual = static_cast<int32_t>(coeff) - cfl_factor;
    return Clamp1(residual, -1024, 1023);
  }

  Status BuildFromJPEG(const jpeg::JPEGData& jpeg_data,
                       JPEGTranscodeACModel hist_model,
                       const JpegCflContext& cfl_ctx, ThreadPool* pool);

  int64_t NZFTab(uint32_t n) const;

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
