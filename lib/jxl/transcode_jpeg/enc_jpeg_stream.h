// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Packed AC-stream format for JPEG lossless recompression.
//
// The transcode pipeline converts per-block AC data into a packed event stream
// that is then scanned repeatedly by threshold optimization and clustering.
// This file defines both sides of that format: the builder that produces the
// packed stream and its derived `(zdc, value)` indexing tables, and the walker
// that decodes the stream for repeated hot scans.
//
// Why this stream needed: the histograms used on DC thresholds optimization
// path can be enormous in size, worst case is
//     `(2048 DC values over current axis) x (32 cells over other axes) x`
//     `(3 x 458 x 2048 of (c,zdc,ai))`,
// and if collect them in parallel in thread pool, the memory pressure is too
// high, and performance drops down as all it gets out of any cache. So
// "bin-by-bin" approach appears to be the best here, we collect histograms per
// each bin in a context by h-terms and per each context by N-terms in sequence,
// tracking just two numbers per histogram and flushing costs as they are ready.
// Cache locality pays off very well: `perf` shows limitation only by memory
// bandwidth of stream reading and lookups.
//
// `ACStreamData`
//   Output of the stream-building pass: packed AC stream, sparse-to-dense AC
//   symbol map, and the maximum per-`zdc` total needed to size `ftab`.
//
// `BuildACStream`
//   Builds `ACStreamData` from the precomputed per-block data in
//   `JPEGOptData`.
//
// `SweepACStream`
//   Stateless decoder over `JPEGOptData::AC_stream`. It walks the packed reset
//   and run frames in order and dispatches them to caller-provided flush and
//   `on_run` callbacks.

#ifndef LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_STREAM_H_
#define LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_STREAM_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "lib/jxl/transcode_jpeg/enc_jpeg_opt_data.h"

namespace jxl {

struct ACStreamData {
  std::vector<ACEntry> stream;
  CompactACHistogramData AC_histogram;
  uint32_t max_zdc_total = 0;
};

StatusOr<ACStreamData> BuildACStream(const JPEGOptData& d, ThreadPool* pool);

// Stateless decoder over the packed AC stream shared by threshold
// optimization and clustering. The stream is sorted by the histogram model
// selected at build time, with raw bins kept contiguous inside each histogram
// bin, and consists of two frame types:
// - Reset frame:
//   `(1<<31) | (ctx_change<<30) | (bin_change<<29) | (bin<<7) | (dc0>>4)`.
// - Normal frame: `(delta_dc0<<27) | (dc1<<16) | (dc2<<5) | (run-1)`,
//   `delta_dc0 <= 15`, so that bit 31 is 0. `run` is always in [1, 32].
//   Runs longer than 32 are split into consecutive normal frames with
//   `delta_dc0 = 0` and the same `dc1`/`dc2` on continuation frames.
// `bin` always stores the raw `(channel, zdc, ai)` bin; `bin_change` tracks
// changes of the selected AC histogram symbol, which can be coarser than the
// raw bin, so a reset frame may update `bin` while leaving `bin_change = 0`.
// Stays in the header as templated callbacks are used in multiple places.
template <class It, class FlushH, class FlushN, class OnRun>
void SweepACStreamRange(It begin, It end, FlushH&& flush_h, FlushN&& flush_N,
                        OnRun&& on_run) {
  uint32_t dc0_idx = 0;
  uint32_t bin_state = 0;
  for (It it = begin; it != end; ++it) {
    const uint32_t frame = *it;
    if (frame >> 31) {
      dc0_idx = (frame & 0x7Fu) << 4;
      bin_state = (frame >> 7) & 0x3FFFFFu;
      if ((frame >> 29) & 1) {
        flush_h();
        if ((frame >> 30) & 1) flush_N();
      }
      continue;
    }
    dc0_idx += (frame >> 27) & 0xFu;
    const uint32_t dc1_idx = (frame >> 16) & 0x7FFu;
    const uint32_t dc2_idx = (frame >> 5) & 0x7FFu;
    const uint32_t run = (frame & 0x1Fu) + 1;
    on_run(dc0_idx, dc1_idx, dc2_idx, run, bin_state);
  }
}

template <class FlushH, class FlushN, class OnRun>
void SweepACStream(const std::vector<ACEntry>& stream, FlushH&& flush_h,
                   FlushN&& flush_N, OnRun&& on_run) {
  SweepACStreamRange(
      stream.begin(), stream.end(), std::forward<FlushH>(flush_h),
      std::forward<FlushN>(flush_N), std::forward<OnRun>(on_run));
}

}  // namespace jxl

#endif  // LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_STREAM_H_
