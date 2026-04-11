// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Packed AC-stream format for JPEG lossless recompression.
//
// The transcode pipeline converts per-block AC data into a packed event stream
// that is then scanned repeatedly by threshold optimization and clustering.
// This file defines both sides of that format: the builder that produces the
// packed stream and its derived `(zdc, token)` indexing tables, and the walker
// that decodes the stream for repeated hot scans.
//
// `ACStreamData`
//   Output of the stream-building pass: packed AC stream, sparse-to-dense AC
//   symbol maps, and the maximum per-`zdc` total needed to size `ftab`.
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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/transcode_jpeg/enc_jpeg_opt_data.h"

namespace jxl {

struct ACStreamData {
  std::vector<ACEntry> stream;
  std::vector<uint32_t> compact_map_h;
  std::vector<uint32_t> dense_to_zdctok;
  uint32_t num_zdctok = 0;
  uint32_t max_zdc_total = 0;
};

StatusOr<ACStreamData> BuildACStream(const JPEGOptData& d, ThreadPool* pool);

// Stateless decoder over the packed AC stream shared by threshold
// optimization and clustering. The stream is sorted by bin index and
// consists of:
// - Reset frame:
//   `(1<<31) | (ctx_change<<30) | (bin_change<<29) | (bin<<7) | (dc0>>4)`.
// - Normal frame: `(delta_dc0<<27) | (dc1<<16) | (dc2<<5) | (run-1)`,
//   `delta_dc0 <= 15`, so that bit 31 is 0.
// - Long-run frame: `(delta_dc0<<27) | (dc1<<16) | (dc2<<5) | 0x1F` followed
//   by `run`.
// Stays in the header as templated callbacks are used in multiple places.
template <class FlushH, class FlushN, class OnRun>
void SweepACStream(const std::vector<ACEntry>& stream, FlushH&& flush_h,
                   FlushN&& flush_N, OnRun&& on_run) {
  uint32_t dc0_idx = 0;
  uint32_t bin_state = 0;
  for (size_t si = 0; si < stream.size(); ++si) {
    const uint32_t frame = stream[si];
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
    const uint32_t run_sym = frame & 0x1Fu;
    const uint32_t run = (run_sym == 0x1Fu) ? stream[++si] : run_sym + 1;
    on_run(dc0_idx, dc1_idx, dc2_idx, run, bin_state);
  }
}

}  // namespace jxl

#endif  // LIB_JXL_TRANSCODE_JPEG_ENC_JPEG_STREAM_H_
