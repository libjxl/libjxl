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

#include "lib/jxl/jpeg/jpeg_quant_matrix.h"

#include "lib/jxl/base/bits.h"

namespace jxl {
namespace jpeg {

// TODO(eustas): consider high-precision (16-bit) tables in Brunsli v3.
void FillQuantMatrix(bool is_chroma, uint32_t q, uint8_t dst[kDCTBlockSize]) {
  JXL_DASSERT(q < kQFactorLimit);
  const uint8_t* const in = kDefaultQuantMatrix[is_chroma];
  for (size_t i = 0; i < kDCTBlockSize; ++i) {
    const uint32_t v = (in[i] * q + 32) >> 6;
    // clamp to prevent illegal quantizer values
    dst[i] = (v < 1) ? 1 : (v > 255) ? 255u : v;
  }
}

// TODO(eustas): consider high-precision (16-bit) tables in Brunsli v3.
uint32_t FindBestMatrix(const int* src, bool is_chroma,
                        uint8_t dst[kDCTBlockSize]) {
  uint32_t best_q = 0;
  const size_t kMaxDiffCost = 33;
  const size_t kWorstLen = (kDCTBlockSize + 1) * (kMaxDiffCost + 1);
  size_t best_len = kWorstLen;
  for (uint32_t q = 0; q < kQFactorLimit; ++q) {
    FillQuantMatrix(is_chroma, q, dst);
    // Copycat encoder behavior.
    int last_diff = 0;  // difference predictor
    size_t len = 0;
    for (size_t k = 0; k < kDCTBlockSize; ++k) {
      const int j = kJPEGNaturalOrder[k];
      const int new_diff = src[j] - dst[j];
      int diff = new_diff - last_diff;
      last_diff = new_diff;
      if (diff != 0) {
        len += 1;
        if (diff < 0) diff = -diff;
        diff -= 1;
        if (diff == 0) {
          len++;
        } else if (diff > 65535) {
          len = kWorstLen;
          break;
        } else {
          uint32_t diff_len = FloorLog2Nonzero<uint32_t>(diff) + 1;
          if (diff_len == 16) diff_len--;
          len += 2 * diff_len + 1;
        }
      }
    }
    if (len < best_len) {
      best_len = len;
      best_q = q;
    }
  }
  FillQuantMatrix(is_chroma, best_q, dst);
  return best_q;
}

}  // namespace jpeg
}  // namespace jxl
