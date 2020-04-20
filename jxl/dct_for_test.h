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

#ifndef JXL_DCT_SLOW_H_
#define JXL_DCT_SLOW_H_

// Unoptimized DCT only for use in tests.

#include <string.h>  // memcpy

#include <cmath>

#include "jxl/common.h"  // Pi

namespace jxl {

static inline double alpha(int u) { return u == 0 ? 0.7071067811865475 : 1.0; }
template <size_t N>
void DCTSlow(double block[N * N]) {
  constexpr size_t kBlockSize = N * N;
  double g[kBlockSize];
  memcpy(g, block, kBlockSize * sizeof(g[0]));
  const double scale = std::sqrt(2.0 / N);
  for (int v = 0; v < N; ++v) {
    for (int u = 0; u < N; ++u) {
      double val = 0.0;
      for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
          val += (alpha(u) * cos((x + 0.5) * u * Pi(1.0 / N)) * alpha(v) *
                  cos((y + 0.5) * v * Pi(1.0 / N)) * g[N * y + x]);
        }
      }
      block[N * v + u] = val * scale * scale;
    }
  }
}

template <size_t N>
void IDCTSlow(double block[N * N]) {
  constexpr size_t kBlockSize = N * N;
  double F[kBlockSize];
  memcpy(F, block, kBlockSize * sizeof(F[0]));
  const double scale = std::sqrt(2.0 / N);
  for (int v = 0; v < N; ++v) {
    for (int u = 0; u < N; ++u) {
      double val = 0.0;
      for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
          val += (alpha(x) * cos(x * (u + 0.5) * Pi(1.0 / N)) * alpha(y) *
                  cos(y * (v + 0.5) * Pi(1.0 / N)) * F[N * y + x]);
        }
      }
      block[N * v + u] = val * scale * scale;
    }
  }
}

}  // namespace jxl

#endif  // JXL_DCT_SLOW_H_
