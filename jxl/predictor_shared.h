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

#ifndef JXL_PREDICTOR_SHARED_H_
#define JXL_PREDICTOR_SHARED_H_

#include <stddef.h>
#include <algorithm>
#include <hwy/base.h>  // HWY_ALIGN_MAX

#include "jxl/base/compiler_specific.h"
#include "jxl/common.h"

namespace jxl {

constexpr size_t kNumPredictors = 8;

// Returns the average of two int32_t values without overflowing:
// truncate((a + b) / 2).
static JXL_INLINE int32_t Average(int64_t a, int64_t b) { return (a + b) / 2; }

// Clamps gradient to the min/max of n, w (and l, implicitly).
static JXL_INLINE int32_t ClampedGradient(const int32_t n, const int32_t w,
                                          const int32_t l) {
  const int32_t m = std::min(n, w);
  const int32_t M = std::max(n, w);
  // The end result of this operation doesn't overflow or underflow if the
  // result is between m and M, but the intermediate value may overflow, so we
  // do the intermediate operations in uint32_t and check later if we had an
  // overflow or underflow condition comparing m, M and l directly.
  // grad = M + m - l = n + w - l
  const int32_t grad =
      static_cast<int32_t>(static_cast<uint32_t>(n) + static_cast<uint32_t>(w) -
                           static_cast<uint32_t>(l));
  // We use two sets of ternary operators to force the evaluation of them in
  // any case, allowing the compiler to avoid branches and use cmovl/cmovg in
  // x86.
  const int32_t grad_clamp_M = (l < m) ? M : grad;
  return (l > M) ? m : grad_clamp_M;
}

// Ensure that 32-bit systems do not overflow when doing simple arithmetic on
// the max error.
static constexpr uint32_t kMaxError = std::numeric_limits<uint32_t>::max() >> 1;

struct YPredictor {
  static JXL_INLINE void Predict(const int32_t n, const int32_t w,
                                 const int32_t l, const int32_t r,
                                 int32_t* JXL_RESTRICT pred) {
    pred[0] = Average(Average(n, w), r);
    pred[1] = Average(w, n);
    pred[2] = Average(n, r);
    pred[3] = Average(w, l);
    pred[4] = Average(n, l);
    pred[5] = w;
    pred[6] = ClampedGradient(n, w, l);
    pred[7] = n;
  }

  // 0 = valid predictor, kMaxError = invalid.

  // All predictors but w are invalid in row 0.
  static JXL_INLINE const uint32_t* Row0Mask() {
    HWY_ALIGN_MAX static constexpr uint32_t kMask[kNumPredictors] = {
        kMaxError, kMaxError, kMaxError, kMaxError,
        kMaxError, 0,         kMaxError, kMaxError};
    return kMask;
  }

  // All predictors that use l and w are invalid in col 0.
  static JXL_INLINE const uint32_t* Col0Mask() {
    HWY_ALIGN_MAX static constexpr uint32_t kMask[kNumPredictors] = {
        kMaxError, kMaxError, 0, kMaxError, kMaxError, kMaxError, kMaxError, 0};
    return kMask;
  }

  // All predictors that use r are invalid in the last column.
  static JXL_INLINE const uint32_t* LastColMask() {
    HWY_ALIGN_MAX static constexpr uint32_t kMask[kNumPredictors] = {
        kMaxError, 0, kMaxError, 0, 0, 0, 0, 0};
    return kMask;
  }
};

struct XBPredictor {
  static void Predict(const int32_t n, const int32_t w, const int32_t l,
                      const int32_t r, int32_t* JXL_RESTRICT pred) {
    pred[0] = ClampedGradient(n, w, l);
    pred[1] = Average(n, w);
    pred[2] = n;
    pred[3] = Average(n, r);
    pred[4] = w;
    pred[5] = Average(w, l);
    pred[6] = r;
    pred[7] = Average(Average(w, r), n);
  }

  // All predictors but w are invalid in row 0.
  static JXL_INLINE const uint32_t* Row0Mask() {
    HWY_ALIGN_MAX static constexpr uint32_t kMask[kNumPredictors] = {
        kMaxError, kMaxError, kMaxError, kMaxError,
        0,         kMaxError, kMaxError, kMaxError};
    return kMask;
  }

  // All predictors that use l and w are invalid in col 0.
  static JXL_INLINE const uint32_t* Col0Mask() {
    HWY_ALIGN_MAX static constexpr uint32_t kMask[kNumPredictors] = {
        kMaxError, kMaxError, 0, 0, kMaxError, kMaxError, 0, kMaxError};
    return kMask;
  }

  // All predictors that use r are invalid in the last column.
  static JXL_INLINE const uint32_t* LastColMask() {
    HWY_ALIGN_MAX static constexpr uint32_t kMask[kNumPredictors] = {
        0, 0, 0, kMaxError, 0, 0, kMaxError, kMaxError};
    return kMask;
  }
};

static constexpr size_t kMaxLine =
    kDcGroupDimInBlocks < kGroupDim ? kGroupDim : kDcGroupDimInBlocks;

enum class RowType {
  kFirstRow,
  kSecondRow,
  kRegularRow,
};

enum ColumnType : uint8_t {
  kRegularColumn = 0x0,
  kFirstColumn = 0x1,
  kSecondColumn = 0x2,
  kLastColumn = 0x4,
};

enum class PixelType { kBorderPixel, kInteriorPixel };

namespace error {
struct Or {
  static const char* Name() { return "Or"; }
  static void Apply(uint32_t* error, const uint32_t* mask) {
    for (size_t i = 0; i < kNumPredictors; i++) error[i] |= mask[i];
  }
};
struct And {
  static const char* Name() { return "And"; }
  static void Apply(uint32_t* error, const uint32_t* mask) {
    for (size_t i = 0; i < kNumPredictors; i++) error[i] &= mask[i];
  }
};
struct AndNot {
  static const char* Name() { return "AndNot"; }
  static void Apply(uint32_t* error, const uint32_t* mask) {
    for (size_t i = 0; i < kNumPredictors; i++) error[i] &= ~mask[i];
  }
};
}  // namespace error

}  // namespace jxl

#endif  // JXL_PREDICTOR_SHARED_H_
