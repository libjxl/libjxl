// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#if defined(LIB_JXL_CONVOLVE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef LIB_JXL_CONVOLVE_INL_H_
#undef LIB_JXL_CONVOLVE_INL_H_
#else
#define LIB_JXL_CONVOLVE_INL_H_
#endif

#include <hwy/highway.h>

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Broadcast;
#if HWY_TARGET != HWY_SCALAR
using hwy::HWY_NAMESPACE::CombineShiftRightBytes;
#endif
using hwy::HWY_NAMESPACE::TableLookupLanes;
using hwy::HWY_NAMESPACE::Vec;

// Synthesizes left/right neighbors from a vector of center pixels.
class Neighbors {
 public:
  using D = HWY_CAPPED(float, 16);
  using V = Vec<D>;

  // Returns l[i] == c[Mirror(i - 1)].
  HWY_INLINE HWY_MAYBE_UNUSED static V FirstL1(const V c) {
#if HWY_CAP_GE256
    const D d;
    HWY_ALIGN constexpr int32_t lanes[16] = {0, 0, 1, 2,  3,  4,  5,  6,
                                             7, 8, 9, 10, 11, 12, 13, 14};
    const auto indices = SetTableIndices(d, lanes);
    // c = PONM'LKJI
    return TableLookupLanes(c, indices);  // ONML'KJII
#elif HWY_TARGET == HWY_SCALAR
    return c;  // Same (the first mirrored value is the last valid one)
#else  // 128 bit
    // c = LKJI
#if HWY_TARGET <= (1 << HWY_HIGHEST_TARGET_BIT_X86)
    return V{_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(2, 1, 0, 0))};  // KJII
#else
    const D d;
    // TODO(deymo): Figure out if this can be optimized using a single vsri
    // instruction to convert LKJI to KJII.
    HWY_ALIGN constexpr int lanes[4] = {0, 0, 1, 2};  // KJII
    const auto indices = SetTableIndices(d, lanes);
    return TableLookupLanes(c, indices);
#endif
#endif
  }

  // Returns l[i] == c[Mirror(i - 2)].
  HWY_INLINE HWY_MAYBE_UNUSED static V FirstL2(const V c) {
#if HWY_CAP_GE256
    const D d;
    HWY_ALIGN constexpr int32_t lanes[16] = {1, 0, 0, 1, 2,  3,  4,  5,
                                             6, 7, 8, 9, 10, 11, 12, 13};
    const auto indices = SetTableIndices(d, lanes);
    // c = PONM'LKJI
    return TableLookupLanes(c, indices);  // NMLK'JIIJ
#elif HWY_TARGET == HWY_SCALAR
    const D d;
    JXL_DEBUG_ABORT("Unsupported");
    return Zero(d);
#else  // 128 bit
    // c = LKJI
#if HWY_TARGET <= (1 << HWY_HIGHEST_TARGET_BIT_X86)
    return V{_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(1, 0, 0, 1))};  // JIIJ
#else
    const D d;
    HWY_ALIGN constexpr int lanes[4] = {1, 0, 0, 1};  // JIIJ
    const auto indices = SetTableIndices(d, lanes);
    return TableLookupLanes(c, indices);
#endif
#endif
  }
};

// Single entry point for convolution.
// "Strategy" (Direct*/Separable*) decides kernel size and how to evaluate it.
template <class Strategy>
class ConvolveT {
  static constexpr int64_t kRadius = Strategy::kRadius;
  using Simd = HWY_CAPPED(float, 16);

 public:
  static size_t MinWidth() {
#if HWY_TARGET == HWY_SCALAR
    // First/Last use mirrored loads of up to +/- kRadius.
    return 2 * kRadius;
#else
    return Lanes(Simd()) + kRadius;
#endif
  }

  template <class Weights>
  static void Run(const ImageF& in, const Rect& rect, const Weights& weights,
                  ThreadPool* pool, ImageF* out) {
    JXL_DASSERT(SameSize(rect, *out));
    JXL_DASSERT(rect.xsize() >= MinWidth());

    static_assert(static_cast<int64_t>(kRadius) <= 3,
                  "Must handle [0, kRadius) and >= kRadius");
    switch (rect.xsize() % Lanes(Simd())) {
      case 0:
        return RunRows<0>(in, rect, weights, pool, out);
      case 1:
        return RunRows<1>(in, rect, weights, pool, out);
      case 2:
        return RunRows<2>(in, rect, weights, pool, out);
      default:
        return RunRows<3>(in, rect, weights, pool, out);
    }
  }

 private:
  template <size_t kSizeModN, class WrapRow, class Weights>
  static JXL_INLINE void RunRow(const float* JXL_RESTRICT in,
                                const size_t xsize, const int64_t stride,
                                const WrapRow& wrap_row, const Weights& weights,
                                float* JXL_RESTRICT out) {
    Strategy::template ConvolveRow<kSizeModN>(in, xsize, stride, wrap_row,
                                              weights, out);
  }

  template <size_t kSizeModN, class Weights>
  static JXL_INLINE void RunBorderRows(const ImageF& in, const Rect& rect,
                                       const int64_t ybegin, const int64_t yend,
                                       const Weights& weights, ImageF* out) {
    const int64_t stride = in.PixelsPerRow();
    const WrapRowMirror wrap_row(in, rect.ysize());
    for (int64_t y = ybegin; y < yend; ++y) {
      RunRow<kSizeModN>(rect.ConstRow(in, y), rect.xsize(), stride, wrap_row,
                        weights, out->Row(y));
    }
  }

  template <size_t kSizeModN, class Weights>
  static JXL_INLINE void RunInteriorRows(const ImageF& in, const Rect& rect,
                                         const int64_t ybegin,
                                         const int64_t yend,
                                         const Weights& weights,
                                         ThreadPool* pool, ImageF* out) {
    const int64_t stride = in.PixelsPerRow();
    const auto process_row = [&](const uint32_t y, size_t /*thread*/) HWY_ATTR {
      RunRow<kSizeModN>(rect.ConstRow(in, y), rect.xsize(), stride,
                        WrapRowUnchanged(), weights, out->Row(y));
      return true;
    };
    Status status = RunOnPool(pool, ybegin, yend, ThreadPool::NoInit,
                              process_row, "Convolve");
    (void)status;
    JXL_DASSERT(status);
  }

  template <size_t kSizeModN, class Weights>
  static JXL_INLINE void RunRows(const ImageF& in, const Rect& rect,
                                 const Weights& weights, ThreadPool* pool,
                                 ImageF* out) {
    constexpr int64_t r = static_cast<int64_t>(kRadius);
    const int64_t ysize = rect.ysize();
    if (ysize <= 2 * r) {
      RunBorderRows<kSizeModN>(in, rect, 0, ysize, weights, out);
    } else {
      RunBorderRows<kSizeModN>(in, rect, 0, r, weights, out);
      RunInteriorRows<kSizeModN>(in, rect, r, ysize - r, weights, pool, out);
      RunBorderRows<kSizeModN>(in, rect, ysize - r, ysize, weights, out);
    }
  }
};

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // LIB_JXL_CONVOLVE_INL_H_
