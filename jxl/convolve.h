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

#ifndef JXL_CONVOLVE_H_
#define JXL_CONVOLVE_H_

// Fast SIMD 2D convolution.

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <hwy/arch.h>
#include <hwy/static_targets.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/profiler.h"
#include "jxl/base/status.h"
#include "jxl/image.h"
#include "jxl/image_ops.h"

namespace jxl {

// Usable by any 3x3 kernel; applied as-is without flipping.
struct Weights3x3 {
  // top/middle/bottom left/center/right, replicated 4x via HWY_REP4.
  float tl[4];
  float tc[4];
  float tr[4];
  float ml[4];
  float mc[4];
  float mr[4];
  float bl[4];
  float bc[4];
  float br[4];
};

struct WeightsSeparable5 {
  // Horizontal 1D, distances 0..2, each replicated 4x.
  float horz[3 * 4];
  float vert[3 * 4];
};

struct WeightsSeparable7 {
  // Horizontal 1D, distances 0..3, each replicated 4x.
  float horz[4 * 4];
  float vert[4 * 4];
};

// For code-folding.
namespace kernel {

// Holds weights computed at runtime (e.g. inverse of another kernel).
class Variable3 {
 public:
  explicit Variable3(const float tl, const float tc, const float tr,
                     const float ml, const float mc, const float mr,
                     const float bl, const float bc, const float br) {
    for (size_t i = 0; i < 4; ++i) {
      weights_.tl[i] = tl;
      weights_.tc[i] = tc;
      weights_.tr[i] = tr;
      weights_.ml[i] = ml;
      weights_.mc[i] = mc;
      weights_.mr[i] = mr;
      weights_.bl[i] = bl;
      weights_.bc[i] = bc;
      weights_.br[i] = br;
    }
  }

  JXL_INLINE const Weights3x3& Weights() const { return weights_; }

 private:
  Weights3x3 weights_;
};

// Concentrates energy in low-frequency components (e.g. for antialiasing).
struct Lowpass3 {
  JXL_INLINE const Weights3x3& Weights() const {
    // Computed by research/convolve_weights.py's cubic spline approximations of
    // prolate spheroidal wave functions.
    constexpr float w0 = 0.36208932f;
    constexpr float w1 = 0.12820096f;
    constexpr float w2 = 0.03127668f;
    static constexpr Weights3x3 weights = {
        {HWY_REP4(w2)}, {HWY_REP4(w1)}, {HWY_REP4(w2)},
        {HWY_REP4(w1)}, {HWY_REP4(w0)}, {HWY_REP4(w1)},
        {HWY_REP4(w2)}, {HWY_REP4(w1)}, {HWY_REP4(w2)}};
    return weights;
  }
};

struct Lowpass5 {
  JXL_INLINE const WeightsSeparable5& Weights() const {
    constexpr float w0 = 0.41714928f;
    constexpr float w1 = 0.25539268f;
    constexpr float w2 = 0.03603267f;
    static constexpr WeightsSeparable5 weights = {
        {HWY_REP4(w0), HWY_REP4(w1), HWY_REP4(w2)},
        {HWY_REP4(w0), HWY_REP4(w1), HWY_REP4(w2)}};
    return weights;
  }
};

struct Gaussian5Sigma1 {
  JXL_INLINE const WeightsSeparable5& Weights() const {
    constexpr float w0 = 0.38774f;
    constexpr float w1 = 0.24477f;
    constexpr float w2 = 0.06136f;
    static constexpr WeightsSeparable5 weights = {
        {HWY_REP4(w0), HWY_REP4(w1), HWY_REP4(w2)},
        {HWY_REP4(w0), HWY_REP4(w1), HWY_REP4(w2)}};
    return weights;
  }
};

struct Gaussian5Sigma2 {
  JXL_INLINE const WeightsSeparable5& Weights() const {
    constexpr float w0 = 0.250301f;
    constexpr float w1 = 0.221461f;
    constexpr float w2 = 0.153388f;
    static constexpr WeightsSeparable5 weights = {
        {HWY_REP4(w0), HWY_REP4(w1), HWY_REP4(w2)},
        {HWY_REP4(w0), HWY_REP4(w1), HWY_REP4(w2)}};
    return weights;
  }
};

struct Gaussian7Sigma8 {
  JXL_INLINE const WeightsSeparable7& Weights() const {
    constexpr float w0 = 0.147332f;
    constexpr float w1 = 0.146187f;
    constexpr float w2 = 0.142805f;
    constexpr float w3 = 0.137341f;
    static constexpr WeightsSeparable7 weights = {
        {HWY_REP4(w0), HWY_REP4(w1), HWY_REP4(w2), HWY_REP4(w3)},
        {HWY_REP4(w0), HWY_REP4(w1), HWY_REP4(w2), HWY_REP4(w3)}};
    return weights;
  }
};

}  // namespace kernel

// Non-vectorized implementations for validation.
namespace slow {

// Separable kernels, any radius.
template <int64_t kRadius, class Wrap>
class SeparableConvolution {
 public:
  template <class Kernel>
  static void Run(const ImageF& in, const Rect& rect, const Kernel& kernel,
                  ImageF* out) {
    PROFILER_ZONE("slow::Separable::Run");
    const float* horz_weights = &kernel.Weights().horz[0];
    const float* vert_weights = &kernel.Weights().vert[0];
    for (size_t y = 0; y < rect.ysize(); ++y) {
      float* const JXL_RESTRICT row_out = out->Row(y);
      for (size_t x = 0; x < rect.xsize(); ++x) {
        row_out[x] = ConvolvePixel(in, rect, x, y, horz_weights, vert_weights);
      }
    }
  }

  template <class Kernel>
  static void Run(const Image3F& in, const Rect& rect, const Kernel& kernel,
                  Image3F* out) {
    for (size_t c = 0; c < 3; ++c) {
      Run(in.Plane(c), rect, kernel, const_cast<ImageF*>(&out->Plane(c)));
    }
  }

 private:
  static float ConvolvePixel(const ImageF& in, const Rect& rect,
                             const int64_t x, const int64_t y,
                             const float* JXL_RESTRICT horz_weights,
                             const float* JXL_RESTRICT vert_weights) {
    const size_t xsize = rect.xsize();
    const size_t ysize = rect.ysize();

    float mul = 0.0f;
    for (int dy = -kRadius; dy <= kRadius; ++dy) {
      const float wy = vert_weights[std::abs(dy) * 4];
      const size_t sy = Wrap()(y + dy, ysize);
      JXL_CHECK(sy < ysize);
      const float* const JXL_RESTRICT row = rect.ConstRow(in, sy);
      for (int dx = -kRadius; dx <= kRadius; ++dx) {
        const float wx = horz_weights[std::abs(dx) * 4];
        const size_t sx = Wrap()(x + dx, xsize);
        JXL_CHECK(sx < xsize);
        mul += row[sx] * wx * wy;
      }
    }
    return mul;
  }
};

// Hardcoded 5x5 "Laplacian".
template <int64_t kRadius, class Wrap>
struct Laplacian5 {
  static_assert(kRadius == 2, "Wrong kRadius");

  template <class Kernel>
  static void Run(const ImageF& in, const Rect& rect, const Kernel& kernel,
                  ImageF* out) {
    PROFILER_ZONE("slow::Laplacian5::Run");
    JXL_CHECK(SameSize(rect, *out));

    const size_t xsize = rect.xsize();
    const size_t ysize = rect.ysize();
    for (int64_t y = 0; y < ysize; ++y) {
      const float* const JXL_RESTRICT row_t =
          rect.ConstRow(in, Wrap()(y - 2, ysize));
      const float* const JXL_RESTRICT row_m = rect.ConstRow(in, y);
      const float* const JXL_RESTRICT row_b =
          rect.ConstRow(in, Wrap()(y + 2, ysize));
      float* const JXL_RESTRICT row_out = out->Row(y);

      for (int64_t x = 0; x < xsize; ++x) {
        const int64_t xm2 = Wrap()(x - 2, xsize);
        const int64_t xp2 = Wrap()(x + 2, xsize);
        float r = 0.0f;
        r += /*               */ 1.0f * row_t[x];
        r += 1.0f * row_m[xm2] - 4.0f * row_m[x] + 1.0f * row_m[xp2];
        r += /*               */ 1.0f * row_b[x];
        row_out[x] = r;
      }
    }
  }

  template <class Kernel>
  static void Run(const Image3F& in, const Rect& rect, const Kernel& kernel,
                  Image3F* out) {
    for (size_t c = 0; c < 3; ++c) {
      Run(in.Plane(c), rect, kernel, const_cast<ImageF*>(&out->Plane(c)));
    }
  }
};

template <int64_t kRadius, class Wrap>
struct General3x3Convolution {
  static_assert(kRadius == 1, "Wrong kRadius");

  template <class Kernel>
  static void Run(const ImageF& in, const Rect& rect, const Kernel& kernel,
                  ImageF* out) {
    PROFILER_ZONE("slow::General3::Run");
    JXL_CHECK(SameSize(rect, *out));
    const Weights3x3& weights = kernel.Weights();

    const size_t xsize = rect.xsize();
    const size_t ysize = rect.ysize();
    for (int64_t y = 0; y < ysize; ++y) {
      const float* const JXL_RESTRICT row_t =
          rect.ConstRow(in, Wrap()(y - 1, ysize));
      const float* const JXL_RESTRICT row_m = rect.ConstRow(in, y);
      const float* const JXL_RESTRICT row_b =
          rect.ConstRow(in, Wrap()(y + 1, ysize));
      float* const JXL_RESTRICT row_out = out->Row(y);

      for (int64_t x = 0; x < xsize; ++x) {
        const int64_t xm1 = Wrap()(x - 1, xsize);
        const int64_t xp1 = Wrap()(x + 1, xsize);
        const float tl = row_t[xm1];
        const float ml = row_m[xm1];
        const float bl = row_b[xm1];
        const float tr = row_t[xp1];
        const float mr = row_m[xp1];
        const float br = row_b[xp1];
        float r = 0.0f;
        r += tl * weights.tl[0] + row_t[x] * weights.tc[0] + tr * weights.tr[0];
        r += ml * weights.ml[0] + row_m[x] * weights.mc[0] + mr * weights.mr[0];
        r += bl * weights.bl[0] + row_b[x] * weights.bc[0] + br * weights.br[0];
        row_out[x] = r;
      }
    }
  }

  template <class Kernel>
  static void Run(const Image3F& in, const Rect& rect, const Kernel& kernel,
                  Image3F* out) {
    for (size_t c = 0; c < 3; ++c) {
      Run(in.Plane(c), rect, kernel, const_cast<ImageF*>(&out->Plane(c)));
    }
  }
};

// Slow N*R^2 algorithm in case weights are not separable, but avoids
// bounds-checking overhead for interior pixels. Weights are the lower-right
// quadrant of the kernel and need not be pre-normalized.
template <int64_t kRadius, class Wrap>
class SymmetricConvolution {
 public:
  static void Run(const ImageF& in, const Rect& rect,
                  const float (&weights)[(kRadius + 1) * (kRadius + 1)],
                  ThreadPool* pool, ImageF* JXL_RESTRICT out) {
    PROFILER_ZONE("slow::Symmetric::Run");

    float normalized[(kRadius + 1) * (kRadius + 1)];
    NormalizeWeights(weights, &normalized);
    const size_t ysize = rect.ysize();
    RunOnPool(
        pool, 0, static_cast<int>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const int64_t iy = task;

          if (iy < kRadius || iy >= ysize - kRadius) {
            ConvolveRow<Wrap>(in, rect, iy, normalized, out->Row(iy));
          } else {
            ConvolveRow<WrapUnchanged>(in, rect, iy, normalized, out->Row(iy));
          }
        },
        "Symmetric conv");
  }

  static void Run(const Image3F& in, const Rect& rect,
                  const float (&weights)[(kRadius + 1) * (kRadius + 1)],
                  ThreadPool* pool, Image3F* JXL_RESTRICT out) {
    PROFILER_ZONE("slow::Symmetric::Run3");

    float normalized[(kRadius + 1) * (kRadius + 1)];
    NormalizeWeights(weights, &normalized);
    const size_t ysize = rect.ysize();
    RunOnPool(
        pool, 0, static_cast<int>(ysize), ThreadPool::SkipInit(),
        [&](const int task, int /*thread*/) {
          const size_t iy = task;

          if (iy < kRadius || iy >= ysize - kRadius) {
            for (size_t c = 0; c < 3; ++c) {
              ConvolveRow<Wrap>(in.Plane(c), rect, iy, normalized,
                                out->PlaneRow(c, iy));
            }
          } else {
            for (size_t c = 0; c < 3; ++c) {
              ConvolveRow<WrapUnchanged>(in.Plane(c), rect, iy, normalized,
                                         out->PlaneRow(c, iy));
            }
          }
        },
        "Symmetric conv3");
  }

 private:
  // Expands quadrant into entire kernel and normalizes.
  static void NormalizeWeights(
      const float (&weights)[(kRadius + 1) * (kRadius + 1)],
      float (*normalized)[(kRadius + 1) * (kRadius + 1)]) {
    double sum = 0.0;
    for (int64_t ky = -kRadius; ky <= kRadius; ky++) {
      const int64_t wy = std::abs(ky);
      for (int64_t kx = -kRadius; kx <= kRadius; kx++) {
        const int64_t wx = std::abs(kx);
        sum += weights[wy * (kRadius + 1) + wx];
      }
    }
    const float mul = sum == 0.0f ? 1.0f : 1.0 / sum;
    for (size_t i = 0; i < (kRadius + 1) * (kRadius + 1); ++i) {
      (*normalized)[i] = weights[i] * mul;
    }
  }

  template <class WrapX, class WrapY>
  static float ConvolvePixel(
      const ImageF& in, const Rect& rect, const int64_t ix, const int64_t iy,
      const float (&weights)[(kRadius + 1) * (kRadius + 1)]) {
    float sum = 0.0;

    // ix: image; kx: kernel; wx: weight
    for (int64_t ky = -kRadius; ky <= kRadius; ky++) {
      const int64_t wy = std::abs(ky);
      const int64_t y = WrapY()(iy + ky, rect.ysize());
      const float* JXL_RESTRICT row_in = in.ConstRow(y);

      for (int64_t kx = -kRadius; kx <= kRadius; kx++) {
        const int64_t wx = std::abs(kx);
        const int64_t x = WrapX()(ix + kx, rect.xsize());

        sum += row_in[x] * weights[wy * (kRadius + 1) + wx];
      }
    }
    return sum;
  }

  template <class WrapY>
  static inline void ConvolveRow(
      const ImageF& in, const Rect& rect, const int64_t iy,
      const float (&weights)[(kRadius + 1) * (kRadius + 1)],
      float* JXL_RESTRICT row_out) {
    const size_t xsize = rect.xsize();
    size_t ix = 0;
    for (; ix < kRadius; ix++) {
      row_out[ix] = ConvolvePixel<WrapMirror, WrapY>(in, rect, ix, iy, weights);
    }
    for (; ix < xsize - kRadius; ix++) {
      row_out[ix] =
          ConvolvePixel<WrapUnchanged, WrapY>(in, rect, ix, iy, weights);
    }
    for (; ix < xsize; ix++) {
      row_out[ix] = ConvolvePixel<WrapMirror, WrapY>(in, rect, ix, iy, weights);
    }
  }
};

}  // namespace slow

// Synthesizes left/right neighbors from a vector of center pixels.
class Neighbors {
 public:
  // TODO(janwas): AVX512
#if HWY_BITS >= 256
  using D = HWY_CAPPED(float, 8);
#else
  using D = HWY_CAPPED(float, 4);
#endif
  using V = hwy::VT<D>;

  // Returns l[i] == c[Mirror(i - 1)].
  static HWY_ATTR JXL_INLINE V FirstL1(const V c) {
#if HWY_BITS >= 256
    const D d;
    HWY_ALIGN constexpr int lanes[8] = {0, 0, 1, 2, 3, 4, 5, 6};
    const auto indices = SetTableIndices(d, lanes);
    // c = PONM'LKJI
    return TableLookupLanes(c, indices);  // ONML'KJII
#elif HWY_BITS == 128
    // c = LKJI
#if HWY_ARCH == HWY_ARCH_X86
    return V{_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(2, 1, 0, 0))};  // KJII
#else
    const D d;
    // TODO(deymo): Figure out if this can be optimized using a single vsri
    // instruction to convert LKJI to KJII.
    HWY_ALIGN constexpr int lanes[4] = {0, 0, 1, 2};  // KJII
    const auto indices = SetTableIndices(d, lanes);
    return TableLookupLanes(c, indices);
#endif
#else
    return c;  // Same (the first mirrored value is the last valid one)
#endif
  }

  // Returns l[i] == c[Mirror(i - 2)].
  static HWY_ATTR JXL_INLINE V FirstL2(const V c) {
#if HWY_BITS >= 256
    const D d;
    HWY_ALIGN constexpr int lanes[8] = {1, 0, 0, 1, 2, 3, 4, 5};
    const auto indices = SetTableIndices(d, lanes);
    // c = PONM'LKJI
    return TableLookupLanes(c, indices);  // NMLK'JIIJ
#elif HWY_BITS == 128
    // c = LKJI
#if HWY_ARCH == HWY_ARCH_X86
    return V{_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(1, 0, 0, 1))};  // JIIJ
#else
    const D d;
    HWY_ALIGN constexpr int lanes[4] = {1, 0, 0, 1};  // JIIJ
    const auto indices = SetTableIndices(d, lanes);
    return TableLookupLanes(c, indices);
#endif
#else
    const D d;
    JXL_ASSERT(false);  // unsupported, avoid calling this.
    return Zero(d);
#endif
  }

  // Returns l[i] == c[Mirror(i - 3)].
  static HWY_ATTR JXL_INLINE V FirstL3(const V c) {
#if HWY_BITS >= 256
    const D d;
    HWY_ALIGN constexpr int lanes[8] = {2, 1, 0, 0, 1, 2, 3, 4};
    const auto indices = SetTableIndices(d, lanes);
    // c = PONM'LKJI
    return TableLookupLanes(c, indices);  // MLKJ'IIJK
#elif HWY_BITS == 128
    // c = LKJI
#if HWY_ARCH == HWY_ARCH_X86
    return V{_mm_shuffle_ps(c.raw, c.raw, _MM_SHUFFLE(0, 0, 1, 2))};  // IIJK
#else
    const D d;
    HWY_ALIGN constexpr int lanes[4] = {2, 1, 0, 0};  // IIJK
    const auto indices = SetTableIndices(d, lanes);
    return TableLookupLanes(c, indices);
#endif
#else
    const D d;
    JXL_ASSERT(false);  // unsupported, avoid calling this.
    return Zero(d);
#endif
  }

  // Returns r[i] == c[i + 1].
  static HWY_ATTR JXL_INLINE V R1(const V c, const V n) {
#if HWY_BITS >= 256
    // c = PONM'LKJI, n = xxxx'xxxQ
    const V Q_M = ConcatLoHi(n, c);  // Right-aligned (lower lane)
    return hwy::CombineShiftRightBytes<4>(Q_M, c);  // QPON'MLKJ
#elif HWY_BITS == 128
    // c = LKJI, n = xxxM
    return hwy::CombineShiftRightBytes<4>(n, c);  // MLKJ
#else
    return n;
#endif
  }

  // Returns r[i] == c[i + 1].
  static HWY_ATTR JXL_INLINE V LastR1(const V c) {
#if HWY_BITS >= 256
    HWY_ALIGN constexpr uint32_t lanes[8] = {1, 2, 3, 4, 5, 6, 7, 7};
    const auto indices = Load(HWY_CAPPED(uint32_t, 8)(), lanes);
    // c = PONM'LKJI
    return V{_mm256_permutevar8x32_ps(c.raw, indices.raw)};  // PPON'MLKJ
#elif HWY_BITS == 128
    // c = LKJI
    const auto L = hwy::Broadcast<3>(c);
    return hwy::CombineShiftRightBytes<4>(L, c);  // LLKJ
#else
    return c;
#endif
  }
};

// No valid values outside [0, xsize), but the strategy may still safely load
// the preceding vector, and/or round xsize up to the vector lane count. This
// avoids needing PadImage.
// Requires xsize >= kConvolveLanes + kConvolveMaxRadius.
static constexpr size_t kConvolveMaxRadius = 3;

// For use by SetTableIndices.
static inline const int32_t* MirrorLanes(const size_t mod) {
#if HWY_BITS >= 256
  HWY_CAPPED(float, 8) d;
  // last  part  mirrored
  // 01234567| 76543210   loadedReg 76543210 mirroredReg 01234567
  // 01234567|8 8765432   loadedReg 87654321 mirroredReg 23456788
  // 01234567|89 987654   loadedReg 98765432 mirroredReg 45678998
  // 01234567|89A A9876   loadedReg A9876543 mirroredReg 6789AA98
  // 01234567|89AB BA98
  // 01234567|89ABC CBA
  // 01234567|89ABCD DC
  // 01234567|89ABCDE E   loadedReg EDCBA987 mirroredReg EEDCBA98
  HWY_ALIGN static constexpr int32_t idx_lanes[d.N * d.N] = {
      7, 6, 5, 4, 3, 2, 1, 0,  // 0
      7, 7, 6, 5, 4, 3, 2, 1,  // 1
      6, 7, 7, 6, 5, 4, 3, 2,  // 2
      5, 6, 7, 7, 6, 5, 4, 3,  // 3
      4, 5, 6, 7, 7, 6, 5, 4,  // 4
      3, 4, 5, 6, 7, 7, 6, 5,  // 5
      2, 3, 4, 5, 6, 7, 7, 6,  // 6
      1, 2, 3, 4, 5, 6, 7, 7,  // 7
  };
  return idx_lanes + mod * d.N;
#elif HWY_BITS == 128
  HWY_CAPPED(float, 4) d;
  // 0123| 3210   loadedReg 3210 mirroredReg 0123
  // 0123|4 432   loadedReg 4321 mirroredReg 2344
  // 0123|45 54   loadedReg 5432 mirroredReg 4554
  // 0123|456 6   loadedReg 6543 mirroredReg 6654
  HWY_ALIGN static constexpr int32_t idx_lanes[d.N * d.N] = {
      3, 2, 1, 0,  // 0
      3, 3, 2, 1,  // 1
      2, 3, 3, 2,  // 2
      1, 2, 3, 3,  // 3
  };
  return idx_lanes + mod * d.N;
#else
  return nullptr;  // do not call
#endif
}

namespace strategy {

struct StrategyBase {
#if HWY_BITS >= 256
  using D = HWY_CAPPED(float, 8);
#else
  using D = HWY_CAPPED(float, 4);
#endif
  using V = hwy::VT<D>;
};

// 3x3 convolution by symmetric kernel with a single scan through the input.
class Symmetric3 : public StrategyBase {
 public:
  static constexpr int64_t kRadius = 1;

  // Only accesses pixels in [0, xsize).
  template <size_t kSizeModN, class WrapRow>
  static HWY_ATTR JXL_INLINE void ConvolveRow(
      const float* const JXL_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row, const Weights3x3& weights,
      float* const JXL_RESTRICT row_out) {
    const D d;
    // t, m, b = top, middle, bottom row;
    const float* const JXL_RESTRICT row_t = wrap_row(row_m - stride, stride);
    const float* const JXL_RESTRICT row_b = wrap_row(row_m + stride, stride);

    // Must load in advance - compiler doesn't understand LoadDup128 and
    // schedules them too late.
    const V w0 = LoadDup128(d, weights.mc);
    const V w1 = LoadDup128(d, weights.tc);
    const V w2 = LoadDup128(d, weights.tl);

    // l, c, r = left, center, right. Leftmost vector: need FirstL1.
    {
      const V tc = LoadU(d, row_t + 0);
      const V mc = LoadU(d, row_m + 0);
      const V bc = LoadU(d, row_b + 0);
      const V tl = Neighbors::FirstL1(tc);
      const V tr = LoadU(d, row_t + 0 + 1);
      const V ml = Neighbors::FirstL1(mc);
      const V mr = LoadU(d, row_m + 0 + 1);
      const V bl = Neighbors::FirstL1(bc);
      const V br = LoadU(d, row_b + 0 + 1);
      const V conv =
          WeightedSum(tl, tc, tr, ml, mc, mr, bl, bc, br, w0, w1, w2);
      Store(conv, d, row_out + 0);
    }

    // Loop as long as we can load enough new values:
    size_t x = d.N;
    for (; x + d.N + kRadius <= xsize; x += d.N) {
      const auto conv = ConvolveValid(row_t, row_m, row_b, x, w0, w1, w2);
      Store(conv, d, row_out + x);
    }

    // For final (partial) vector:
    const V tc = LoadU(d, row_t + x);
    const V mc = LoadU(d, row_m + x);
    const V bc = LoadU(d, row_b + x);

    V tr, mr, br;
#if HWY_BITS == 0
    tr = tc;  // Single-lane => mirrored right neighbor = center value.
    mr = mc;
    br = bc;
#else
    if (kSizeModN == 0) {
      // The above loop didn't handle the last vector because it needs an
      // additional right neighbor (generated via mirroring).
      auto mirror = SetTableIndices(d, MirrorLanes(d.N - 1));
      tr = TableLookupLanes(tc, mirror);
      mr = TableLookupLanes(mc, mirror);
      br = TableLookupLanes(bc, mirror);
    } else {
      auto mirror = SetTableIndices(d, MirrorLanes((xsize % d.N) - 1));
      // Loads last valid value into uppermost lane and mirrors.
      tr = TableLookupLanes(LoadU(d, row_t + xsize - d.N), mirror);
      mr = TableLookupLanes(LoadU(d, row_m + xsize - d.N), mirror);
      br = TableLookupLanes(LoadU(d, row_b + xsize - d.N), mirror);
    }
#endif

    const V tl = LoadU(d, row_t + x - 1);
    const V ml = LoadU(d, row_m + x - 1);
    const V bl = LoadU(d, row_b + x - 1);
    const V conv = WeightedSum(tl, tc, tr, ml, mc, mr, bl, bc, br, w0, w1, w2);
    Store(conv, d, row_out + x);
  }

 private:
  // Returns sum{x_i * w_i}.
  template <class V>
  static HWY_ATTR JXL_INLINE V WeightedSum(const V tl, const V tc, const V tr,
                                           const V ml, const V mc, const V mr,
                                           const V bl, const V bc, const V br,
                                           const V w0, const V w1, const V w2) {
    const V sum_tb = tc + bc;

    // Faster than 5 mul + 4 FMA.
    const V mul0 = mc * w0;
    const V sum_lr = ml + mr;

    const V x1 = sum_tb + sum_lr;
    const V mul1 = MulAdd(x1, w1, mul0);

    const V sum_t2 = tl + tr;
    const V sum_b2 = bl + br;
    const V x2 = sum_t2 + sum_b2;
    const V mul2 = MulAdd(x2, w2, mul1);
    return mul2;
  }

  static HWY_ATTR JXL_INLINE V ConvolveValid(const float* JXL_RESTRICT row_t,
                                             const float* JXL_RESTRICT row_m,
                                             const float* JXL_RESTRICT row_b,
                                             const int64_t x, const V w0,
                                             const V w1, const V w2) {
    const D d;
    const V tc = LoadU(d, row_t + x);
    const V mc = LoadU(d, row_m + x);
    const V bc = LoadU(d, row_b + x);
    const V tl = LoadU(d, row_t + x - 1);
    const V tr = LoadU(d, row_t + x + 1);
    const V ml = LoadU(d, row_m + x - 1);
    const V mr = LoadU(d, row_m + x + 1);
    const V bl = LoadU(d, row_b + x - 1);
    const V br = LoadU(d, row_b + x + 1);
    return WeightedSum(tl, tc, tr, ml, mc, mr, bl, bc, br, w0, w1, w2);
  }
};

// 5x5 convolution by separable kernel with a single scan through the input.
class Separable5 : public StrategyBase {
 public:
  static constexpr int64_t kRadius = 2;

  template <size_t kSizeModN, class WrapRow>
  static HWY_ATTR JXL_INLINE void ConvolveRow(
      const float* const JXL_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row,
      const WeightsSeparable5& weights, float* const JXL_RESTRICT row_out) {
    const D d;
    const int64_t neg_stride = -stride;  // allows LEA addressing.
    const float* const JXL_RESTRICT row_t2 =
        wrap_row(row_m + 2 * neg_stride, stride);
    const float* const JXL_RESTRICT row_t1 =
        wrap_row(row_m + 1 * neg_stride, stride);
    const float* const JXL_RESTRICT row_b1 =
        wrap_row(row_m + 1 * stride, stride);
    const float* const JXL_RESTRICT row_b2 =
        wrap_row(row_m + 2 * stride, stride);

    const V wh0 = LoadDup128(d, weights.horz + 0 * 4);
    const V wh1 = LoadDup128(d, weights.horz + 1 * 4);
    const V wh2 = LoadDup128(d, weights.horz + 2 * 4);
    const V wv0 = LoadDup128(d, weights.vert + 0 * 4);
    const V wv1 = LoadDup128(d, weights.vert + 1 * 4);
    const V wv2 = LoadDup128(d, weights.vert + 2 * 4);

    size_t x = 0;

    // Need to loop more than once for scalars (d.N == 1).
    for (; x < kRadius; x += d.N) {
      const V conv0 = HorzConvolveFirst(row_m, x, xsize, wh0, wh1, wh2) * wv0;

      const V conv1t = HorzConvolveFirst(row_t1, x, xsize, wh0, wh1, wh2);
      const V conv1b = HorzConvolveFirst(row_b1, x, xsize, wh0, wh1, wh2);
      const V conv1 = MulAdd(conv1t + conv1b, wv1, conv0);

      const V conv2t = HorzConvolveFirst(row_t2, x, xsize, wh0, wh1, wh2);
      const V conv2b = HorzConvolveFirst(row_b2, x, xsize, wh0, wh1, wh2);
      const V conv2 = MulAdd(conv2t + conv2b, wv2, conv1);
      Store(conv2, d, row_out + x);
    }

    // Main loop: load inputs without padding
    for (; x + d.N + kRadius <= xsize; x += d.N) {
      const V conv0 = HorzConvolve(row_m + x, wh0, wh1, wh2) * wv0;

      const V conv1t = HorzConvolve(row_t1 + x, wh0, wh1, wh2);
      const V conv1b = HorzConvolve(row_b1 + x, wh0, wh1, wh2);
      const V conv1 = MulAdd(conv1t + conv1b, wv1, conv0);

      const V conv2t = HorzConvolve(row_t2 + x, wh0, wh1, wh2);
      const V conv2b = HorzConvolve(row_b2 + x, wh0, wh1, wh2);
      const V conv2 = MulAdd(conv2t + conv2b, wv2, conv1);
      Store(conv2, d, row_out + x);
    }

    // Last full vector to write (the above loop handled mod >= kRadius)
#if HWY_BITS == 0
    while (x < xsize) {
#else
    if (kSizeModN < kRadius) {
#endif
      const V conv0 =
          HorzConvolveLast<kSizeModN>(row_m, x, xsize, wh0, wh1, wh2) * wv0;

      const V conv1t =
          HorzConvolveLast<kSizeModN>(row_t1, x, xsize, wh0, wh1, wh2);
      const V conv1b =
          HorzConvolveLast<kSizeModN>(row_b1, x, xsize, wh0, wh1, wh2);
      const V conv1 = MulAdd(conv1t + conv1b, wv1, conv0);

      const V conv2t =
          HorzConvolveLast<kSizeModN>(row_t2, x, xsize, wh0, wh1, wh2);
      const V conv2b =
          HorzConvolveLast<kSizeModN>(row_b2, x, xsize, wh0, wh1, wh2);
      const V conv2 = MulAdd(conv2t + conv2b, wv2, conv1);
      Store(conv2, d, row_out + x);
      x += d.N;
    }

    // If mod = 0, the above vector was the last.
    if (kSizeModN != 0) {
      for (; x < xsize; ++x) {
        float mul = 0.0f;
        for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
          const float wy = weights.vert[std::abs(dy) * 4];
          const float* clamped_row = wrap_row(row_m + dy * stride, stride);
          for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
            const float wx = weights.horz[std::abs(dx) * 4];
            const int64_t clamped_x = Mirror(x + dx, xsize);
            mul += clamped_row[clamped_x] * wx * wy;
          }
        }
        row_out[x] = mul;
      }
    }
  }

 private:
  // Same as HorzConvolve for the first/last vector in a row.
  static HWY_ATTR JXL_INLINE V HorzConvolveFirst(
      const float* const JXL_RESTRICT row, const int64_t x, const int64_t xsize,
      const V wh0, const V wh1, const V wh2) {
    const D d;
    const V c = LoadU(d, row + x);
    const V mul0 = c * wh0;

#if HWY_BITS == 0
    const V l1 = LoadU(d, row + Mirror(x - 1, xsize));
    const V l2 = LoadU(d, row + Mirror(x - 2, xsize));
#else
    (void)xsize;
    const V l1 = Neighbors::FirstL1(c);
    const V l2 = Neighbors::FirstL2(c);
#endif

    const V r1 = LoadU(d, row + x + 1);
    const V r2 = LoadU(d, row + x + 2);

    const V mul1 = MulAdd(l1 + r1, wh1, mul0);
    const V mul2 = MulAdd(l2 + r2, wh2, mul1);
    return mul2;
  }

  template <size_t kSizeModN>
  static HWY_ATTR JXL_INLINE V
  HorzConvolveLast(const float* const JXL_RESTRICT row, const int64_t x,
                   const int64_t xsize, const V wh0, const V wh1, const V wh2) {
    const D d;
    const V c = LoadU(d, row + x);
    const V mul0 = c * wh0;

    const V l1 = LoadU(d, row + x - 1);
    const V l2 = LoadU(d, row + x - 2);

    V r1, r2;
#if HWY_BITS == 0
    r1 = LoadU(d, row + Mirror(x + 1, xsize));
    r2 = LoadU(d, row + Mirror(x + 2, xsize));
#else
    if (kSizeModN == 0) {
      r2 = TableLookupLanes(c, SetTableIndices(d, MirrorLanes(d.N - 2)));
      r1 = TableLookupLanes(c, SetTableIndices(d, MirrorLanes(d.N - 1)));
    } else {  // == 1
      const auto last = LoadU(d, row + xsize - d.N);
      r2 = TableLookupLanes(last, SetTableIndices(d, MirrorLanes(d.N - 1)));
      r1 = last;
    }
#endif

    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = l1 + r1;
    const V mul1 = MulAdd(sum1, wh1, mul0);
    const V sum2 = l2 + r2;
    const V mul2 = MulAdd(sum2, wh2, mul1);
    return mul2;
  }

  // Requires kRadius valid pixels before/after pos.
  static HWY_ATTR JXL_INLINE V HorzConvolve(const float* const JXL_RESTRICT pos,
                                            const V wh0, const V wh1,
                                            const V wh2) {
    const D d;
    const V c = LoadU(d, pos);
    const V mul0 = c * wh0;

    // Loading anew is faster than combining vectors.
    const V l1 = LoadU(d, pos - 1);
    const V r1 = LoadU(d, pos + 1);
    const V l2 = LoadU(d, pos - 2);
    const V r2 = LoadU(d, pos + 2);
    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = l1 + r1;
    const V mul1 = MulAdd(sum1, wh1, mul0);
    const V sum2 = l2 + r2;
    const V mul2 = MulAdd(sum2, wh2, mul1);
    return mul2;
  }
};  // namespace strategy

// 7x7 convolution by separable kernel with a single scan through the input.
class Separable7 : public StrategyBase {
 public:
  static constexpr int64_t kRadius = 3;

  template <size_t kSizeModN, class WrapRow>
  static HWY_ATTR JXL_INLINE void ConvolveRow(
      const float* const JXL_RESTRICT row_m, const size_t xsize,
      const int64_t stride, const WrapRow& wrap_row,
      const WeightsSeparable7& weights, float* const JXL_RESTRICT row_out) {
    const D d;
    const int64_t neg_stride = -stride;  // allows LEA addressing.
    const float* const JXL_RESTRICT row_t3 =
        wrap_row(row_m + 3 * neg_stride, stride);
    const float* const JXL_RESTRICT row_t2 =
        wrap_row(row_m + 2 * neg_stride, stride);
    const float* const JXL_RESTRICT row_t1 =
        wrap_row(row_m + 1 * neg_stride, stride);
    const float* const JXL_RESTRICT row_b1 =
        wrap_row(row_m + 1 * stride, stride);
    const float* const JXL_RESTRICT row_b2 =
        wrap_row(row_m + 2 * stride, stride);
    const float* const JXL_RESTRICT row_b3 =
        wrap_row(row_m + 3 * stride, stride);

    const V wh0 = LoadDup128(d, weights.horz + 0 * 4);
    const V wh1 = LoadDup128(d, weights.horz + 1 * 4);
    const V wh2 = LoadDup128(d, weights.horz + 2 * 4);
    const V wh3 = LoadDup128(d, weights.horz + 3 * 4);
    const V wv0 = LoadDup128(d, weights.vert + 0 * 4);
    const V wv1 = LoadDup128(d, weights.vert + 1 * 4);
    const V wv2 = LoadDup128(d, weights.vert + 2 * 4);
    const V wv3 = LoadDup128(d, weights.vert + 3 * 4);

    size_t x = 0;

    // Need to loop more than once for scalars (d.N == 1).
    for (; x < kRadius; x += d.N) {
      const V conv0 =
          HorzConvolveFirst(row_m, x, xsize, wh0, wh1, wh2, wh3) * wv0;

      const V conv1t = HorzConvolveFirst(row_t1, x, xsize, wh0, wh1, wh2, wh3);
      const V conv1b = HorzConvolveFirst(row_b1, x, xsize, wh0, wh1, wh2, wh3);
      const V conv1 = MulAdd(conv1t + conv1b, wv1, conv0);

      const V conv2t = HorzConvolveFirst(row_t2, x, xsize, wh0, wh1, wh2, wh3);
      const V conv2b = HorzConvolveFirst(row_b2, x, xsize, wh0, wh1, wh2, wh3);
      const V conv2 = MulAdd(conv2t + conv2b, wv2, conv1);

      const V conv3t = HorzConvolveFirst(row_t3, x, xsize, wh0, wh1, wh2, wh3);
      const V conv3b = HorzConvolveFirst(row_b3, x, xsize, wh0, wh1, wh2, wh3);
      const V conv3 = MulAdd(conv3t + conv3b, wv3, conv2);

      Store(conv3, d, row_out + x);
    }

    // Main loop: load inputs without padding
    for (; x + d.N + kRadius <= xsize; x += d.N) {
      const V conv0 = HorzConvolve(row_m + x, wh0, wh1, wh2, wh3) * wv0;

      const V conv1t = HorzConvolve(row_t1 + x, wh0, wh1, wh2, wh3);
      const V conv1b = HorzConvolve(row_b1 + x, wh0, wh1, wh2, wh3);
      const V conv1 = MulAdd(conv1t + conv1b, wv1, conv0);

      const V conv2t = HorzConvolve(row_t2 + x, wh0, wh1, wh2, wh3);
      const V conv2b = HorzConvolve(row_b2 + x, wh0, wh1, wh2, wh3);
      const V conv2 = MulAdd(conv2t + conv2b, wv2, conv1);

      const V conv3t = HorzConvolve(row_t3 + x, wh0, wh1, wh2, wh3);
      const V conv3b = HorzConvolve(row_b3 + x, wh0, wh1, wh2, wh3);
      const V conv3 = MulAdd(conv3t + conv3b, wv3, conv2);

      Store(conv3, d, row_out + x);
    }

    // Last full vector to write (the above loop handled mod >= kRadius)
#if HWY_BITS == 0
    while (x < xsize) {
#else
    if (kSizeModN < kRadius) {
#endif
      const V conv0 =
          HorzConvolveLast<kSizeModN>(row_m, x, xsize, wh0, wh1, wh2, wh3) *
          wv0;

      const V conv1t =
          HorzConvolveLast<kSizeModN>(row_t1, x, xsize, wh0, wh1, wh2, wh3);
      const V conv1b =
          HorzConvolveLast<kSizeModN>(row_b1, x, xsize, wh0, wh1, wh2, wh3);
      const V conv1 = MulAdd(conv1t + conv1b, wv1, conv0);

      const V conv2t =
          HorzConvolveLast<kSizeModN>(row_t2, x, xsize, wh0, wh1, wh2, wh3);
      const V conv2b =
          HorzConvolveLast<kSizeModN>(row_b2, x, xsize, wh0, wh1, wh2, wh3);
      const V conv2 = MulAdd(conv2t + conv2b, wv2, conv1);

      const V conv3t =
          HorzConvolveLast<kSizeModN>(row_t3, x, xsize, wh0, wh1, wh2, wh3);
      const V conv3b =
          HorzConvolveLast<kSizeModN>(row_b3, x, xsize, wh0, wh1, wh2, wh3);
      const V conv3 = MulAdd(conv3t + conv3b, wv3, conv2);

      Store(conv3, d, row_out + x);
      x += d.N;
    }

    // If mod = 0, the above vector was the last.
    if (kSizeModN != 0) {
      for (; x < xsize; ++x) {
        float mul = 0.0f;
        for (int64_t dy = -kRadius; dy <= kRadius; ++dy) {
          const float wy = weights.vert[std::abs(dy) * 4];
          const float* clamped_row = wrap_row(row_m + dy * stride, stride);
          for (int64_t dx = -kRadius; dx <= kRadius; ++dx) {
            const float wx = weights.horz[std::abs(dx) * 4];
            const int64_t clamped_x = Mirror(x + dx, xsize);
            mul += clamped_row[clamped_x] * wx * wy;
          }
        }
        row_out[x] = mul;
      }
    }
  }

 private:
  // Same as HorzConvolve for the first/last vector in a row.
  static HWY_ATTR JXL_INLINE V HorzConvolveFirst(
      const float* const JXL_RESTRICT row, const int64_t x, const int64_t xsize,
      const V wh0, const V wh1, const V wh2, const V wh3) {
    const D d;
    const V c = LoadU(d, row + x);
    const V mul0 = c * wh0;

#if HWY_BITS == 0
    const V l1 = LoadU(d, row + Mirror(x - 1, xsize));
    const V l2 = LoadU(d, row + Mirror(x - 2, xsize));
    const V l3 = LoadU(d, row + Mirror(x - 3, xsize));
#else
    const V l1 = Neighbors::FirstL1(c);
    const V l2 = Neighbors::FirstL2(c);
    const V l3 = Neighbors::FirstL3(c);
#endif

    const V r1 = LoadU(d, row + Mirror(x + 1, xsize));
    const V r2 = LoadU(d, row + Mirror(x + 2, xsize));
    const V r3 = LoadU(d, row + Mirror(x + 3, xsize));

    const V mul1 = MulAdd(l1 + r1, wh1, mul0);
    const V mul2 = MulAdd(l2 + r2, wh2, mul1);
    const V mul3 = MulAdd(l3 + r3, wh3, mul2);
    return mul3;
  }

  template <size_t kSizeModN>
  static HWY_ATTR JXL_INLINE V HorzConvolveLast(
      const float* const JXL_RESTRICT row, const int64_t x, const int64_t xsize,
      const V wh0, const V wh1, const V wh2, const V wh3) {
    const D d;
    const V c = LoadU(d, row + x);
    const V mul0 = c * wh0;

    JXL_DASSERT(x >= kRadius);
    const V l1 = LoadU(d, row + x - 1);
    const V l2 = LoadU(d, row + x - 2);
    const V l3 = LoadU(d, row + x - 3);

    V r1, r2, r3;
#if HWY_BITS == 0
    r1 = LoadU(d, row + Mirror(x + 1, xsize));
    r2 = LoadU(d, row + Mirror(x + 2, xsize));
    r3 = LoadU(d, row + Mirror(x + 3, xsize));
#else
    switch (kSizeModN) {
      case 0:
        r3 = TableLookupLanes(c, SetTableIndices(d, MirrorLanes(d.N - 3)));
        r2 = TableLookupLanes(c, SetTableIndices(d, MirrorLanes(d.N - 2)));
        r1 = TableLookupLanes(c, SetTableIndices(d, MirrorLanes(d.N - 1)));
        break;
      case 1: {
        const auto last = LoadU(d, row + xsize - d.N);
        r3 = TableLookupLanes(last, SetTableIndices(d, MirrorLanes(d.N - 2)));
        r2 = TableLookupLanes(last, SetTableIndices(d, MirrorLanes(d.N - 1)));
        r1 = last;
        break;
      }
      default:
        JXL_DASSERT(kSizeModN == 2);
        {
          const auto last = LoadU(d, row + xsize - d.N);
          r3 = TableLookupLanes(last, SetTableIndices(d, MirrorLanes(d.N - 1)));
          r2 = last;
          r1 = LoadU(d, row + x + 1);
          break;
        }
    }
#endif

    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = l1 + r1;
    const V mul1 = MulAdd(sum1, wh1, mul0);
    const V sum2 = l2 + r2;
    const V mul2 = MulAdd(sum2, wh2, mul1);
    const V sum3 = l3 + r3;
    const V mul3 = MulAdd(sum3, wh3, mul2);
    return mul3;
  }

  // Requires kRadius valid pixels before/after pos.
  static HWY_ATTR JXL_INLINE V HorzConvolve(const float* const JXL_RESTRICT pos,
                                            const V wh0, const V wh1,
                                            const V wh2, const V wh3) {
    const D d;
    const V c = LoadU(d, pos);
    const V mul0 = c * wh0;

    // Loading anew is faster than combining vectors.
    const V l1 = LoadU(d, pos - 1);
    const V r1 = LoadU(d, pos + 1);
    const V l2 = LoadU(d, pos - 2);
    const V r2 = LoadU(d, pos + 2);
    const V l3 = LoadU(d, pos - 3);
    const V r3 = LoadU(d, pos + 3);
    // Sum of pixels with Manhattan distance i, multiplied by weights[i].
    const V sum1 = l1 + r1;
    const V mul1 = MulAdd(sum1, wh1, mul0);
    const V sum2 = l2 + r2;
    const V mul2 = MulAdd(sum2, wh2, mul1);
    const V sum3 = l3 + r3;
    const V mul3 = MulAdd(sum3, wh3, mul2);
    return mul3;
  }
};  // namespace jxl

}  // namespace strategy

// TODO(janwas): AVX-512
static constexpr size_t kConvolveLanes = HWY_CAPPED(float, 8)::N;
// 3x3 kernels require inputs at least this wide - for the first vector, they
// load right neighbors (N lanes starting from x + 1).
static constexpr size_t kConvolveMinWidth = kConvolveLanes + 1;

// Single entry point for convolution.
// "Strategy" (Direct*/Separable*) decides kernel size and how to evaluate it.
template <class Strategy>
class ConvolveT {
  static constexpr int64_t kRadius = Strategy::kRadius;

 public:
  // "Image" is ImageF or Image3F.
  template <class Image, class Kernel>
  static HWY_ATTR void Run(const Image& in, const Rect& rect,
                           const Kernel& kernel, ThreadPool* pool,
                           const Image* out) {
    PROFILER_ZONE("ConvolveT::Run");
    JXL_CHECK(SameSize(rect, *out));
    JXL_CHECK(rect.xsize() >= kConvolveMinWidth);

    static_assert(int64_t(kRadius) <= 3,
                  "Must handle [0, kRadius) and >= kRadius");
    switch (rect.xsize() % kConvolveLanes) {
      case 0:
        return RunRows<0>(in, rect, kernel, pool, out);
      case 1:
        return RunRows<1>(in, rect, kernel, pool, out);
      case 2:
        return RunRows<2>(in, rect, kernel, pool, out);
      default:
        return RunRows<3>(in, rect, kernel, pool, out);
    }
  }

 private:
  template <size_t kSizeModN, class WrapRow, class Kernel>
  static HWY_ATTR JXL_INLINE void RunRow(const float* JXL_RESTRICT in,
                                         const size_t xsize,
                                         const int64_t stride,
                                         const WrapRow& wrap_row,
                                         const Kernel& kernel,
                                         const float* JXL_RESTRICT out) {
    Strategy::template ConvolveRow<kSizeModN>(
        in, xsize, stride, wrap_row, kernel.Weights(), const_cast<float*>(out));
  }

  template <size_t kSizeModN, class Kernel>
  static HWY_ATTR JXL_INLINE void RunBorderRows(
      const ImageF& in, const Rect& rect, const int64_t ybegin,
      const int64_t yend, const Kernel& kernel, const ImageF* out) {
    const int64_t stride = in.PixelsPerRow();
    const WrapRowMirror wrap_row(in, rect.ysize());
    for (int64_t y = ybegin; y < yend; ++y) {
      RunRow<kSizeModN>(rect.ConstRow(in, y), rect.xsize(), stride, wrap_row,
                        kernel, out->Row(y));
    }
  }

  // Image3F.
  template <size_t kSizeModN, class Kernel>
  static HWY_ATTR JXL_INLINE void RunBorderRows(
      const Image3F& in, const Rect& rect, const int64_t ybegin,
      const int64_t yend, const Kernel& kernel, const Image3F* out) {
    const int64_t stride = in.PixelsPerRow();
    for (int64_t y = ybegin; y < yend; ++y) {
      for (size_t c = 0; c < 3; ++c) {
        const WrapRowMirror wrap_row(in.Plane(c), rect.ysize());
        RunRow<kSizeModN>(rect.ConstPlaneRow(in, c, y), rect.xsize(), stride,
                          wrap_row, kernel, out->PlaneRow(c, y));
      }
    }
  }

  template <size_t kSizeModN, class Kernel>
  static HWY_ATTR JXL_INLINE void RunInteriorRows(
      const ImageF& in, const Rect& rect, const int64_t ybegin,
      const int64_t yend, const Kernel& kernel, ThreadPool* pool,
      const ImageF* out) {
    const int64_t stride = in.PixelsPerRow();
    RunOnPool(
        pool, ybegin, yend, ThreadPool::SkipInit(),
        [&](const int y, int /*thread*/) HWY_ATTR {
          RunRow<kSizeModN>(rect.ConstRow(in, y), rect.xsize(), stride,
                            WrapRowUnchanged(), kernel, out->Row(y));
        },
        "Convolve");
  }

  // Image3F.
  template <size_t kSizeModN, class Kernel>
  static HWY_ATTR JXL_INLINE void RunInteriorRows(
      const Image3F& in, const Rect& rect, const int64_t ybegin,
      const int64_t yend, const Kernel& kernel, ThreadPool* pool,
      const Image3F* out) {
    const int64_t stride = in.PixelsPerRow();
    RunOnPool(
        pool, ybegin, yend, ThreadPool::SkipInit(),
        [&](const int y, int /*thread*/) HWY_ATTR {
          for (size_t c = 0; c < 3; ++c) {
            RunRow<kSizeModN>(rect.ConstPlaneRow(in, c, y), rect.xsize(),
                              stride, WrapRowUnchanged(), kernel,
                              out->PlaneRow(c, y));
          }
        },
        "Convolve3");
  }

  template <size_t kSizeModN, class Image, class Kernel>
  static HWY_ATTR JXL_INLINE void RunRows(const Image& in, const Rect& rect,
                                          const Kernel& kernel,
                                          ThreadPool* pool, const Image* out) {
    const int64_t ysize = rect.ysize();
    RunBorderRows<kSizeModN>(in, rect, 0, std::min(int64_t(kRadius), ysize),
                             kernel, out);
    if (ysize > 2 * int64_t(kRadius)) {
      RunInteriorRows<kSizeModN>(in, rect, int64_t(kRadius),
                                 ysize - int64_t(kRadius), kernel, pool, out);
    }
    if (ysize > int64_t(kRadius)) {
      RunBorderRows<kSizeModN>(in, rect, ysize - int64_t(kRadius), ysize,
                               kernel, out);
    }
  }
};

}  // namespace jxl

#endif  // JXL_CONVOLVE_H_
