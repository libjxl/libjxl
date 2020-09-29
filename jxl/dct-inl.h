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

// Fast SIMD floating-point (I)DCT, any power of two.

#if defined(JXL_DCT_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef JXL_DCT_INL_H_
#undef JXL_DCT_INL_H_
#else
#define JXL_DCT_INL_H_
#endif

#include <stddef.h>

#include <hwy/highway.h>

#include "jxl/dct_block-inl.h"
#include "jxl/dct_scales.h"
#include "jxl/transpose-inl.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

template <size_t SZ>
using FV = HWY_CAPPED(float, SZ);

// Implementation of Lowest Complexity Self Recursive Radix-2 DCT II/III
// Algorithms, by Siriani M. Perera and Jianhua Liu.

template <size_t N, size_t SZ>
struct CoeffBundle {
  static void AddReverse(const float* JXL_RESTRICT ain1,
                         const float* JXL_RESTRICT ain2,
                         float* JXL_RESTRICT aout) {
    for (size_t i = 0; i < N; i++) {
      auto in1 = Load(FV<SZ>(), ain1 + i * SZ);
      auto in2 = Load(FV<SZ>(), ain2 + (N - i - 1) * SZ);
      Store(in1 + in2, FV<SZ>(), aout + i * SZ);
    }
  }
  static void SubReverse(const float* JXL_RESTRICT ain1,
                         const float* JXL_RESTRICT ain2,
                         float* JXL_RESTRICT aout) {
    for (size_t i = 0; i < N; i++) {
      auto in1 = Load(FV<SZ>(), ain1 + i * SZ);
      auto in2 = Load(FV<SZ>(), ain2 + (N - i - 1) * SZ);
      Store(in1 - in2, FV<SZ>(), aout + i * SZ);
    }
  }
  static void B(float* JXL_RESTRICT coeff) {
    auto sqrt2 = Set(FV<SZ>(), square_root<2>::value);
    auto in1 = Load(FV<SZ>(), coeff);
    auto in2 = Load(FV<SZ>(), coeff + SZ);
    Store(MulAdd(in1, sqrt2, in2), FV<SZ>(), coeff);
    for (size_t i = 1; i + 1 < N; i++) {
      auto in1 = Load(FV<SZ>(), coeff + i * SZ);
      auto in2 = Load(FV<SZ>(), coeff + (i + 1) * SZ);
      Store(in1 + in2, FV<SZ>(), coeff + i * SZ);
    }
  }
  static void BTranspose(float* JXL_RESTRICT coeff) {
    for (size_t i = N - 1; i > 0; i--) {
      auto in1 = Load(FV<SZ>(), coeff + i * SZ);
      auto in2 = Load(FV<SZ>(), coeff + (i - 1) * SZ);
      Store(in1 + in2, FV<SZ>(), coeff + i * SZ);
    }
    auto sqrt2 = Set(FV<SZ>(), square_root<2>::value);
    auto in1 = Load(FV<SZ>(), coeff);
    Store(in1 * sqrt2, FV<SZ>(), coeff);
  }
  // Ideally optimized away by compiler.
  static void InverseEvenOdd(const float* JXL_RESTRICT ain,
                             float* JXL_RESTRICT aout) {
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = Load(FV<SZ>(), ain + i * SZ);
      Store(in1, FV<SZ>(), aout + 2 * i * SZ);
    }
    for (size_t i = N / 2; i < N; i++) {
      auto in1 = Load(FV<SZ>(), ain + i * SZ);
      Store(in1, FV<SZ>(), aout + (2 * (i - N / 2) + 1) * SZ);
    }
  }
  // Ideally optimized away by compiler.
  static void ForwardEvenOdd(const float* JXL_RESTRICT ain,
                             float* JXL_RESTRICT aout) {
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = Load(FV<SZ>(), ain + 2 * i * SZ);
      Store(in1, FV<SZ>(), aout + i * SZ);
    }
    for (size_t i = N / 2; i < N; i++) {
      auto in1 = Load(FV<SZ>(), ain + (2 * (i - N / 2) + 1) * SZ);
      Store(in1, FV<SZ>(), aout + i * SZ);
    }
  }
  // Invoked on full vector.
  static void Multiply(float* JXL_RESTRICT coeff) {
    for (size_t i = 0; i < N / 2; i++) {
      auto in1 = Load(FV<SZ>(), coeff + (N / 2 + i) * SZ);
      auto mul = Set(FV<SZ>(), WcMultipliers<N>::kMultipliers[i]);
      Store(in1 * mul, FV<SZ>(), coeff + (N / 2 + i) * SZ);
    }
  }
  static void MultiplyAndAdd(const float* JXL_RESTRICT coeff,
                             float* JXL_RESTRICT out) {
    for (size_t i = 0; i < N / 2; i++) {
      auto mul = Set(FV<SZ>(), WcMultipliers<N>::kMultipliers[i]);
      auto in1 = Load(FV<SZ>(), coeff + i * SZ);
      auto in2 = Load(FV<SZ>(), coeff + (N / 2 + i) * SZ);
      auto out1 = MulAdd(mul, in2, in1);
      auto out2 = NegMulAdd(mul, in2, in1);
      Store(out1, FV<SZ>(), out + i * SZ);
      Store(out2, FV<SZ>(), out + (N - i - 1) * SZ);
    }
  }
  template <typename Block>
  static void LoadFromBlock(const Block& in, size_t off,
                            float* JXL_RESTRICT coeff) {
    for (size_t i = 0; i < N; i++) {
      Store(in.LoadPart(FV<SZ>(), i, off), FV<SZ>(), coeff + i * SZ);
    }
  }
  template <typename Block>
  static void StoreToBlock(const float* JXL_RESTRICT coeff, const Block& out,
                           size_t off) {
    for (size_t i = 0; i < N; i++) {
      out.StorePart(FV<SZ>(), Load(FV<SZ>(), coeff + i * SZ), i, off);
    }
  }
};

template <size_t N, size_t SZ>
struct DCT1DImpl;

template <size_t SZ>
struct DCT1DImpl<1, SZ> {
  JXL_INLINE void operator()(float* JXL_RESTRICT mem) {}
};

template <size_t SZ>
struct DCT1DImpl<2, SZ> {
  JXL_INLINE void operator()(float* JXL_RESTRICT mem) {
    auto in1 = Load(FV<SZ>(), mem);
    auto in2 = Load(FV<SZ>(), mem + SZ);
    Store(in1 + in2, FV<SZ>(), mem);
    Store(in1 - in2, FV<SZ>(), mem + SZ);
  }
};

template <size_t N, size_t SZ>
struct DCT1DImpl {
  void operator()(float* JXL_RESTRICT mem) {
    // This is relatively small (4kB with 64-DCT and AVX-512)
    HWY_ALIGN float tmp[N * SZ];
    CoeffBundle<N / 2, SZ>::AddReverse(mem, mem + N / 2 * SZ, tmp);
    DCT1DImpl<N / 2, SZ>()(tmp);
    CoeffBundle<N / 2, SZ>::SubReverse(mem, mem + N / 2 * SZ, tmp + N / 2 * SZ);
    CoeffBundle<N, SZ>::Multiply(tmp);
    DCT1DImpl<N / 2, SZ>()(tmp + N / 2 * SZ);
    CoeffBundle<N / 2, SZ>::B(tmp + N / 2 * SZ);
    CoeffBundle<N, SZ>::InverseEvenOdd(tmp, mem);
  }
};

template <size_t N, size_t SZ>
struct IDCT1DImpl;

template <size_t SZ>
struct IDCT1DImpl<1, SZ> {
  JXL_INLINE void operator()(float* JXL_RESTRICT mem) {}
};

template <size_t SZ>
struct IDCT1DImpl<2, SZ> {
  JXL_INLINE void operator()(float* JXL_RESTRICT mem) {
    auto in1 = Load(FV<SZ>(), mem);
    auto in2 = Load(FV<SZ>(), mem + SZ);
    Store(in1 + in2, FV<SZ>(), mem);
    Store(in1 - in2, FV<SZ>(), mem + SZ);
  }
};

template <size_t N, size_t SZ>
struct IDCT1DImpl {
  void operator()(float* JXL_RESTRICT mem) {
    // This is relatively small (4kB with 64-DCT and AVX-512)
    HWY_ALIGN float tmp[N * SZ];
    CoeffBundle<N, SZ>::ForwardEvenOdd(mem, tmp);
    IDCT1DImpl<N / 2, SZ>()(tmp);
    CoeffBundle<N / 2, SZ>::BTranspose(tmp + N / 2 * SZ);
    IDCT1DImpl<N / 2, SZ>()(tmp + N / 2 * SZ);
    CoeffBundle<N, SZ>::MultiplyAndAdd(tmp, mem);
  }
};

template <size_t N, size_t M, typename FromBlock, typename ToBlock>
void DCT1D(const FromBlock& from, const ToBlock& to) {
  constexpr size_t SZ = MaxLanes(FV<M>());
  HWY_ALIGN float tmp[N * SZ];
  for (size_t i = 0; i < M; i += Lanes(FV<M>())) {
    CoeffBundle<N, SZ>::LoadFromBlock(from, i, tmp);
    DCT1DImpl<N, SZ>()(tmp);
    CoeffBundle<N, SZ>::StoreToBlock(tmp, to, i);
  }
}

template <size_t N, size_t M, typename FromBlock, typename ToBlock>
void IDCT1D(const FromBlock& from, const ToBlock& to) {
  constexpr size_t SZ = MaxLanes(FV<M>());
  HWY_ALIGN float tmp[N * SZ];
  for (size_t i = 0; i < M; i += Lanes(FV<M>())) {
    CoeffBundle<N, SZ>::LoadFromBlock(from, i, tmp);
    IDCT1DImpl<N, SZ>()(tmp);
    CoeffBundle<N, SZ>::StoreToBlock(tmp, to, i);
  }
}
// Computes the in-place NxN transposed-scaled-DCT (tsDCT) of block.
// Requires that block is HWY_ALIGN'ed.
//
// Final DCT coefficients could be obtained the following way:
//   unscaled(f)[x, y] = f[x, y] * DCTScales<N>[x] * DCTScales<N>[y]
//   untransposed(f)[x, y] = f[y, x]
//   DCT(input) = unscaled(untransposed(tsDCT(input)))
//
// NB: DCT denotes scaled variant of DCT-II, which is orthonormal.
//
// See also DCTSlow, ComputeDCT
template <size_t N>
struct ComputeTransposedScaledDCT {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    // TODO(user): it is possible to avoid using temporary array,
    // after generalizing "To" to be bi-directional; all sub-transforms could
    // be performed "in-place".
    HWY_ALIGN float block[N * N];
    HWY_ALIGN float transposed_block[N * N];
    DCT1D<N, N>(from, ToBlock(N, N, block));
    Transpose<N, N>::Run(FromBlock(N, N, block),
                         ToBlock(N, N, transposed_block));
    DCT1D<N, N>(FromBlock(N, N, transposed_block), to);
  }
};

// Computes the in-place NxN transposed-scaled-iDCT (tsIDCT)of block.
// Requires that block is HWY_ALIGN'ed.
//
// Final DCT coefficients could be obtained the following way:
//   unscaled(f)[x, y] = f[x, y] * IDCTScales<N>[x] * IDCTScales<N>[y]
//   untransposed(f)[x, y] = f[y, x]
//   IDCT(input) = tsIDCT(untransposed(unscaled(input)))
//
// NB: IDCT denotes scaled variant of DCT-III, which is orthonormal.
//
// See also IDCTSlow, ComputeIDCT.

template <size_t N>
struct ComputeTransposedScaledIDCT {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    // TODO(user): it is possible to avoid using temporary array,
    // after generalizing "To" to be bi-directional; all sub-transforms could
    // be performed "in-place".
    HWY_ALIGN float block[N * N];
    HWY_ALIGN float transposed_block[N * N];
    IDCT1D<N, N>(from, ToBlock(N, N, block));
    Transpose<N, N>::Run(FromBlock(N, N, block),
                         ToBlock(N, N, transposed_block));
    IDCT1D<N, N>(FromBlock(N, N, transposed_block), to);
  }
};
// Computes the non-transposed, scaled DCT of a block, that needs to be
// HWY_ALIGN'ed. Used for rectangular blocks.
template <size_t ROWS, size_t COLS>
struct ComputeScaledDCT {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[ROWS * COLS];
    HWY_ALIGN float transposed_block[ROWS * COLS];
    DCT1D<ROWS, COLS>(from, ToBlock(ROWS, COLS, block));
    Transpose<ROWS, COLS>::Run(FromBlock(ROWS, COLS, block),
                               ToBlock(COLS, ROWS, transposed_block));
    // Reusing block to reduce stack usage.
    if (ROWS < COLS) {
      DCT1D<COLS, ROWS>(FromBlock(COLS, ROWS, transposed_block),
                        ToBlock(COLS, ROWS, block));
      Transpose<COLS, ROWS>::Run(FromBlock(COLS, ROWS, block), to);
    } else {
      DCT1D<COLS, ROWS>(FromBlock(COLS, ROWS, transposed_block), to);
    }
  }
};
// Computes the non-transposed, scaled DCT of a block, that needs to be
// HWY_ALIGN'ed. Used for rectangular blocks.
template <size_t ROWS, size_t COLS>
struct ComputeScaledIDCT {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[ROWS * COLS];
    HWY_ALIGN float transposed_block[ROWS * COLS];
    // Reverse the steps done in ComputeScaledDCT.
    if (ROWS < COLS) {
      Transpose<ROWS, COLS>::Run(from, ToBlock(COLS, ROWS, block));
      IDCT1D<COLS, ROWS>(FromBlock(COLS, ROWS, block),
                         ToBlock(COLS, ROWS, transposed_block));
    } else {
      IDCT1D<COLS, ROWS>(from, ToBlock(COLS, ROWS, transposed_block));
    }
    Transpose<COLS, ROWS>::Run(FromBlock(COLS, ROWS, transposed_block),
                               ToBlock(ROWS, COLS, block));
    IDCT1D<ROWS, COLS>(FromBlock(ROWS, COLS, block), to);
  }
};

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();
#endif
