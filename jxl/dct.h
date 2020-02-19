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

#ifndef JXL_DCT_H_
#define JXL_DCT_H_

// Fast SIMD floating-point DCT8-32.

#include <string.h>

#include <cmath>
#include <hwy/static_targets.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/status.h"
#include "jxl/block.h"
#include "jxl/dct_simd_4.h"
#include "jxl/dct_simd_8.h"
#include "jxl/dct_simd_any.h"

namespace jxl {

// Final scaling factors of outputs/inputs in the Arai, Agui, and Nakajima
// algorithm computing the DCT/IDCT (described in the book JPEG: Still Image
// Data Compression Standard, section 4.3.5) and the "A low multiplicative
// complexity fast recursive DCT-2 algorithm" (Maxim Vashkevich, Alexander
// Pertrovsky) algorithm. Note that the DCT and the IDCT scales of these two
// algorithms are flipped. We use the first algorithm for DCT8, and the second
// one for all other DCTs.
/* Python snippet to produce these tables for the Arai, Agui, Nakajima
 * algorithm:
 *
from mpmath import *
N = 8
def iscale(u):
  eps = sqrt(mpf(0.5)) if u == 0 else mpf(1.0)
  return sqrt(mpf(2) / mpf(N)) * eps * cos(mpf(u) * pi / mpf(2 * N))
def scale(u):
  return mpf(1) / (mpf(N) * iscale(i))
mp.dps = 18
print(", ".join([str(scale(i)) + 'f' for i in range(N)]))
print(", ".join([str(iscale(i)) + 'f' for i in range(N)]))
 */
static constexpr const float kDCTScales1[1] = {1.0f};
static constexpr const float kIDCTScales1[1] = {1.0f};
static constexpr const float kDCTScales2[2] = {0.707106781186547524f,
                                               0.707106781186547524f};
static constexpr const float kIDCTScales2[2] = {0.707106781186547524f,
                                                0.707106781186547524f};
static constexpr const float kDCTScales4[4] = {0.5f, 0.653281482438188264f,
                                               0.5f, 0.270598050073098492f};
static constexpr const float kIDCTScales4[4] = {0.5f, 0.382683432365089772f,
                                                0.5f, 0.923879532511286756f};
static constexpr const float kDCTScales8[8] = {
    0.353553390593273762f, 0.254897789552079584f, 0.270598050073098492f,
    0.30067244346752264f,  0.353553390593273762f, 0.449988111568207852f,
    0.653281482438188264f, 1.28145772387075309f};

static constexpr const float kIDCTScales8[8] = {
    0.353553390593273762f, 0.490392640201615225f, 0.461939766255643378f,
    0.415734806151272619f, 0.353553390593273762f, 0.277785116509801112f,
    0.191341716182544886f, 0.0975451610080641339f};

static constexpr const float kIDCTScales16[16] = {0.25f,
                                                  0.177632042131274808f,
                                                  0.180239955501736978f,
                                                  0.184731156892216368f,
                                                  0.191341716182544886f,
                                                  0.200444985785954314f,
                                                  0.212607523691814112f,
                                                  0.228686034616512494f,
                                                  0.25f,
                                                  0.278654739432954475f,
                                                  0.318189645143208485f,
                                                  0.375006192208515097f,
                                                  0.461939766255643378f,
                                                  0.608977011699708658f,
                                                  0.906127446352887843f,
                                                  1.80352839005774887f};

static constexpr const float kDCTScales16[16] = {0.25f,
                                                 0.351850934381595615f,
                                                 0.346759961330536865f,
                                                 0.33832950029358817f,
                                                 0.326640741219094132f,
                                                 0.311806253246667808f,
                                                 0.293968900604839679f,
                                                 0.273300466750439372f,
                                                 0.25f,
                                                 0.224291896585659071f,
                                                 0.196423739596775545f,
                                                 0.166663914619436624f,
                                                 0.135299025036549246f,
                                                 0.102631131880589345f,
                                                 0.0689748448207357531f,
                                                 0.0346542922997728657f};

static constexpr const float kIDCTScales32[32] = {
    0.176776695296636881f, 0.125150749558799075f, 0.125604821547038926f,
    0.126367739974385915f, 0.127448894776039792f, 0.128861827480656137f,
    0.13062465373492222f,  0.132760647772446044f, 0.135299025036549246f,
    0.138275974008611132f, 0.141736008704089426f, 0.145733742051533468f,
    0.15033622173376132f,  0.155626030758916204f, 0.161705445839997532f,
    0.168702085363751436f, 0.176776695296636881f, 0.186134067750574612f,
    0.197038655862812556f, 0.20983741135388176f,  0.224994055784103926f,
    0.243142059465490173f, 0.265169421497586868f, 0.292359983358221239f,
    0.326640741219094132f, 0.371041154078541569f, 0.430611774559583482f,
    0.514445252488352888f, 0.640728861935376545f, 0.851902104617179697f,
    1.27528715467229096f,  2.5475020308870142f};

static constexpr const float kDCTScales32[32] = {
    0.176776695296636881f,  0.249698864051293098f,  0.248796181668049222f,
    0.247294127491195243f,  0.245196320100807612f,  0.242507813298635998f,
    0.239235083933052216f,  0.235386016295755195f,  0.230969883127821689f,
    0.225997323280860833f,  0.220480316087088757f,  0.214432152500068017f,
    0.207867403075636309f,  0.200801882870161227f,  0.19325261334068424f,
    0.185237781338739773f,  0.176776695296636881f,  0.1678897387117546f,
    0.158598321040911375f,  0.148924826123108336f,  0.138892558254900556f,
    0.128525686048305432f,  0.117849184206499412f,  0.106888773357570524f,
    0.0956708580912724429f, 0.0842224633480550127f, 0.0725711693136155919f,
    0.0607450449758159725f, 0.048772580504032067f,  0.0366826186138404379f,
    0.0245042850823901505f, 0.0122669185818545036f};

template <size_t N>
constexpr const float* DCTScales() {
  return N == 1 ? kDCTScales1
                : (N == 2 ? kDCTScales2
                          : (N == 4 ? kDCTScales4
                                    : (N == 8 ? kDCTScales8
                                              : (N == 16 ? kDCTScales16
                                                         : kDCTScales32))));
}

template <size_t N>
constexpr const float* IDCTScales() {
  return N == 1 ? kIDCTScales1
                : (N == 2 ? kIDCTScales2
                          : (N == 4 ? kIDCTScales4
                                    : (N == 8 ? kIDCTScales8
                                              : (N == 16 ? kIDCTScales16
                                                         : kIDCTScales32))));
}

// For n != 0, the n-th basis function of a N-DCT, evaluated in pixel k, has a
// value of cos((k+1/2) n/(2N) pi). When downsampling by 2x, we average
// the values for pixel k and k+1 to get the value for pixel (k/2), thus we get
//
// [cos((k+1/2) n/N pi) + cos((k+3/2) n/N pi)]/2 =
// cos(n/(2N) pi) cos((k+1) n/N pi) =
// cos(n/(2N) pi) cos(((k/2)+1/2) n/(N/2) pi)
//
// which is exactly the same as the value of pixel k/2 of a N/2-sized DCT,
// except for the cos(n/(2N) pi) scaling factor (which does *not*
// depend on the pixel). Thus, when using the lower-frequency coefficients of a
// DCT-N to compute a DCT-(N/2), they should be scaled by this constant. Scaling
// factors for a DCT-(N/4) etc can then be obtained by successive
// multiplications. The structs below contain the above-mentioned scaling
// factors.
template <size_t FROM, size_t TO>
struct DCTResampleScales;

template <>
struct DCTResampleScales<8, 1> {
  static constexpr float kScales[1] = {
      1.000000000000000000,
  };
};

template <>
struct DCTResampleScales<8, 2> {
  static constexpr float kScales[2] = {
      1.000000000000000000,
      0.906127446352887778,
  };
};

template <>
struct DCTResampleScales<16, 2> {
  static constexpr float kScales[2] = {
      1.000000000000000000,
      0.901764195028874394,
  };
};

template <>
struct DCTResampleScales<16, 4> {
  static constexpr float kScales[4] = {
      1.000000000000000000,
      0.976062531202202877,
      0.906127446352887778,
      0.795666809947927156,
  };
};

template <>
struct DCTResampleScales<32, 4> {
  static constexpr float kScales[4] = {
      1.000000000000000000,
      0.974886821136879522,
      0.901764195028874394,
      0.787054918159101335,
  };
};

template <>
struct DCTResampleScales<32, 8> {
  static constexpr float kScales[8] = {
      1.000000000000000000, 0.993985983084976765, 0.976062531202202877,
      0.946582901544112176, 0.906127446352887778, 0.855491189274751540,
      0.795666809947927156, 0.727823404688121345,
  };
};

// Inverses of the above.
template <>
struct DCTResampleScales<1, 8> {
  static constexpr float kScales[1] = {
      1.000000000000000000,
  };
};

template <>
struct DCTResampleScales<2, 8> {
  static constexpr float kScales[2] = {
      1.000000000000000000,
      1.103597517131772232,
  };
};

template <>
struct DCTResampleScales<2, 16> {
  static constexpr float kScales[2] = {
      1.000000000000000000,
      1.108937353592731823,
  };
};

template <>
struct DCTResampleScales<4, 16> {
  static constexpr float kScales[4] = {
      1.000000000000000000,
      1.024524523821556565,
      1.103597517131772232,
      1.256807482098500017,
  };
};

template <>
struct DCTResampleScales<4, 32> {
  static constexpr float kScales[4] = {
      1.000000000000000000,
      1.025760096781116015,
      1.108937353592731823,
      1.270559368765487251,
  };
};

template <>
struct DCTResampleScales<8, 32> {
  static constexpr float kScales[8] = {
      1.000000000000000000, 1.006050404147911470, 1.024524523821556565,
      1.056431505754806377, 1.103597517131772232, 1.168919110491081437,
      1.256807482098500017, 1.373959663235216677,
  };
};

// https://en.wikipedia.org/wiki/In-place_matrix_transposition#Square_matrices
template <size_t N, class From, class To>
HWY_ATTR JXL_INLINE void GenericTransposeBlockInplace(const From& from,
                                                      const To& to) {
  // This does not guarantee anything, just saves from the most stupid mistakes.
  JXL_ASSERT(from.Address(0, 0) == to.Address(0, 0));
  for (size_t n = 0; n < N - 1; ++n) {
    for (size_t m = n + 1; m < N; ++m) {
      // Swap
      const float tmp = from.Read(m, n);
      to.Write(from.Read(n, m), m, n);
      to.Write(tmp, n, m);
    }
  }
}

template <size_t ROWS, size_t COLS, class From, class To>
HWY_ATTR JXL_INLINE void GenericTransposeBlock(const From& from, const To& to) {
  // This does not guarantee anything, just saves from the most stupid mistakes.
  JXL_DASSERT(from.Address(0, 0) != to.Address(0, 0));
  for (size_t n = 0; n < ROWS; ++n) {
    for (size_t m = 0; m < COLS; ++m) {
      to.Write(from.Read(n, m), m, n);
    }
  }
}

// TODO(eustas): issue#40 temporary workaround.
template <size_t ROWS, size_t COLS, class From, class To>
HWY_ATTR JXL_NOINLINE void GenericTransposeBlockNoinline(const From& from,
                                                         const To& to) {
  // This does not guarantee anything, just saves from the most stupid mistakes.
  JXL_DASSERT(from.Address(0, 0) != to.Address(0, 0));
  for (size_t n = 0; n < ROWS; ++n) {
    for (size_t m = 0; m < COLS; ++m) {
      to.Write(from.Read(n, m), m, n);
    }
  }
}

template <class From, class To>
HWY_ATTR JXL_INLINE void TransposeBlock8(const From& from, const To& to) {
#if HWY_BITS >= 256
  TransposeBlock8_V8(from, to);
#elif HWY_BITS == 128
  TransposeBlock8_V4(from, to);
#else
  if (from.Address(0, 0) == to.Address(0, 0)) {
    GenericTransposeBlockInplace<8>(from, to);
  } else {
    GenericTransposeBlock<8, 8>(from, to);
  }
#endif
}

template <class From, class To>
HWY_ATTR void TransposeBlock16(const From& from, const To& to) {
  HWY_ALIGN float tmp[8 * 8];
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(0, 8), ToBlock<8>(tmp));
  TransposeBlock8(from.View(8, 0), to.View(0, 8));
  CopyBlock8(FromBlock<8>(tmp), to.View(8, 0));
  TransposeBlock8(from.View(8, 8), to.View(8, 8));
}

template <class From, class To>
HWY_ATTR void TransposeBlock168(const From& from, const To& to) {
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(8, 0), to.View(0, 8));
}

template <class From, class To>
HWY_ATTR void TransposeBlock328(const From& from, const To& to) {
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(8, 0), to.View(0, 8));
  TransposeBlock8(from.View(16, 0), to.View(0, 16));
  TransposeBlock8(from.View(24, 0), to.View(0, 24));
}

template <class From, class To>
HWY_ATTR void TransposeBlock3216(const From& from, const To& to) {
  TransposeBlock16(from, to);
  TransposeBlock16(from.View(16, 0), to.View(0, 16));
}

template <class From, class To>
HWY_ATTR void TransposeBlock816(const From& from, const To& to) {
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(0, 8), to.View(8, 0));
}

template <class From, class To>
HWY_ATTR void TransposeBlock832(const From& from, const To& to) {
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(0, 8), to.View(8, 0));
  TransposeBlock8(from.View(0, 16), to.View(16, 0));
  TransposeBlock8(from.View(0, 24), to.View(24, 0));
}
template <class From, class To>
HWY_ATTR void TransposeBlock1632(const From& from, const To& to) {
  TransposeBlock16(from, to);
  TransposeBlock16(from.View(0, 16), to.View(16, 0));
}

template <class From, class To>
HWY_ATTR void TransposeBlock32(const From& from, const To& to) {
  HWY_ALIGN float tmp[8 * 8];
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(0, 8), ToBlock<8>(tmp));
  TransposeBlock8(from.View(8, 0), to.View(0, 8));
  CopyBlock8(FromBlock<8>(tmp), to.View(8, 0));
  TransposeBlock8(from.View(8, 8), to.View(8, 8));
  TransposeBlock8(from.View(0, 16), ToBlock<8>(tmp));
  TransposeBlock8(from.View(16, 0), to.View(0, 16));
  CopyBlock8(FromBlock<8>(tmp), to.View(16, 0));
  TransposeBlock8(from.View(8, 16), ToBlock<8>(tmp));
  TransposeBlock8(from.View(16, 8), to.View(8, 16));
  CopyBlock8(FromBlock<8>(tmp), to.View(16, 8));
  TransposeBlock8(from.View(16, 16), to.View(16, 16));
  TransposeBlock8(from.View(0, 24), ToBlock<8>(tmp));
  TransposeBlock8(from.View(24, 0), to.View(0, 24));
  CopyBlock8(FromBlock<8>(tmp), to.View(24, 0));
  TransposeBlock8(from.View(8, 24), ToBlock<8>(tmp));
  TransposeBlock8(from.View(24, 8), to.View(8, 24));
  CopyBlock8(FromBlock<8>(tmp), to.View(24, 8));
  TransposeBlock8(from.View(16, 24), ToBlock<8>(tmp));
  TransposeBlock8(from.View(24, 16), to.View(16, 24));
  CopyBlock8(FromBlock<8>(tmp), to.View(24, 16));
  TransposeBlock8(from.View(24, 24), to.View(24, 24));
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
struct ComputeTransposedScaledDCT;

template <>
struct ComputeTransposedScaledDCT<32> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[32 * 32];
    ColumnDCT32(from, ToBlock<32>(block));
    TransposeBlock32(FromBlock<32>(block), ToBlock<32>(block));
    ColumnDCT32(FromBlock<32>(block), to);
  }
};

template <>
struct ComputeTransposedScaledDCT<16> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[16 * 16];
    ColumnDCT16(from, ToBlock<16>(block));
    TransposeBlock16(FromBlock<16>(block), ToBlock<16>(block));
    ColumnDCT16(FromBlock<16>(block), to);
  }
};

template <>
struct ComputeTransposedScaledDCT<8> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
#if HWY_BITS >= 256
    ComputeTransposedScaledDCT8_V8(from, to);
#elif HWY_BITS == 128
    ComputeTransposedScaledDCT8_V4(from, to);
#else
    HWY_ALIGN float block[8 * 8];
    ColumnDCT8(from, ToBlock<8>(block));
    TransposeBlock8(FromBlock<8>(block), ToBlock<8>(block));
    ColumnDCT8(FromBlock<8>(block), to);
#endif
  }
};

template <>
struct ComputeTransposedScaledDCT<4> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 4];
    ColumnDCT4(from, ToBlock<4>(block));
    GenericTransposeBlockInplace<4>(FromBlock<4>(block), ToBlock<4>(block));
    ColumnDCT4(FromBlock<4>(block), to);
  }
};

template <>
struct ComputeTransposedScaledDCT<2> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    const float a00 = from.Read(0, 0);
    const float a01 = from.Read(0, 1);
    const float a10 = from.Read(1, 0);
    const float a11 = from.Read(1, 1);
    to.Write(a00 + a01 + a10 + a11, 0, 0);
    to.Write(a00 + a01 - a10 - a11, 0, 1);
    to.Write(a00 - a01 + a10 - a11, 1, 0);
    to.Write(a00 - a01 - a10 + a11, 1, 1);
  }
};

// Computes the non-transposed, scaled DCT of a block, that needs to be
// HWY_ALIGN'ed. Used for rectangular blocks.
template <size_t ROWS, size_t COLS>
struct ComputeScaledDCT;

template <>
struct ComputeScaledDCT<8, 16> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[8 * 16];
    HWY_ALIGN float transposed_block[8 * 16];
    using FromOriginal = FromBlock<8, 16>;
    using FromTransposed = FromBlock<16, 8>;
    using ToOriginal = ToBlock<8, 16>;
    using ToTransposed = ToBlock<16, 8>;
    ColumnDCT8<From, ToOriginal, /*COLS=*/16>(from, ToOriginal(block));
    TransposeBlock816(FromOriginal(block), ToTransposed(transposed_block));
    // Reusing block to reduce stack usage.
    ColumnDCT16<FromTransposed, ToTransposed, /*COLS=*/8>(
        FromTransposed(transposed_block), ToTransposed(block));
    TransposeBlock168(FromTransposed(block), to);
  }
};

template <>
struct ComputeScaledDCT<8, 32> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[8 * 32];
    HWY_ALIGN float transposed_block[8 * 32];
    using FromOriginal = FromBlock<8, 32>;
    using FromTransposed = FromBlock<32, 8>;
    using ToOriginal = ToBlock<8, 32>;
    using ToTransposed = ToBlock<32, 8>;
    ColumnDCT8<From, ToOriginal, /*COLS=*/32>(from, ToOriginal(block));
    TransposeBlock832(FromOriginal(block), ToTransposed(transposed_block));
    // Reusing block to reduce stack usage.
    ColumnDCT32<FromTransposed, ToTransposed, /*COLS=*/8>(
        FromTransposed(transposed_block), ToTransposed(block));
    TransposeBlock328(FromTransposed(block), to);
  }
};

template <>
struct ComputeScaledDCT<16, 32> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[16 * 32];
    HWY_ALIGN float transposed_block[16 * 32];
    using FromOriginal = FromBlock<16, 32>;
    using FromTransposed = FromBlock<32, 16>;
    using ToOriginal = ToBlock<16, 32>;
    using ToTransposed = ToBlock<32, 16>;
    ColumnDCT16<From, ToOriginal, /*COLS=*/32>(from, ToOriginal(block));
    TransposeBlock1632(FromOriginal(block), ToTransposed(transposed_block));
    // Reusing block to reduce stack usage.
    ColumnDCT32<FromTransposed, ToTransposed, /*COLS=*/16>(
        FromTransposed(transposed_block), ToTransposed(block));
    TransposeBlock3216(FromTransposed(block), to);
  }
};

// Blocks of the form XxY with X > Y are stored transposed.

template <>
struct ComputeScaledDCT<16, 8> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[8 * 16];
    HWY_ALIGN float transposed_block[8 * 16];
    using FromOriginal = FromBlock<16, 8>;
    using FromTransposed = FromBlock<8, 16>;
    using ToOriginal = ToBlock<16, 8>;
    using ToTransposed = ToBlock<8, 16>;
    ColumnDCT16<From, ToOriginal, /*COLS=*/8>(from, ToOriginal(block));
    TransposeBlock168(FromOriginal(block), ToTransposed(transposed_block));
    ColumnDCT8<FromTransposed, To, /*COLS=*/16>(
        FromTransposed(transposed_block), to);
  }
};

template <>
struct ComputeScaledDCT<32, 8> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[8 * 32];
    HWY_ALIGN float transposed_block[8 * 32];
    using FromOriginal = FromBlock<32, 8>;
    using FromTransposed = FromBlock<8, 32>;
    using ToOriginal = ToBlock<32, 8>;
    using ToTransposed = ToBlock<8, 32>;
    ColumnDCT32<From, ToOriginal, /*COLS=*/8>(from, ToOriginal(block));
    TransposeBlock328(FromOriginal(block), ToTransposed(transposed_block));
    ColumnDCT8<FromTransposed, To, /*COLS=*/32>(
        FromTransposed(transposed_block), to);
  }
};

template <>
struct ComputeScaledDCT<32, 16> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[16 * 32];
    HWY_ALIGN float transposed_block[16 * 32];
    using FromOriginal = FromBlock<32, 16>;
    using FromTransposed = FromBlock<16, 32>;
    using ToOriginal = ToBlock<32, 16>;
    using ToTransposed = ToBlock<16, 32>;
    ColumnDCT32<From, ToOriginal, /*COLS=*/16>(from, ToOriginal(block));
    TransposeBlock3216(FromOriginal(block), ToTransposed(transposed_block));
    ColumnDCT16<FromTransposed, To, /*COLS=*/32>(
        FromTransposed(transposed_block), to);
  }
};

template <>
struct ComputeScaledDCT<8, 4> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 8];
    HWY_ALIGN float transposed_block[4 * 8];
    using FromOriginal = FromBlock<8, 4>;
    using FromTransposed = FromBlock<4, 8>;
    using ToOriginal = ToBlock<8, 4>;
    using ToTransposed = ToBlock<4, 8>;
    ColumnDCT8<From, ToOriginal, /*COLS=*/4>(from, ToOriginal(block));
    GenericTransposeBlock<8, 4>(FromOriginal(block),
                                ToTransposed(transposed_block));
    ColumnDCT4<FromTransposed, To, /*COLS=*/8>(FromTransposed(transposed_block),
                                               to);
  }
};

template <>
struct ComputeScaledDCT<4, 8> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[8 * 4];
    HWY_ALIGN float transposed_block[8 * 4];
    using FromOriginal = FromBlock<4, 8>;
    using FromTransposed = FromBlock<8, 4>;
    using ToOriginal = ToBlock<4, 8>;
    using ToTransposed = ToBlock<8, 4>;
    ColumnDCT4<From, ToOriginal, /*COLS=*/8>(from, ToOriginal(block));
#if !defined(__wasm_simd128__)
    GenericTransposeBlock<4, 8>(FromOriginal(block),
                                ToTransposed(transposed_block));
#else
    // TODO(eustas): issue#40 temporary workaround.
    GenericTransposeBlockNoinline<4, 8>(FromOriginal(block),
                                        ToTransposed(transposed_block));
#endif
    // Reusing block to reduce stack usage.
    ColumnDCT8<FromTransposed, ToTransposed, /*COLS=*/4>(
        FromTransposed(transposed_block), ToTransposed(block));
    GenericTransposeBlock<8, 4>(FromTransposed(block), to);
  }
};

template <>
struct ComputeScaledDCT<4, 2> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 2];
    using ToOriginal = ToBlock<4, 2>;
    ColumnDCT4<From, ToOriginal, /*COLS=*/2>(from, ToOriginal(block));
    for (size_t y = 0; y < 4; ++y) {
      const float a0 = block[2 * y];
      const float a1 = block[2 * y + 1];
      to.Write(a0 + a1, 0, y);
      to.Write(a0 - a1, 1, y);
    }
  }
};

template <>
struct ComputeScaledDCT<2, 4> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    // `block` and `coeffs` are transposed.
    HWY_ALIGN float block[4 * 2];
    for (size_t y = 0; y < 4; ++y) {
      const float a0 = from.Read(0, y);
      const float a1 = from.Read(1, y);
      block[2 * y] = a0 + a1;
      block[2 * y + 1] = a0 - a1;
    }
    using FromTransposed = FromBlock<4, 2>;
    using ToTransposed = ToBlock<4, 2>;
    HWY_ALIGN float coeffs[4 * 2];
    ColumnDCT4<FromTransposed, ToTransposed, /*COLS=*/2>(FromTransposed(block),
                                                         ToTransposed(coeffs));
    GenericTransposeBlock<4, 2>(FromTransposed(coeffs), to);
  }
};

template <>
struct ComputeScaledDCT<4, 1> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 1];
    ColumnDCT4<From, ToBlock<4, 1>, /*COLS=*/1>(from, ToBlock<4, 1>(block));
    GenericTransposeBlock<4, 1>(FromBlock<4, 1>(block), to);
  }
};

template <>
struct ComputeScaledDCT<1, 4> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float coeffs[4 * 1];
    using FromTransposed = FromBlock<4, 1>;
    using ToTransposed = ToBlock<4, 1>;
    GenericTransposeBlock<1, 4>(from, ToTransposed(coeffs));
    HWY_ALIGN float block[4 * 1];
    ColumnDCT4<FromTransposed, ToTransposed, /*COLS=*/1>(FromTransposed(coeffs),
                                                         ToTransposed(block));
    GenericTransposeBlock<4, 1>(FromTransposed(block), to);
  }
};

template <>
struct ComputeScaledDCT<2, 1> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    const float a0 = from.Read(0, 0);
    const float a1 = from.Read(1, 0);
    to.Write(a0 + a1, 0, 0);
    to.Write(a0 - a1, 0, 1);
  }
};

template <>
struct ComputeScaledDCT<1, 2> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    const float a0 = from.Read(0, 0);
    const float a1 = from.Read(0, 1);
    to.Write(a0 + a1, 0, 0);
    to.Write(a0 - a1, 0, 1);
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
struct ComputeTransposedScaledIDCT;

template <>
struct ComputeTransposedScaledIDCT<32> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[32 * 32];
    ColumnIDCT32(from, ToBlock<32>(block));
    TransposeBlock32(FromBlock<32>(block), ToBlock<32>(block));
    ColumnIDCT32(FromBlock<32>(block), to);
  }
};

template <>
struct ComputeTransposedScaledIDCT<16> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[16 * 16];
    ColumnIDCT16(from, ToBlock<16>(block));
    TransposeBlock16(FromBlock<16>(block), ToBlock<16>(block));
    ColumnIDCT16(FromBlock<16>(block), to);
  }
};

template <>
struct ComputeTransposedScaledIDCT<8> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
#if HWY_BITS >= 256
    ComputeTransposedScaledIDCT8_V8(from, to);
#elif HWY_BITS == 128
    ComputeTransposedScaledIDCT8_V4(from, to);
#else
    HWY_ALIGN float block[8 * 8];
    ColumnIDCT8(from, ToBlock<8>(block));
    TransposeBlock8(FromBlock<8>(block), ToBlock<8>(block));
    ColumnIDCT8(FromBlock<8>(block), to);
#endif
  }
};

template <>
struct ComputeTransposedScaledIDCT<4> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 4];
    ColumnIDCT4(from, ToBlock<4>(block));
    GenericTransposeBlockInplace<4>(FromBlock<4>(block), ToBlock<4>(block));
    ColumnIDCT4(FromBlock<4>(block), to);
  }
};

template <>
struct ComputeTransposedScaledIDCT<2> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    const float a00 = from.Read(0, 0);
    const float a01 = from.Read(0, 1);
    const float a10 = from.Read(1, 0);
    const float a11 = from.Read(1, 1);
    to.Write(a00 + a01 + a10 + a11, 0, 0);
    to.Write(a00 + a01 - a10 - a11, 0, 1);
    to.Write(a00 - a01 + a10 - a11, 1, 0);
    to.Write(a00 - a01 - a10 + a11, 1, 1);
  }
};

template <>
struct ComputeTransposedScaledIDCT<1> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    to.Write(from.Read(0, 0), 0, 0);
  }
};

// Computes the non-transposed, scaled DCT of a block, that needs to be
// HWY_ALIGN'ed. Used for rectangular blocks.
template <size_t ROWS, size_t COLS>
struct ComputeScaledIDCT;

template <>
struct ComputeScaledIDCT<8, 16> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[8 * 16];
    HWY_ALIGN float transposed_block[8 * 16];
    using FromOriginal = FromBlock<8, 16>;
    using FromTransposed = FromBlock<16, 8>;
    using ToOriginal = ToBlock<8, 16>;
    using ToTransposed = ToBlock<16, 8>;
    // Reverse the steps done in ComputeScaledDCT.
    TransposeBlock816(from, ToTransposed(block));
    ColumnIDCT16<FromTransposed, ToTransposed, /*COLS=*/8>(
        FromTransposed(block), ToTransposed(transposed_block));
    TransposeBlock168(FromTransposed(transposed_block), ToOriginal(block));
    ColumnIDCT8<FromOriginal, To, /*COLS=*/16>(FromOriginal(block), to);
  }
};

template <>
struct ComputeScaledIDCT<8, 32> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[8 * 32];
    HWY_ALIGN float transposed_block[8 * 32];
    using FromOriginal = FromBlock<8, 32>;
    using FromTransposed = FromBlock<32, 8>;
    using ToOriginal = ToBlock<8, 32>;
    using ToTransposed = ToBlock<32, 8>;
    // Reverse the steps done in ComputeScaledDCT.
    TransposeBlock832(from, ToTransposed(block));
    ColumnIDCT32<FromTransposed, ToTransposed, /*COLS=*/8>(
        FromTransposed(block), ToTransposed(transposed_block));
    TransposeBlock328(FromTransposed(transposed_block), ToOriginal(block));
    ColumnIDCT8<FromOriginal, To, /*COLS=*/32>(FromOriginal(block), to);
  }
};

template <>
struct ComputeScaledIDCT<16, 32> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[16 * 32];
    HWY_ALIGN float transposed_block[16 * 32];
    using FromOriginal = FromBlock<16, 32>;
    using FromTransposed = FromBlock<32, 16>;
    using ToOriginal = ToBlock<16, 32>;
    using ToTransposed = ToBlock<32, 16>;
    // Reverse the steps done in ComputeScaledDCT.
    TransposeBlock1632(from, ToTransposed(block));
    ColumnIDCT32<FromTransposed, ToTransposed, /*COLS=*/16>(
        FromTransposed(block), ToTransposed(transposed_block));
    TransposeBlock3216(FromTransposed(transposed_block), ToOriginal(block));
    ColumnIDCT16<FromOriginal, To, /*COLS=*/32>(FromOriginal(block), to);
  }
};

template <>
struct ComputeScaledIDCT<16, 8> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[8 * 16];
    HWY_ALIGN float transposed_block[8 * 16];
    using FromOriginal = FromBlock<16, 8>;
    using FromTransposed = FromBlock<8, 16>;
    using ToOriginal = ToBlock<16, 8>;
    using ToTransposed = ToBlock<8, 16>;
    ColumnIDCT8<From, ToTransposed, /*COLS=*/16>(
        from, ToTransposed(transposed_block));
    TransposeBlock816(FromTransposed(transposed_block), ToOriginal(block));
    ColumnIDCT16<FromOriginal, To, /*COLS=*/8>(FromOriginal(block), to);
  }
};

template <>
struct ComputeScaledIDCT<32, 8> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[8 * 32];
    HWY_ALIGN float transposed_block[8 * 32];
    using FromOriginal = FromBlock<32, 8>;
    using FromTransposed = FromBlock<8, 32>;
    using ToOriginal = ToBlock<32, 8>;
    using ToTransposed = ToBlock<8, 32>;
    ColumnIDCT8<From, ToTransposed, /*COLS=*/32>(
        from, ToTransposed(transposed_block));
    TransposeBlock832(FromTransposed(transposed_block), ToOriginal(block));
    ColumnIDCT32<FromOriginal, To, /*COLS=*/8>(FromOriginal(block), to);
  }
};

template <>
struct ComputeScaledIDCT<32, 16> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[16 * 32];
    HWY_ALIGN float transposed_block[16 * 32];
    using FromOriginal = FromBlock<32, 16>;
    using FromTransposed = FromBlock<16, 32>;
    using ToOriginal = ToBlock<32, 16>;
    using ToTransposed = ToBlock<16, 32>;
    ColumnIDCT16<From, ToTransposed, /*COLS=*/32>(
        from, ToTransposed(transposed_block));
    TransposeBlock1632(FromTransposed(transposed_block), ToOriginal(block));
    ColumnIDCT32<FromOriginal, To, /*COLS=*/16>(FromOriginal(block), to);
  }
};

template <>
struct ComputeScaledIDCT<8, 4> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 8];
    HWY_ALIGN float transposed_block[4 * 8];
    using FromOriginal = FromBlock<8, 4>;
    using FromTransposed = FromBlock<4, 8>;
    using ToOriginal = ToBlock<8, 4>;
    using ToTransposed = ToBlock<4, 8>;
    ColumnIDCT4<From, ToTransposed, /*COLS=*/8>(from,
                                                ToTransposed(transposed_block));
    GenericTransposeBlock<4, 8>(FromTransposed(transposed_block),
                                ToOriginal(block));
    ColumnIDCT8<FromOriginal, To, /*COLS=*/4>(FromOriginal(block), to);
  }
};

template <>
struct ComputeScaledIDCT<4, 8> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[8 * 4];
    HWY_ALIGN float transposed_block[8 * 4];
    using FromOriginal = FromBlock<4, 8>;
    using FromTransposed = FromBlock<8, 4>;
    using ToOriginal = ToBlock<4, 8>;
    using ToTransposed = ToBlock<8, 4>;
    GenericTransposeBlock<4, 8>(from, ToTransposed(block));
    ColumnIDCT8<FromTransposed, ToTransposed, /*COLS=*/4>(
        FromTransposed(block), ToTransposed(transposed_block));
    GenericTransposeBlock<8, 4>(FromTransposed(transposed_block),
                                ToOriginal(block));
    ColumnIDCT4<FromOriginal, To, /*COLS=*/8>(FromOriginal(block), to);
  }
};

template <>
struct ComputeScaledIDCT<4, 2> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 2];
    using FromOriginal = FromBlock<4, 2>;
    for (size_t y = 0; y < 4; ++y) {
      const float a0 = from.Read(0, y);
      const float a1 = from.Read(1, y);
      block[2 * y] = a0 + a1;
      block[2 * y + 1] = a0 - a1;
    }
    ColumnIDCT4<FromOriginal, To, /*COLS=*/2>(FromOriginal(block), to);
  }
};

template <>
struct ComputeScaledIDCT<2, 4> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float coeffs[4 * 2];
    using FromTransposed = FromBlock<4, 2>;
    using ToTransposed = ToBlock<4, 2>;
    GenericTransposeBlock<2, 4>(from, ToTransposed(coeffs));
    HWY_ALIGN float block[4 * 2];
    ColumnIDCT4<FromTransposed, ToTransposed, /*COLS=*/2>(
        FromTransposed(coeffs), ToTransposed(block));
    for (size_t y = 0; y < 4; ++y) {
      const float a0 = block[2 * y];
      const float a1 = block[2 * y + 1];
      to.Write(a0 + a1, 0, y);
      to.Write(a0 - a1, 1, y);
    }
  }
};

template <>
struct ComputeScaledIDCT<4, 1> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 1];
    GenericTransposeBlock<1, 4>(from, ToBlock<4, 1>(block));
    ColumnIDCT4<FromBlock<4, 1>, To, /*COLS=*/1>(FromBlock<4, 1>(block), to);
  }
};

template <>
struct ComputeScaledIDCT<1, 4> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    HWY_ALIGN float coeffs[4 * 1];
    using FromTransposed = FromBlock<4, 1>;
    using ToTransposed = ToBlock<4, 1>;
    GenericTransposeBlock<1, 4>(from, ToTransposed(coeffs));
    HWY_ALIGN float block[4 * 1];
    ColumnIDCT4<FromTransposed, ToTransposed, /*COLS=*/1>(
        FromTransposed(coeffs), ToTransposed(block));
    GenericTransposeBlock<4, 1>(FromTransposed(block), to);
  }
};

template <>
struct ComputeScaledIDCT<2, 1> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    const float a0 = from.Read(0, 0);
    const float a1 = from.Read(0, 1);
    to.Write(a0 + a1, 0, 0);
    to.Write(a0 - a1, 1, 0);
  }
};

template <>
struct ComputeScaledIDCT<1, 2> {
  template <class From, class To>
  HWY_ATTR JXL_INLINE void operator()(const From& from, const To& to) {
    const float a0 = from.Read(0, 0);
    const float a1 = from.Read(0, 1);
    to.Write(a0 + a1, 0, 0);
    to.Write(a0 - a1, 0, 1);
  }
};

namespace slow_dct {
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
}  // namespace slow_dct

}  // namespace jxl

#endif  // JXL_DCT_H_
