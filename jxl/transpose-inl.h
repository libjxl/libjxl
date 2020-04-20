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

// Block transpose for DCT/IDCT

#if defined(JXL_TRANSPOSE_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef JXL_TRANSPOSE_INL_H_
#undef JXL_TRANSPOSE_INL_H_
#else
#define JXL_TRANSPOSE_INL_H_
#endif

#include <hwy/highway.h>
#include <stddef.h>
#include "jxl/base/status.h"
#include "jxl/dct_block-inl.h"

namespace jxl {

#include <hwy/begin_target-inl.h>

#ifndef JXL_INLINE_TRANSPOSE
// Workaround for issue #42 - (excessive?) inlining causes invalid codegen.
#if defined(__arm__)
#define JXL_INLINE_TRANSPOSE HWY_NOINLINE
#else
#define JXL_INLINE_TRANSPOSE HWY_INLINE
#endif
#endif  // JXL_INLINE_TRANSPOSE

// https://en.wikipedia.org/wiki/In-place_matrix_transposition#Square_matrices
template <size_t N, class From, class To>
HWY_FUNC void GenericTransposeBlockInplace(const From& from, const To& to) {
  // This does not guarantee anything, just saves from the most stupid mistakes.
  JXL_DASSERT(from.Address(0, 0) == to.Address(0, 0));
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
HWY_ATTR JXL_INLINE_TRANSPOSE void GenericTransposeBlock(const From& from,
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
HWY_FUNC void CopyBlock4(const From& from, const To& to) {
  const BlockDesc<4> d;
  for (size_t i = 0; i < 4; i += d.N) {
    const auto i0 = from.template LoadPart<4>(0, i);
    const auto i1 = from.template LoadPart<4>(1, i);
    const auto i2 = from.template LoadPart<4>(2, i);
    const auto i3 = from.template LoadPart<4>(3, i);
    to.template StorePart<4>(i0, 0, i);
    to.template StorePart<4>(i1, 1, i);
    to.template StorePart<4>(i2, 2, i);
    to.template StorePart<4>(i3, 3, i);
  }
}

template <class From, class To>
HWY_FUNC void CopyBlock8(const From& from, const To& to) {
  const BlockDesc<8> d;
  for (size_t i = 0; i < 8; i += d.N) {
    const auto i0 = from.template LoadPart<8>(0, i);
    const auto i1 = from.template LoadPart<8>(1, i);
    const auto i2 = from.template LoadPart<8>(2, i);
    const auto i3 = from.template LoadPart<8>(3, i);
    const auto i4 = from.template LoadPart<8>(4, i);
    const auto i5 = from.template LoadPart<8>(5, i);
    const auto i6 = from.template LoadPart<8>(6, i);
    const auto i7 = from.template LoadPart<8>(7, i);
    to.template StorePart<8>(i0, 0, i);
    to.template StorePart<8>(i1, 1, i);
    to.template StorePart<8>(i2, 2, i);
    to.template StorePart<8>(i3, 3, i);
    to.template StorePart<8>(i4, 4, i);
    to.template StorePart<8>(i5, 5, i);
    to.template StorePart<8>(i6, 6, i);
    to.template StorePart<8>(i7, 7, i);
  }
}

#if HWY_CAPS & HWY_CAP_GE256

// Each vector holds one row of the input/output block.
template <class V>
HWY_FUNC void TransposeBlock8_V8(V& i0, V& i1, V& i2, V& i3, V& i4, V& i5,
                                 V& i6, V& i7) {
  // Surprisingly, this straightforward implementation (24 cycles on port5) is
  // faster than load128+insert and LoadDup128+ConcatUpperLower+blend.
  const auto q0 = InterleaveLower(i0, i2);
  const auto q1 = InterleaveLower(i1, i3);
  const auto q2 = InterleaveUpper(i0, i2);
  const auto q3 = InterleaveUpper(i1, i3);
  const auto q4 = InterleaveLower(i4, i6);
  const auto q5 = InterleaveLower(i5, i7);
  const auto q6 = InterleaveUpper(i4, i6);
  const auto q7 = InterleaveUpper(i5, i7);

  const auto r0 = InterleaveLower(q0, q1);
  const auto r1 = InterleaveUpper(q0, q1);
  const auto r2 = InterleaveLower(q2, q3);
  const auto r3 = InterleaveUpper(q2, q3);
  const auto r4 = InterleaveLower(q4, q5);
  const auto r5 = InterleaveUpper(q4, q5);
  const auto r6 = InterleaveLower(q6, q7);
  const auto r7 = InterleaveUpper(q6, q7);

  i0 = ConcatLowerLower(r4, r0);
  i1 = ConcatLowerLower(r5, r1);
  i2 = ConcatLowerLower(r6, r2);
  i3 = ConcatLowerLower(r7, r3);
  i4 = ConcatUpperUpper(r4, r0);
  i5 = ConcatUpperUpper(r5, r1);
  i6 = ConcatUpperUpper(r6, r2);
  i7 = ConcatUpperUpper(r7, r3);
}

template <class From, class To>
HWY_FUNC void TransposeBlock8_V8(const From& from, const To& to) {
  auto i0 = from.template LoadPart<8>(0, 0);
  auto i1 = from.template LoadPart<8>(1, 0);
  auto i2 = from.template LoadPart<8>(2, 0);
  auto i3 = from.template LoadPart<8>(3, 0);
  auto i4 = from.template LoadPart<8>(4, 0);
  auto i5 = from.template LoadPart<8>(5, 0);
  auto i6 = from.template LoadPart<8>(6, 0);
  auto i7 = from.template LoadPart<8>(7, 0);
  TransposeBlock8_V8(i0, i1, i2, i3, i4, i5, i6, i7);
  to.template StorePart<8>(i0, 0, 0);
  to.template StorePart<8>(i1, 1, 0);
  to.template StorePart<8>(i2, 2, 0);
  to.template StorePart<8>(i3, 3, 0);
  to.template StorePart<8>(i4, 4, 0);
  to.template StorePart<8>(i5, 5, 0);
  to.template StorePart<8>(i6, 6, 0);
  to.template StorePart<8>(i7, 7, 0);
}

#else

template <class From, class To>
HWY_ATTR JXL_INLINE_TRANSPOSE HWY_MAYBE_UNUSED static void TransposeBlock4_V4(
    const From& from, const To& to) {
  const auto p0 = from.LoadVec(0, 0);
  const auto p1 = from.LoadVec(1, 0);
  const auto p2 = from.LoadVec(2, 0);
  const auto p3 = from.LoadVec(3, 0);

  const auto q0 = InterleaveLower(p0, p2);
  const auto q1 = InterleaveLower(p1, p3);
  const auto q2 = InterleaveUpper(p0, p2);
  const auto q3 = InterleaveUpper(p1, p3);

  const auto r0 = InterleaveLower(q0, q1);
  const auto r1 = InterleaveUpper(q0, q1);
  const auto r2 = InterleaveLower(q2, q3);
  const auto r3 = InterleaveUpper(q2, q3);

  to.StoreVec(r0, 0, 0);
  to.StoreVec(r1, 1, 0);
  to.StoreVec(r2, 2, 0);
  to.StoreVec(r3, 3, 0);
}

template <class From, class To>
HWY_ATTR JXL_INLINE_TRANSPOSE HWY_MAYBE_UNUSED static void TransposeBlock8_V4(
    const From& from, const To& to) {
  HWY_ALIGN float tmp[4 * 4];
  TransposeBlock4_V4(from, to);
  TransposeBlock4_V4(from.View(0, 4), ToBlock<4>(tmp));
  TransposeBlock4_V4(from.View(4, 0), to.View(0, 4));
  CopyBlock4(FromBlock<4>(tmp), to.View(4, 0));
  TransposeBlock4_V4(from.View(4, 4), to.View(4, 4));
}

#endif  // !HWY_CAP_GE256

template <class From, class To>
HWY_FUNC void TransposeBlock8(const From& from, const To& to) {
#if HWY_CAPS & HWY_CAP_GE256
  TransposeBlock8_V8(from, to);
#elif HWY_TARGET == HWY_SCALAR
  if (from.Address(0, 0) == to.Address(0, 0)) {
    GenericTransposeBlockInplace<8>(from, to);
  } else {
    GenericTransposeBlock<8, 8>(from, to);
  }
#else  // 128-bit
  TransposeBlock8_V4(from, to);
#endif
}

template <class From, class To>
HWY_ATTR HWY_MAYBE_UNUSED void TransposeBlock16(const From& from,
                                                const To& to) {
  HWY_ALIGN float tmp[8 * 8];
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(0, 8), ToBlock<8>(tmp));
  TransposeBlock8(from.View(8, 0), to.View(0, 8));
  CopyBlock8(FromBlock<8>(tmp), to.View(8, 0));
  TransposeBlock8(from.View(8, 8), to.View(8, 8));
}

template <class From, class To>
HWY_ATTR HWY_MAYBE_UNUSED void TransposeBlock168(const From& from,
                                                 const To& to) {
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(8, 0), to.View(0, 8));
}

template <class From, class To>
HWY_ATTR HWY_MAYBE_UNUSED void TransposeBlock328(const From& from,
                                                 const To& to) {
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(8, 0), to.View(0, 8));
  TransposeBlock8(from.View(16, 0), to.View(0, 16));
  TransposeBlock8(from.View(24, 0), to.View(0, 24));
}

template <class From, class To>
HWY_ATTR HWY_MAYBE_UNUSED void TransposeBlock3216(const From& from,
                                                  const To& to) {
  TransposeBlock16(from, to);
  TransposeBlock16(from.View(16, 0), to.View(0, 16));
}

template <class From, class To>
HWY_ATTR HWY_MAYBE_UNUSED void TransposeBlock816(const From& from,
                                                 const To& to) {
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(0, 8), to.View(8, 0));
}

template <class From, class To>
HWY_ATTR HWY_MAYBE_UNUSED void TransposeBlock832(const From& from,
                                                 const To& to) {
  TransposeBlock8(from, to);
  TransposeBlock8(from.View(0, 8), to.View(8, 0));
  TransposeBlock8(from.View(0, 16), to.View(16, 0));
  TransposeBlock8(from.View(0, 24), to.View(24, 0));
}

template <class From, class To>
HWY_ATTR HWY_MAYBE_UNUSED void TransposeBlock1632(const From& from,
                                                  const To& to) {
  TransposeBlock16(from, to);
  TransposeBlock16(from.View(0, 16), to.View(16, 0));
}

template <class From, class To>
HWY_ATTR HWY_MAYBE_UNUSED void TransposeBlock32(const From& from,
                                                const To& to) {
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

#include <hwy/end_target-inl.h>

}  // namespace jxl

#endif  // include guard
