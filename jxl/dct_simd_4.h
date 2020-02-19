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

#ifndef JXL_DCT_HWY_4_H_
#define JXL_DCT_HWY_4_H_

#include <hwy/static_targets.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/block.h"
#include "jxl/dct_simd_any.h"

#if HWY_BITS == 128

namespace jxl {

// DCT building blocks that require SIMD vector length to be 4, e.g. SSE4.
static_assert(BlockDesc<8>::N == 4, "Wrong vector size, must be 4");

template <class From, class To>
static HWY_ATTR JXL_INLINE void TransposeBlock4_V4(const From& from,
                                                   const To& to) {
  const auto p0 = from.LoadVec(0, 0);
  const auto p1 = from.LoadVec(1, 0);
  const auto p2 = from.LoadVec(2, 0);
  const auto p3 = from.LoadVec(3, 0);

  const auto q0 = InterleaveLo(p0, p2);
  const auto q1 = InterleaveLo(p1, p3);
  const auto q2 = InterleaveHi(p0, p2);
  const auto q3 = InterleaveHi(p1, p3);

  const auto r0 = InterleaveLo(q0, q1);
  const auto r1 = InterleaveHi(q0, q1);
  const auto r2 = InterleaveLo(q2, q3);
  const auto r3 = InterleaveHi(q2, q3);

  to.StoreVec(r0, 0, 0);
  to.StoreVec(r1, 1, 0);
  to.StoreVec(r2, 2, 0);
  to.StoreVec(r3, 3, 0);
}

// TODO(eustas): issue#40 temporary workaround.
template <class From, class To>
static HWY_ATTR JXL_NOINLINE void TransposeBlock4_V4_noinline(const From& from,
                                                              const To& to) {
  TransposeBlock4_V4(from, to);
}

template <class From, class To>
static HWY_ATTR JXL_INLINE void TransposeBlock8_V4(const From& from,
                                                   const To& to) {
  HWY_ALIGN float tmp[4 * 4];
#if !defined(__wasm_simd128__)
  TransposeBlock4_V4(from, to);
#else
  // TODO(eustas): issue#40 temporary workaround.
  TransposeBlock4_V4_noinline(from, to);
#endif
  TransposeBlock4_V4(from.View(0, 4), ToBlock<4>(tmp));
  TransposeBlock4_V4(from.View(4, 0), to.View(0, 4));
  CopyBlock4(FromBlock<4>(tmp), to.View(4, 0));
#if !defined(__wasm_simd128__)
  TransposeBlock4_V4(from.View(4, 4), to.View(4, 4));
#else
  // TODO(eustas): issue#40 temporary workaround.
  TransposeBlock4_V4_noinline(from.View(4, 4), to.View(4, 4));
#endif
}

template <class From, class To>
HWY_ATTR JXL_INLINE void ComputeTransposedScaledDCT8_V4(const From& from,
                                                        const To& to) {
  // TODO(user): it is possible to avoid using temporary array,
  // after generalizing "To" to be bi-directional; all sub-transforms could
  // be performed "in-place".
  HWY_ALIGN float block[8 * 8];
  ColumnDCT8(from, ToBlock<8>(block));
  TransposeBlock8_V4(FromBlock<8>(block), ToBlock<8>(block));
  ColumnDCT8(FromBlock<8>(block), to);
}

template <class From, class To>
HWY_ATTR JXL_INLINE void ComputeTransposedScaledIDCT8_V4(const From& from,
                                                         const To& to) {
  // TODO(user): it is possible to avoid using temporary array,
  // after generalizing "To" to be bi-directional; all sub-transforms could
  // be performed "in-place".
  HWY_ALIGN float block[8 * 8];
  ColumnIDCT8(from, ToBlock<8>(block));
  TransposeBlock8_V4(FromBlock<8>(block), ToBlock<8>(block));
  ColumnIDCT8(FromBlock<8>(block), to);
}

}  // namespace jxl

#endif

#endif  // JXL_DCT_HWY_4_H_
