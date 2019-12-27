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
static HWY_ATTR JXL_INLINE void TransposeBlock8_V4(const From& from,
                                                   const To& to) {
  const auto p0L = from.LoadVec(0, 0);
  const auto p0H = from.LoadVec(0, 4);
  const auto p1L = from.LoadVec(1, 0);
  const auto p1H = from.LoadVec(1, 4);
  const auto p2L = from.LoadVec(2, 0);
  const auto p2H = from.LoadVec(2, 4);
  const auto p3L = from.LoadVec(3, 0);
  const auto p3H = from.LoadVec(3, 4);
  const auto p4L = from.LoadVec(4, 0);
  const auto p4H = from.LoadVec(4, 4);
  const auto p5L = from.LoadVec(5, 0);
  const auto p5H = from.LoadVec(5, 4);
  const auto p6L = from.LoadVec(6, 0);
  const auto p6H = from.LoadVec(6, 4);
  const auto p7L = from.LoadVec(7, 0);
  const auto p7H = from.LoadVec(7, 4);

  const auto q0L = InterleaveLo(p0L, p2L);
  const auto q0H = InterleaveLo(p0H, p2H);
  const auto q1L = InterleaveLo(p1L, p3L);
  const auto q1H = InterleaveLo(p1H, p3H);
  const auto q2L = InterleaveHi(p0L, p2L);
  const auto q2H = InterleaveHi(p0H, p2H);
  const auto q3L = InterleaveHi(p1L, p3L);
  const auto q3H = InterleaveHi(p1H, p3H);
  const auto q4L = InterleaveLo(p4L, p6L);
  const auto q4H = InterleaveLo(p4H, p6H);
  const auto q5L = InterleaveLo(p5L, p7L);
  const auto q5H = InterleaveLo(p5H, p7H);
  const auto q6L = InterleaveHi(p4L, p6L);
  const auto q6H = InterleaveHi(p4H, p6H);
  const auto q7L = InterleaveHi(p5L, p7L);
  const auto q7H = InterleaveHi(p5H, p7H);

  const auto r0L = InterleaveLo(q0L, q1L);
  const auto r0H = InterleaveLo(q0H, q1H);
  const auto r1L = InterleaveHi(q0L, q1L);
  const auto r1H = InterleaveHi(q0H, q1H);
  const auto r2L = InterleaveLo(q2L, q3L);
  const auto r2H = InterleaveLo(q2H, q3H);
  const auto r3L = InterleaveHi(q2L, q3L);
  const auto r3H = InterleaveHi(q2H, q3H);
  const auto r4L = InterleaveLo(q4L, q5L);
  const auto r4H = InterleaveLo(q4H, q5H);
  const auto r5L = InterleaveHi(q4L, q5L);
  const auto r5H = InterleaveHi(q4H, q5H);
  const auto r6L = InterleaveLo(q6L, q7L);
  const auto r6H = InterleaveLo(q6H, q7H);
  const auto r7L = InterleaveHi(q6L, q7L);
  const auto r7H = InterleaveHi(q6H, q7H);

  to.StoreVec(r0L, 0, 0);
  to.StoreVec(r4L, 0, 4);
  to.StoreVec(r1L, 1, 0);
  to.StoreVec(r5L, 1, 4);
  to.StoreVec(r2L, 2, 0);
  to.StoreVec(r6L, 2, 4);
  to.StoreVec(r3L, 3, 0);
  to.StoreVec(r7L, 3, 4);
  to.StoreVec(r0H, 4, 0);
  to.StoreVec(r4H, 4, 4);
  to.StoreVec(r1H, 5, 0);
  to.StoreVec(r5H, 5, 4);
  to.StoreVec(r2H, 6, 0);
  to.StoreVec(r6H, 6, 4);
  to.StoreVec(r3H, 7, 0);
  to.StoreVec(r7H, 7, 4);
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
