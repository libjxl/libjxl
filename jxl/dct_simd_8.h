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

#ifndef JXL_DCT_HWY_8_H_
#define JXL_DCT_HWY_8_H_

#include <hwy/static_targets.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/block.h"
#include "jxl/dct_simd_any.h"

#if HWY_BITS >= 256

namespace jxl {

// DCT building blocks that require SIMD vector length to be 8, e.g. AVX2.
static_assert(BlockDesc<8>::N == 8, "Wrong vector size, must be 8");

// Each vector holds one row of the input/output block.
template <class V>
HWY_ATTR JXL_INLINE void TransposeBlock8_V8(V& i0, V& i1, V& i2, V& i3, V& i4,
                                            V& i5, V& i6, V& i7) {
  // Surprisingly, this straightforward implementation (24 cycles on port5) is
  // faster than load128+insert and LoadDup128+ConcatHiLo+blend.
  const auto q0 = InterleaveLo(i0, i2);
  const auto q1 = InterleaveLo(i1, i3);
  const auto q2 = InterleaveHi(i0, i2);
  const auto q3 = InterleaveHi(i1, i3);
  const auto q4 = InterleaveLo(i4, i6);
  const auto q5 = InterleaveLo(i5, i7);
  const auto q6 = InterleaveHi(i4, i6);
  const auto q7 = InterleaveHi(i5, i7);

  const auto r0 = InterleaveLo(q0, q1);
  const auto r1 = InterleaveHi(q0, q1);
  const auto r2 = InterleaveLo(q2, q3);
  const auto r3 = InterleaveHi(q2, q3);
  const auto r4 = InterleaveLo(q4, q5);
  const auto r5 = InterleaveHi(q4, q5);
  const auto r6 = InterleaveLo(q6, q7);
  const auto r7 = InterleaveHi(q6, q7);

  i0 = ConcatLoLo(r4, r0);
  i1 = ConcatLoLo(r5, r1);
  i2 = ConcatLoLo(r6, r2);
  i3 = ConcatLoLo(r7, r3);
  i4 = ConcatHiHi(r4, r0);
  i5 = ConcatHiHi(r5, r1);
  i6 = ConcatHiHi(r6, r2);
  i7 = ConcatHiHi(r7, r3);
}

template <class From, class To>
HWY_ATTR JXL_INLINE void TransposeBlock8_V8(const From& from, const To& to) {
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

template <class From, class To>
HWY_ATTR JXL_INLINE void ComputeTransposedScaledDCT8_V8(const From& from,
                                                        const To& to) {
  const BlockDesc<8> d;

  const float c1234_lanes[4] = {
      0.707106781186548f,  // 1 / sqrt(2)
      0.382683432365090f,  // cos(3 * pi / 8)
      1.30656296487638f,   // 1 / (2 * cos(3 * pi / 8))
      0.541196100146197f   // sqrt(2) * cos(3 * pi / 8)
  };
  const auto c1234 = LoadDup128(d, c1234_lanes);
  const auto k1 = Set(d, 1.0f);

  auto i0 = from.template LoadPart<8>(0, 0);
  auto i7 = from.template LoadPart<8>(7, 0);
  auto t00 = i0 + i7;                // 2 (faster than Add)
  auto t01 = NegMulAdd(i7, k1, i0);  // 4
  HWY_FENCE;

  auto i3 = from.template LoadPart<8>(3, 0);
  auto i4 = from.template LoadPart<8>(4, 0);
  auto t02 = i3 + i4;
  auto t03 = NegMulAdd(i4, k1, i3);  // 1
  HWY_FENCE;

  auto i2 = from.template LoadPart<8>(2, 0);
  auto i5 = from.template LoadPart<8>(5, 0);
  auto t04 = i2 + i5;  // 1
  auto t05 = NegMulAdd(i5, k1, i2);
  HWY_FENCE;

  auto i1 = from.template LoadPart<8>(1, 0);
  auto i6 = from.template LoadPart<8>(6, 0);
  auto t06 = i1 + i6;  // !
  HWY_FENCE;

  auto t07 = i1 - i6;
  auto t09 = NegMulAdd(t02, k1, t00);
  const auto c4 = hwy::Broadcast<3>(c1234);

  auto t11 = t06 - t04;             // !
  auto t08 = MulAdd(t00, k1, t02);  // 2
  const auto c3 = hwy::Broadcast<2>(c1234);

  auto t14 = t05 + t03;
  auto t10 = MulAdd(t06, k1, t04);  // 1; dep-1

  auto t13 = t01 + t07;  // limits odd d
  const auto c1 = hwy::Broadcast<0>(c1234);

  auto t15 = t11 + t09;  // !
  const auto c2 = hwy::Broadcast<1>(c1234);

  auto t12 = t07 + t05;  // !
  auto ct14 = c4 * t14;

  auto t16 = t14 - t13;  // 1
  auto ct13 = c3 * t13;

  auto d0 = MulAdd(t08, k1, t10);
  auto d2 = MulAdd(c1, t15, t09);

  auto t21 = NegMulAdd(c1, t12, t01);  // 2

  auto d6 = NegMulAdd(c1, t15, t09);
  auto t20 = MulAdd(c1, t12, t01);  // 2

  auto t23 = MulAdd(c2, t16, ct14);

  auto d4 = t08 - t10;
  auto t22 = MulAdd(c2, t16, ct13);  // !

  const auto q0 = InterleaveLo(d0, d2);

  const auto q2 = InterleaveHi(d0, d2);

  const auto q4 = InterleaveLo(d4, d6);

  auto d3 = t21 - t23;
  const auto q6 = InterleaveHi(d4, d6);

  auto d1 = t20 + t22;
  const auto q1 = InterleaveLo(d1, d3);

  const auto r0 = InterleaveLo(q0, q1);
  const auto r1 = InterleaveHi(q0, q1);

  auto d7 = t20 - t22;
  const auto q3 = InterleaveHi(d1, d3);
  const auto r2 = InterleaveLo(q2, q3);
  const auto r3 = InterleaveHi(q2, q3);

  auto d5 = t21 + t23;
  const auto q5 = InterleaveLo(d5, d7);
  const auto r4 = InterleaveLo(q4, q5);
  const auto r5 = InterleaveHi(q4, q5);

  const auto q7 = InterleaveHi(d5, d7);
  const auto r6 = InterleaveLo(q6, q7);
  const auto r7 = InterleaveHi(q6, q7);

  // Second column-DCT after transpose
  i0 = ConcatLoLo(r4, r0);
  i7 = ConcatHiHi(r7, r3);
  t01 = i0 - i7;             // 1
  t00 = MulAdd(i0, k1, i7);  // 2

  i1 = ConcatLoLo(r5, r1);
  i6 = ConcatHiHi(r6, r2);
  t07 = i1 - i6;             // !
  t06 = MulAdd(i1, k1, i6);  // 2

  i3 = ConcatLoLo(r7, r3);
  i4 = ConcatHiHi(r4, r0);
  t03 = i3 - i4;             // 1
  t02 = MulAdd(i3, k1, i4);  // !

  i2 = ConcatLoLo(r6, r2);
  i5 = ConcatHiHi(r5, r1);
  t05 = i2 - i5;

  t13 = t01 + t07;  // 1

  t04 = i2 + i5;

  t14 = t05 + t03;
  t12 = MulAdd(t07, k1, t05);  // 2

  t09 = NegMulAdd(t02, k1, t00);
  ct13 = c3 * t13;  // 1

  t11 = t06 - t04;  // 1
  t10 = MulAdd(t06, k1, t04);

  t16 = t14 - t13;  // !
  ct14 = c4 * t14;

  t08 = t00 + t02;

  t20 = MulAdd(c1, t12, t01);  // 1

  t15 = t11 + t09;
  t22 = MulAdd(c2, t16, ct13);

  i0 = t08 + t10;

  t21 = NegMulAdd(c1, t12, t01);
  t23 = MulAdd(c2, t16, ct14);

  i4 = t08 - t10;
  i2 = MulAdd(c1, t15, t09);

  i6 = NegMulAdd(c1, t15, t09);
  to.template StorePart<8>(i0, 0, 0);
  HWY_FENCE;

  i1 = t20 + t22;

  i7 = t20 - t22;
  to.template StorePart<8>(i2, 2, 0);
  to.template StorePart<8>(i4, 4, 0);
  HWY_FENCE;

  i3 = t21 - t23;
  to.template StorePart<8>(i1, 1, 0);
  HWY_FENCE;

  i5 = t21 + t23;
  to.template StorePart<8>(i6, 6, 0);
  to.template StorePart<8>(i7, 7, 0);
  to.template StorePart<8>(i3, 3, 0);
  to.template StorePart<8>(i5, 5, 0);
}

template <class From, class To>
HWY_ATTR JXL_INLINE void ComputeTransposedScaledIDCT8_V8(const From& from,
                                                         const To& to) {
  const BlockDesc<8> d;

  const float k1_lanes[4] = {HWY_REP4(1.0f)};
  const auto k1 = LoadDup128(d, k1_lanes);
  const float c1234_lanes[4] = {
      1.41421356237310f,  // sqrt(2)
      2.61312592975275f,  // 1 / cos(3 * pi / 8)
      0.76536686473018f,  // 2 * cos(3 * pi / 8)
      1.08239220029239f   // 2 * sqrt(2) * cos(3 * pi / 8)
  };
  const auto c1234 = LoadDup128(d, c1234_lanes);
  HWY_FENCE;

  // Finish d5,d7 and d0,d2 first so we can overlap more port5 (shuffles) with
  // other computations; they have a shorter dependency chain than d13/46.

  auto i1 = from.template LoadPart<8>(1, 0);
  auto i7 = from.template LoadPart<8>(7, 0);
  auto t05 = i7 - i1;             // !
  auto t04 = MulAdd(i7, k1, i1);  // 1

  auto i3 = from.template LoadPart<8>(3, 0);
  auto i5 = from.template LoadPart<8>(5, 0);
  auto t07 = i5 - i3;             // +1
  auto t06 = MulAdd(i5, k1, i3);  // +1

  auto i2 = from.template LoadPart<8>(2, 0);
  auto i6 = from.template LoadPart<8>(6, 0);
  auto t02 = i6 + i2;  // 1
  const auto c2 = hwy::Broadcast<1>(c1234);
  HWY_FENCE;

  auto i0 = from.template LoadPart<8>(0, 0);
  auto i4 = from.template LoadPart<8>(4, 0);
  auto t03 = i6 - i2;    // !
  auto ct05 = c2 * t05;  // !
  HWY_FENCE;

  auto t12 = t07 - t05;                      // 1
  const auto c1 = hwy::Broadcast<0>(c1234);  // 1

  auto t00 = MulAdd(i0, k1, i4);             // +2
  const auto c3 = hwy::Broadcast<2>(c1234);  // 2

  auto t09 = NegMulAdd(t06, k1, t04);
  auto t14 = MulAdd(c1, t03, t02);  // +3

  auto t08 = MulAdd(t04, k1, t06);           // 1
  const auto c4 = hwy::Broadcast<3>(c1234);  // 2

  auto t01 = i0 - i4;                // +1
  auto t17 = MulAdd(c3, t12, ct05);  // !
  HWY_FENCE;

  //

  auto t10 = MulAdd(t00, k1, t02);
  auto ct07 = c4 * t07;  // !

  auto t15 = NegMulAdd(t14, k1, t01);  // 1
  auto ct09 = c1 * t09;

  auto t11 = NegMulAdd(t02, k1, t00);  // 6

  auto t19 = t08 + t17;  // !

  auto t16 = MulAdd(t01, k1, t14);

  auto d0 = MulAdd(t10, k1, t08);    // dep-3; 4
  auto t18 = MulAdd(c3, t12, ct07);  // !

  auto t20 = ct09 + t19;              // !
  auto d7 = NegMulAdd(t08, k1, t10);  // 1

  auto d1 = NegMulAdd(t19, k1, t15);  // 5

  //

  auto d5 = t16 - t20;  // !
  auto d2 = MulAdd(t16, k1, t20);

  auto t21 = t18 - t20;  // !

  //

  // Begin transposing finished d#

  auto d6 = t15 + t19;  // 1
  const auto q5 = InterleaveLo(d5, d7);

  auto d4 = t11 - t21;                   // !
  const auto q7 = InterleaveHi(d5, d7);  // 8

  auto d3 = t11 + t21;  // !
  const auto q0 = InterleaveLo(d0, d2);

  const auto q2 = InterleaveHi(d0, d2);  // 8

  const auto q4 = InterleaveLo(d4, d6);

  const auto q1 = InterleaveLo(d1, d3);

  const auto r4 = InterleaveLo(q4, q5);

  const auto r0 = InterleaveLo(q0, q1);

  i0 = ConcatLoLo(r4, r0);

  i4 = ConcatHiHi(r4, r0);
  const auto _c1234 = LoadDup128(d, c1234_lanes);

  const auto q3 = InterleaveHi(d1, d3);

  // Begin second column-IDCT for transposed r#

  const auto q6 = InterleaveHi(d4, d6);

  t00 = MulAdd(i0, k1, i4);
  const auto r2 = InterleaveLo(q2, q3);

  t01 = NegMulAdd(i4, k1, i0);
  const auto r6 = InterleaveLo(q6, q7);

  i2 = ConcatLoLo(r6, r2);

  i6 = ConcatHiHi(r6, r2);

  const auto r7 = InterleaveHi(q6, q7);

  const auto r3 = InterleaveHi(q2, q3);

  t03 = i6 - i2;
  i7 = ConcatHiHi(r7, r3);

  t02 = i6 + i2;
  const auto r5 = InterleaveHi(q4, q5);

  const auto r1 = InterleaveHi(q0, q1);
  const auto _c1 = hwy::Broadcast<0>(_c1234);

  i1 = ConcatLoLo(r5, r1);
  auto ct03 = _c1 * t03;

  t10 = MulAdd(t00, k1, t02);  // 5
  i5 = ConcatHiHi(r5, r1);

  i3 = ConcatLoLo(r7, r3);

  t05 = i7 - i1;  // !
  const auto _c2 = hwy::Broadcast<1>(_c1234);

  t04 = MulAdd(i7, k1, i1);  // 1

  t07 = i5 - i3;

  t06 = i5 + i3;
  ct05 = _c2 * t05;  // !

  t14 = ct03 + t02;  // 1

  t12 = t07 - t05;

  t08 = t04 + t06;

  t09 = t04 - t06;

  t15 = NegMulAdd(t14, k1, t01);  // 3
  t17 = MulAdd(c3, t12, ct05);    // !

  d0 = t10 + t08;

  d7 = t10 - t08;

  ct09 = _c1 * t09;

  const auto _c4 = hwy::Broadcast<3>(_c1234);
  to.StoreVec(d0, 0, 0);
  HWY_FENCE;

  t19 = t08 + t17;   // !
  ct07 = _c4 * t07;  // !
  to.StoreVec(d7, 7, 0);
  HWY_FENCE;

  t11 = t00 - t02;  // 8

  t16 = t01 + t14;  // 3

  d1 = t15 - t19;
  t20 = ct09 + t19;  // !

  d6 = t15 + t19;
  const auto _c3 = hwy::Broadcast<2>(_c1234);

  t18 = MulAdd(_c3, t12, ct07);  // !

  d2 = t16 + t20;
  to.StoreVec(d1, 1, 0);
  HWY_FENCE;

  d5 = t16 - t20;
  to.StoreVec(d6, 6, 0);
  HWY_FENCE;

  t21 = t18 - t20;  // !

  d4 = t11 - t21;
  to.StoreVec(d2, 2, 0);

  d3 = t11 + t21;
  to.StoreVec(d5, 5, 0);

  to.StoreVec(d4, 4, 0);
  to.StoreVec(d3, 3, 0);
}

}  // namespace jxl

#endif

#endif  // JXL_DCT_HWY_8_H_
