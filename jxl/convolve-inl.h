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

// No include guard - included within HWY_NAMESPACE.

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
