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

// Fast SIMD floating-point IDCT8-32.

#if defined(JXL_DEC_DCT_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef JXL_DEC_DCT_INL_H_
#undef JXL_DEC_DCT_INL_H_
#else
#define JXL_DEC_DCT_INL_H_
#endif

#include <hwy/highway.h>
#include <stddef.h>

#include "jxl/dct_block-inl.h"
#include "jxl/transpose-inl.h"

namespace jxl {

#include <hwy/begin_target-inl.h>

// Column IDCTs with (part of) one row per vector argument. Called by the
// facades below which take From/To template arguments.
struct VectorIDCT {
  template <size_t SZ, class V>
  HWY_FUNC static void ColumnIDCT4(V& i0, V& i1, V& i2, V& i3) {
    const BlockDesc<SZ> d;
    const auto c2_8 = Set(d, 0.7071067811865475244f);  // 0.5 / cos(2 * pi / 8)
    auto t0 = i0 + i2;
    auto t1 = i0 - i2;
    auto t2 = i1 + i3;
    auto t3 = i1 - i3;
    auto t4 = t3 * c2_8;
    auto t5 = t2 + t4;
    i0 = t0 + t5;
    i1 = t1 + t4;
    i3 = t0 - t5;
    i2 = t1 - t4;
  }

  // NB: ColumnIDCT8(ColumnDCT8(I)) = 8.0 * I
  template <size_t SZ, class V>
  HWY_FUNC static void ColumnIDCT8(V& i0, V& i1, V& i2, V& i3, V& i4, V& i5,
                                   V& i6, V& i7) {
    const BlockDesc<SZ> d;

    const auto c1 = Set(d, 1.41421356237310f);  // sqrt(2)
    const auto c2 = Set(d, 2.61312592975275f);  // 1 / cos(3 * pi / 8)
    const auto c3 = Set(d, 0.76536686473018f);  // 2 * cos(3 * pi / 8)
    const auto c4 = Set(d, 1.08239220029239f);  // 2 * sqrt(2) * cos(3 * pi / 8)

    const auto t00 = i0 + i4;
    const auto t01 = i0 - i4;
    const auto t02 = i6 + i2;
    const auto t03 = i6 - i2;
    const auto t04 = i7 + i1;
    const auto t05 = i7 - i1;
    const auto t06 = i5 + i3;
    const auto t07 = i5 - i3;
    const auto t08 = t04 + t06;
    const auto t09 = t04 - t06;
    const auto t10 = t00 + t02;
    const auto t11 = t00 - t02;
    const auto t12 = t07 - t05;
    const auto t13 = c3 * t12;
    const auto t14 = MulAdd(c1, t03, t02);
    const auto t15 = t01 - t14;
    const auto t16 = t01 + t14;
    const auto t17 = MulAdd(c2, t05, t13);
    const auto t18 = MulAdd(c4, t07, t13);
    const auto t19 = t08 + t17;
    const auto t20 = MulAdd(c1, t09, t19);
    const auto t21 = t18 - t20;
    i0 = t10 + t08;
    i1 = t15 - t19;
    i2 = t16 + t20;
    i3 = t11 + t21;
    i4 = t11 - t21;
    i5 = t16 - t20;
    i6 = t15 + t19;
    i7 = t10 - t08;
  }

  // TODO(veluca): recursive implementation for fewer spills?
  template <size_t SZ, class V>
  HWY_FUNC static void ColumnIDCT32(V& i00, V& i01, V& i02, V& i03, V& i04,
                                    V& i05, V& i06, V& i07, V& i08, V& i09,
                                    V& i10, V& i11, V& i12, V& i13, V& i14,
                                    V& i15, V& i16, V& i17, V& i18, V& i19,
                                    V& i20, V& i21, V& i22, V& i23, V& i24,
                                    V& i25, V& i26, V& i27, V& i28, V& i29,
                                    V& i30, V& i31) {
    const BlockDesc<SZ> d;
    const auto c2_64 = Set(d, 0.502419286188155678f);  // 0.5 / cos(2 * pi / 64)
    const auto c4_64 = Set(d, 0.509795579104159180f);  // 0.5 / cos(4 * pi / 64)
    const auto c6_64 = Set(d, 0.522498614939688855f);  // 0.5 / cos(6 * pi / 64)
    const auto c8_64 = Set(d, 0.541196100146197012f);  // 0.5 / cos(8 * pi / 64)
    const auto c10_64 =
        Set(d, 0.566944034816357689f);  // 0.5 / cos(10 * pi / 64)
    const auto c12_64 =
        Set(d, 0.601344886935045286f);  // 0.5 / cos(12 * pi / 64)
    const auto c14_64 =
        Set(d, 0.646821783359990077f);  // 0.5 / cos(14 * pi / 64)
    const auto c16_64 =
        Set(d, 0.707106781186547462f);  // 0.5 / cos(16 * pi / 64)
    const auto c18_64 =
        Set(d, 0.788154623451250202f);  // 0.5 / cos(18 * pi / 64)
    const auto c20_64 =
        Set(d, 0.899976223136415565f);  // 0.5 / cos(20 * pi / 64)
    const auto c22_64 =
        Set(d, 1.060677685990347063f);  // 0.5 / cos(22 * pi / 64)
    const auto c24_64 =
        Set(d, 1.306562964876376354f);  // 0.5 / cos(24 * pi / 64)
    const auto c26_64 =
        Set(d, 1.722447098238334195f);  // 0.5 / cos(26 * pi / 64)
    const auto c28_64 =
        Set(d, 2.562915447741505481f);  // 0.5 / cos(28 * pi / 64)
    const auto c30_64 =
        Set(d, 5.101148618689155256f);  // 0.5 / cos(30 * pi / 64)

    const auto t00 = i00 + i16;
    const auto t01 = i00 - i16;
    const auto t02 = i08 + i24;
    const auto t03 = i08 - i24;
    const auto t04 = t03 * c16_64;
    const auto t05 = t02 + t04;
    const auto t06 = t00 + t05;
    const auto t07 = t01 + t04;
    const auto t08 = t00 - t05;
    const auto t09 = t01 - t04;
    const auto t10 = i04 + i28;
    const auto t11 = i04 - i28;
    const auto t12 = t11 * c8_64;
    const auto t13 = t10 + t12;
    const auto t14 = i12 + i20;
    const auto t15 = i12 - i20;
    const auto t16 = t15 * c24_64;
    const auto t17 = t14 + t16;
    const auto t18 = t13 + t17;
    const auto t19 = t12 + t16;
    const auto t20 = t13 - t17;
    const auto t21 = t12 - t16;
    const auto t22 = t20 * c16_64;
    const auto t23 = t21 * c16_64;
    const auto t24 = t18 + t23;
    const auto t25 = t19 + t22;
    const auto t26 = t06 + t24;
    const auto t27 = t07 + t25;
    const auto t28 = t09 + t22;
    const auto t29 = t08 + t23;
    const auto t30 = t06 - t24;
    const auto t31 = t07 - t25;
    const auto t32 = t09 - t22;
    const auto t33 = t08 - t23;
    const auto t34 = i02 + i30;
    const auto t35 = i02 - i30;
    const auto t36 = t35 * c4_64;
    const auto t37 = t34 + t36;
    const auto t38 = i14 + i18;
    const auto t39 = i14 - i18;
    const auto t40 = t39 * c28_64;
    const auto t41 = t38 + t40;
    const auto t42 = t37 + t41;
    const auto t43 = t36 + t40;
    const auto t44 = t37 - t41;
    const auto t45 = t36 - t40;
    const auto t46 = t44 * c8_64;
    const auto t47 = t45 * c8_64;
    const auto t48 = t42 + t47;
    const auto t49 = t43 + t46;
    const auto t50 = i06 + i26;
    const auto t51 = i06 - i26;
    const auto t52 = t51 * c12_64;
    const auto t53 = t50 + t52;
    const auto t54 = i10 + i22;
    const auto t55 = i10 - i22;
    const auto t56 = t55 * c20_64;
    const auto t57 = t54 + t56;
    const auto t58 = t53 + t57;
    const auto t59 = t52 + t56;
    const auto t60 = t53 - t57;
    const auto t61 = t52 - t56;
    const auto t62 = t60 * c24_64;
    const auto t63 = t61 * c24_64;
    const auto t64 = t58 + t63;
    const auto t65 = t59 + t62;
    const auto t66 = t48 + t64;
    const auto t67 = t49 + t65;
    const auto t68 = t46 + t62;
    const auto t69 = t47 + t63;
    const auto t70 = t48 - t64;
    const auto t71 = t49 - t65;
    const auto t72 = t46 - t62;
    const auto t73 = t47 - t63;
    const auto t74 = t70 * c16_64;
    const auto t75 = t71 * c16_64;
    const auto t76 = t72 * c16_64;
    const auto t77 = t73 * c16_64;
    const auto t78 = t66 + t77;
    const auto t79 = t67 + t76;
    const auto t80 = t68 + t75;
    const auto t81 = t69 + t74;
    const auto t82 = t26 + t78;
    const auto t83 = t27 + t79;
    const auto t84 = t28 + t80;
    const auto t85 = t29 + t81;
    const auto t86 = t33 + t74;
    const auto t87 = t32 + t75;
    const auto t88 = t31 + t76;
    const auto t89 = t30 + t77;
    const auto t90 = t26 - t78;
    const auto t91 = t27 - t79;
    const auto t92 = t28 - t80;
    const auto t93 = t29 - t81;
    const auto t94 = t33 - t74;
    const auto t95 = t32 - t75;
    const auto t96 = t31 - t76;
    const auto t97 = t30 - t77;
    const auto t98 = i01 + i31;
    const auto t99 = i01 - i31;
    const auto t100 = t99 * c2_64;
    const auto t101 = t98 + t100;
    const auto t102 = i15 + i17;
    const auto t103 = i15 - i17;
    const auto t104 = t103 * c30_64;
    const auto t105 = t102 + t104;
    const auto t106 = t101 + t105;
    const auto t107 = t100 + t104;
    const auto t108 = t101 - t105;
    const auto t109 = t100 - t104;
    const auto t110 = t108 * c4_64;
    const auto t111 = t109 * c4_64;
    const auto t112 = t106 + t111;
    const auto t113 = t107 + t110;
    const auto t114 = i07 + i25;
    const auto t115 = i07 - i25;
    const auto t116 = t115 * c14_64;
    const auto t117 = t114 + t116;
    const auto t118 = i09 + i23;
    const auto t119 = i09 - i23;
    const auto t120 = t119 * c18_64;
    const auto t121 = t118 + t120;
    const auto t122 = t117 + t121;
    const auto t123 = t116 + t120;
    const auto t124 = t117 - t121;
    const auto t125 = t116 - t120;
    const auto t126 = t124 * c28_64;
    const auto t127 = t125 * c28_64;
    const auto t128 = t122 + t127;
    const auto t129 = t123 + t126;
    const auto t130 = t112 + t128;
    const auto t131 = t113 + t129;
    const auto t132 = t110 + t126;
    const auto t133 = t111 + t127;
    const auto t134 = t112 - t128;
    const auto t135 = t113 - t129;
    const auto t136 = t110 - t126;
    const auto t137 = t111 - t127;
    const auto t138 = t134 * c8_64;
    const auto t139 = t135 * c8_64;
    const auto t140 = t136 * c8_64;
    const auto t141 = t137 * c8_64;
    const auto t142 = t130 + t141;
    const auto t143 = t131 + t140;
    const auto t144 = t132 + t139;
    const auto t145 = t133 + t138;
    const auto t146 = i03 + i29;
    const auto t147 = i03 - i29;
    const auto t148 = t147 * c6_64;
    const auto t149 = t146 + t148;
    const auto t150 = i13 + i19;
    const auto t151 = i13 - i19;
    const auto t152 = t151 * c26_64;
    const auto t153 = t150 + t152;
    const auto t154 = t149 + t153;
    const auto t155 = t148 + t152;
    const auto t156 = t149 - t153;
    const auto t157 = t148 - t152;
    const auto t158 = t156 * c12_64;
    const auto t159 = t157 * c12_64;
    const auto t160 = t154 + t159;
    const auto t161 = t155 + t158;
    const auto t162 = i05 + i27;
    const auto t163 = i05 - i27;
    const auto t164 = t163 * c10_64;
    const auto t165 = t162 + t164;
    const auto t166 = i11 + i21;
    const auto t167 = i11 - i21;
    const auto t168 = t167 * c22_64;
    const auto t169 = t166 + t168;
    const auto t170 = t165 + t169;
    const auto t171 = t164 + t168;
    const auto t172 = t165 - t169;
    const auto t173 = t164 - t168;
    const auto t174 = t172 * c20_64;
    const auto t175 = t173 * c20_64;
    const auto t176 = t170 + t175;
    const auto t177 = t171 + t174;
    const auto t178 = t160 + t176;
    const auto t179 = t161 + t177;
    const auto t180 = t158 + t174;
    const auto t181 = t159 + t175;
    const auto t182 = t160 - t176;
    const auto t183 = t161 - t177;
    const auto t184 = t158 - t174;
    const auto t185 = t159 - t175;
    const auto t186 = t182 * c24_64;
    const auto t187 = t183 * c24_64;
    const auto t188 = t184 * c24_64;
    const auto t189 = t185 * c24_64;
    const auto t190 = t178 + t189;
    const auto t191 = t179 + t188;
    const auto t192 = t180 + t187;
    const auto t193 = t181 + t186;
    const auto t194 = t142 + t190;
    const auto t195 = t143 + t191;
    const auto t196 = t144 + t192;
    const auto t197 = t145 + t193;
    const auto t198 = t138 + t186;
    const auto t199 = t139 + t187;
    const auto t200 = t140 + t188;
    const auto t201 = t141 + t189;
    const auto t202 = t142 - t190;
    const auto t203 = t143 - t191;
    const auto t204 = t144 - t192;
    const auto t205 = t145 - t193;
    const auto t206 = t138 - t186;
    const auto t207 = t139 - t187;
    const auto t208 = t140 - t188;
    const auto t209 = t141 - t189;
    const auto t210 = t202 * c16_64;
    const auto t211 = t203 * c16_64;
    const auto t212 = t204 * c16_64;
    const auto t213 = t205 * c16_64;
    const auto t214 = t206 * c16_64;
    const auto t215 = t207 * c16_64;
    const auto t216 = t208 * c16_64;
    const auto t217 = t209 * c16_64;
    const auto t218 = t194 + t217;
    const auto t219 = t195 + t216;
    const auto t220 = t196 + t215;
    const auto t221 = t197 + t214;
    const auto t222 = t198 + t213;
    const auto t223 = t199 + t212;
    const auto t224 = t200 + t211;
    const auto t225 = t201 + t210;
    const auto t226 = t82 + t218;
    const auto t227 = t83 + t219;
    const auto t228 = t84 + t220;
    const auto t229 = t85 + t221;
    const auto t230 = t86 + t222;
    const auto t231 = t87 + t223;
    const auto t232 = t88 + t224;
    const auto t233 = t89 + t225;
    const auto t234 = t97 + t210;
    const auto t235 = t96 + t211;
    const auto t236 = t95 + t212;
    const auto t237 = t94 + t213;
    const auto t238 = t93 + t214;
    const auto t239 = t92 + t215;
    const auto t240 = t91 + t216;
    const auto t241 = t90 + t217;
    const auto t242 = t82 - t218;
    const auto t243 = t83 - t219;
    const auto t244 = t84 - t220;
    const auto t245 = t85 - t221;
    const auto t246 = t86 - t222;
    const auto t247 = t87 - t223;
    const auto t248 = t88 - t224;
    const auto t249 = t89 - t225;
    const auto t250 = t97 - t210;
    const auto t251 = t96 - t211;
    const auto t252 = t95 - t212;
    const auto t253 = t94 - t213;
    const auto t254 = t93 - t214;
    const auto t255 = t92 - t215;
    const auto t256 = t91 - t216;
    const auto t257 = t90 - t217;

    i00 = t226;
    i01 = t227;
    i02 = t228;
    i03 = t229;
    i04 = t230;
    i05 = t231;
    i06 = t232;
    i07 = t233;
    i08 = t234;
    i09 = t235;
    i10 = t236;
    i11 = t237;
    i12 = t238;
    i13 = t239;
    i14 = t240;
    i15 = t241;
    i16 = t257;
    i17 = t256;
    i18 = t255;
    i19 = t254;
    i20 = t253;
    i21 = t252;
    i22 = t251;
    i23 = t250;
    i24 = t249;
    i25 = t248;
    i26 = t247;
    i27 = t246;
    i28 = t245;
    i29 = t244;
    i30 = t243;
    i31 = t242;
  }

};  // VectorIDCT

// Call VectorIDCT::ColumnIDCT* after loading rows.

template <class From, class To, size_t COLS = 4>
HWY_FUNC void ColumnIDCT4(const From& from, const To& to) {
  const BlockDesc<4> d;
  constexpr size_t kSize = COLS < d.N ? COLS : d.N;
  for (size_t i = 0; i < COLS; i += d.N) {
    auto i0 = from.template LoadPart<kSize>(0, i);
    auto i1 = from.template LoadPart<kSize>(1, i);
    auto i2 = from.template LoadPart<kSize>(2, i);
    auto i3 = from.template LoadPart<kSize>(3, i);
    VectorIDCT::ColumnIDCT4<kSize>(i0, i1, i2, i3);
    to.template StorePart<kSize>(i0, 0, i);
    to.template StorePart<kSize>(i1, 1, i);
    to.template StorePart<kSize>(i2, 2, i);
    to.template StorePart<kSize>(i3, 3, i);
  }
}

template <class From, class To, size_t COLS = 8>
HWY_FUNC void ColumnIDCT8(const From& from, const To& to) {
  const BlockDesc<COLS> d;

  for (size_t i = 0; i < COLS; i += d.N) {
    auto i0 = from.template LoadPart<COLS>(0, i);
    auto i1 = from.template LoadPart<COLS>(1, i);
    auto i2 = from.template LoadPart<COLS>(2, i);
    auto i3 = from.template LoadPart<COLS>(3, i);
    auto i4 = from.template LoadPart<COLS>(4, i);
    auto i5 = from.template LoadPart<COLS>(5, i);
    auto i6 = from.template LoadPart<COLS>(6, i);
    auto i7 = from.template LoadPart<COLS>(7, i);
    VectorIDCT::ColumnIDCT8<COLS>(i0, i1, i2, i3, i4, i5, i6, i7);
    to.template StorePart<COLS>(i0, 0, i);
    to.template StorePart<COLS>(i1, 1, i);
    to.template StorePart<COLS>(i2, 2, i);
    to.template StorePart<COLS>(i3, 3, i);
    to.template StorePart<COLS>(i4, 4, i);
    to.template StorePart<COLS>(i5, 5, i);
    to.template StorePart<COLS>(i6, 6, i);
    to.template StorePart<COLS>(i7, 7, i);
  }
}

// This is the main implementation, we skip the adapter because it leads to
// more spills.
// "A low multiplicative complexity fast recursive DCT-2 algorithm"
// Maxim Vashkevich, Alexander Pertrovsky, 27 Jul 2012
template <class From, class To, size_t COLS = 16>
HWY_ATTR HWY_MAYBE_UNUSED void ColumnIDCT16(const From& from, const To& to) {
  const BlockDesc<COLS> d;

  // Odd constants are only used once, so broadcast as needed.
  HWY_ALIGN constexpr float k1357[4] = {
      0.5097955791041592f,  // 0.5 / cos(1 * pi / 16)
      0.6013448869350453f,  // 0.5 / cos(3 * pi / 16)
      0.8999762231364156f,  // 0.5 / cos(5 * pi / 16)
      2.5629154477415055f,  // 0.5 / cos(7 * pi / 16)
  };

  HWY_ALIGN constexpr float c2_lanes[4] = {
      HWY_REP4(0.541196100146197f)};  // 0.5 / cos(2 * pi / 16)
  HWY_ALIGN constexpr float c4_lanes[4] = {
      HWY_REP4(0.7071067811865475f)};  // 0.5 / cos(4 * pi / 16)
  HWY_ALIGN constexpr float c6_lanes[4] = {
      HWY_REP4(1.3065629648763764f)};  // 0.5 / cos(6 * pi / 16)

  // Unrolling does not help.
  for (size_t i = 0; i < COLS; i += d.N) {
    auto i00 = from.template LoadPart<COLS>(0, i);
    auto i08 = from.template LoadPart<COLS>(8, i);
    auto i04 = from.template LoadPart<COLS>(4, i);
    auto i12 = from.template LoadPart<COLS>(12, i);
    const auto t00 = i00 + i08;
    const auto t01 = i00 - i08;
    const auto t02 = i04 + i12;
    const auto t03 = i04 - i12;
    HWY_FENCE;

    auto i02 = from.template LoadPart<COLS>(2, i);
    auto i06 = from.template LoadPart<COLS>(6, i);
    auto i10 = from.template LoadPart<COLS>(10, i);
    auto i14 = from.template LoadPart<COLS>(14, i);
    const auto t10 = i02 + i14;
    const auto t11 = i02 - i14;
    const auto t14 = i06 + i10;
    const auto t15 = i06 - i10;
    HWY_FENCE;

    auto i01 = from.template LoadPart<COLS>(1, i);
    auto i07 = from.template LoadPart<COLS>(7, i);
    auto i09 = from.template LoadPart<COLS>(9, i);
    auto i15 = from.template LoadPart<COLS>(15, i);

    auto i03 = from.template LoadPart<COLS>(3, i);
    auto i05 = from.template LoadPart<COLS>(5, i);
    auto i11 = from.template LoadPart<COLS>(11, i);
    auto i13 = from.template LoadPart<COLS>(13, i);

#if HWY_TARGET == HWY_SCALAR
    const auto c1_16 = Set(d, k1357[0]);
    const auto c3_16 = Set(d, k1357[1]);
    const auto c5_16 = Set(d, k1357[2]);
    const auto c7_16 = Set(d, k1357[3]);
    const auto c2_16 = Set(d, c2_lanes[0]);
    const auto c4_16 = Set(d, c4_lanes[0]);
    const auto c6_16 = Set(d, c6_lanes[0]);
#else
    const auto c7531 = LoadDup128(d, k1357);  // c1 is in lane0.
    const auto c1_16 = Broadcast<0>(c7531);
    const auto c3_16 = Broadcast<1>(c7531);
    const auto c5_16 = Broadcast<2>(c7531);
    const auto c7_16 = Broadcast<3>(c7531);
    const auto c2_16 = LoadDup128(d, c2_lanes);
    const auto c4_16 = LoadDup128(d, c4_lanes);
    const auto c6_16 = LoadDup128(d, c6_lanes);
#endif

    const auto t05 = MulAdd(t03, c4_16, t02);
    const auto t07 = MulAdd(t03, c4_16, t01);
    const auto t06 = t00 + t05;
    const auto t08 = t00 - t05;
    const auto t09 = NegMulAdd(t03, c4_16, t01);
    const auto t13 = MulAdd(t11, c2_16, t10);
    const auto t16 = t15 * c6_16;  // used in subsequent FMA
    const auto t17 = t14 + t16;
    const auto t18 = t13 + t17;
    const auto t19 = MulAdd(t11, c2_16, t16);
    const auto t20 = t13 - t17;
    const auto t21 = MulSub(t11, c2_16, t16);
    const auto t23 = t21 * c4_16;
    const auto t24 = t18 + t23;
    const auto t25 = MulAdd(t20, c4_16, t19);
    const auto t26 = t06 + t24;
    const auto t27 = t07 + t25;
    const auto t28 = MulAdd(t20, c4_16, t09);
    const auto t29 = t08 + t23;
    const auto t30 = t06 - t24;
    const auto t31 = t07 - t25;
    const auto t32 = NegMulAdd(t20, c4_16, t09);
    const auto t33 = t08 - t23;
    const auto t34 = i01 + i15;
    const auto t35 = i01 - i15;
    const auto t37 = MulAdd(t35, c1_16, t34);
    const auto t38 = i07 + i09;
    const auto t39 = i07 - i09;
    const auto t40 = t39 * c7_16;  // used in subsequent FMA
    const auto t41 = t38 + t40;
    const auto t42 = t37 + t41;
    const auto t43 = MulAdd(t35, c1_16, t40);
    const auto t44 = t37 - t41;
    const auto t45 = MulSub(t35, c1_16, t40);
    const auto t48 = MulAdd(t45, c2_16, t42);
    const auto t49 = MulAdd(t44, c2_16, t43);
    const auto t50 = i03 + i13;
    const auto t51 = i03 - i13;
    const auto t52 = t51 * c3_16;
    const auto t53 = t50 + t52;
    const auto t54 = i05 + i11;
    const auto t55 = i05 - i11;
    const auto t57 = MulAdd(t55, c5_16, t54);
    const auto t58 = t53 + t57;
    const auto t60 = t53 - t57;
    const auto t59 = MulAdd(t55, c5_16, t52);
    const auto t61 = NegMulAdd(t55, c5_16, t52);
    const auto t62 = t60 * c6_16;  // used in subsequent FMA
    const auto t63 = t61 * c6_16;  // used in subsequent FMA
    const auto t64 = t58 + t63;
    const auto t65 = t59 + t62;
    const auto t66 = t48 + t64;
    const auto t67 = t49 + t65;
    const auto t68 = MulAdd(t44, c2_16, t62);
    const auto t69 = MulAdd(t45, c2_16, t63);
    const auto t70 = t48 - t64;
    const auto t71 = t49 - t65;
    const auto t72 = MulSub(t44, c2_16, t62);
    const auto t73 = MulSub(t45, c2_16, t63);
    const auto t78 = MulAdd(t73, c4_16, t66);
    const auto t79 = MulAdd(t72, c4_16, t67);
    const auto t80 = MulAdd(t71, c4_16, t68);
    const auto t81 = MulAdd(t70, c4_16, t69);

    to.template StorePart<COLS>(t26 + t78, 0, i);
    to.template StorePart<COLS>(t27 + t79, 1, i);
    to.template StorePart<COLS>(t28 + t80, 2, i);
    to.template StorePart<COLS>(t29 + t81, 3, i);
    to.template StorePart<COLS>(MulAdd(t70, c4_16, t33), 4, i);
    to.template StorePart<COLS>(MulAdd(t71, c4_16, t32), 5, i);
    to.template StorePart<COLS>(MulAdd(t72, c4_16, t31), 6, i);
    to.template StorePart<COLS>(MulAdd(t73, c4_16, t30), 7, i);
    to.template StorePart<COLS>(NegMulAdd(t73, c4_16, t30), 8, i);
    to.template StorePart<COLS>(NegMulAdd(t72, c4_16, t31), 9, i);
    to.template StorePart<COLS>(NegMulAdd(t71, c4_16, t32), 10, i);
    to.template StorePart<COLS>(NegMulAdd(t70, c4_16, t33), 11, i);
    to.template StorePart<COLS>(t29 - t81, 12, i);
    to.template StorePart<COLS>(t28 - t80, 13, i);
    to.template StorePart<COLS>(t27 - t79, 14, i);
    to.template StorePart<COLS>(t26 - t78, 15, i);
  }
}

template <class From, class To, size_t COLS = 32>
HWY_ATTR HWY_MAYBE_UNUSED void ColumnIDCT32(const From& from, const To& to) {
  const BlockDesc<COLS> d;

  for (size_t i = 0; i < COLS; i += d.N) {
    auto i00 = from.template LoadPart<COLS>(0, i);
    auto i01 = from.template LoadPart<COLS>(1, i);
    auto i02 = from.template LoadPart<COLS>(2, i);
    auto i03 = from.template LoadPart<COLS>(3, i);
    auto i04 = from.template LoadPart<COLS>(4, i);
    auto i05 = from.template LoadPart<COLS>(5, i);
    auto i06 = from.template LoadPart<COLS>(6, i);
    auto i07 = from.template LoadPart<COLS>(7, i);
    auto i08 = from.template LoadPart<COLS>(8, i);
    auto i09 = from.template LoadPart<COLS>(9, i);
    auto i10 = from.template LoadPart<COLS>(10, i);
    auto i11 = from.template LoadPart<COLS>(11, i);
    auto i12 = from.template LoadPart<COLS>(12, i);
    auto i13 = from.template LoadPart<COLS>(13, i);
    auto i14 = from.template LoadPart<COLS>(14, i);
    auto i15 = from.template LoadPart<COLS>(15, i);
    auto i16 = from.template LoadPart<COLS>(16, i);
    auto i17 = from.template LoadPart<COLS>(17, i);
    auto i18 = from.template LoadPart<COLS>(18, i);
    auto i19 = from.template LoadPart<COLS>(19, i);
    auto i20 = from.template LoadPart<COLS>(20, i);
    auto i21 = from.template LoadPart<COLS>(21, i);
    auto i22 = from.template LoadPart<COLS>(22, i);
    auto i23 = from.template LoadPart<COLS>(23, i);
    auto i24 = from.template LoadPart<COLS>(24, i);
    auto i25 = from.template LoadPart<COLS>(25, i);
    auto i26 = from.template LoadPart<COLS>(26, i);
    auto i27 = from.template LoadPart<COLS>(27, i);
    auto i28 = from.template LoadPart<COLS>(28, i);
    auto i29 = from.template LoadPart<COLS>(29, i);
    auto i30 = from.template LoadPart<COLS>(30, i);
    auto i31 = from.template LoadPart<COLS>(31, i);
    VectorIDCT::ColumnIDCT32<COLS>(i00, i01, i02, i03, i04, i05, i06, i07, i08,
                                   i09, i10, i11, i12, i13, i14, i15, i16, i17,
                                   i18, i19, i20, i21, i22, i23, i24, i25, i26,
                                   i27, i28, i29, i30, i31);
    to.template StorePart<COLS>(i00, 0, i);
    to.template StorePart<COLS>(i01, 1, i);
    to.template StorePart<COLS>(i02, 2, i);
    to.template StorePart<COLS>(i03, 3, i);
    to.template StorePart<COLS>(i04, 4, i);
    to.template StorePart<COLS>(i05, 5, i);
    to.template StorePart<COLS>(i06, 6, i);
    to.template StorePart<COLS>(i07, 7, i);
    to.template StorePart<COLS>(i08, 8, i);
    to.template StorePart<COLS>(i09, 9, i);
    to.template StorePart<COLS>(i10, 10, i);
    to.template StorePart<COLS>(i11, 11, i);
    to.template StorePart<COLS>(i12, 12, i);
    to.template StorePart<COLS>(i13, 13, i);
    to.template StorePart<COLS>(i14, 14, i);
    to.template StorePart<COLS>(i15, 15, i);
    to.template StorePart<COLS>(i16, 16, i);
    to.template StorePart<COLS>(i17, 17, i);
    to.template StorePart<COLS>(i18, 18, i);
    to.template StorePart<COLS>(i19, 19, i);
    to.template StorePart<COLS>(i20, 20, i);
    to.template StorePart<COLS>(i21, 21, i);
    to.template StorePart<COLS>(i22, 22, i);
    to.template StorePart<COLS>(i23, 23, i);
    to.template StorePart<COLS>(i24, 24, i);
    to.template StorePart<COLS>(i25, 25, i);
    to.template StorePart<COLS>(i26, 26, i);
    to.template StorePart<COLS>(i27, 27, i);
    to.template StorePart<COLS>(i28, 28, i);
    to.template StorePart<COLS>(i29, 29, i);
    to.template StorePart<COLS>(i30, 30, i);
    to.template StorePart<COLS>(i31, 31, i);
  }
}

// Special case for 8-lane SIMD - combines IDCT and transpose.
#if HWY_CAPS & HWY_CAP_GE256

template <class From, class To>
HWY_FUNC void ComputeTransposedScaledIDCT8_V8(const From& from, const To& to) {
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
  const auto c2 = Broadcast<1>(c1234);
  HWY_FENCE;

  auto i0 = from.template LoadPart<8>(0, 0);
  auto i4 = from.template LoadPart<8>(4, 0);
  auto t03 = i6 - i2;    // !
  auto ct05 = c2 * t05;  // !
  HWY_FENCE;

  auto t12 = t07 - t05;                 // 1
  const auto c1 = Broadcast<0>(c1234);  // 1

  auto t00 = MulAdd(i0, k1, i4);        // +2
  const auto c3 = Broadcast<2>(c1234);  // 2

  auto t09 = NegMulAdd(t06, k1, t04);
  auto t14 = MulAdd(c1, t03, t02);  // +3

  auto t08 = MulAdd(t04, k1, t06);      // 1
  const auto c4 = Broadcast<3>(c1234);  // 2

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
  const auto q5 = InterleaveLower(d5, d7);

  auto d4 = t11 - t21;                      // !
  const auto q7 = InterleaveUpper(d5, d7);  // 8

  auto d3 = t11 + t21;  // !
  const auto q0 = InterleaveLower(d0, d2);

  const auto q2 = InterleaveUpper(d0, d2);  // 8

  const auto q4 = InterleaveLower(d4, d6);

  const auto q1 = InterleaveLower(d1, d3);

  const auto r4 = InterleaveLower(q4, q5);

  const auto r0 = InterleaveLower(q0, q1);

  i0 = ConcatLowerLower(r4, r0);

  i4 = ConcatUpperUpper(r4, r0);
  const auto _c1234 = LoadDup128(d, c1234_lanes);

  const auto q3 = InterleaveUpper(d1, d3);

  // Begin second column-IDCT for transposed r#

  const auto q6 = InterleaveUpper(d4, d6);

  t00 = MulAdd(i0, k1, i4);
  const auto r2 = InterleaveLower(q2, q3);

  t01 = NegMulAdd(i4, k1, i0);
  const auto r6 = InterleaveLower(q6, q7);

  i2 = ConcatLowerLower(r6, r2);

  i6 = ConcatUpperUpper(r6, r2);

  const auto r7 = InterleaveUpper(q6, q7);

  const auto r3 = InterleaveUpper(q2, q3);

  t03 = i6 - i2;
  i7 = ConcatUpperUpper(r7, r3);

  t02 = i6 + i2;
  const auto r5 = InterleaveUpper(q4, q5);

  const auto r1 = InterleaveUpper(q0, q1);
  const auto _c1 = Broadcast<0>(_c1234);

  i1 = ConcatLowerLower(r5, r1);
  auto ct03 = _c1 * t03;

  t10 = MulAdd(t00, k1, t02);  // 5
  i5 = ConcatUpperUpper(r5, r1);

  i3 = ConcatLowerLower(r7, r3);

  t05 = i7 - i1;  // !
  const auto _c2 = Broadcast<1>(_c1234);

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

  const auto _c4 = Broadcast<3>(_c1234);
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
  const auto _c3 = Broadcast<2>(_c1234);

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

#endif  // HWY_CAPS & HWY_CAP_GE256

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
  HWY_FUNC void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[32 * 32];
    ColumnIDCT32(from, ToBlock<32>(block));
    TransposeBlock32(FromBlock<32>(block), ToBlock<32>(block));
    ColumnIDCT32(FromBlock<32>(block), to);
  }
};

template <>
struct ComputeTransposedScaledIDCT<16> {
  template <class From, class To>
  HWY_FUNC void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[16 * 16];
    ColumnIDCT16(from, ToBlock<16>(block));
    TransposeBlock16(FromBlock<16>(block), ToBlock<16>(block));
    ColumnIDCT16(FromBlock<16>(block), to);
  }
};

template <>
struct ComputeTransposedScaledIDCT<8> {
  template <class From, class To>
  HWY_FUNC void operator()(const From& from, const To& to) {
#if HWY_CAPS & HWY_CAP_GE256
    ComputeTransposedScaledIDCT8_V8(from, to);
#elif HWY_TARGET == HWY_SCALAR
    HWY_ALIGN float block[8 * 8];
    ColumnIDCT8(from, ToBlock<8>(block));
    TransposeBlock8(FromBlock<8>(block), ToBlock<8>(block));
    ColumnIDCT8(FromBlock<8>(block), to);
#else  // 128-bit
    // TODO(user): it is possible to avoid using temporary array,
    // after generalizing "To" to be bi-directional; all sub-transforms could
    // be performed "in-place".
    HWY_ALIGN float block[8 * 8];
    ColumnIDCT8(from, ToBlock<8>(block));
    TransposeBlock8_V4(FromBlock<8>(block), ToBlock<8>(block));
    ColumnIDCT8(FromBlock<8>(block), to);
#endif
  }
};

template <>
struct ComputeTransposedScaledIDCT<4> {
  template <class From, class To>
  HWY_FUNC void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 4];
    ColumnIDCT4(from, ToBlock<4>(block));
    GenericTransposeBlockInplace<4>(FromBlock<4>(block), ToBlock<4>(block));
    ColumnIDCT4(FromBlock<4>(block), to);
  }
};

template <>
struct ComputeTransposedScaledIDCT<2> {
  template <class From, class To>
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 1];
    GenericTransposeBlock<1, 4>(from, ToBlock<4, 1>(block));
    ColumnIDCT4<FromBlock<4, 1>, To, /*COLS=*/1>(FromBlock<4, 1>(block), to);
  }
};

template <>
struct ComputeScaledIDCT<1, 4> {
  template <class From, class To>
  HWY_FUNC void operator()(const From& from, const To& to) {
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
  HWY_FUNC void operator()(const From& from, const To& to) {
    const float a0 = from.Read(0, 0);
    const float a1 = from.Read(0, 1);
    to.Write(a0 + a1, 0, 0);
    to.Write(a0 - a1, 1, 0);
  }
};

template <>
struct ComputeScaledIDCT<1, 2> {
  template <class From, class To>
  HWY_FUNC void operator()(const From& from, const To& to) {
    const float a0 = from.Read(0, 0);
    const float a1 = from.Read(0, 1);
    to.Write(a0 + a1, 0, 0);
    to.Write(a0 - a1, 0, 1);
  }
};

#include <hwy/end_target-inl.h>

}  // namespace jxl

#endif  // include guard
