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

// Fast SIMD floating-point DCT8-32.

#if defined(JXL_ENC_DCT_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef JXL_ENC_DCT_INL_H_
#undef JXL_ENC_DCT_INL_H_
#else
#define JXL_ENC_DCT_INL_H_
#endif

#include <stddef.h>

#include <hwy/highway.h>

#include "jxl/dct_block-inl.h"
#include "jxl/transpose-inl.h"
HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

// These templates are not found via ADL.
using hwy::HWY_NAMESPACE::Broadcast;

// Column IDCTs with (part of) one row per vector argument. Called by the
// facades below which take From/To template arguments.
struct VectorDCT {
  template <size_t SZ, class V>
  HWY_MAYBE_UNUSED static void ColumnDCT4(V& i0, V& i1, V& i2, V& i3) {
    const BlockDesc<SZ> d;
    const auto c2_8 = Set(d, 1.414213562373095048f);  // 2 * cos(2 * pi / 8)
    auto t0 = i0 + i3;
    auto t1 = i1 + i2;
    auto t2 = i0 - i3;
    auto t3 = i1 - i2;
    i0 = t0 + t1;
    i2 = t0 - t1;
    auto t6 = t2 - t3;
    auto t7 = t3 * c2_8;
    i1 = t6 + t7;
    i3 = t6 - t7;
  }

  template <size_t SZ, class V>
  HWY_MAYBE_UNUSED static void ColumnDCT8(V& i0, V& i1, V& i2, V& i3, V& i4,
                                          V& i5, V& i6, V& i7) {
    const BlockDesc<SZ> d;

    const auto c1 = Set(d, 0.707106781186548f);  // 1 / sqrt(2)
    const auto c2 = Set(d, 0.382683432365090f);  // cos(3 * pi / 8)
    const auto c3 = Set(d, 1.30656296487638f);   // 1 / (2 * cos(3 * pi / 8))
    const auto c4 = Set(d, 0.541196100146197f);  // sqrt(2) * cos(3 * pi / 8)

    const auto t00 = i0 + i7;
    const auto t01 = i0 - i7;
    const auto t02 = i3 + i4;
    const auto t03 = i3 - i4;
    const auto t04 = i2 + i5;
    const auto t05 = i2 - i5;
    const auto t06 = i1 + i6;
    const auto t07 = i1 - i6;
    const auto t08 = t00 + t02;
    const auto t09 = t00 - t02;
    const auto t10 = t06 + t04;
    const auto t11 = t06 - t04;
    const auto t12 = t07 + t05;
    const auto t13 = t01 + t07;
    const auto t14 = t05 + t03;
    const auto t15 = t11 + t09;
    const auto t16 = t14 - t13;
    const auto t17 = c1 * t15;
    const auto t18 = c1 * t12;
    const auto t19 = c2 * t16;
    const auto t20 = t01 + t18;
    const auto t21 = t01 - t18;
    const auto t22 = MulAdd(c3, t13, t19);
    const auto t23 = MulAdd(c4, t14, t19);
    i0 = t08 + t10;
    i1 = t20 + t22;
    i2 = t09 + t17;
    i3 = t21 - t23;
    i4 = t08 - t10;
    i5 = t21 + t23;
    i6 = t09 - t17;
    i7 = t20 - t22;
  }

  // "A low multiplicative complexity fast recursive DCT-2 algorithm"
  // Maxim Vashkevich, Alexander Pertrovsky, 27 Jul 2012
  template <size_t SZ, class V>
  HWY_MAYBE_UNUSED static void ColumnDCT16(V& i00, V& i01, V& i02, V& i03,
                                           V& i04, V& i05, V& i06, V& i07,
                                           V& i08, V& i09, V& i10, V& i11,
                                           V& i12, V& i13, V& i14, V& i15) {
    const BlockDesc<SZ> d;

    const auto c1_16 = Set(d, 1.9615705608064609f);   // 2 * cos(1 * pi / 16)
    const auto c2_16 = Set(d, 1.8477590650225735f);   // 2 * cos(2 * pi / 16)
    const auto c3_16 = Set(d, 1.6629392246050905f);   // 2 * cos(3 * pi / 16)
    const auto c4_16 = Set(d, 1.4142135623730951f);   // 2 * cos(4 * pi / 16)
    const auto c5_16 = Set(d, 1.1111404660392046f);   // 2 * cos(5 * pi / 16)
    const auto c6_16 = Set(d, 0.7653668647301797f);   // 2 * cos(6 * pi / 16)
    const auto c7_16 = Set(d, 0.39018064403225666f);  // 2 * cos(7 * pi / 16)

    const auto t00 = i00 + i15;
    const auto t01 = i01 + i14;
    const auto t02 = i02 + i13;
    const auto t03 = i03 + i12;
    const auto t04 = i04 + i11;
    const auto t05 = i05 + i10;
    const auto t06 = i06 + i09;
    const auto t07 = i07 + i08;
    const auto t08 = i00 - i15;
    const auto t09 = i01 - i14;
    const auto t10 = i02 - i13;
    const auto t11 = i03 - i12;
    const auto t12 = i04 - i11;
    const auto t13 = i05 - i10;
    const auto t14 = i06 - i09;
    const auto t15 = i07 - i08;
    const auto t16 = t00 + t07;
    const auto t17 = t01 + t06;
    const auto t18 = t02 + t05;
    const auto t19 = t03 + t04;
    const auto t20 = t00 - t07;
    const auto t21 = t01 - t06;
    const auto t22 = t02 - t05;
    const auto t23 = t03 - t04;
    const auto t24 = t16 + t19;
    const auto t25 = t17 + t18;
    const auto t26 = t16 - t19;
    const auto t27 = t17 - t18;
    i00 = t24 + t25;
    i08 = t24 - t25;
    const auto t30 = t26 - t27;
    const auto t31 = t27 * c4_16;
    i04 = t30 + t31;
    i12 = t30 - t31;
    const auto t34 = t20 - t23;
    const auto t35 = t21 - t22;
    const auto t36 = t22 * c4_16;
    const auto t37 = t23 * c4_16;
    const auto t38 = t34 + t36;
    const auto t39 = t35 + t37;
    const auto t40 = t34 - t36;
    const auto t41 = t35 - t37;
    const auto t42 = t38 - t39;
    const auto t43 = t39 * c2_16;
    i02 = t42 + t43;
    i14 = t42 - t43;
    const auto t46 = t40 - t41;
    const auto t47 = t41 * c6_16;
    i06 = t46 + t47;
    i10 = t46 - t47;
    const auto t50 = t08 - t15;
    const auto t51 = t09 - t14;
    const auto t52 = t10 - t13;
    const auto t53 = t11 - t12;
    const auto t54 = t12 * c4_16;
    const auto t55 = t13 * c4_16;
    const auto t56 = t14 * c4_16;
    const auto t57 = t15 * c4_16;
    const auto t58 = t50 + t54;
    const auto t59 = t51 + t55;
    const auto t60 = t52 + t56;
    const auto t61 = t53 + t57;
    const auto t62 = t50 - t54;
    const auto t63 = t51 - t55;
    const auto t64 = t52 - t56;
    const auto t65 = t53 - t57;
    const auto t66 = t58 - t61;
    const auto t67 = t59 - t60;
    const auto t68 = t60 * c2_16;
    const auto t69 = t61 * c2_16;
    const auto t70 = t66 + t68;
    const auto t71 = t67 + t69;
    const auto t72 = t66 - t68;
    const auto t73 = t67 - t69;
    const auto t74 = t70 - t71;
    const auto t75 = t71 * c1_16;
    i01 = t74 + t75;
    i15 = t74 - t75;
    const auto t78 = t72 - t73;
    const auto t79 = t73 * c7_16;
    i07 = t78 + t79;
    i09 = t78 - t79;
    const auto t82 = t62 - t65;
    const auto t83 = t63 - t64;
    const auto t84 = t64 * c6_16;
    const auto t85 = t65 * c6_16;
    const auto t86 = t82 + t84;
    const auto t87 = t83 + t85;
    const auto t88 = t82 - t84;
    const auto t89 = t83 - t85;
    const auto t90 = t86 - t87;
    const auto t91 = t87 * c3_16;
    i03 = t90 + t91;
    i13 = t90 - t91;
    const auto t94 = t88 - t89;
    const auto t95 = t89 * c5_16;
    i05 = t94 + t95;
    i11 = t94 - t95;
  }

  template <size_t SZ, class V>
  HWY_MAYBE_UNUSED static void ColumnDCT32(
      V& i00, V& i01, V& i02, V& i03, V& i04, V& i05, V& i06, V& i07, V& i08,
      V& i09, V& i10, V& i11, V& i12, V& i13, V& i14, V& i15, V& i16, V& i17,
      V& i18, V& i19, V& i20, V& i21, V& i22, V& i23, V& i24, V& i25, V& i26,
      V& i27, V& i28, V& i29, V& i30, V& i31) {
    const BlockDesc<SZ> d;
    const auto c2_64 = Set(d, 1.990369453344393857f);   // 2 * cos(2 * pi / 64)
    const auto c4_64 = Set(d, 1.961570560806460861f);   // 2 * cos(4 * pi / 64)
    const auto c6_64 = Set(d, 1.913880671464417649f);   // 2 * cos(6 * pi / 64)
    const auto c8_64 = Set(d, 1.847759065022573477f);   // 2 * cos(8 * pi / 64)
    const auto c10_64 = Set(d, 1.763842528696710099f);  // 2 * cos(10 * pi / 64)
    const auto c12_64 = Set(d, 1.662939224605090471f);  // 2 * cos(12 * pi / 64)
    const auto c14_64 = Set(d, 1.546020906725473987f);  // 2 * cos(14 * pi / 64)
    const auto c16_64 = Set(d, 1.414213562373095145f);  // 2 * cos(16 * pi / 64)
    const auto c18_64 = Set(d, 1.268786568327290976f);  // 2 * cos(18 * pi / 64)
    const auto c20_64 = Set(d, 1.111140466039204577f);  // 2 * cos(20 * pi / 64)
    const auto c22_64 = Set(d, 0.942793473651995617f);  // 2 * cos(22 * pi / 64)
    const auto c24_64 = Set(d, 0.765366864730179675f);  // 2 * cos(24 * pi / 64)
    const auto c26_64 = Set(d, 0.580569354508924662f);  // 2 * cos(26 * pi / 64)
    const auto c28_64 = Set(d, 0.390180644032256663f);  // 2 * cos(28 * pi / 64)
    const auto c30_64 = Set(d, 0.196034280659121540f);  // 2 * cos(30 * pi / 64)

    const auto t00 = i00 + i31;
    const auto t01 = i01 + i30;
    const auto t02 = i02 + i29;
    const auto t03 = i03 + i28;
    const auto t04 = i04 + i27;
    const auto t05 = i05 + i26;
    const auto t06 = i06 + i25;
    const auto t07 = i07 + i24;
    const auto t08 = i08 + i23;
    const auto t09 = i09 + i22;
    const auto t10 = i10 + i21;
    const auto t11 = i11 + i20;
    const auto t12 = i12 + i19;
    const auto t13 = i13 + i18;
    const auto t14 = i14 + i17;
    const auto t15 = i15 + i16;
    const auto t16 = i00 - i31;
    const auto t17 = i01 - i30;
    const auto t18 = i02 - i29;
    const auto t19 = i03 - i28;
    const auto t20 = i04 - i27;
    const auto t21 = i05 - i26;
    const auto t22 = i06 - i25;
    const auto t23 = i07 - i24;
    const auto t24 = i08 - i23;
    const auto t25 = i09 - i22;
    const auto t26 = i10 - i21;
    const auto t27 = i11 - i20;
    const auto t28 = i12 - i19;
    const auto t29 = i13 - i18;
    const auto t30 = i14 - i17;
    const auto t31 = i15 - i16;
    const auto t32 = t00 + t15;
    const auto t33 = t01 + t14;
    const auto t34 = t02 + t13;
    const auto t35 = t03 + t12;
    const auto t36 = t04 + t11;
    const auto t37 = t05 + t10;
    const auto t38 = t06 + t09;
    const auto t39 = t07 + t08;
    const auto t40 = t00 - t15;
    const auto t41 = t01 - t14;
    const auto t42 = t02 - t13;
    const auto t43 = t03 - t12;
    const auto t44 = t04 - t11;
    const auto t45 = t05 - t10;
    const auto t46 = t06 - t09;
    const auto t47 = t07 - t08;
    const auto t48 = t32 + t39;
    const auto t49 = t33 + t38;
    const auto t50 = t34 + t37;
    const auto t51 = t35 + t36;
    const auto t52 = t32 - t39;
    const auto t53 = t33 - t38;
    const auto t54 = t34 - t37;
    const auto t55 = t35 - t36;
    const auto t56 = t48 + t51;
    const auto t57 = t49 + t50;
    const auto t58 = t48 - t51;
    const auto t59 = t49 - t50;
    const auto t60 = t56 + t57;
    const auto t61 = t56 - t57;
    const auto t62 = t58 - t59;
    const auto t63 = t59 * c16_64;
    const auto t64 = t62 + t63;
    const auto t65 = t62 - t63;
    const auto t66 = t52 - t55;
    const auto t67 = t53 - t54;
    const auto t68 = t54 * c16_64;
    const auto t69 = t55 * c16_64;
    const auto t70 = t66 + t68;
    const auto t71 = t67 + t69;
    const auto t72 = t66 - t68;
    const auto t73 = t67 - t69;
    const auto t74 = t70 - t71;
    const auto t75 = t71 * c8_64;
    const auto t76 = t74 + t75;
    const auto t77 = t74 - t75;
    const auto t78 = t72 - t73;
    const auto t79 = t73 * c24_64;
    const auto t80 = t78 + t79;
    const auto t81 = t78 - t79;
    const auto t82 = t40 - t47;
    const auto t83 = t41 - t46;
    const auto t84 = t42 - t45;
    const auto t85 = t43 - t44;
    const auto t86 = t44 * c16_64;
    const auto t87 = t45 * c16_64;
    const auto t88 = t46 * c16_64;
    const auto t89 = t47 * c16_64;
    const auto t90 = t82 + t86;
    const auto t91 = t83 + t87;
    const auto t92 = t84 + t88;
    const auto t93 = t85 + t89;
    const auto t94 = t82 - t86;
    const auto t95 = t83 - t87;
    const auto t96 = t84 - t88;
    const auto t97 = t85 - t89;
    const auto t98 = t90 - t93;
    const auto t99 = t91 - t92;
    const auto t100 = t92 * c8_64;
    const auto t101 = t93 * c8_64;
    const auto t102 = t98 + t100;
    const auto t103 = t99 + t101;
    const auto t104 = t98 - t100;
    const auto t105 = t99 - t101;
    const auto t106 = t102 - t103;
    const auto t107 = t103 * c4_64;
    const auto t108 = t106 + t107;
    const auto t109 = t106 - t107;
    const auto t110 = t104 - t105;
    const auto t111 = t105 * c28_64;
    const auto t112 = t110 + t111;
    const auto t113 = t110 - t111;
    const auto t114 = t94 - t97;
    const auto t115 = t95 - t96;
    const auto t116 = t96 * c24_64;
    const auto t117 = t97 * c24_64;
    const auto t118 = t114 + t116;
    const auto t119 = t115 + t117;
    const auto t120 = t114 - t116;
    const auto t121 = t115 - t117;
    const auto t122 = t118 - t119;
    const auto t123 = t119 * c12_64;
    const auto t124 = t122 + t123;
    const auto t125 = t122 - t123;
    const auto t126 = t120 - t121;
    const auto t127 = t121 * c20_64;
    const auto t128 = t126 + t127;
    const auto t129 = t126 - t127;
    const auto t130 = t16 - t31;
    const auto t131 = t17 - t30;
    const auto t132 = t18 - t29;
    const auto t133 = t19 - t28;
    const auto t134 = t20 - t27;
    const auto t135 = t21 - t26;
    const auto t136 = t22 - t25;
    const auto t137 = t23 - t24;
    const auto t138 = t24 * c16_64;
    const auto t139 = t25 * c16_64;
    const auto t140 = t26 * c16_64;
    const auto t141 = t27 * c16_64;
    const auto t142 = t28 * c16_64;
    const auto t143 = t29 * c16_64;
    const auto t144 = t30 * c16_64;
    const auto t145 = t31 * c16_64;
    const auto t146 = t130 + t138;
    const auto t147 = t131 + t139;
    const auto t148 = t132 + t140;
    const auto t149 = t133 + t141;
    const auto t150 = t134 + t142;
    const auto t151 = t135 + t143;
    const auto t152 = t136 + t144;
    const auto t153 = t137 + t145;
    const auto t154 = t130 - t138;
    const auto t155 = t131 - t139;
    const auto t156 = t132 - t140;
    const auto t157 = t133 - t141;
    const auto t158 = t134 - t142;
    const auto t159 = t135 - t143;
    const auto t160 = t136 - t144;
    const auto t161 = t137 - t145;
    const auto t162 = t146 - t153;
    const auto t163 = t147 - t152;
    const auto t164 = t148 - t151;
    const auto t165 = t149 - t150;
    const auto t166 = t150 * c8_64;
    const auto t167 = t151 * c8_64;
    const auto t168 = t152 * c8_64;
    const auto t169 = t153 * c8_64;
    const auto t170 = t162 + t166;
    const auto t171 = t163 + t167;
    const auto t172 = t164 + t168;
    const auto t173 = t165 + t169;
    const auto t174 = t162 - t166;
    const auto t175 = t163 - t167;
    const auto t176 = t164 - t168;
    const auto t177 = t165 - t169;
    const auto t178 = t170 - t173;
    const auto t179 = t171 - t172;
    const auto t180 = t172 * c4_64;
    const auto t181 = t173 * c4_64;
    const auto t182 = t178 + t180;
    const auto t183 = t179 + t181;
    const auto t184 = t178 - t180;
    const auto t185 = t179 - t181;
    const auto t186 = t182 - t183;
    const auto t187 = t183 * c2_64;
    const auto t188 = t186 + t187;
    const auto t189 = t186 - t187;
    const auto t190 = t184 - t185;
    const auto t191 = t185 * c30_64;
    const auto t192 = t190 + t191;
    const auto t193 = t190 - t191;
    const auto t194 = t174 - t177;
    const auto t195 = t175 - t176;
    const auto t196 = t176 * c28_64;
    const auto t197 = t177 * c28_64;
    const auto t198 = t194 + t196;
    const auto t199 = t195 + t197;
    const auto t200 = t194 - t196;
    const auto t201 = t195 - t197;
    const auto t202 = t198 - t199;
    const auto t203 = t199 * c14_64;
    const auto t204 = t202 + t203;
    const auto t205 = t202 - t203;
    const auto t206 = t200 - t201;
    const auto t207 = t201 * c18_64;
    const auto t208 = t206 + t207;
    const auto t209 = t206 - t207;
    const auto t210 = t154 - t161;
    const auto t211 = t155 - t160;
    const auto t212 = t156 - t159;
    const auto t213 = t157 - t158;
    const auto t214 = t158 * c24_64;
    const auto t215 = t159 * c24_64;
    const auto t216 = t160 * c24_64;
    const auto t217 = t161 * c24_64;
    const auto t218 = t210 + t214;
    const auto t219 = t211 + t215;
    const auto t220 = t212 + t216;
    const auto t221 = t213 + t217;
    const auto t222 = t210 - t214;
    const auto t223 = t211 - t215;
    const auto t224 = t212 - t216;
    const auto t225 = t213 - t217;
    const auto t226 = t218 - t221;
    const auto t227 = t219 - t220;
    const auto t228 = t220 * c12_64;
    const auto t229 = t221 * c12_64;
    const auto t230 = t226 + t228;
    const auto t231 = t227 + t229;
    const auto t232 = t226 - t228;
    const auto t233 = t227 - t229;
    const auto t234 = t230 - t231;
    const auto t235 = t231 * c6_64;
    const auto t236 = t234 + t235;
    const auto t237 = t234 - t235;
    const auto t238 = t232 - t233;
    const auto t239 = t233 * c26_64;
    const auto t240 = t238 + t239;
    const auto t241 = t238 - t239;
    const auto t242 = t222 - t225;
    const auto t243 = t223 - t224;
    const auto t244 = t224 * c20_64;
    const auto t245 = t225 * c20_64;
    const auto t246 = t242 + t244;
    const auto t247 = t243 + t245;
    const auto t248 = t242 - t244;
    const auto t249 = t243 - t245;
    const auto t250 = t246 - t247;
    const auto t251 = t247 * c10_64;
    const auto t252 = t250 + t251;
    const auto t253 = t250 - t251;
    const auto t254 = t248 - t249;
    const auto t255 = t249 * c22_64;
    const auto t256 = t254 + t255;
    const auto t257 = t254 - t255;

    i00 = t60;
    i01 = t188;
    i02 = t108;
    i03 = t236;
    i04 = t76;
    i05 = t252;
    i06 = t124;
    i07 = t204;
    i08 = t64;
    i09 = t208;
    i10 = t128;
    i11 = t256;
    i12 = t80;
    i13 = t240;
    i14 = t112;
    i15 = t192;
    i16 = t61;
    i17 = t193;
    i18 = t113;
    i19 = t241;
    i20 = t81;
    i21 = t257;
    i22 = t129;
    i23 = t209;
    i24 = t65;
    i25 = t205;
    i26 = t125;
    i27 = t253;
    i28 = t77;
    i29 = t237;
    i30 = t109;
    i31 = t189;
  }
};  // VectorDCT

// Call VectorDCT::ColumnDCT* after loading rows.

template <class From, class To, size_t COLS = 4>
HWY_MAYBE_UNUSED void ColumnDCT(DCTSizeTag<4>, const From& from, const To& to) {
  const BlockDesc<4> d;
  constexpr size_t kSize = HWY_MIN(COLS, MaxLanes(d));
  for (size_t i = 0; i < COLS; i += Lanes(d)) {
    auto i0 = from.LoadPart(BlockDesc<kSize>(), 0, i);
    auto i1 = from.LoadPart(BlockDesc<kSize>(), 1, i);
    auto i2 = from.LoadPart(BlockDesc<kSize>(), 2, i);
    auto i3 = from.LoadPart(BlockDesc<kSize>(), 3, i);
    VectorDCT::ColumnDCT4<kSize>(i0, i1, i2, i3);
    to.StorePart(BlockDesc<kSize>(), i0, 0, i);
    to.StorePart(BlockDesc<kSize>(), i1, 1, i);
    to.StorePart(BlockDesc<kSize>(), i2, 2, i);
    to.StorePart(BlockDesc<kSize>(), i3, 3, i);
  }
}

template <class From, class To, size_t COLS = 8>
HWY_MAYBE_UNUSED void ColumnDCT(DCTSizeTag<8>, const From& from, const To& to) {
  const BlockDesc<8> d;
  constexpr size_t kSize = HWY_MIN(COLS, MaxLanes(d));

  for (size_t i = 0; i < COLS; i += Lanes(d)) {
    auto i0 = from.LoadPart(BlockDesc<kSize>(), 0, i);
    auto i1 = from.LoadPart(BlockDesc<kSize>(), 1, i);
    auto i2 = from.LoadPart(BlockDesc<kSize>(), 2, i);
    auto i3 = from.LoadPart(BlockDesc<kSize>(), 3, i);
    auto i4 = from.LoadPart(BlockDesc<kSize>(), 4, i);
    auto i5 = from.LoadPart(BlockDesc<kSize>(), 5, i);
    auto i6 = from.LoadPart(BlockDesc<kSize>(), 6, i);
    auto i7 = from.LoadPart(BlockDesc<kSize>(), 7, i);
    VectorDCT::ColumnDCT8<kSize>(i0, i1, i2, i3, i4, i5, i6, i7);
    to.StorePart(BlockDesc<kSize>(), i0, 0, i);
    to.StorePart(BlockDesc<kSize>(), i1, 1, i);
    to.StorePart(BlockDesc<kSize>(), i2, 2, i);
    to.StorePart(BlockDesc<kSize>(), i3, 3, i);
    to.StorePart(BlockDesc<kSize>(), i4, 4, i);
    to.StorePart(BlockDesc<kSize>(), i5, 5, i);
    to.StorePart(BlockDesc<kSize>(), i6, 6, i);
    to.StorePart(BlockDesc<kSize>(), i7, 7, i);
  }
}

template <class From, class To, size_t COLS = 16>
HWY_MAYBE_UNUSED void ColumnDCT(DCTSizeTag<16>, const From& from,
                                const To& to) {
  const BlockDesc<COLS> d;

  for (size_t i = 0; i < COLS; i += Lanes(d)) {
    auto i00 = from.LoadPart(d, 0, i);
    auto i01 = from.LoadPart(d, 1, i);
    auto i02 = from.LoadPart(d, 2, i);
    auto i03 = from.LoadPart(d, 3, i);
    auto i04 = from.LoadPart(d, 4, i);
    auto i05 = from.LoadPart(d, 5, i);
    auto i06 = from.LoadPart(d, 6, i);
    auto i07 = from.LoadPart(d, 7, i);
    auto i08 = from.LoadPart(d, 8, i);
    auto i09 = from.LoadPart(d, 9, i);
    auto i10 = from.LoadPart(d, 10, i);
    auto i11 = from.LoadPart(d, 11, i);
    auto i12 = from.LoadPart(d, 12, i);
    auto i13 = from.LoadPart(d, 13, i);
    auto i14 = from.LoadPart(d, 14, i);
    auto i15 = from.LoadPart(d, 15, i);
    VectorDCT::ColumnDCT16<MaxLanes(d)>(i00, i01, i02, i03, i04, i05, i06, i07,
                                        i08, i09, i10, i11, i12, i13, i14, i15);
    to.StorePart(d, i00, 0, i);
    to.StorePart(d, i01, 1, i);
    to.StorePart(d, i02, 2, i);
    to.StorePart(d, i03, 3, i);
    to.StorePart(d, i04, 4, i);
    to.StorePart(d, i05, 5, i);
    to.StorePart(d, i06, 6, i);
    to.StorePart(d, i07, 7, i);
    to.StorePart(d, i08, 8, i);
    to.StorePart(d, i09, 9, i);
    to.StorePart(d, i10, 10, i);
    to.StorePart(d, i11, 11, i);
    to.StorePart(d, i12, 12, i);
    to.StorePart(d, i13, 13, i);
    to.StorePart(d, i14, 14, i);
    to.StorePart(d, i15, 15, i);
  }
}

template <class From, class To, size_t COLS = 32>
HWY_MAYBE_UNUSED void ColumnDCT(DCTSizeTag<32>, const From& from,
                                const To& to) {
  const BlockDesc<COLS> d;

  for (size_t i = 0; i < COLS; i += Lanes(d)) {
    auto i00 = from.LoadPart(d, 0, i);
    auto i01 = from.LoadPart(d, 1, i);
    auto i02 = from.LoadPart(d, 2, i);
    auto i03 = from.LoadPart(d, 3, i);
    auto i04 = from.LoadPart(d, 4, i);
    auto i05 = from.LoadPart(d, 5, i);
    auto i06 = from.LoadPart(d, 6, i);
    auto i07 = from.LoadPart(d, 7, i);
    auto i08 = from.LoadPart(d, 8, i);
    auto i09 = from.LoadPart(d, 9, i);
    auto i10 = from.LoadPart(d, 10, i);
    auto i11 = from.LoadPart(d, 11, i);
    auto i12 = from.LoadPart(d, 12, i);
    auto i13 = from.LoadPart(d, 13, i);
    auto i14 = from.LoadPart(d, 14, i);
    auto i15 = from.LoadPart(d, 15, i);
    auto i16 = from.LoadPart(d, 16, i);
    auto i17 = from.LoadPart(d, 17, i);
    auto i18 = from.LoadPart(d, 18, i);
    auto i19 = from.LoadPart(d, 19, i);
    auto i20 = from.LoadPart(d, 20, i);
    auto i21 = from.LoadPart(d, 21, i);
    auto i22 = from.LoadPart(d, 22, i);
    auto i23 = from.LoadPart(d, 23, i);
    auto i24 = from.LoadPart(d, 24, i);
    auto i25 = from.LoadPart(d, 25, i);
    auto i26 = from.LoadPart(d, 26, i);
    auto i27 = from.LoadPart(d, 27, i);
    auto i28 = from.LoadPart(d, 28, i);
    auto i29 = from.LoadPart(d, 29, i);
    auto i30 = from.LoadPart(d, 30, i);
    auto i31 = from.LoadPart(d, 31, i);
    VectorDCT::ColumnDCT32<MaxLanes(d)>(i00, i01, i02, i03, i04, i05, i06, i07,
                                        i08, i09, i10, i11, i12, i13, i14, i15,
                                        i16, i17, i18, i19, i20, i21, i22, i23,
                                        i24, i25, i26, i27, i28, i29, i30, i31);
    to.StorePart(d, i00, 0, i);
    to.StorePart(d, i01, 1, i);
    to.StorePart(d, i02, 2, i);
    to.StorePart(d, i03, 3, i);
    to.StorePart(d, i04, 4, i);
    to.StorePart(d, i05, 5, i);
    to.StorePart(d, i06, 6, i);
    to.StorePart(d, i07, 7, i);
    to.StorePart(d, i08, 8, i);
    to.StorePart(d, i09, 9, i);
    to.StorePart(d, i10, 10, i);
    to.StorePart(d, i11, 11, i);
    to.StorePart(d, i12, 12, i);
    to.StorePart(d, i13, 13, i);
    to.StorePart(d, i14, 14, i);
    to.StorePart(d, i15, 15, i);
    to.StorePart(d, i16, 16, i);
    to.StorePart(d, i17, 17, i);
    to.StorePart(d, i18, 18, i);
    to.StorePart(d, i19, 19, i);
    to.StorePart(d, i20, 20, i);
    to.StorePart(d, i21, 21, i);
    to.StorePart(d, i22, 22, i);
    to.StorePart(d, i23, 23, i);
    to.StorePart(d, i24, 24, i);
    to.StorePart(d, i25, 25, i);
    to.StorePart(d, i26, 26, i);
    to.StorePart(d, i27, 27, i);
    to.StorePart(d, i28, 28, i);
    to.StorePart(d, i29, 29, i);
    to.StorePart(d, i30, 30, i);
    to.StorePart(d, i31, 31, i);
  }
}

// Special case for 8-lane SIMD - combines DCT and transpose.
#if HWY_CAP_GE256

template <class From, class To>
HWY_MAYBE_UNUSED void ComputeTransposedScaledDCT8_V8(const From& from,
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

  auto i0 = from.LoadPart(BlockDesc<8>(), 0, 0);
  auto i7 = from.LoadPart(BlockDesc<8>(), 7, 0);
  auto t00 = i0 + i7;                // 2 (faster than Add)
  auto t01 = NegMulAdd(i7, k1, i0);  // 4
  HWY_FENCE;

  auto i3 = from.LoadPart(BlockDesc<8>(), 3, 0);
  auto i4 = from.LoadPart(BlockDesc<8>(), 4, 0);
  auto t02 = i3 + i4;
  auto t03 = NegMulAdd(i4, k1, i3);  // 1
  HWY_FENCE;

  auto i2 = from.LoadPart(BlockDesc<8>(), 2, 0);
  auto i5 = from.LoadPart(BlockDesc<8>(), 5, 0);
  auto t04 = i2 + i5;  // 1
  auto t05 = NegMulAdd(i5, k1, i2);
  HWY_FENCE;

  auto i1 = from.LoadPart(BlockDesc<8>(), 1, 0);
  auto i6 = from.LoadPart(BlockDesc<8>(), 6, 0);
  auto t06 = i1 + i6;  // !
  HWY_FENCE;

  auto t07 = i1 - i6;
  auto t09 = NegMulAdd(t02, k1, t00);
  const auto c4 = Broadcast<3>(c1234);

  auto t11 = t06 - t04;             // !
  auto t08 = MulAdd(t00, k1, t02);  // 2
  const auto c3 = Broadcast<2>(c1234);

  auto t14 = t05 + t03;
  auto t10 = MulAdd(t06, k1, t04);  // 1; dep-1

  auto t13 = t01 + t07;  // limits odd d
  const auto c1 = Broadcast<0>(c1234);

  auto t15 = t11 + t09;  // !
  const auto c2 = Broadcast<1>(c1234);

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

  const auto q0 = InterleaveLower(d0, d2);

  const auto q2 = InterleaveUpper(d0, d2);

  const auto q4 = InterleaveLower(d4, d6);

  auto d3 = t21 - t23;
  const auto q6 = InterleaveUpper(d4, d6);

  auto d1 = t20 + t22;
  const auto q1 = InterleaveLower(d1, d3);

  const auto r0 = InterleaveLower(q0, q1);
  const auto r1 = InterleaveUpper(q0, q1);

  auto d7 = t20 - t22;
  const auto q3 = InterleaveUpper(d1, d3);
  const auto r2 = InterleaveLower(q2, q3);
  const auto r3 = InterleaveUpper(q2, q3);

  auto d5 = t21 + t23;
  const auto q5 = InterleaveLower(d5, d7);
  const auto r4 = InterleaveLower(q4, q5);
  const auto r5 = InterleaveUpper(q4, q5);

  const auto q7 = InterleaveUpper(d5, d7);
  const auto r6 = InterleaveLower(q6, q7);
  const auto r7 = InterleaveUpper(q6, q7);

  // Second column-DCT after transpose
  i0 = ConcatLowerLower(r4, r0);
  i7 = ConcatUpperUpper(r7, r3);
  t01 = i0 - i7;             // 1
  t00 = MulAdd(i0, k1, i7);  // 2

  i1 = ConcatLowerLower(r5, r1);
  i6 = ConcatUpperUpper(r6, r2);
  t07 = i1 - i6;             // !
  t06 = MulAdd(i1, k1, i6);  // 2

  i3 = ConcatLowerLower(r7, r3);
  i4 = ConcatUpperUpper(r4, r0);
  t03 = i3 - i4;             // 1
  t02 = MulAdd(i3, k1, i4);  // !

  i2 = ConcatLowerLower(r6, r2);
  i5 = ConcatUpperUpper(r5, r1);
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
  to.StorePart(BlockDesc<8>(), i0, 0, 0);
  HWY_FENCE;

  i1 = t20 + t22;

  i7 = t20 - t22;
  to.StorePart(BlockDesc<8>(), i2, 2, 0);
  to.StorePart(BlockDesc<8>(), i4, 4, 0);
  HWY_FENCE;

  i3 = t21 - t23;
  to.StorePart(BlockDesc<8>(), i1, 1, 0);
  HWY_FENCE;

  i5 = t21 + t23;
  to.StorePart(BlockDesc<8>(), i6, 6, 0);
  to.StorePart(BlockDesc<8>(), i7, 7, 0);
  to.StorePart(BlockDesc<8>(), i3, 3, 0);
  to.StorePart(BlockDesc<8>(), i5, 5, 0);
}

#endif  // HWY_CAP_GE256

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
    ColumnDCT(DCTSizeTag<N>(), from, ToBlock(N, N, block));
    Transpose<N, N>::Run(FromBlock(N, N, block),
                         ToBlock(N, N, transposed_block));
    ColumnDCT(DCTSizeTag<N>(), FromBlock(N, N, transposed_block), to);
  }
};

#if HWY_CAP_GE256
template <>
struct ComputeTransposedScaledDCT<8> {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    return ComputeTransposedScaledDCT8_V8(from, to);
  }
};
#endif

// Special case for 2x2 DCT.
template <>
struct ComputeTransposedScaledDCT<2> {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
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
struct ComputeScaledDCT {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[ROWS * COLS];
    HWY_ALIGN float transposed_block[ROWS * COLS];
    ColumnDCT<From, ToBlock, COLS>(DCTSizeTag<ROWS>(), from,
                                   ToBlock(ROWS, COLS, block));
    Transpose<ROWS, COLS>::Run(FromBlock(ROWS, COLS, block),
                               ToBlock(COLS, ROWS, transposed_block));
    // Reusing block to reduce stack usage.
    if (ROWS < COLS) {
      ColumnDCT<FromBlock, ToBlock, ROWS>(
          DCTSizeTag<COLS>(), FromBlock(COLS, ROWS, transposed_block),
          ToBlock(COLS, ROWS, block));
      Transpose<COLS, ROWS>::Run(FromBlock(COLS, ROWS, block), to);
    } else {
      ColumnDCT<FromBlock, To, ROWS>(
          DCTSizeTag<COLS>(), FromBlock(COLS, ROWS, transposed_block), to);
    }
  }
};

// Special cases for ROWS (or COLS) <= 2.
template <>
struct ComputeScaledDCT<4, 2> {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 2];
    ColumnDCT<From, ToBlock, /*COLS=*/2>(DCTSizeTag<4>(), from,
                                         ToBlock(4, 2, block));
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
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    // `block` and `coeffs` are transposed.
    HWY_ALIGN float block[4 * 2];
    for (size_t y = 0; y < 4; ++y) {
      const float a0 = from.Read(0, y);
      const float a1 = from.Read(1, y);
      block[2 * y] = a0 + a1;
      block[2 * y + 1] = a0 - a1;
    }
    HWY_ALIGN float coeffs[4 * 2];
    ColumnDCT<FromBlock, ToBlock, /*COLS=*/2>(
        DCTSizeTag<4>(), FromBlock(4, 2, block), ToBlock(4, 2, coeffs));
    Transpose<4, 2>::Run(FromBlock(4, 2, coeffs), to);
  }
};

template <>
struct ComputeScaledDCT<4, 1> {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    HWY_ALIGN float block[4 * 1];
    ColumnDCT<From, ToBlock, /*COLS=*/1>(DCTSizeTag<4>(), from,
                                         ToBlock(4, 1, block));
    Transpose<4, 1>::Run(FromBlock(4, 1, block), to);
  }
};

template <>
struct ComputeScaledDCT<1, 4> {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    HWY_ALIGN float coeffs[4 * 1];
    Transpose<1, 4>::Run(from, ToBlock(4, 1, coeffs));
    HWY_ALIGN float block[4 * 1];
    ColumnDCT<FromBlock, ToBlock, /*COLS=*/1>(
        DCTSizeTag<4>(), FromBlock(4, 1, coeffs), ToBlock(4, 1, block));
    Transpose<4, 1>::Run(FromBlock(4, 1, block), to);
  }
};

template <>
struct ComputeScaledDCT<2, 1> {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    const float a0 = from.Read(0, 0);
    const float a1 = from.Read(1, 0);
    to.Write(a0 + a1, 0, 0);
    to.Write(a0 - a1, 0, 1);
  }
};

template <>
struct ComputeScaledDCT<1, 2> {
  template <class From, class To>
  HWY_MAYBE_UNUSED void operator()(const From& from, const To& to) {
    const float a0 = from.Read(0, 0);
    const float a1 = from.Read(0, 1);
    to.Write(a0 + a1, 0, 0);
    to.Write(a0 - a1, 0, 1);
  }
};

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#endif  // include guard
