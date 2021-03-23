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

// Defined by build system; this avoids IDE warnings. Must come before
// color_management.h (affects header definitions).
#ifndef JPEGXL_ENABLE_SKCMS
#define JPEGXL_ENABLE_SKCMS 0
#endif

#include "lib/jxl/color_management.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/color_management.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/linalg.h"
#include "lib/jxl/transfer_functions-inl.h"
#if JPEGXL_ENABLE_SKCMS
#include "skcms.h"
#else  // JPEGXL_ENABLE_SKCMS
#include "lcms2.h"
#include "lcms2_plugin.h"
#endif  // JPEGXL_ENABLE_SKCMS

#define JXL_CMS_VERBOSE 0

// Define these only once. We can't use HWY_ONCE here because it is defined as
// 1 only on the last pass.
#ifndef LIB_JXL_COLOR_MANAGEMENT_CC_
#define LIB_JXL_COLOR_MANAGEMENT_CC_

namespace jxl {
#if JPEGXL_ENABLE_SKCMS
struct ColorSpaceTransform::SkcmsICC {
  // Parsed skcms_ICCProfiles retain pointers to the original data.
  PaddedBytes icc_src_, icc_dst_;
  skcms_ICCProfile profile_src_, profile_dst_;
};
#endif  // JPEGXL_ENABLE_SKCMS
}  // namespace jxl

#endif  // LIB_JXL_COLOR_MANAGEMENT_CC_

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

#if JXL_CMS_VERBOSE >= 2
const size_t kX = 0;  // pixel index, multiplied by 3 for RGB
#endif

// xform_src = UndoGammaCompression(buf_src).
void BeforeTransform(ColorSpaceTransform* t, const float* buf_src,
                     float* xform_src) {
  switch (t->preprocess_) {
    case ExtraTF::kNone:
      JXL_DASSERT(false);  // unreachable
      break;

    case ExtraTF::kPQ: {
      // By default, PQ content has an intensity target of 10000, stored
      // exactly.
      HWY_FULL(float) df;
      const auto multiplier = Set(df, t->intensity_target_ == 10000.f
                                          ? 1.0f
                                          : 10000.f / t->intensity_target_);
      for (size_t i = 0; i < t->buf_src_.xsize(); i += Lanes(df)) {
        const auto val = Load(df, buf_src + i);
        const auto result = multiplier * TF_PQ().DisplayFromEncoded(df, val);
        Store(result, df, xform_src + i);
      }
#if JXL_CMS_VERBOSE >= 2
      printf("pre in %.4f %.4f %.4f undoPQ %.4f %.4f %.4f\n", buf_src[3 * kX],
             buf_src[3 * kX + 1], buf_src[3 * kX + 2], xform_src[3 * kX],
             xform_src[3 * kX + 1], xform_src[3 * kX + 2]);
#endif
      break;
    }

    case ExtraTF::kHLG:
      for (size_t i = 0; i < t->buf_src_.xsize(); ++i) {
        xform_src[i] = static_cast<float>(
            TF_HLG().DisplayFromEncoded(static_cast<double>(buf_src[i])));
      }
#if JXL_CMS_VERBOSE >= 2
      printf("pre in %.4f %.4f %.4f undoHLG %.4f %.4f %.4f\n", buf_src[3 * kX],
             buf_src[3 * kX + 1], buf_src[3 * kX + 2], xform_src[3 * kX],
             xform_src[3 * kX + 1], xform_src[3 * kX + 2]);
#endif
      break;

    case ExtraTF::kSRGB:
      HWY_FULL(float) df;
      for (size_t i = 0; i < t->buf_src_.xsize(); i += Lanes(df)) {
        const auto val = Load(df, buf_src + i);
        const auto result = TF_SRGB().DisplayFromEncoded(val);
        Store(result, df, xform_src + i);
      }
#if JXL_CMS_VERBOSE >= 2
      printf("pre in %.4f %.4f %.4f undoSRGB %.4f %.4f %.4f\n", buf_src[3 * kX],
             buf_src[3 * kX + 1], buf_src[3 * kX + 2], xform_src[3 * kX],
             xform_src[3 * kX + 1], xform_src[3 * kX + 2]);
#endif
      break;
  }
}

// Applies gamma compression in-place.
void AfterTransform(ColorSpaceTransform* t, float* JXL_RESTRICT buf_dst) {
  switch (t->postprocess_) {
    case ExtraTF::kNone:
      JXL_DASSERT(false);  // unreachable
      break;
    case ExtraTF::kPQ: {
      HWY_FULL(float) df;
      const auto multiplier = Set(df, t->intensity_target_ == 10000.f
                                          ? 1.0f
                                          : t->intensity_target_ * 1e-4f);
      for (size_t i = 0; i < t->buf_dst_.xsize(); i += Lanes(df)) {
        const auto val = Load(df, buf_dst + i);
        const auto result = TF_PQ().EncodedFromDisplay(df, multiplier * val);
        Store(result, df, buf_dst + i);
      }
#if JXL_CMS_VERBOSE >= 2
      printf("after PQ enc %.4f %.4f %.4f\n", buf_dst[3 * kX],
             buf_dst[3 * kX + 1], buf_dst[3 * kX + 2]);
#endif
      break;
    }
    case ExtraTF::kHLG:
      for (size_t i = 0; i < t->buf_dst_.xsize(); ++i) {
        buf_dst[i] = static_cast<float>(
            TF_HLG().EncodedFromDisplay(static_cast<double>(buf_dst[i])));
      }
#if JXL_CMS_VERBOSE >= 2
      printf("after HLG enc %.4f %.4f %.4f\n", buf_dst[3 * kX],
             buf_dst[3 * kX + 1], buf_dst[3 * kX + 2]);
#endif
      break;
    case ExtraTF::kSRGB:
      HWY_FULL(float) df;
      for (size_t i = 0; i < t->buf_dst_.xsize(); i += Lanes(df)) {
        const auto val = Load(df, buf_dst + i);
        const auto result =
            TF_SRGB().EncodedFromDisplay(HWY_FULL(float)(), val);
        Store(result, df, buf_dst + i);
      }
#if JXL_CMS_VERBOSE >= 2
      printf("after SRGB enc %.4f %.4f %.4f\n", buf_dst[3 * kX],
             buf_dst[3 * kX + 1], buf_dst[3 * kX + 2]);
#endif
      break;
  }
}

void DoColorSpaceTransform(ColorSpaceTransform* t, const size_t thread,
                           const float* buf_src, float* buf_dst) {
  // No lock needed.

  float* xform_src = const_cast<float*>(buf_src);  // Read-only.
  if (t->preprocess_ != ExtraTF::kNone) {
    xform_src = t->buf_src_.Row(thread);  // Writable buffer.
    BeforeTransform(t, buf_src, xform_src);
  }

#if JXL_CMS_VERBOSE >= 2
  // Save inputs for printing before in-place transforms overwrite them.
  const float in0 = xform_src[3 * kX + 0];
  const float in1 = xform_src[3 * kX + 1];
  const float in2 = xform_src[3 * kX + 2];
#endif

  if (t->skip_lcms_) {
    if (buf_dst != xform_src) {
      memcpy(buf_dst, xform_src, t->buf_dst_.xsize() * sizeof(*buf_dst));
    }  // else: in-place, no need to copy
  } else {
#if JPEGXL_ENABLE_SKCMS
    JXL_CHECK(skcms_Transform(
        xform_src, skcms_PixelFormat_RGB_fff, skcms_AlphaFormat_Opaque,
        &t->skcms_icc_->profile_src_, buf_dst, skcms_PixelFormat_RGB_fff,
        skcms_AlphaFormat_Opaque, &t->skcms_icc_->profile_dst_, t->xsize_));
#else   // JPEGXL_ENABLE_SKCMS
    JXL_DASSERT(thread < t->transforms_.size());
    cmsHTRANSFORM xform = t->transforms_[thread];
    cmsDoTransform(xform, xform_src, buf_dst,
                   static_cast<cmsUInt32Number>(t->xsize_));
#endif  // JPEGXL_ENABLE_SKCMS
  }
#if JXL_CMS_VERBOSE >= 2
  printf("xform skip%d: %.4f %.4f %.4f (%p) -> (%p) %.4f %.4f %.4f\n",
         t->skip_lcms_, in0, in1, in2, xform_src, buf_dst, buf_dst[3 * kX],
         buf_dst[3 * kX + 1], buf_dst[3 * kX + 2]);
#endif

  if (t->postprocess_ != ExtraTF::kNone) {
    AfterTransform(t, buf_dst);
  }
}

// NOTE: this is only used to provide a reasonable ICC profile that other
// software can read. Our own transforms use ExtraTF instead because that is
// more precise and supports unbounded mode.
std::vector<uint16_t> CreateTableCurve(uint32_t N, const ExtraTF tf) {
  JXL_ASSERT(N <= 4096);  // ICC MFT2 only allows 4K entries
  JXL_ASSERT(tf == ExtraTF::kPQ || tf == ExtraTF::kHLG);
  // No point using float - LCMS converts to 16-bit for A2B/MFT.
  std::vector<uint16_t> table(N);
  for (uint32_t i = 0; i < N; ++i) {
    const float x = static_cast<float>(i) / (N - 1);  // 1.0 at index N - 1.
    const double dx = static_cast<double>(x);
    // LCMS requires EOTF (e.g. 2.4 exponent).
    double y = (tf == ExtraTF::kHLG) ? TF_HLG().DisplayFromEncoded(dx)
                                     : TF_PQ().DisplayFromEncoded(dx);
    JXL_ASSERT(y >= 0.0);
    // Clamp to table range - necessary for HLG.
    if (y > 1.0) y = 1.0;
    // 1.0 corresponds to table value 0xFFFF.
    table[i] = static_cast<uint16_t>(std::round(y * 65535.0));
  }
  return table;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(DoColorSpaceTransform);
void DoColorSpaceTransform(ColorSpaceTransform* t, size_t thread,
                           const float* buf_src, float* buf_dst) {
  return HWY_DYNAMIC_DISPATCH(DoColorSpaceTransform)(t, thread, buf_src,
                                                     buf_dst);
}

HWY_EXPORT(CreateTableCurve);  // Local function.

namespace {

// Define to 1 on OS X as a workaround for older LCMS lacking MD5.
#define JXL_CMS_OLD_VERSION 0

// cms functions (even *THR) are not thread-safe, except cmsDoTransform.
// To ensure all functions are covered without frequent lock-taking nor risk of
// recursive lock, we lock in the top-level APIs.
static std::mutex& LcmsMutex() {
  static std::mutex m;
  return m;
}

Status CIEXYZFromWhiteCIExy(const CIExy& xy, float XYZ[3]) {
  // Target Y = 1.
  if (std::abs(xy.y) < 1e-12) return JXL_FAILURE("Y value is too small");
  const float factor = 1 / xy.y;
  XYZ[0] = xy.x * factor;
  XYZ[1] = 1;
  XYZ[2] = (1 - xy.x - xy.y) * factor;
  return true;
}

#if JPEGXL_ENABLE_SKCMS
JXL_MUST_USE_RESULT CIExy CIExyFromXYZ(const float XYZ[3]) {
  const float factor = 1.f / (XYZ[0] + XYZ[1] + XYZ[2]);
  CIExy xy;
  xy.x = XYZ[0] * factor;
  xy.y = XYZ[1] * factor;
  return xy;
}

#else  // JPEGXL_ENABLE_SKCMS
// (LCMS interface requires xyY but we omit the Y for white points/primaries.)

JXL_MUST_USE_RESULT CIExy CIExyFromxyY(const cmsCIExyY& xyY) {
  CIExy xy;
  xy.x = xyY.x;
  xy.y = xyY.y;
  return xy;
}

JXL_MUST_USE_RESULT CIExy CIExyFromXYZ(const cmsCIEXYZ& XYZ) {
  cmsCIExyY xyY;
  cmsXYZ2xyY(/*Dest=*/&xyY, /*Source=*/&XYZ);
  return CIExyFromxyY(xyY);
}

JXL_MUST_USE_RESULT cmsCIEXYZ D50_XYZ() {
  // Quantized D50 as stored in ICC profiles.
  return {0.96420288, 1.0, 0.82490540};
}

JXL_MUST_USE_RESULT cmsCIExyY xyYFromCIExy(const CIExy& xy) {
  const cmsCIExyY xyY = {xy.x, xy.y, 1.0};
  return xyY;
}

// RAII

struct ProfileDeleter {
  void operator()(void* p) { cmsCloseProfile(p); }
};
using Profile = std::unique_ptr<void, ProfileDeleter>;

struct TransformDeleter {
  void operator()(void* p) { cmsDeleteTransform(p); }
};
using Transform = std::unique_ptr<void, TransformDeleter>;

struct CurveDeleter {
  void operator()(cmsToneCurve* p) { cmsFreeToneCurve(p); }
};
using Curve = std::unique_ptr<cmsToneCurve, CurveDeleter>;

Status CreateProfileXYZ(const cmsContext context,
                        Profile* JXL_RESTRICT profile) {
  profile->reset(cmsCreateXYZProfileTHR(context));
  if (profile->get() == nullptr) return JXL_FAILURE("Failed to create XYZ");
  return true;
}


#endif  // !JPEGXL_ENABLE_SKCMS

#if JPEGXL_ENABLE_SKCMS
// IMPORTANT: icc must outlive profile.
Status DecodeProfile(const PaddedBytes& icc, skcms_ICCProfile* const profile) {
  if (!skcms_Parse(icc.data(), icc.size(), profile)) {
    return JXL_FAILURE("Failed to parse ICC profile with %zu bytes",
                       icc.size());
  }
  return true;
}
#else   // JPEGXL_ENABLE_SKCMS
Status DecodeProfile(const cmsContext context, const PaddedBytes& icc,
                     Profile* profile) {
  profile->reset(cmsOpenProfileFromMemTHR(context, icc.data(), icc.size()));
  if (profile->get() == nullptr) {
    return JXL_FAILURE("Failed to decode profile");
  }

  // WARNING: due to the LCMS MD5 issue mentioned above, many existing
  // profiles have incorrect MD5, so do not even bother checking them nor
  // generating warning clutter.

  return true;
}
#endif  // JPEGXL_ENABLE_SKCMS

void ICCComputeMD5(const PaddedBytes& data, uint8_t sum[16]) {
  PaddedBytes data64 = data;
  data64.push_back(128);
  // Add bytes such that ((size + 8) & 63) == 0.
  size_t extra = ((64 - ((data64.size() + 8) & 63)) & 63);
  data64.resize(data64.size() + extra, 0);
  for (uint64_t i = 0; i < 64; i += 8) {
    data64.push_back(static_cast<uint64_t>(data.size() << 3u) >> i);
  }

  static const uint32_t sineparts[64] = {
      0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
      0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
      0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
      0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
      0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
      0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
      0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
      0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
      0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
      0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
      0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
  };
  static const uint32_t shift[64] = {
      7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
      5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20, 5, 9,  14, 20,
      4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
      6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
  };

  uint32_t a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;

  for (size_t i = 0; i < data64.size(); i += 64) {
    uint32_t a = a0, b = b0, c = c0, d = d0, f, g;
    for (size_t j = 0; j < 64; j++) {
      if (j < 16) {
        f = (b & c) | ((~b) & d);
        g = j;
      } else if (j < 32) {
        f = (d & b) | ((~d) & c);
        g = (5 * j + 1) & 0xf;
      } else if (j < 48) {
        f = b ^ c ^ d;
        g = (3 * j + 5) & 0xf;
      } else {
        f = c ^ (b | (~d));
        g = (7 * j) & 0xf;
      }
      uint32_t dg0 = data64[i + g * 4 + 0], dg1 = data64[i + g * 4 + 1],
               dg2 = data64[i + g * 4 + 2], dg3 = data64[i + g * 4 + 3];
      uint32_t u = dg0 | (dg1 << 8u) | (dg2 << 16u) | (dg3 << 24u);
      f += a + sineparts[j] + u;
      a = d;
      d = c;
      c = b;
      b += (f << shift[j]) | (f >> (32u - shift[j]));
    }
    a0 += a;
    b0 += b;
    c0 += c;
    d0 += d;
  }
  sum[0] = a0;
  sum[1] = a0 >> 8u;
  sum[2] = a0 >> 16u;
  sum[3] = a0 >> 24u;
  sum[4] = b0;
  sum[5] = b0 >> 8u;
  sum[6] = b0 >> 16u;
  sum[7] = b0 >> 24u;
  sum[8] = c0;
  sum[9] = c0 >> 8u;
  sum[10] = c0 >> 16u;
  sum[11] = c0 >> 24u;
  sum[12] = d0;
  sum[13] = d0 >> 8u;
  sum[14] = d0 >> 16u;
  sum[15] = d0 >> 24u;
}

/* Chromatic adaptation matrices*/
static float kBradford[9] = {
    0.8951f, 0.2664f, -0.1614f, -0.7502f, 1.7135f,
    0.0367f, 0.0389f, -0.0685f, 1.0296f,
};

static float kBradfordInv[9] = {
    0.9869929f, -0.1470543f, 0.1599627f, 0.4323053f, 0.5183603f,
    0.0492912f, -0.0085287f, 0.0400428f, 0.9684867f,
};

// Adapts whitepoint x, y to D50
static Status AdaptToXYZD50(float wx, float wy, float matrix[9]) {
  if (wx < 0 || wx > 1 || wy < 0 || wy > 1) {
    return JXL_FAILURE("xy color out of range");
  }

  float w[3] = {wx / wy, 1.0f, (1.0f - wx - wy) / wy};
  float w50[3] = {0.96422f, 1.0f, 0.82521f};

  float lms[3];
  float lms50[3];

  MatMul(kBradford, w, 3, 3, 1, lms);
  MatMul(kBradford, w50, 3, 3, 1, lms50);

  float a[9] = {
      lms50[0] / lms[0], 0, 0, 0, lms50[1] / lms[1], 0, 0, 0, lms50[2] / lms[2],
  };

  float b[9];
  MatMul(a, kBradford, 3, 3, 3, b);
  MatMul(kBradfordInv, b, 3, 3, 3, matrix);

  return true;
}

static Status PrimariesToXYZD50(float rx, float ry, float gx, float gy,
                                float bx, float by, float wx, float wy,
                                float matrix[9]) {
  if (rx < 0 || rx > 1 || ry < 0 || ry > 1 || gx < 0 || gx > 1 || gy < 0 ||
      gy > 1 || bx < 0 || bx > 1 || by < 0 || by > 1 || wx < 0 || wx > 1 ||
      wy < 0 || wy > 1) {
    return JXL_FAILURE("xy color out of range");
  }

  float primaries[9] = {
      rx, gx, bx, ry, gy, by, 1.0f - rx - ry, 1.0f - gx - gy, 1.0f - bx - by};
  float primaries_inv[9];
  memcpy(primaries_inv, primaries, sizeof(float) * 9);
  Inv3x3Matrix(primaries_inv);

  float w[3] = {wx / wy, 1.0f, (1.0f - wx - wy) / wy};
  float xyz[3];
  MatMul(primaries_inv, w, 3, 3, 1, xyz);

  float a[9] = {
      xyz[0], 0, 0, 0, xyz[1], 0, 0, 0, xyz[2],
  };

  float toXYZ[9];
  MatMul(primaries, a, 3, 3, 3, toXYZ);

  float d50[9];
  JXL_RETURN_IF_ERROR(AdaptToXYZD50(wx, wy, d50));

  MatMul(d50, toXYZ, 3, 3, 3, matrix);
  return true;
}

Status CreateICCChadMatrix(CIExy w, float result[9]) {
  float m[9];
  if (w.y == 0) {  // WhitePoint can not be pitch-black.
    return JXL_FAILURE("Invalid WhitePoint");
  }
  JXL_RETURN_IF_ERROR(AdaptToXYZD50(w.x, w.y, m));
  memcpy(result, m, sizeof(float) * 9);
  return true;
}

// Creates RGB to XYZ matrix given RGB primaries and whitepoint in xy.
Status CreateICCRGBMatrix(CIExy r, CIExy g, CIExy b, CIExy w, float result[9]) {
  float m[9];
  JXL_RETURN_IF_ERROR(
      PrimariesToXYZD50(r.x, r.y, g.x, g.y, b.x, b.y, w.x, w.y, m));
  memcpy(result, m, sizeof(float) * 9);
  return true;
}

void WriteICCUint32(uint32_t value, size_t pos, PaddedBytes* JXL_RESTRICT icc) {
  if (icc->size() < pos + 4) icc->resize(pos + 4);
  (*icc)[pos + 0] = (value >> 24u) & 255;
  (*icc)[pos + 1] = (value >> 16u) & 255;
  (*icc)[pos + 2] = (value >> 8u) & 255;
  (*icc)[pos + 3] = value & 255;
}

void WriteICCUint16(uint16_t value, size_t pos, PaddedBytes* JXL_RESTRICT icc) {
  if (icc->size() < pos + 2) icc->resize(pos + 2);
  (*icc)[pos + 0] = (value >> 8u) & 255;
  (*icc)[pos + 1] = value & 255;
}

// Writes a 4-character tag
void WriteICCTag(const char* value, size_t pos, PaddedBytes* JXL_RESTRICT icc) {
  if (icc->size() < pos + 4) icc->resize(pos + 4);
  memcpy(icc->data() + pos, value, 4);
}

Status WriteICCS15Fixed16(float value, size_t pos,
                          PaddedBytes* JXL_RESTRICT icc) {
  // "nextafterf" for 32768.0f towards zero are:
  // 32767.998046875, 32767.99609375, 32767.994140625
  // Even the first value works well,...
  bool ok = (-32767.995f <= value) && (value <= 32767.995f);
  if (!ok) return JXL_FAILURE("ICC value is out of range / NaN");
  int32_t i = value * 65536.0f + 0.5f;
  // Use two's complement
  uint32_t u = static_cast<uint32_t>(i);
  WriteICCUint32(u, pos, icc);
  return true;
}

Status CreateICCHeader(const ColorEncoding& c,
                       PaddedBytes* JXL_RESTRICT header) {
  // TODO(lode): choose color management engine name, e.g. "skia" if
  // integrated in skia.
  static const char* kCmm = "jxl ";

  header->resize(128, 0);

  WriteICCUint32(0, 0, header);  // size, correct value filled in at end
  WriteICCTag(kCmm, 4, header);
  WriteICCUint32(0x04300000u, 8, header);
  WriteICCTag("mntr", 12, header);
  WriteICCTag(c.IsGray() ? "GRAY" : "RGB ", 16, header);
  WriteICCTag("XYZ ", 20, header);

  // Three uint32_t's date/time encoding.
  // TODO(lode): encode actual date and time, this is a placeholder
  uint32_t year = 2019, month = 12, day = 1;
  uint32_t hour = 0, minute = 0, second = 0;
  WriteICCUint16(year, 24, header);
  WriteICCUint16(month, 26, header);
  WriteICCUint16(day, 28, header);
  WriteICCUint16(hour, 30, header);
  WriteICCUint16(minute, 32, header);
  WriteICCUint16(second, 34, header);

  WriteICCTag("acsp", 36, header);
  WriteICCTag("APPL", 40, header);
  WriteICCUint32(0, 44, header);  // flags
  WriteICCUint32(0, 48, header);  // device manufacturer
  WriteICCUint32(0, 52, header);  // device model
  WriteICCUint32(0, 56, header);  // device attributes
  WriteICCUint32(0, 60, header);  // device attributes
  WriteICCUint32(static_cast<uint32_t>(c.rendering_intent), 64, header);

  // Mandatory D50 white point of profile connection space
  WriteICCUint32(0x0000f6d6, 68, header);
  WriteICCUint32(0x00010000, 72, header);
  WriteICCUint32(0x0000d32d, 76, header);

  WriteICCTag(kCmm, 80, header);

  return true;
}

void AddToICCTagTable(const char* tag, size_t offset, size_t size,
                      PaddedBytes* JXL_RESTRICT tagtable,
                      std::vector<size_t>* offsets) {
  WriteICCTag(tag, tagtable->size(), tagtable);
  // writing true offset deferred to later
  WriteICCUint32(0, tagtable->size(), tagtable);
  offsets->push_back(offset);
  WriteICCUint32(size, tagtable->size(), tagtable);
}

void FinalizeICCTag(PaddedBytes* JXL_RESTRICT tags, size_t* offset,
                    size_t* size) {
  while ((tags->size() & 3) != 0) {
    tags->push_back(0);
  }
  *offset += *size;
  *size = tags->size() - *offset;
}

// The input text must be ASCII, writing other characters to UTF-16 is not
// implemented.
void CreateICCMlucTag(const std::string& text, PaddedBytes* JXL_RESTRICT tags) {
  WriteICCTag("mluc", tags->size(), tags);
  WriteICCUint32(0, tags->size(), tags);
  WriteICCUint32(1, tags->size(), tags);
  WriteICCUint32(12, tags->size(), tags);
  WriteICCTag("enUS", tags->size(), tags);
  WriteICCUint32(text.size() * 2, tags->size(), tags);
  WriteICCUint32(28, tags->size(), tags);
  for (size_t i = 0; i < text.size(); i++) {
    tags->push_back(0);  // prepend 0 for UTF-16
    tags->push_back(text[i]);
  }
}

Status CreateICCXYZTag(float xyz[3], PaddedBytes* JXL_RESTRICT tags) {
  WriteICCTag("XYZ ", tags->size(), tags);
  WriteICCUint32(0, tags->size(), tags);
  for (size_t i = 0; i < 3; ++i) {
    JXL_RETURN_IF_ERROR(WriteICCS15Fixed16(xyz[i], tags->size(), tags));
  }
  return true;
}

Status CreateICCChadTag(float chad[9], PaddedBytes* JXL_RESTRICT tags) {
  WriteICCTag("sf32", tags->size(), tags);
  WriteICCUint32(0, tags->size(), tags);
  for (size_t i = 0; i < 9; i++) {
    JXL_RETURN_IF_ERROR(WriteICCS15Fixed16(chad[i], tags->size(), tags));
  }
  return true;
}

void CreateICCCurvCurvTag(const std::vector<uint16_t>& curve,
                          PaddedBytes* JXL_RESTRICT tags) {
  size_t pos = tags->size();
  tags->resize(tags->size() + 12 + curve.size() * 2, 0);
  WriteICCTag("curv", pos, tags);
  WriteICCUint32(0, pos + 4, tags);
  WriteICCUint32(curve.size(), pos + 8, tags);
  for (size_t i = 0; i < curve.size(); i++) {
    WriteICCUint16(curve[i], pos + 12 + i * 2, tags);
  }
}

Status CreateICCCurvParaTag(std::vector<float> params, size_t curve_type,
                            PaddedBytes* JXL_RESTRICT tags) {
  WriteICCTag("para", tags->size(), tags);
  WriteICCUint32(0, tags->size(), tags);
  WriteICCUint16(curve_type, tags->size(), tags);
  WriteICCUint16(0, tags->size(), tags);
  for (size_t i = 0; i < params.size(); i++) {
    JXL_RETURN_IF_ERROR(WriteICCS15Fixed16(params[i], tags->size(), tags));
  }
  return true;
}

Status MaybeCreateProfile(const ColorEncoding& c,
                          PaddedBytes* JXL_RESTRICT icc) {
  PaddedBytes header, tagtable, tags;

  if (c.GetColorSpace() == ColorSpace::kUnknown || c.tf.IsUnknown()) {
    return false;  // Not an error
  }

  switch (c.GetColorSpace()) {
    case ColorSpace::kRGB:
    case ColorSpace::kGray:
      break;  // OK
    case ColorSpace::kXYB:
      return JXL_FAILURE("XYB ICC not yet implemented");
    default:
      return JXL_FAILURE("Invalid CS %u",
                         static_cast<unsigned int>(c.GetColorSpace()));
  }

  JXL_RETURN_IF_ERROR(CreateICCHeader(c, &header));

  std::vector<size_t> offsets;
  // tag count, deferred to later
  WriteICCUint32(0, tagtable.size(), &tagtable);

  size_t tag_offset = 0, tag_size = 0;

  CreateICCMlucTag(Description(c), &tags);
  FinalizeICCTag(&tags, &tag_offset, &tag_size);
  AddToICCTagTable("desc", tag_offset, tag_size, &tagtable, &offsets);

  const std::string copyright =
      "Copyright 2019 Google LLC, CC-BY-SA 3.0 Unported "
      "license(https://creativecommons.org/licenses/by-sa/3.0/legalcode)";
  CreateICCMlucTag(copyright, &tags);
  FinalizeICCTag(&tags, &tag_offset, &tag_size);
  AddToICCTagTable("cprt", tag_offset, tag_size, &tagtable, &offsets);

  // TODO(eustas): isn't it the other way round: gray image has d50 WhitePoint?
  if (c.IsGray()) {
    float wtpt[3];
    JXL_RETURN_IF_ERROR(CIEXYZFromWhiteCIExy(c.GetWhitePoint(), wtpt));
    JXL_RETURN_IF_ERROR(CreateICCXYZTag(wtpt, &tags));
  } else {
    float d50[3] = {0.964203, 1.0, 0.824905};
    JXL_RETURN_IF_ERROR(CreateICCXYZTag(d50, &tags));
  }
  FinalizeICCTag(&tags, &tag_offset, &tag_size);
  AddToICCTagTable("wtpt", tag_offset, tag_size, &tagtable, &offsets);

  if (!c.IsGray()) {
    // Chromatic adaptation matrix
    float chad[9];
    JXL_RETURN_IF_ERROR(CreateICCChadMatrix(c.GetWhitePoint(), chad));

    const PrimariesCIExy primaries = c.GetPrimaries();
    float m[9];
    JXL_RETURN_IF_ERROR(CreateICCRGBMatrix(primaries.r, primaries.g,
                                           primaries.b, c.GetWhitePoint(), m));
    float r[3] = {m[0], m[3], m[6]};
    float g[3] = {m[1], m[4], m[7]};
    float b[3] = {m[2], m[5], m[8]};

    JXL_RETURN_IF_ERROR(CreateICCChadTag(chad, &tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("chad", tag_offset, tag_size, &tagtable, &offsets);

    JXL_RETURN_IF_ERROR(CreateICCXYZTag(r, &tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("rXYZ", tag_offset, tag_size, &tagtable, &offsets);

    JXL_RETURN_IF_ERROR(CreateICCXYZTag(g, &tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("gXYZ", tag_offset, tag_size, &tagtable, &offsets);

    JXL_RETURN_IF_ERROR(CreateICCXYZTag(b, &tags));
    FinalizeICCTag(&tags, &tag_offset, &tag_size);
    AddToICCTagTable("bXYZ", tag_offset, tag_size, &tagtable, &offsets);
  }

  if (c.tf.IsGamma()) {
    float gamma = 1.0 / c.tf.GetGamma();
    JXL_RETURN_IF_ERROR(
        CreateICCCurvParaTag({gamma, 1.0, 0.0, 1.0, 0.0}, 3, &tags));
  } else {
    switch (c.tf.GetTransferFunction()) {
      case TransferFunction::kHLG:
        CreateICCCurvCurvTag(
            HWY_DYNAMIC_DISPATCH(CreateTableCurve)(4096, ExtraTF::kHLG), &tags);
        break;
      case TransferFunction::kPQ:
        CreateICCCurvCurvTag(
            HWY_DYNAMIC_DISPATCH(CreateTableCurve)(4096, ExtraTF::kPQ), &tags);
        break;
      case TransferFunction::kSRGB:
        JXL_RETURN_IF_ERROR(CreateICCCurvParaTag(
            {2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045}, 3, &tags));
        break;
      case TransferFunction::k709:
        JXL_RETURN_IF_ERROR(CreateICCCurvParaTag(
            {1.0 / 0.45, 1.0 / 1.099, 0.099 / 1.099, 1.0 / 4.5, 0.081}, 3,
            &tags));
        break;
      case TransferFunction::kLinear:
        JXL_RETURN_IF_ERROR(
            CreateICCCurvParaTag({1.0, 1.0, 0.0, 1.0, 0.0}, 3, &tags));
        break;
      case TransferFunction::kDCI:
        JXL_RETURN_IF_ERROR(
            CreateICCCurvParaTag({2.6, 1.0, 0.0, 1.0, 0.0}, 3, &tags));
        break;
      default:
        JXL_ABORT("Unknown TF %d", c.tf.GetTransferFunction());
    }
  }
  FinalizeICCTag(&tags, &tag_offset, &tag_size);
  if (c.IsGray()) {
    AddToICCTagTable("kTRC", tag_offset, tag_size, &tagtable, &offsets);
  } else {
    AddToICCTagTable("rTRC", tag_offset, tag_size, &tagtable, &offsets);
    AddToICCTagTable("gTRC", tag_offset, tag_size, &tagtable, &offsets);
    AddToICCTagTable("bTRC", tag_offset, tag_size, &tagtable, &offsets);
  }

  // Tag count
  WriteICCUint32(offsets.size(), 0, &tagtable);
  for (size_t i = 0; i < offsets.size(); i++) {
    WriteICCUint32(offsets[i] + header.size() + tagtable.size(), 4 + 12 * i + 4,
                   &tagtable);
  }

  // ICC profile size
  WriteICCUint32(header.size() + tagtable.size() + tags.size(), 0, &header);

  *icc = header;
  icc->append(tagtable);
  icc->append(tags);

  // The MD5 checksum must be computed on the profile with profile flags,
  // rendering intent, and region of the checksum itself, set to 0.
  // TODO(lode): manually verify with a reliable tool that this creates correct
  // signature (profile id) for ICC profiles.
  PaddedBytes icc_sum = *icc;
  memset(icc_sum.data() + 44, 0, 4);
  memset(icc_sum.data() + 64, 0, 4);
  uint8_t checksum[16];
  ICCComputeMD5(icc_sum, checksum);

  memcpy(icc->data() + 84, checksum, sizeof(checksum));

  return true;
}

#if JPEGXL_ENABLE_SKCMS

ColorSpace ColorSpaceFromProfile(const skcms_ICCProfile& profile) {
  switch (profile.data_color_space) {
    case skcms_Signature_RGB:
      return ColorSpace::kRGB;
    case skcms_Signature_Gray:
      return ColorSpace::kGray;
    default:
      return ColorSpace::kUnknown;
  }
}

// "profile1" is pre-decoded to save time in DetectTransferFunction.
Status ProfileEquivalentToICC(const skcms_ICCProfile& profile1,
                              const PaddedBytes& icc) {
  skcms_ICCProfile profile2;
  JXL_RETURN_IF_ERROR(skcms_Parse(icc.data(), icc.size(), &profile2));
  return skcms_ApproximatelyEqualProfiles(&profile1, &profile2);
}

// vector_out := matmul(matrix, vector_in)
void MatrixProduct(const skcms_Matrix3x3& matrix, const float vector_in[3],
                   float vector_out[3]) {
  for (int i = 0; i < 3; ++i) {
    vector_out[i] = 0;
    for (int j = 0; j < 3; ++j) {
      vector_out[i] += matrix.vals[i][j] * vector_in[j];
    }
  }
}

// Returns white point that was specified when creating the profile.
JXL_MUST_USE_RESULT Status UnadaptedWhitePoint(const skcms_ICCProfile& profile,
                                               CIExy* out) {
  float media_white_point_XYZ[3];
  if (!skcms_GetWTPT(&profile, media_white_point_XYZ)) {
    return JXL_FAILURE("ICC profile does not contain WhitePoint tag");
  }
  skcms_Matrix3x3 CHAD;
  if (!skcms_GetCHAD(&profile, &CHAD)) {
    // If there is no chromatic adaptation matrix, it means that the white point
    // is already unadapted.
    *out = CIExyFromXYZ(media_white_point_XYZ);
    return true;
  }
  // Otherwise, it has been adapted to the PCS white point using said matrix,
  // and the adaptation needs to be undone.
  skcms_Matrix3x3 inverse_CHAD;
  if (!skcms_Matrix3x3_invert(&CHAD, &inverse_CHAD)) {
    return JXL_FAILURE("Non-invertible ChromaticAdaptation matrix");
  }
  float unadapted_white_point_XYZ[3];
  MatrixProduct(inverse_CHAD, media_white_point_XYZ, unadapted_white_point_XYZ);
  *out = CIExyFromXYZ(unadapted_white_point_XYZ);
  return true;
}

Status IdentifyPrimaries(const skcms_ICCProfile& profile,
                         const CIExy& wp_unadapted, ColorEncoding* c) {
  if (!c->HasPrimaries()) return true;

  skcms_Matrix3x3 CHAD, inverse_CHAD;
  if (skcms_GetCHAD(&profile, &CHAD)) {
    JXL_RETURN_IF_ERROR(skcms_Matrix3x3_invert(&CHAD, &inverse_CHAD));
  } else {
    static constexpr skcms_Matrix3x3 kLMSFromXYZ = {
        {{0.8951, 0.2664, -0.1614},
         {-0.7502, 1.7135, 0.0367},
         {0.0389, -0.0685, 1.0296}}};
    static constexpr skcms_Matrix3x3 kXYZFromLMS = {
        {{0.9869929, -0.1470543, 0.1599627},
         {0.4323053, 0.5183603, 0.0492912},
         {-0.0085287, 0.0400428, 0.9684867}}};
    static constexpr float kWpD50XYZ[3] = {0.96420288, 1.0, 0.82490540};
    float wp_unadapted_XYZ[3];
    JXL_RETURN_IF_ERROR(CIEXYZFromWhiteCIExy(wp_unadapted, wp_unadapted_XYZ));
    float wp_D50_LMS[3], wp_unadapted_LMS[3];
    MatrixProduct(kLMSFromXYZ, kWpD50XYZ, wp_D50_LMS);
    MatrixProduct(kLMSFromXYZ, wp_unadapted_XYZ, wp_unadapted_LMS);
    inverse_CHAD = {{{wp_unadapted_LMS[0] / wp_D50_LMS[0], 0, 0},
                     {0, wp_unadapted_LMS[1] / wp_D50_LMS[1], 0},
                     {0, 0, wp_unadapted_LMS[2] / wp_D50_LMS[2]}}};
    inverse_CHAD = skcms_Matrix3x3_concat(&kXYZFromLMS, &inverse_CHAD);
    inverse_CHAD = skcms_Matrix3x3_concat(&inverse_CHAD, &kLMSFromXYZ);
  }

  float XYZ[3];
  PrimariesCIExy primaries;
  CIExy* const chromaticities[] = {&primaries.r, &primaries.g, &primaries.b};
  for (int i = 0; i < 3; ++i) {
    float RGB[3] = {};
    RGB[i] = 1;
    skcms_Transform(RGB, skcms_PixelFormat_RGB_fff, skcms_AlphaFormat_Opaque,
                    &profile, XYZ, skcms_PixelFormat_RGB_fff,
                    skcms_AlphaFormat_Opaque, skcms_XYZD50_profile(), 1);
    float unadapted_XYZ[3];
    MatrixProduct(inverse_CHAD, XYZ, unadapted_XYZ);
    *chromaticities[i] = CIExyFromXYZ(unadapted_XYZ);
  }
  return c->SetPrimaries(primaries);
}

void DetectTransferFunction(const skcms_ICCProfile& profile,
                            ColorEncoding* JXL_RESTRICT c) {
  if (c->tf.SetImplicit()) return;

  for (TransferFunction tf : Values<TransferFunction>()) {
    // Can only create profile from known transfer function.
    if (tf == TransferFunction::kUnknown) continue;

    c->tf.SetTransferFunction(tf);

    skcms_ICCProfile profile_test;
    PaddedBytes bytes;
    if (MaybeCreateProfile(*c, &bytes) && DecodeProfile(bytes, &profile_test) &&
        skcms_ApproximatelyEqualProfiles(&profile, &profile_test)) {
      return;
    }
  }

  c->tf.SetTransferFunction(TransferFunction::kUnknown);
}

#else  // JPEGXL_ENABLE_SKCMS

uint32_t Type32(const ColorEncoding& c) {
  if (c.IsGray()) return TYPE_GRAY_FLT;
  return TYPE_RGB_FLT;
}

uint32_t Type64(const ColorEncoding& c) {
  if (c.IsGray()) return TYPE_GRAY_DBL;
  return TYPE_RGB_DBL;
}

ColorSpace ColorSpaceFromProfile(const Profile& profile) {
  switch (cmsGetColorSpace(profile.get())) {
    case cmsSigRgbData:
      return ColorSpace::kRGB;
    case cmsSigGrayData:
      return ColorSpace::kGray;
    default:
      return ColorSpace::kUnknown;
  }
}

// "profile1" is pre-decoded to save time in DetectTransferFunction.
Status ProfileEquivalentToICC(const cmsContext context, const Profile& profile1,
                              const PaddedBytes& icc, const ColorEncoding& c) {
  const uint32_t type_src = Type64(c);

  Profile profile2;
  JXL_RETURN_IF_ERROR(DecodeProfile(context, icc, &profile2));

  Profile profile_xyz;
  JXL_RETURN_IF_ERROR(CreateProfileXYZ(context, &profile_xyz));

  const uint32_t intent = INTENT_RELATIVE_COLORIMETRIC;
  const uint32_t flags = cmsFLAGS_NOOPTIMIZE | cmsFLAGS_BLACKPOINTCOMPENSATION |
                         cmsFLAGS_HIGHRESPRECALC;
  Transform xform1(cmsCreateTransformTHR(context, profile1.get(), type_src,
                                         profile_xyz.get(), TYPE_XYZ_DBL,
                                         intent, flags));
  Transform xform2(cmsCreateTransformTHR(context, profile2.get(), type_src,
                                         profile_xyz.get(), TYPE_XYZ_DBL,
                                         intent, flags));
  if (xform1 == nullptr || xform2 == nullptr) {
    return JXL_FAILURE("Failed to create transform");
  }

  double in[3];
  double out1[3];
  double out2[3];

  // Uniformly spaced samples from very dark to almost fully bright.
  const double init = 1E-3;
  const double step = 0.2;

  if (c.IsGray()) {
    // Finer sampling and replicate each component.
    for (in[0] = init; in[0] < 1.0; in[0] += step / 8) {
      cmsDoTransform(xform1.get(), in, out1, 1);
      cmsDoTransform(xform2.get(), in, out2, 1);
      if (!ApproxEq(out1[0], out2[0], 2E-4)) {
        return false;
      }
    }
  } else {
    for (in[0] = init; in[0] < 1.0; in[0] += step) {
      for (in[1] = init; in[1] < 1.0; in[1] += step) {
        for (in[2] = init; in[2] < 1.0; in[2] += step) {
          cmsDoTransform(xform1.get(), in, out1, 1);
          cmsDoTransform(xform2.get(), in, out2, 1);
          for (size_t i = 0; i < 3; ++i) {
            if (!ApproxEq(out1[i], out2[i], 2E-4)) {
              return false;
            }
          }
        }
      }
    }
  }

  return true;
}

// Returns white point that was specified when creating the profile.
// NOTE: we can't just use cmsSigMediaWhitePointTag because its interpretation
// differs between ICC versions.
JXL_MUST_USE_RESULT cmsCIEXYZ UnadaptedWhitePoint(const cmsContext context,
                                                  const Profile& profile,
                                                  const ColorEncoding& c) {
  cmsCIEXYZ XYZ = {1.0, 1.0, 1.0};

  Profile profile_xyz;
  if (!CreateProfileXYZ(context, &profile_xyz)) return XYZ;
  // Array arguments are one per profile.
  cmsHPROFILE profiles[2] = {profile.get(), profile_xyz.get()};
  // Leave white point unchanged - that is what we're trying to extract.
  cmsUInt32Number intents[2] = {INTENT_ABSOLUTE_COLORIMETRIC,
                                INTENT_ABSOLUTE_COLORIMETRIC};
  cmsBool black_compensation[2] = {0, 0};
  cmsFloat64Number adaption[2] = {0.0, 0.0};
  // Only transforming a single pixel, so skip expensive optimizations.
  cmsUInt32Number flags = cmsFLAGS_NOOPTIMIZE | cmsFLAGS_HIGHRESPRECALC;
  Transform xform(cmsCreateExtendedTransform(
      context, 2, profiles, black_compensation, intents, adaption, nullptr, 0,
      Type64(c), TYPE_XYZ_DBL, flags));
  if (!xform) return XYZ;  // TODO(lode): return error

  // xy are relative, so magnitude does not matter if we ignore output Y.
  const cmsFloat64Number in[3] = {1.0, 1.0, 1.0};
  cmsDoTransform(xform.get(), in, &XYZ.X, 1);
  return XYZ;
}

Status IdentifyPrimaries(const Profile& profile, const cmsCIEXYZ& wp_unadapted,
                         ColorEncoding* c) {
  if (!c->HasPrimaries()) return true;
  if (ColorSpaceFromProfile(profile) == ColorSpace::kUnknown) return true;

  // These were adapted to the profile illuminant before storing in the profile.
  const cmsCIEXYZ* adapted_r = static_cast<const cmsCIEXYZ*>(
      cmsReadTag(profile.get(), cmsSigRedColorantTag));
  const cmsCIEXYZ* adapted_g = static_cast<const cmsCIEXYZ*>(
      cmsReadTag(profile.get(), cmsSigGreenColorantTag));
  const cmsCIEXYZ* adapted_b = static_cast<const cmsCIEXYZ*>(
      cmsReadTag(profile.get(), cmsSigBlueColorantTag));
  if (adapted_r == nullptr || adapted_g == nullptr || adapted_b == nullptr) {
    return JXL_FAILURE("Failed to retrieve colorants");
  }

  // TODO(janwas): no longer assume Bradford and D50.
  // Undo the chromatic adaptation.
  const cmsCIEXYZ d50 = D50_XYZ();

  cmsCIEXYZ r, g, b;
  cmsAdaptToIlluminant(&r, &d50, &wp_unadapted, adapted_r);
  cmsAdaptToIlluminant(&g, &d50, &wp_unadapted, adapted_g);
  cmsAdaptToIlluminant(&b, &d50, &wp_unadapted, adapted_b);

  const PrimariesCIExy rgb = {CIExyFromXYZ(r), CIExyFromXYZ(g),
                              CIExyFromXYZ(b)};
  return c->SetPrimaries(rgb);
}

void DetectTransferFunction(const cmsContext context, const Profile& profile,
                            ColorEncoding* JXL_RESTRICT c) {
  if (c->tf.SetImplicit()) return;

  for (TransferFunction tf : Values<TransferFunction>()) {
    // Can only create profile from known transfer function.
    if (tf == TransferFunction::kUnknown) continue;

    c->tf.SetTransferFunction(tf);

    PaddedBytes icc_test;
    if (MaybeCreateProfile(*c, &icc_test) &&
        ProfileEquivalentToICC(context, profile, icc_test, *c)) {
      return;
    }
  }

  c->tf.SetTransferFunction(TransferFunction::kUnknown);
}

void ErrorHandler(cmsContext context, cmsUInt32Number code, const char* text) {
  JXL_WARNING("LCMS error %u: %s", code, text);
}

// Returns a context for the current thread, creating it if necessary.
cmsContext GetContext() {
  static thread_local void* context_;
  if (context_ == nullptr) {
    context_ = cmsCreateContext(nullptr, nullptr);
    JXL_ASSERT(context_ != nullptr);

    cmsSetLogErrorHandlerTHR(static_cast<cmsContext>(context_), &ErrorHandler);
  }
  return static_cast<cmsContext>(context_);
}

#endif  // JPEGXL_ENABLE_SKCMS

}  // namespace

// All functions that call lcms directly (except ColorSpaceTransform::Run) must
// lock LcmsMutex().

Status ColorEncoding::SetFieldsFromICC() {
  // In case parsing fails, mark the ColorEncoding as invalid.
  SetColorSpace(ColorSpace::kUnknown);
  tf.SetTransferFunction(TransferFunction::kUnknown);

  if (icc_.empty()) return JXL_FAILURE("Empty ICC profile");

#if JPEGXL_ENABLE_SKCMS
  if (icc_.size() < 128) {
    return JXL_FAILURE("ICC file too small");
  }

  skcms_ICCProfile profile;
  JXL_RETURN_IF_ERROR(skcms_Parse(icc_.data(), icc_.size(), &profile));

  // skcms does not return the rendering intent, so get it from the file. It
  // is encoded as big-endian 32-bit integer in bytes 60..63.
  uint32_t rendering_intent32 = icc_[67];
  if (rendering_intent32 > 3 || icc_[64] != 0 || icc_[65] != 0 ||
      icc_[66] != 0) {
    return JXL_FAILURE("Invalid rendering intent %u\n", rendering_intent32);
  }

  SetColorSpace(ColorSpaceFromProfile(profile));

  CIExy wp_unadapted;
  JXL_RETURN_IF_ERROR(UnadaptedWhitePoint(profile, &wp_unadapted));
  JXL_RETURN_IF_ERROR(SetWhitePoint(wp_unadapted));

  // Relies on color_space.
  JXL_RETURN_IF_ERROR(IdentifyPrimaries(profile, wp_unadapted, this));

  // Relies on color_space/white point/primaries being set already.
  DetectTransferFunction(profile, this);
  // ICC and RenderingIntent have the same values (0..3).
  rendering_intent = static_cast<RenderingIntent>(rendering_intent32);
#else   // JPEGXL_ENABLE_SKCMS

  std::lock_guard<std::mutex> guard(LcmsMutex());
  const cmsContext context = GetContext();

  Profile profile;
  JXL_RETURN_IF_ERROR(DecodeProfile(context, icc_, &profile));

  const cmsUInt32Number rendering_intent32 =
      cmsGetHeaderRenderingIntent(profile.get());
  if (rendering_intent32 > 3) {
    return JXL_FAILURE("Invalid rendering intent %u\n", rendering_intent32);
  }

  SetColorSpace(ColorSpaceFromProfile(profile));

  const cmsCIEXYZ wp_unadapted = UnadaptedWhitePoint(context, profile, *this);
  JXL_RETURN_IF_ERROR(SetWhitePoint(CIExyFromXYZ(wp_unadapted)));

  // Relies on color_space.
  JXL_RETURN_IF_ERROR(IdentifyPrimaries(profile, wp_unadapted, this));

  // Relies on color_space/white point/primaries being set already.
  DetectTransferFunction(context, profile, this);

  // ICC and RenderingIntent have the same values (0..3).
  rendering_intent = static_cast<RenderingIntent>(rendering_intent32);
#endif  // JPEGXL_ENABLE_SKCMS

  return true;
}

Status ColorEncoding::CreateICC() {
  std::lock_guard<std::mutex> guard(LcmsMutex());
  InternalRemoveICC();
  if (!MaybeCreateProfile(*this, &icc_)) {
    return JXL_FAILURE("Failed to create profile from fields");
  }
  return true;
}

void ColorEncoding::DecideIfWantICC() {
  PaddedBytes icc_new;
  bool equivalent;
#if JPEGXL_ENABLE_SKCMS
  skcms_ICCProfile profile;
  if (!DecodeProfile(ICC(), &profile)) return;
  if (!MaybeCreateProfile(*this, &icc_new)) return;
  equivalent = ProfileEquivalentToICC(profile, icc_new);
#else   // JPEGXL_ENABLE_SKCMS
  const cmsContext context = GetContext();
  Profile profile;
  if (!DecodeProfile(context, ICC(), &profile)) return;
  if (!MaybeCreateProfile(*this, &icc_new)) return;
  equivalent = ProfileEquivalentToICC(context, profile, icc_new, *this);
#endif  // JPEGXL_ENABLE_SKCMS

  // Successfully created a profile => reconstruction should be equivalent.
  JXL_ASSERT(equivalent);
  want_icc_ = false;
}

ColorSpaceTransform::~ColorSpaceTransform() {
#if !JPEGXL_ENABLE_SKCMS
  std::lock_guard<std::mutex> guard(LcmsMutex());
  for (void* p : transforms_) {
    TransformDeleter()(p);
  }
#endif
}

ColorSpaceTransform::ColorSpaceTransform()
#if JPEGXL_ENABLE_SKCMS
    : skcms_icc_(new SkcmsICC())
#endif  // JPEGXL_ENABLE_SKCMS
{
}

Status ColorSpaceTransform::Init(const ColorEncoding& c_src,
                                 const ColorEncoding& c_dst,
                                 float intensity_target, size_t xsize,
                                 const size_t num_threads) {
  std::lock_guard<std::mutex> guard(LcmsMutex());
#if JXL_CMS_VERBOSE
  printf("%s -> %s\n", Description(c_src).c_str(), Description(c_dst).c_str());
#endif

#if JPEGXL_ENABLE_SKCMS
  skcms_icc_->icc_src_ = c_src.ICC();
  skcms_icc_->icc_dst_ = c_dst.ICC();
  JXL_RETURN_IF_ERROR(
      DecodeProfile(skcms_icc_->icc_src_, &skcms_icc_->profile_src_));
  JXL_RETURN_IF_ERROR(
      DecodeProfile(skcms_icc_->icc_dst_, &skcms_icc_->profile_dst_));
#else   // JPEGXL_ENABLE_SKCMS
  const cmsContext context = GetContext();
  Profile profile_src, profile_dst;
  JXL_RETURN_IF_ERROR(DecodeProfile(context, c_src.ICC(), &profile_src));
  JXL_RETURN_IF_ERROR(DecodeProfile(context, c_dst.ICC(), &profile_dst));
#endif  // JPEGXL_ENABLE_SKCMS

  skip_lcms_ = false;
  if (c_src.SameColorEncoding(c_dst)) {
    skip_lcms_ = true;
#if JXL_CMS_VERBOSE
    printf("Skip CMS\n");
#endif
  }

  // Special-case for BT.2100 HLG/PQ and SRGB <=> linear:
  const bool src_linear = c_src.tf.IsLinear();
  const bool dst_linear = c_dst.tf.IsLinear();
  if (((c_src.tf.IsPQ() || c_src.tf.IsHLG()) && dst_linear) ||
      ((c_dst.tf.IsPQ() || c_dst.tf.IsHLG()) && src_linear) ||
      ((c_src.tf.IsPQ() != c_dst.tf.IsPQ()) && intensity_target_ != 10000) ||
      (c_src.tf.IsSRGB() && dst_linear) || (c_dst.tf.IsSRGB() && src_linear)) {
    // Construct new profiles as if the data were already/still linear.
    ColorEncoding c_linear_src = c_src;
    ColorEncoding c_linear_dst = c_dst;
    c_linear_src.tf.SetTransferFunction(TransferFunction::kLinear);
    c_linear_dst.tf.SetTransferFunction(TransferFunction::kLinear);
    PaddedBytes icc_src, icc_dst;
#if JPEGXL_ENABLE_SKCMS
    skcms_ICCProfile new_src, new_dst;
#else   // JPEGXL_ENABLE_SKCMS
    Profile new_src, new_dst;
#endif  // JPEGXL_ENABLE_SKCMS
        // Only enable ExtraTF if profile creation succeeded.
    if (MaybeCreateProfile(c_linear_src, &icc_src) &&
        MaybeCreateProfile(c_linear_dst, &icc_dst) &&
#if JPEGXL_ENABLE_SKCMS
        DecodeProfile(icc_src, &new_src) && DecodeProfile(icc_dst, &new_dst)) {
#else   // JPEGXL_ENABLE_SKCMS
        DecodeProfile(context, icc_src, &new_src) &&
        DecodeProfile(context, icc_dst, &new_dst)) {
#endif  // JPEGXL_ENABLE_SKCMS
      if (c_src.SameColorSpace(c_dst)) {
        skip_lcms_ = true;
      }
#if JXL_CMS_VERBOSE
      printf("Special linear <-> HLG/PQ/sRGB; skip=%d\n", skip_lcms_);
#endif
#if JPEGXL_ENABLE_SKCMS
      skcms_icc_->icc_src_ = PaddedBytes();
      skcms_icc_->profile_src_ = new_src;
      skcms_icc_->icc_dst_ = PaddedBytes();
      skcms_icc_->profile_dst_ = new_dst;
#else   // JPEGXL_ENABLE_SKCMS
      profile_src.swap(new_src);
      profile_dst.swap(new_dst);
#endif  // JPEGXL_ENABLE_SKCMS
      if (!c_src.tf.IsLinear()) {
        preprocess_ = c_src.tf.IsSRGB()
                          ? ExtraTF::kSRGB
                          : (c_src.tf.IsPQ() ? ExtraTF::kPQ : ExtraTF::kHLG);
      }
      if (!c_dst.tf.IsLinear()) {
        postprocess_ = c_dst.tf.IsSRGB()
                           ? ExtraTF::kSRGB
                           : (c_dst.tf.IsPQ() ? ExtraTF::kPQ : ExtraTF::kHLG);
      }
    } else {
      JXL_WARNING("Failed to create extra linear profiles");
    }
  }

#if JPEGXL_ENABLE_SKCMS
  if (!skcms_MakeUsableAsDestination(&skcms_icc_->profile_dst_)) {
    return JXL_FAILURE(
        "Failed to make %s usable as a color transform destination",
        Description(c_dst).c_str());
  }
#endif  // JPEGXL_ENABLE_SKCMS

  // Not including alpha channel (copied separately).
  const size_t channels_src = c_src.Channels();
  const size_t channels_dst = c_dst.Channels();
  JXL_CHECK(channels_src == channels_dst);
#if JXL_CMS_VERBOSE
  printf("Channels: %zu; Threads: %zu\n", channels_src, num_threads);
#endif

#if !JPEGXL_ENABLE_SKCMS
  // Type includes color space (XYZ vs RGB), so can be different.
  const uint32_t type_src = Type32(c_src);
  const uint32_t type_dst = Type32(c_dst);
  transforms_.clear();
  for (size_t i = 0; i < num_threads; ++i) {
    const uint32_t intent = static_cast<uint32_t>(c_dst.rendering_intent);
    const uint32_t flags =
        cmsFLAGS_BLACKPOINTCOMPENSATION | cmsFLAGS_HIGHRESPRECALC;
    // NOTE: we're using the current thread's context and assuming all state
    // modified by cmsDoTransform resides in the transform, not the context.
    transforms_.emplace_back(cmsCreateTransformTHR(context, profile_src.get(),
                                                   type_src, profile_dst.get(),
                                                   type_dst, intent, flags));
    if (transforms_.back() == nullptr) {
      return JXL_FAILURE("Failed to create transform");
    }
  }
#endif  // !JPEGXL_ENABLE_SKCMS

  // Ideally LCMS would convert directly from External to Image3. However,
  // cmsDoTransformLineStride only accepts 32-bit BytesPerPlaneIn, whereas our
  // planes can be more than 4 GiB apart. Hence, transform inputs/outputs must
  // be interleaved. Calling cmsDoTransform for each pixel is expensive
  // (indirect call). We therefore transform rows, which requires per-thread
  // buffers. To avoid separate allocations, we use the rows of an image.
  // Because LCMS apparently also cannot handle <= 16 bit inputs and 32-bit
  // outputs (or vice versa), we use floating point input/output.
#if JPEGXL_ENABLE_SKCMS
  // SkiaCMS doesn't support grayscale float buffers, so we create space for RGB
  // float buffers anyway.
  buf_src_ = ImageF(xsize * 3, num_threads);
  buf_dst_ = ImageF(xsize * 3, num_threads);
#else
  buf_src_ = ImageF(xsize * channels_src, num_threads);
  buf_dst_ = ImageF(xsize * channels_dst, num_threads);
#endif
  intensity_target_ = intensity_target;
  xsize_ = xsize;
  return true;
}

}  // namespace jxl
#endif  // HWY_ONCE
