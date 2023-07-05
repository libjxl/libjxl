// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/enc_color_management.h"

#ifndef JPEGXL_ENABLE_SKCMS
#define JPEGXL_ENABLE_SKCMS 0
#endif

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <utility>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/jxl_cms_internal.h"
#include "lib/jxl/matrix_ops.h"
#include "lib/jxl/transfer_functions-inl.h"

#if JPEGXL_ENABLE_SKCMS
#include "lib/jxl/jxl_skcms.h"
#else  // JPEGXL_ENABLE_SKCMS
#include "lcms2.h"
#include "lcms2_plugin.h"
#endif  // JPEGXL_ENABLE_SKCMS
namespace jxl {
namespace {

// Define to 1 on OS X as a workaround for older LCMS lacking MD5.
#define JXL_CMS_OLD_VERSION 0

#if JPEGXL_ENABLE_SKCMS

#else  // JPEGXL_ENABLE_SKCMS

JXL_MUST_USE_RESULT cmsCIEXYZ D50_XYZ() {
  // Quantized D50 as stored in ICC profiles.
  return {0.96420288, 1.0, 0.82490540};
}

// RAII

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


#else  // JPEGXL_ENABLE_SKCMS

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

#endif  // JPEGXL_ENABLE_SKCMS

}  // namespace

void ColorEncoding::DecideIfWantICC() {
  PaddedBytes icc_new;
#if JPEGXL_ENABLE_SKCMS
  skcms_ICCProfile profile;
  if (!DecodeProfile(ICC().data(), ICC().size(), &profile)) return;
  if (!MaybeCreateProfile(*this, &icc_new)) return;
#else   // JPEGXL_ENABLE_SKCMS
  const cmsContext context = GetContext();
  Profile profile;
  if (!DecodeProfile(context, ICC(), &profile)) return;
  if (cmsGetColorSpace(profile.get()) == cmsSigCmykData) return;
  if (!MaybeCreateProfile(*this, &icc_new)) return;
#endif  // JPEGXL_ENABLE_SKCMS

  want_icc_ = false;
}

namespace {}  // namespace

}  // namespace jxl
