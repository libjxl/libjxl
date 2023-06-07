
// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#ifndef LIB_JXL_JXL_CMS_INTERNAL_H_
#define LIB_JXL_JXL_CMS_INTERNAL_H_
#include <array>

#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"
#include "lib/jxl/image.h"

#if JPEGXL_ENABLE_SKCMS
#include "lib/jxl/enc_jxl_skcms.h"
#else  // JPEGXL_ENABLE_SKCMS
#include "lcms2.h"
#include "lcms2_plugin.h"
#endif  // JPEGXL_ENABLE_SKCMS
namespace jxl {
namespace {
struct JxlCms {
#if JPEGXL_ENABLE_SKCMS
  PaddedBytes icc_src, icc_dst;
  skcms_ICCProfile profile_src, profile_dst;
#else
  void* lcms_transform;
#endif

  // These fields are used when the HLG OOTF or inverse OOTF must be applied.
  bool apply_hlg_ootf;
  size_t hlg_ootf_num_channels;
  // Y component of the primaries.
  std::array<float, 3> hlg_ootf_luminances;

  size_t channels_src;
  size_t channels_dst;
  ImageF buf_src;
  ImageF buf_dst;
  float intensity_target;
  bool skip_lcms = false;
  ExtraTF preprocess = ExtraTF::kNone;
  ExtraTF postprocess = ExtraTF::kNone;
};

Status ApplyHlgOotf(JxlCms* t, float* JXL_RESTRICT buf, size_t xsize,
                    bool forward);
}  // namespace

#if JPEGXL_ENABLE_SKCMS
Status DecodeProfile(const uint8_t* icc, size_t size,
                     skcms_ICCProfile* const profile);
#else  // JPEGXL_ENABLE_SKCMS

struct ProfileDeleter {
  void operator()(void* p) { cmsCloseProfile(p); }
};
using Profile = std::unique_ptr<void, ProfileDeleter>;

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

Status DecodeProfile(const cmsContext context, const PaddedBytes& icc,
                     Profile* profile);
#endif

}  // namespace jxl

#endif  // LIB_JXL_JXL_CMS_INTERNAL_H_
