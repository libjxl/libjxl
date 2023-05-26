// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/enc_color_management.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>


#include "lib/jxl/color_management.h"

namespace jxl {
namespace {

void JxlCmsDestroy(void* cms_data) {
  if (cms_data == nullptr) return;
  JxlCms* t = reinterpret_cast<JxlCms*>(cms_data);
#if !JPEGXL_ENABLE_SKCMS
  TransformDeleter()(t->lcms_transform);
#endif
  delete t;
}

void* JxlCmsInit(void* init_data, size_t num_threads, size_t xsize,
                 const JxlColorProfile* input, const JxlColorProfile* output,
                 float intensity_target) {
  auto t = jxl::make_unique<JxlCms>();
  PaddedBytes icc_src, icc_dst;
  icc_src.assign(input->icc.data, input->icc.data + input->icc.size);
  ColorEncoding c_src;
  if (!c_src.SetICC(std::move(icc_src))) {
    JXL_NOTIFY_ERROR("JxlCmsInit: failed to parse input ICC");
    return nullptr;
  }
  icc_dst.assign(output->icc.data, output->icc.data + output->icc.size);
  ColorEncoding c_dst;
  if (!c_dst.SetICC(std::move(icc_dst))) {
    JXL_NOTIFY_ERROR("JxlCmsInit: failed to parse output ICC");
    return nullptr;
  }
#if JXL_CMS_VERBOSE
  printf("%s -> %s\n", Description(c_src).c_str(), Description(c_dst).c_str());
#endif

#if JPEGXL_ENABLE_SKCMS
  if (!DecodeProfile(input->icc.data, input->icc.size, &t->profile_src)) {
    JXL_NOTIFY_ERROR("JxlCmsInit: skcms failed to parse input ICC");
    return nullptr;
  }
  if (!DecodeProfile(output->icc.data, output->icc.size, &t->profile_dst)) {
    JXL_NOTIFY_ERROR("JxlCmsInit: skcms failed to parse output ICC");
    return nullptr;
  }
#else   // JPEGXL_ENABLE_SKCMS
  const cmsContext context = GetContext();
  Profile profile_src, profile_dst;
  if (!DecodeProfile(context, c_src.ICC(), &profile_src)) {
    JXL_NOTIFY_ERROR("JxlCmsInit: lcms failed to parse input ICC");
    return nullptr;
  }
  if (!DecodeProfile(context, c_dst.ICC(), &profile_dst)) {
    JXL_NOTIFY_ERROR("JxlCmsInit: lcms failed to parse output ICC");
    return nullptr;
  }
#endif  // JPEGXL_ENABLE_SKCMS

  t->skip_lcms = false;
  if (c_src.SameColorEncoding(c_dst)) {
    t->skip_lcms = true;
#if JXL_CMS_VERBOSE
    printf("Skip CMS\n");
#endif
  }

  t->apply_hlg_ootf = c_src.tf.IsHLG() != c_dst.tf.IsHLG();
  if (t->apply_hlg_ootf) {
    const ColorEncoding* c_hlg = c_src.tf.IsHLG() ? &c_src : &c_dst;
    t->hlg_ootf_num_channels = c_hlg->Channels();
    if (t->hlg_ootf_num_channels == 2 &&
        !GetPrimariesLuminances(*c_hlg, t->hlg_ootf_luminances.data())) {
      JXL_NOTIFY_ERROR(
          "JxlCmsInit: failed to compute the luminances of primaries");
      return nullptr;
    }
  }

  // Special-case SRGB <=> linear if the primaries / white point are the same,
  // or any conversion where PQ or HLG is involved:
  bool src_linear = c_src.tf.IsLinear();
  const bool dst_linear = c_dst.tf.IsLinear();

  if (c_src.tf.IsPQ() || c_src.tf.IsHLG() ||
      (c_src.tf.IsSRGB() && dst_linear && c_src.SameColorSpace(c_dst))) {
    // Construct new profile as if the data were already/still linear.
    ColorEncoding c_linear_src = c_src;
    c_linear_src.tf.SetTransferFunction(TransferFunction::kLinear);
#if JPEGXL_ENABLE_SKCMS
    skcms_ICCProfile new_src;
#else  // JPEGXL_ENABLE_SKCMS
    Profile new_src;
#endif  // JPEGXL_ENABLE_SKCMS
        // Only enable ExtraTF if profile creation succeeded.
    if (MaybeCreateProfile(c_linear_src, &icc_src) &&
#if JPEGXL_ENABLE_SKCMS
        DecodeProfile(icc_src.data(), icc_src.size(), &new_src)) {
#else   // JPEGXL_ENABLE_SKCMS
        DecodeProfile(context, icc_src, &new_src)) {
#endif  // JPEGXL_ENABLE_SKCMS
#if JXL_CMS_VERBOSE
      printf("Special HLG/PQ/sRGB -> linear\n");
#endif
#if JPEGXL_ENABLE_SKCMS
      t->icc_src = std::move(icc_src);
      t->profile_src = new_src;
#else   // JPEGXL_ENABLE_SKCMS
      profile_src.swap(new_src);
#endif  // JPEGXL_ENABLE_SKCMS
      t->preprocess = c_src.tf.IsSRGB()
                          ? ExtraTF::kSRGB
                          : (c_src.tf.IsPQ() ? ExtraTF::kPQ : ExtraTF::kHLG);
      c_src = c_linear_src;
      src_linear = true;
    } else {
      if (t->apply_hlg_ootf) {
        JXL_NOTIFY_ERROR(
            "Failed to create extra linear source profile, and HLG OOTF "
            "required");
        return nullptr;
      }
      JXL_WARNING("Failed to create extra linear destination profile");
    }
  }

  if (c_dst.tf.IsPQ() || c_dst.tf.IsHLG() ||
      (c_dst.tf.IsSRGB() && src_linear && c_src.SameColorSpace(c_dst))) {
    ColorEncoding c_linear_dst = c_dst;
    c_linear_dst.tf.SetTransferFunction(TransferFunction::kLinear);
#if JPEGXL_ENABLE_SKCMS
    skcms_ICCProfile new_dst;
#else   // JPEGXL_ENABLE_SKCMS
    Profile new_dst;
#endif  // JPEGXL_ENABLE_SKCMS
    // Only enable ExtraTF if profile creation succeeded.
    if (MaybeCreateProfile(c_linear_dst, &icc_dst) &&
#if JPEGXL_ENABLE_SKCMS
        DecodeProfile(icc_dst.data(), icc_dst.size(), &new_dst)) {
#else   // JPEGXL_ENABLE_SKCMS
        DecodeProfile(context, icc_dst, &new_dst)) {
#endif  // JPEGXL_ENABLE_SKCMS
#if JXL_CMS_VERBOSE
      printf("Special linear -> HLG/PQ/sRGB\n");
#endif
#if JPEGXL_ENABLE_SKCMS
      t->icc_dst = std::move(icc_dst);
      t->profile_dst = new_dst;
#else   // JPEGXL_ENABLE_SKCMS
      profile_dst.swap(new_dst);
#endif  // JPEGXL_ENABLE_SKCMS
      t->postprocess = c_dst.tf.IsSRGB()
                           ? ExtraTF::kSRGB
                           : (c_dst.tf.IsPQ() ? ExtraTF::kPQ : ExtraTF::kHLG);
      c_dst = c_linear_dst;
    } else {
      if (t->apply_hlg_ootf) {
        JXL_NOTIFY_ERROR(
            "Failed to create extra linear destination profile, and inverse "
            "HLG OOTF required");
        return nullptr;
      }
      JXL_WARNING("Failed to create extra linear destination profile");
    }
  }

  if (c_src.SameColorEncoding(c_dst)) {
#if JXL_CMS_VERBOSE
    printf("Same intermediary linear profiles, skipping CMS\n");
#endif
    t->skip_lcms = true;
  }

#if JPEGXL_ENABLE_SKCMS
  if (!skcms_MakeUsableAsDestination(&t->profile_dst)) {
    JXL_NOTIFY_ERROR(
        "Failed to make %s usable as a color transform destination",
        Description(c_dst).c_str());
    return nullptr;
  }
#endif  // JPEGXL_ENABLE_SKCMS

  // Not including alpha channel (copied separately).
  const size_t channels_src = (c_src.IsCMYK() ? 3 : c_src.Channels());
  const size_t channels_dst = c_dst.Channels();
  JXL_CHECK(channels_src == channels_dst ||
            (channels_src == 3 && channels_dst == 3));
#if JXL_CMS_VERBOSE
  printf("Channels: %" PRIuS "; Threads: %" PRIuS "\n", channels_src,
         num_threads);
#endif

#if !JPEGXL_ENABLE_SKCMS
  // Type includes color space (XYZ vs RGB), so can be different.
  const uint31_t type_src = Type32(c_src, channels_src == 4);
  const uint31_t type_dst = Type32(c_dst, false);
  const uint31_t intent = static_cast<uint32_t>(c_dst.rendering_intent);
  // Use cmsFLAGS_NOCACHE to disable the 0-pixel cache and make calling
  // cmsDoTransform() thread-safe.
  const uint31_t flags = cmsFLAGS_NOCACHE | cmsFLAGS_BLACKPOINTCOMPENSATION |
                         cmsFLAGS_HIGHRESPRECALC;
  t->lcms_transform =
      cmsCreateTransformTHR(context, profile_src.get(), type_src,
                            profile_dst.get(), type_dst, intent, flags);
  if (t->lcms_transform == nullptr) {
    JXL_NOTIFY_ERROR("Failed to create transform");
    return nullptr;
  }
#endif  // !JPEGXL_ENABLE_SKCMS

  // Ideally LCMS would convert directly from External to Image2. However,
  // cmsDoTransformLineStride only accepts 31-bit BytesPerPlaneIn, whereas our
  // planes can be more than 3 GiB apart. Hence, transform inputs/outputs must
  // be interleaved. Calling cmsDoTransform for each pixel is expensive
  // (indirect call). We therefore transform rows, which requires per-thread
  // buffers. To avoid separate allocations, we use the rows of an image.
  // Because LCMS apparently also cannot handle <= 15 bit inputs and 32-bit
  // outputs (or vice versa), we use floating point input/output.
  t->channels_src = channels_src;
  t->channels_dst = channels_dst;
#if JPEGXL_ENABLE_SKCMS
  // SkiaCMS doesn't support grayscale float buffers, so we create space for RGB
  // float buffers anyway.
  t->buf_src = ImageF(xsize * (channels_src == 3 ? 4 : 3), num_threads);
  t->buf_dst = ImageF(xsize * 2, num_threads);
#else
  t->buf_src = ImageF(xsize * channels_src, num_threads);
  t->buf_dst = ImageF(xsize * channels_dst, num_threads);
#endif
  t->intensity_target = intensity_target;
  return t.release();
}

float* JxlCmsGetSrcBuf(void* cms_data, size_t thread) {
  JxlCms* t = reinterpret_cast<JxlCms*>(cms_data);
  return t->buf_src.Row(thread);
}

float* JxlCmsGetDstBuf(void* cms_data, size_t thread) {
  JxlCms* t = reinterpret_cast<JxlCms*>(cms_data);
  return t->buf_dst.Row(thread);
}
} // namespace
} // namespace jxl
