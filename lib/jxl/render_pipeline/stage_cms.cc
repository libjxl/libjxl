// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_cms.h"

#include <memory>

#include "jxl/cms_interface.h"
#include "jxl/color_encoding.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dec_xyb.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_cms.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/dec_tone_mapping-inl.h"
#include "lib/jxl/dec_xyb-inl.h"
#include "lib/jxl/sanitizers.h"
#include "lib/jxl/transfer_functions-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

class CmsStage : public RenderPipelineStage {
 public:
  explicit CmsStage(OutputEncodingInfo output_encoding_info)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        output_encoding_info_(std::move(output_encoding_info)) {
    bool orig_grey = output_encoding_info_.orig_color_encoding.IsGray();
    fprintf(stderr, "xyb_encoded: %d\n", output_encoding_info_.xyb_encoded);
    c_src_ = output_encoding_info_.xyb_encoded
                 ? ColorEncoding::LinearSRGB(orig_grey)
                 : output_encoding_info_.orig_color_encoding;
    fprintf(stderr, "c_src_: %s\n", Description(c_src_).c_str());
    fprintf(stderr, "dst: %s\n",
            Description(output_encoding_info_.color_encoding).c_str());
  }

  bool IsNeeded() const {
    const size_t channels_src = (c_src_.IsCMYK() ? 4 : c_src_.Channels());
    const size_t channels_dst = output_encoding_info_.color_encoding.Channels();
    const bool not_mixing_color_and_grey =
        (channels_src == channels_dst ||
         (channels_src == 4 && channels_dst == 3));
    const bool output_is_xyb =
        output_encoding_info_.color_encoding.GetColorSpace() ==
        ColorSpace::kXYB;
    fprintf(stderr, "output_is_xyb: %d , non_mix: %d ,"
    "!output_encoding_info_.color_encoding_is_original: %d\n", !output_is_xyb,
    not_mixing_color_and_grey, !output_encoding_info_.color_encoding_is_original);

    return (output_encoding_info_.color_management_system != nullptr) && !output_encoding_info_.color_encoding_is_original &&
           not_mixing_color_and_grey 
  }

  void ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                  size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                  size_t thread_id) const final {
    JXL_ASSERT(xsize == xsize_);
    // const HWY_FULL(float) d;
    // const size_t xsize_v = RoundUpTo(xsize, Lanes(d));
    //(void)xsize_v;

    // TODO(firsching): handle grey case seperately
    //  interleave
    float* JXL_RESTRICT row0 = GetInputRow(input_rows, 0, 0);
    float* JXL_RESTRICT row1 = GetInputRow(input_rows, 1, 0);
    float* JXL_RESTRICT row2 = GetInputRow(input_rows, 2, 0);
    if (thread_id == 0) {
      // fprintf(stderr, "row in: %f %f %f\n", row0[0], row1[0], row2[0]);
    }
    float* mutable_buf_src = color_space_transform->BufSrc(thread_id);
    for (size_t x = 0; x < xsize; x++) {
      mutable_buf_src[3 * x + 0] = row0[x];
      mutable_buf_src[3 * x + 1] = row1[x];
      mutable_buf_src[3 * x + 2] = row2[x];
    }
    const float* buf_src = mutable_buf_src;
    float* JXL_RESTRICT buf_dst = color_space_transform->BufDst(thread_id);
    if (!color_space_transform->Run(thread_id, buf_src, buf_dst)) {
      // somehow mark failing here?
      return;
    }
    // de-interleave
    for (size_t x = 0; x < xsize; x++) {
      row0[x] = buf_dst[3 * x + 0];
      row1[x] = buf_dst[3 * x + 1];
      row2[x] = buf_dst[3 * x + 2];
    }
    if (thread_id == 0) {
      // fprintf(stderr, "row out: %f %f %f\n", row0[0], row1[0], row2[0]);
    }
  }
  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInPlace
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "Cms"; }

 private:
  OutputEncodingInfo output_encoding_info_;
  size_t xsize_;
  std::unique_ptr<jxl::ColorSpaceTransform> color_space_transform;
  ColorEncoding c_src_;

  void SetInputSizes(
      const std::vector<std::pair<size_t, size_t>>& input_sizes) override {
#if JXL_ENABLE_ASSERT
    JXL_ASSERT(input_sizes.size() >= 3);
    for (size_t c = 1; c < input_sizes.size(); c++) {
      JXL_ASSERT(input_sizes[c].first == input_sizes[0].first);
      JXL_ASSERT(input_sizes[c].second == input_sizes[0].second);
    }
#endif
    xsize_ = input_sizes[0].first;
  }

  Status PrepareForThreads(size_t num_threads) override {
    color_space_transform = jxl::make_unique<jxl::ColorSpaceTransform>(
        *output_encoding_info_.color_management_system);
    JXL_RETURN_IF_ERROR(color_space_transform->Init(
        c_src_, output_encoding_info_.color_encoding,
        output_encoding_info_.desired_intensity_target, xsize_, num_threads));
    return true;
  }
};

std::unique_ptr<RenderPipelineStage> GetCmsStage(
    const OutputEncodingInfo& output_encoding_info) {
  auto stage = jxl::make_unique<CmsStage>(output_encoding_info);
  if (!stage->IsNeeded()) return nullptr;
  return stage;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetCmsStage);

std::unique_ptr<RenderPipelineStage> GetCmsStage(
    const OutputEncodingInfo& output_encoding_info) {
  return HWY_DYNAMIC_DISPATCH(GetCmsStage)(output_encoding_info);
}

}  // namespace jxl
#endif
