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
  CmsStage(const JxlCmsInterface* cms, const ColorEncoding& input,
           const ColorEncoding& output, float intensity_target)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        cms_(cms),
        c_input_(input),
        c_output_(output),
        intensity_target_(intensity_target) {
    fprintf(stderr, "c_src_: %s\n", Description(c_input_).c_str());
    fprintf(stderr, "dst: %s\n", Description(c_output_).c_str());
  }

  bool IsNeeded() const {
    // TODO(veluca): check if input and output color encodings are approximately
    // the same.
    return true;
  }

  void ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                  size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                  size_t thread_id) const final {
    JXL_ASSERT(false);
    JXL_ASSERT(xsize == xsize_);
    // const HWY_FULL(float) d;
    // const size_t xsize_v = RoundUpTo(xsize, Lanes(d));
    //(void)xsize_v;

    // TODO(firsching): CMYK, grayscale
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
    // TODO(veluca): CMYK? Gray?
    return c < 3 ? RenderPipelineChannelMode::kInPlace
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override { return "Cms"; }

 private:
  size_t xsize_;
  std::unique_ptr<jxl::ColorSpaceTransform> color_space_transform;
  const JxlCmsInterface* cms_;
  ColorEncoding c_input_;
  ColorEncoding c_output_;
  float intensity_target_;

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
    color_space_transform = jxl::make_unique<jxl::ColorSpaceTransform>(*cms_);
    JXL_RETURN_IF_ERROR(color_space_transform->Init(
        c_input_, c_output_, intensity_target_, xsize_, num_threads));
    return true;
  }
};

std::unique_ptr<RenderPipelineStage> GetCmsStage(const JxlCmsInterface* cms,
                                                 const ColorEncoding& input,
                                                 const ColorEncoding& output,
                                                 float intensity_target) {
  auto stage = jxl::make_unique<CmsStage>(cms, input, output, intensity_target);
  if (!stage->IsNeeded()) return nullptr;
  fprintf(stderr, "returning cms stage...\n");
  return stage;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetCmsStage);

std::unique_ptr<RenderPipelineStage> GetCmsStage(const JxlCmsInterface* cms,
                                                 const ColorEncoding& input,
                                                 const ColorEncoding& output,
                                                 float intensity_target) {
  return HWY_DYNAMIC_DISPATCH(GetCmsStage)(cms, input, output,
                                           intensity_target);
}

}  // namespace jxl
#endif
