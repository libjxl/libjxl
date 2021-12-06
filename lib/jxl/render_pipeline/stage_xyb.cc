// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/stage_xyb.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/render_pipeline/stage_xyb.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/dec_xyb-inl.h"
#include "lib/jxl/fast_math-inl.h"
#include "lib/jxl/transfer_functions-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

struct OpLinear {
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
    return linear;
  }
};

struct OpRgb {
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
#if JXL_HIGH_PRECISION
    return TF_SRGB().EncodedFromDisplay(d, linear);
#else
    return FastLinearToSRGB(d, linear);
#endif
  }
};

struct OpPq {
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
    return TF_PQ().EncodedFromDisplay(d, linear);
  }
};

struct OpHlg {
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
    return TF_HLG().EncodedFromDisplay(d, linear);
  }
};

struct Op709 {
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
    return TF_709().EncodedFromDisplay(d, linear);
  }
};

struct OpGamma {
  const float inverse_gamma;
  template <typename D, typename T>
  T Transform(D d, const T& linear) const {
    return IfThenZeroElse(linear <= Set(d, 1e-5f),
                          FastPowf(d, linear, Set(d, inverse_gamma)));
  }
};

template <typename Op>
class XYBStage : public RenderPipelineStage {
 public:
  XYBStage(OpsinParams opsin_params, Op op)
      : RenderPipelineStage(RenderPipelineStage::Settings()),
        opsin_params_(opsin_params),
        op_(op) {}

  void ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                  size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                  float* JXL_RESTRICT temp) const final {
    PROFILER_ZONE("UndoXYB");

    const HWY_FULL(float) d;
    const size_t xsize_v = RoundUpTo(xsize, Lanes(d));
    float* JXL_RESTRICT row0 = GetInputRow(input_rows, 0, 0);
    float* JXL_RESTRICT row1 = GetInputRow(input_rows, 1, 0);
    float* JXL_RESTRICT row2 = GetInputRow(input_rows, 2, 0);
    // All calculations are lane-wise, still some might require
    // value-dependent behaviour (e.g. NearestInt). Temporary unpoison last
    // vector tail.
    msan::UnpoisonMemory(row0 + xsize, sizeof(float) * (xsize_v - xsize));
    msan::UnpoisonMemory(row1 + xsize, sizeof(float) * (xsize_v - xsize));
    msan::UnpoisonMemory(row2 + xsize, sizeof(float) * (xsize_v - xsize));
    for (int64_t x = -xextra; x < (int64_t)(xsize + xextra); x += Lanes(d)) {
      const auto in_opsin_x = Load(d, row0 + x + kRenderPipelineXOffset);
      const auto in_opsin_y = Load(d, row1 + x + kRenderPipelineXOffset);
      const auto in_opsin_b = Load(d, row2 + x + kRenderPipelineXOffset);
      JXL_COMPILER_FENCE;
      auto linear_r = Undefined(d);
      auto linear_g = Undefined(d);
      auto linear_b = Undefined(d);
      XybToRgb(d, in_opsin_x, in_opsin_y, in_opsin_b, opsin_params_, &linear_r,
               &linear_g, &linear_b);
      Store(op_.Transform(d, linear_r), d, row0 + x + kRenderPipelineXOffset);
      Store(op_.Transform(d, linear_g), d, row1 + x + kRenderPipelineXOffset);
      Store(op_.Transform(d, linear_b), d, row2 + x + kRenderPipelineXOffset);
    }
    msan::PoisonMemory(row0 + xsize, sizeof(float) * (xsize_v - xsize));
    msan::PoisonMemory(row1 + xsize, sizeof(float) * (xsize_v - xsize));
    msan::PoisonMemory(row2 + xsize, sizeof(float) * (xsize_v - xsize));
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInPlace
                 : RenderPipelineChannelMode::kIgnored;
  }

 private:
  OpsinParams opsin_params_;
  Op op_;
};

std::unique_ptr<RenderPipelineStage> GetXYBStage(
    const OutputEncodingInfo& output_encoding_info) {
  if (output_encoding_info.color_encoding.tf.IsLinear()) {
    return jxl::make_unique<XYBStage<OpLinear>>(
        output_encoding_info.opsin_params, OpLinear());
  } else if (output_encoding_info.color_encoding.tf.IsSRGB()) {
    return jxl::make_unique<XYBStage<OpRgb>>(output_encoding_info.opsin_params,
                                             OpRgb());
  } else if (output_encoding_info.color_encoding.tf.IsPQ()) {
    return jxl::make_unique<XYBStage<OpPq>>(output_encoding_info.opsin_params,
                                            OpPq());
  } else if (output_encoding_info.color_encoding.tf.IsHLG()) {
    return jxl::make_unique<XYBStage<OpHlg>>(output_encoding_info.opsin_params,
                                             OpHlg());
  } else if (output_encoding_info.color_encoding.tf.Is709()) {
    return jxl::make_unique<XYBStage<Op709>>(output_encoding_info.opsin_params,
                                             Op709());
  } else if (output_encoding_info.color_encoding.tf.IsGamma() ||
             output_encoding_info.color_encoding.tf.IsDCI()) {
    OpGamma op{output_encoding_info.inverse_gamma};
    return jxl::make_unique<XYBStage<OpGamma>>(
        output_encoding_info.opsin_params, op);
  } else {
    // This is a programming error.
    JXL_ABORT("Invalid target encoding");
  }
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

HWY_EXPORT(GetXYBStage);

std::unique_ptr<RenderPipelineStage> GetXYBStage(
    const OutputEncodingInfo& output_encoding_info) {
  return HWY_DYNAMIC_DISPATCH(GetXYBStage)(output_encoding_info);
}

}  // namespace jxl
#endif
