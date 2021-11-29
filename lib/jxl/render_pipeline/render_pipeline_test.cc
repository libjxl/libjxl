// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/render_pipeline.h"

#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace jxl {
namespace {

class TrivialStage : public RenderPipelineStage {
 public:
  TrivialStage(size_t shift, size_t border)
      : RenderPipelineStage(
            RenderPipelineStage::Settings::Symmetric(shift, border)) {}

  void ProcessRow(RowInfo input_rows, RowInfo output_rows, size_t xextra,
                  size_t xsize, size_t xpos, size_t ypos,
                  float* JXL_RESTRICT temp) const final {}

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInOut;
  }
};
class FinalTrivialStage : public RenderPipelineStage {
 public:
  FinalTrivialStage() : RenderPipelineStage(RenderPipelineStage::Settings()) {}

  void ProcessRow(RowInfo input_rows, RowInfo output_rows, size_t xextra,
                  size_t xsize, size_t xpos, size_t ypos,
                  float* JXL_RESTRICT temp) const final {}

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInput;
  }
};

TEST(RenderPipelineTest, Build) {
  RenderPipeline::Builder builder(/*channel_shifts=*/{{1, 1}});
  builder.AddStage(jxl::make_unique<TrivialStage>(1, 1));  // shift, border
  builder.AddStage(jxl::make_unique<TrivialStage>(0, 2));  // no shift, border
  builder.AddStage(jxl::make_unique<FinalTrivialStage>());
  builder.UseSimpleImplementation();
  FrameDimensions frame_dimensions;
  frame_dimensions.Set(/*xsize=*/1024, /*ysize=*/1024, /*group_size_shift=*/0,
                       /*max_hshift=*/0, /*max_vshift=*/0,
                       /*modular_mode=*/false, /*upsampling=*/2);
  std::move(builder).Finalize(frame_dimensions);
}

TEST(RenderPipelineTest, CallAllGroups) {
  RenderPipeline::Builder builder({{1, 1}});
  builder.AddStage(jxl::make_unique<TrivialStage>(1, 1));  // shift, border
  builder.AddStage(jxl::make_unique<TrivialStage>(0, 2));  // no shift, border
  builder.AddStage(jxl::make_unique<FinalTrivialStage>());
  builder.UseSimpleImplementation();
  FrameDimensions frame_dimensions;
  frame_dimensions.Set(/*xsize=*/1024, /*ysize=*/1024, /*group_size_shift=*/0,
                       /*max_hshift=*/0, /*max_vshift=*/0,
                       /*modular_mode=*/false, /*upsampling=*/2);
  auto pipeline = std::move(builder).Finalize(frame_dimensions);
  pipeline->PrepareForThreads(1);

  for (size_t i = 0; i < frame_dimensions.num_groups; i++) {
    const auto& input_buffers = pipeline->GetInputBuffers(i, 0);
    (void)input_buffers;
    /*
    TODO(veluca): call this function once the render pipeline actually prepares
    buffers.

    FillPlane(0.0f, input_buffers.GetBuffer(0).first,
              input_buffers.GetBuffer(0).second);
    */
  }

  EXPECT_TRUE(pipeline->IsDone());
}

}  // namespace
}  // namespace jxl
