// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_render_pipeline.h"

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

TEST(RenderPipelineTest, BuildAndCheckPadding) {
  RenderPipeline::Builder builder({{1, 1}});
  builder.AddStage(make_unique<TrivialStage>(1, 1));  // shift, border
  builder.AddStage(make_unique<TrivialStage>(0, 2));  // no shift, border
  builder.AddStage(make_unique<FinalTrivialStage>());
  RenderPipeline pipeline =
      std::move(builder).Finalize(/*max_out_xsize=*/kGroupDim);
  // Shift in stage 1 should halve the border of stage 2.
  EXPECT_EQ(pipeline.Padding(), 2);
}

}  // namespace
}  // namespace jxl
