// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

class UpsampleXSlowStage : public RenderPipelineStage {
 public:
  UpsampleXSlowStage()
      : RenderPipelineStage(RenderPipelineStage::Settings::ShiftX(1, 1)) {}

  void ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                  size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                  float* JXL_RESTRICT temp) const final {
    for (size_t c = 0; c < input_rows.size(); c++) {
      const float* row = GetInputRow(input_rows, c, 0);
      float* row_out = GetOutputRow(output_rows, c, 0);
      for (int64_t x = -xextra; x < (int64_t)(xsize + xextra); x++) {
        float xp = row[x + kRenderPipelineXOffset - 1];
        float xc = row[x + kRenderPipelineXOffset];
        float xn = row[x + kRenderPipelineXOffset + 1];
        float xout0 = xp * 0.25f + xc * 0.75f;
        float xout1 = xc * 0.75f + xn * 0.25f;
        row_out[kRenderPipelineXOffset + 2 * x + 0] = xout0;
        row_out[kRenderPipelineXOffset + 2 * x + 1] = xout1;
      }
    }
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInOut;
  }
};

class UpsampleYSlowStage : public RenderPipelineStage {
 public:
  UpsampleYSlowStage()
      : RenderPipelineStage(RenderPipelineStage::Settings::ShiftY(1, 1)) {}

  void ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                  size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                  float* JXL_RESTRICT temp) const final {
    for (size_t c = 0; c < input_rows.size(); c++) {
      const float* rowp = GetInputRow(input_rows, c, -1);
      const float* rowc = GetInputRow(input_rows, c, 0);
      const float* rown = GetInputRow(input_rows, c, 1);
      float* row_out0 = GetOutputRow(output_rows, c, 0);
      float* row_out1 = GetOutputRow(output_rows, c, 1);
      for (int64_t x = -xextra; x < (int64_t)(xsize + xextra); x++) {
        float xp = rowp[x + kRenderPipelineXOffset];
        float xc = rowc[x + kRenderPipelineXOffset];
        float xn = rown[x + kRenderPipelineXOffset];
        float yout0 = xp * 0.25f + xc * 0.75f;
        float yout1 = xc * 0.75f + xn * 0.25f;
        row_out0[kRenderPipelineXOffset + x] = yout0;
        row_out1[kRenderPipelineXOffset + x] = yout1;
      }
    }
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInOut;
  }
};

class Check0FinalStage : public RenderPipelineStage {
 public:
  Check0FinalStage() : RenderPipelineStage(RenderPipelineStage::Settings()) {}

  void ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                  size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                  float* JXL_RESTRICT temp) const final {
    for (size_t c = 0; c < input_rows.size(); c++) {
      for (size_t x = 0; x < xsize; x++) {
        JXL_CHECK(fabsf(GetInputRow(input_rows, c,
                                    0)[x + kRenderPipelineXOffset]) < 1e-8);
      }
    }
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInput;
  }
};

}  // namespace jxl
