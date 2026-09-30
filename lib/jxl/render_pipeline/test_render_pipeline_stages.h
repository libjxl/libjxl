// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cmath>
#include <cstdint>
#include <cstdio>

#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

class UpsampleXSlowStage : public RenderPipelineStage {
 public:
  UpsampleXSlowStage()
      : RenderPipelineStage(
            RenderPipelineStage::Settings::ShiftX(/*shift=*/1, /*border=*/1)) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    ptrdiff_t x_start = -static_cast<ptrdiff_t>(xextra_left);
    ptrdiff_t x_end = static_cast<ptrdiff_t>(xsize + xextra_right);
    for (size_t c = 0; c < input_rows.size(); c++) {
      const float* row = GetInputRow(input_rows, c, 0);
      float* row_out = GetOutputRow(output_rows, c, 0);
      for (ptrdiff_t x = x_start; x < x_end; x++) {
        float xp = *(row + x - 1);
        float xc = *(row + x);
        float xn = *(row + x + 1);
        float xout0 = xp * 0.25f + xc * 0.75f;
        float xout1 = xc * 0.75f + xn * 0.25f;
        *(row_out + 2 * x + 0) = xout0;
        *(row_out + 2 * x + 1) = xout1;
      }
    }
    return true;
  }

  const char* GetName() const override { return "TEST::UpsampleXSlowStage"; }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInOut;
  }
};

class UpsampleYSlowStage : public RenderPipelineStage {
 public:
  UpsampleYSlowStage()
      : RenderPipelineStage(
            RenderPipelineStage::Settings::ShiftY(/*shift=*/1, /*border=*/1)) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    ptrdiff_t x_start = -static_cast<ptrdiff_t>(xextra_left);
    ptrdiff_t x_end = static_cast<ptrdiff_t>(xsize + xextra_right);
    for (size_t c = 0; c < input_rows.size(); c++) {
      const float* rowp = GetInputRow(input_rows, c, -1);
      const float* rowc = GetInputRow(input_rows, c, 0);
      const float* rown = GetInputRow(input_rows, c, 1);
      float* row_out0 = GetOutputRow(output_rows, c, 0);
      float* row_out1 = GetOutputRow(output_rows, c, 1);
      for (ptrdiff_t x = x_start; x < x_end; x++) {
        float xp = *(rowp + x);
        float xc = *(rowc + x);
        float xn = *(rown + x);
        float y_out0 = xp * 0.25f + xc * 0.75f;
        float y_out1 = xc * 0.75f + xn * 0.25f;
        *(row_out0 + x) = y_out0;
        *(row_out1 + x) = y_out1;
      }
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInOut;
  }

  const char* GetName() const override { return "TEST::UpsampleYSlowStage"; }
};

class Check0FinalStage : public RenderPipelineStage {
 public:
  Check0FinalStage() : RenderPipelineStage(RenderPipelineStage::Settings()) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    ptrdiff_t x_start = -static_cast<ptrdiff_t>(xextra_left);
    ptrdiff_t x_end = static_cast<ptrdiff_t>(xsize + xextra_right);
    for (size_t c = 0; c < input_rows.size(); c++) {
      float* JXL_RESTRICT row = GetInputRow(input_rows, c, 0);
      for (ptrdiff_t x = x_start; x < x_end; x++) {
        JXL_ENSURE(std::abs(row[x]) < 1e-8);
      }
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return RenderPipelineChannelMode::kInput;
  }
  const char* GetName() const override { return "TEST::Check0FinalStage"; }
};

/* Fake stages quickly copy input to output. */
class FakeVChromaUpsampleStage : public RenderPipelineStage {
 public:
  explicit FakeVChromaUpsampleStage(size_t c)
      : RenderPipelineStage(RenderPipelineStage::Settings().ShiftY(
            /*shift=*/1, /*border=*/1)),
        c_(c) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    const float* row_top = GetInputRow(input_rows, c_, -1);
    const float* row_mid = GetInputRow(input_rows, c_, 0);
    const float* row_bot = GetInputRow(input_rows, c_, 1);
    float* row_out0 = GetOutputRow(output_rows, c_, 0);
    float* row_out1 = GetOutputRow(output_rows, c_, 1);
    size_t num_bytes = sizeof(float) * (xextra_left + xsize + xextra_right);
    msan::MemoryIsInitialized(row_top - xextra_left, num_bytes);
    msan::MemoryIsInitialized(row_mid - xextra_left, num_bytes);
    msan::MemoryIsInitialized(row_bot - xextra_left, num_bytes);
    memcpy(row_out0 - xextra_left, row_mid - xextra_left, num_bytes);
    memcpy(row_out1 - xextra_left, row_mid - xextra_left, num_bytes);
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c == c_ ? RenderPipelineChannelMode::kInOut
                   : RenderPipelineChannelMode::kIgnored;
  }
  const char* GetName() const override { return "FakeVChromaUpsampleStage"; }

 private:
  size_t c_;
};

class FakeHChromaUpsampleStage : public RenderPipelineStage {
 public:
  explicit FakeHChromaUpsampleStage(size_t c)
      : RenderPipelineStage(RenderPipelineStage::Settings().ShiftX(
            /*shift=*/1, /*border=*/1)),
        c_(c) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    const float* row_in = GetInputRow(input_rows, c_, 0);
    float* row_out = GetOutputRow(output_rows, c_, 0);
    size_t num_bytes = sizeof(float) * (xextra_left + xsize + xextra_right);
    msan::MemoryIsInitialized(row_in - xextra_left - 1,
                              num_bytes + 2 * sizeof(float));
    intptr_t x_start = -static_cast<intptr_t>(xextra_left);
    intptr_t x_end = static_cast<intptr_t>(xsize + xextra_right);
    for (intptr_t x = x_start; x < x_end; x++) {
      row_out[x * 2 + 0] = row_in[x];
      row_out[x * 2 + 1] = row_in[x];
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c == c_ ? RenderPipelineChannelMode::kInOut
                   : RenderPipelineChannelMode::kIgnored;
  }
  const char* GetName() const override { return "FakeHChromaUpsampleStage"; }

 private:
  size_t c_;
};

// Gaborish + EPF{0,1,2}
template <size_t border>
class FakeGaborishStage : public RenderPipelineStage {
 public:
  explicit FakeGaborishStage()
      : RenderPipelineStage(
            RenderPipelineStage::Settings().SymmetricBorderOnly(border)) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    static_assert(border >= 1 && border <= 3);
    size_t num_bytes = sizeof(float) * (xextra_left + xsize + xextra_right);
    for (size_t c = 0; c < 3; ++c) {
      const float* row_top = GetInputRow(input_rows, c, -1);
      const float* row_mid = GetInputRow(input_rows, c, 0);
      const float* row_bot = GetInputRow(input_rows, c, 1);
      float* row_out = GetOutputRow(output_rows, c, 0);
      msan::MemoryIsInitialized(row_top - xextra_left - border,
                                num_bytes + border * sizeof(float));
      msan::MemoryIsInitialized(row_mid - xextra_left - border,
                                num_bytes + border * sizeof(float));
      msan::MemoryIsInitialized(row_bot - xextra_left - border,
                                num_bytes + border * sizeof(float));
      memcpy(row_out - xextra_left, row_mid - xextra_left, num_bytes);
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c < 3 ? RenderPipelineChannelMode::kInOut
                 : RenderPipelineChannelMode::kIgnored;
  }

  const char* GetName() const override {
    return (border == 1)   ? "FakeGaborishStage1"
           : (border == 2) ? "FakeGaborishStage2"
                           : "FakeGaborishStage3";
  }
};

template <size_t shift>
class FakeUpsampleStage : public RenderPipelineStage {
 public:
  explicit FakeUpsampleStage(size_t c)
      : RenderPipelineStage(RenderPipelineStage::Settings().Symmetric(
            /*shift*/ shift, /*border*/ 2)),
        c_(c) {}

  Status ProcessRow(const RowInfo& input_rows, const RowInfo& output_rows,
                    size_t xextra_left, size_t xextra_right, size_t xsize,
                    size_t xpos, size_t ypos, size_t thread_id) const final {
    size_t num_bytes = sizeof(float) * (xextra_left + xsize + xextra_right);
    for (ptrdiff_t iy = -2; iy <= 2; ++iy) {
      const float* row_in = GetInputRow(input_rows, c_, iy);
      msan::MemoryIsInitialized(row_in - xextra_left - 2,
                                num_bytes + 4 * sizeof(float));
    }
    size_t bundle = 1 << shift;
    intptr_t x_start = -static_cast<intptr_t>(xextra_left);
    intptr_t x_end = static_cast<intptr_t>(xsize + xextra_right);
    for (size_t oy = 0; oy < bundle; ++oy) {
      const float* row_in = GetInputRow(input_rows, c_, 0);
      float* row_out = GetOutputRow(output_rows, c_, oy);
      for (intptr_t x = x_start; x < x_end; x++) {
        for (size_t ox = 0; ox < bundle; ++ox) {
          row_out[x * bundle + ox] = row_in[x];
        }
      }
    }
    return true;
  }

  RenderPipelineChannelMode GetChannelMode(size_t c) const final {
    return c == c_ ? RenderPipelineChannelMode::kInOut
                   : RenderPipelineChannelMode::kIgnored;
  }
  const char* GetName() const override { return "FakeUpsampleStage"; }

 private:
  size_t c_;
};

}  // namespace jxl
