// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DEC_RENDER_PIPELINE_H_
#define LIB_JXL_DEC_RENDER_PIPELINE_H_

#include <stdint.h>

#include "lib/jxl/filters.h"

namespace jxl {

// The first pixel in the input to RenderPipelineStage will be located at
// this position. Pixels before this position may be accessed as padding.
constexpr size_t kRenderPipelineXOffset = 16;

enum class RenderPipelineChannelMode {
  kIgnored = 0,
  kInPlace = 1,
  kInOut = 2,
  kInput = 3,
};

class RenderPipeline;

class RenderPipelineStage {
 protected:
  using Row = float*;
  using ChannelRows = const Row*;
  using RowInfo = const ChannelRows*;

 public:
  struct Settings {
    // Amount of padding required in the various directions by all channels
    // that have kInOut mode.
    size_t border_x = 0;
    size_t border_y = 0;

    // Log2 of the number of columns/rows of output that this stage will produce
    // for every input row for kInOut channels.
    size_t shift_x = 0;
    size_t shift_y = 0;

    // Size (in floats) of the (aligned) per-thread temporary buffer to pass to
    // ProcessRow.
    size_t temp_buffer_size = 0;

    static Settings ShiftX(size_t shift, size_t border) {
      Settings settings;
      settings.border_x = border;
      settings.shift_x = shift;
      return settings;
    }

    static Settings ShiftY(size_t shift, size_t border) {
      Settings settings;
      settings.border_x = border;
      settings.shift_x = shift;
      return settings;
    }

    static Settings Symmetric(size_t shift, size_t border,
                              size_t temp_buffer_size = 0) {
      Settings settings;
      settings.border_x = settings.border_y = border;
      settings.shift_x = settings.shift_y = shift;
      return settings;
    }

    static Settings SymmetricBorderOnly(size_t border) {
      return Symmetric(0, border);
    }
  };

  virtual ~RenderPipelineStage() = default;

 protected:
  // Process one row of input, producing the appropriate number of rows of
  // output. Input/output rows can be obtained by calls to
  // `GetInputRow`/`GetOutputRow`. `xsize+2*xextra` represents the total number
  // of pixels to be processed in the input row, where the first pixel is at
  // position `kRenderPipelineXOffset-xextra`. All pixels in the
  // `[kRenderPipelineXOffset-xextra-border_x,
  // kRenderPipelineXOffset+xsize+xextra+borderx)` range are initialized and
  // accessible. `xpos` and `ypos` represent the position of the first
  // (non-extra, i.e. in position kRenderPipelineXOffset) pixel in the center
  // row of the input in the full image. `xpos` is a multiple of
  // `GroupBorderAssigner::kPaddingXRound`. If `temp_buffer_size` is nonzero,
  // `temp` will point to an aligned buffer of at least that number of floats;
  // concurrent calls will have different buffers.
  virtual void ProcessRow(RowInfo input_rows, RowInfo output_rows,
                          size_t xextra, size_t xsize, size_t xpos, size_t ypos,
                          float* JXL_RESTRICT temp) const = 0;

  // How each channel will be processed. Channels are numbered starting from
  // color channels (always 3) and followed by all other channels.
  virtual RenderPipelineChannelMode GetChannelMode(size_t c) const = 0;

  explicit RenderPipelineStage(Settings settings) : settings_(settings) {}

  // Returns a pointer to the input row of channel `c` with offset `y`.
  // `y` must be in [-settings_.border_y, settings_.border_y]. `c` must be such
  // that `GetChannelMode(c) != kIgnored`. The returned pointer points to the
  // *beginning* of the row.
  float* GetInputRow(RowInfo input_rows, size_t c, int offset) const {
    JXL_DASSERT(GetChannelMode(c) != RenderPipelineChannelMode::kIgnored);
    JXL_DASSERT(-offset >= static_cast<int>(settings_.border_y));
    JXL_DASSERT(offset <= static_cast<int>(settings_.border_y));
    return input_rows[c][settings_.border_y + offset];
  }
  // Similar to `GetOutputRow`, but can only be used if `GetChannelMode(c) ==
  // kInOut`. Offset must be less than `1<<settings_.shift_y`.
  float* GetOutputRow(RowInfo output_rows, size_t c, size_t offset) const {
    JXL_DASSERT(GetChannelMode(c) == RenderPipelineChannelMode::kInOut);
    JXL_DASSERT(offset <= 1 << settings_.shift_y);
    return output_rows[c][offset];
  }

 private:
  const Settings settings_;
  friend class RenderPipeline;
};

class RenderPipeline {
  struct ThreadMemory {
    // Storage buffers for every stage and every channel. More precisely,
    // `buffers[i][c]` contains the storage for the `c`-th channel of the `i`-th
    // stage.
    std::vector<std::vector<ImageF>> buffers;
    // Buffers for storing pointers to input/output rows.
    // `rows[i][c][r]` points to the `r`-th row of channel `c` of the input to
    // stage `i`.
    // The innermost vectors may be resized in calls to Run.
    std::vector<std::vector<std::vector<float*>>> rows;
    // Buffer to be used as RowInfo for calls to single stages.
    // `row_info_(in|out)put.data()` should be used as the RowInfo vector for
    // calls to a stage.
    std::vector<float* const*> row_info_input;
    std::vector<float* const*> row_info_output;
    // Storage for Y ranges at every stage.
    std::vector<std::pair<size_t, size_t>> y_ranges;
  };

 public:
  class Builder {
   public:
    // Initial shifts for the channels (following the same convention as
    // RenderPipelineStage for naming the channels).
    explicit Builder(std::vector<std::pair<size_t, size_t>> channel_shifts)
        : channel_shifts_(std::move(channel_shifts)) {}

    // Adds a stage to the pipeline. Must be called at least once; the last
    // added stage cannot have kInOut channels.
    void AddStage(std::unique_ptr<RenderPipelineStage> stage);

    // Finalizes setup of the pipeline. Shifts for all channels should be 0 at
    // this point.
    RenderPipeline Finalize(size_t max_out_xsize) &&;

   private:
    std::vector<std::unique_ptr<RenderPipelineStage>> stages_;
    std::vector<std::pair<size_t, size_t>> channel_shifts_;
  };

  friend class Builder;

  // Valid pixels needed on each side of the input (unless the input is on image
  // borders) for running this pipeline.
  size_t Padding() const { return padding_[0]; }

  // Allocates storage to run with `num` threads.
  void PrepareForThreads(size_t num);

  // A single channel of input to the pipeline. Image data is assumed to be in
  // `image:rect`, with a border of `Padding()` pixels around it being either
  // valid data or outside the image's frame. Pixels outside the valid
  // pixels may be accessed along the `x` direction, so sufficient padding
  // should be provided (`RoundUpTo(Padding(), kPaddingXRound) + kPaddingXRound`
  // pixels are always sufficient).
  struct Input {
    ImageF* image;
    Rect rect;
  };

  // Runs the pipeline on the given input data, which is assumed to correspond
  // to the portion of the full frame starting at `xpos`, `ypos`.
  // Channels should have a relative size that is adequate for the
  // `channel_shifts` provided at the input, i.e. such that all channel sizes
  // scaled up by the channel shifts are equal (up to rounding).
  void Run(const Input* input_data, size_t thread, size_t xpos, size_t ypos,
           size_t final_frame_xsize, size_t final_frame_ysize);

 private:
  std::vector<std::unique_ptr<RenderPipelineStage>> stages_;
  // Shifts for every channel at the input of each stage.
  std::vector<std::vector<std::pair<size_t, size_t>>> channel_shifts_;
  // Amount of (cumulative) padding required by each stage.
  std::vector<size_t> padding_;
  size_t max_out_xsize_ = 0;
  size_t num_c_ = 0;
  std::vector<ThreadMemory> thread_memory_;
};

}  // namespace jxl

#endif  // LIB_JXL_DEC_RENDER_PIPELINE_H_
