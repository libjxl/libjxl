// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_render_pipeline.h"

#include <algorithm>

namespace jxl {

void RenderPipeline::Builder::AddStage(
    std::unique_ptr<RenderPipelineStage> stage) {
  stages_.push_back(std::move(stage));
}

RenderPipeline RenderPipeline::Builder::Finalize(size_t max_out_xsize) && {
#if JXL_ENABLE_ASSERT
  // Check channel shifts.
  auto channel_shifts = channel_shifts_;
  for (const auto& stage : stages_) {
    std::pair<size_t, size_t> current_shift = {-1, -1};
    for (size_t c = 0; c < channel_shifts.size(); c++) {
      if (stage->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
        if (current_shift.first == -1UL) current_shift = channel_shifts[c];
        JXL_ASSERT(current_shift == channel_shifts[c]);
        channel_shifts[c].first -= stage->settings_.shift_x;
        channel_shifts[c].second -= stage->settings_.shift_y;
      }
    }
  }
  for (const auto& cc : channel_shifts) {
    JXL_ASSERT(cc.first == 0);
    JXL_ASSERT(cc.second == 0);
  }
  // Check that the last stage is not an kInOut stage for any channel, and that
  // there is at least one stage.
  JXL_ASSERT(!stages_.empty());
  for (size_t c = 0; c < channel_shifts.size(); c++) {
    JXL_ASSERT(stages_.back()->GetChannelMode(c) !=
               RenderPipelineChannelMode::kInOut);
  }
#endif

  RenderPipeline res;
  res.padding_.resize(stages_.size());
  std::vector<size_t> channel_border(channel_shifts.size());
  for (size_t i = stages_.size(); i > 0; i--) {
    const auto& stage = stages_[i - 1];
    for (size_t c = 0; c < channel_shifts.size(); c++) {
      if (stage->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
        channel_border[c] = DivCeil(
            channel_border[c],
            1 << std::max(stage->settings_.shift_x, stage->settings_.shift_y));
        channel_border[c] +=
            std::max(stage->settings_.border_x, stage->settings_.border_y);
      }
    }
    res.padding_[i - 1] =
        *std::max_element(channel_border.begin(), channel_border.end());
  }
  res.num_c_ = channel_shifts.size();
  res.max_out_xsize_ = max_out_xsize;
  res.channel_shifts_.resize(stages_.size());
  res.channel_shifts_[0] = std::move(channel_shifts);
  for (size_t i = 1; i < stages_.size(); i++) {
    auto& stage = stages_[i - 1];
    res.channel_shifts_[i].resize(res.num_c_);
    for (size_t c = 0; c < res.num_c_; c++) {
      if (stage->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
        res.channel_shifts_[i][c].first =
            res.channel_shifts_[i - 1][c].first - stage->settings_.shift_x;
        res.channel_shifts_[i][c].second =
            res.channel_shifts_[i - 1][c].first - stage->settings_.shift_y;
      }
    }
  }
  res.stages_ = std::move(stages_);
  return res;
}

void RenderPipeline::PrepareForThreads(size_t num) {
  size_t cur = thread_memory_.size();
  if (cur > num) return;
  thread_memory_.resize(num);
  for (size_t i = cur; i < num; i++) {
    thread_memory_[i].row_info_output.resize(channel_shifts_.size());
    thread_memory_[i].row_info_input.resize(channel_shifts_.size());

    thread_memory_[i].y_ranges.resize(stages_.size());

    thread_memory_[i].rows.resize(
        stages_.size(),
        std::vector<std::vector<float*>>(channel_shifts_.size()));

    thread_memory_[i].buffers.resize(stages_.size());
    std::vector<size_t> next_border_y(num_c_);
    for (size_t j = stages_.size(); j > 0; j--) {
      auto& stage = stages_[j - 1];
      thread_memory_[i].buffers[j - 1].resize(num_c_);
      for (size_t c = 0; c < num_c_; c++) {
        if (stage->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
          // TODO(veluca): figure out a better upper bound than `max_out_xsize_`
          // when we are not yet fully upsampled.
          thread_memory_[i].buffers[j - 1][c] =
              ImageF(max_out_xsize_ + kRenderPipelineXOffset * 2,
                     next_border_y[c] * 2 + (1 << stage->settings_.shift_y));
          next_border_y[c] = stage->settings_.border_y;
        }
      }
    }
  }
}

void RenderPipeline::Run(const Input* input_data, size_t thread, size_t xpos,
                         size_t ypos, size_t final_frame_xsize,
                         size_t final_frame_ysize) {
  ThreadMemory& memory = thread_memory_[thread];
  size_t num_out_y = 0;
  for (size_t c = 0; c < num_c_; c++) {
    size_t noy = input_data[c].rect.ysize() << channel_shifts_[0][c].second;
    if (num_out_y == 0) num_out_y = noy;
    JXL_ASSERT(num_out_y == noy);
  }
  // Prepare input.
  for (size_t c = 0; c < num_c_; c++) {
    size_t y_begin = ypos == 0 ? ypos : ypos - Padding();
    size_t y_end = std::min(ypos + input_data[c].rect.ysize() + Padding(),
                            final_frame_ysize);
    memory.y_ranges[0] = {y_begin, y_end};
  }
  /*
  for (size_t i = 0; i < stages_.size() - 1; i++) {
    for (size_t c = 0; c < num_c_; c++) {
      if (stages_[i]->GetChannelMode(c) == RenderPipelineChannelMode::kInOut) {
      } else {
        //
      }
    }
  }
  */
}

}  // namespace jxl
