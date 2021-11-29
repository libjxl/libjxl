// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/render_pipeline/simple_render_pipeline.h"

namespace jxl {
void SimpleRenderPipeline::PrepareForThreads(size_t num) {
  // TODO(veluca): actually allocate input buffers.
}

std::vector<std::pair<ImageF*, Rect>> SimpleRenderPipeline::PrepareBuffers(
    size_t group_id, size_t thread_id) {
  return {};
}

void SimpleRenderPipeline::ProcessBuffers(size_t group_id, size_t thread_id) {
  // TODO(veluca): actually run the pipeline.
}
}  // namespace jxl
