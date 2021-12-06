// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_RENDER_PIPELINE_STAGE_WRITE_TO_IB_H_
#define LIB_JXL_RENDER_PIPELINE_STAGE_WRITE_TO_IB_H_

#include "lib/jxl/image_bundle.h"
#include "lib/jxl/render_pipeline/render_pipeline_stage.h"

namespace jxl {

std::unique_ptr<RenderPipelineStage> GetWriteToImageBundleStage(
    ImageBundle* image_bundle);
}

#endif  // LIB_JXL_RENDER_PIPELINE_STAGE_WRITE_TO_IB_H_
