// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "jxl/luminance.h"

#include "jxl/color_encoding.h"

namespace jxl {

namespace {

// Chooses a default intensity target based on the transfer function of the
// image, if known. For SDR images or images not known to be HDR, returns
// kDefaultIntensityTarget, for images known to have PQ or HLG transfer function
// returns a higher value.
float ChooseDefaultIntensityTarget(const ImageMetadata& metadata) {
  if (metadata.color_encoding.tf.IsPQ() || metadata.color_encoding.tf.IsHLG()) {
    // HDR
    return 4000;
  }
  // SDR
  return kDefaultIntensityTarget;
}

Status ScaleInLinearSpace(const float scaling_factor, ImageBundle* const ib,
                          ThreadPool* const pool) {
  JXL_ASSERT(!ib->IsJPEG());
  if (std::abs(scaling_factor - 1) < 1e-6) return true;
  if (ib->c_current().tf.IsLinear()) {
    ScaleImage(scaling_factor, ib->MutableColor());
    return true;
  }

  const ColorEncoding original_encoding = ib->c_current();
  ColorEncoding linear = original_encoding;
  linear.tf.SetTransferFunction(TransferFunction::kLinear);
  JXL_RETURN_IF_ERROR(ib->TransformTo(linear, pool));
  ScaleImage(scaling_factor, ib->MutableColor());
  return ib->TransformTo(original_encoding, pool);
}

}  // namespace

Status Map255ToTargetNits(CodecInOut* const io, ThreadPool* const pool) {
  int target_nits = io->target_nits;
  if (target_nits == 0) {
    target_nits = ChooseDefaultIntensityTarget(io->metadata);
  }
  io->metadata.SetIntensityTarget(target_nits);

  const float scaling_factor = target_nits * (1 / 255.f);
  for (ImageBundle& ib : io->frames) {
    JXL_RETURN_IF_ERROR(ScaleInLinearSpace(scaling_factor, &ib, pool));
  }
  return true;
}

Status Map255ToTargetNits(ImageBundle* const ib, ThreadPool* const pool) {
  const float scaling_factor = ib->metadata()->IntensityTarget() * (1 / 255.f);
  return ScaleInLinearSpace(scaling_factor, ib, pool);
}

Status MapTargetNitsTo255(ImageBundle* ib, ThreadPool* pool) {
  JXL_ASSERT(ib->metadata()->IntensityTarget() != 0);
  const float scaling_factor = 255.f / ib->metadata()->IntensityTarget();
  return ScaleInLinearSpace(scaling_factor, ib, pool);
}

Status MapTargetNitsTo255(CodecInOut* io, ThreadPool* pool) {
  JXL_ASSERT(io->metadata.IntensityTarget() != 0);
  const float scaling_factor = 255.f / io->metadata.IntensityTarget();
  for (ImageBundle& ib : io->frames) {
    JXL_RETURN_IF_ERROR(ScaleInLinearSpace(scaling_factor, &ib, pool));
  }
  return true;
}

}  // namespace jxl
