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

#ifndef JXL_LUMINANCE_H_
#define JXL_LUMINANCE_H_

#include "jxl/codec_in_out.h"

namespace jxl {

// Expects a CodecInOut with a nominal white encoding value of (255, 255, 255)
// and scales the values so that white is remapped to (target_nits, target_nits,
// target_nits) in linear space, where target_nits = io->target_nits (or an
// appropriate value if io->target_nits is 0). Also sets the intensity target in
// io->metadata.
// 0 to 255 is the native storage range of many formats using 8 bits per sample.
Status Map255ToTargetNits(CodecInOut* io, ThreadPool* pool);

// Likewise, but takes an ImageBundle, and uses
// ib->metadata()->IntensityTarget() as the intensity target (and, of course,
// does not set it since it assumes that it is already set).
Status Map255ToTargetNits(ImageBundle* ib, ThreadPool* pool);

// Does the opposite, mapping
// (intensity_target, intensity_target, intensity_target) to (255, 255, 255) in
// linear space, where intensity_target = io->metadata.IntensityTarget().
Status MapTargetNitsTo255(CodecInOut* io, ThreadPool* pool);
Status MapTargetNitsTo255(ImageBundle* ib, ThreadPool* pool);

}  // namespace jxl

#endif
