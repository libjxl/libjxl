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

#ifndef JXL_COLOR_MANAGEMENT_H_
#define JXL_COLOR_MANAGEMENT_H_

// ICC profiles and color space conversions.

#include <stddef.h>

#include <vector>

#include "jxl/base/padded_bytes.h"
#include "jxl/base/status.h"
#include "jxl/color_encoding.h"
#include "jxl/common.h"
#include "jxl/image.h"
#if JPEGXL_ENABLE_SKCMS
#include "skcms.h"
#endif

namespace jxl {

// Run is thread-safe.
class ColorSpaceTransform {
 public:
  ColorSpaceTransform() = default;
  ~ColorSpaceTransform();

  // Cannot copy (transforms_ holds pointers).
  ColorSpaceTransform(const ColorSpaceTransform&) = delete;
  ColorSpaceTransform& operator=(const ColorSpaceTransform&) = delete;

  // "Constructor"; allocates for up to `num_threads`, or returns false.
  Status Init(const ColorEncoding& c_src, const ColorEncoding& c_dst,
              size_t xsize, size_t num_threads);

  float* BufSrc(const size_t thread) { return buf_src_.Row(thread); }

  float* BufDst(const size_t thread) { return buf_dst_.Row(thread); }

  // buf_X can either be from BufX() or caller-allocated, interleaved storage.
  // `thread` must be less than the `num_threads` passed to Init.
  void Run(size_t thread, const float* buf_src, float* buf_dst);

 private:
  enum class ExtraTF {
    kNone,
    kPQ,
    kHLG,
    kSRGB,
  };

#if JPEGXL_ENABLE_SKCMS
  // Parsed skcms_ICCProfiles retain pointers to the original data.
  PaddedBytes icc_src_, icc_dst_;
  skcms_ICCProfile profile_src_, profile_dst_;
#else
  // One per thread - cannot share because of caching.
  std::vector<void*> transforms_;
#endif

  ImageF buf_src_;
  ImageF buf_dst_;
  size_t xsize_;
  bool skip_lcms_ = false;
  ExtraTF preprocess_ = ExtraTF::kNone;
  ExtraTF postprocess_ = ExtraTF::kNone;
};

}  // namespace jxl

#endif  // JXL_COLOR_MANAGEMENT_H_
