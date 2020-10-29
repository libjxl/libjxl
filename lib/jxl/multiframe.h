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

#ifndef LIB_JXL_MULTIFRAME_H_
#define LIB_JXL_MULTIFRAME_H_

#include <stddef.h>
#include <stdint.h>

#include <limits>
#include <memory>
#include <vector>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_util.h"
#include "lib/jxl/dot_dictionary.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"
#include "lib/jxl/patch_dictionary.h"
#include "lib/jxl/splines.h"

// A multiframe handler/manager to encode single images. It will run heuristics
// for quantization, AC strategy and color correlation map only the first time
// we want to encode a lossy pass, and will then re-use the existing heuristics
// for further passes. All the passes of a single image are added together.

namespace jxl {

constexpr size_t kNoDownsamplingFactor = std::numeric_limits<size_t>::max();

struct PassDefinition {
  // Side of the square of the coefficients that should be kept in each 8x8
  // block. Must be greater than 1, and at most 8. Should be in non-decreasing
  // order.
  size_t num_coefficients;

  // How much to shift the encoded values by, with rounding.
  size_t shift;

  // Whether or not we should include only salient blocks.
  // TODO(veluca): ignored for now.
  bool salient_only;

  // If specified, this indicates that if the requested downsampling factor is
  // sufficiently high, then it is fine to stop decoding after this pass.
  // By default, passes are not marked as being suitable for any downsampling.
  size_t suitable_for_downsampling_of_at_least;
};

struct ProgressiveMode {
  size_t num_passes = 1;
  PassDefinition passes[kMaxNumPasses] = {PassDefinition{
      /*num_coefficients=*/8, /*shift=*/0, /*salient_only=*/false,
      /*suitable_for_downsampling_of_at_least=*/1}};

  ProgressiveMode() = default;

  template <size_t nump>
  explicit ProgressiveMode(const PassDefinition (&p)[nump]) {
    JXL_ASSERT(nump <= kMaxNumPasses);
    num_passes = nump;
    PassDefinition previous_pass{
        /*num_coefficients=*/1, /*shift=*/0,
        /*salient_only=*/false,
        /*suitable_for_downsampling_of_at_least=*/kNoDownsamplingFactor};
    size_t last_downsampling_factor = kNoDownsamplingFactor;
    for (size_t i = 0; i < nump; i++) {
      JXL_ASSERT(p[i].num_coefficients > previous_pass.num_coefficients ||
                 (p[i].num_coefficients == previous_pass.num_coefficients &&
                  !p[i].salient_only && previous_pass.salient_only) ||
                 (p[i].num_coefficients == previous_pass.num_coefficients &&
                  p[i].shift < previous_pass.shift));
      JXL_ASSERT(p[i].suitable_for_downsampling_of_at_least ==
                     kNoDownsamplingFactor ||
                 p[i].suitable_for_downsampling_of_at_least <=
                     last_downsampling_factor);
      if (p[i].suitable_for_downsampling_of_at_least != kNoDownsamplingFactor) {
        last_downsampling_factor = p[i].suitable_for_downsampling_of_at_least;
      }
      previous_pass = passes[i] = p[i];
    }
  }
};

// Multiframe holds information about passes and manages
// MultiframeHandlers. It is assumed that parallelization goes below the manager
// level (at group level), so all the methods of Multiframe should be
// invoked from a single thread.
class Multiframe {
 public:
  Multiframe() : current_header_(nullptr) {}

  // Called at the start of each frame.
  void StartFrame(const FrameHeader& frame_header) {
    current_header_ = frame_header;
  }

  bool NeedsSaving() {
    return current_header_.dc_level != 0 ||
           (current_header_.save_as_reference != 0) ||
           (!current_header_.animation_frame.is_last &&
            !current_header_.animation_frame.have_crop &&
            current_header_.animation_frame.new_base == NewBase::kCurrentFrame);
  }

  bool NeedsRestoring() {
    return current_header_.dc_level == 0 &&
           current_header_.save_as_reference == 0 &&
           !current_header_.animation_frame.have_crop && has_previous_frame_;
  }

  Image3F* FrameStorage(size_t xsize, size_t ysize) {
    if (!NeedsSaving() && !NeedsRestoring()) return nullptr;
    if (current_header_.dc_level != 0) {
      dc_storage_[current_header_.dc_level - 1] = Image3F(xsize, ysize);
      return &dc_storage_[current_header_.dc_level - 1];
    }
    if (current_header_.save_as_reference != 0) {
      reference_frames_[current_header_.save_as_reference - 1] =
          Image3F(xsize, ysize);
      return &reference_frames_[current_header_.save_as_reference - 1];
    }
    if (!NeedsRestoring()) {
      frame_storage_ = Image3F(xsize, ysize);
    }
    JXL_CHECK(frame_storage_.xsize() == xsize);
    JXL_CHECK(frame_storage_.ysize() == ysize);
    return &frame_storage_;
  }

  const Image3F* SavedDc(size_t level) const { return &dc_storage_[level - 1]; }

  // Called when a frame is done.
  void SetDecodedFrame() {
    if (!IsDisplayed()) return;
    if (!current_header_.animation_frame.is_last &&
        !current_header_.animation_frame.have_crop) {
      switch (current_header_.animation_frame.new_base) {
        case NewBase::kExisting:
          break;
        case NewBase::kCurrentFrame:
          has_previous_frame_ = true;
          break;
        case NewBase::kNone:
          frame_storage_ = Image3F();
          has_previous_frame_ = false;
          break;
      }
    } else {
      if (current_header_.animation_frame.new_base != NewBase::kExisting) {
        frame_storage_ = Image3F();
        has_previous_frame_ = false;
      }
    }
  }

  // Modifies img by subtracting the current reference frame.
  void DecorrelateOpsin(Image3F* img) {
    if (NeedsRestoring()) {
      SubtractFrom(frame_storage_, img);
    }
  }

  void SetProgressiveMode(ProgressiveMode mode) { mode_ = mode; }

  void SetSaliencyMap(const ImageF* saliency_map) {
    saliency_map_ = saliency_map;
  }

  void SetSaliencyThreshold(float threshold) {
    saliency_threshold_ = threshold;
  }

  size_t GetNumPasses() const { return mode_.num_passes; }

  void InitPasses(Passes* JXL_RESTRICT passes) const {
    passes->num_passes = static_cast<uint32_t>(GetNumPasses());
    passes->num_downsample = 0;
    JXL_ASSERT(passes->num_passes != 0);
    if (passes->num_passes == 1) return;  // Done, arrays are empty

    for (uint32_t i = 0; i < mode_.num_passes - 1; ++i) {
      const size_t min_downsampling_factor =
          mode_.passes[i].suitable_for_downsampling_of_at_least;
      passes->shift[i] = mode_.passes[i].shift;
      if (1 < min_downsampling_factor &&
          min_downsampling_factor != kNoDownsamplingFactor) {
        passes->downsample[passes->num_downsample] = min_downsampling_factor;
        passes->last_pass[passes->num_downsample] = i;
        passes->num_downsample += 1;
      }
    }
  }

  void SplitACCoefficients(const ac_qcoeff_t* JXL_RESTRICT block, size_t size,
                           const AcStrategy& acs, size_t bx, size_t by,
                           size_t offset,
                           ac_qcoeff_t* JXL_RESTRICT output[kMaxNumPasses][3]);

  const Image3F* GetReferenceFrames() const { return reference_frames_; }

  bool IsDisplayed() const { return current_header_.IsDisplayed(); }

 private:
  friend class MultiframeHandler;

  bool SuperblockIsSalient(size_t row_start, size_t col_start, size_t num_rows,
                           size_t num_cols) const;

  FrameHeader current_header_;
  ProgressiveMode mode_;

  // Not owned, must remain valid.
  const ImageF* saliency_map_ = nullptr;
  float saliency_threshold_ = 0.0;

  Image3F frame_storage_;
  Image3F dc_storage_[3];
  Image3F reference_frames_[kMaxNumReferenceFrames];
  bool has_previous_frame_ = false;
};

}  // namespace jxl

#endif  // LIB_JXL_MULTIFRAME_H_
