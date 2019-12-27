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

#ifndef JXL_PATCH_DICTIONARY_H_
#define JXL_PATCH_DICTIONARY_H_

// Chooses reference patches, and avoids encoding them once per occurrence.

#include <stddef.h>
#include <string.h>
#include <sys/types.h>

#include <tuple>
#include <vector>

#include "jxl/aux_out_fwd.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/common.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/enc_params.h"
#include "jxl/image.h"
#include "jxl/opsin_params.h"

namespace jxl {

constexpr size_t kMaxPatchSize = 32;

enum class PatchBlendMode : uint8_t {
  kAdd,
  kReplace,
  // TODO(veluca): Add kBlendAbove, kBlendBelow
  kNumBlendModes,
};

struct QuantizedPatch {
  size_t xsize;
  size_t ysize;
  int8_t pixels[3][kMaxPatchSize * kMaxPatchSize] = {};
  // Not compared. Used only to retrieve original pixels to construct the
  // reference image.
  float fpixels[3][kMaxPatchSize * kMaxPatchSize] = {};
  bool operator==(const QuantizedPatch& other) const {
    if (xsize != other.xsize) return false;
    if (ysize != other.ysize) return false;
    for (size_t c = 0; c < 3; c++) {
      if (memcmp(pixels[c], other.pixels[c], sizeof(int8_t) * xsize * ysize) !=
          0)
        return false;
    }
    return true;
  }

  bool operator<(const QuantizedPatch& other) const {
    if (xsize != other.xsize) return xsize < other.xsize;
    if (ysize != other.ysize) return ysize < other.ysize;
    for (size_t c = 0; c < 3; c++) {
      int cmp =
          memcmp(pixels[c], other.pixels[c], sizeof(int8_t) * xsize * ysize);
      if (cmp > 0) return false;
      if (cmp < 0) return true;
    }
    return false;
  }
};

// Pair (patch, vector of occurences).
using PatchInfo =
    std::pair<QuantizedPatch, std::vector<std::pair<uint32_t, uint32_t>>>;

// Position and size of the patch in the reference frame.
struct PatchReferencePosition {
  size_t ref, x0, y0, xsize, ysize;
  bool operator<(const PatchReferencePosition& oth) const {
    return std::make_tuple(ref, x0, y0, xsize, ysize) <
           std::make_tuple(oth.ref, oth.x0, oth.y0, oth.xsize, oth.ysize);
  }
  bool operator==(const PatchReferencePosition& oth) const {
    return !(*this < oth) && !(oth < *this);
  }
};

struct PatchPosition {
  // Position of top-left corner of the patch in the image.
  size_t x, y;
  PatchBlendMode blend_mode;
  PatchReferencePosition ref_pos;
  bool operator<(const PatchPosition& oth) const {
    return std::make_tuple(ref_pos, x, y) <
           std::make_tuple(oth.ref_pos, oth.x, oth.y);
  }
};

class PatchDictionary {
 public:
  PatchDictionary() = default;

  void SetReferenceFrames(const Image3F* JXL_RESTRICT reference_frames) {
    reference_frames_ = reference_frames;
  }
  void SetPositions(std::vector<PatchPosition> positions) {
    positions_ = std::move(positions);
    std::sort(positions_.begin(), positions_.end());
    ComputePatchCache();
  }

  bool HasAny() const { return !positions_.empty(); }
  // Only call if HasAny().
  void Encode(BitWriter* writer, size_t layer, AuxOut* aux_out) const;

  Status Decode(BitReader* br, size_t xsize, size_t ysize,
                size_t save_as_reference);

  // Only adds patches that belong to the `image_rect` area of the decoded
  // image, writing them to the `opsin_rect` area of `opsin`.
  void AddTo(Image3F* opsin, const Rect& opsin_rect,
             const Rect& image_rect) const;

  void SubtractFrom(Image3F* opsin) const;

 private:
  const Image3F* JXL_RESTRICT reference_frames_;
  std::vector<PatchPosition> positions_;

  // Patch occurences sorted by y.
  std::vector<size_t> sorted_patches_;
  // Index of the first patch for each y value.
  std::vector<size_t> patch_starts_;

  // Patch IDs in position [patch_starts_[y], patch_start_[y+1]) of
  // sorted_patches_ are all the patches that intersect the horizontal line at
  // y.

  // Compute sorted_patches_ and patch_start_ after updating positions_.
  void ComputePatchCache();

  template <bool>
  void Apply(Image3F* opsin, const Rect& opsin_rect,
             const Rect& image_rect) const;
};

// Avoid cyclic header inclusion.
struct PassesEncoderState;

void FindBestPatchDictionary(const Image3F& opsin,
                             PassesEncoderState* JXL_RESTRICT state,
                             ThreadPool* pool, AuxOut* aux_out,
                             bool is_xyb = true);

}  // namespace jxl

#endif  // JXL_PATCH_DICTIONARY_H_
