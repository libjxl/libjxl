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

#ifndef JXL_SPLINES_H_
#define JXL_SPLINES_H_

#include <stddef.h>
#include <stdint.h>

#include <utility>
#include <vector>

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/status.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_ans.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/entropy_coder.h"
#include "jxl/image.h"

namespace jxl {

struct Spline {
  struct Point {
    float x, y;
  };
  std::vector<Point> control_points;
  // X, Y, B.
  float color_dct[3][32];
  // Splines are draws by normalized Gaussian splatting. This controls the
  // Gaussian's parameter along the spline.
  float sigma_dct[32];
};

class QuantizedSpline {
 public:
  QuantizedSpline() = default;
  explicit QuantizedSpline(const Spline& original,
                           int32_t quantization_adjustment, float ytox,
                           float ytob);

  Spline Dequantize(const Spline::Point& starting_point,
                    int32_t quantization_adjustment, float ytox,
                    float ytob) const;

  void Tokenize(std::vector<Token>* tokens) const;

  void Decode(const std::vector<uint8_t>& context_map, ANSSymbolReader* decoder,
              BitReader* br);

 private:
  std::vector<std::pair<int64_t, int64_t>>
      control_points_;  // Double delta-encoded.
  int color_dct_[3][32] = {};
  int sigma_dct_[32] = {};
};

class Splines {
 public:
  Splines() = default;
  explicit Splines(const int32_t quantization_adjustment,
                   std::vector<QuantizedSpline> splines,
                   std::vector<Spline::Point> starting_points)
      : quantization_adjustment_(quantization_adjustment),
        splines_(std::move(splines)),
        starting_points_(std::move(starting_points)) {}

  bool HasAny() const { return !splines_.empty(); }

  // Only call if HasAny().
  void Encode(BitWriter* writer, size_t layer, AuxOut* aux_out) const;
  Status Decode(BitReader* br);

  void AddTo(Image3F* opsin, const Rect& opsin_rect, const Rect& image_rect,
             const ColorCorrelationMap& cmap) const;
  void SubtractFrom(Image3F* opsin, const ColorCorrelationMap& cmap) const;

  const std::vector<QuantizedSpline>& TestOnlyQuantizedSplines() const {
    return splines_;
  }
  const std::vector<Spline::Point>& TestOnlyStartingPoints() const {
    return starting_points_;
  }

 private:
  template <bool>
  void Apply(Image3F* opsin, const Rect& opsin_rect, const Rect& image_rect,
             const ColorCorrelationMap& cmap) const;

  // If positive, quantization weights are multiplied by 1 + this/8, which
  // increases precision. If negative, they are divided by 1 - this/8. If 0,
  // they are unchanged.
  int32_t quantization_adjustment_ = 0;
  std::vector<QuantizedSpline> splines_;
  std::vector<Spline::Point> starting_points_;
};

Splines FindSplines(const Image3F& image);

}  // namespace jxl

#endif  // JXL_SPLINES_H_
