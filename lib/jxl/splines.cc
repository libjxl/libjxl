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

#include "lib/jxl/splines.h"

#include <algorithm>

#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/common.h"
#include "lib/jxl/dct_scales.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/opsin_params.h"
#include "lib/jxl/splines_fastmath.h"

namespace jxl {

namespace {

constexpr float kDesiredRenderingDistance = 1.f;

// X, Y, B, sigma.
float ColorQuantizationWeight(const int32_t adjustment, const int channel,
                              const int i) {
  const float multiplier = adjustment >= 0 ? 1.f + .125f * adjustment
                                           : 1.f / (1.f + .125f * -adjustment);

  static constexpr float kChannelWeight[] = {0.0042f, 0.075f, 0.07f, .3333f};

  return multiplier / kChannelWeight[channel];
}

enum SplineEntropyContexts : size_t {
  kQuantizationAdjustmentContext = 0,
  kStartingPositionContext,
  kNumSplinesContext,
  kNumControlPointsContext,
  kControlPointsContext,
  kDCTContext,
  kNumSplineContexts
};

void EncodeAllStartingPoints(const std::vector<Spline::Point>& points,
                             std::vector<Token>* tokens) {
  int64_t last_x = 0;
  int64_t last_y = 0;
  for (size_t i = 0; i < points.size(); i++) {
    const int64_t x = std::lround(points[i].x);
    const int64_t y = std::lround(points[i].y);
    if (i == 0) {
      tokens->emplace_back(kStartingPositionContext, x);
      tokens->emplace_back(kStartingPositionContext, y);
    } else {
      tokens->emplace_back(kStartingPositionContext, PackSigned(x - last_x));
      tokens->emplace_back(kStartingPositionContext, PackSigned(y - last_y));
    }
    last_x = x;
    last_y = y;
  }
}

Status DecodeAllStartingPoints(std::vector<Spline::Point>* const points,
                               BitReader* const br, ANSSymbolReader* reader,
                               const std::vector<uint8_t>& context_map,
                               size_t num_splines) {
  points->resize(num_splines);
  int64_t last_x = 0;
  int64_t last_y = 0;
  for (size_t i = 0; i < points->size(); i++) {
    int64_t x =
        reader->ReadHybridUint(kStartingPositionContext, br, context_map);
    int64_t y =
        reader->ReadHybridUint(kStartingPositionContext, br, context_map);
    if (i != 0) {
      x = UnpackSigned(x) + last_x;
      y = UnpackSigned(y) + last_y;
    }
    (*points)[i].x = x;
    (*points)[i].y = y;
    last_x = x;
    last_y = y;
  }
  return true;
}

struct Vector {
  float x, y;
  Vector operator-() const { return {-x, -y}; }
  Vector operator+(const Vector& other) const {
    return {x + other.x, y + other.y};
  }
  float SquaredNorm() const { return x * x + y * y; }
};
Vector operator*(const float k, const Vector& vec) {
  return {k * vec.x, k * vec.y};
}

Spline::Point operator+(const Spline::Point& p, const Vector& vec) {
  return {p.x + vec.x, p.y + vec.y};
}
Spline::Point operator-(const Spline::Point& p, const Vector& vec) {
  return p + -vec;
}
Vector operator-(const Spline::Point& a, const Spline::Point& b) {
  return {a.x - b.x, a.y - b.y};
}

std::vector<Spline::Point> DrawCentripetalCatmullRomSpline(
    std::vector<Spline::Point> points) {
  if (points.size() <= 1) return points;
  // Number of points to compute between each control point.
  static constexpr int kNumPoints = 16;
  std::vector<Spline::Point> result;
  result.reserve((points.size() - 1) * kNumPoints + 1);
  points.insert(points.begin(), points[0] + (points[0] - points[1]));
  points.push_back(points[points.size() - 1] +
                   (points[points.size() - 1] - points[points.size() - 2]));
  // points has at least 4 elements at this point.
  for (size_t start = 0; start < points.size() - 3; ++start) {
    // 4 of them are used, and we draw from p[1] to p[2].
    const Spline::Point* const p = &points[start];
    result.push_back(p[1]);
    float t[4] = {0};
    for (int k = 1; k < 4; ++k) {
      t[k] = std::sqrt(std::hypot(p[k].x - p[k - 1].x, p[k].y - p[k - 1].y)) +
             t[k - 1];
    }
    for (int i = 1; i < kNumPoints; ++i) {
      const float tt =
          t[1] + (static_cast<float>(i) / kNumPoints) * (t[2] - t[1]);
      Spline::Point a[3];
      for (int k = 0; k < 3; ++k) {
        a[k] = p[k] + ((tt - t[k]) / (t[k + 1] - t[k])) * (p[k + 1] - p[k]);
      }
      Spline::Point b[2];
      for (int k = 0; k < 2; ++k) {
        b[k] = a[k] + ((tt - t[k]) / (t[k + 2] - t[k])) * (a[k + 1] - a[k]);
      }
      result.push_back(b[0] + ((tt - t[1]) / (t[2] - t[1])) * (b[1] - b[0]));
    }
  }
  result.push_back(points[points.size() - 2]);
  return result;
}

// Given a set of DCT coefficients, this returns the result of performing cosine
// interpolation on the original samples.
template <int N>
float ContinuousIDCT(const float dct[N], float t) {
  // We compute here the DCT-3 of the `dct` vector, rescaled by a factor of
  // sqrt(32). This is such that an input vector vector {x, 0, ..., 0} produces
  // a constant result of x.
  float result = dct[0];
  for (int i = 1; i < N; ++i) {
    result += square_root<2>::value * dct[i] *
              splines_internal::Cos((kPi / N) * i * (t + 0.5f));
  }
  return result;
}

// Used for Gaussian splatting. This gives the intensity of the Gaussian for a
// given distance from its center.
float BrushIntensity(const float distance, const float sigma) {
  const float one_dimensional_delta = (1.f / 1.4142135623730951f) * distance;
  const float inv_sqrt2_times_sigma = 1.f / (1.4142135623730951f * sigma);
  const float one_dimensional_factor =
      splines_internal::Erf((one_dimensional_delta + .5f) *
                            inv_sqrt2_times_sigma) -
      splines_internal::Erf((one_dimensional_delta - .5f) *
                            inv_sqrt2_times_sigma);
  return .25f * sigma * one_dimensional_factor * one_dimensional_factor;
}

// Splats a single Gaussian on the image.
void DrawGaussian(Image3F* const opsin, const Rect& opsin_rect,
                  const Rect& image_rect, const Spline::Point& center,
                  const float intensity, const float color[3],
                  const float sigma) {
  constexpr float kDistanceMultiplier = 4.605170185988091f;  // -2 * log(0.1)
  // Distance beyond which exp(-d^2 / (2 * sigma^2)) drops below 0.1.
  const float maximum_distance = sigma * sigma * kDistanceMultiplier;
  const auto xbegin =
      static_cast<size_t>(std::max(0.f, center.x - maximum_distance + .5f));
  const auto xend =
      std::min<size_t>(center.x + maximum_distance + .5f, opsin->xsize() - 1);
  const auto ybegin =
      static_cast<size_t>(std::max(0.f, center.y - maximum_distance + .5f));
  const auto yend =
      std::min<size_t>(center.y + maximum_distance + .5f, opsin->ysize() - 1);
  for (size_t y = ybegin; y <= yend; ++y) {
    if (y < image_rect.y0() || y >= image_rect.y0() + image_rect.ysize())
      continue;
    float* JXL_RESTRICT rows[3] = {
        opsin_rect.PlaneRow(opsin, 0, y - image_rect.y0()),
        opsin_rect.PlaneRow(opsin, 1, y - image_rect.y0()),
        opsin_rect.PlaneRow(opsin, 2, y - image_rect.y0()),
    };
    for (size_t x = xbegin; x <= xend; ++x) {
      if (x < image_rect.x0() || x >= image_rect.x0() + image_rect.xsize())
        continue;
      const Spline::Point point{static_cast<float>(x), static_cast<float>(y)};
      const float distance = std::sqrt((point - center).SquaredNorm());
      const float local_intensity = intensity * BrushIntensity(distance, sigma);
      for (size_t c = 0; c < 3; ++c) {
        rows[c][x - image_rect.x0()] += local_intensity * color[c];
      }
    }
  }
}

// Move along the line segments defined by `points`, `kDesiredRenderingDistance`
// pixels at a time, and call `functor` with each point and the actual distance
// to the previous point (which will always be kDesiredRenderingDistance except
// possibly for the very last point).
template <typename Points, typename Functor>
void ForEachEquallySpacedPoint(const Points& points, const Functor& functor) {
  JXL_ASSERT(!points.empty());
  Spline::Point current = points.front();
  functor(current, kDesiredRenderingDistance);
  auto next = points.begin();
  while (next != points.end()) {
    const Spline::Point* previous = &current;
    float arclength_from_previous = 0.f;
    for (;;) {
      if (next == points.end()) {
        functor(*previous, arclength_from_previous);
        return;
      }
      const float arclength_to_next =
          std::sqrt((*next - *previous).SquaredNorm());
      if (arclength_from_previous + arclength_to_next >=
          kDesiredRenderingDistance) {
        current =
            *previous + ((kDesiredRenderingDistance - arclength_from_previous) /
                         arclength_to_next) *
                            (*next - *previous);
        functor(current, kDesiredRenderingDistance);
        break;
      }
      arclength_from_previous += arclength_to_next;
      previous = &*next;
      ++next;
    }
  }
}

}  // namespace

QuantizedSpline::QuantizedSpline(const Spline& original,
                                 const int32_t quantization_adjustment,
                                 float ytox, float ytob) {
  JXL_ASSERT(!original.control_points.empty());
  control_points_.reserve(original.control_points.size() - 1);
  const Spline::Point& starting_point = original.control_points.front();
  int previous_x = static_cast<int>(std::round(starting_point.x)),
      previous_y = static_cast<int>(std::round(starting_point.y));
  int previous_delta_x = 0, previous_delta_y = 0;
  for (auto it = original.control_points.begin() + 1;
       it != original.control_points.end(); ++it) {
    const int new_x = static_cast<int>(std::round(it->x));
    const int new_y = static_cast<int>(std::round(it->y));
    const int new_delta_x = new_x - previous_x;
    const int new_delta_y = new_y - previous_y;
    control_points_.emplace_back(new_delta_x - previous_delta_x,
                                 new_delta_y - previous_delta_y);
    previous_delta_x = new_delta_x;
    previous_delta_y = new_delta_y;
    previous_x = new_x;
    previous_y = new_y;
  }

  for (int c = 0; c < 3; ++c) {
    float factor = c == 0 ? ytox : c == 1 ? 0 : ytob;
    for (int i = 0; i < 32; ++i) {
      const float coefficient =
          original.color_dct[c][i] -
          factor * color_dct_[1][i] /
              ColorQuantizationWeight(quantization_adjustment, 1, i);
      color_dct_[c][i] = static_cast<int>(
          std::round(coefficient *
                     ColorQuantizationWeight(quantization_adjustment, c, i)));
    }
  }
  for (int i = 0; i < 32; ++i) {
    sigma_dct_[i] = static_cast<int>(
        std::round(original.sigma_dct[i] *
                   ColorQuantizationWeight(quantization_adjustment, 3, i)));
  }
}

Spline QuantizedSpline::Dequantize(const Spline::Point& starting_point,
                                   const int32_t quantization_adjustment,
                                   float ytox, float ytob) const {
  Spline result;

  result.control_points.reserve(control_points_.size() + 1);
  int current_x = static_cast<int>(std::round(starting_point.x)),
      current_y = static_cast<int>(std::round(starting_point.y));
  result.control_points.push_back(Spline::Point{static_cast<float>(current_x),
                                                static_cast<float>(current_y)});
  int current_delta_x = 0, current_delta_y = 0;
  for (const auto& point : control_points_) {
    current_delta_x += point.first;
    current_delta_y += point.second;
    current_x += current_delta_x;
    current_y += current_delta_y;
    result.control_points.push_back(Spline::Point{
        static_cast<float>(current_x), static_cast<float>(current_y)});
  }

  for (int c = 0; c < 3; ++c) {
    for (int i = 0; i < 32; ++i) {
      result.color_dct[c][i] =
          color_dct_[c][i] /
          ColorQuantizationWeight(quantization_adjustment, c, i);
    }
  }
  for (int i = 0; i < 32; ++i) {
    result.color_dct[0][i] += ytox * result.color_dct[1][i];
    result.color_dct[2][i] += ytob * result.color_dct[1][i];
  }
  for (int i = 0; i < 32; ++i) {
    result.sigma_dct[i] =
        sigma_dct_[i] / ColorQuantizationWeight(quantization_adjustment, 3, i);
  }

  return result;
}

void QuantizedSpline::Tokenize(std::vector<Token>* const tokens) const {
  tokens->emplace_back(kNumControlPointsContext, control_points_.size());
  for (const auto& point : control_points_) {
    tokens->emplace_back(kControlPointsContext, PackSigned(point.first));
    tokens->emplace_back(kControlPointsContext, PackSigned(point.second));
  }
  const auto encode_dct = [tokens](const int dct[32]) {
    for (int i = 0; i < 32; ++i) {
      tokens->emplace_back(kDCTContext, PackSigned(dct[i]));
    }
  };
  for (int c = 0; c < 3; ++c) {
    encode_dct(color_dct_[c]);
  }
  encode_dct(sigma_dct_);
}

Status QuantizedSpline::Decode(const std::vector<uint8_t>& context_map,
                               ANSSymbolReader* const decoder,
                               BitReader* const br) {
  const size_t num_control_points =
      decoder->ReadHybridUint(kNumControlPointsContext, br, context_map);
  control_points_.resize(num_control_points);
  for (std::pair<int64_t, int64_t>& control_point : control_points_) {
    control_point.first = UnpackSigned(
        decoder->ReadHybridUint(kControlPointsContext, br, context_map));
    control_point.second = UnpackSigned(
        decoder->ReadHybridUint(kControlPointsContext, br, context_map));
  }

  const auto decode_dct = [decoder, br, &context_map](int dct[32]) -> Status {
    for (int i = 0; i < 32; ++i) {
      dct[i] =
          UnpackSigned(decoder->ReadHybridUint(kDCTContext, br, context_map));
    }
    return true;
  };
  for (int c = 0; c < 3; ++c) {
    JXL_RETURN_IF_ERROR(decode_dct(color_dct_[c]));
  }
  JXL_RETURN_IF_ERROR(decode_dct(sigma_dct_));
  return true;
}

void Splines::Encode(BitWriter* writer, const size_t layer,
                     AuxOut* aux_out) const {
  JXL_ASSERT(HasAny());

  std::vector<QuantizedSpline> splines = splines_;
  std::vector<std::vector<Token>> tokens(1);
  tokens[0].emplace_back(kNumSplinesContext, splines.size() - 1);
  EncodeAllStartingPoints(starting_points_, &tokens[0]);

  tokens[0].emplace_back(kQuantizationAdjustmentContext,
                         PackSigned(quantization_adjustment_));

  for (const QuantizedSpline& spline : splines) {
    spline.Tokenize(&tokens[0]);
  }

  EntropyEncodingData codes;
  std::vector<uint8_t> context_map;
  BuildAndEncodeHistograms(HistogramParams(), kNumSplineContexts, tokens,
                           &codes, &context_map, writer, layer, aux_out);
  WriteTokens(tokens[0], codes, context_map, writer, layer, aux_out);
}

Status Splines::Decode(jxl::BitReader* br) {
  std::vector<uint8_t> context_map;
  ANSCode code;
  JXL_RETURN_IF_ERROR(
      DecodeHistograms(br, kNumSplineContexts, &code, &context_map));
  ANSSymbolReader decoder(&code, br);
  const int num_splines =
      1 + decoder.ReadHybridUint(kNumSplinesContext, br, context_map);
  JXL_RETURN_IF_ERROR(DecodeAllStartingPoints(&starting_points_, br, &decoder,
                                              context_map, num_splines));

  quantization_adjustment_ = UnpackSigned(
      decoder.ReadHybridUint(kQuantizationAdjustmentContext, br, context_map));

  splines_.reserve(num_splines);
  for (int i = 0; i < num_splines; ++i) {
    QuantizedSpline spline;
    JXL_RETURN_IF_ERROR(spline.Decode(context_map, &decoder, br));
    splines_.push_back(std::move(spline));
  }

  JXL_RETURN_IF_ERROR(decoder.CheckANSFinalState());

  if (!HasAny()) {
    return JXL_FAILURE("Decoded splines but got none");
  }

  return true;
}

Status Splines::AddTo(Image3F* const opsin, const Rect& opsin_rect,
                      const Rect& image_rect,
                      const ColorCorrelationMap& cmap) const {
  return Apply</*add=*/true>(opsin, opsin_rect, image_rect, cmap);
}

Status Splines::SubtractFrom(Image3F* const opsin,
                             const ColorCorrelationMap& cmap) const {
  return Apply</*add=*/false>(opsin, Rect(*opsin), Rect(*opsin), cmap);
}

template <bool add>
Status Splines::Apply(Image3F* const opsin, const Rect& opsin_rect,
                      const Rect& image_rect,
                      const ColorCorrelationMap& cmap) const {
  for (size_t i = 0; i < splines_.size(); ++i) {
    const Spline spline =
        splines_[i].Dequantize(starting_points_[i], quantization_adjustment_,
                               cmap.YtoXRatio(0), cmap.YtoBRatio(0));
    if (std::adjacent_find(spline.control_points.begin(),
                           spline.control_points.end()) !=
        spline.control_points.end()) {
      return JXL_FAILURE("identical successive control points in spline %zu",
                         i);
    }
    std::vector<std::pair<Spline::Point, float>> points_to_draw;
    ForEachEquallySpacedPoint(
        DrawCentripetalCatmullRomSpline(spline.control_points),
        [&](const Spline::Point& point, const float multiplier) {
          points_to_draw.emplace_back(point, multiplier);
        });
    const float arc_length =
        (points_to_draw.size() - 2) * kDesiredRenderingDistance +
        points_to_draw.back().second;
    if (arc_length <= 0.f) {
      // This spline wouldn't have any effect.
      continue;
    }
    int k = 0;
    for (const auto& point_to_draw : points_to_draw) {
      const Spline::Point& point = point_to_draw.first;
      const float multiplier =
          add ? point_to_draw.second : -point_to_draw.second;
      const float progress_along_arc =
          std::min(1.f, (k * kDesiredRenderingDistance) / arc_length);
      ++k;
      float color[3];
      for (size_t c = 0; c < 3; ++c) {
        color[c] = ContinuousIDCT<32>(spline.color_dct[c],
                                      (32 - 1) * progress_along_arc);
      }
      const float sigma =
          ContinuousIDCT<32>(spline.sigma_dct, (32 - 1) * progress_along_arc);
      DrawGaussian(opsin, opsin_rect, image_rect, point, multiplier, color,
                   sigma);
    }
  }
  return true;
}

Splines FindSplines(const Image3F& opsin) {
  // TODO: implement spline detection.
  return {};
}

}  // namespace jxl
