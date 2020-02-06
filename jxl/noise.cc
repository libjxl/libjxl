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

#include "jxl/noise.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <hwy/compiler_specific.h>
#include <hwy/static_targets.h>
#include <numeric>
#include <utility>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/robust_statistics.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/convolve.h"
#include "jxl/image_ops.h"
#include "jxl/opsin_params.h"
#include "jxl/optimize.h"

namespace jxl {
namespace {

namespace HWY_NAMESPACE {
#include "jxl/xorshift128plus-inl.h"
}  // namespace HWY_NAMESPACE
using HWY_NAMESPACE::Xorshift128Plus;

using D = HWY_CAPPED(float, 1);

// Converts one vector's worth of random bits to floats in [0, 1).
HWY_ATTR void BitsToFloat(const uint32_t* JXL_RESTRICT random_bits,
                          float* JXL_RESTRICT floats) {
  const HWY_FULL(float) df;
  const HWY_FULL(uint32_t) du;

  const auto bits = Load(du, random_bits);
  // 1.0 + 23 random mantissa bits = [1, 2)
  const auto rand12 =
      BitCast(df, hwy::ShiftRight<9>(bits) | Set(du, 0x3F800000));
  const auto rand01 = rand12 - Set(df, 1.0f);
  Store(rand01, df, floats);
}

HWY_ATTR void RandomImage(ImageF* JXL_RESTRICT temp, Xorshift128Plus* rng,
                          const Rect& rect, const ImageF* JXL_RESTRICT noise) {
  const size_t xsize = temp->xsize();
  const size_t ysize = temp->ysize();

  // May exceed the vector size, hence we have two loops over x below.
  constexpr size_t kFloatsPerBatch =
      Xorshift128Plus::N * sizeof(uint64_t) / sizeof(float);
  HWY_ALIGN uint64_t batch[Xorshift128Plus::N];

  const HWY_FULL(float) df;

  for (size_t y = 0; y < ysize; ++y) {
    float* JXL_RESTRICT row = temp->Row(y);

    size_t x = 0;
    // Only entire batches (avoids exceeding the image padding).
    for (; x + kFloatsPerBatch <= xsize; x += kFloatsPerBatch) {
      rng->Fill(batch);
      for (size_t i = 0; i < kFloatsPerBatch; i += df.N) {
        BitsToFloat(reinterpret_cast<const uint32_t*>(batch) + i, row + x + i);
      }
    }

    // Any remaining pixels, rounded up to vectors (safe due to padding).
    rng->Fill(batch);
    size_t batch_pos = 0;  // < kFloatsPerBatch
    for (; x < xsize; x += df.N) {
      BitsToFloat(reinterpret_cast<const uint32_t*>(batch) + batch_pos,
                  row + x);
      batch_pos += df.N;
    }
  }

  // TODO(veluca): SIMD-fy & avoid intermediate copy.
  ImageF out(xsize, ysize);
  slow::Laplacian5<2, WrapMirror>::Run(*temp, Rect(*temp),
                                       /*unused*/ int(), &out);
  CopyImageTo(out, rect, noise);
}

float GetScoreSumsOfAbsoluteDifferences(const Image3F& opsin, const int x,
                                        const int y, const int block_size) {
  const int small_bl_size_x = 3;
  const int small_bl_size_y = 4;
  const int kNumSAD =
      (block_size - small_bl_size_x) * (block_size - small_bl_size_y);
  // block_size x block_size reference pixels
  int counter = 0;
  const int offset = 2;

  std::vector<float> sad(kNumSAD, 0);
  for (int y_bl = 0; y_bl + small_bl_size_y < block_size; ++y_bl) {
    for (int x_bl = 0; x_bl + small_bl_size_x < block_size; ++x_bl) {
      float sad_sum = 0;
      // size of the center patch, we compare all the patches inside window with
      // the center one
      for (int cy = 0; cy < small_bl_size_y; ++cy) {
        for (int cx = 0; cx < small_bl_size_x; ++cx) {
          float wnd = 0.5f * (opsin.PlaneRow(1, y + y_bl + cy)[x + x_bl + cx] +
                              opsin.PlaneRow(0, y + y_bl + cy)[x + x_bl + cx]);
          float center =
              0.5f * (opsin.PlaneRow(1, y + offset + cy)[x + offset + cx] +
                      opsin.PlaneRow(0, y + offset + cy)[x + offset + cx]);
          sad_sum += std::abs(center - wnd);
        }
      }
      sad[counter++] = sad_sum;
    }
  }
  const int kSamples = (kNumSAD) / 2;
  // As with ROAD (rank order absolute distance), we keep the smallest half of
  // the values in SAD (we use here the more robust patch SAD instead of
  // absolute single-pixel differences).
  std::sort(sad.begin(), sad.end());
  const float total_sad_sum =
      std::accumulate(sad.begin(), sad.begin() + kSamples, 0.0f);
  return total_sad_sum / kSamples;
}

class NoiseHistogram {
 public:
  static constexpr int kBins = 256;

  NoiseHistogram() { std::fill(bins, bins + kBins, 0); }

  void Increment(const float x) { bins[Index(x)] += 1; }
  int Get(const float x) const { return bins[Index(x)]; }
  int Bin(const size_t bin) const { return bins[bin]; }

  void Print() const {
    for (unsigned int bin : bins) {
      printf("%d\n", bin);
    }
  }

  int Mode() const {
    uint32_t cdf[kBins];
    std::partial_sum(bins, bins + kBins, cdf);
    return HalfRangeMode()(cdf, kBins);
  }

  double Quantile(double q01) const {
    const int64_t total = std::accumulate(bins, bins + kBins, 1LL);
    const int64_t target = static_cast<int64_t>(q01 * total);
    // Until sum >= target:
    int64_t sum = 0;
    size_t i = 0;
    for (; i < kBins; ++i) {
      sum += bins[i];
      // Exact match: assume middle of bin i
      if (sum == target) {
        return i + 0.5;
      }
      if (sum > target) break;
    }

    // Next non-empty bin (in case histogram is sparsely filled)
    size_t next = i + 1;
    while (next < kBins && bins[next] == 0) {
      ++next;
    }

    // Linear interpolation according to how far into next we went
    const double excess = target - sum;
    const double weight_next = bins[Index(next)] / excess;
    return ClampX(next * weight_next + i * (1.0 - weight_next));
  }

  // Inter-quartile range
  double IQR() const { return Quantile(0.75) - Quantile(0.25); }

 private:
  template <typename T>
  T ClampX(const T x) const {
    return std::min(std::max(T(0), x), T(kBins - 1));
  }
  size_t Index(const float x) const { return ClampX(static_cast<int>(x)); }

  uint32_t bins[kBins];
};

std::vector<float> GetSADScoresForPatches(const Image3F& opsin,
                                          const size_t block_s,
                                          const size_t num_bin,
                                          NoiseHistogram* sad_histogram) {
  std::vector<float> sad_scores(
      (opsin.ysize() / block_s) * (opsin.xsize() / block_s), 0.0f);

  int block_index = 0;

  for (size_t y = 0; y + block_s <= opsin.ysize(); y += block_s) {
    for (size_t x = 0; x + block_s <= opsin.xsize(); x += block_s) {
      float sad_sc = GetScoreSumsOfAbsoluteDifferences(opsin, x, y, block_s);
      sad_scores[block_index++] = sad_sc;
      sad_histogram->Increment(sad_sc * num_bin);
    }
  }
  return sad_scores;
}

float GetSADThreshold(const NoiseHistogram& histogram, const int num_bin) {
  // Here we assume that the most patches with similar SAD value is a "flat"
  // patches. However, some images might contain regular texture part and
  // generate second strong peak at the histogram
  // TODO(user) handle bimodal and heavy-tailed case
  const int mode = histogram.Mode();
  return static_cast<float>(mode) / NoiseHistogram::kBins;
}

// [0, max_value]
template <class D, class V>
static HWY_ATTR HWY_INLINE V Clamp0ToMax(D d, const V x, const V max_value) {
  const auto clamped = Min(x, max_value);
  return ZeroIfNegative(clamped);
}

// x is in [0+delta, 1+delta], delta ~= 0.06
template <class StrengthEval>
HWY_ATTR typename StrengthEval::V NoiseStrength(
    const StrengthEval& eval, const typename StrengthEval::V x) {
  return Clamp0ToMax(D(), eval(x), Set(D(), 1.0f));
}

std::pair<int, float> IndexAndFrac(float x) {
  // TODO: instead of 1, this should be a proper Y range.
  constexpr float kScale = (NoiseParams::kNumNoisePoints - 2) / 1;
  float scaled_x = std::max(0.f, x * kScale);
  size_t floor_x = static_cast<size_t>(scaled_x);
  if (JXL_UNLIKELY(floor_x > NoiseParams::kNumNoisePoints - 2)) {
    floor_x = NoiseParams::kNumNoisePoints - 2;
  }
  return std::make_pair(floor_x, scaled_x - floor_x);
}

// TODO(veluca): SIMD-fy.
class StrengthEvalLut {
 public:
  using V = hwy::VT<D>;

  explicit StrengthEvalLut(const NoiseParams& noise_params)
      : noise_params_(noise_params) {}

  HWY_ATTR V operator()(const V vx) const {
    float x;
    Store(vx, D(), &x);
    std::pair<int, float> pos = IndexAndFrac(x);
    JXL_DASSERT(pos.first >= 0 && static_cast<size_t>(pos.first) <
                                      NoiseParams::kNumNoisePoints - 1);
    float low = noise_params_.lut[pos.first];
    float hi = noise_params_.lut[pos.first + 1];
    return Set(D(), low * (1.0f - pos.second) + hi * pos.second);
  }

 private:
  const NoiseParams noise_params_;
};

template <class D>
HWY_ATTR void AddNoiseToRGB(const D d, const hwy::VT<D> rnd_noise_r,
                            const hwy::VT<D> rnd_noise_g,
                            const hwy::VT<D> rnd_noise_cor,
                            const hwy::VT<D> noise_strength_g,
                            const hwy::VT<D> noise_strength_r, float ytox,
                            float ytob, float* JXL_RESTRICT out_x,
                            float* JXL_RESTRICT out_y,
                            float* JXL_RESTRICT out_b) {
  const auto kRGCorr = Set(d, 0.9921875);    // 127/128
  const auto kRGNCorr = Set(d, 0.0078125f);  // 1/128

  const auto red_noise = kRGNCorr * rnd_noise_r * noise_strength_r +
                         kRGCorr * rnd_noise_cor * noise_strength_r;
  const auto green_noise = kRGNCorr * rnd_noise_g * noise_strength_g +
                           kRGCorr * rnd_noise_cor * noise_strength_g;

  auto vx = Load(d, out_x);
  auto vy = Load(d, out_y);
  auto vb = Load(d, out_b);

  vx += red_noise - green_noise + Set(d, ytox) * (red_noise + green_noise);
  vy += red_noise + green_noise;
  vb += Set(d, ytob) * (red_noise + green_noise);

  Store(vx, d, out_x);
  Store(vy, d, out_y);
  Store(vb, d, out_b);
}

}  // namespace

HWY_ATTR void AddNoise(const NoiseParams& noise_params, const Rect& noise_rect,
                       const Image3F& noise, const Rect& opsin_rect,
                       const ColorCorrelationMap& cmap, Image3F* opsin) {
  if (!noise_params.HasAny()) return;
  const StrengthEvalLut noise_model(noise_params);
  D d;
  const auto half = Set(d, 0.5f);

  const size_t xsize = opsin_rect.xsize();
  const size_t ysize = opsin_rect.ysize();

  // With the prior subtract-random Laplacian approximation, rnd_* ranges were
  // about [-1.5, 1.6]; Laplacian3 about doubles this to [-3.6, 3.6], so the
  // normalizer is half of what it was before (0.5).
  const auto norm_const = Set(d, 0.22f);

  float ytox = cmap.YtoXRatio(kColorOffset);
  float ytob = cmap.YtoBRatio(kColorOffset);

  for (size_t y = 0; y < ysize; ++y) {
    float* JXL_RESTRICT row_x = opsin_rect.PlaneRow(opsin, 0, y);
    float* JXL_RESTRICT row_y = opsin_rect.PlaneRow(opsin, 1, y);
    float* JXL_RESTRICT row_b = opsin_rect.PlaneRow(opsin, 2, y);
    const float* JXL_RESTRICT row_rnd_r = noise_rect.PlaneRow(opsin, 0, y);
    const float* JXL_RESTRICT row_rnd_g = noise_rect.PlaneRow(opsin, 1, y);
    const float* JXL_RESTRICT row_rnd_c = noise_rect.PlaneRow(opsin, 2, y);
    for (size_t x = 0; x < xsize; x += d.N) {
      const auto vx = Load(d, row_x + x);
      const auto vy = Load(d, row_y + x);
      const auto in_g = vy - vx;
      const auto in_r = vy + vx;
      const auto noise_strength_g = NoiseStrength(noise_model, in_g * half);
      const auto noise_strength_r = NoiseStrength(noise_model, in_r * half);
      const auto addit_rnd_noise_red = Load(d, row_rnd_r + x) * norm_const;
      const auto addit_rnd_noise_green = Load(d, row_rnd_g + x) * norm_const;
      const auto addit_rnd_noise_correlated =
          Load(d, row_rnd_c + x) * norm_const;
      AddNoiseToRGB(D(), addit_rnd_noise_red, addit_rnd_noise_green,
                    addit_rnd_noise_correlated, noise_strength_g,
                    noise_strength_r, ytox, ytob, row_x + x, row_y + x,
                    row_b + x);
    }
  }
}

void RandomImage3(const Rect& rect, Image3F* JXL_RESTRICT noise) {
  const size_t xsize = rect.xsize();
  const size_t ysize = rect.ysize();

  Xorshift128Plus rng((uint64_t(rect.y0()) << 32) + rect.x0());
  ImageF temp(xsize, ysize);
  RandomImage(&temp, &rng, rect, &noise->Plane(0));
  RandomImage(&temp, &rng, rect, &noise->Plane(1));
  RandomImage(&temp, &rng, rect, &noise->Plane(2));
}

// loss = sum asym * (F(x) - nl)^2 + kReg * num_points * sum (w[i] - w[i+1])^2
// where asym = 1 if F(x) < nl, kAsym if F(x) > nl.
struct LossFunction {
  explicit LossFunction(std::vector<NoiseLevel> nl0) : nl(std::move(nl0)) {}

  double Compute(const std::vector<double>& w, std::vector<double>* df,
                 bool skip_regularization = false) const {
    constexpr double kReg = 0.005;
    constexpr double kAsym = 1.1;
    double loss_function = 0;
    for (size_t i = 0; i < w.size(); i++) {
      (*df)[i] = 0;
    }
    for (auto ind : nl) {
      std::pair<int, float> pos = IndexAndFrac(ind.intensity);
      JXL_DASSERT(pos.first >= 0 && static_cast<size_t>(pos.first) <
                                        NoiseParams::kNumNoisePoints - 1);
      double low = w[pos.first];
      double hi = w[pos.first + 1];
      double val = low * (1.0f - pos.second) + hi * pos.second;
      double dist = val - ind.noise_level;
      if (dist > 0) {
        loss_function += kAsym * dist * dist;
        (*df)[pos.first] -= kAsym * (1.0f - pos.second) * dist;
        (*df)[pos.first + 1] -= kAsym * pos.second * dist;
      } else {
        loss_function += dist * dist;
        (*df)[pos.first] -= (1.0f - pos.second) * dist;
        (*df)[pos.first + 1] -= pos.second * dist;
      }
    }
    if (skip_regularization) return loss_function;
    for (size_t i = 0; i + 1 < w.size(); i++) {
      double diff = w[i] - w[i + 1];
      loss_function += kReg * nl.size() * diff * diff;
      (*df)[i] -= kReg * diff * nl.size();
      (*df)[i + 1] += kReg * diff * nl.size();
    }
    return loss_function;
  }

  std::vector<NoiseLevel> nl;
};

Status GetNoiseParameter(const Image3F& opsin, NoiseParams* noise_params,
                         float quality_coef) {
  // The size of a patch in decoder might be different from encoder's patch
  // size.
  // For encoder: the patch size should be big enough to estimate
  //              noise level, but, at the same time, it should be not too big
  //              to be able to estimate intensity value of the patch
  const size_t block_s = 8;
  const size_t kNumBin = 256;
  NoiseHistogram sad_histogram;
  std::vector<float> sad_scores =
      GetSADScoresForPatches(opsin, block_s, kNumBin, &sad_histogram);
  float sad_threshold = GetSADThreshold(sad_histogram, kNumBin);
  // If threshold is too large, the image has a strong pattern. This pattern
  // fools our model and it will add too much noise. Therefore, we do not add
  // noise for such images
  if (sad_threshold > 0.15f || sad_threshold <= 0.0f) {
    noise_params->Clear();
    return false;
  }
  std::vector<NoiseLevel> nl =
      GetNoiseLevel(opsin, sad_scores, sad_threshold, block_s);

  OptimizeNoiseParameters(nl, noise_params);
  for (float& i : noise_params->lut) {
    i *= quality_coef;
  }
  return noise_params->HasAny();
}

const float kNoisePrecision = 1 << 10;

void EncodeFloatParam(float val, float precision, BitWriter* writer) {
  JXL_ASSERT(val >= 0);
  const int absval_quant = static_cast<int>(val * precision + 0.5f);
  JXL_ASSERT(absval_quant < (1 << 10));
  writer->Write(10, absval_quant);
}

HWY_ATTR void DecodeFloatParam(float precision, float* val, BitReader* br) {
  const int absval_quant = br->ReadFixedBits<10>();
  *val = absval_quant / precision;
}

void EncodeNoise(const NoiseParams& noise_params, BitWriter* writer,
                 size_t layer, AuxOut* aux_out) {
  JXL_ASSERT(noise_params.HasAny());

  BitWriter::Allotment allotment(writer, NoiseParams::kNumNoisePoints * 16);
  for (float i : noise_params.lut) {
    EncodeFloatParam(i, kNoisePrecision, writer);
  }
  ReclaimAndCharge(writer, &allotment, layer, aux_out);
}

Status DecodeNoise(BitReader* br, NoiseParams* noise_params) {
  for (float& i : noise_params->lut) {
    DecodeFloatParam(kNoisePrecision, &i, br);
  }
  if (!noise_params->HasAny()) {
    return JXL_FAILURE("DecodedNoise got no noise");
  }
  return true;
}

void OptimizeNoiseParameters(const std::vector<NoiseLevel>& noise_level,
                             NoiseParams* noise_params) {
  constexpr double kMaxError = 1e-3;
  static const double kPrecision = 1e-8;
  static const int kMaxIter = 40;

  float avg = 0;
  for (const NoiseLevel& nl : noise_level) {
    avg += nl.noise_level;
  }
  avg /= noise_level.size();

  LossFunction loss_function(noise_level);
  std::vector<double> parameter_vector(NoiseParams::kNumNoisePoints, avg);

  parameter_vector = optimize::OptimizeWithScaledConjugateGradientMethod(
      loss_function, parameter_vector, kPrecision, kMaxIter);

  std::vector<double> df = parameter_vector;
  float loss = loss_function.Compute(parameter_vector, &df,
                                     /*skip_regularization=*/true) /
               noise_level.size();

  // Approximation went too badly: escape with no noise at all.
  if (loss > kMaxError) {
    noise_params->Clear();
    return;
  }

  for (size_t i = 0; i < parameter_vector.size(); i++) {
    noise_params->lut[i] = std::max(parameter_vector[i], 0.0);
  }
}

std::vector<float> GetTextureStrength(const Image3F& opsin,
                                      const size_t block_s) {
  std::vector<float> texture_strength_index((opsin.ysize() / block_s) *
                                            (opsin.xsize() / block_s));
  size_t block_index = 0;

  for (size_t y = 0; y + block_s <= opsin.ysize(); y += block_s) {
    for (size_t x = 0; x + block_s <= opsin.xsize(); x += block_s) {
      float texture_strength = 0;
      for (size_t y_bl = 0; y_bl < block_s; ++y_bl) {
        for (size_t x_bl = 0; x_bl + 1 < block_s; ++x_bl) {
          float diff = opsin.PlaneRow(1, y)[x + x_bl + 1] -
                       opsin.PlaneRow(1, y)[x + x_bl];
          texture_strength += diff * diff;
        }
      }
      for (size_t y_bl = 0; y_bl + 1 < block_s; ++y_bl) {
        for (size_t x_bl = 0; x_bl < block_s; ++x_bl) {
          float diff = opsin.PlaneRow(1, y + 1)[x + x_bl] -
                       opsin.PlaneRow(1, y)[x + x_bl];
          texture_strength += diff * diff;
        }
      }
      texture_strength_index[block_index] = texture_strength;
      ++block_index;
    }
  }
  return texture_strength_index;
}

float GetThresholdFlatIndices(const std::vector<float>& texture_strength,
                              const int n_patches) {
  std::vector<float> kth_statistic = texture_strength;
  std::stable_sort(kth_statistic.begin(), kth_statistic.end());
  return kth_statistic[n_patches];
}

std::vector<NoiseLevel> GetNoiseLevel(
    const Image3F& opsin, const std::vector<float>& texture_strength,
    const float threshold, const size_t block_s) {
  std::vector<NoiseLevel> noise_level_per_intensity;

  const int filt_size = 1;
  static const float kLaplFilter[filt_size * 2 + 1][filt_size * 2 + 1] = {
      {-0.25f, -1.0f, -0.25f},
      {-1.0f, 5.0f, -1.0f},
      {-0.25f, -1.0f, -0.25f},
  };

  // The noise model is built based on channel 0.5 * (X+Y) as we notice that it
  // is similar to the model 0.5 * (Y-X)
  size_t patch_index = 0;

  for (size_t y = 0; y + block_s <= opsin.ysize(); y += block_s) {
    for (size_t x = 0; x + block_s <= opsin.xsize(); x += block_s) {
      if (texture_strength[patch_index] <= threshold) {
        // Calculate mean value
        float mean_int = 0;
        for (size_t y_bl = 0; y_bl < block_s; ++y_bl) {
          for (size_t x_bl = 0; x_bl < block_s; ++x_bl) {
            mean_int += 0.5f * (opsin.PlaneRow(1, y + y_bl)[x + x_bl] +
                                opsin.PlaneRow(0, y + y_bl)[x + x_bl]);
          }
        }
        mean_int /= block_s * block_s;

        // Calculate Noise level
        float noise_level = 0;
        size_t count = 0;
        for (size_t y_bl = 0; y_bl < block_s; ++y_bl) {
          for (size_t x_bl = 0; x_bl < block_s; ++x_bl) {
            float filtered_value = 0;
            for (int y_f = -1 * filt_size; y_f <= filt_size; ++y_f) {
              if (((y_bl + y_f) < block_s) && ((y_bl + y_f) >= 0)) {
                for (int x_f = -1 * filt_size; x_f <= filt_size; ++x_f) {
                  if ((x_bl + x_f) >= 0 && (x_bl + x_f) < block_s) {
                    filtered_value +=
                        0.5f *
                        (opsin.PlaneRow(1, y + y_bl + y_f)[x + x_bl + x_f] +
                         opsin.PlaneRow(0, y + y_bl + y_f)[x + x_bl + x_f]) *
                        kLaplFilter[y_f + filt_size][x_f + filt_size];
                  } else {
                    filtered_value +=
                        0.5f *
                        (opsin.PlaneRow(1, y + y_bl + y_f)[x + x_bl - x_f] +
                         opsin.PlaneRow(0, y + y_bl + y_f)[x + x_bl - x_f]) *
                        kLaplFilter[y_f + filt_size][x_f + filt_size];
                  }
                }
              } else {
                for (int x_f = -1 * filt_size; x_f <= filt_size; ++x_f) {
                  if ((x_bl + x_f) >= 0 && (x_bl + x_f) < block_s) {
                    filtered_value +=
                        0.5f *
                        (opsin.PlaneRow(1, y + y_bl - y_f)[x + x_bl + x_f] +
                         opsin.PlaneRow(0, y + y_bl - y_f)[x + x_bl + x_f]) *
                        kLaplFilter[y_f + filt_size][x_f + filt_size];
                  } else {
                    filtered_value +=
                        0.5f *
                        (opsin.PlaneRow(1, y + y_bl - y_f)[x + x_bl - x_f] +
                         opsin.PlaneRow(0, y + y_bl - y_f)[x + x_bl - x_f]) *
                        kLaplFilter[y_f + filt_size][x_f + filt_size];
                  }
                }
              }
            }
            noise_level += std::abs(filtered_value);
            ++count;
          }
        }
        noise_level /= count;
        NoiseLevel nl;
        nl.intensity = mean_int;
        nl.noise_level = noise_level;
        noise_level_per_intensity.push_back(nl);
      }
      ++patch_index;
    }
  }
  return noise_level_per_intensity;
}

}  // namespace jxl
