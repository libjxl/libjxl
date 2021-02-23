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

#include "lib/extras/tone_mapping.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/extras/tone_mapping.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/transfer_functions-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

Status ToneMapFrame(const std::pair<float, float> display_nits,
                    ImageBundle* const ib, ThreadPool* const pool) {
  // Perform tone mapping as described in Report ITU-R BT.2390-8, section 5.4.

  ColorEncoding linear_rec2020;
  linear_rec2020.SetColorSpace(ColorSpace::kRGB);
  linear_rec2020.primaries = Primaries::k2100;
  linear_rec2020.white_point = WhitePoint::kD65;
  linear_rec2020.tf.SetTransferFunction(TransferFunction::kLinear);
  JXL_RETURN_IF_ERROR(linear_rec2020.CreateICC());
  JXL_RETURN_IF_ERROR(ib->TransformTo(linear_rec2020, pool));

  const auto eotf_inv = [](const float luminance) -> float {
    return TF_PQ().EncodedFromDisplay(luminance * (1. / 10000));
  };

  const float pq_mastering_min =
      eotf_inv(ib->metadata()->tone_mapping.min_nits);
  const float pq_mastering_max =
      eotf_inv(ib->metadata()->tone_mapping.intensity_target);
  const float pq_mastering_range = pq_mastering_max - pq_mastering_min;
  const float min_lum =
      (eotf_inv(display_nits.first) - pq_mastering_min) / pq_mastering_range;
  const float max_lum =
      (eotf_inv(display_nits.second) - pq_mastering_min) / pq_mastering_range;
  const float ks = 1.5f * max_lum - 0.5f;
  const float b = min_lum;

  const auto T = [ks](const float a) { return (a - ks) / (1 - ks); };
  const auto P = [&T, ks, max_lum](const float b) {
    const float t_b = T(b);
    const float t_b_2 = t_b * t_b;
    const float t_b_3 = t_b_2 * t_b;
    return (2 * t_b_3 - 3 * t_b_2 + 1) * ks +
           (t_b_3 - 2 * t_b_2 + t_b) * (1 - ks) +
           (-2 * t_b_3 + 3 * t_b_2) * max_lum;
  };

  JXL_RETURN_IF_ERROR(RunOnPool(
      pool, 0, ib->ysize(), ThreadPool::SkipInit(),
      [&](const int y, const int thread) {
        float* const JXL_RESTRICT row_r = ib->color()->PlaneRow(0, y);
        float* const JXL_RESTRICT row_g = ib->color()->PlaneRow(1, y);
        float* const JXL_RESTRICT row_b = ib->color()->PlaneRow(2, y);
        for (size_t x = 0; x < ib->xsize(); ++x) {
          const float luminance =
              ib->metadata()->IntensityTarget() *
              (0.2627f * row_r[x] + 0.6780f * row_g[x] + 0.0593f * row_b[x]);
          const float normalized_pq =
              (eotf_inv(luminance) - pq_mastering_min) / pq_mastering_range;
          const float e2 =
              normalized_pq < ks ? normalized_pq : P(normalized_pq);
          const float one_minus_e2 = 1 - e2;
          const float one_minus_e2_2 = one_minus_e2 * one_minus_e2;
          const float one_minus_e2_4 = one_minus_e2_2 * one_minus_e2_2;
          const float e3 = e2 + b * one_minus_e2_4;
          const float e4 = e3 * pq_mastering_range + pq_mastering_min;
          const float new_luminance = 10000 * TF_PQ().DisplayFromEncoded(e4);

          if (luminance > 1e-6) {
            const float ratio = new_luminance / luminance;
            const float multiplier =
                ib->metadata()->IntensityTarget() * ratio / display_nits.second;
            for (float* const val : {&row_r[x], &row_g[x], &row_b[x]}) {
              *val *= multiplier;
            }
          } else {
            const float new_val = new_luminance / display_nits.second;
            for (float* const val : {&row_r[x], &row_g[x], &row_b[x]}) {
              *val = new_val;
            }
          }
        }
      },
      "ToneMap"));

  return true;
}

// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {

namespace {
HWY_EXPORT(ToneMapFrame);
}

Status ToneMapTo(const std::pair<float, float> display_nits,
                 CodecInOut* const io, ThreadPool* const pool) {
  const auto tone_map_frame = HWY_DYNAMIC_DISPATCH(ToneMapFrame);
  for (ImageBundle& ib : io->frames) {
    JXL_RETURN_IF_ERROR(tone_map_frame(display_nits, &ib, pool));
  }
  io->metadata.m.SetIntensityTarget(display_nits.second);
  return true;
}

}  // namespace jxl
#endif
