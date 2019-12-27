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

#include "tools/speed_stats.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <algorithm>

#include "jxl/base/robust_statistics.h"

namespace jpegxl {
namespace tools {

void SpeedStats::NotifyElapsed(double elapsed_seconds) {
  JXL_ASSERT(elapsed_seconds > 0.0);
  elapsed_.push_back(elapsed_seconds);
}

jxl::Status SpeedStats::GetSummary(SpeedStats::Summary* s) {
  if (elapsed_.empty()) return JXL_FAILURE("Didn't call NotifyElapsed");

  s->min = *std::min_element(elapsed_.begin(), elapsed_.end());
  s->max = *std::max_element(elapsed_.begin(), elapsed_.end());

  // Single rep
  if (elapsed_.size() == 1) {
    s->central_tendency = elapsed_[0];
    s->variability = 0.0;
    s->type = "";
    return true;
  }

  // Two: skip first (noisier)
  if (elapsed_.size() == 2) {
    s->central_tendency = elapsed_[1];
    s->variability = 0.0;
    s->type = "second: ";
    return true;
  }

  // Prefer geomean unless numerically unreliable (too many reps)
  if (std::pow(elapsed_[0], elapsed_.size()) < 1E100) {
    double product = 1.0;
    for (size_t i = 1; i < elapsed_.size(); ++i) {
      product *= elapsed_[i];
    }

    s->central_tendency = std::pow(product, 1.0 / (elapsed_.size() - 1));
    s->variability = 0.0;
    s->type = "geomean: ";
    return true;
  }

  // Else: mode
  std::sort(elapsed_.begin(), elapsed_.end());
  s->central_tendency = jxl::HalfSampleMode()(elapsed_.data(), elapsed_.size());
  s->variability = jxl::MedianAbsoluteDeviation(elapsed_, s->central_tendency);
  s->type = "mode: ";
  return true;
}

jxl::Status SpeedStats::Print(const size_t xsize, const size_t ysize,
                              const size_t worker_threads) {
  Summary s;
  JXL_RETURN_IF_ERROR(GetSummary(&s));
  char variability[20] = {'\0'};
  if (s.variability != 0.0) {
    snprintf(variability, sizeof(variability), " (var %.2f)", s.variability);
  }

  const double megapixels = xsize * ysize * 1E-6;
  const double mpps = megapixels / s.central_tendency;
  // Note flipped order: higher elapsed = lower mpps.
  const double mpps_min = megapixels / s.max;
  const double mpps_max = megapixels / s.min;

  fprintf(stderr,
          "%zu x %zu, %s%.2f MP/s [%.2f, %.2f]%s, %zu reps, %zu threads.\n",
          xsize, ysize, s.type, mpps, mpps_min, mpps_max, variability,
          elapsed_.size(), worker_threads);
  return true;
}

}  // namespace tools
}  // namespace jpegxl
