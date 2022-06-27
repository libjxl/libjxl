// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/speed_stats.h"

#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>

#include <algorithm>
#include <string>

namespace jpegxl {
namespace tools {

void SpeedStats::NotifyElapsed(double elapsed_seconds) {
  if (elapsed_seconds > 0.0) {
    elapsed_.push_back(elapsed_seconds);
  }
}

bool SpeedStats::GetSummary(SpeedStats::Summary* s) {
  if (elapsed_.empty()) return false;

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
    s->type = " second:";
    return true;
  }

  // Prefer geomean unless numerically unreliable (too many reps)
  if (pow(elapsed_[0], elapsed_.size()) < 1E100) {
    double product = 1.0;
    for (size_t i = 1; i < elapsed_.size(); ++i) {
      product *= elapsed_[i];
    }

    s->central_tendency = pow(product, 1.0 / (elapsed_.size() - 1));
    s->variability = 0.0;
    s->type = " geomean:";
    return true;
  }

  // Else: median
  std::sort(elapsed_.begin(), elapsed_.end());
  s->central_tendency = elapsed_.data()[elapsed_.size() / 2];
  std::vector<double> deviations(elapsed_.size());
  for (size_t i = 0; i < elapsed_.size(); i++) {
    deviations[i] = fabs(elapsed_[i] - s->central_tendency);
  }
  std::nth_element(deviations.begin(),
                   deviations.begin() + deviations.size() / 2,
                   deviations.end());
  s->variability = deviations[deviations.size() / 2];
  s->type = "median: ";
  return true;
}

namespace {

std::string SummaryStat(double value, const char* unit,
                        const SpeedStats::Summary& s) {
  if (value == 0.) return "";

  char stat_str[100] = {'\0'};
  const double value_tendency = value / s.central_tendency;
  // Note flipped order: higher elapsed = lower mpps.
  const double value_min = value / s.max;
  const double value_max = value / s.min;

  snprintf(stat_str, sizeof(stat_str), ",%s %.2f %s/s [%.2f, %.2f]", s.type,
           value_tendency, unit, value_min, value_max);
  return stat_str;
}

}  // namespace

bool SpeedStats::Print(size_t worker_threads) {
  Summary s;
  if (!GetSummary(&s)) {
    return false;
  }
  std::string mps_stats = SummaryStat(xsize_ * ysize_ * 1e-6, "MP", s);
  std::string mbs_stats = SummaryStat(file_size_ * 1e-6, "MB", s);

  char variability[20] = {'\0'};
  if (s.variability != 0.0) {
    snprintf(variability, sizeof(variability), " (var %.2f)", s.variability);
  }

  fprintf(stderr,
          "%" PRIu64 " x %" PRIu64 "%s%s%s, %" PRIu64 " reps, %" PRIu64
          " threads.\n",
          static_cast<uint64_t>(xsize_), static_cast<uint64_t>(ysize_),
          mps_stats.c_str(), mbs_stats.c_str(), variability,
          static_cast<uint64_t>(elapsed_.size()),
          static_cast<uint64_t>(worker_threads));
  return true;
}

}  // namespace tools
}  // namespace jpegxl
