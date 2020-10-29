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

#ifndef TOOLS_SSIMULACRA_H_
#define TOOLS_SSIMULACRA_H_

#include <vector>

#include "lib/jxl/image.h"

namespace ssimulacra {

struct SsimulacraScale {
  double avg_ssim[3];
  double min_ssim[3];
};

struct Ssimulacra {
  std::vector<SsimulacraScale> scales;
  double avg_edgediff[3];
  double row_p2[2][3];
  double col_p2[2][3];

  double Score() const;
  void PrintDetails() const;
};

Ssimulacra ComputeDiff(const jxl::Image3F& orig, const jxl::Image3F& distorted);

}  // namespace ssimulacra

#endif  // TOOLS_SSIMULACRA_H_
