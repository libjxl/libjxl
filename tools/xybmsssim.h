// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_XYBMSSSIM_H_
#define TOOLS_XYBMSSSIM_H_

#include <vector>

#include "lib/jxl/image.h"

namespace xybmsssim {

struct MsssimScale {
  double avg_ssim[3 * 4];
};

struct Msssim {
  std::vector<MsssimScale> scales;

  double Score() const;
};

// expects input images in XYB space
Msssim ComputeMSSSIM(jxl::Image3F& orig, jxl::Image3F& distorted);

}  // namespace xybmsssim

#endif  // TOOLS_XYBMSSSIM_H_
