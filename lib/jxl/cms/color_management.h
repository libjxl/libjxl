// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_COLOR_MANAGEMENT_H_
#define LIB_JXL_COLOR_MANAGEMENT_H_

// ICC profiles and color space conversions.
// TODO(eustas): rename to something_internal.h; those methods are not allowed
//               to be used outside of cms/ folder.

#include <jxl/color_encoding.h>

#include <cstdint>
#include <string>
#include <vector>

#include "lib/jxl/base/status.h"

namespace jxl {

enum class ExtraTF {
  kNone,
  kPQ,
  kHLG,
  kSRGB,
};

// NOTE: for XYB colorspace, the created profile can be used to transform a
// *scaled* XYB image (created by ScaleXYB()) to another colorspace.
Status MaybeCreateProfile(const JxlColorEncoding& c, std::vector<uint8_t>* icc);
Status CIEXYZFromWhiteCIExy(double wx, double wy, float XYZ[3]);

Status PrimariesToXYZ(float rx, float ry, float gx, float gy, float bx,
                      float by, float wx, float wy, float matrix[9]);
// Adapts whitepoint x, y to D50
Status AdaptToXYZD50(float wx, float wy, float matrix[9]);
Status PrimariesToXYZD50(float rx, float ry, float gx, float gy, float bx,
                         float by, float wx, float wy, float matrix[9]);
std::string ColorEncodingDescription(const JxlColorEncoding& c);

}  // namespace jxl

#endif  // LIB_JXL_COLOR_MANAGEMENT_H_
