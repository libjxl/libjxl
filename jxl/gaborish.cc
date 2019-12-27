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

#include "jxl/gaborish.h"

#include <stddef.h>

#include <hwy/static_targets.h>

#include "jxl/base/status.h"
#include "jxl/convolve.h"
#include "jxl/image_ops.h"

namespace jxl {

class GaborishKernel {
 public:
  // weight1,2 need not be normalized.
  GaborishKernel(float weight1, float weight2) {
    constexpr float weight0 = 1.0f;

    // Normalize
    const float mul = 1.0 / (weight0 + 4 * (weight1 + weight2));
    const float w0 = weight0 * mul;
    const float w1 = weight1 * mul;
    const float w2 = weight2 * mul;

    // Fill all lanes
    for (size_t i = 0; i < 4; ++i) {
      // clang-format off
      weights_.tl[i] = w2; weights_.tc[i] = w1; weights_.tr[i] = w2;
      weights_.ml[i] = w1; weights_.mc[i] = w0; weights_.mr[i] = w1;
      weights_.bl[i] = w2; weights_.bc[i] = w1; weights_.br[i] = w2;
      // clang-format on
    }
  }

  JXL_INLINE const Weights3x3& Weights() const { return weights_; }

 private:
  Weights3x3 weights_;
};

Image3F GaborishInverse(const Image3F& in, double mul, ThreadPool* pool) {
  JXL_ASSERT(mul >= 0.0);

  // Only an approximation. One or even two 3x3, and rank-1 (separable) 5x5
  // are insufficient.
  static const double kGaborish[5] = {
      -0.092359145662814029,  -0.039253623634014627, 0.016176494530216929,
      0.00083458437774987476, 0.004512465323949319,
  };
  static const double kCenter = 1.0;
  /*
    better would be:
      1.0 - mul * (4 * (kGaborish[0] + kGaborish[1] +
                        kGaborish[2] + kGaborish[4]) +
                   8 * (kGaborish[3]));
  */

  const float sharpen_weights5[9] = {
      static_cast<float>(kCenter),
      static_cast<float>(mul * kGaborish[0]),
      static_cast<float>(mul * kGaborish[2]),

      static_cast<float>(mul * kGaborish[0]),
      static_cast<float>(mul * kGaborish[1]),
      static_cast<float>(mul * kGaborish[3]),

      static_cast<float>(mul * kGaborish[2]),
      static_cast<float>(mul * kGaborish[3]),
      static_cast<float>(mul * kGaborish[4]),
  };
  Image3F sharpened(in.xsize(), in.ysize());
  slow::SymmetricConvolution<2, WrapClamp>::Run(in, Rect(in), sharpen_weights5,
                                                pool, &sharpened);
  return sharpened;
}

HWY_ATTR void ConvolveGaborish(const ImageF& in, float weight1, float weight2,
                               ThreadPool* pool, ImageF* JXL_RESTRICT out) {
  JXL_CHECK(SameSize(in, *out));

  const GaborishKernel gaborish(weight1, weight2);
  if (in.xsize() < kConvolveMinWidth) {
    using Convolution = slow::General3x3Convolution<1, WrapMirror>;
    Convolution::Run(in, Rect(in), gaborish, out);
  } else {
    using Conv3 = ConvolveT<strategy::Symmetric3>;
    Conv3::Run(in, Rect(in), gaborish, pool, out);
  }
}

}  // namespace jxl
