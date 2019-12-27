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

#ifndef JXL_GAUSS_BLUR_H_
#define JXL_GAUSS_BLUR_H_

#include <math.h>
#include <stddef.h>

#include <vector>

#include "jxl/base/status.h"
#include "jxl/image.h"

namespace jxl {

template <typename T>
std::vector<T> GaussianKernel(int radius, T sigma) {
  JXL_ASSERT(sigma > 0.0);
  std::vector<T> kernel(2 * radius + 1);
  const T scaler = -1.0 / (2 * sigma * sigma);
  double sum = 0.0;
  for (int i = -radius; i <= radius; ++i) {
    const T val = std::exp(scaler * i * i);
    kernel[i + radius] = val;
    sum += val;
  }
  for (size_t i = 0; i < kernel.size(); ++i) {
    kernel[i] /= sum;
  }
  return kernel;
}

// All convolution functions below apply mirroring of the input on the borders
// in the following way:
//
//     input: [a0 a1 a2 ...  aN]
//     mirrored input: [aR ... a1 | a0 a1 a2 .... aN | aN-1 ... aN-R]
//
// where R is the radius of the kernel (i.e. kernel size is 2*R+1).

// TODO(janwas): Deprecated, use ConvolveT instead (if |kernel| <= 5).
ImageF Convolve(const ImageF& in, const std::vector<float>& kernel);
Image3F Convolve(const Image3F& in, const std::vector<float>& kernel);

// TODO(janwas): Deprecated, use ConvolveT instead (if |kernel| <= 5).
ImageF Convolve(const ImageF& in, const std::vector<float>& kernel_x,
                const std::vector<float>& kernel_y);
Image3F Convolve(const Image3F& in, const std::vector<float>& kernel_x,
                 const std::vector<float>& kernel_y);

// TODO(janwas): Use ConvolveT instead (if |kernel| <= 5 and res == 1).
// REQUIRES: in.xsize() and in.ysize() are integer multiples of res.
ImageF ConvolveAndSample(const ImageF& in, const std::vector<float>& kernel,
                         const size_t res);
ImageF ConvolveAndSample(const ImageF& in, const std::vector<float>& kernel_x,
                         const std::vector<float>& kernel_y, const size_t res);

// TODO(janwas): Use ConvolveT instead (if |kernel| <= 5 and res == 1).
ImageF ConvolveXSampleAndTranspose(const ImageF& in,
                                   const std::vector<float>& kernel,
                                   const size_t res);
Image3F ConvolveXSampleAndTranspose(const Image3F& in,
                                    const std::vector<float>& kernel,
                                    const size_t res);

}  // namespace jxl

#endif  // JXL_GAUSS_BLUR_H_
