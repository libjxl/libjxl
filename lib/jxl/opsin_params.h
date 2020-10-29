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

#ifndef LIB_JXL_OPSIN_PARAMS_H_
#define LIB_JXL_OPSIN_PARAMS_H_

// Constants that define the XYB color space.

#include <stdlib.h>

#include <cmath>

#include "lib/jxl/base/compiler_specific.h"

namespace jxl {

static const float kScale = 255.0f;

static const float kScaleR = 1.0f;
static const float kScaleG = 1.0f;
static const float kInvScaleR = 1.0f;
static const float kInvScaleG = 1.0f;

// Parameters for opsin absorbance.
static const float kM02 = 0.078;
static const float kM00 = 0.30;
static const float kM01 = 1.0 - kM02 - kM00;

static const float kM12 = 0.078;
static const float kM10 = 0.23;
static const float kM11 = 1.0 - kM12 - kM10;

static const float kM20 = 0.24342268924547819;
static const float kM21 = 0.20476744424496821;
static const float kM22 = 1.0 - kM20 - kM21;

static const float kBScale = 1.0f;
static const float kYToBRatio = 1.0f;  // works better with 0.50017729543783418
static const float kBToYRatio = 1.0f / kYToBRatio;

// Unscaled values for kOpsinAbsorbanceBias
static const float kB0 = 0.96723368009523958;
static const float kB1 = kB0;
static const float kB2 = kB0;

// Opsin absorbance matrix is now frozen.
static const float kOpsinAbsorbanceMatrix[9] = {
    kM00 / kScale, kM01 / kScale, kM02 / kScale, kM10 / kScale, kM11 / kScale,
    kM12 / kScale, kM20 / kScale, kM21 / kScale, kM22 / kScale,
};

// Must be the inverse matrix of kOpsinAbsorbanceMatrix and match the spec.
static inline const float* DefaultInverseOpsinAbsorbanceMatrix() {
  static float kDefaultInverseOpsinAbsorbanceMatrix[9] = {
      2813.04956,  -2516.07070, -41.9788641, -829.807582, 1126.78645,
      -41.9788641, -933.007078, 691.795377,  496.211701};
  return kDefaultInverseOpsinAbsorbanceMatrix;
}

// Returns 3x3 row-major matrix inverse of kOpsinAbsorbanceMatrix.
// opsin_image_test verifies this is actually the inverse.
const float* GetOpsinAbsorbanceInverseMatrix();

void InitSIMDInverseMatrix(const float* JXL_RESTRICT inverse,
                           float* JXL_RESTRICT simd_inverse,
                           float intensity_target);

static const float kOpsinAbsorbanceBias[3] = {
    kB0 / kScale,
    kB1 / kScale,
    kB2 / kScale,
};

static const float kNegOpsinAbsorbanceBiasRGB[4] = {
    -kOpsinAbsorbanceBias[0], -kOpsinAbsorbanceBias[1],
    -kOpsinAbsorbanceBias[2], 255.0f};

}  // namespace jxl

#endif  // LIB_JXL_OPSIN_PARAMS_H_
