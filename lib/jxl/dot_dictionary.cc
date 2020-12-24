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

#include "lib/jxl/dot_dictionary.h"

#include <stddef.h>
#include <string.h>

#include <array>
#include <utility>

#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_xyb.h"
#include "lib/jxl/detect_dots.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/enc_xyb.h"
#include "lib/jxl/image.h"

namespace jxl {

// Private implementation of Dictionary Encode/Decode
namespace {

/* Quantization constants for Ellipse dots */
const size_t kEllipsePosQ = 2;        // Quantization level for the position
const double kEllipseMinSigma = 0.1;  // Minimum sigma value
const double kEllipseMaxSigma = 3.1;  // Maximum Sigma value
const size_t kEllipseSigmaQ = 16;     // Number of quantization levels for sigma
const size_t kEllipseAngleQ = 8;      // Quantization level for the angle
// TODO: fix these values.
const std::array<double, 3> kEllipseMinIntensity{-0.05, 0.0, -0.5};
const std::array<double, 3> kEllipseMaxIntensity{0.05, 1.0, 0.4};
const std::array<size_t, 3> kEllipseIntensityQ{10, 36, 10};
}  // namespace

namespace {

GaussianDetectParams GetConstDetectParams() {
  GaussianDetectParams ans;
  ans.t_high = 0.04;
  ans.t_low = 0.02;
  ans.maxWinSize = 5;
  ans.maxL2Loss = 0.005;
  ans.maxCustomLoss = 300;
  ans.minIntensity = 0.12;
  ans.maxDistMeanMode = 1.0;
  ans.maxNegPixels = 0;
  ans.minScore = 12.0;
  ans.maxCC = 100;
  ans.percCC = 100;
  return ans;
}

const GaussianDetectParams kEllipseDetectParams = GetConstDetectParams();

}  // namespace

std::vector<PatchInfo> FindDotDictionary(const CompressParams& cparams,
                                         const Image3F& opsin,
                                         const ColorCorrelationMap& cmap,
                                         ThreadPool* pool) {
  if (ApplyOverride(cparams.dots,
                    cparams.butteraugli_distance >= kMinButteraugliForDots)) {
    EllipseQuantParams qParams{
        opsin.xsize(),      opsin.ysize(),        kEllipsePosQ,
        kEllipseMinSigma,   kEllipseMaxSigma,     kEllipseSigmaQ,
        kEllipseAngleQ,     kEllipseMinIntensity, kEllipseMaxIntensity,
        kEllipseIntensityQ, kEllipsePosQ <= 5,    cmap.YtoXRatio(0),
        cmap.YtoBRatio(0)};
    GaussianDetectParams eDetectParams = kEllipseDetectParams;

    return DetectGaussianEllipses(opsin, eDetectParams, qParams, pool);
  }
  return {};
}
}  // namespace jxl
