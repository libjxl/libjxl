// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Author: Jyrki Alakuijala (jyrki.alakuijala@gmail.com)

#ifndef LIB_JXL_BUTTERAUGLI_BUTTERAUGLI_H_
#define LIB_JXL_BUTTERAUGLI_BUTTERAUGLI_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>
#include <cmath>
#include <memory>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/common.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

#define BUTTERAUGLI_ENABLE_CHECKS 0
#define BUTTERAUGLI_RESTRICT JXL_RESTRICT

// This is the main interface to butteraugli image similarity
// analysis function.

namespace jxl {

struct ButteraugliParams {
  // Multiplier for penalizing new HF artifacts more than blurring away
  // features. 1.0=neutral.
  float hf_asymmetry = 1.0f;

  // Multiplier for the psychovisual difference in the X channel.
  float xmul = 1.0f;

  // Number of nits that correspond to 1.0f input values.
  float intensity_target = 80.0f;
};

// Converts the butteraugli score into fuzzy class values that are continuous
// at the class boundary. The class boundary location is based on human
// raters, but the slope is arbitrary. Particularly, it does not reflect
// the expectation value of probabilities of the human raters. It is just
// expected that a smoother class boundary will allow for higher-level
// optimization algorithms to work faster.
//
// Returns 2.0 for a perfect match, and 1.0 for 'ok', 0.0 for bad. Because the
// scoring is fuzzy, a butteraugli score of 0.96 would return a class of
// around 1.9.
double ButteraugliFuzzyClass(double score);

// Input values should be in range 0 (bad) to 2 (good). Use
// kButteraugliNormalization as normalization.
double ButteraugliFuzzyInverse(double seek);

// Implementation details, don't use anything below or your code will
// break in the future.

#ifdef _MSC_VER
#define BUTTERAUGLI_INLINE __forceinline
#else
#define BUTTERAUGLI_INLINE inline
#endif

#ifdef __clang__
// Early versions of Clang did not support __builtin_assume_aligned.
#define BUTTERAUGLI_HAS_ASSUME_ALIGNED __has_builtin(__builtin_assume_aligned)
#elif defined(__GNUC__)
#define BUTTERAUGLI_HAS_ASSUME_ALIGNED 1
#else
#define BUTTERAUGLI_HAS_ASSUME_ALIGNED 0
#endif

// Returns a void* pointer which the compiler then assumes is N-byte aligned.
// Example: float* JXL_RESTRICT aligned = (float*)JXL_ASSUME_ALIGNED(in, 32);
//
// The assignment semantics are required by GCC/Clang. ICC provides an in-place
// __assume_aligned, whereas MSVC's __assume appears unsuitable.
#if BUTTERAUGLI_HAS_ASSUME_ALIGNED
#define BUTTERAUGLI_ASSUME_ALIGNED(ptr, align) \
  __builtin_assume_aligned((ptr), (align))
#else
#define BUTTERAUGLI_ASSUME_ALIGNED(ptr, align) (ptr)
#endif  // BUTTERAUGLI_HAS_ASSUME_ALIGNED

struct PsychoImage {
  ImageF uhf[2];  // XY
  ImageF hf[2];   // XY
  Image3F mf;     // XYB
  Image3F lf;     // XYB
};

// Blur needs a transposed image.
// Hold it here and only allocate on demand to reduce memory usage.
struct BlurTemp {
  ImageF *GetTransposed(const ImageF &in) {
    if (transposed_temp.xsize() == 0) {
      transposed_temp = ImageF(in.ysize(), in.xsize());
    }
    return &transposed_temp;
  }

  ImageF transposed_temp;
};

class ButteraugliComparator {
 public:
  // Butteraugli is calibrated at xmul = 1.0. We add a multiplier here so that
  // we can test the hypothesis that a higher weighing of the X channel would
  // improve results at higher Butteraugli values.
  ButteraugliComparator(const Image3F &rgb0, const ButteraugliParams &params);
  virtual ~ButteraugliComparator() = default;

  // Computes the butteraugli map between the original image given in the
  // constructor and the distorted image give here.
  void Diffmap(const Image3F &rgb1, ImageF &result) const;

  // Same as above, but OpsinDynamicsImage() was already applied.
  void DiffmapOpsinDynamicsImage(const Image3F &xyb1, ImageF &result) const;

  // Same as above, but the frequency decomposition was already applied.
  void DiffmapPsychoImage(const PsychoImage &pi1, ImageF &diffmap) const;

  void Mask(ImageF *BUTTERAUGLI_RESTRICT mask) const;

 private:
  Image3F *Temp() const;
  void ReleaseTemp() const;

  const size_t xsize_;
  const size_t ysize_;
  ButteraugliParams params_;
  PsychoImage pi0_;

  // Shared temporary image storage to reduce the number of allocations;
  // obtained via Temp(), must call ReleaseTemp when no longer needed.
  mutable Image3F temp_;
  mutable std::atomic_flag temp_in_use_ = ATOMIC_FLAG_INIT;

  mutable BlurTemp blur_temp_;
  std::unique_ptr<ButteraugliComparator> sub_;
};

double ButteraugliScoreFromDiffmap(const ImageF &diffmap,
                                   const ButteraugliParams *params = nullptr);

// Generate rgb-representation of the distance between two images.
Image3F CreateHeatMapImage(const ImageF &distmap, double good_threshold,
                           double bad_threshold);

}  // namespace jxl

#endif  // LIB_JXL_BUTTERAUGLI_BUTTERAUGLI_H_
