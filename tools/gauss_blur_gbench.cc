// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/memory_manager.h>

#include <hwy/targets.h>

#include "benchmark/benchmark.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image_ops.h"
#include "tools/gauss_blur.h"
#include "tools/no_memory_manager.h"

namespace jxl {
namespace {

#define QUIT(M)           \
  state.SkipWithError(M); \
  return;

#define BM_CHECK(C) \
  if (!(C)) {       \
    QUIT(#C)        \
  }

void BM_GaussBlur1d(benchmark::State& state) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  // Uncomment to disable SIMD and force and scalar implementation
  // hwy::DisableTargets(~HWY_SCALAR);
  // Uncomment to run AVX2
  // hwy::DisableTargets(HWY_AVX3);

  const size_t length = state.range();
  const double sigma = 7.0;  // (from Butteraugli application)
  JXL_ASSIGN_OR_QUIT(ImageF in, ImageF::Create(memory_manager, length, 1),
                     "Failed to allocate image.");
  const float expected = length;
  FillImage(expected, &in);

  JXL_ASSIGN_OR_QUIT(ImageF temp, ImageF::Create(memory_manager, length, 1),
                     "Failed to allocate image.");
  JXL_ASSIGN_OR_QUIT(ImageF out, ImageF::Create(memory_manager, length, 1),
                     "Failed to allocate image.");
  const auto rg = CreateRecursiveGaussian(sigma);
  for (auto _ : state) {
    FastGaussian1D(rg, length, in.Row(0), out.Row(0));
    // Prevent optimizing out
    BM_CHECK(std::abs(out.ConstRow(0)[length / 2] - expected) / expected <
             9E-5);
  }
  state.SetItemsProcessed(length * state.iterations());
}

void BM_GaussBlur2d(benchmark::State& state) {
  JxlMemoryManager* memory_manager = jpegxl::tools::NoMemoryManager();
  // See GaussBlur1d for SIMD changes.

  const size_t xsize = state.range();
  const size_t ysize = xsize;
  const double sigma = 7.0;  // (from Butteraugli application)
  JXL_ASSIGN_OR_QUIT(ImageF in, ImageF::Create(memory_manager, xsize, ysize),
                     "Failed to allocate image.");
  const float expected = xsize + ysize;
  FillImage(expected, &in);

  JXL_ASSIGN_OR_QUIT(ImageF temp, ImageF::Create(memory_manager, xsize, ysize),
                     "Failed to allocate image.");
  JXL_ASSIGN_OR_QUIT(ImageF out, ImageF::Create(memory_manager, xsize, ysize),
                     "Failed to allocate image.");
  const auto rg = CreateRecursiveGaussian(sigma);
  for (auto _ : state) {
    BM_CHECK(FastGaussian(
        rg, in.xsize(), in.ysize(), [&](size_t y) { return in.ConstRow(y); },
        [&](size_t y) { return temp.Row(y); },
        [&](size_t y) { return out.Row(y); }));
    // Prevent optimizing out
    BM_CHECK(std::abs(out.ConstRow(ysize / 2)[xsize / 2] - expected) /
                 expected <
             9E-5);
  }
  state.SetItemsProcessed(xsize * ysize * state.iterations());
}

BENCHMARK(BM_GaussBlur1d)->Range(1 << 8, 1 << 14);
BENCHMARK(BM_GaussBlur2d)->Range(1 << 7, 1 << 10);

}  // namespace
}  // namespace jxl
