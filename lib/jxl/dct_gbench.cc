// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <cstdint>

#include "benchmark/benchmark.h"
#include "lib/jxl/base/compiler_specific.h"

#ifndef SIZE_LIST
#define SIZE_LIST(APPLY) \
  APPLY(2, 2)            \
  APPLY(4, 4)            \
  APPLY(8, 8)            \
  APPLY(16, 16)          \
  APPLY(32, 32)          \
  APPLY(16, 8)           \
  APPLY(8, 16)           \
  APPLY(32, 8)           \
  APPLY(8, 32)           \
  APPLY(32, 16)          \
  APPLY(16, 32)          \
  APPLY(4, 8)            \
  APPLY(8, 4)            \
  APPLY(64, 64)          \
  APPLY(64, 32)          \
  APPLY(32, 64)          \
  APPLY(128, 128)        \
  APPLY(128, 64)         \
  APPLY(64, 128)         \
  APPLY(256, 256)        \
  APPLY(256, 128)        \
  APPLY(128, 256)
#endif

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/dct_gbench.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/dct-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {

namespace {

template <size_t ROWS, size_t COLS>
void RunDct(benchmark::State& state, float* JXL_RESTRICT values1,
            size_t values_stride, float* JXL_RESTRICT values2,
            float* JXL_RESTRICT scratch_space) {
  // Swap src-dst to make sure loop is not optimized out.
  float* JXL_RESTRICT values[2] = {values1, values2};
  size_t i = 0;
  for (auto _ : state) {
    (void)_;
    ComputeScaledDCT<ROWS, COLS>()(DCTFrom(values[i & 1], values_stride),
                                   values[(i + 1) & 1], scratch_space);
    i++;
  }
  // Measure per-item performance for better understanding on scaling.
  state.SetItemsProcessed(state.iterations() * ROWS * COLS);
}

template <size_t ROWS, size_t COLS>
void RunIdct(benchmark::State& state, float* JXL_RESTRICT values1,
             size_t values_stride, float* JXL_RESTRICT values2,
             float* JXL_RESTRICT scratch_space) {
  // Swap src-dst to make sure loop is not optimized out.
  float* JXL_RESTRICT values[2] = {values1, values2};
  size_t i = 0;
  for (auto _ : state) {
    (void)_;
    ComputeScaledIDCT<ROWS, COLS>()(values[i & 1],
                                    DCTTo(values[(i + 1) & 1], values_stride),
                                    scratch_space);
    i++;
  }
  // Measure per-item performance for better understanding on scaling.
  state.SetItemsProcessed(state.iterations() * ROWS * COLS);
}

#define IMPL_BM(R, C)                                                         \
  HWY_NOINLINE void BM_DCT##R##x##C(                                          \
      benchmark::State& state, float* JXL_RESTRICT pixels,                    \
      size_t pixels_stride, float* JXL_RESTRICT coefficients,                 \
      float* JXL_RESTRICT scratch_space) {                                    \
    RunDct<R, C>(state, pixels, pixels_stride, coefficients, scratch_space);  \
  }                                                                           \
  HWY_NOINLINE void BM_IDCT##R##x##C(                                         \
      benchmark::State& state, float* JXL_RESTRICT pixels,                    \
      size_t pixels_stride, float* JXL_RESTRICT coefficients,                 \
      float* JXL_RESTRICT scratch_space) {                                    \
    RunIdct<R, C>(state, pixels, pixels_stride, coefficients, scratch_space); \
  }
SIZE_LIST(IMPL_BM)
#undef IMPL_BM

}  // namespace
// NOLINTNEXTLINE(google-readability-namespace-comments)
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
namespace {

#define DEFINE_BM(R, C)                                            \
  HWY_EXPORT(BM_DCT##R##x##C);                                     \
  void BM_DCT##R##x##C(benchmark::State& state, int64_t target) {  \
    hwy::SetSupportedTargetsForTest(target);                       \
    HWY_ALIGN_MAX float pixels[256 * 256] = {0};                   \
    HWY_ALIGN_MAX float coeffs[256 * 256] = {0};                   \
    HWY_ALIGN_MAX float scratch[4 * 256 * 256] = {0};              \
    HWY_DYNAMIC_DISPATCH(BM_DCT##R##x##C)                          \
    (state, pixels, 256, coeffs, scratch);                         \
    hwy::SetSupportedTargetsForTest(0);                            \
  }                                                                \
                                                                   \
  HWY_EXPORT(BM_IDCT##R##x##C);                                    \
  void BM_IDCT##R##x##C(benchmark::State& state, int64_t target) { \
    hwy::SetSupportedTargetsForTest(target);                       \
    HWY_ALIGN_MAX float pixels[256 * 256] = {0};                   \
    HWY_ALIGN_MAX float coeffs[256 * 256] = {0};                   \
    HWY_ALIGN_MAX float scratch[4 * 256 * 256] = {0};              \
    HWY_DYNAMIC_DISPATCH(BM_IDCT##R##x##C)                         \
    (state, pixels, 256, coeffs, scratch);                         \
    hwy::SetSupportedTargetsForTest(0);                            \
  }
SIZE_LIST(DEFINE_BM)
#undef DEFINE_BM

}  // namespace
}  // namespace jxl

void RegisterDctBenchmarks();

void JXL_MAYBE_UNUSED RegisterDctBenchmarks() {
#define REGISTER_BM(R, C)                                                  \
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {             \
    std::string target_name(hwy::TargetName(target));                      \
    std::string dct_name = "DCT" #R "x" #C "/" + target_name;              \
    benchmark::RegisterBenchmark(dct_name.c_str(), jxl::BM_DCT##R##x##C,   \
                                 target);                                  \
  }                                                                        \
  for (int64_t target : hwy::SupportedAndGeneratedTargets()) {             \
    std::string target_name(hwy::TargetName(target));                      \
    std::string idct_name = "IDCT" #R "x" #C "/" + target_name;            \
    benchmark::RegisterBenchmark(idct_name.c_str(), jxl::BM_IDCT##R##x##C, \
                                 target);                                  \
  }
  SIZE_LIST(REGISTER_BM)
#undef REGISTER_BM
}

#endif
