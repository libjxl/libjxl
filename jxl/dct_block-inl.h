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

// Adapters for DCT input/output: from/to contiguous blocks or image rows.

#if defined(JXL_DCT_BLOCK_INL_H_) == defined(HWY_TARGET_TOGGLE)
#ifdef JXL_DCT_BLOCK_INL_H_
#undef JXL_DCT_BLOCK_INL_H_
#else
#define JXL_DCT_BLOCK_INL_H_
#endif

#include <stddef.h>

#include "jxl/base/status.h"

// SIMD code
#include <hwy/before_namespace-inl.h>
namespace jxl {
#include <hwy/begin_target-inl.h>

// Block: (x, y) <-> (N * y + x)
// Lines: (x, y) <-> (stride * y + x)
//
// I.e. Block is a specialization of Lines with fixed stride.
//
// FromXXX should implement Read and Load (Read vector).
// ToXXX should implement Write and Store (Write vector).

template <size_t N>
using BlockDesc = HWY_CAPPED(float, N);

template <size_t N>
struct DCTSizeTag {};

// Here and in the following, the SZ template parameter specifies the number of
// values to load/store. Needed because we want to handle 4x4 sub-blocks of
// 16x16 blocks.
class FromBlock {
 public:
  FromBlock(size_t rows, size_t cols, const float* block)
      : cols_(cols), block_(block) {}

  template <typename D>
  HWY_INLINE Vec<D> LoadPart(D, const size_t row, size_t i) const {
    JXL_DASSERT(Lanes(D()) <= cols_);
    return Load(D(), Address(row, i));
  }

  HWY_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  constexpr HWY_INLINE const float* Address(const size_t row,
                                            const size_t i) const {
    return block_ + row * cols_ + i;
  }

 private:
  size_t cols_;
  const float* block_;
};

class ToBlock {
 public:
  ToBlock(size_t rows, size_t cols, float* block)
      : cols_(cols), block_(block) {}

  template <typename D>
  HWY_INLINE void StorePart(D, const Vec<D>& v, const size_t row,
                            const size_t i) const {
    JXL_DASSERT(Lanes(D()) <= cols_);
    Store(v, D(), Address(row, i));
  }

  HWY_INLINE void Write(float v, const size_t row, const size_t i) const {
    *Address(row, i) = v;
  }

  constexpr HWY_INLINE float* Address(const size_t row, const size_t i) const {
    return block_ + row * cols_ + i;
  }

 private:
  size_t cols_;
  float* block_;
};

// Same as ToBlock, but multiplies result by (N * N)
class ScaleToBlock {
 public:
  ScaleToBlock(size_t rows, size_t cols, float* block)
      : cols_(cols), mul_(1.0f / (cols * rows)), block_(block) {}

  template <typename D>
  HWY_INLINE void StorePart(D, const Vec<D>& v, const size_t row,
                            const size_t i) const {
    JXL_DASSERT(Lanes(D()) <= cols_);
    const auto mul = Set(D(), mul_);
    StoreU(v * mul, D(), Address(row, i));
  }

  HWY_INLINE void Write(float v, const size_t row, const size_t i) const {
    *Address(row, i) = v * mul_;
  }

  constexpr HWY_INLINE float* Address(const size_t row, const size_t i) const {
    return block_ + row * cols_ + i;
  }

 private:
  size_t cols_;
  float mul_;
  float* block_;
};

class FromLines {
 public:
  FromLines(const float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  template <typename D>
  HWY_INLINE Vec<D> LoadPart(D, const size_t row, const size_t i) const {
    return LoadU(D(), Address(row, i));
  }

  HWY_INLINE float Read(const size_t row, const size_t i) const {
    return *Address(row, i);
  }

  HWY_INLINE const float* Address(const size_t row, const size_t i) const {
    return top_left_ + row * stride_ + i;
  }

 private:
  const float* HWY_RESTRICT top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

// Pointers are restrict-qualified: assumes we don't use both FromLines and
// ToLines in the same DCT. NOTE: Transpose uses From/ToBlock, not *Lines.
class ToLines {
 public:
  ToLines(float* top_left, size_t stride)
      : top_left_(top_left), stride_(stride) {}

  template <typename D>
  HWY_INLINE void StorePart(D, const Vec<D>& v, const size_t row,
                            const size_t i) const {
    StoreU(v, D(), Address(row, i));
  }

  HWY_INLINE void Write(float v, const size_t row, const size_t i) const {
    *Address(row, i) = v;
  }

  HWY_INLINE float* Address(const size_t row, const size_t i) const {
    return top_left_ + row * stride_ + i;
  }

 private:
  float* HWY_RESTRICT top_left_;
  size_t stride_;  // move to next line by adding this to pointer
};

#include <hwy/end_target-inl.h>
}  // namespace jxl
#include <hwy/after_namespace-inl.h>

#endif  // include guard
