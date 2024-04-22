// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_IMAGE_H_
#define LIB_JXL_IMAGE_H_

// SIMD/multicore-friendly planar image representation with row accessors.

#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
#include <cinttypes>  // PRIu64
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>  // std::move

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/cache_aligned.h"

namespace jxl {

// DO NOT use PlaneBase outside of image.{h|cc}
namespace detail {

// Type-independent parts of Plane<> - reduces code duplication and facilitates
// moving member function implementations to cc file.
struct PlaneBase {
  PlaneBase()
      : xsize_(0),
        ysize_(0),
        orig_xsize_(0),
        orig_ysize_(0),
        bytes_per_row_(0),
        bytes_(nullptr),
        sizeof_t_(0) {}

  // Copy construction/assignment is forbidden to avoid inadvertent copies,
  // which can be very expensive. Use CopyImageTo() instead.
  PlaneBase(const PlaneBase& other) = delete;
  PlaneBase& operator=(const PlaneBase& other) = delete;

  // Move constructor (required for returning Image from function)
  PlaneBase(PlaneBase&& other) noexcept = default;

  // Move assignment (required for std::vector)
  PlaneBase& operator=(PlaneBase&& other) noexcept = default;

  void Swap(PlaneBase& other);

  // Useful for pre-allocating image with some padding for alignment purposes
  // and later reporting the actual valid dimensions. May also be used to
  // un-shrink the image. Caller is responsible for ensuring xsize/ysize are <=
  // the original dimensions.
  void ShrinkTo(const size_t xsize, const size_t ysize) {
    JXL_CHECK(xsize <= orig_xsize_);
    JXL_CHECK(ysize <= orig_ysize_);
    xsize_ = static_cast<uint32_t>(xsize);
    ysize_ = static_cast<uint32_t>(ysize);
    // NOTE: we can't recompute bytes_per_row for more compact storage and
    // better locality because that would invalidate the image contents.
  }

  // How many pixels.
  JXL_INLINE size_t xsize() const { return xsize_; }
  JXL_INLINE size_t ysize() const { return ysize_; }

  // NOTE: do not use this for copying rows - the valid xsize may be much less.
  JXL_INLINE size_t bytes_per_row() const { return bytes_per_row_; }

  // Raw access to byte contents, for interfacing with other libraries.
  // Unsigned char instead of char to avoid surprises (sign extension).
  JXL_INLINE uint8_t* bytes() {
    void* p = bytes_.get();
    return static_cast<uint8_t * JXL_RESTRICT>(JXL_ASSUME_ALIGNED(p, 64));
  }
  JXL_INLINE const uint8_t* bytes() const {
    const void* p = bytes_.get();
    return static_cast<const uint8_t * JXL_RESTRICT>(JXL_ASSUME_ALIGNED(p, 64));
  }

 protected:
  PlaneBase(size_t xsize, size_t ysize, size_t sizeof_t);
  Status Allocate();

  // Returns pointer to the start of a row.
  JXL_INLINE void* VoidRow(const size_t y) const {
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
    if (y >= ysize_) {
      JXL_ABORT("Row(%" PRIu64 ") in (%u x %u) image\n",
                static_cast<uint64_t>(y), xsize_, ysize_);
    }
#endif

    void* row = bytes_.get() + y * bytes_per_row_;
    return JXL_ASSUME_ALIGNED(row, 64);
  }

  // (Members are non-const to enable assignment during move-assignment.)
  uint32_t xsize_;  // In valid pixels, not including any padding.
  uint32_t ysize_;
  uint32_t orig_xsize_;
  uint32_t orig_ysize_;
  size_t bytes_per_row_;  // Includes padding.
  CacheAlignedUniquePtr bytes_;
  size_t sizeof_t_;
};

}  // namespace detail

// Single channel, aligned rows separated by padding. T must be POD.
//
// 'Single channel' (one 2D array per channel) simplifies vectorization
// (repeating the same operation on multiple adjacent components) without the
// complexity of a hybrid layout (8 R, 8 G, 8 B, ...). In particular, clients
// can easily iterate over all components in a row and Image requires no
// knowledge of the pixel format beyond the component type "T".
//
// 'Aligned' means each row is aligned to the L1 cache line size. This prevents
// false sharing between two threads operating on adjacent rows.
//
// 'Padding' is still relevant because vectors could potentially be larger than
// a cache line. By rounding up row sizes to the vector size, we allow
// reading/writing ALIGNED vectors whose first lane is a valid sample. This
// avoids needing a separate loop to handle remaining unaligned lanes.
//
// This image layout could also be achieved with a vector and a row accessor
// function, but a class wrapper with support for "deleter" allows wrapping
// existing memory allocated by clients without copying the pixels. It also
// provides convenient accessors for xsize/ysize, which shortens function
// argument lists. Supports move-construction so it can be stored in containers.
template <typename ComponentType>
class Plane : public detail::PlaneBase {
 public:
  using T = ComponentType;
  static constexpr size_t kNumPlanes = 1;

  Plane() = default;

  static StatusOr<Plane> Create(const size_t xsize, const size_t ysize) {
    Plane plane(xsize, ysize, sizeof(T));
    JXL_RETURN_IF_ERROR(plane.Allocate());
    return plane;
  }

  JXL_INLINE T* Row(const size_t y) { return static_cast<T*>(VoidRow(y)); }

  // Returns pointer to const (see above).
  JXL_INLINE const T* Row(const size_t y) const {
    return static_cast<const T*>(VoidRow(y));
  }

  // Documents that the access is const.
  JXL_INLINE const T* ConstRow(const size_t y) const {
    return static_cast<const T*>(VoidRow(y));
  }

  // Returns number of pixels (some of which are padding) per row. Useful for
  // computing other rows via pointer arithmetic. WARNING: this must
  // NOT be used to determine xsize.
  JXL_INLINE intptr_t PixelsPerRow() const {
    return static_cast<intptr_t>(bytes_per_row_ / sizeof(T));
  }

 private:
  Plane(size_t xsize, size_t ysize, size_t sizeof_t)
      : detail::PlaneBase(xsize, ysize, sizeof_t) {}
};

using ImageSB = Plane<int8_t>;
using ImageB = Plane<uint8_t>;
using ImageS = Plane<int16_t>;  // signed integer or half-float
using ImageU = Plane<uint16_t>;
using ImageI = Plane<int32_t>;
using ImageF = Plane<float>;
using ImageD = Plane<double>;

// Currently, we abuse Image to either refer to an image that owns its storage
// or one that doesn't. In similar vein, we abuse Image* function parameters to
// either mean "assign to me" or "fill the provided image with data".
// Hopefully, the "assign to me" meaning will go away and most images in the
// codebase will not be backed by own storage. When this happens we can redesign
// Image to be a non-storage-holding view class and introduce BackedImage in
// those places that actually need it.

// NOTE: we can't use Image as a view because invariants are violated
// (alignment and the presence of padding before/after each "row").

// A bundle of 3 same-sized images. Typically constructed by moving from three
// rvalue references to Image. To overwrite an existing Image3 using
// single-channel producers, we also need access to Image*. Constructing
// temporary non-owning Image pointing to one plane of an existing Image3 risks
// dangling references, especially if the wrapper is moved. Therefore, we
// store an array of Image (which are compact enough that size is not a concern)
// and provide Plane+Row accessors.
template <typename ComponentType>
class Image3 {
 public:
  using T = ComponentType;
  using PlaneT = jxl::Plane<T>;
  static constexpr size_t kNumPlanes = 3;

  Image3() : planes_{PlaneT(), PlaneT(), PlaneT()} {}

  Image3(Image3&& other) noexcept {
    for (size_t i = 0; i < kNumPlanes; i++) {
      planes_[i] = std::move(other.planes_[i]);
    }
  }

  // Copy construction/assignment is forbidden to avoid inadvertent copies,
  // which can be very expensive. Use CopyImageTo instead.
  Image3(const Image3& other) = delete;
  Image3& operator=(const Image3& other) = delete;

  Image3& operator=(Image3&& other) noexcept {
    for (size_t i = 0; i < kNumPlanes; i++) {
      planes_[i] = std::move(other.planes_[i]);
    }
    return *this;
  }

  static StatusOr<Image3> Create(const size_t xsize, const size_t ysize) {
    StatusOr<PlaneT> plane0 = PlaneT::Create(xsize, ysize);
    JXL_RETURN_IF_ERROR(plane0.status());
    StatusOr<PlaneT> plane1 = PlaneT::Create(xsize, ysize);
    JXL_RETURN_IF_ERROR(plane1.status());
    StatusOr<PlaneT> plane2 = PlaneT::Create(xsize, ysize);
    JXL_RETURN_IF_ERROR(plane2.status());
    return Image3(std::move(plane0).value(), std::move(plane1).value(),
                  std::move(plane2).value());
  }

  // Returns row pointer; usage: PlaneRow(idx_plane, y)[x] = val.
  JXL_INLINE T* PlaneRow(const size_t c, const size_t y) {
    // Custom implementation instead of calling planes_[c].Row ensures only a
    // single multiplication is needed for PlaneRow(0..2, y).
    PlaneRowBoundsCheck(c, y);
    const size_t row_offset = y * planes_[0].bytes_per_row();
    void* row = planes_[c].bytes() + row_offset;
    return static_cast<T * JXL_RESTRICT>(JXL_ASSUME_ALIGNED(row, 64));
  }

  // Returns const row pointer; usage: val = PlaneRow(idx_plane, y)[x].
  JXL_INLINE const T* PlaneRow(const size_t c, const size_t y) const {
    PlaneRowBoundsCheck(c, y);
    const size_t row_offset = y * planes_[0].bytes_per_row();
    const void* row = planes_[c].bytes() + row_offset;
    return static_cast<const T * JXL_RESTRICT>(JXL_ASSUME_ALIGNED(row, 64));
  }

  // Returns const row pointer, even if called from a non-const Image3.
  JXL_INLINE const T* ConstPlaneRow(const size_t c, const size_t y) const {
    PlaneRowBoundsCheck(c, y);
    return PlaneRow(c, y);
  }

  JXL_INLINE const PlaneT& Plane(size_t idx) const { return planes_[idx]; }

  JXL_INLINE PlaneT& Plane(size_t idx) { return planes_[idx]; }

  void Swap(Image3& other) {
    for (size_t c = 0; c < 3; ++c) {
      other.planes_[c].Swap(planes_[c]);
    }
  }

  // Useful for pre-allocating image with some padding for alignment purposes
  // and later reporting the actual valid dimensions. May also be used to
  // un-shrink the image. Caller is responsible for ensuring xsize/ysize are <=
  // the original dimensions.
  void ShrinkTo(const size_t xsize, const size_t ysize) {
    for (PlaneT& plane : planes_) {
      plane.ShrinkTo(xsize, ysize);
    }
  }

  // Sizes of all three images are guaranteed to be equal.
  JXL_INLINE size_t xsize() const { return planes_[0].xsize(); }
  JXL_INLINE size_t ysize() const { return planes_[0].ysize(); }
  // Returns offset [bytes] from one row to the next row of the same plane.
  // WARNING: this must NOT be used to determine xsize, nor for copying rows -
  // the valid xsize may be much less.
  JXL_INLINE size_t bytes_per_row() const { return planes_[0].bytes_per_row(); }
  // Returns number of pixels (some of which are padding) per row. Useful for
  // computing other rows via pointer arithmetic. WARNING: this must NOT be used
  // to determine xsize.
  JXL_INLINE intptr_t PixelsPerRow() const { return planes_[0].PixelsPerRow(); }

 private:
  Image3(PlaneT&& plane0, PlaneT&& plane1, PlaneT&& plane2) {
    planes_[0] = std::move(plane0);
    planes_[1] = std::move(plane1);
    planes_[2] = std::move(plane2);
  }

  void PlaneRowBoundsCheck(const size_t c, const size_t y) const {
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
    if (c >= kNumPlanes || y >= ysize()) {
      JXL_ABORT("PlaneRow(%" PRIu64 ", %" PRIu64 ") in (%" PRIu64 " x %" PRIu64
                ") image\n",
                static_cast<uint64_t>(c), static_cast<uint64_t>(y),
                static_cast<uint64_t>(xsize()), static_cast<uint64_t>(ysize()));
    }
#endif
  }

  PlaneT planes_[kNumPlanes];
};

using Image3B = Image3<uint8_t>;
using Image3S = Image3<int16_t>;
using Image3U = Image3<uint16_t>;
using Image3I = Image3<int32_t>;
using Image3F = Image3<float>;
using Image3D = Image3<double>;

}  // namespace jxl

#endif  // LIB_JXL_IMAGE_H_
