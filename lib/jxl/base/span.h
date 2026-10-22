// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_SPAN_H_
#define LIB_JXL_BASE_SPAN_H_

// Span (array view) is a non-owning container that provides cheap "cut"
// operations and could be used as "ArrayLike" data source for PaddedBytes.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "lib/jxl/base/status.h"

namespace jxl {

template <typename T>
class Span {
 public:
  constexpr Span() noexcept : Span(nullptr, 0) {}

  constexpr Span(T* array, size_t length) noexcept
      : ptr_(array), len_(length) {}

  template <size_t N>
  explicit constexpr Span(T (&a)[N]) noexcept : Span(a, N) {}

  template <typename U>
  constexpr Span(U* array, size_t length) noexcept
      : ptr_(reinterpret_cast<T*>(array)), len_(length) {
    static_assert(sizeof(U) == sizeof(T), "Incompatible type of source.");
  }

  template <typename ArrayLike>
  explicit constexpr Span(const ArrayLike& other) noexcept
      : Span(reinterpret_cast<T*>(other.data()), other.size()) {
    static_assert(sizeof(*other.data()) == sizeof(T),
                  "Incompatible type of source.");
  }

  using NCT = typename std::remove_const<T>::type;

  constexpr T* data() const noexcept { return ptr_; }

  constexpr size_t size() const noexcept { return len_; }

  constexpr bool empty() const noexcept { return len_ == 0; }

  constexpr T* begin() const noexcept { return data(); }

  constexpr T* end() const noexcept { return data() + size(); }

  constexpr T& operator[](size_t i) const noexcept {
    JXL_DASSERT(i < len_);
    // MSVC 2015 accepts this as constexpr, but not ptr_[i]
    return *(data() + i);
  }

  Status remove_prefix(size_t n) noexcept {
    JXL_ENSURE(size() >= n);
    ptr_ += n;
    len_ -= n;
    return true;
  }

  // Bounds-checked replacement for `Span<T>(span.data() + offset, count)`.
  //
  // Returns an error rather than aborting: callers derive `offset` and `count`
  // from untrusted input, so an out-of-range request is ordinary rejection of a
  // malformed file, not a programmer error. (JXL_ENSURE would be wrong here --
  // it aborts in debug and fuzzer builds.)
  //
  // `len_ - offset` is only evaluated once `offset <= len_` is known to hold,
  // so a caller passing an underflowed `count` is rejected rather than
  // wrapping.
  StatusOr<Span<T>> subspan(size_t offset, size_t count) const {
    if (offset > len_ || count > len_ - offset) {
      return JXL_FAILURE("Span::subspan out of range");
    }
    return Span<T>(ptr_ + offset, count);
  }

  // Bounds-checked suffix starting at `offset`.
  StatusOr<Span<T>> subspan(size_t offset) const {
    if (offset > len_) return JXL_FAILURE("Span::subspan out of range");
    return Span<T>(ptr_ + offset, len_ - offset);
  }

  void AppendTo(std::vector<NCT>& dst) const {
    dst.insert(dst.end(), begin(), end());
  }

  std::vector<NCT> Copy() const { return std::vector<NCT>(begin(), end()); }

 private:
  T* ptr_;
  size_t len_;
};

using Bytes = Span<const uint8_t>;

}  // namespace jxl

#endif  // LIB_JXL_BASE_SPAN_H_
