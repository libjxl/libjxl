// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_MODULAR_MODULAR_IMAGE_H_
#define LIB_JXL_MODULAR_MODULAR_IMAGE_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <utility>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_ops.h"

namespace jxl {

typedef int32_t pixel_type;  // can use int16_t if it's only for 8-bit images.
                             // Need some wiggle room for YCoCg / Squeeze etc

typedef int64_t pixel_type_w;

namespace weighted {
struct Header;
}

class Channel {
 public:
  size_t w, h;
  int hshift, vshift;  // w ~= image.w >> hshift;  h ~= image.h >> vshift
  Channel(size_t iw, size_t ih, int hsh = 0, int vsh = 0)
      : w(iw),
        h(ih),
        hshift(hsh),
        vshift(vsh),
        plane(iw, ih),
        ref_plane(nullptr),
        ref_rect() {}

  Channel(Plane<pixel_type>* ref, Rect& rect, int hsh = 0, int vsh = 0)
      : w(rect.xsize()),
        h(rect.ysize()),
        hshift(hsh),
        vshift(vsh),
        plane(0, 0),
        ref_plane(ref),
        ref_rect(rect) {}

  Channel(const Channel& other) = delete;
  Channel& operator=(const Channel& other) = delete;

  // Move assignment
  Channel& operator=(Channel&& other) noexcept {
    w = other.w;
    h = other.h;
    hshift = other.hshift;
    vshift = other.vshift;
    plane = std::move(other.plane);
    ref_plane = other.ref_plane;
    other.ref_plane = nullptr;
    ref_rect = other.ref_rect;
    return *this;
  }

  // Move constructor
  Channel(Channel&& other) noexcept = default;

  void shrink() {
    JXL_ASSERT(ref_plane == nullptr);
    if (plane.xsize() == w && plane.ysize() == h) return;
    jxl::Plane<pixel_type> resizedplane(w, h);
    plane = std::move(resizedplane);
  }
  void shrink(int nw, int nh) {
    w = nw;
    h = nh;
    shrink();
  }
  intptr_t PixelsPerRow() const {
    if (ref_plane)
      return ref_plane->PixelsPerRow();
    else
      return plane.PixelsPerRow();
  }
  void ZeroFill() {
    JXL_ASSERT(ref_plane == nullptr);
    ZeroFillImage(&plane);
  }
  Plane<pixel_type>* GetPlane() {
    JXL_ASSERT(ref_plane == nullptr);
    return &plane;
  }
  JXL_INLINE pixel_type* Row(const size_t y) {
    if (ref_plane) return ref_rect.Row(ref_plane, y);
    return plane.Row(y);
  }
  JXL_INLINE const pixel_type* Row(const size_t y) const {
    if (ref_plane) return ref_rect.ConstRow(*ref_plane, y);
    return plane.Row(y);
  }

 private:
  // Twp options: 1) Self-owned plane
  Plane<pixel_type> plane;

  // 2) Not self-owned plane (to avoid unnecessary copying)
  Plane<pixel_type>* ref_plane;
  Rect ref_rect;
};

class Transform;

class Image {
 public:
  // image data, transforms can dramatically change the number of channels and
  // their semantics
  std::vector<Channel> channel;
  // transforms that have been applied (and that have to be undone)
  std::vector<Transform> transform;

  // image dimensions (channels may have different dimensions due to transforms)
  size_t w, h;
  int bitdepth;
  size_t nb_meta_channels;  // first few channels might contain palette(s)
  bool error;               // true if a fatal error occurred, false otherwise

  Image(size_t iw, size_t ih, int bitdepth, int nb_chans);
  Image();

  Image(const Image& other) = delete;
  Image& operator=(const Image& other) = delete;

  Image& operator=(Image&& other) noexcept;
  Image(Image&& other) noexcept = default;

  Image clone();

  void undo_transforms(const weighted::Header& wp_header,
                       jxl::ThreadPool* pool = nullptr);
};

}  // namespace jxl

#endif  // LIB_JXL_MODULAR_MODULAR_IMAGE_H_
