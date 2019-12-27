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

#ifndef JXL_MODULAR_IMAGE_IMAGE_H_
#define JXL_MODULAR_IMAGE_IMAGE_H_

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <utility>
#include <vector>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/image.h"
#include "jxl/modular/config.h"
#include "jxl/modular/util.h"

namespace jxl {

#ifdef HDR

typedef int32_t pixel_type;  // can use int16_t if it's only for 8-bit images.
                             // Need some wiggle room for YCoCg / Squeeze etc

// largest possible pixel value (2147483647)
#define LARGEST_VAL 0x7FFFFFFF

// smallest possible pixel value (-2147483647)
#define SMALLEST_VAL 0x80000001

#else

typedef int16_t
    pixel_type;  // enough for up to 14-bit with YCoCg/Squeeze (I think)
#define LARGEST_VAL 0x7FFF
#define SMALLEST_VAL 0x8001

#endif

class Channel {
 public:
  jxl::Plane<pixel_type> plane;
  size_t w, h;
  pixel_type minval, maxval;  // range
  mutable pixel_type
      zero;  // should be zero if zero is a valid value for this channel; the
             // valid value closest to zero otherwise
  int hshift, vshift;  // w ~= image.w >> hshift;  h ~= image.h >> vshift
  int hcshift,
      vcshift;  // cumulative, i.e. when decoding up to this point, we have data
                // available with these shifts (for this component)
  Channel(size_t iw, size_t ih, pixel_type iminval, pixel_type imaxval,
          int hsh = 0, int vsh = 0, int hcsh = 0, int vcsh = 0)
      : plane(iw, ih),
        w(iw),
        h(ih),
        minval(iminval),
        maxval(imaxval),
        hshift(hsh),
        vshift(vsh),
        hcshift(hcsh),
        vcshift(vcsh) {
    setzero();
  }
  Channel()
      : plane(0, 0),
        w(0),
        h(0),
        minval(0),
        maxval(0),
        zero(0),
        hshift(0),
        vshift(0),
        hcshift(0),
        vcshift(0) {}

  Channel(const Channel& other) = delete;
  Channel& operator=(const Channel& other) = delete;

  // Move assignment
  Channel& operator=(Channel&& other) noexcept {
    w = other.w;
    h = other.h;
    minval = other.minval;
    maxval = other.maxval;
    hshift = other.hshift;
    vshift = other.vshift;
    hcshift = other.hcshift;
    vcshift = other.vcshift;
    plane = std::move(other.plane);
    return *this;
  }

  // Move constructor
  Channel(Channel&& other) noexcept = default;

  void setzero() const {
    if (minval > 0)
      zero = minval;
    else if (maxval < 0)
      zero = maxval;
    else
      zero = 0;
  }

  void resize() {
    if (plane.xsize() == w && plane.ysize() == h) return;
    jxl::Plane<pixel_type> resizedplane(w, h);
    if (plane.xsize() || plane.ysize() || zero) {
      // copy pixels over from old plane to new plane
      size_t y = 0;
      for (; y < plane.ysize() && y < h; y++) {
        const pixel_type* JXL_RESTRICT p = plane.Row(y);
        pixel_type* JXL_RESTRICT rp = resizedplane.Row(y);
        size_t x = y;
        for (; x < plane.xsize() && x < w; x++) rp[x] = p[x];
        for (; x < w; x++) rp[x] = zero;
      }
      for (; y < h; y++) {
        pixel_type* JXL_RESTRICT p = resizedplane.Row(y);
        for (size_t x = 0; x < w; x++) p[x] = zero;
      }
    } else if (w && h) {
      size_t ppr = resizedplane.bytes_per_row();
      memset(resizedplane.bytes(), 0, ppr * h);
    }
    plane = std::move(resizedplane);
  }
  void resize(int nw, int nh) {
    w = nw;
    h = nh;
    resize();
  }
  bool is_empty() const { return (plane.ysize() == 0); }

  JXL_INLINE pixel_type* Row(const size_t y) { return plane.Row(y); }
  JXL_INLINE const pixel_type* Row(const size_t y) const {
    return plane.Row(y);
  }
  void actual_minmax(pixel_type* min, pixel_type* max) const;
};

class Transform;

class Image {
 public:
  std::vector<Channel>
      channel;  // image data, transforms can dramatically change the number of
                // channels and their semantics
  std::vector<Transform>
      transform;  // keeps track of the transforms that have been applied (and
                  // that have to be undone when rendering the image)

  size_t w, h;  // actual dimensions of the image (channels may have different
                // dimensions due to transforms like chroma subsampling and DCT)
  int minval, maxval;  // actual (largest) range of the channels (actual ranges
                       // might be different due to transforms; after undoing
                       // transforms, might still be different due to lossy)
  size_t nb_channels;  // actual number of distinct channels (after undoing all
                       // transforms except Palette; can be different from
                       // channel.size())
  size_t real_nb_channels;  // real number of channels (after undoing all
                            // transforms)
  size_t nb_meta_channels;  // first few channels might contain things like
                            // palettes or compaction data that are not yet real
                            // image data
  bool error;               // true if a fatal error occurred, false otherwise

  Image(size_t iw, size_t ih, int maxval, int nb_chans)
      : w(iw),
        h(ih),
        minval(0),
        maxval(maxval),
        nb_channels(nb_chans),
        real_nb_channels(nb_chans),
        nb_meta_channels(0),
        error(false) {
    for (int i = 0; i < nb_chans; i++)
      channel.emplace_back(Channel(iw, ih, 0, maxval));
  }

  Image()
      : w(0),
        h(0),
        minval(0),
        maxval(255),
        nb_channels(0),
        real_nb_channels(0),
        nb_meta_channels(0),
        error(true) {}

  Image(const Image& other) = delete;
  Image& operator=(const Image& other) = delete;

  Image& operator=(Image&& other) noexcept {
    w = other.w;
    h = other.h;
    minval = other.minval;
    maxval = other.maxval;
    nb_channels = other.nb_channels;
    real_nb_channels = other.real_nb_channels;
    nb_meta_channels = other.nb_meta_channels;
    error = other.error;
    channel = std::move(other.channel);
    transform = std::move(other.transform);
    return *this;
  }
  Image(Image&& other) noexcept = default;

  bool do_transform(const Transform& t);
  // undo all except the first 'keep' transforms
  void undo_transforms(int keep = 0, jxl::ThreadPool* pool = nullptr);
  void recompute_minmax() {
    for (auto& ch : channel) ch.actual_minmax(&ch.minval, &ch.maxval);
  }
};

}  // namespace jxl

#include "jxl/modular/transform/transform.h"

#endif  // JXL_MODULAR_IMAGE_IMAGE_H_
