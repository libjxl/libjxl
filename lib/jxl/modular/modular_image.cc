// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/modular_image.h"

#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

void Image::undo_transforms(const weighted::Header &wp_header, int keep,
                            jxl::ThreadPool *pool) {
  if (keep == -2) return;
  while ((int)transform.size() > keep && transform.size() > 0) {
    Transform t = transform.back();
    JXL_DEBUG_V(4, "Undoing transform");
    Status result = t.Inverse(*this, wp_header, pool);
    if (result == false) {
      JXL_NOTIFY_ERROR("Error while undoing transform.");
      error = true;
      return;
    }
    JXL_DEBUG_V(8, "Undoing transform: done");
    transform.pop_back();
  }
  if (!keep && bitdepth < 32) {
    // clamp the values to the valid range (lossy compression can produce values
    // outside the range)
    pixel_type maxval = (1u << bitdepth) - 1;
    for (size_t i = 0; i < channel.size(); i++) {
      for (size_t y = 0; y < channel[i].h; y++) {
        pixel_type *JXL_RESTRICT p = channel[i].plane.Row(y);
        for (size_t x = 0; x < channel[i].w; x++, p++) {
          *p = Clamp1(*p, 0, maxval);
        }
      }
    }
  }
}

Image::Image(size_t iw, size_t ih, int bd, int nb_chans)
    : w(iw), h(ih), bitdepth(bd), nb_meta_channels(0), error(false) {
  for (int i = 0; i < nb_chans; i++) channel.emplace_back(Channel(iw, ih));
}

Image::Image() : w(0), h(0), bitdepth(8), nb_meta_channels(0), error(true) {}

Image &Image::operator=(Image &&other) noexcept {
  w = other.w;
  h = other.h;
  bitdepth = other.bitdepth;
  nb_meta_channels = other.nb_meta_channels;
  error = other.error;
  channel = std::move(other.channel);
  transform = std::move(other.transform);
  return *this;
}

}  // namespace jxl
