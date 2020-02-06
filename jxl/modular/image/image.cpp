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

#include "jxl/modular/image/image.h"

#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/modular/transform/transform.h"

namespace jxl {

#ifdef HAS_ENCODER
void Channel::actual_minmax(pixel_type *min, pixel_type *max) const {
  pixel_type realmin = LARGEST_VAL;
  pixel_type realmax = SMALLEST_VAL;
  for (size_t y = 0; y < h; y++) {
    const pixel_type *JXL_RESTRICT p = plane.Row(y);
    for (size_t x = 0; x < w; x++) {
      if (p[x] < realmin) realmin = p[x];
      if (p[x] > realmax) realmax = p[x];
    }
  }

  *min = realmin;
  *max = realmax;
}
#endif

void Image::undo_transforms(int keep, jxl::ThreadPool *pool) {
  if (keep == -2) return;
  while ((int)transform.size() > keep && transform.size() > 0) {
    Transform t = transform.back();
    JXL_DEBUG_V(4, "Undoing transform %s", t.Name());
    Status result = t.Apply(*this, true, pool);
    if (result == false) {
      JXL_FAILURE("Error while undoing transform %s.", t.Name());
      error = true;
      return;
    }
    JXL_DEBUG_V(8, "Undoing transform %s: done", t.Name());
    transform.pop_back();
  }
  if (!keep) {  // clamp the values to the valid range (lossy
                // compression can produce values outside the range)
    for (size_t i = 0; i < channel.size(); i++) {
      for (size_t y = 0; y < channel[i].h; y++) {
        pixel_type *JXL_RESTRICT p = channel[i].plane.Row(y);
        for (size_t x = 0; x < channel[i].w; x++, p++) {
          *p = Clamp(*p, minval, maxval);
        }
      }
    }
  }
}

#ifdef HAS_ENCODER
bool Image::do_transform(const Transform &tr) {
  Transform t = tr;
  bool did_it = t.Apply(*this, false);
  if (did_it) transform.push_back(t);
  return did_it;
}
#endif

}  // namespace jxl
