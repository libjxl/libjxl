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

#ifndef JXL_MODULAR_TRANSFORM_YCOCG_H_
#define JXL_MODULAR_TRANSFORM_YCOCG_H_

#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/modular/config.h"
#include "jxl/modular/image/image.h"

namespace jxl {

bool inv_YCoCg(Image& input, ThreadPool* pool) {
  size_t m = input.nb_meta_channels;
  int nb_channels = input.nb_channels;
  if (nb_channels < 3) {
    return JXL_FAILURE("Invalid number of channels to apply inverse YCoCg.");
  }
  size_t w = input.channel[m + 0].w;
  size_t h = input.channel[m + 0].h;
  if (input.channel[m + 1].w < w || input.channel[m + 1].h < h ||
      input.channel[m + 2].w < w || input.channel[m + 2].h < h) {
    return JXL_FAILURE(
        "Invalid channel dimensions to apply inverse YCoCg (maybe chroma is "
        "subsampled?).\n");
  }
  RunOnPool(
      pool, 0, h, ThreadPool::SkipInit(),
      [&](const int task, const int thread) {
        const size_t y = task;
        pixel_type* JXL_RESTRICT p0 = input.channel[m].Row(y);
        pixel_type* JXL_RESTRICT p1 = input.channel[m + 1].Row(y);
        pixel_type* JXL_RESTRICT p2 = input.channel[m + 2].Row(y);
        for (size_t x = 0; x < w; x++) {
          pixel_type Y = p0[x];
          pixel_type Co = p1[x];
          pixel_type Cg = p2[x];
          pixel_type tmp = Y - (Cg >> 1);
          pixel_type G = Cg + tmp;
          pixel_type B = tmp - (Co >> 1);
          pixel_type R = B + Co;
          p0[x] = R;
          p1[x] = G;
          p2[x] = B;
        }
      },
      "InvYCoCg");
  return true;
}

#ifdef HAS_ENCODER
bool fwd_YCoCg(Image& input) {
  int m = input.nb_meta_channels;
  int nb_channels = input.nb_channels;
  if (nb_channels < 3) {
    // return JXL_FAILURE("Invalid number of channels to apply YCoCg.");
    return false;
  }
  size_t w = input.channel[m + 0].w;
  size_t h = input.channel[m + 0].h;
  if (input.channel[m + 1].w < w || input.channel[m + 1].h < h ||
      input.channel[m + 2].w < w || input.channel[m + 2].h < h) {
    return JXL_FAILURE("Invalid channel dimensions to apply YCoCg.");
  }
  for (size_t y = 0; y < h; y++) {
    pixel_type* JXL_RESTRICT p0 = input.channel[m].Row(y);
    pixel_type* JXL_RESTRICT p1 = input.channel[m + 1].Row(y);
    pixel_type* JXL_RESTRICT p2 = input.channel[m + 2].Row(y);
    for (size_t x = 0; x < w; x++) {
      pixel_type R = p0[x];
      pixel_type G = p1[x];
      pixel_type B = p2[x];
      p1[x] = R - B;
      pixel_type tmp = B + (p1[x] >> 1);
      p2[x] = G - tmp;
      p0[x] = tmp + (p2[x] >> 1);
    }
  }
  return true;
}
#endif

bool YCoCg(Image& input, bool inverse, ThreadPool* pool) {
  if (inverse) return inv_YCoCg(input, pool);
#ifdef HAS_ENCODER
  else
    return fwd_YCoCg(input);
#else
  return false;
#endif
}

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_YCOCG_H_
