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

#ifndef JXL_MODULAR_TRANSFORM_QUANTIZE_H_
#define JXL_MODULAR_TRANSFORM_QUANTIZE_H_

#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/modular/image/image.h"

namespace jxl {

pixel_type rdiv(pixel_type x, int q) {
  if (x < 0) {
    return -((-x + q / 2) / q);
  } else {
    return (x + q / 2) / q;
  }
}

bool InvQuantize(Image &input, const TransformParams &parameters,
                 ThreadPool *pool) {
  size_t c = input.nb_meta_channels;
  pixel_type *JXL_RESTRICT qs = input.channel[0].Row(0);
  size_t qx = 0;
  for (; c < input.channel.size(); c++) {
    Channel &ch = input.channel[c];
    if (ch.is_empty()) continue;
    int q = qs[qx] + 1;
    qx++;
    if (q == 1) continue;
    JXL_DEBUG_V(3, "De-quantizing channel %zu with quantization constant %i", c,
                q);
    RunOnPool(
        pool, 0, ch.h, ThreadPool::SkipInit(),
        [&](const int task, const int thread) {
          const size_t y = task;
          pixel_type *JXL_RESTRICT p = ch.Row(y);
          for (size_t x = 0; x < ch.w; x++) {
            p[x] = SaturatingMul<pixel_type>(p[x], q);
          }
        },
        "ModularDequant");
  }
  input.nb_meta_channels--;
  input.channel.erase(input.channel.begin(), input.channel.begin() + 1);
  return true;
}

static Status MetaQuantize(Image &input) {
  Channel qs(input.channel.size() - input.nb_meta_channels, 1);
  qs.hshift = -1;
  input.channel.insert(input.channel.begin(), std::move(qs));
  input.nb_meta_channels++;
  return true;
}

Status FwdQuantize(Image &input, const TransformParams &parameters) {
  JXL_RETURN_IF_ERROR(MetaQuantize(input));
  pixel_type *JXL_RESTRICT qs = input.channel[0].Row(0);
  size_t qx = 0;
  for (size_t c = input.nb_meta_channels; c < input.channel.size(); c++) {
    Channel &ch = input.channel[c];
    if (ch.is_empty()) continue;
    int q = (c - input.nb_meta_channels < parameters.size()
                 ? parameters[c - input.nb_meta_channels]
                 : parameters.back());
    if (q < 1) q = 1;
    JXL_DEBUG_V(3, "Quantizing channel %zu with quantization constant %i", c,
                q);
    for (size_t y = 0; y < ch.h; y++) {
      pixel_type *JXL_RESTRICT p = ch.Row(y);
      for (size_t x = 0; x < ch.w; x++) {
        //            p[x] /= q;
        p[x] = rdiv(p[x], q);
      }
    }
    qs[qx] = q - 1;
    qx++;
  }
  return true;
}

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_QUANTIZE_H_
