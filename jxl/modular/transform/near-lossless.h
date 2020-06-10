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

#ifndef JXL_MODULAR_TRANSFORM_NEAR_LOSSLESS_H_
#define JXL_MODULAR_TRANSFORM_NEAR_LOSSLESS_H_

// Very simple lossy preprocessing step.
// Quantizes the prediction residual (so the entropy coder has an easier job)
// Obviously there's room for encoder improvement here
// The decoder doesn't need to know about this step

#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/modular/image/image.h"

namespace jxl {

void DeltaQuantize(int max_error, pixel_type& d) {
  int a = (d < 0 ? -d : d);
  a = (a + (max_error / 2)) / max_error * max_error;
  d = (d < 0 ? -a : a);
}

static Status FwdNearLossless(Image& input, size_t begin_c, size_t end_c,
                              int max_delta_error) {
  if (begin_c < input.nb_meta_channels ||
      begin_c > static_cast<int>(input.channel.size()) ||
      end_c < input.nb_meta_channels ||
      end_c >= static_cast<int>(input.channel.size()) || end_c < begin_c) {
    return JXL_FAILURE("Invalid channel range");
  }

  JXL_DEBUG_V(8, "Applying loss on channels %zu-%zu with max delta=%i.",
              begin_c, end_c, max_delta_error);
  uint64_t total_error = 0;
  for (int c = begin_c; c <= end_c; c++) {
    size_t w = input.channel[c].w;
    size_t h = input.channel[c].h;

    Channel out(w, h);
    for (size_t y = 0; y < h; y++) {
      pixel_type* JXL_RESTRICT p_in = input.channel[c].Row(y);
      pixel_type* JXL_RESTRICT p_out = out.Row(y);
      pixel_type* JXL_RESTRICT prev_out = y ? out.Row(y - 1) : nullptr;
      for (size_t x = 0; x < w; x++) {
        // assuming the default predictor
        pixel_type left = (x ? p_out[x - 1] : 0);
        pixel_type top = (y ? prev_out[x] : left);
        pixel_type prediction = (left + top) / 2;
        pixel_type delta = p_in[x] - prediction;
        DeltaQuantize(max_delta_error, delta);
        pixel_type reconstructed = prediction + delta;
        int e = p_in[x] - reconstructed;
        total_error += abs(e);
        p_out[x] = reconstructed;
      }
    }
    input.channel[c] = std::move(out);
    JXL_DEBUG_V(9, "  Avg error: %f", total_error * 1.0 / (w * h));
  }
  return false;  // don't signal this 'transform' in the bitstream, there is no
                 // inverse transform to be done
}

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_NEAR_LOSSLESS_H_
