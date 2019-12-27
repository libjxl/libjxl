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

#include "jxl/modular/image/image.h"

namespace jxl {

#ifdef HAS_ENCODER
void delta_quantize(int max_error, pixel_type &d) {
  int a = (d < 0 ? -d : d);
  a = (a + (max_error / 2)) / max_error * max_error;
  d = (d < 0 ? -a : a);
}

static bool fwd_near_lossless(Image &input, std::vector<int> &parameters) {
  JXL_DASSERT(parameters.size() == 3);
  int begin_c = input.nb_meta_channels + parameters[0];
  int end_c = input.nb_meta_channels + parameters[1];
  int max_delta_error = parameters[2];

  input.recompute_minmax();
  JXL_DEBUG_V(8, "Applying loss on channels %i-%i with max delta=%i.", begin_c,
              end_c, max_delta_error);
  uint64_t total_error = 0;
  for (int c = begin_c; c <= end_c; c++) {
    if (c >= input.channel.size()) return false;
    int w = input.channel[c].w;
    int h = input.channel[c].h;
    intptr_t onerow = input.channel[c].plane.PixelsPerRow();

    Channel out(w, h, 0, 1);
    for (int y = 0; y < h; y++) {
      pixel_type *JXL_RESTRICT p_in = input.channel[c].Row(y);
      pixel_type *JXL_RESTRICT p_out = out.Row(y);
      for (int x = 0; x < w; x++) {
        // assuming the default predictor
        pixel_type left = (x ? p_out[x - 1] : 0);
        pixel_type top = (y ? p_out[x - onerow] : left);
        pixel_type prediction = (left + top) / 2;
        pixel_type delta = p_in[x] - prediction;
        delta_quantize(max_delta_error, delta);
        pixel_type reconstructed =
            CLAMP(prediction + delta, input.channel[c].minval,
                  input.channel[c].maxval);
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
#endif

static bool near_lossless(Image &input, bool inverse,
                          std::vector<int> &parameters) {
  JXL_DASSERT(inverse == false);
#ifdef HAS_ENCODER
  return fwd_near_lossless(input, parameters);
#else
  return false;
#endif
}

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_NEAR_LOSSLESS_H_
