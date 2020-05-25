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

#ifndef JXL_MODULAR_TRANSFORM_SUBTRACTGREEN_H_
#define JXL_MODULAR_TRANSFORM_SUBTRACTGREEN_H_

#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/modular/image/image.h"

namespace jxl {

Status InvSubtractGreen(Image& input, const TransformParams& parameters) {
  size_t m = input.nb_meta_channels;
  int nb_channels = input.nb_channels;
  if (nb_channels < 3) {
    return JXL_FAILURE(
        "Invalid number of channels to apply inverse subtract_green.");
  }
  size_t w = input.channel[m + 0].w;
  size_t h = input.channel[m + 0].h;
  if (input.channel[m + 1].w < w || input.channel[m + 1].h < h ||
      input.channel[m + 2].w < w || input.channel[m + 2].h < h) {
    return JXL_FAILURE(
        "Invalid channel dimensions to apply inverse subtract_green (maybe "
        "chroma is subsampled?).");
  }
  // Permutation: 0=RGB, 1=GBR, 2=BRG, 3=RBG, 4=GRB, 5=BGR
  int permutation = 1;
  // Second: 0=nop, 1=SubtractFirst, 2=SubtractAvgFirstThird
  int second = 1;
  // Third: 0=nop, 1=SubtractFirst
  int third = 1;
  if (parameters.size() == 1) {
    int custom = parameters[0];
    permutation = custom / 6;
    second = (custom % 6) >> 1;
    third = (custom & 1);
  }
  for (size_t y = 0; y < h; y++) {
    const pixel_type* in0 = input.channel[m].Row(y);
    const pixel_type* in1 = input.channel[m + 1].Row(y);
    const pixel_type* in2 = input.channel[m + 2].Row(y);
    pixel_type* out0 = input.channel[m + (permutation % 3)].Row(y);
    pixel_type* out1 =
        input.channel[m + ((permutation + 1 + permutation / 3) % 3)].Row(y);
    pixel_type* out2 =
        input.channel[m + ((permutation + 2 - permutation / 3) % 3)].Row(y);
    for (size_t x = 0; x < w; x++) {
      pixel_type_w First = in0[x];
      pixel_type_w Second = in1[x];
      pixel_type_w Third = in2[x];
      if (third) Third = Third + First;
      if (second == 1) {
        Second = Second + First;
      } else if (second == 2) {
        Second = Second + ((First + Third) >> 1);
      }
      out0[x] = ClampToRange<pixel_type>(First);
      out1[x] = ClampToRange<pixel_type>(Second);
      out2[x] = ClampToRange<pixel_type>(Third);
    }
  }
  return true;
}

Status FwdSubtractGreen(Image& input, const TransformParams& parameters) {
  size_t nb_channels = input.nb_channels;
  if (nb_channels < 3) {
    return false;
  }
  // Permutation: 0=RGB, 1=GBR, 2=BRG, 3=RBG, 4=GRB, 5=BGR
  int permutation = 1;
  // Second: 0=nop, 1=SubtractFirst, 2=SubtractAvgFirstThird
  int second = 1;
  // Third: 0=nop, 1=SubtractFirst
  int third = 1;
  if (parameters.size() == 1) {
    int custom = parameters[0];
    permutation = custom / 6;
    second = (custom % 6) >> 1;
    third = (custom & 1);
  }
  size_t m = input.nb_meta_channels;
  size_t w = input.channel[m + 0].w;
  size_t h = input.channel[m + 0].h;
  if (input.channel[m + 1].w < w || input.channel[m + 1].h < h ||
      input.channel[m + 2].w < w || input.channel[m + 2].h < h) {
    return JXL_FAILURE("Invalid channel dimensions to apply subtract_green.");
  }
  for (size_t y = 0; y < h; y++) {
    const pixel_type* in0 = input.channel[m + (permutation % 3)].Row(y);
    const pixel_type* in1 =
        input.channel[m + ((permutation + 1 + permutation / 3) % 3)].Row(y);
    const pixel_type* in2 =
        input.channel[m + ((permutation + 2 - permutation / 3) % 3)].Row(y);
    pixel_type* out0 = input.channel[m].Row(y);
    pixel_type* out1 = input.channel[m + 1].Row(y);
    pixel_type* out2 = input.channel[m + 2].Row(y);
    for (size_t x = 0; x < w; x++) {
      pixel_type First = in0[x];
      pixel_type Second = in1[x];
      pixel_type Third = in2[x];
      if (second == 1) {
        Second = Second - First;
      } else if (second == 2) {
        Second = Second - ((First + Third) >> 1);
      }
      if (third) Third = Third - First;
      out0[x] = First;
      out1[x] = Second;
      out2[x] = Third;
    }
  }
  return true;
}

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_SUBTRACTGREEN_H_
