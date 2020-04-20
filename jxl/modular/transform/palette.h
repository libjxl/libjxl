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

#ifndef JXL_MODULAR_TRANSFORM_PALETTE_H_
#define JXL_MODULAR_TRANSFORM_PALETTE_H_

#include <set>

#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/modular/image/image.h"

namespace jxl {

static Status inv_palette(Image &input, const TransformParams &parameters,
                          ThreadPool *pool) {
  if (input.nb_meta_channels < 1) {
    return JXL_FAILURE("Error: Palette transform without palette.");
  }
  if (parameters.size() != 3) {
    return JXL_FAILURE("Error: Palette transform with incorrect parameters.");
  }
  int nb = input.channel[0].h;
  uint32_t c0 = static_cast<uint32_t>(parameters[0] + 1);
  if (c0 >= input.channel.size()) {
    return JXL_FAILURE("Channel is out of range.");
  }
  size_t w = input.channel[c0].w;
  size_t h = input.channel[c0].h;
  // might be false in case of lossy
  // JXL_DASSERT(input.channel[c0].minval == 0);
  // JXL_DASSERT(input.channel[c0].maxval == palette.w-1);
  for (int i = 1; i < nb; i++) {
    input.channel.insert(input.channel.begin() + c0 + 1, Channel(w, h));
  }
  const Channel &palette = input.channel[0];
  int zero = 0;
  if (nb == 1) {
    int min, max;
    input.channel[c0].compute_trivial(&min, &max);
    zero = -min;
  }
  const pixel_type *JXL_RESTRICT p_palette = input.channel[0].Row(0);
  intptr_t onerow = input.channel[0].plane.PixelsPerRow();

  if (nb == 1) {
    RunOnPool(
        pool, 0, h, ThreadPool::SkipInit(),
        [&](const int task, const int thread) {
          const size_t y = task;
          pixel_type *p = input.channel[c0].Row(y);
          for (size_t x = 0; x < w; x++) {
            int index = Clamp(p[x] + zero, 0, static_cast<int>(palette.w) - 1);
            p[x] = p_palette[index];
          }
        },
        "UndoChannelPalette");
  } else {
    RunOnPool(
        pool, 0, h, ThreadPool::SkipInit(),
        [&](const int task, const int thread) {
          const size_t y = task;
          std::vector<pixel_type *> p_out(nb);
          const pixel_type *p_index = input.channel[c0].Row(y);
          for (int c = 0; c < nb; c++) p_out[c] = input.channel[c0 + c].Row(y);
          for (int x = 0; x < w; x++) {
            int index =
                Clamp(p_index[x] + zero, 0, static_cast<int>(palette.w) - 1);
            for (int c = 0; c < nb; c++)
              p_out[c][x] = p_palette[c * onerow + index];
          }
        },
        "UndoPalette");
  }
  input.nb_channels += nb - 1;
  input.nb_meta_channels--;
  input.channel.erase(input.channel.begin(), input.channel.begin() + 1);
  return true;
}

static Status CheckPaletteParams(const Image &image,
                                 const TransformParams &parameters) {
  if (parameters.size() != 3) {
    return JXL_FAILURE("Error: Palette transform with incorrect parameters.");
  }
  int c1 = parameters[0];
  int c2 = parameters[1];
  // The range is including c1 and c2, so c2 may not be num_channels.
  if (c1 < 0 || c1 > image.channel.size() || c2 < 0 ||
      c2 >= image.channel.size() || c2 < c1) {
    return JXL_FAILURE("Invalid channel range");
  }

  return true;
}

static Status meta_palette(Image &input, const TransformParams &parameters) {
  JXL_RETURN_IF_ERROR(CheckPaletteParams(input, parameters));

  uint32_t begin_c = parameters[0];
  uint32_t end_c = parameters[1];
  uint32_t nb = end_c - begin_c + 1;
  int nb_colors = parameters[2];
  if (nb_colors <= 0) {
    return JXL_FAILURE("Invalid number of nb_colors");
  }
  input.nb_meta_channels++;
  input.nb_channels -= nb - 1;
  input.channel.erase(input.channel.begin() + begin_c + 1,
                      input.channel.begin() + end_c + 1);
  Channel pch(nb_colors, nb);
  pch.hshift = -1;
  input.channel.insert(input.channel.begin(), std::move(pch));
  return true;
}

#ifdef HAS_ENCODER
static Status fwd_palette(Image &input, TransformParams &parameters) {
  JXL_RETURN_IF_ERROR(CheckPaletteParams(input, parameters));
  uint32_t begin_c = parameters[0];
  uint32_t end_c = parameters[1];
  int &nb_colors = parameters[2];
  bool ordered = true;
  if (nb_colors < 0) {
    nb_colors = -nb_colors;
    ordered = false;
  }
  uint32_t nb = end_c - begin_c + 1;

  size_t w = input.channel[begin_c].w;
  size_t h = input.channel[begin_c].h;
  for (int c = begin_c + 1; c <= end_c; c++) {
    if (input.channel[c].w != w) return false;
    if (input.channel[c].h != h) return false;
  }
  JXL_DEBUG_V(
      7, "Trying to represent channels %i-%i using at most a %i-color palette.",
      begin_c, end_c, nb_colors);
  std::set<std::vector<pixel_type> >
      candidate_palette;  // ordered lexicographically
  std::vector<std::vector<pixel_type> > candidate_palette_imageorder;
  std::vector<pixel_type> color(nb);
  std::vector<const pixel_type *> p_in(nb);
  size_t count = 0;
  for (size_t y = 0; y < h; y++) {
    for (uint32_t c = 0; c < nb; c++) {
      p_in[c] = input.channel[begin_c + c].Row(y);
    }
    for (size_t x = 0; x < w; x++) {
      for (uint32_t c = 0; c < nb; c++) {
        color[c] = p_in[c][x];
      }
      candidate_palette.insert(color);
      if (candidate_palette.size() > count) {
        count++;
        candidate_palette_imageorder.push_back(color);
      }
      if (static_cast<int>(candidate_palette.size()) > nb_colors) {
        return false;  // too many colors
      }
    }
  }
  nb_colors = candidate_palette.size();
  if (nb_colors == 1) return false;  // don't need palette for this
  JXL_DEBUG_V(6, "Channels %i-%i can be represented using a %i-color palette.",
              begin_c, end_c, nb_colors);

  Channel pch(nb_colors, nb);
  pch.hshift = -1;
  int x = 0;
  pixel_type *JXL_RESTRICT p_palette = pch.Row(0);
  intptr_t onerow = pch.plane.PixelsPerRow();
  int zero = 0;
  std::vector<pixel_type> lookup;
  int minval, maxval;
  input.channel[begin_c].compute_trivial(&minval, &maxval);
  if (nb == 1) {
    lookup.resize(maxval - minval + 1);
  }
  if (ordered) {
    JXL_DEBUG_V(7, "Palette of %i colors, using lexicographic order",
                nb_colors);
    for (auto pcol : candidate_palette) {
      JXL_DEBUG_V(9, "  Color %i :  ", x);
      for (int i = 0; i < nb; i++) {
        p_palette[i * onerow + x] = pcol[i];
      }
      if (nb == 1) lookup[pcol[0] - minval] = x;
      if (nb == 1 && pcol[0] <= 0) zero = x;
      for (int i = 0; i < nb; i++) {
        JXL_DEBUG_V(9, "%i ", pcol[i]);
      }
      x++;
    }
  } else {
    JXL_DEBUG_V(7, "Palette of %i colors, using image order", nb_colors);
    for (auto pcol : candidate_palette_imageorder) {
      JXL_DEBUG_V(9, "  Color %i :  ", x);
      for (int i = 0; i < nb; i++) p_palette[i * onerow + x] = pcol[i];
      if (nb == 1) lookup[pcol[0] - minval] = x;
      for (int i = 0; i < nb; i++) JXL_DEBUG_V(9, "%i ", pcol[i]);
      x++;
    }
  }
  for (size_t y = 0; y < h; y++) {
    for (int c = 0; c < nb; c++) p_in[c] = input.channel[begin_c + c].Row(y);
    pixel_type *JXL_RESTRICT p = input.channel[begin_c].Row(y);
    if (nb == 1) {
      for (size_t x = 0; x < w; x++) p[x] = lookup[p[x] - minval] - zero;
    } else {
      for (size_t x = 0; x < w; x++) {
        for (int c = 0; c < nb; c++) color[c] = p_in[c][x];
        int index = 0;
        for (; index < nb_colors; index++) {
          bool found = true;
          for (int c = 0; c < nb; c++)
            if (color[c] != p_palette[c * onerow + index]) {
              found = false;
              break;
            }
          if (found) break;
        }
        p[x] = index - zero;
      }
    }
  }
  input.nb_meta_channels++;
  input.nb_channels -= nb - 1;
  input.channel.erase(input.channel.begin() + begin_c + 1,
                      input.channel.begin() + end_c + 1);
  input.channel.insert(input.channel.begin(), std::move(pch));
  return true;
}
#endif

static Status palette(Image &input, bool inverse, TransformParams &parameters,
                      ThreadPool *pool) {
  if (inverse) {
    return inv_palette(input, parameters, pool);
  } else {
#ifdef HAS_ENCODER
    return fwd_palette(input, parameters);
#else
    return false;
#endif
  }
}

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_PALETTE_H_
