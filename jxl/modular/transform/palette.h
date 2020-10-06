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
#include "jxl/modular/encoding/context_predict.h"
#include "jxl/modular/image/image.h"

namespace jxl {

namespace palette_internal {

// Inclusive.
static constexpr int kMinImplicitPaletteIndex = -52;

// Exclusive.
static constexpr int kMaxImplicitPaletteIndexPastExplicitEntries = 1 << 9;

// The purpose of this function is solely to extend the interpretation of
// palette indices to implicit values. If index < nb_deltas, indicating that the
// result is a delta palette entry, it is the responsibility of the caller to
// treat it as such.
static pixel_type GetPaletteValue(const pixel_type *const palette, int index,
                                  const int c, const int palette_size,
                                  const int onerow, const int bit_depth) {
  if (index < 0) {
    static constexpr std::array<std::array<pixel_type, 3>, 52> kDeltaPalette = {
        {
            {0, 0, 0},      {-1, -1, -1},    {1, 1, 1},       {-2, -2, -2},
            {2, 2, 2},      {4, 4, 4},       {-4, -4, -4},    {-6, 0, 0},
            {0, -6, 0},     {0, 0, -6},      {6, 0, 0},       {0, 6, 0},
            {0, 0, 6},      {0, 0, -8},      {0, 0, 8},       {0, -8, 0},
            {0, -8, -8},    {0, -8, 8},      {0, 8, 0},       {0, 8, -8},
            {0, 8, 8},      {8, 8, 8},       {-8, 0, 0},      {-8, 0, -8},
            {-8, 0, 8},     {-8, -8, 0},     {-8, 8, 0},      {8, 0, 0},
            {8, 0, -8},     {8, 0, 8},       {8, -8, 0},      {8, 8, 0},
            {23, 23, 23},   {13, 13, 13},    {-24, -24, -24}, {-12, -12, -12},
            {-11, -11, -6}, {-20, -20, -20}, {16, 16, 16},    {-16, 16, 16},
            {16, -16, 16},  {-16, -16, 16},  {16, 16, -16},   {-16, 16, -16},
            {16, -16, -16}, {-16, -16, -16}, {0, 0, -24},     {0, -24, 0},
            {-24, 0, 0},    {0, 0, 24},      {0, 24, 0},      {24, 0, 0},
        }};
    index = ~index;
    if (index < kDeltaPalette.size() && c < kDeltaPalette[0].size()) {
      return kDeltaPalette[index][c];
    } else {
      return 0;
    }
  } else if (index >= palette_size) {
    index -= palette_size;
    return ((index >> (3 * c)) & 0b111) * ((1 << bit_depth) - 1) / 0b111;
  }

  return palette[c * onerow + index];
}

int QuantizeColorToImplicitPaletteIndex(const std::vector<pixel_type> &color,
                                        const int palette_size,
                                        const int bit_depth) {
  int index = 0;
  for (int c = 0; c < color.size(); c++) {
    const int quantized =
        (0b111 * color[c] + (1 << (bit_depth - 1))) / ((1 << bit_depth) - 1);
    JXL_DASSERT((quantized & 0b111) == quantized);
    index |= (quantized & 0b111) << (3 * c);
  }
  return palette_size + index;
}

}  // namespace palette_internal

static Status InvPalette(Image &input, uint32_t begin_c, uint32_t nb_colors,
                         uint32_t nb_deltas, Predictor predictor,
                         const weighted::Header &wp_header, ThreadPool *pool) {
  if (input.nb_meta_channels < 1) {
    return JXL_FAILURE("Error: Palette transform without palette.");
  }
  int nb = input.channel[0].h;
  uint32_t c0 = begin_c + 1;
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
  const pixel_type *JXL_RESTRICT p_palette = input.channel[0].Row(0);
  intptr_t onerow = input.channel[0].plane.PixelsPerRow();
  intptr_t onerow_image = input.channel[c0].plane.PixelsPerRow();
  const int bit_depth =
      CeilLog2Nonzero(static_cast<unsigned>(input.maxval - input.minval + 1));

  if (nb_deltas == 0 && predictor == Predictor::Zero) {
    if (nb == 1) {
      RunOnPool(
          pool, 0, h, ThreadPool::SkipInit(),
          [&](const int task, const int thread) {
            const size_t y = task;
            pixel_type *p = input.channel[c0].Row(y);
            for (size_t x = 0; x < w; x++) {
              const int index = p[x];
              p[x] = palette_internal::GetPaletteValue(
                  p_palette, std::max(0, index), /*c=*/0,
                  /*palette_size=*/palette.w, /*onerow=*/onerow,
                  /*bit_depth=*/bit_depth);
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
            for (int c = 0; c < nb; c++)
              p_out[c] = input.channel[c0 + c].Row(y);
            for (int x = 0; x < w; x++) {
              const int index = std::max(0, p_index[x]);
              for (int c = 0; c < nb; c++)
                p_out[c][x] = palette_internal::GetPaletteValue(
                    p_palette, std::max(0, index), /*c=*/c,
                    /*palette_size=*/palette.w, /*onerow=*/onerow,
                    /*bit_depth=*/bit_depth);
            }
          },
          "UndoPalette");
    }
  } else {
    // Parallelized per channel.
    ImageI indices = CopyImage(input.channel[c0].plane);
    if (predictor == Predictor::Weighted) {
      RunOnPool(
          pool, 0, nb, ThreadPool::SkipInit(),
          [&](size_t c, size_t _) {
            Channel &channel = input.channel[c0 + c];
            weighted::State wp_state(wp_header, channel.w, channel.h);
            for (size_t y = 0; y < channel.h; y++) {
              pixel_type *JXL_RESTRICT p = channel.Row(y);
              const pixel_type *JXL_RESTRICT idx = indices.Row(y);
              for (size_t x = 0; x < channel.w; x++) {
                int index = idx[x];
                pixel_type_w val = 0;
                const pixel_type palette_entry =
                    palette_internal::GetPaletteValue(
                        p_palette, index, /*c=*/c, /*palette_size=*/palette.w,
                        /*onerow=*/onerow, /*bit_depth=*/bit_depth);
                if (index < static_cast<int32_t>(nb_deltas)) {
                  PredictionResult pred =
                      PredictNoTreeWP(channel.w, p + x, onerow_image, x, y,
                                      predictor, &wp_state);
                  val = pred.guess + palette_entry;
                } else {
                  val = palette_entry;
                }
                p[x] = val;
                wp_state.UpdateErrors(p[x], x, y, channel.w);
              }
            }
          },
          "UndoDeltaPaletteWP");
    } else if (predictor == Predictor::Gradient) {
      // Gradient is the most common predictor for now. This special case gives
      // about 20% extra speed.
      RunOnPool(
          pool, 0, nb, ThreadPool::SkipInit(),
          [&](size_t c, size_t _) {
            Channel &channel = input.channel[c0 + c];
            for (size_t y = 0; y < channel.h; y++) {
              pixel_type *JXL_RESTRICT p = channel.Row(y);
              const pixel_type *JXL_RESTRICT idx = indices.Row(y);
              for (size_t x = 0; x < channel.w; x++) {
                int index = idx[x];
                pixel_type_w val = 0;
                const pixel_type palette_entry =
                    palette_internal::GetPaletteValue(
                        p_palette, index, /*c=*/c, /*palette_size=*/palette.w,
                        /*onerow=*/onerow, /*bit_depth=*/bit_depth);
                if (index < static_cast<int32_t>(nb_deltas)) {
                  pixel_type_w left =
                      x ? p[x - 1] : (y ? *(p + x - onerow_image) : 0);
                  pixel_type_w top = y ? *(p + x - onerow_image) : left;
                  pixel_type_w topleft =
                      x && y ? *(p + x - 1 - onerow_image) : left;
                  val = ClampedGradient(left, top, topleft) + palette_entry;
                } else {
                  val = palette_entry;
                }
                p[x] = val;
              }
            }
          },
          "UndoDeltaPaletteGradient");
    } else {
      RunOnPool(
          pool, 0, nb, ThreadPool::SkipInit(),
          [&](size_t c, size_t _) {
            Channel &channel = input.channel[c0 + c];
            for (size_t y = 0; y < channel.h; y++) {
              pixel_type *JXL_RESTRICT p = channel.Row(y);
              const pixel_type *JXL_RESTRICT idx = indices.Row(y);
              for (size_t x = 0; x < channel.w; x++) {
                int index = idx[x];
                pixel_type_w val = 0;
                const pixel_type palette_entry =
                    palette_internal::GetPaletteValue(
                        p_palette, index, /*c=*/c, /*palette_size=*/palette.w,
                        /*onerow=*/onerow, /*bit_depth=*/bit_depth);
                if (index < static_cast<int32_t>(nb_deltas)) {
                  PredictionResult pred = PredictNoTreeNoWP(
                      channel.w, p + x, onerow_image, x, y, predictor);
                  val = pred.guess + palette_entry;
                } else {
                  val = palette_entry;
                }
                p[x] = val;
              }
            }
          },
          "UndoDeltaPaletteNoWP");
    }
  }
  input.nb_channels += nb - 1;
  input.nb_meta_channels--;
  input.channel.erase(input.channel.begin(), input.channel.begin() + 1);
  return true;
}

static Status CheckPaletteParams(const Image &image, uint32_t begin_c,
                                 uint32_t end_c) {
  int c1 = begin_c;
  int c2 = end_c;
  // The range is including c1 and c2, so c2 may not be num_channels.
  if (c1 < 0 || c1 > image.channel.size() || c2 < 0 ||
      c2 >= image.channel.size() || c2 < c1) {
    return JXL_FAILURE("Invalid channel range");
  }

  return true;
}

static Status MetaPalette(Image &input, uint32_t begin_c, uint32_t end_c,
                          uint32_t nb_colors, uint32_t nb_deltas, bool lossy) {
  JXL_RETURN_IF_ERROR(CheckPaletteParams(input, begin_c, end_c));

  size_t nb = end_c - begin_c + 1;
  input.nb_meta_channels++;
  input.nb_channels -= nb - 1;
  input.channel.erase(input.channel.begin() + begin_c + 1,
                      input.channel.begin() + end_c + 1);
  Channel pch(nb_colors + nb_deltas, nb);
  pch.hshift = -1;
  input.channel.insert(input.channel.begin(), std::move(pch));
  return true;
}

static Status FwdPalette(Image &input, uint32_t begin_c, uint32_t end_c,
                         uint32_t &nb_colors, bool ordered, bool lossy,
                         Predictor &predictor,
                         const weighted::Header &wp_header) {
  JXL_RETURN_IF_ERROR(CheckPaletteParams(input, begin_c, end_c));
  uint32_t nb = end_c - begin_c + 1;

  size_t w = input.channel[begin_c].w;
  size_t h = input.channel[begin_c].h;
  for (int c = begin_c + 1; c <= end_c; c++) {
    if (input.channel[c].w != w) return false;
    if (input.channel[c].h != h) return false;
  }

  Image quantized_input;
  if (lossy) {
    quantized_input = Image(w, h, input.maxval, nb);
    for (int c = 0; c < nb; c++) {
      CopyImageTo(input.channel[begin_c + c].plane,
                  &quantized_input.channel[c].plane);
    }
  }

  JXL_DEBUG_V(
      7, "Trying to represent channels %i-%i using at most a %i-color palette.",
      begin_c, end_c, nb_colors);
  int nb_deltas = 0;
  bool delta_used = false;
  std::set<std::vector<pixel_type> >
      candidate_palette;  // ordered lexicographically
  std::vector<std::vector<pixel_type> > candidate_palette_imageorder;
  std::vector<pixel_type> color(nb);
  std::vector<const pixel_type *> p_in(nb);
  for (size_t y = 0; y < h; y++) {
    for (uint32_t c = 0; c < nb; c++) {
      p_in[c] = input.channel[begin_c + c].Row(y);
    }
    for (size_t x = 0; x < w; x++) {
      if (lossy && candidate_palette.size() >= nb_colors) break;
      for (uint32_t c = 0; c < nb; c++) {
        color[c] = p_in[c][x];
      }
      const bool new_color = candidate_palette.insert(color).second;
      if (new_color) {
        candidate_palette_imageorder.push_back(color);
      }
      if (static_cast<int>(candidate_palette.size()) > nb_colors) {
        return false;  // too many colors
      }
    }
  }
  nb_colors = candidate_palette.size();
  JXL_DEBUG_V(6, "Channels %i-%i can be represented using a %i-color palette.",
              begin_c, end_c, nb_colors);

  Channel pch(nb_colors, nb);
  pch.hshift = -1;
  int x = 0;
  pixel_type *JXL_RESTRICT p_palette = pch.Row(0);
  intptr_t onerow = pch.plane.PixelsPerRow();
  intptr_t onerow_image = input.channel[begin_c].plane.PixelsPerRow();
  const int bit_depth =
      CeilLog2Nonzero(static_cast<unsigned>(input.maxval - input.minval + 1));
  std::vector<pixel_type> lookup;
  int minval, maxval;
  input.channel[begin_c].compute_minmax(&minval, &maxval);
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
  std::vector<weighted::State> wp_states;
  for (int c = 0; c < nb; c++) {
    wp_states.emplace_back(wp_header, w, h);
  }
  std::vector<pixel_type *> p_quant(nb);
  for (size_t y = 0; y < h; y++) {
    for (int c = 0; c < nb; c++) {
      p_in[c] = input.channel[begin_c + c].Row(y);
      if (lossy) p_quant[c] = quantized_input.channel[c].Row(y);
    }
    pixel_type *JXL_RESTRICT p = input.channel[begin_c].Row(y);
    if (nb == 1 && !lossy) {
      for (size_t x = 0; x < w; x++) p[x] = lookup[p[x] - minval];
    } else {
      for (size_t x = 0; x < w; x++) {
        for (int c = 0; c < nb; c++) color[c] = p_in[c][x];
        int index = 0;
        if (!lossy) {
          // Exact search.
          for (; index < nb_colors; index++) {
            bool found = true;
            for (int c = 0; c < nb; c++)
              if (color[c] != p_palette[c * onerow + index]) {
                found = false;
                break;
              }
            if (found) break;
          }
        } else {
          pixel_type_w best_l2_squared =
              std::numeric_limits<pixel_type_w>::max();
          int best_index = 0;
          std::vector<pixel_type> best_val(nb, 0);
          std::vector<pixel_type> quantized_val(nb);
          std::vector<pixel_type_w> predictions(nb);
          for (int c = 0; c < nb; ++c) {
            predictions[c] = PredictNoTreeWP(w, p_quant[c] + x, onerow_image, x,
                                             y, predictor, &wp_states[c])
                                 .guess;
          }
          const auto TryIndex = [&](const int index) {
            pixel_type_w l2_squared = 0;
            for (int c = 0; c < nb; c++) {
              quantized_val[c] = palette_internal::GetPaletteValue(
                  p_palette, index, /*c=*/c, /*palette_size=*/nb_colors,
                  /*onerow=*/onerow, /*bit_depth=*/bit_depth);
              if (index < nb_deltas) {
                quantized_val[c] += predictions[c];
              }
              const pixel_type_w channel_difference =
                  static_cast<pixel_type_w>(color[c]) - quantized_val[c];
              l2_squared +=
                  channel_difference * channel_difference * (c <= 1 ? 2 : 1);
            }
            if (l2_squared < best_l2_squared) {
              best_l2_squared = l2_squared;
              best_index = index;
              std::copy(quantized_val.begin(), quantized_val.end(),
                        best_val.begin());
            }
          };
          for (index = palette_internal::kMinImplicitPaletteIndex;
               index < static_cast<int32_t>(nb_colors); index++) {
            TryIndex(index);
          }
          TryIndex(palette_internal::QuantizeColorToImplicitPaletteIndex(
              color, nb_colors, bit_depth));
          index = best_index;
          for (int c = 0; c < nb; ++c) {
            wp_states[c].UpdateErrors(best_val[c], x, y, w);
            p_quant[c][x] = best_val[c];
          }
        }
        p[x] = index;
        if (index < nb_deltas) {
          delta_used = true;
        }
      }
    }
  }
  if (!delta_used) {
    predictor = Predictor::Zero;
  }
  input.nb_meta_channels++;
  input.nb_channels -= nb - 1;
  input.channel.erase(input.channel.begin() + begin_c + 1,
                      input.channel.begin() + end_c + 1);
  input.channel.insert(input.channel.begin(), std::move(pch));
  return true;
}

}  // namespace jxl

#endif  // JXL_MODULAR_TRANSFORM_PALETTE_H_
