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

#ifndef JXL_MODULAR_ENCODING_CONTEXT_PREDICT_H_
#define JXL_MODULAR_ENCODING_CONTEXT_PREDICT_H_

#include "jxl/modular/encoding/encoding.h"
#include "jxl/modular/image/image.h"
#include "jxl/modular/ma/compound.h"
#include "jxl/modular/ma/util.h"

namespace jxl {

enum class Predictor : uint32_t {
  Zero = 0,
  Left = 1,
  Top = 2,
  Average = 3,
  Select = 4,
  Gradient = 5,
  Variable = 6,
  Weighted = 7,
  Best = 8  // Best of Gradient and Weighted (encoder only)
};

// Something that looks like absolute value. Just returning the absolute value
// works. Wrapped into a function to make it easier to experiment with
// alternatives.
inline pixel_type UNSIGNED_VAL(pixel_type x) { return abs(x); }

// Something that preserves the sign but potentially reduces the range
// Just returning the value works.
// Wrapped into a function to make it easier to experiment with alternatives
inline pixel_type SIGNED_VAL(pixel_type x) { return x; }

inline void init_properties(Ranges &pr, const Image &image, int beginc,
                            int endc, modular_options &options) {
  int offset = 0;
  for (int j = beginc - 1; j >= 0 && offset < options.max_properties; j--) {
    if (image.channel[j].minval == image.channel[j].maxval) continue;
    if (image.channel[j].w == 0 || image.channel[j].h == 0) continue;
    if (image.channel[j].hshift < 0) continue;
    int minval = image.channel[j].minval;
    if (minval > 0) minval = 0;
    int maxval = image.channel[j].maxval;
    if (maxval < 0) maxval = 0;
    pr.push_back(std::pair<PropertyVal, PropertyVal>(
        0, UNSIGNED_VAL((maxval > -minval ? maxval : minval))));
    offset++;
    pr.push_back(std::pair<PropertyVal, PropertyVal>(SIGNED_VAL(minval),
                                                     SIGNED_VAL(maxval)));
    offset++;
    pr.push_back(std::pair<PropertyVal, PropertyVal>(
        0, MAX(UNSIGNED_VAL(minval - maxval), UNSIGNED_VAL(maxval - minval))));
    offset++;
    pr.push_back(std::pair<PropertyVal, PropertyVal>(
        SIGNED_VAL(minval - maxval), SIGNED_VAL(maxval - minval)));
    offset++;
  }

  pixel_type minval = LARGEST_VAL;
  pixel_type maxval = SMALLEST_VAL;
  size_t maxh = 0;
  size_t maxw = 0;
  for (int j = beginc; j <= endc; j++) {
    if (image.channel[j].minval < minval) minval = image.channel[j].minval;
    if (image.channel[j].maxval > maxval) maxval = image.channel[j].maxval;
    if (image.channel[j].h > maxh) maxh = image.channel[j].h;
    if (image.channel[j].w > maxw) maxw = image.channel[j].w;
  }
  if (minval > 0) minval = 0;
  if (maxval < 0) maxval = 0;

  // neighbors
  pr.push_back(std::pair<PropertyVal, PropertyVal>(
      0, MAX(UNSIGNED_VAL(minval), UNSIGNED_VAL(maxval))));
  pr.push_back(std::pair<PropertyVal, PropertyVal>(
      0, MAX(UNSIGNED_VAL(minval), UNSIGNED_VAL(maxval))));
  pr.push_back(std::pair<PropertyVal, PropertyVal>(SIGNED_VAL(minval),
                                                   SIGNED_VAL(maxval)));
  pr.push_back(std::pair<PropertyVal, PropertyVal>(SIGNED_VAL(minval),
                                                   SIGNED_VAL(maxval)));

  // location
  pr.push_back(std::pair<PropertyVal, PropertyVal>(0, maxh - 1));
  pr.push_back(std::pair<PropertyVal, PropertyVal>(0, maxw - 1));

  // local gradient
  pr.push_back(std::pair<PropertyVal, PropertyVal>(minval + minval - maxval,
                                                   maxval + maxval - minval));

  // FFV1
  pr.push_back(std::pair<PropertyVal, PropertyVal>(
      SIGNED_VAL(minval - maxval), SIGNED_VAL(maxval - minval)));
  pr.push_back(std::pair<PropertyVal, PropertyVal>(
      SIGNED_VAL(minval - maxval), SIGNED_VAL(maxval - minval)));
  pr.push_back(std::pair<PropertyVal, PropertyVal>(
      SIGNED_VAL(minval - maxval), SIGNED_VAL(maxval - minval)));
  pr.push_back(std::pair<PropertyVal, PropertyVal>(
      SIGNED_VAL(minval - maxval), SIGNED_VAL(maxval - minval)));
}

inline pixel_type select(pixel_type a, pixel_type b, pixel_type c) {
  pixel_type p = a + b - c;
  pixel_type pa = abs(p - a);
  pixel_type pb = abs(p - b);
  if (pa < pb)
    return a;
  else
    return b;
}

inline pixel_type predict_and_compute_properties(
    Properties &p, const Channel &ch, const pixel_type *JXL_RESTRICT pp,
    const intptr_t onerow, const size_t x, const size_t y, Predictor predictor,
    int offset) {
  pixel_type left = (x ? pp[-1] : ch.zero);
  pixel_type top = (y ? pp[-onerow] : left);
  pixel_type topleft = (x && y ? pp[-1 - onerow] : left);
  pixel_type topright = (x + 1 < ch.w && y ? pp[1 - onerow] : top);
  pixel_type leftleft = (x > 1 ? pp[-2] : left);
  //  pixel_type toptop = (y > 1 ? pp[-onerow - onerow] : top);

  // neighbors
  p[offset++] = UNSIGNED_VAL(top);
  p[offset++] = UNSIGNED_VAL(left);
  p[offset++] = SIGNED_VAL(top);
  p[offset++] = SIGNED_VAL(left);

  // location
  p[offset++] = y;
  p[offset++] = x;

  // local gradient
  p[offset++] = left + top - topleft;
  //  p[offset++] = topleft + topright - top;

  // FFV1 context properties
  p[offset++] = SIGNED_VAL(left - topleft);
  p[offset++] = SIGNED_VAL(topleft - top);
  p[offset++] = SIGNED_VAL(top - topright);
  //  p[offset++] = SIGNED_VAL(top - toptop);
  p[offset++] = SIGNED_VAL(left - leftleft);

  switch (predictor) {
    case Predictor::Zero:
      return ch.zero;
    case Predictor::Left:
      return left;
    case Predictor::Top:
      return top;
    case Predictor::Average:
      return (left + top) / 2;
    case Predictor::Select:
      return select(left, top, topleft);
    case Predictor::Gradient:
    default:
      return median3((pixel_type)(left + top - topleft), left, top);
  }
}

inline pixel_type predict(const Channel &ch, const pixel_type *JXL_RESTRICT pp,
                          const intptr_t onerow, const int x, const int y,
                          int predictor) ATTRIBUTE_HOT;
inline pixel_type predict(const Channel &ch, const pixel_type *JXL_RESTRICT pp,
                          const intptr_t onerow, const int x, const int y,
                          Predictor predictor) {
  pixel_type left = (x ? pp[-1] : ch.zero);
  pixel_type top = (y ? pp[-onerow] : left);
  pixel_type topleft = (x && y ? pp[-1 - onerow] : left);
  switch (predictor) {
    case Predictor::Zero:
      return ch.zero;
    case Predictor::Left:
      return left;
    case Predictor::Top:
      return top;
    case Predictor::Average:
      return (left + top) / 2;
    case Predictor::Select:
      return select(left, top, topleft);
    case Predictor::Gradient:
    default:
      return median3((pixel_type)(left + top - topleft), left, top);
  }
}

inline pixel_type predict_and_compute_properties_no_edge_case(
    Properties &p, const pixel_type *JXL_RESTRICT pp, const intptr_t onerow,
    const size_t x, const size_t y, int offset = 0) ATTRIBUTE_HOT;
inline pixel_type predict_and_compute_properties_no_edge_case(
    Properties &p, const pixel_type *JXL_RESTRICT pp, const intptr_t onerow,
    const size_t x, const size_t y, int offset) {
  JXL_DASSERT(x > 1);
  JXL_DASSERT(y > 1);

  pixel_type left = pp[-1];
  pixel_type top = pp[-onerow];
  pixel_type topleft = pp[-1 - onerow];
  pixel_type topright = pp[1 - onerow];

  pixel_type leftleft = pp[-2];
  //  pixel_type toptop = pp[-onerow - onerow];

  // neighbors
  p[offset++] = UNSIGNED_VAL(top);
  p[offset++] = UNSIGNED_VAL(left);
  p[offset++] = SIGNED_VAL(top);
  p[offset++] = SIGNED_VAL(left);

  // location
  p[offset++] = y;
  p[offset++] = x;

  // local gradient
  p[offset++] = left + top - topleft;
  //  p[offset++] = topleft + topright - top;

  // FFV1 context properties
  p[offset++] = SIGNED_VAL(left - topleft);
  p[offset++] = SIGNED_VAL(topleft - top);
  p[offset++] = SIGNED_VAL(top - topright);
  //  p[offset++] = SIGNED_VAL(top - toptop);
  p[offset++] = SIGNED_VAL(left - leftleft);

  return 0;
}

#define NB_NONREF_PROPERTIES 11

inline void precompute_references(const Channel &ch, size_t y,
                                  const Image &image, int i,
                                  const modular_options &options,
                                  Channel &references) {
  int offset = 0;
  size_t oy = y << ch.vshift;
  intptr_t onerow = references.plane.PixelsPerRow();
  pixel_type *lastrow = references.Row(0) + references.h * onerow;
  for (int j = i - 1; j >= 0 && offset < options.max_properties; j--) {
    if (image.channel[j].minval == image.channel[j].maxval) continue;
    if (image.channel[j].w == 0 || image.channel[j].h == 0) continue;
    if (image.channel[j].hshift < 0) continue;
    size_t ry = oy >> image.channel[j].vshift;
    if (ry >= image.channel[j].h) ry = image.channel[j].h - 1;
    pixel_type *JXL_RESTRICT rp = references.Row(0) + offset;
    const pixel_type *JXL_RESTRICT rpp = image.channel[j].Row(ry);
    const pixel_type *JXL_RESTRICT rpprev =
        image.channel[j].Row(ry ? ry - 1 : 0);

    if (ch.hshift == image.channel[j].hshift && ch.w <= image.channel[j].w) {
      for (size_t x = 0; x < ch.w; x++, rp += onerow) {
        size_t rx = x;
        pixel_type v = rpp[rx];
        rp[0] = UNSIGNED_VAL(v);
        rp[1] = SIGNED_VAL(v);
        pixel_type vleft = (rx ? rpp[rx - 1] : image.channel[j].zero);
        pixel_type vtop = (ry ? rpprev[rx] : vleft);
        pixel_type vtopleft = (rx && ry ? rpprev[rx - 1] : vleft);
        pixel_type vpredicted =
            median3((pixel_type)(vleft + vtop - vtopleft), vleft, vtop);
        rp[2] = UNSIGNED_VAL(v - vpredicted);
        rp[3] = SIGNED_VAL(v - vpredicted);
      }
    } else if (ch.hshift < image.channel[j].hshift) {
      size_t stepsize = (1 << image.channel[j].hshift) >> ch.hshift;
      size_t rx = 0;
      pixel_type v;
      if (stepsize == 2)
        for (; rx < image.channel[j].w - 1; rx++) {
          v = rpp[rx];
          pixel_type vleft = (rx ? rpp[rx - 1] : image.channel[j].zero);
          pixel_type vtop = (ry ? rpprev[rx] : vleft);
          pixel_type vtopleft = (rx && ry ? rpprev[rx - 1] : vleft);
          pixel_type vpredicted =
              median3((pixel_type)(vleft + vtop - vtopleft), vleft, vtop);
          rp[0] = UNSIGNED_VAL(v);
          rp[1] = SIGNED_VAL(v);
          rp[2] = UNSIGNED_VAL(v - vpredicted);
          rp[3] = SIGNED_VAL(v - vpredicted);
          rp += onerow;
          rp[0] = UNSIGNED_VAL(v);
          rp[1] = SIGNED_VAL(v);
          rp[2] = UNSIGNED_VAL(v - vpredicted);
          rp[3] = SIGNED_VAL(v - vpredicted);
          rp += onerow;
        }
      else
        for (; rx < image.channel[j].w - 1; rx++) {
          v = rpp[rx];
          pixel_type vleft = (rx ? rpp[rx - 1] : image.channel[j].zero);
          pixel_type vtop = (ry ? rpprev[rx] : vleft);
          pixel_type vtopleft = (rx && ry ? rpprev[rx - 1] : vleft);
          pixel_type vpredicted =
              median3((pixel_type)(vleft + vtop - vtopleft), vleft, vtop);
          for (size_t s = 0; s < stepsize; s++, rp += onerow) {
            rp[0] = UNSIGNED_VAL(v);
            rp[1] = SIGNED_VAL(v);
            rp[2] = UNSIGNED_VAL(v - vpredicted);
            rp[3] = SIGNED_VAL(v - vpredicted);
          }
        }
      // assert (x-1 < ch.w);
      v = rpp[rx];
      pixel_type vleft = (rx ? rpp[rx - 1] : image.channel[j].zero);
      pixel_type vtop = (ry ? rpprev[rx] : vleft);
      pixel_type vtopleft = (rx && ry ? rpprev[rx - 1] : vleft);
      pixel_type vpredicted =
          median3((pixel_type)(vleft + vtop - vtopleft), vleft, vtop);
      while (rp < lastrow) {
        rp[0] = UNSIGNED_VAL(v);
        rp[1] = SIGNED_VAL(v);
        rp[2] = UNSIGNED_VAL(v - vpredicted);
        rp[3] = SIGNED_VAL(v - vpredicted);
        rp += onerow;
      }
    } else
      // all the above are just some special cases of this:
      for (size_t x = 0; x < ch.w; x++, rp += onerow) {
        size_t ox = x << ch.hshift;
        size_t rx = ox >> image.channel[j].hshift;
        if (rx >= image.channel[j].w) rx = image.channel[j].w - 1;
        pixel_type v = rpp[rx];
        rp[0] = UNSIGNED_VAL(v);
        rp[1] = SIGNED_VAL(v);
        pixel_type vleft = (rx ? rpp[rx - 1] : image.channel[j].zero);
        pixel_type vtop = (ry ? rpprev[rx] : vleft);
        pixel_type vtopleft = (rx && ry ? rpprev[rx - 1] : vleft);
        pixel_type vpredicted =
            median3((pixel_type)(vleft + vtop - vtopleft), vleft, vtop);
        rp[2] = UNSIGNED_VAL(v - vpredicted);
        rp[3] = SIGNED_VAL(v - vpredicted);
      }

    offset += 4;
  }
}

inline pixel_type predict_and_compute_properties_with_precomputed_reference(
    Properties &p, const Channel &ch, const pixel_type *JXL_RESTRICT pp,
    const intptr_t onerow, const int x, const int y, Predictor predictor,
    const Image &image, int i, modular_options &options,
    const Channel &references) {
  size_t offset;
  const pixel_type *JXL_RESTRICT rp = references.Row(x);
  for (offset = 0; offset < references.w;) {
    p[offset] = rp[offset];
    offset++;
    p[offset] = rp[offset];
    offset++;
    p[offset] = rp[offset];
    offset++;
    p[offset] = rp[offset];
    offset++;
  }
  return predict_and_compute_properties(p, ch, pp, onerow, x, y, predictor,
                                        offset);
}

inline pixel_type
predict_and_compute_properties_with_precomputed_reference_no_edge_case(
    Properties &p, const pixel_type *JXL_RESTRICT pp, const intptr_t onerow,
    const int x, const int y, const Channel &references) {
  size_t offset;
  const pixel_type *JXL_RESTRICT rp = references.Row(x);
  for (offset = 0; offset < references.w;) {
    p[offset] = rp[offset];
    offset++;
    p[offset] = rp[offset];
    offset++;
    p[offset] = rp[offset];
    offset++;
    p[offset] = rp[offset];
    offset++;
  }
  return predict_and_compute_properties_no_edge_case(p, pp, onerow, x, y,
                                                     offset);
}

}  // namespace jxl

#endif  // JXL_MODULAR_ENCODING_CONTEXT_PREDICT_H_
