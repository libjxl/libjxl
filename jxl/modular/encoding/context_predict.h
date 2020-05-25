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

#include <utility>
#include <vector>

#include "jxl/fields.h"
#include "jxl/modular/encoding/ma.h"
#include "jxl/modular/image/image.h"
#include "jxl/modular/options.h"
#include "jxl/predictor_shared.h"

namespace jxl {

namespace weighted {
constexpr static size_t kNumPredictors = 4;
constexpr static int64_t kPredExtraBits = 3;
constexpr static int64_t kPredictionRound = ((1 << kPredExtraBits) >> 1) - 1;
constexpr static size_t kNumProperties = 1;

struct Header {
  static const char *Name() { return "WeightedPredictorHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor *JXL_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;
    auto visit_p = [visitor](pixel_type val, pixel_type *p) {
      uint32_t up = *p;
      visitor->Bits(5, val, &up);
      *p = up;
    };
    visit_p(16, &p1C);
    visit_p(10, &p2C);
    visit_p(7, &p3Ca);
    visit_p(7, &p3Cb);
    visit_p(7, &p3Cc);
    visit_p(0, &p3Cd);
    visit_p(0, &p3Ce);
    visitor->Bits(4, 0xd, &w[0]);
    visitor->Bits(4, 0xc, &w[1]);
    visitor->Bits(4, 0xc, &w[2]);
    visitor->Bits(4, 0xc, &w[3]);
    return true;
  }

  Header() { Bundle::Init(this); }

  bool all_default;
  pixel_type p1C = 0, p2C = 0, p3Ca = 0, p3Cb = 0, p3Cc = 0, p3Cd = 0, p3Ce = 0;
  uint32_t w[kNumPredictors] = {};
};

struct State {
  pixel_type_w prediction[kNumPredictors];
  pixel_type_w pred;  // *before* removing the added bits.
  std::vector<uint32_t> pred_errors[kNumPredictors];
  std::vector<int32_t> error;
  Header header;

  // Allows to approximate division by a number from 1 to 64.
  uint32_t divlookup[64];

  constexpr static pixel_type_w AddBits(pixel_type_w x) {
    return uint64_t(x) << kPredExtraBits;
  }

  State(Header header, size_t xsize, size_t ysize) : header(header) {
    // Extra margin to avoid out-of-bounds writes.
    // All have space for two rows of data.
    for (size_t i = 0; i < 4; i++) {
      pred_errors[i].resize((xsize + 2) * 2);
    }
    error.resize((xsize + 2) * 2);
    // Initialize division lookup table.
    for (int i = 0; i < 64; i++) {
      divlookup[i] = (1 << 24) / (i + 1);
    }
  }

  // Approximates 4+(maxweight<<24)/(x+1), avoiding division
  JXL_INLINE uint32_t ErrorWeight(uint32_t x, uint32_t maxweight) const {
    int shift = FloorLog2Nonzero(x + 1) - 5;
    if (shift < 0) shift = 0;
    return 4 + ((maxweight * divlookup[x >> shift]) >> shift);
  }

  // Approximates the weighted average of the input values with the given
  // weights, avoiding division. Weights must sum to at least 16.
  JXL_INLINE pixel_type_w
  WeightedAverage(const pixel_type_w *JXL_RESTRICT p,
                  std::array<uint32_t, kNumPredictors> w) const {
    uint32_t weight_sum = 0;
    for (size_t i = 0; i < kNumPredictors; i++) {
      weight_sum += w[i];
    }
    JXL_DASSERT(weight_sum > 15);
    uint32_t log_weight = FloorLog2Nonzero(weight_sum);  // at least 4.
    weight_sum = 0;
    for (size_t i = 0; i < kNumPredictors; i++) {
      w[i] >>= log_weight - 4;
      weight_sum += w[i];
    }
    // for rounding.
    pixel_type_w sum = (weight_sum >> 1) - 1;
    for (size_t i = 0; i < kNumPredictors; i++) {
      sum += p[i] * w[i];
    }
    return (sum * divlookup[weight_sum - 1]) >> 24;
  }

  template <bool compute_properties>
  JXL_INLINE pixel_type_w Predict(size_t x, size_t y, size_t xsize,
                                  pixel_type_w N, pixel_type_w W,
                                  pixel_type_w NE, pixel_type_w NW,
                                  pixel_type_w NN, Properties *properties,
                                  size_t *offset) {
    size_t cur_row = y & 1 ? 0 : (xsize + 2);
    size_t prev_row = y & 1 ? (xsize + 2) : 0;
    size_t pos_N = prev_row + x;
    size_t pos_NE = x < xsize - 1 ? pos_N + 1 : pos_N;
    size_t pos_NW = x > 0 ? pos_N - 1 : pos_N;
    std::array<uint32_t, kNumPredictors> weights;
    for (size_t i = 0; i < kNumPredictors; i++) {
      // pred_errors[pos_N] also contains the error of pixel W.
      // pred_errors[pos_NW] also contains the error of pixel WW.
      weights[i] = pred_errors[i][pos_N] + pred_errors[i][pos_NE] +
                   pred_errors[i][pos_NW];
      weights[i] = ErrorWeight(weights[i], header.w[i]);
    }

    N = AddBits(N);
    W = AddBits(W);
    NE = AddBits(NE);
    NW = AddBits(NW);
    NN = AddBits(NN);

    pixel_type_w teW = x == 0 ? 0 : error[cur_row + x - 1];
    pixel_type_w teN = error[pos_N];
    pixel_type_w teNW = error[pos_NW];
    pixel_type_w sumWN = teN + teW;
    pixel_type_w teNE = error[pos_NE];

    if (compute_properties) {
      (*properties)[(*offset)++] =
          std::max(std::max(std::abs(teW), std::abs(teN)),
                   std::max(std::abs(teNW), std::abs(teNE)));
    }

    prediction[0] = W + NE - N;
    prediction[1] = N - (((sumWN + teNE) * header.p1C) >> 5);
    prediction[2] = W - (((sumWN + teNW) * header.p2C) >> 5);
    prediction[3] =
        N - ((teNW * header.p3Ca + teN * header.p3Cb + teNE * header.p3Cc +
              (NN - N) * header.p3Cd + (NW - W) * header.p3Ce) >>
             5);

    pred = WeightedAverage(prediction, weights);

    // If all three have the same sign, skip clamping.
    if (((teN ^ teW) | (teN ^ teNW)) > 0) {
      return (pred + kPredictionRound) >> kPredExtraBits;
    }

    // Otherwise, clamp to min/max of neighbouring pixels (just W, NE, N).
    pixel_type_w mx = std::max(W, std::max(NE, N));
    pixel_type_w mn = std::min(W, std::min(NE, N));
    pred = std::max(mn, std::min(mx, pred));
    return (pred + kPredictionRound) >> kPredExtraBits;
  }

  JXL_INLINE void UpdateErrors(pixel_type_w val, size_t x, size_t y,
                               size_t xsize) {
    size_t cur_row = y & 1 ? 0 : (xsize + 2);
    size_t prev_row = y & 1 ? (xsize + 2) : 0;
    val = AddBits(val);
    error[cur_row + x] = ClampToRange<pixel_type>(pred - val);
    for (size_t i = 0; i < kNumPredictors; i++) {
      pixel_type_w err =
          (std::abs(prediction[i] - val) + kPredictionRound) >> kPredExtraBits;
      // For predicting in the next row.
      pred_errors[i][cur_row + x] = err;
      // Add the error on this pixel to the error on the NE pixel. This has the
      // effect of adding the error on this pixel to the E and EE pixels.
      pred_errors[i][prev_row + x + 1] += err;
    }
  }
};

// Encoder helper function to set the parameters to some presets.
inline void PredictorMode(int i, Header *header) {
  switch (i) {
    case 0:
      // ~ lossless16 predictor
      header->w[0] = 0xd;
      header->w[1] = 0xc;
      header->w[2] = 0xc;
      header->w[3] = 0xc;
      header->p1C = 16;
      header->p2C = 10;
      header->p3Ca = 7;
      header->p3Cb = 7;
      header->p3Cc = 7;
      header->p3Cd = 0;
      header->p3Ce = 0;
      break;
    case 1:
      // ~ default lossless8 predictor
      header->w[0] = 0xd;
      header->w[1] = 0xc;
      header->w[2] = 0xc;
      header->w[3] = 0xb;
      header->p1C = 8;
      header->p2C = 8;
      header->p3Ca = 4;
      header->p3Cb = 0;
      header->p3Cc = 3;
      header->p3Cd = 23;
      header->p3Ce = 2;
      break;
    case 2:
      // ~ west lossless8 predictor
      header->w[0] = 0xd;
      header->w[1] = 0xc;
      header->w[2] = 0xd;
      header->w[3] = 0xc;
      header->p1C = 10;
      header->p2C = 9;
      header->p3Ca = 7;
      header->p3Cb = 0;
      header->p3Cc = 0;
      header->p3Cd = 16;
      header->p3Ce = 9;
      break;
    case 3:
      // ~ north lossless8 predictor
      header->w[0] = 0xd;
      header->w[1] = 0xd;
      header->w[2] = 0xc;
      header->w[3] = 0xc;
      header->p1C = 16;
      header->p2C = 8;
      header->p3Ca = 0;
      header->p3Cb = 16;
      header->p3Cc = 0;
      header->p3Cd = 23;
      header->p3Ce = 0;
      break;
    case 4:
    default:
      // something else, because why not
      header->w[0] = 0xd;
      header->w[1] = 0xc;
      header->w[2] = 0xc;
      header->w[3] = 0xc;
      header->p1C = 10;
      header->p2C = 10;
      header->p3Ca = 5;
      header->p3Cb = 5;
      header->p3Cc = 5;
      header->p3Cd = 12;
      header->p3Ce = 4;
      break;
  }
}
}  // namespace weighted

class MATreeLookup {
 public:
  explicit MATreeLookup(const Tree &tree) : inner_nodes_(tree) {}
  struct LookupResult {
    int context;
    Predictor predictor;
    int64_t offset;
  };
  LookupResult Lookup(const Properties &properties) const {
    Tree::size_type pos = 0;
    while (true) {
      const PropertyDecisionNode &node = inner_nodes_[pos];
      if (node.property < 0) {
        return {node.childID, node.predictor, node.predictor_offset};
      }
      if (properties[node.property] > node.splitval) {
        pos = node.childID;
      } else {
        pos = node.childID + 1;
      }
    }
  }

 private:
  const Tree &inner_nodes_;
};

// Something that looks like absolute value. Just returning the absolute value
// works. Wrapped into a function to make it easier to experiment with
// alternatives.
inline pixel_type UnsignedVal(pixel_type_w x) {
  return ClampToRange<pixel_type>(abs(x));
}

// Something that preserves the sign but potentially reduces the range
// Just returning the value works.
// Wrapped into a function to make it easier to experiment with alternatives
inline pixel_type SignedVal(pixel_type_w x) {
  return ClampToRange<pixel_type>(x);
}

static constexpr size_t kNumNonrefProperties = 11 + weighted::kNumProperties;

inline size_t NumProperties(const Image &image, int beginc, int endc,
                            const ModularOptions &options) {
  int num = 0;
  for (int j = beginc - 1; j >= 0 && num < options.max_properties; j--) {
    if (image.channel[j].is_trivial) continue;
    if (image.channel[j].w == 0 || image.channel[j].h == 0) continue;
    if (image.channel[j].hshift < 0) continue;
    // 4 properties per previous channel.
    num += 4;
  }
  // Add properties for the current channels.
  return num + kNumNonrefProperties;
}

inline pixel_type_w Select(pixel_type_w a, pixel_type_w b, pixel_type_w c) {
  pixel_type_w p = a + b - c;
  pixel_type_w pa = abs(p - a);
  pixel_type_w pb = abs(p - b);
  return pa < pb ? a : b;
}

inline void PrecomputeReferences(const Channel &ch, size_t y,
                                 const Image &image, int i,
                                 const ModularOptions &options,
                                 Channel *references) {
  int offset = 0;
  size_t oy = y << ch.vshift;
  intptr_t onerow = references->plane.PixelsPerRow();
  pixel_type *lastrow = references->Row(0) + references->h * onerow;
  for (int j = i - 1; j >= 0 && offset < options.max_properties; j--) {
    if (image.channel[j].is_trivial) continue;
    if (image.channel[j].w == 0 || image.channel[j].h == 0) continue;
    if (image.channel[j].hshift < 0) continue;
    size_t ry = oy >> image.channel[j].vshift;
    if (ry >= image.channel[j].h) ry = image.channel[j].h - 1;
    pixel_type *JXL_RESTRICT rp = references->Row(0) + offset;
    const pixel_type *JXL_RESTRICT rpp = image.channel[j].Row(ry);
    const pixel_type *JXL_RESTRICT rpprev =
        image.channel[j].Row(ry ? ry - 1 : 0);

    if (ch.hshift == image.channel[j].hshift && ch.w <= image.channel[j].w) {
      for (size_t x = 0; x < ch.w; x++, rp += onerow) {
        size_t rx = x;
        pixel_type_w v = rpp[rx];
        rp[0] = UnsignedVal(v);
        rp[1] = SignedVal(v);
        pixel_type_w vleft = (rx ? rpp[rx - 1] : 0);
        pixel_type_w vtop = (ry ? rpprev[rx] : vleft);
        pixel_type_w vtopleft = (rx && ry ? rpprev[rx - 1] : vleft);
        pixel_type_w vpredicted = ClampedGradient(vleft, vtop, vtopleft);
        rp[2] = UnsignedVal(v - vpredicted);
        rp[3] = SignedVal(v - vpredicted);
      }
    } else if (ch.hshift < image.channel[j].hshift) {
      size_t stepsize = 1 << (image.channel[j].hshift - ch.hshift);
      size_t rx = 0;
      pixel_type_w v;
      if (stepsize == 2) {
        for (; rx < image.channel[j].w - 1 &&
               rx < (ch.w >> (image.channel[j].hshift - ch.hshift));
             rx++) {
          v = rpp[rx];
          pixel_type_w vleft = (rx ? rpp[rx - 1] : 0);
          pixel_type_w vtop = (ry ? rpprev[rx] : vleft);
          pixel_type_w vtopleft = (rx && ry ? rpprev[rx - 1] : vleft);
          pixel_type_w vpredicted = ClampedGradient(vleft, vtop, vtopleft);
          rp[0] = UnsignedVal(v);
          rp[1] = SignedVal(v);
          rp[2] = UnsignedVal(v - vpredicted);
          rp[3] = SignedVal(v - vpredicted);
          rp += onerow;
          rp[0] = UnsignedVal(v);
          rp[1] = SignedVal(v);
          rp[2] = UnsignedVal(v - vpredicted);
          rp[3] = SignedVal(v - vpredicted);
          rp += onerow;
        }
      } else {
        for (; rx < image.channel[j].w - 1 &&
               rx < (ch.w >> (image.channel[j].hshift - ch.hshift));
             rx++) {
          v = rpp[rx];
          pixel_type_w vleft = (rx ? rpp[rx - 1] : 0);
          pixel_type_w vtop = (ry ? rpprev[rx] : vleft);
          pixel_type_w vtopleft = (rx && ry ? rpprev[rx - 1] : vleft);
          pixel_type_w vpredicted = ClampedGradient(vleft, vtop, vtopleft);
          for (size_t s = 0; s < stepsize; s++, rp += onerow) {
            rp[0] = UnsignedVal(v);
            rp[1] = SignedVal(v);
            rp[2] = UnsignedVal(v - vpredicted);
            rp[3] = SignedVal(v - vpredicted);
          }
        }
      }
      // assert (x-1 < ch.w);
      v = rpp[rx];
      pixel_type_w vleft = (rx ? rpp[rx - 1] : 0);
      pixel_type_w vtop = (ry ? rpprev[rx] : vleft);
      pixel_type_w vtopleft = (rx && ry ? rpprev[rx - 1] : vleft);
      pixel_type_w vpredicted = ClampedGradient(vleft, vtop, vtopleft);
      while (rp < lastrow) {
        rp[0] = UnsignedVal(v);
        rp[1] = SignedVal(v);
        rp[2] = UnsignedVal(v - vpredicted);
        rp[3] = SignedVal(v - vpredicted);
        rp += onerow;
      }
    } else
      // all the above are just some special cases of this:
      for (size_t x = 0; x < ch.w; x++, rp += onerow) {
        size_t ox = x << ch.hshift;
        size_t rx = ox >> image.channel[j].hshift;
        if (rx >= image.channel[j].w) rx = image.channel[j].w - 1;
        pixel_type_w v = rpp[rx];
        rp[0] = UnsignedVal(v);
        rp[1] = SignedVal(v);
        pixel_type_w vleft = (rx ? rpp[rx - 1] : 0);
        pixel_type_w vtop = (ry ? rpprev[rx] : vleft);
        pixel_type_w vtopleft = (rx && ry ? rpprev[rx - 1] : vleft);
        pixel_type_w vpredicted = ClampedGradient(vleft, vtop, vtopleft);
        rp[2] = UnsignedVal(v - vpredicted);
        rp[3] = SignedVal(v - vpredicted);
      }

    offset += 4;
  }
}

struct PredictionResult {
  int context = 0;
  pixel_type_w guess = 0;
};

namespace detail {
enum PredictorMode {
  kUseTree = 1,
  kUseWP = 2,
  kForceComputeProperties = 4,
  kAllPredictions = 8,
};

template <int mode>
inline PredictionResult Predict(Properties *p, const Channel &ch,
                                const pixel_type *JXL_RESTRICT pp,
                                const intptr_t onerow, const int x, const int y,
                                Predictor predictor, const MATreeLookup *lookup,
                                const Channel *references,
                                weighted::State *wp_state,
                                pixel_type_w *predictions) {
  size_t offset;
  constexpr bool compute_properties =
      mode & kUseTree || mode & kForceComputeProperties;
  if (compute_properties) {
    const pixel_type *JXL_RESTRICT rp = references->Row(x);
    for (offset = 0; offset < references->w;) {
      (*p)[offset] = rp[offset];
      offset++;
    }
  }
  pixel_type_w left = (x ? pp[-1] : 0);
  pixel_type_w top = (y ? pp[-onerow] : left);
  pixel_type_w topleft = (x && y ? pp[-1 - onerow] : left);
  pixel_type_w topright = (x + 1 < ch.w && y ? pp[1 - onerow] : top);
  pixel_type_w leftleft = (x > 1 ? pp[-2] : left);
  pixel_type_w toptop = (y > 1 ? pp[-onerow - onerow] : top);

  if (compute_properties) {
    // neighbors
    (*p)[offset++] = UnsignedVal(top);
    (*p)[offset++] = UnsignedVal(left);
    (*p)[offset++] = SignedVal(top);
    (*p)[offset++] = SignedVal(left);

    // location
    (*p)[offset++] = y;
    (*p)[offset++] = x;

    // local gradient
    (*p)[offset++] = left + top - topleft;
    //  (*p)[offset++] = topleft + topright - top;

    // FFV1 context properties
    (*p)[offset++] = SignedVal(left - topleft);
    (*p)[offset++] = SignedVal(topleft - top);
    (*p)[offset++] = SignedVal(top - topright);
    //  (*p)[offset++] = SignedVal(top - toptop);
    (*p)[offset++] = SignedVal(left - leftleft);
  }

  pixel_type_w wp_pred = 0;
  if (mode & kUseWP) {
    wp_pred = wp_state->Predict<compute_properties>(
        x, y, ch.w, top, left, topright, topleft, toptop, p, &offset);
  }
  PredictionResult result;
  if (mode & kUseTree) {
    MATreeLookup::LookupResult lr = lookup->Lookup(*p);
    result.context = lr.context;
    result.guess = lr.offset;
    predictor = lr.predictor;
  }
  pixel_type_w pred_storage[(int)Predictor::Best];
  if (!(mode & kAllPredictions)) {
    predictions = pred_storage;
  }
  predictions[(int)Predictor::Zero] = 0;
  predictions[(int)Predictor::Left] = left;
  predictions[(int)Predictor::Top] = top;
  predictions[(int)Predictor::Average] = (left + top) / 2;
  predictions[(int)Predictor::Select] = Select(left, top, topleft);
  predictions[(int)Predictor::Weighted] = wp_pred;
  predictions[(int)Predictor::Gradient] = ClampedGradient(left, top, topleft);

  result.guess += predictions[(int)predictor];

  return result;
}
}  // namespace detail

inline PredictionResult PredictNoTreeNoWP(const Channel &ch,
                                          const pixel_type *JXL_RESTRICT pp,
                                          const intptr_t onerow, const int x,
                                          const int y, Predictor predictor) {
  return detail::Predict</*mode=*/0>(
      /*p=*/nullptr, ch, pp, onerow, x, y, predictor, /*lookup=*/nullptr,
      /*references=*/nullptr, /*wp_state=*/nullptr, /*predictions=*/nullptr);
}

inline PredictionResult PredictNoTreeWP(const Channel &ch,
                                        const pixel_type *JXL_RESTRICT pp,
                                        const intptr_t onerow, const int x,
                                        const int y, Predictor predictor,
                                        weighted::State *wp_state) {
  return detail::Predict<detail::kUseWP>(
      /*p=*/nullptr, ch, pp, onerow, x, y, predictor, /*lookup=*/nullptr,
      /*references=*/nullptr, wp_state, /*predictions=*/nullptr);
}

inline PredictionResult PredictTreeNoWP(Properties *p, const Channel &ch,
                                        const pixel_type *JXL_RESTRICT pp,
                                        const intptr_t onerow, const int x,
                                        const int y,
                                        const MATreeLookup &tree_lookup,
                                        const Channel &references) {
  return detail::Predict<detail::kUseTree>(
      p, ch, pp, onerow, x, y, Predictor::Zero, &tree_lookup, &references,
      /*wp_state=*/nullptr, /*predictions=*/nullptr);
}

inline PredictionResult PredictTreeWP(Properties *p, const Channel &ch,
                                      const pixel_type *JXL_RESTRICT pp,
                                      const intptr_t onerow, const int x,
                                      const int y,
                                      const MATreeLookup &tree_lookup,
                                      const Channel &references,
                                      weighted::State *wp_state) {
  return detail::Predict<detail::kUseTree | detail::kUseWP>(
      p, ch, pp, onerow, x, y, Predictor::Zero, &tree_lookup, &references,
      wp_state, /*predictions=*/nullptr);
}

inline PredictionResult PredictLearn(Properties *p, const Channel &ch,
                                     const pixel_type *JXL_RESTRICT pp,
                                     const intptr_t onerow, const int x,
                                     const int y, Predictor predictor,
                                     const Channel &references,
                                     weighted::State *wp_state) {
  return detail::Predict<detail::kForceComputeProperties | detail::kUseWP>(
      p, ch, pp, onerow, x, y, predictor, /*lookup=*/nullptr, &references,
      wp_state, /*predictions=*/nullptr);
}

inline void PredictLearnAll(Properties *p, const Channel &ch,
                            const pixel_type *JXL_RESTRICT pp,
                            const intptr_t onerow, const int x, const int y,
                            const Channel &references,
                            weighted::State *wp_state,
                            pixel_type_w *predictions) {
  detail::Predict<detail::kForceComputeProperties | detail::kUseWP |
                  detail::kAllPredictions>(p, ch, pp, onerow, x, y,
                                           Predictor::Zero, /*lookup=*/nullptr,
                                           &references, wp_state, predictions);
}

inline void PredictAllNoWP(const Channel &ch, const pixel_type *JXL_RESTRICT pp,
                           const intptr_t onerow, const int x, const int y,
                           pixel_type_w *predictions) {
  detail::Predict<detail::kAllPredictions>(
      /*p=*/nullptr, ch, pp, onerow, x, y, Predictor::Zero, /*lookup=*/nullptr,
      /*references=*/nullptr, /*wp_state=*/nullptr, predictions);
}
}  // namespace jxl

#endif  // JXL_MODULAR_ENCODING_CONTEXT_PREDICT_H_
