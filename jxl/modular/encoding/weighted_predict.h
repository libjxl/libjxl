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

// @author Alexander Rhatushnyak

#ifndef JXL_MODULAR_ENCODING_WEIGHTED_PREDICT_H_
#define JXL_MODULAR_ENCODING_WEIGHTED_PREDICT_H_

#include "jxl/base/bits.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_ans.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/entropy_coder.h"
#include "jxl/fields.h"
#include "jxl/modular/image/image.h"

namespace jxl {

namespace {
constexpr int kWithSign = 7;
constexpr int kBitsMax = 13;
constexpr int kNumContexts = 1 + kWithSign + kBitsMax;

// SET ME TO ZERO FOR A FASTER VERSION WITH NO ROUNDING!
constexpr int PBits = 3;
constexpr int toRound = ((1 << PBits) >> 1) - 1;
constexpr int toRound_m1 = toRound;  // (toRound ? toRound - 1 : 0);
constexpr pixel_type AddPBits(pixel_type x) { return uint32_t(x) << PBits; }
}  // namespace

struct WeightedPredictorHeader {
  static const char* Name() { return "WeightedPredictorHeader"; }

  template <class Visitor>
  Status VisitFields(Visitor* JXL_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;
    auto visit_p = [visitor](pixel_type val, pixel_type* p) {
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
    visitor->Bits(4, 0xd, &w0);
    visitor->Bits(4, 0xc, &w1);
    visitor->Bits(4, 0xc, &w2);
    visitor->Bits(4, 0xc, &w3);
    return true;
  }

  WeightedPredictorHeader() { Bundle::Init(this); }

  bool all_default;
  pixel_type p1C = 0, p2C = 0, p3Ca = 0, p3Cb = 0, p3Cc = 0, p3Cd = 0, p3Ce = 0;
  uint32_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
};

struct WeightedPredictorState {
  pixel_type prediction0, prediction1, prediction2, prediction3;
  WeightedPredictorHeader header;
  const size_t lastX;  // xsize - 1, for prediction borders
  pixel_type* JXL_RESTRICT rowImg;
  const pixel_type *JXL_RESTRICT rowPrev, *JXL_RESTRICT rowPP;

  std::vector<uint32_t> errors0;  // Errors of predictor 0
  std::vector<uint32_t> errors1;  // Errors of predictor 1
  std::vector<uint32_t> errors2;  // Errors of predictor 2
  std::vector<uint32_t> errors3;  // Errors of predictor 3
  std::vector<uint8_t> nbitErr;
  std::vector<int32_t> trueErr;
  uint32_t divlookup[64];

  WeightedPredictorState(size_t imageSizeX, size_t imageSizeY)
      : lastX(imageSizeX - 1) {
    errors0.resize(imageSizeX * 2 + 4);
    errors1.resize(imageSizeX * 2 + 4);
    errors2.resize(imageSizeX * 2 + 4);
    errors3.resize(imageSizeX * 2 + 4);
    nbitErr.resize(imageSizeX * 2);
    trueErr.resize(imageSizeX * 2);
    for (int i = 0; i < 64; i++) {
      divlookup[i] = (1 << 24) / (i + 1);
    }
  }

  // approximates (maxweight<<24)/(x+1), avoiding division
  JXL_INLINE uint32_t errorWeight(uint32_t x, uint32_t maxweight) {
    // cheapest option:
    //    return maxweight << (24 - FloorLog2Nonzero(x+1));

    // better and somewhat more expensive
    //    int shift=FloorLog2Nonzero(x+1)+12;
    //    return maxweight * ((0x1000000000LLU - ((uint64_t)x<<(46-shift))) >>
    //    shift);

    // almost as good as real division
    int shift = FloorLog2Nonzero(x + 1) - 5;
    if (shift < 0) shift = 0;
    return 4 + (maxweight * divlookup[x >> shift] >> shift);
  }
  JXL_INLINE int numBits(int x) {
    if (x < 2) return x;
    return std::min(2 + (int)FloorLog2Nonzero((unsigned int)x - 1), kBitsMax);
  }

  JXL_INLINE pixel_type predict1y0(size_t x, size_t yp, size_t yp1,
                                   int* maxErr) {
    *maxErr = (x == 0 ? kNumContexts - 1
                      : x == 1 ? nbitErr[yp - 1]
                               : std::max(nbitErr[yp - 1], nbitErr[yp - 2]));
    prediction0 = prediction1 = prediction2 = prediction3 =
        (x == 0 ? 0
                : x == 1 ? AddPBits(rowImg[x - 1])
                         : AddPBits(rowImg[x - 1]) +
                               AddPBits(rowImg[x - 1] - rowImg[x - 2]) / 4);
    return prediction0;
  }

  JXL_INLINE pixel_type predict1x0(size_t x, size_t yp, size_t yp1,
                                   int* maxErr) {
    *maxErr = std::max(nbitErr[yp1], nbitErr[yp1 + (x < lastX ? 1 : 0)]);
    prediction0 = prediction2 = prediction3 = prediction1 =
        AddPBits(rowPrev[x]);
    return prediction0;
  }

  JXL_INLINE pixel_type WeightedAverage(pixel_type p1, pixel_type p2,
                                        pixel_type p3, pixel_type p4,
                                        uint32_t w1, uint32_t w2, uint32_t w3,
                                        uint32_t w4) {
    uint32_t sumWeights = w1 + w2 + w3 + w4;
    JXL_DASSERT(sumWeights > 15);
    uint32_t log_weight = FloorLog2Nonzero(sumWeights);  // at least 4.
    w1 = w1 >> (log_weight - 4);
    w2 = w2 >> (log_weight - 4);
    w3 = w3 >> (log_weight - 4);
    w4 = w4 >> (log_weight - 4);
    uint32_t new_sum_weights = w1 + w2 + w3 + w4;
    int64_t s = new_sum_weights >> 1;
    s += static_cast<int64_t>(p1) * w1;
    s += static_cast<int64_t>(p2) * w2;
    s += static_cast<int64_t>(p3) * w3;
    s += static_cast<int64_t>(p4) * w4;
    return (s * divlookup[new_sum_weights - 1]) >> 24;
  }

  JXL_INLINE pixel_type predict1(size_t x, size_t yp, size_t yp1, int* maxErr) {
    if (!rowPrev) return predict1y0(x, yp, yp1, maxErr);
    if (x == 0LL) return predict1x0(x, yp, yp1, maxErr);

    int a1 = (x < lastX ? 1 : 0);
    uint32_t weight0 = errors0[yp1] + errors0[yp1 - 1] + errors0[yp1 + a1];
    uint32_t weight1 = errors1[yp1] + errors1[yp1 - 1] + errors1[yp1 + a1];
    uint32_t weight2 = errors2[yp1] + errors2[yp1 - 1] + errors2[yp1 + a1];
    uint32_t weight3 = errors3[yp1] + errors3[yp1 - 1] + errors3[yp1 + a1];

    uint8_t mxe = nbitErr[yp - 1];
    mxe = std::max(mxe, nbitErr[yp1]);
    mxe = std::max(mxe, nbitErr[yp1 - 1]);
    mxe = std::max(mxe, nbitErr[yp1 + a1]);

    pixel_type N = AddPBits(rowPrev[x]), W = AddPBits(rowImg[x - 1]),
               NE = AddPBits(rowPrev[x + a1]), NW = AddPBits(rowPrev[x - 1]),
               NN = AddPBits(rowPP[x]);

    weight0 = errorWeight(weight0, header.w0);
    weight1 = errorWeight(weight1, header.w1);
    weight2 = errorWeight(weight2, header.w2);
    weight3 = errorWeight(weight3, header.w3);

    int teW = trueErr[yp - 1];
    int teN = trueErr[yp1];
    int teNW = trueErr[yp1 - 1];
    int sumWN = teN + teW;
    int teNE = trueErr[yp1 + a1];

    prediction0 = W + NE - N;
    prediction1 = N - (((sumWN + teNE) * header.p1C) >> 5);
    prediction2 = W - (((sumWN + teNW) * header.p2C) >> 5);
    prediction3 =
        N - ((teNW * header.p3Ca + teN * header.p3Cb + teNE * header.p3Cc +
              (NN - N) * header.p3Cd + (NW - W) * header.p3Ce) >>
             5);

    pixel_type prediction =
        WeightedAverage(prediction0, prediction1, prediction2, prediction3,
                        weight0, weight1, weight2, weight3);
    if (mxe && mxe <= kWithSign * 2) {
      if (sumWN * 2 + teNW + teNE < 0) --mxe;  // 2 2 1 1
    }
    *maxErr = mxe;

    if (((teN ^ teW) | (teN ^ teNW)) > 0) {  // if all three have the same sign
      return prediction;
    }

    pixel_type mx = (W > NE ? W : NE), mn = W + NE - mx;
    if (N > mx) mx = N;
    if (N < mn) mn = N;
    prediction = std::max(mn, std::min(mx, prediction));
    return prediction;
  }

  JXL_INLINE void UpdateSizeAndErrors(pixel_type err, size_t yp, size_t yp1,
                                      size_t x, pixel_type prediction0,
                                      pixel_type prediction1,
                                      pixel_type prediction2,
                                      pixel_type prediction3,
                                      pixel_type truePixelValue) {
    trueErr[yp + x] = err;
    err = numBits(err >= 0 ? err : -err);
    nbitErr[yp + x] = (err <= kWithSign ? err * 2 : err + kWithSign);
    err = ((prediction0 + toRound) >> PBits) - truePixelValue;
    if (err < 0) err = -err; /* abs() and min()? worse speed! */
    errors0[yp + x] = err;
    errors0[1 + yp1 + x] += err;
    err = ((prediction1 + toRound) >> PBits) - truePixelValue;
    if (err < 0) err = -err;
    errors1[yp + x] = err;
    errors1[1 + yp1 + x] += err;
    err = ((prediction2 + toRound) >> PBits) - truePixelValue;
    if (err < 0) err = -err;
    errors2[yp + x] = err;
    errors2[1 + yp1 + x] += err;
    err = ((prediction3 + toRound) >> PBits) - truePixelValue;
    if (err < 0) err = -err;
    errors3[yp + x] = err;
    errors3[1 + yp1 + x] += err;
  }

  // Encoder helper function to set the parameters to some presets.
  void predictor_mode(int i) {
    switch (i) {
      case 0:
        // ~ lossless16 predictor
        header.w0 = 0xd;
        header.w1 = 0xc;
        header.w2 = 0xc;
        header.w3 = 0xc;
        header.p1C = 16;
        header.p2C = 10;
        header.p3Ca = 7;
        header.p3Cb = 7;
        header.p3Cc = 7;
        header.p3Cd = 0;
        header.p3Ce = 0;
        break;
      case 1:
        // ~ default lossless8 predictor
        header.w0 = 0xd;
        header.w1 = 0xc;
        header.w2 = 0xc;
        header.w3 = 0xb;
        header.p1C = 8;
        header.p2C = 8;
        header.p3Ca = 4;
        header.p3Cb = 0;
        header.p3Cc = 3;
        header.p3Cd = 23;
        header.p3Ce = 2;
        break;
      case 2:
        // ~ west lossless8 predictor
        header.w0 = 0xd;
        header.w1 = 0xc;
        header.w2 = 0xd;
        header.w3 = 0xc;
        header.p1C = 10;
        header.p2C = 9;
        header.p3Ca = 7;
        header.p3Cb = 0;
        header.p3Cc = 0;
        header.p3Cd = 16;
        header.p3Ce = 9;
        break;
      case 3:
        // ~ north lossless8 predictor
        header.w0 = 0xd;
        header.w1 = 0xd;
        header.w2 = 0xc;
        header.w3 = 0xc;
        header.p1C = 16;
        header.p2C = 8;
        header.p3Ca = 0;
        header.p3Cb = 16;
        header.p3Cc = 0;
        header.p3Cd = 23;
        header.p3Ce = 0;
        break;
      case 4:
        // something else, because why not
        header.w0 = 0xd;
        header.w1 = 0xc;
        header.w2 = 0xc;
        header.w3 = 0xc;
        header.p1C = 10;
        header.p2C = 10;
        header.p3Ca = 5;
        header.p3Cb = 5;
        header.p3Cc = 5;
        header.p3Cd = 12;
        header.p3Ce = 4;
        break;
    }
  }

  bool wp_compress(const Channel& img, int nb_modes, size_t base_ctx,
                   const HybridUintConfig& uint_config,
                   std::vector<Token>* tokens,
                   WeightedPredictorHeader* header) {
    size_t xsize = img.w;
    size_t ysize = img.h;
    std::vector<Token> best_tokens;
    float best_cost;
    for (int mode = 0; mode < nb_modes; mode++) {
      predictor_mode(mode);
      std::vector<Token> local_tokens;

      for (size_t y = 0, yp = 0, yp1 = xsize; y < ysize;
           ++y, yp = xsize - yp, yp1 = xsize - yp) {
        rowImg = const_cast<pixel_type*>(img.Row(y));
        rowPrev = (y == 0 ? nullptr : img.Row(y - 1));
        rowPP = (y <= 1 ? rowPrev : img.Row(y - 2));
        for (size_t x = 0; x < xsize; ++x) {
          int maxErr;
          pixel_type prediction = predict1(x, yp + x, yp1 + x, &maxErr);
          JXL_DASSERT(0 <= maxErr && maxErr <= kNumContexts - 1);
          pixel_type truePixelValue = rowImg[x];
          TokenizeWithConfig(
              uint_config, base_ctx + maxErr,
              PackSigned(truePixelValue - ((prediction + toRound_m1) >> PBits)),
              &local_tokens);
          pixel_type err = prediction - AddPBits(truePixelValue);
          UpdateSizeAndErrors(err, yp, yp1, x, prediction0, prediction1,
                              prediction2, prediction3, truePixelValue);
        }  // x
      }    // y

      size_t extension_bits, header_bits;
      JXL_RETURN_IF_ERROR(
          Bundle::CanEncode(this->header, &extension_bits, &header_bits));

      float local_cost = TokenCost(local_tokens) + extension_bits + header_bits;
      if (mode == 0 || local_cost < best_cost) {
        best_tokens = std::move(local_tokens);
        best_cost = local_cost;
        *header = this->header;
      }
    }
    tokens->insert(tokens->end(), best_tokens.begin(), best_tokens.end());
    return true;
  }

  bool wp_decompress(BitReader* br, ANSSymbolReader* reader,
                     const std::vector<uint8_t>& context_map, size_t base_ctx,
                     const WeightedPredictorHeader& header, Channel* img) {
    size_t xsize = img->w;
    size_t ysize = img->h;

    if (!xsize || !ysize) return JXL_FAILURE("invalid image size");

    this->header = header;

    for (size_t y = 0, yp = 0, yp1 = xsize; y < ysize;
         ++y, yp = xsize - yp, yp1 = xsize - yp) {
      rowImg = img->Row(y);
      rowPrev = (y == 0 ? nullptr : img->Row(y - 1));
      rowPP = (y <= 1 ? rowPrev : img->Row(y - 2));
      for (size_t x = 0; x < xsize; ++x) {
        int maxErr;
        pixel_type prediction = predict1(x, yp + x, yp1 + x, &maxErr);
        JXL_DASSERT(0 <= maxErr && maxErr <= kNumContexts - 1);

        size_t q = reader->ReadHybridUint(maxErr + base_ctx, br, context_map);
        pixel_type truePixelValue =
            ((prediction + toRound_m1) >> PBits) + UnpackSigned(q);
        rowImg[x] = truePixelValue;
        pixel_type err = prediction - AddPBits(truePixelValue);
        UpdateSizeAndErrors(err, yp, yp1, x, prediction0, prediction1,
                            prediction2, prediction3, truePixelValue);
      }  // x
    }    // y
    return true;
  }
};

}  // namespace jxl

#endif  // JXL_MODULAR_ENCODING_WEIGHTED_PREDICT_H_
