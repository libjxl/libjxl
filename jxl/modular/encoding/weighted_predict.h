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

struct State {
  pixel_type prediction0, prediction1, prediction2, prediction3;
  pixel_type p1C, p2C, p3Ca, p3Cb, p3Cc, p3Cd, p3Ce;
  uint32_t w0, w1, w2, w3;
  const size_t lastX;  // xsize - 1, for prediction borders
  pixel_type* JXL_RESTRICT rowImg;
  const pixel_type *JXL_RESTRICT rowPrev, *JXL_RESTRICT rowPP;
  const pixel_type minTpv, maxTpv;
  const pixel_type shiftedMinTpv = AddPBits(minTpv);
  const pixel_type shiftedMaxTpv = AddPBits(maxTpv);

  std::vector<uint32_t> errors0;  // Errors of predictor 0
  std::vector<uint32_t> errors1;  // Errors of predictor 1
  std::vector<uint32_t> errors2;  // Errors of predictor 2
  std::vector<uint32_t> errors3;  // Errors of predictor 3
  std::vector<uint8_t> nbitErr;
  std::vector<int32_t> trueErr;
  uint32_t divlookup[64];

  State(size_t imageSizeX, size_t imageSizeY, pixel_type minval,
        pixel_type maxval)
      : lastX(imageSizeX - 1), minTpv(minval), maxTpv(maxval) {
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
        (x == 0 ? AddPBits(minTpv + maxTpv) / 2
                : x == 1 ? AddPBits(rowImg[x - 1])
                         : AddPBits(rowImg[x - 1]) +
                               AddPBits(rowImg[x - 1] - rowImg[x - 2]) / 4);
    return std::max(shiftedMinTpv, std::min(prediction0, shiftedMaxTpv));
  }

  JXL_INLINE pixel_type predict1x0(size_t x, size_t yp, size_t yp1,
                                   int* maxErr) {
    *maxErr = std::max(nbitErr[yp1], nbitErr[yp1 + (x < lastX ? 1 : 0)]);
    prediction0 = prediction2 = prediction3 = prediction1 =
        AddPBits(rowPrev[x]);
    return std::max(shiftedMinTpv, std::min(prediction0, shiftedMaxTpv));
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

    weight0 = errorWeight(weight0, w0);
    weight1 = errorWeight(weight1, w1);
    weight2 = errorWeight(weight2, w2);
    weight3 = errorWeight(weight3, w3);

    int teW = trueErr[yp - 1];
    int teN = trueErr[yp1];
    int teNW = trueErr[yp1 - 1];
    int sumWN = teN + teW;
    int teNE = trueErr[yp1 + a1];

    prediction0 = W + NE - N;
    prediction1 = N - (((sumWN + teNE) * p1C) >> 5);
    prediction2 = W - (((sumWN + teNW) * p2C) >> 5);
    prediction3 = N - ((teNW * p3Ca + teN * p3Cb + teNE * p3Cc +
                        (NN - N) * p3Cd + (NW - W) * p3Ce) >>
                       5);

    pixel_type prediction =
        WeightedAverage(prediction0, prediction1, prediction2, prediction3,
                        weight0, weight1, weight2, weight3);
    if (mxe && mxe <= kWithSign * 2) {
      if (sumWN * 2 + teNW + teNE < 0) --mxe;  // 2 2 1 1
    }
    *maxErr = mxe;

    if (((teN ^ teW) | (teN ^ teNW)) > 0) {  // if all three have the same sign
      return std::max(shiftedMinTpv, std::min(prediction, shiftedMaxTpv));
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
  // Decoder only uses predictor_mode(0), which is the default setting.
  void predictor_mode(int i) {
    switch (i) {
      case 0:
        // ~ lossless16 predictor
        w0 = 0xd;
        w1 = 0xc;
        w2 = 0xc;
        w3 = 0xc;
        p1C = 16;
        p2C = 10;
        p3Ca = 7;
        p3Cb = 7;
        p3Cc = 7;
        p3Cd = 0;
        p3Ce = 0;
        break;
      case 1:
        // ~ default lossless8 predictor
        w0 = 0xd;
        w1 = 0xc;
        w2 = 0xc;
        w3 = 0xb;
        p1C = 8;
        p2C = 8;
        p3Ca = 4;
        p3Cb = 0;
        p3Cc = 3;
        p3Cd = 23;
        p3Ce = 2;
        break;
      case 2:
        // ~ west lossless8 predictor
        w0 = 0xd;
        w1 = 0xc;
        w2 = 0xd;
        w3 = 0xc;
        p1C = 10;
        p2C = 9;
        p3Ca = 7;
        p3Cb = 0;
        p3Cc = 0;
        p3Cd = 16;
        p3Ce = 9;
        break;
      case 3:
        // ~ north lossless8 predictor
        w0 = 0xd;
        w1 = 0xd;
        w2 = 0xc;
        w3 = 0xc;
        p1C = 16;
        p2C = 8;
        p3Ca = 0;
        p3Cb = 16;
        p3Cc = 0;
        p3Cd = 23;
        p3Ce = 0;
        break;
      case 4:
        // something else, because why not
        w0 = 0xd;
        w1 = 0xc;
        w2 = 0xc;
        w3 = 0xc;
        p1C = 10;
        p2C = 10;
        p3Ca = 5;
        p3Cb = 5;
        p3Cc = 5;
        p3Cd = 12;
        p3Ce = 4;
        break;
    }
  }

  bool wp_compress(const Channel& img, PaddedBytes* bytes, int nb_modes) {
    size_t xsize = img.w;
    size_t ysize = img.h;
    size_t pos = bytes->size();
    size_t best_size = 0;
    for (int mode = 0; mode < nb_modes; mode++) {
      predictor_mode(mode);
      std::vector<std::vector<Token>> tokens(1);

      for (size_t y = 0, yp = 0, yp1 = xsize; y < ysize;
           ++y, yp = xsize - yp, yp1 = xsize - yp) {
        rowImg = const_cast<pixel_type*>(img.Row(y));
        rowPrev = (y == 0 ? nullptr : img.Row(y - 1));
        rowPP = (y <= 1 ? rowPrev : img.Row(y - 2));
        for (size_t x = 0; x < xsize; ++x) {
          int maxErr;
          pixel_type prediction = predict1(x, yp + x, yp1 + x, &maxErr);
          JXL_DASSERT(0 <= maxErr && maxErr <= kNumContexts - 1);
          JXL_DASSERT(shiftedMinTpv <= prediction &&
                      prediction <= shiftedMaxTpv);
          pixel_type truePixelValue = rowImg[x];
          TokenizeHybridUint(
              maxErr,
              PackSigned(truePixelValue - ((prediction + toRound_m1) >> PBits)),
              tokens.data());
          pixel_type err = prediction - AddPBits(truePixelValue);
          UpdateSizeAndErrors(err, yp, yp1, x, prediction0, prediction1,
                              prediction2, prediction3, truePixelValue);
        }  // x
      }    // y

      BitWriter writer;
      EntropyEncodingData codes;
      std::vector<uint8_t> context_map;
      BitWriter::Allotment allotment(&writer, 52);
      if (mode != 0) {
        writer.Write(1, 0);
        writer.Write(5, p1C);
        writer.Write(5, p2C);
        writer.Write(5, p3Ca);
        writer.Write(5, p3Cb);
        writer.Write(5, p3Cc);
        writer.Write(5, p3Cd);
        writer.Write(5, p3Ce);
        writer.Write(4, w0);
        writer.Write(4, w1);
        writer.Write(4, w2);
        writer.Write(4, w3);
      } else {
        writer.Write(1, 1);
      }
      ReclaimAndCharge(&writer, &allotment, 0, nullptr);
      BuildAndEncodeHistograms(HistogramParams(), kNumContexts, tokens, &codes,
                               &context_map, &writer, 0, nullptr);
      WriteTokens(tokens[0], codes, context_map, &writer, 0, nullptr);
      writer.ZeroPadToByte();
      Span<const uint8_t> span = writer.GetSpan();
      if (mode == 0 || span.size() < best_size) {
        bytes->resize(pos + span.size());
        best_size = span.size();
        memcpy(bytes->data() + pos, span.data(), span.size());
      }
    }
    return true;
  }

  HWY_ATTR bool wp_decompress(const Span<const uint8_t> bytes,
                              size_t* bytes_pos, Channel& img) {
    if (*bytes_pos > bytes.size()) return JXL_FAILURE("out of bounds");
    size_t xsize = img.w;
    size_t ysize = img.h;
    size_t compressedSize = bytes.size() - *bytes_pos;
    const uint8_t* compressedData = bytes.data() + *bytes_pos;

    if (!xsize || !ysize) return JXL_FAILURE("invalid image size");

    predictor_mode(0);

    Status ret = true;
    {
      BitReader bitreader(Span<const uint8_t>(compressedData, compressedSize));
      BitReaderScopedCloser bitreader_closer(&bitreader, &ret);
      ANSCode code;
      std::vector<uint8_t> context_map;
      if (!bitreader.ReadBits(1)) {
        p1C = bitreader.ReadBits(5);
        p2C = bitreader.ReadBits(5);
        p3Ca = bitreader.ReadBits(5);
        p3Cb = bitreader.ReadBits(5);
        p3Cc = bitreader.ReadBits(5);
        p3Cd = bitreader.ReadBits(5);
        p3Ce = bitreader.ReadBits(5);
        w0 = bitreader.ReadBits(4);
        w1 = bitreader.ReadBits(4);
        w2 = bitreader.ReadBits(4);
        w3 = bitreader.ReadBits(4);
      }

      JXL_RETURN_IF_ERROR(DecodeHistograms(
          &bitreader, kNumContexts, ANS_MAX_ALPHA_SIZE, &code, &context_map));
      ANSSymbolReader ansreader(&code, &bitreader);

      for (size_t y = 0, yp = 0, yp1 = xsize; y < ysize;
           ++y, yp = xsize - yp, yp1 = xsize - yp) {
        rowImg = img.Row(y);
        rowPrev = (y == 0 ? nullptr : img.Row(y - 1));
        rowPP = (y <= 1 ? rowPrev : img.Row(y - 2));
        for (size_t x = 0; x < xsize; ++x) {
          int maxErr;
          pixel_type prediction = predict1(x, yp + x, yp1 + x, &maxErr);
          JXL_DASSERT(0 <= maxErr && maxErr <= kNumContexts - 1);
          JXL_DASSERT(shiftedMinTpv <= prediction &&
                      prediction <= shiftedMaxTpv);

          size_t q = ansreader.ReadHybridUint(maxErr, &bitreader, context_map);
          pixel_type truePixelValue =
              ((prediction + toRound_m1) >> PBits) + UnpackSigned(q);
          rowImg[x] = truePixelValue;
          pixel_type err = prediction - AddPBits(truePixelValue);
          UpdateSizeAndErrors(err, yp, yp1, x, prediction0, prediction1,
                              prediction2, prediction3, truePixelValue);
        }  // x
      }    // y
      if (!ansreader.CheckANSFinalState()) {
        return JXL_FAILURE("ANS final state invalid");
      }
      JXL_RETURN_IF_ERROR(bitreader.JumpToByteBoundary());
      *bytes_pos += bitreader.TotalBitsConsumed() / 8;
    }
    return ret;
  }
};

bool wp_compress(const Channel& img, PaddedBytes* bytes, int nb_modes) {
  std::unique_ptr<State> state(new State(img.w, img.h, img.minval, img.maxval));
  return state->wp_compress(img, bytes, nb_modes);
}

HWY_ATTR bool wp_decompress(const Span<const uint8_t> bytes, size_t* pos,
                            Channel& img) {
  std::unique_ptr<State> state(new State(img.w, img.h, img.minval, img.maxval));
  return state->wp_decompress(bytes, pos, img);
}

}  // namespace jxl

#endif  // JXL_MODULAR_ENCODING_WEIGHTED_PREDICT_H_
