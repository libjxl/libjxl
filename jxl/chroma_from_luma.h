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

#ifndef JXL_CHROMA_FROM_LUMA_H_
#define JXL_CHROMA_FROM_LUMA_H_

// Chroma-from-luma, computed using heuristics to determine the best linear
// model for the X and B channels from the Y channel.

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_ans.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/entropy_coder.h"
#include "jxl/field_encodings.h"
#include "jxl/fields.h"
#include "jxl/image.h"
#include "jxl/opsin_params.h"
#include "jxl/quant_weights.h"

namespace jxl {

// Tile is the rectangular grid of blocks that share color correlation
// parameters ("factor_x/b" such that residual_b = blue - Y * factor_b).
static constexpr size_t kColorTileDim = 64;

static_assert(kColorTileDim % kBlockDim == 0,
              "Color tile dim should be divisible by block dim");
static constexpr size_t kColorTileDimInBlocks = kColorTileDim / kBlockDim;

static_assert(kGroupDimInBlocks % kColorTileDimInBlocks == 0,
              "Group dim should be divisible by color tile dim");

static constexpr uint8_t kColorOffset = 127;
static constexpr uint8_t kDefaultColorFactor = 84;

static constexpr U32Enc kColorFactorDist(Val(kDefaultColorFactor), Val(256),
                                         BitsOffset(2, 8), BitsOffset(258, 12));

struct ColorCorrelationMap {
  ColorCorrelationMap() = default;
  // xsize/ysize are in pixels
  // set XYB=false to do something close to no-op cmap (needed for now since
  // cmap is mandatory)
  ColorCorrelationMap(size_t xsize, size_t ysize, bool XYB = true);

  float YtoXRatio(int32_t x_factor) const {
    return base_correlation_x_ + (x_factor - kColorOffset) * color_scale_;
  }

  float YtoBRatio(int32_t b_factor) const {
    return base_correlation_b_ + (b_factor - kColorOffset) * color_scale_;
  }

  void EncodeDC(BitWriter* writer, size_t layer, AuxOut* aux_out) const {
    BitWriter::Allotment allotment(writer, 1 + 2 * kBitsPerByte + 12 + 32);
    if (ytox_dc_ == kColorOffset && ytob_dc_ == kColorOffset &&
        color_factor_ == kDefaultColorFactor && base_correlation_x_ == 0.0f &&
        base_correlation_b_ == kYToBRatio) {
      writer->Write(1, 1);
      ReclaimAndCharge(writer, &allotment, layer, aux_out);
      return;
    }
    writer->Write(1, 0);
    JXL_CHECK(U32Coder::Write(kColorFactorDist, color_factor_, writer));
    JXL_CHECK(F16Coder::Write(base_correlation_x_, writer));
    JXL_CHECK(F16Coder::Write(base_correlation_b_, writer));
    writer->Write(kBitsPerByte, ytox_dc_);
    writer->Write(kBitsPerByte, ytob_dc_);
    ReclaimAndCharge(writer, &allotment, layer, aux_out);
  }

  Status DecodeDC(BitReader* br) {
    if (br->ReadFixedBits<1>() == 1) {
      // All default.
      return true;
    }
    SetColorFactor(U32Coder::Read(kColorFactorDist, br));
    JXL_RETURN_IF_ERROR(F16Coder::Read(br, &base_correlation_x_));
    JXL_RETURN_IF_ERROR(F16Coder::Read(br, &base_correlation_b_));
    ytox_dc_ = br->ReadFixedBits<kBitsPerByte>();
    ytob_dc_ = br->ReadFixedBits<kBitsPerByte>();
    RecomputeDCFactors();
    return true;
  }

  void SetColorFactor(uint32_t factor) {
    color_factor_ = factor;
    color_scale_ = 1.0f / color_factor_;
    RecomputeDCFactors();
  }

  void SetYToBDC(int32_t ytob_dc) {
    ytob_dc_ = ytob_dc;
    RecomputeDCFactors();
  }
  void SetYToXDC(int32_t ytox_dc) {
    ytox_dc_ = ytox_dc;
    RecomputeDCFactors();
  }

  const float* DCFactors() const { return dc_factors_; }

  void RecomputeDCFactors() {
    dc_factors_[0] = YtoXRatio(ytox_dc_);
    dc_factors_[2] = YtoBRatio(ytob_dc_);
  }

  ImageB ytox_map;
  ImageB ytob_map;

 private:
  float dc_factors_[4] = {};
  // range of factor: -1.51 to +1.52
  uint32_t color_factor_ = kDefaultColorFactor;
  float color_scale_ = 1.0f / color_factor_;
  float base_correlation_x_ = 0.0f;
  float base_correlation_b_ = kYToBRatio;
  int32_t ytox_dc_ = kColorOffset;
  int32_t ytob_dc_ = kColorOffset;
};

typedef void FindBestColorCorrelationMapFunc(
    const Image3F& opsin, const DequantMatrices& dequant,
    const AcStrategyImage* ac_strategy, const ImageI* raw_quant_field,
    const Quantizer* quantizer, ThreadPool* pool, ColorCorrelationMap* cmap);
FindBestColorCorrelationMapFunc* ChooseFindBestColorCorrelationMap();

typedef void EncodeColorMapFunc(const ColorCorrelationMap& cmap,
                                const Rect& rect, std::vector<Token>* tokens,
                                size_t base_context,
                                AuxOut* JXL_RESTRICT aux_out);
EncodeColorMapFunc* ChooseEncodeColorMap();

typedef Status DecodeColorMapFunc(BitReader* JXL_RESTRICT br,
                                  ANSSymbolReader* decoder,
                                  const std::vector<uint8_t>& context_map,
                                  ColorCorrelationMap* cmap, const Rect& rect,
                                  size_t base_context,
                                  AuxOut* JXL_RESTRICT aux_out);
DecodeColorMapFunc* ChooseDecodeColorMap();

// Declared here to avoid including predictor.h.
static constexpr size_t kCmapContexts = 24;

}  // namespace jxl

#endif  // JXL_CHROMA_FROM_LUMA_H_
