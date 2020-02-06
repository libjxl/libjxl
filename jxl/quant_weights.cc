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
#include "jxl/quant_weights.h"

#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/bits.h"
#include "jxl/base/status.h"
#include "jxl/common.h"
#include "jxl/dct.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/fields.h"
#include "jxl/image.h"
#include "jxl/modular/encoding/encoding.h"

namespace jxl {

// kQuantWeights[N * N * c + N * y + x] is the relative weight of the (x, y)
// coefficient in component c. Higher weights correspond to finer quantization
// intervals and more bits spent in encoding.

namespace {
void GetQuantWeightsDCT2(const QuantEncoding::DCT2Weights& dct2weights,
                         double* weights) {
  for (size_t c = 0; c < 3; c++) {
    size_t start = c * 64;
    weights[start] = 0xBAD;
    weights[start + 1] = weights[start + 8] = dct2weights[c][0];
    weights[start + 9] = dct2weights[c][1];
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        weights[start + y * 8 + x + 2] = dct2weights[c][2];
        weights[start + (y + 2) * 8 + x] = dct2weights[c][2];
      }
    }
    for (size_t y = 0; y < 2; y++) {
      for (size_t x = 0; x < 2; x++) {
        weights[start + (y + 2) * 8 + x + 2] = dct2weights[c][3];
      }
    }
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        weights[start + y * 8 + x + 4] = dct2weights[c][4];
        weights[start + (y + 4) * 8 + x] = dct2weights[c][4];
      }
    }
    for (size_t y = 0; y < 4; y++) {
      for (size_t x = 0; x < 4; x++) {
        weights[start + (y + 4) * 8 + x + 4] = dct2weights[c][5];
      }
    }
  }
}

void GetQuantWeightsIdentity(const QuantEncoding::IdWeights& idweights,
                             double* weights) {
  for (size_t c = 0; c < 3; c++) {
    for (int i = 0; i < 64; i++) {
      weights[64 * c + i] = idweights[c][0];
    }
    weights[64 * c + 1] = idweights[c][1];
    weights[64 * c + 8] = idweights[c][1];
    weights[64 * c + 9] = idweights[c][2];
  }
}

double Mult(double v) {
  if (v > 0) return 1 + v;
  return 1 / (1 - v);
}

double Interpolate(double pos, double max, const double* array, size_t len) {
  double scaled_pos = pos * (len - 1) / max;
  size_t idx = scaled_pos;
  JXL_ASSERT(idx + 1 < len);
  double a = array[idx];
  double b = array[idx + 1];
  return a * pow(b / a, scaled_pos - idx);
}

// Computes quant weights for a COLS*ROWS-sized transform, using num_bands
// eccentricity bands and num_ebands eccentricity bands. If print_mode is 1,
// prints the resulting matrix; if print_mode is 2, prints the matrix in a
// format suitable for a 3d plot with gnuplot.
template <size_t print_mode = 0>
Status GetQuantWeights(
    size_t ROWS, size_t COLS,
    const DctQuantWeightParams::DistanceBandsArray& distance_bands,
    size_t num_bands, double* out) {
  for (size_t c = 0; c < 3; c++) {
    if (print_mode) {
      fprintf(stderr, "Channel %zu\n", c);
    }
    double bands[DctQuantWeightParams::kMaxDistanceBands] = {
        distance_bands[c][0]};
    for (size_t i = 1; i < num_bands; i++) {
      bands[i] = bands[i - 1] * Mult(distance_bands[c][i]);
      if (bands[i] < 0) return JXL_FAILURE("Invalid distance bands");
    }
    for (size_t y = 0; y < ROWS; y++) {
      for (size_t x = 0; x < COLS; x++) {
        double dx = 1.0 * x / (COLS - 1);
        double dy = 1.0 * y / (ROWS - 1);
        double distance = std::sqrt(dx * dx + dy * dy);
        double weight =
            num_bands == 1
                ? bands[0]
                : Interpolate(distance, std::sqrt(2) + 1e-6, bands, num_bands);

        if (print_mode == 1) {
          fprintf(stderr, "%15.12f, ", weight);
        }
        if (print_mode == 2) {
          fprintf(stderr, "%zu %zu %15.12f\n", x, y, weight);
        }
        out[c * COLS * ROWS + y * COLS + x] = weight;
      }
      if (print_mode) fprintf(stderr, "\n");
      if (print_mode == 1) fprintf(stderr, "\n");
    }
    if (print_mode) fprintf(stderr, "\n");
  }
  return true;
}

Status EncodeDctParams(const DctQuantWeightParams& params, BitWriter* writer) {
  JXL_ASSERT(params.num_distance_bands >= 1);
  writer->Write(DctQuantWeightParams::kLog2MaxDistanceBands,
                params.num_distance_bands - 1);
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params.num_distance_bands; i++) {
      JXL_RETURN_IF_ERROR(F16Coder::Write(
          params.distance_bands[c][i] * (i == 0 ? (1 / 64.0f) : 1.0f), writer));
    }
  }
  return true;
}

HWY_ATTR Status DecodeDctParams(BitReader* br, DctQuantWeightParams* params) {
  params->num_distance_bands =
      br->ReadFixedBits<DctQuantWeightParams::kLog2MaxDistanceBands>() + 1;
  for (size_t c = 0; c < 3; c++) {
    for (size_t i = 0; i < params->num_distance_bands; i++) {
      JXL_RETURN_IF_ERROR(F16Coder::Read(br, &params->distance_bands[c][i]));
    }
    params->distance_bands[c][0] *= 64.0f;
  }
  return true;
}

Status EncodeQuant(const QuantEncoding& encoding, size_t idx, size_t size_x,
                   size_t size_y, BitWriter* writer) {
  writer->Write(kLog2NumQuantModes, encoding.mode);
  size_x *= kBlockDim;
  size_y *= kBlockDim;
  switch (encoding.mode) {
    case QuantEncoding::kQuantModeLibrary: {
      writer->Write(kCeilLog2NumPredefinedTables, encoding.predefined);
      break;
    }
    case QuantEncoding::kQuantModeID: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 3; i++) {
          JXL_RETURN_IF_ERROR(
              F16Coder::Write(encoding.idweights[c][i] * (1.0f / 64), writer));
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 6; i++) {
          JXL_RETURN_IF_ERROR(F16Coder::Write(
              encoding.dct2weights[c][i] * (1.0f / 64), writer));
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT4X8: {
      for (size_t c = 0; c < 3; c++) {
        JXL_RETURN_IF_ERROR(
            F16Coder::Write(encoding.dct4x8multipliers[c], writer));
      }
      JXL_RETURN_IF_ERROR(EncodeDctParams(encoding.dct_params, writer));
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 2; i++) {
          JXL_RETURN_IF_ERROR(
              F16Coder::Write(encoding.dct4multipliers[c][i], writer));
        }
      }
      JXL_RETURN_IF_ERROR(EncodeDctParams(encoding.dct_params, writer));
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      JXL_RETURN_IF_ERROR(EncodeDctParams(encoding.dct_params, writer));
      break;
    }
    case QuantEncoding::kQuantModeRAW: {
      JXL_ASSERT(encoding.qraw.qtable != nullptr);
      JXL_ASSERT(size_x * size_y * 3 == encoding.qraw.qtable->size());
      writer->Write(3, encoding.qraw.qtable_den_shift);
      writer->ZeroPadToByte();
      Image3I img(size_x, size_y);
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < size_y; y++) {
          int* JXL_RESTRICT row = img.PlaneRow(c, y);
          for (size_t x = 0; x < size_x; x++) {
            row[x] =
                (*encoding.qraw.qtable)[c * size_x * size_y + y * size_x + x];
          }
        }
      }
      modular_options cfopts;
      set_default_modular_options(cfopts);
      cfopts.nb_repeats = 0;
      cfopts.entropy_coder = 2;
      PaddedBytes enc;
      modular_rect_compress_3(img, Rect(img), &enc, &cfopts, /*loss=*/0);
      writer->ZeroPadToByte();
      *writer += enc;
      break;
    }
    case QuantEncoding::kQuantModeAFV: {
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 9; i++) {
          JXL_RETURN_IF_ERROR(F16Coder::Write(
              encoding.afv_weights[c][i] * (i < 6 ? 1.0f / 64 : 1.0f), writer));
        }
        JXL_RETURN_IF_ERROR(EncodeDctParams(encoding.dct_params, writer));
        JXL_RETURN_IF_ERROR(
            EncodeDctParams(encoding.dct_params_afv_4x4, writer));
      }
      break;
    }
  }
  return true;
}

HWY_ATTR Status Decode(BitReader* br, QuantEncoding* encoding,
                       size_t required_size_x, size_t required_size_y,
                       size_t idx) {
  size_t required_size = required_size_x * required_size_y;
  required_size_x *= kBlockDim;
  required_size_y *= kBlockDim;
  int mode = br->ReadFixedBits<kLog2NumQuantModes>();
  switch (mode) {
    case QuantEncoding::kQuantModeLibrary: {
      encoding->predefined = br->ReadFixedBits<kCeilLog2NumPredefinedTables>();
      if (encoding->predefined >= kNumPredefinedTables) {
        return JXL_FAILURE("Invalid predefined table");
      }
      break;
    }
    case QuantEncoding::kQuantModeID: {
      if (required_size != 1) return JXL_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 3; i++) {
          JXL_RETURN_IF_ERROR(F16Coder::Read(br, &encoding->idweights[c][i]));
          encoding->idweights[c][i] *= 64;
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      if (required_size != 1) return JXL_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 6; i++) {
          JXL_RETURN_IF_ERROR(F16Coder::Read(br, &encoding->dct2weights[c][i]));
          encoding->dct2weights[c][i] *= 64;
        }
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT4X8: {
      if (required_size != 1) return JXL_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        JXL_RETURN_IF_ERROR(
            F16Coder::Read(br, &encoding->dct4x8multipliers[c]));
      }
      JXL_RETURN_IF_ERROR(DecodeDctParams(br, &encoding->dct_params));
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      if (required_size != 1) return JXL_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 2; i++) {
          JXL_RETURN_IF_ERROR(
              F16Coder::Read(br, &encoding->dct4multipliers[c][i]));
        }
      }
      JXL_RETURN_IF_ERROR(DecodeDctParams(br, &encoding->dct_params));
      break;
    }
    case QuantEncoding::kQuantModeAFV: {
      if (required_size != 1) return JXL_FAILURE("Invalid mode");
      for (size_t c = 0; c < 3; c++) {
        for (size_t i = 0; i < 9; i++) {
          JXL_RETURN_IF_ERROR(F16Coder::Read(br, &encoding->afv_weights[c][i]));
        }
        for (size_t i = 0; i < 6; i++) {
          encoding->afv_weights[c][i] *= 64;
        }
        JXL_RETURN_IF_ERROR(DecodeDctParams(br, &encoding->dct_params));
        JXL_RETURN_IF_ERROR(DecodeDctParams(br, &encoding->dct_params_afv_4x4));
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      JXL_RETURN_IF_ERROR(DecodeDctParams(br, &encoding->dct_params));
      break;
    }
    case QuantEncoding::kQuantModeRAW: {
      encoding->qraw.qtable_den_shift = br->ReadFixedBits<3>();
      Image3I img(required_size_x, required_size_y);
      JXL_RETURN_IF_ERROR(br->JumpToByteBoundary());
      JXL_RETURN_IF_ERROR(br->AllReadsWithinBounds());
      size_t pos = 0;
      const Span<const uint8_t> compressed = br->GetSpan();
      if (!modular_rect_decompress_3(compressed, &pos, &img, Rect(img)))
        return JXL_FAILURE("Failed to decode DC");
      br->SkipBits(pos * kBitsPerByte);
      if (!encoding->qraw.qtable) {
        encoding->qraw.qtable = new std::vector<int>();
      }
      encoding->qraw.qtable->resize(required_size_x * required_size_y * 3);
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < required_size_y; y++) {
          int* JXL_RESTRICT row = img.PlaneRow(c, y);
          for (size_t x = 0; x < required_size_x; x++) {
            (*encoding->qraw.qtable)[c * required_size_x * required_size_y +
                                     y * required_size_x + x] = row[x];
            if (row[x] <= 0) {
              return JXL_FAILURE("Invalid raw quantization table");
            }
          }
        }
      }
      JXL_RETURN_IF_ERROR(br->JumpToByteBoundary());
      break;
    }
    default:
      return JXL_FAILURE("Invalid quantization table encoding");
  }
  encoding->mode = QuantEncoding::Mode(mode);
  return true;
}

Status ComputeQuantTable(const QuantEncoding& encoding, float* table,
                         size_t* offsets, size_t table_num,
                         DequantMatrices::QuantTable kind, size_t* pos) {
  double weights[3 * kMaxQuantTableSize];
  double numerators[kMaxQuantTableSize];

  constexpr size_t N = kBlockDim;
  const float* idct4_scales = IDCTScales<N / 2>();
  const float* idct_scales = IDCTScales<N>();
  const float* idct16_scales = IDCTScales<2 * N>();
  const float* idct32_scales = IDCTScales<4 * N>();
  size_t wrows = 8, wcols = 8;
  size_t num = 0;
  switch (kind) {
    case DequantMatrices::DCT: {
      num = kDCTBlockSize;
      wrows = 8;
      wcols = 8;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % N;
        const size_t y = i / N;
        const float idct_scale = idct_scales[x] * idct_scales[y] * 8;
        numerators[i] = idct_scale;
      }
      break;
    }
    case DequantMatrices::DCT16X16: {
      num = 4 * kDCTBlockSize;
      wrows = 16;
      wcols = 16;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % (2 * N);
        const size_t y = i / (2 * N);
        const float idct_scale = idct16_scales[x] * idct16_scales[y] * 16;
        numerators[i] = idct_scale;
      }
      break;
    }
    case DequantMatrices::DCT32X32: {
      num = 16 * kDCTBlockSize;
      wrows = 32;
      wcols = 32;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % (4 * N);
        const size_t y = i / (4 * N);
        const float idct_scale = idct32_scales[x] * idct32_scales[y] * 32;
        numerators[i] = idct_scale;
      }
      break;
    }
    case DequantMatrices::DCT8X16: {
      wrows = 8;
      wcols = 16;
      num = 2 * kDCTBlockSize;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % (2 * N);
        const size_t y = i / (2 * N);
        const float idct_scale =
            idct16_scales[x] * idct_scales[y] * std::sqrt(8 * 16);
        numerators[i] = idct_scale;
      }
      break;
    }
    case DequantMatrices::DCT8X32: {
      num = 4 * kDCTBlockSize;
      wrows = 8;
      wcols = 32;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % (4 * N);
        const size_t y = i / (4 * N);
        const float idct_scale = idct32_scales[x] * idct_scales[y] * 16;
        numerators[i] = idct_scale;
      }
      break;
    }
    case DequantMatrices::DCT16X32: {
      num = 8 * kDCTBlockSize;
      wrows = 16;
      wcols = 32;
      for (size_t i = 0; i < num; i++) {
        const size_t x = i % (4 * N);
        const size_t y = i / (4 * N);
        const float idct_scale =
            idct32_scales[x] * idct16_scales[y] * std::sqrt(16 * 32);
        numerators[i] = idct_scale;
      }
      break;
    }
    case DequantMatrices::DCT4X4: {
      num = kDCTBlockSize;
      for (size_t i = 0; i < N * N; i++) {
        const size_t x = i % N;
        const size_t y = i / N;
        float idct_scale = idct4_scales[x / 2] * idct4_scales[y / 2] * 4;
        numerators[i] = idct_scale;
      }
      break;
    }
    case DequantMatrices::DCT4X8: {
      num = kDCTBlockSize;
      for (size_t i = 0; i < N * N; i++) {
        const size_t x = i % N;
        const size_t y = i / N;
        float idct_scale =
            idct_scales[x] * idct4_scales[y / 2] * std::sqrt(4 * 8);
        numerators[i] = idct_scale;
      }
      break;
    }
    case DequantMatrices::IDENTITY:
    case DequantMatrices::DCT2X2:
      num = kDCTBlockSize;
      std::fill_n(numerators, kDCTBlockSize, 1.0);
      break;
    case DequantMatrices::AFV0: {
      num = kDCTBlockSize;
      for (size_t i = 0; i < N * N; i++) {
        const size_t x = i % N;
        const size_t y = i / N;
        if (y & 1) {
          float idct_scale =
              idct_scales[x] * idct4_scales[y / 2] * std::sqrt(4 * 8);
          numerators[i] = idct_scale;
        } else {
          if (x & 1) {
            float idct_scale = idct4_scales[x / 2] * idct4_scales[y / 2] * 4;
            numerators[i] = idct_scale;
          } else {
            numerators[i] = 1;
          }
        }
      }
      break;
    }
    default: {
      JXL_ABORT("Invalid AC strategy value");
    }
  }

  switch (encoding.mode) {
    case QuantEncoding::kQuantModeLibrary: {
      // Library and copy quant encoding should get replaced by the actual
      // parameters by the caller.
      JXL_ASSERT(false);
      break;
    }
    case QuantEncoding::kQuantModeID: {
      JXL_ASSERT(num == kDCTBlockSize);
      GetQuantWeightsIdentity(encoding.idweights, weights);
      break;
    }
    case QuantEncoding::kQuantModeDCT2: {
      JXL_ASSERT(num == kDCTBlockSize);
      GetQuantWeightsDCT2(encoding.dct2weights, weights);
      break;
    }
    case QuantEncoding::kQuantModeDCT4: {
      JXL_ASSERT(num == kDCTBlockSize);
      double weights4x4[3 * 4 * 4];
      // Always use 4x4 GetQuantWeights for DCT4 quantization tables.
      JXL_RETURN_IF_ERROR(
          GetQuantWeights(4, 4, encoding.dct_params.distance_bands,
                          encoding.dct_params.num_distance_bands, weights4x4));
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < kBlockDim; y++) {
          for (size_t x = 0; x < kBlockDim; x++) {
            weights[c * num + y * kBlockDim + x] =
                weights4x4[c * 16 + (y / 2) * 4 + (x / 2)];
          }
        }
        weights[c * num + 1] /= encoding.dct4multipliers[c][0];
        weights[c * num + N] /= encoding.dct4multipliers[c][0];
        weights[c * num + N + 1] /= encoding.dct4multipliers[c][1];
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT4X8: {
      JXL_ASSERT(num == kDCTBlockSize);
      double weights4x8[3 * 4 * 8];
      // Always use 4x8 GetQuantWeights for DCT4X8 quantization tables.
      JXL_RETURN_IF_ERROR(
          GetQuantWeights(4, 8, encoding.dct_params.distance_bands,
                          encoding.dct_params.num_distance_bands, weights4x8));
      for (size_t c = 0; c < 3; c++) {
        for (size_t y = 0; y < kBlockDim; y++) {
          for (size_t x = 0; x < kBlockDim; x++) {
            weights[c * num + y * kBlockDim + x] =
                weights4x8[c * 32 + (y / 2) * 8 + x];
          }
        }
        weights[c * num + N] /= encoding.dct4x8multipliers[c];
      }
      break;
    }
    case QuantEncoding::kQuantModeDCT: {
      JXL_RETURN_IF_ERROR(
          GetQuantWeights(wrows, wcols, encoding.dct_params.distance_bands,
                          encoding.dct_params.num_distance_bands, weights));
      break;
    }
    case QuantEncoding::kQuantModeRAW: {
      float den = 1 << encoding.qraw.qtable_den_shift;
      if (!encoding.qraw.qtable || encoding.qraw.qtable->size() != 3 * num) {
        return JXL_FAILURE("Invalid table encoding");
      }
      for (size_t i = 0; i < 3 * num; i++) {
        weights[i] = 8.0 / (den * (*encoding.qraw.qtable)[i]);
      }
      break;
    }
    case QuantEncoding::kQuantModeAFV: {
      constexpr float kFreqs[] = {
          0xBAD,
          0xBAD,
          0.8517778890324296,
          5.37778436506804,
          0xBAD,
          0xBAD,
          4.734747904497923,
          5.449245381693219,
          1.6598270267479331,
          4,
          7.275749096817861,
          10.423227632456525,
          2.662932286148962,
          7.630657783650829,
          8.962388608184032,
          12.97166202570235,
      };

      double weights4x8[3 * 4 * 8];
      JXL_RETURN_IF_ERROR((
          GetQuantWeights(4, 8, encoding.dct_params.distance_bands,
                          encoding.dct_params.num_distance_bands, weights4x8)));
      double weights4x4[3 * 4 * 4];
      JXL_RETURN_IF_ERROR((GetQuantWeights(
          4, 4, encoding.dct_params_afv_4x4.distance_bands,
          encoding.dct_params_afv_4x4.num_distance_bands, weights4x4)));

      constexpr float lo = 0.8517778890324296;
      constexpr float hi = 12.97166202570235 - lo + 1e-6;
      for (size_t c = 0; c < 3; c++) {
        double bands[4];
        bands[0] = encoding.afv_weights[c][5];
        if (bands[0] < 0) return JXL_FAILURE("Invalid AFV bands");
        for (size_t i = 1; i < 4; i++) {
          bands[i] = bands[i - 1] * Mult(encoding.afv_weights[c][i + 5]);
          if (bands[i] < 0) return JXL_FAILURE("Invalid AFV bands");
        }
        size_t start = c * 64;
        auto set_weight = [&start, &weights](size_t x, size_t y, float val) {
          weights[start + y * 8 + x] = val;
        };
        weights[start] = 1;  // Not used, but causes MSAN error otherwise.
        // Weights for (0, 1) and (1, 0).
        set_weight(0, 1, encoding.afv_weights[c][0]);
        set_weight(1, 0, encoding.afv_weights[c][1]);
        // AFV special weights for 3-pixel corner.
        set_weight(0, 2, encoding.afv_weights[c][2]);
        set_weight(2, 0, encoding.afv_weights[c][3]);
        set_weight(2, 2, encoding.afv_weights[c][4]);

        // All other AFV weights.
        for (size_t y = 0; y < 4; y++) {
          for (size_t x = 0; x < 4; x++) {
            if (x < 2 && y < 2) continue;
            float val = Interpolate(kFreqs[y * 4 + x] - lo, hi, bands, 4);
            set_weight(2 * x, 2 * y, val);
          }
        }

        // Put 4x8 weights in odd rows, except (1, 0).
        for (size_t y = 0; y < kBlockDim / 2; y++) {
          for (size_t x = 0; x < kBlockDim; x++) {
            if (x == 0 && y == 0) continue;
            weights[c * num + (2 * y + 1) * kBlockDim + x] =
                weights4x8[c * 32 + y * 8 + x];
          }
        }
        // Put 4x4 weights in even rows / odd columns, except (0, 1).
        for (size_t y = 0; y < kBlockDim / 2; y++) {
          for (size_t x = 0; x < kBlockDim / 2; x++) {
            if (x == 0 && y == 0) continue;
            weights[c * num + (2 * y) * kBlockDim + 2 * x + 1] =
                weights4x4[c * 16 + y * 4 + x];
          }
        }
      }
      break;
    }
  }
  for (size_t c = 0; c < 3; c++) {
    offsets[table_num * 3 + c] = *pos;
    for (size_t i = 0; i < num; i++) {
      double val = numerators[i] / weights[c * num + i];
      if (val > std::numeric_limits<float>::max() || val < 0) {
        return JXL_FAILURE("Invalid quantization table");
      }
      table[(*pos)++] = val;
    }
  }
  return true;
}

}  // namespace

// These definitions are needed before C++17.
constexpr size_t DequantMatrices::required_size_[];
constexpr size_t DequantMatrices::required_size_x_[];
constexpr size_t DequantMatrices::required_size_y_[];
constexpr DequantMatrices::QuantTable DequantMatrices::kQuantTable[];

Status DequantMatrices::Encode(BitWriter* writer, size_t layer,
                               AuxOut* aux_out) const {
  bool all_default = true;
  for (size_t i = 0; i < encodings_.size(); i++) {
    if (encodings_[i].mode != QuantEncoding::kQuantModeLibrary ||
        encodings_[i].predefined != 0) {
      all_default = false;
    }
  }
  // TODO(janwas): better bound
  BitWriter::Allotment allotment(writer, 512 * 1024);
  writer->Write(1, all_default);
  if (!all_default) {
    for (size_t i = 0; i < encodings_.size(); i++) {
      JXL_RETURN_IF_ERROR(EncodeQuant(encodings_[i], i, required_size_x_[i],
                                      required_size_y_[i], writer));
    }
  }
  ReclaimAndCharge(writer, &allotment, layer, aux_out);
  return true;
}

Status DequantMatrices::EncodeDC(BitWriter* writer, size_t layer,
                                 AuxOut* aux_out) const {
  bool all_default = true;
  for (size_t c = 0; c < 3; c++) {
    if (dc_quant_[c] != kDCQuant[c]) {
      all_default = false;
    }
  }
  BitWriter::Allotment allotment(writer, 1 + sizeof(float) * kBitsPerByte * 3);
  writer->Write(1, all_default);
  if (!all_default) {
    for (size_t c = 0; c < 3; c++) {
      JXL_RETURN_IF_ERROR(F16Coder::Write(dc_quant_[c] * 128.0f, writer));
    }
  }
  ReclaimAndCharge(writer, &allotment, layer, aux_out);
  return true;
}

HWY_ATTR Status DequantMatrices::Decode(BitReader* br) {
  size_t all_default = br->ReadBits(1);
  size_t num_tables = all_default ? 0 : kNum;
  encodings_.clear();
  encodings_.resize(kNum, QuantEncoding::Library(0));
  for (size_t i = 0; i < num_tables; i++) {
    JXL_RETURN_IF_ERROR(jxl::Decode(br, &encodings_[i],
                                    required_size_x_[i % kNum],
                                    required_size_y_[i % kNum], i));
  }
  return DequantMatrices::Compute();
}

HWY_ATTR Status DequantMatrices::DecodeDC(BitReader* br) {
  bool all_default = br->ReadBits(1);
  if (!all_default) {
    for (size_t c = 0; c < 3; c++) {
      JXL_RETURN_IF_ERROR(F16Coder::Read(br, &dc_quant_[c]));
      dc_quant_[c] *= 1.0f / 128.0f;
      if (dc_quant_[c] == 0.)
        return JXL_FAILURE("Invalid dc_quant coefficient 0.");
      inv_dc_quant_[c] = 1.0f / dc_quant_[c];
    }
  }
  return true;
}

constexpr float V(double v) { return static_cast<float>(v); }

namespace {
struct DequantMatricesLibraryDef {
  // DCT8
  static constexpr const QuantEncodingInternal DCT() {
    return QuantEncodingInternal::DCT(DctQuantWeightParams({{{
                                                                 V(3150.0),
                                                                 V(0.0),
                                                                 V(-0.4),
                                                                 V(-0.4),
                                                                 V(-0.4),
                                                                 V(-2.0),
                                                             },
                                                             {
                                                                 V(560.0),
                                                                 V(0.0),
                                                                 V(-0.3),
                                                                 V(-0.3),
                                                                 V(-0.3),
                                                                 V(-0.3),
                                                             },
                                                             {
                                                                 V(512.0),
                                                                 V(-2.0),
                                                                 V(-1.0),
                                                                 V(0.0),
                                                                 V(-1.0),
                                                                 V(-2.0),
                                                             }}},
                                                           6));
  }

  // Identity
  static constexpr const QuantEncodingInternal IDENTITY() {
    return QuantEncodingInternal::Identity({{{
                                                 V(280.0),
                                                 V(3160.0),
                                                 V(3160.0),
                                             },
                                             {
                                                 V(60.0),
                                                 V(864.0),
                                                 V(864.0),
                                             },
                                             {
                                                 V(18.0),
                                                 V(200.0),
                                                 V(200.0),
                                             }}});
  }

  // DCT2
  static constexpr const QuantEncodingInternal DCT2X2() {
    return QuantEncodingInternal::DCT2({{{
                                             V(3840.0),
                                             V(2560.0),
                                             V(1280.0),
                                             V(640.0),
                                             V(480.0),
                                             V(300.0),
                                         },
                                         {
                                             V(960.0),
                                             V(640.0),
                                             V(320.0),
                                             V(180.0),
                                             V(140.0),
                                             V(120.0),
                                         },
                                         {
                                             V(640.0),
                                             V(320.0),
                                             V(128.0),
                                             V(64.0),
                                             V(32.0),
                                             V(16.0),
                                         }}});
  }

  // DCT4 (quant_kind 3)
  static constexpr const QuantEncodingInternal DCT4X4() {
    return QuantEncodingInternal::DCT4(DctQuantWeightParams({{{
                                                                  V(2200.0),
                                                                  V(0.0),
                                                                  V(0.0),
                                                                  V(0.0),
                                                              },
                                                              {
                                                                  V(392.0),
                                                                  V(0.0),
                                                                  V(0.0),
                                                                  V(0.0),
                                                              },
                                                              {
                                                                  V(112.0),
                                                                  V(-0.25),
                                                                  V(-0.25),
                                                                  V(-0.5),
                                                              }}},
                                                            4),
                                       /* kMul */
                                       {{{
                                             V(1.0),
                                             V(1.0),
                                         },
                                         {
                                             V(1.0),
                                             V(1.0),
                                         },
                                         {
                                             V(1.0),
                                             V(1.0),
                                         }}});
  }

  // DCT16
  static constexpr const QuantEncodingInternal DCT16X16() {
    return QuantEncodingInternal::DCT(
        DctQuantWeightParams({{{
                                   V(8996.8725711814115328),
                                   V(-1.3000777393353804),
                                   V(-0.49424529824571225),
                                   V(-0.439093774457103443),
                                   V(-0.6350101832695744),
                                   V(-0.90177264050827612),
                                   V(-1.6162099239887414),
                               },
                               {
                                   V(3191.48366296844234752),
                                   V(-0.67424582104194355),
                                   V(-0.80745813428471001),
                                   V(-0.44925837484843441),
                                   V(-0.35865440981033403),
                                   V(-0.31322389111877305),
                                   V(-0.37615025315725483),
                               },
                               {
                                   V(1157.50408145487200256),
                                   V(-2.0531423165804414),
                                   V(-1.4),
                                   V(-0.50687130033378396),
                                   V(-0.42708730624733904),
                                   V(-1.4856834539296244),
                                   V(-4.9209142884401604),
                               }}},
                             7));
  }

  // DCT32
  static constexpr const QuantEncodingInternal DCT32X32() {
    return QuantEncodingInternal::DCT(
        DctQuantWeightParams({{{
                                   V(15718.40830982518931456),
                                   V(-1.025),
                                   V(-0.98),
                                   V(-0.9012),
                                   V(-0.4),
                                   V(-0.48819395464),
                                   V(-0.421064),
                                   V(-0.27),
                               },
                               {
                                   V(7305.7636810695983104),
                                   V(-0.8041958212306401),
                                   V(-0.7633036457487539),
                                   V(-0.55660379990111464),
                                   V(-0.49785304658857626),
                                   V(-0.43699592683512467),
                                   V(-0.40180866526242109),
                                   V(-0.27321683125358037),
                               },
                               {
                                   V(3803.53173721215041536),
                                   V(-3.060733579805728),
                                   V(-2.0413270132490346),
                                   V(-2.0235650159727417),
                                   V(-0.5495389509954993),
                                   V(-0.4),
                                   V(-0.4),
                                   V(-0.3),
                               }}},
                             8));
  }

  // DCT16X8
  static constexpr const QuantEncodingInternal DCT8X16() {
    return QuantEncodingInternal::DCT(
        DctQuantWeightParams({{{
                                   V(7240.7734393502),
                                   V(-0.7),
                                   V(-0.7),
                                   V(-0.2),
                                   V(-0.2),
                                   V(-0.2),
                                   V(-0.5),
                               },
                               {
                                   V(1448.15468787004),
                                   V(-0.5),
                                   V(-0.5),
                                   V(-0.5),
                                   V(-0.2),
                                   V(-0.2),
                                   V(-0.2),
                               },
                               {
                                   V(506.854140754517),
                                   V(-1.4),
                                   V(-0.2),
                                   V(-0.5),
                                   V(-0.5),
                                   V(-1.5),
                                   V(-3.6),
                               }}},
                             7));
  }

  // DCT32X8
  static constexpr const QuantEncodingInternal DCT8X32() {
    return QuantEncodingInternal::DCT(
        DctQuantWeightParams({{{
                                   V(16283.2494710648897),
                                   V(-1.7812845336559429),
                                   V(-1.6309059012653515),
                                   V(-1.0382179034313539),
                                   V(-0.85),
                                   V(-0.7),
                                   V(-0.9),
                                   V(-1.2360638576849587),
                               },
                               {
                                   V(5089.15750884921511936),
                                   V(-0.320049391452786891),
                                   V(-0.35362849922161446),
                                   V(-0.30340000000000003),
                                   V(-0.61),
                                   V(-0.5),
                                   V(-0.5),
                                   V(-0.6),
                               },
                               {
                                   V(3397.77603275308720128),
                                   V(-0.321327362693153371),
                                   V(-0.34507619223117997),
                                   V(-0.70340000000000003),
                                   V(-0.9),
                                   V(-1.0),
                                   V(-1.0),
                                   V(-1.1754605576265209),
                               }}},
                             8));
  }

  // DCT32X16
  static constexpr const QuantEncodingInternal DCT16X32() {
    return QuantEncodingInternal::DCT(
        DctQuantWeightParams({{{
                                   V(13844.97076442300573),
                                   V(-0.97113799999999995),
                                   V(-0.658),
                                   V(-0.42026),
                                   V(-0.22712),
                                   V(-0.2206),
                                   V(-0.226),
                                   V(-0.6),
                               },
                               {
                                   V(4798.964084220744293),
                                   V(-0.61125308982767057),
                                   V(-0.83770786552491361),
                                   V(-0.79014862079498627),
                                   V(-0.2692727459704829),
                                   V(-0.38272769465388551),
                                   V(-0.22924222653091453),
                                   V(-0.20719098826199578),
                               },
                               {
                                   V(1807.236946760964614),
                                   V(-1.2),
                                   V(-1.2),
                                   V(-0.7),
                                   V(-0.7),
                                   V(-0.7),
                                   V(-0.4),
                                   V(-0.5),
                               }}},
                             8));
  }

  // DCT4X8 and 8x4
  static constexpr const QuantEncodingInternal DCT4X8() {
    return QuantEncodingInternal::DCT4X8(
        DctQuantWeightParams({{
                                 {
                                     V(2198.050556016380522),
                                     V(-0.96269623020744692),
                                     V(-0.76194253026666783),
                                     V(-0.6551140670773547),
                                 },
                                 {
                                     V(764.3655248643528689),
                                     V(-0.92630200888366945),
                                     V(-0.9675229603596517),
                                     V(-0.27845290869168118),
                                 },
                                 {
                                     V(527.107573587542228),
                                     V(-1.4594385811273854),
                                     V(-1.450082094097871593),
                                     V(-1.5843722511996204),
                                 },
                             }},
                             4),
        /* kMuls */
        {{
            V(1.0),
            V(1.0),
            V(1.0),
        }});
  }

  // AFV
  static const QuantEncodingInternal AFV0() {
    return QuantEncodingInternal::AFV(DCT4X8().dct_params, DCT4X4().dct_params,
                                      {{{
                                            // 4x4/4x8 DC tendency.
                                            V(3072.0),
                                            V(3072.0),
                                            // AFV corner.
                                            V(256.0),
                                            V(256.0),
                                            V(256.0),
                                            // AFV high freqs.
                                            V(414.0),
                                            V(0.0),
                                            V(0.0),
                                            V(0.0),
                                        },
                                        {
                                            // 4x4/4x8 DC tendency.
                                            V(1024.0),
                                            V(1024.0),
                                            // AFV corner.
                                            V(50),
                                            V(50),
                                            V(50),
                                            // AFV high freqs.
                                            V(58.0),
                                            V(0.0),
                                            V(0.0),
                                            V(0.0),
                                        },
                                        {
                                            // 4x4/4x8 DC tendency.
                                            V(384.0),
                                            V(384.0),
                                            // AFV corner.
                                            V(12.0),
                                            V(12.0),
                                            V(12.0),
                                            // AFV high freqs.
                                            V(22.0),
                                            V(-0.25),
                                            V(-0.25),
                                            V(-0.25),
                                        }}});
  }
};
}  // namespace

const DequantMatrices::DequantLibraryInternal DequantMatrices::LibraryInit() {
  static_assert(kNum == 11,
                "Update this function when adding new quantization kinds.");
  static_assert(kNumPredefinedTables == 1,
                "Update this function when adding new quantization matrices to "
                "the library.");

  // The library and the indices need to be kept in sync manually.
  static_assert(0 == DCT, "Update the DequantLibrary array below.");
  static_assert(1 == IDENTITY, "Update the DequantLibrary array below.");
  static_assert(2 == DCT2X2, "Update the DequantLibrary array below.");
  static_assert(3 == DCT4X4, "Update the DequantLibrary array below.");
  static_assert(4 == DCT16X16, "Update the DequantLibrary array below.");
  static_assert(5 == DCT32X32, "Update the DequantLibrary array below.");
  static_assert(6 == DCT8X16, "Update the DequantLibrary array below.");
  static_assert(7 == DCT8X32, "Update the DequantLibrary array below.");
  static_assert(8 == DCT16X32, "Update the DequantLibrary array below.");
  static_assert(9 == DCT4X8, "Update the DequantLibrary array below.");
  static_assert(10 == AFV0, "Update the DequantLibrary array below.");
  return DequantMatrices::DequantLibraryInternal{
      DequantMatricesLibraryDef::DCT(),
      DequantMatricesLibraryDef::IDENTITY(),
      DequantMatricesLibraryDef::DCT2X2(),
      DequantMatricesLibraryDef::DCT4X4(),
      DequantMatricesLibraryDef::DCT16X16(),
      DequantMatricesLibraryDef::DCT32X32(),
      DequantMatricesLibraryDef::DCT8X16(),
      DequantMatricesLibraryDef::DCT8X32(),
      DequantMatricesLibraryDef::DCT16X32(),
      DequantMatricesLibraryDef::DCT4X8(),
      DequantMatricesLibraryDef::AFV0(),
  };
}

namespace {
const DequantMatrices::DequantLibraryInternal kDequantLibrary =
    DequantMatrices::LibraryInit();
}  // namespace

const QuantEncoding* DequantMatrices::Library() {
  // Downcast the result to a const QuantEncoding* from QuantEncodingInternal*
  // since the subclass (QuantEncoding) doesn't add any new members and users
  // will need to upcast to QuantEncodingInternal to access the members of that
  // class. This allows to have kDequantLibrary as a constexpr value while still
  // allowing to create QuantEncoding::RAW() instances that use std::vector in
  // C++11.
  return reinterpret_cast<const QuantEncoding*>(kDequantLibrary.data());
}

Status DequantMatrices::Compute() {
  size_t pos = 0;

  const QuantEncoding* library = Library();

  // Avoid changing encodings_.
  auto encodings = encodings_;

  std::vector<size_t> offsets(kNum * 3);

  for (size_t table = 0; table < encodings.size(); table++) {
    if (encodings[table].mode == QuantEncoding::kQuantModeLibrary) {
      encodings[table] = library[encodings[table].predefined *
                                     AcStrategy::kNumValidStrategies +
                                 table % kNum];
    }
    JXL_RETURN_IF_ERROR(ComputeQuantTable(encodings[table], table_,
                                          offsets.data(), table,
                                          QuantTable(table % kNum), &pos));
  }

  JXL_ASSERT(pos == kTotalTableSize);

  size_ = pos;
  for (size_t i = 0; i < pos; i++) {
    inv_table_[i] = 1.0f / table_[i];
  }
  for (size_t i = 0; i < AcStrategy::kNumValidStrategies; i++) {
    for (size_t c = 0; c < 3; c++) {
      table_offsets_[i * 3 + c] = offsets[kQuantTable[i] * 3 + c];
    }
  }
  return true;
}

void FindBestDequantMatrices(const CompressParams& cparams,
                             const Image3F& opsin,
                             DequantMatrices* dequant_matrices) {
  // TODO(veluca): heuristics for in-bitstream quant tables.
  *dequant_matrices = DequantMatrices();
  if (cparams.max_error_mode) {
    // Set numerators of all quantization matrices to constant values.
    float weights[3][1] = {{1.0f / cparams.max_error[0]},
                           {1.0f / cparams.max_error[1]},
                           {1.0f / cparams.max_error[2]}};
    DctQuantWeightParams dct_params(weights);
    std::vector<QuantEncoding> encodings(DequantMatrices::kNum,
                                         QuantEncoding::DCT(dct_params));
    dequant_matrices->SetCustom(encodings);
    float dc_weights[3] = {1.0f / cparams.max_error[0],
                           1.0f / cparams.max_error[1],
                           1.0f / cparams.max_error[2]};
    dequant_matrices->SetCustomDC(dc_weights);
  }
}

}  // namespace jxl
