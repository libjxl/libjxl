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

#ifndef JXL_NOISE_H_
#define JXL_NOISE_H_

// Noise synthesis. Currently disabled.

#include <stddef.h>

#include <vector>

#include "jxl/aux_out_fwd.h"
#include "jxl/base/status.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/image.h"

namespace jxl {

struct NoiseParams {
  // LUT index is an intensity of pixel / mean intensity of patch
  static constexpr size_t kNumNoisePoints = 8;
  float lut[kNumNoisePoints];

  void Clear() {
    for (float& i : lut) i = 0;
  }
  bool HasAny() const {
    for (float i : lut) {
      if (i != 0) return true;
    }
    return false;
  }
};

struct NoiseLevel {
  float noise_level;
  float intensity;
};

// Add a noise to Opsin image, loading generated random noise from `noise_rect`
// in `noise`.
HWY_ATTR void AddNoise(const NoiseParams& noise_params, const Rect& noise_rect,
                       const Image3F& noise, const Rect& opsin_rect,
                       const ColorCorrelationMap& cmap, Image3F* opsin);

void RandomImage3(const Rect& rect, Image3F* JXL_RESTRICT noise);

// Get parameters of the noise for NoiseParams model
// Returns whether a valid noise model (with HasAny()) is set.
Status GetNoiseParameter(const Image3F& opsin, NoiseParams* noise_params,
                         float quality_coef);

// Does not write anything if `noise_params` are empty. Otherwise, caller must
// set FrameHeader.flags.kNoise.
void EncodeNoise(const NoiseParams& noise_params, BitWriter* writer,
                 size_t layer, AuxOut* aux_out);

// Must only call if FrameHeader.flags.kNoise.
Status DecodeNoise(BitReader* br, NoiseParams* noise_params);

// Texture Strength is defined as tr(A), A = [Gh, Gv]^T[[Gh, Gv]]
std::vector<float> GetTextureStrength(const Image3F& opsin, size_t block_s);

float GetThresholdFlatIndices(const std::vector<float>& texture_strength,
                              int n_patches);

std::vector<NoiseLevel> GetNoiseLevel(
    const Image3F& opsin, const std::vector<float>& texture_strength,
    float threshold, size_t block_s);

void OptimizeNoiseParameters(const std::vector<NoiseLevel>& noise_level,
                             NoiseParams* noise_params);
}  // namespace jxl

#endif  // JXL_NOISE_H_
