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

#ifndef JXL_ENC_PARAMS_H_
#define JXL_ENC_PARAMS_H_

// Parameters and flags that govern JXL compression.

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "jxl/base/override.h"
#include "jxl/frame_header.h"
#include "jxl/modular/encoding/encoding.h"

namespace jxl {

enum class SpeedTier {
  // Turns on FindBestQuantizationHQ loop. Equivalent to "guetzli" mode.
  kTortoise,
  // Turns on FindBestQuantization butteraugli loop. Default.
  kKitten,
  // Turns on dots, patches, and spline detection by default, as well as full
  // context clustering. Equivalent to "fast" mode.
  kSquirrel,
  // Turns on error diffusion and full AC strategy heuristics.
  kWombat,
  // Turns on gaborish by default, non-default cmap, initial quant field.
  kHare,
  // Turns on simple heuristics for AC strategy, quant field, and clustering;
  // also enables coefficient reordering.
  kCheetah,
  // Turns off most encoder features, for the fastest possible encoding time.
  kFalcon,
};

inline bool ParseSpeedTier(const std::string& s, SpeedTier* out) {
  if (s == "falcon") {
    *out = SpeedTier::kFalcon;
    return true;
  } else if (s == "cheetah") {
    *out = SpeedTier::kCheetah;
    return true;
  } else if (s == "hare") {
    *out = SpeedTier::kHare;
    return true;
  } else if (s == "wombat") {
    *out = SpeedTier::kWombat;
    return true;
  } else if (s == "fast" || s == "squirrel") {
    *out = SpeedTier::kSquirrel;
    return true;
  } else if (s == "kitten") {
    *out = SpeedTier::kKitten;
    return true;
  } else if (s == "guetzli" || s == "tortoise") {
    *out = SpeedTier::kTortoise;
    return true;
  }
  return false;
}

inline const char* SpeedTierName(SpeedTier speed_tier) {
  switch (speed_tier) {
    case SpeedTier::kFalcon:
      return "falcon";
    case SpeedTier::kCheetah:
      return "cheetah";
    case SpeedTier::kHare:
      return "hare";
    case SpeedTier::kWombat:
      return "wombat";
    case SpeedTier::kSquirrel:
      return "squirrel";
    case SpeedTier::kKitten:
      return "kitten";
    case SpeedTier::kTortoise:
      return "tortoise";
  }
}

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct CompressParams {
  // Only used for benchmarking (comparing vs libjpeg)
  int jpeg_quality = 100;
  bool jpeg_chroma_subsampling = false;
  bool clear_metadata = false;

  float butteraugli_distance = 1.0f;
  size_t target_size = 0;
  float target_bitrate = 0.0f;

  // 0.0 means search for the adaptive quantization map that matches the
  // butteraugli distance, positive values mean quantize everywhere with that
  // value.
  float uniform_quant = 0.0f;
  float quant_border_bias = 0.0f;

  // What reference frame number this should be saved as.
  size_t save_as_reference = 0;

  // Try to achieve a maximum pixel-by-pixel error on each channel.
  bool max_error_mode = false;
  float max_error[3] = {0.0, 0.0, 0.0};

  // Encode a special DC frame (recursively, if > 1).
  // Up to 3 pyramid levels - for up to 4096x downsampling.
  size_t dc_level = 0;

  SpeedTier speed_tier = SpeedTier::kKitten;

  int max_butteraugli_iters = 4;

  int max_butteraugli_iters_guetzli_mode = 100;

  ColorTransform color_transform = ColorTransform::kXYB;
  YCbCrChromaSubsampling chroma_subsampling = YCbCrChromaSubsampling::k444;

  // If true, the "modular mode options" members below are used.
  bool modular_group_mode = false;

  // Use "lossless JPEG groups".
  bool brunsli_group_mode = false;

  Override preview = Override::kDefault;
  Override noise = Override::kDefault;
  Override dots = Override::kDefault;
  Override patches = Override::kDefault;
  Override adaptive_reconstruction = Override::kDefault;
  Override gaborish = Override::kDefault;

  // TODO(deymo): Remove "gradient" once all clients stop setting this value.
  // This flag is already deprecated and is unused in the encoder.
  Override gradient = Override::kOff;

  // Progressive mode.
  bool progressive_mode = false;

  // Quantized-progressive mode.
  bool qprogressive_mode = false;

  size_t progressive_dc = 0;

  // Progressive-mode saliency.
  //
  // How many progressive saliency-encoding steps to perform.
  // - 1: Encode only DC and lowest-frequency AC. Does not need a saliency-map.
  // - 2: Encode only DC+LF, dropping all HF AC data.
  //      Does not need a saliency-map.
  // - 3: Encode DC+LF+{salient HF}, dropping all non-salient HF data.
  // - 4: Encode DC+LF+{salient HF}+{other HF}.
  // - 5: Encode DC+LF+{quantized HF}+{low HF bits}.
  size_t saliency_num_progressive_steps = 3;
  // Every saliency-heatmap cell with saliency >= threshold will be considered
  // as 'salient'. The default value of 0.0 will consider every AC-block
  // as salient, hence not require a saliency-map, and not actually generate
  // a 4th progressive step.
  float saliency_threshold = 0.0f;
  // Filename for the saliency-map (must be generated separately).
  std::string saliency_map_filename;
  // Saliency-map (owned by caller).
  ImageF* saliency_map = nullptr;

  // Input and output file name. Will be used to provide pluggable saliency
  // extractor with paths.
  const char* file_in = nullptr;
  const char* file_out = nullptr;

  // Prints extra information during/after encoding.
  bool verbose = false;

  // Multiplier for penalizing new HF artifacts more than blurring away
  // features. 1.0=neutral.
  float hf_asymmetry = 1.0f;

  // Intended intensity target of the viewer after decoding, in nits (cd/m^2).
  // There is no other way of knowing the target brightness - depends on source
  // material. 709 typically targets 100 nits, BT.2100 PQ up to 10K, but HDR
  // content is more typically mastered to 4K nits. The default requires no
  // scaling for Butteraugli.
  float intensity_target = kDefaultIntensityTarget;

  float GetIntensityMultiplier() const {
    return intensity_target * kIntensityMultiplier;
  }

  // modular mode options below
  modular_options options;
  int responsive = -1;
  std::pair<float, float> quality_pair;
  int colorspace = -1;
  // Use Global channel palette if #colors < this percentage of range
  float channel_colors_pre_transform_percent = 95.f;
  // Use Local channel palette if #colors < this percentage of range
  float channel_colors_percent = 80.f;
  int near_lossless = 0;
  int palette_colors = 1 << 10;  // up to 10-bit palette is probably worthwhile
  bool ans = false;

  CompressParams() {
    quality_pair.first = 100.f;   // quality
    quality_pair.second = 100.f;  // cquality
    set_default_modular_options(options);
  }
};

// Always on so we notice any changes.
static constexpr float kMinButteraugliForAdaptiveReconstruction = 0.0f;
static constexpr float kMinButteraugliForDots = 3.0f;
static constexpr float kMinButteraugliToSubtractOriginalPatches = 3.0f;

// Always off
static constexpr float kMinButteraugliForNoise = 99.0f;

// Minimum butteraugli distance the encoder accepts.
static constexpr float kMinButteraugliDistance = 0.01f;

}  // namespace jxl

#endif  // JXL_ENC_PARAMS_H_
