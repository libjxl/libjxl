// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_PARAMS_H_
#define LIB_JXL_ENC_PARAMS_H_

// Parameters and flags that govern JXL compression.

#include <jxl/cms_interface.h>
#include <jxl/encode.h>
#include <stddef.h>

#include <array>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/base/override.h"
#include "lib/jxl/common.h"
#include "lib/jxl/enc_progressive_split.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/splines.h"

namespace jxl {

// NOLINTNEXTLINE(clang-analyzer-optin.performance.Padding)
struct CompressParams {
  float butteraugli_distance = 1.0f;

  // explicit distances for extra channels (defaults to butteraugli_distance
  // when not set; value of -1 can be used to represent 'default')
  std::vector<float> ec_distance;

  // Try to achieve a maximum pixel-by-pixel error on each channel.
  bool max_error_mode = false;
  float max_error[3] = {0.0, 0.0, 0.0};

  bool disable_perceptual_optimizations = false;

  SpeedTier speed_tier = SpeedTier::kSquirrel;
  int brotli_effort = -1;

  // 0 = default.
  // 1 = slightly worse quality.
  // 4 = fastest speed, lowest quality
  size_t decoding_speed_tier = 0;

  ColorTransform color_transform = ColorTransform::kXYB;

  // If true, the "modular mode options" members below are used.
  bool modular_mode = false;

  // Change group size in modular mode (0=128, 1=256, 2=512, 3=1024, -1=encoder
  // chooses).
  int modular_group_size_shift = -1;

  Override preview = Override::kDefault;
  Override noise = Override::kDefault;
  Override dots = Override::kDefault;
  Override patches = Override::kDefault;
  Override gaborish = Override::kDefault;
  int epf = -1;

  // Progressive mode.
  Override progressive_mode = Override::kDefault;

  // Quantized-progressive mode.
  Override qprogressive_mode = Override::kDefault;

  // Put center groups first in the bitstream.
  bool centerfirst = false;

  // Pixel coordinates of the center. First group will contain that center.
  size_t center_x = static_cast<size_t>(-1);
  size_t center_y = static_cast<size_t>(-1);

  int progressive_dc = -1;

  // If on: preserve color of invisible pixels (if off: don't care)
  // Default: on
  Override keep_invisible = Override::kDefault;

  JxlCmsInterface cms;
  bool cms_set = false;
  void SetCms(const JxlCmsInterface& new_cms) {
    cms = new_cms;
    cms_set = true;
  }

  // Force usage of CfL when doing JPEG recompression. This can have unexpected
  // effects on the decoded pixels, while still being JPEG-compliant and
  // allowing reconstruction of the original JPEG.
  bool force_cfl_jpeg_recompression = true;
  // Apply LF Smoothing when doing JPEG recompression. Same applies as above.
  // Default to non-subsampled smoothing only until decoders update.
  int force_lfs_jpeg_recompression = -1;

  // Use brotli compression for any boxes derived from a JPEG frame.
  bool jpeg_compress_boxes = true;

  // Preserve this metadata when doing JPEG recompression.
  bool jpeg_keep_exif = true;
  bool jpeg_keep_xmp = true;
  bool jpeg_keep_jumbf = true;

  // Set the noise to what it would approximately be if shooting at the nominal
  // exposure for a given ISO setting on a 35mm camera.
  float photon_noise_iso = 0;

  // modular mode options below
  ModularOptions options;

  // TODO(eustas): use Override?
  int responsive = -1;
  int colorspace = -1;
  int move_to_front_from_channel = -1;

  // Use Global channel palette if #colors < this percentage of range
  float channel_colors_pre_transform_percent = 95.f;
  // Use Local channel palette if #colors < this percentage of range
  float channel_colors_percent = 80.f;
  int palette_colors = 1 << 10;  // up to 10-bit palette is probably worthwhile
  bool lossy_palette = false;

  // Returns whether these params are lossless as defined by SetLossless();
  bool IsLossless() const { return modular_mode && ModularPartIsLossless(); }

  bool ModularPartIsLossless() const {
    if (modular_mode) {
      // YCbCr is also considered lossless here since it's intended for
      // source material that is already YCbCr (we don't do the fwd transform)
      if (butteraugli_distance != 0 ||
          color_transform == jxl::ColorTransform::kXYB)
        return false;
    }
    for (float f : ec_distance) {
      if (f > 0) return false;
      if (f < 0 && butteraugli_distance != 0) return false;
    }
    // all modular channels are encoded at distance 0
    return true;
  }

  // Sets the parameters required to make the codec lossless.
  void SetLossless() {
    modular_mode = true;
    butteraugli_distance = 0.0f;
    for (float& f : ec_distance) f = 0.0f;
    color_transform = jxl::ColorTransform::kNone;
  }

  // Down/upsample the image before encoding / after decoding by this factor.
  // The resampling value can also be set to <= 0 to automatically choose based
  // on distance, however EncodeFrame doesn't support this, so it is
  // required to call PostInit() to set a valid positive resampling
  // value and altered butteraugli score if this is used.
  int resampling = -1;
  int ec_resampling = -1;
  // Skip the downsampling before encoding if this is true.
  bool already_downsampled = false;
  // Butteraugli target distance on the original full size image, this can be
  // different from butteraugli_distance if resampling was used.
  float original_butteraugli_distance = -1.0f;

  float quant_ac_rescale = 1.0;

  // Codestream level to conform to.
  // -1: don't care
  int level = -1;

  // See JXL_ENC_FRAME_SETTING_BUFFERING option value.
  int buffering = -1;
  // Output streaming mode: 0=buffered, 1=seek-based streaming, 2=OOO jxlp.
  int output_mode = 0;
  // See JXL_ENC_FRAME_SETTING_USE_FULL_IMAGE_HEURISTICS option value.
  bool use_full_image_heuristics = true;

  std::vector<float> manual_noise;
  std::vector<float> manual_xyb_factors;

  // If not empty, this tree will be used for dc global section.
  // Used in jxl_from_tree tool.
  Tree custom_fixed_tree;
  // Optimal LZ77 (slowest efforts) always uses the slower, more careful greedy
  // first pass, instead of only for small inputs. Helps small files with large
  // token streams, such as jxl_from_tree art. Used in jxl_from_tree tool.
  bool lz77_careful_first_pass = false;
  // Modular frames: if enabled, exactly this palette transform (on num_c
  // channels starting at the first non-meta channel) instead of the palette
  // heuristics. The palette entries themselves are the palette meta channel
  // (nb_deltas + nb_colors wide, num_c high), which is coded like any other
  // channel. Used in jxl_from_tree tool (with custom_fixed_tree and zero
  // tokens): if `entries` is empty, the tree defines the entries (whatever it
  // gives for the meta channel with zero residuals); otherwise the meta channel
  // holds `entries` (row c = component c, column i = entry i) and its residuals
  // are coded (see ModularOptions::code_meta_channels).
  struct CustomPalette {
    bool enabled = false;
    uint32_t num_c = 3;
    uint32_t nb_colors = 0;
    uint32_t nb_deltas = 0;
    Predictor predictor = Predictor::Zero;
    std::vector<std::vector<int32_t>> entries;
  };
  CustomPalette custom_palette;
  // Modular data (with custom_fixed_tree and zero tokens, i.e. jxl_from_tree):
  // if not empty, the residuals of the streams, by stream ID (the tree's
  // property 1; -1 for every stream not listed): a prefix, then a period that
  // repeats. Streams not covered keep all-zero residuals.
  std::map<int, ResidualPattern> custom_residuals;
  // Modular frames (jxl_from_tree's NibbleCode): prefix codes, the default
  // hybrid uint config, and every histogram that uses more than two of the
  // tokens 0..15 (and no others) becomes the flat code of all 16 (4 bits
  // each), so that those tokens are raw nibbles in the bitstream.
  bool flat_nibble_code = false;
  // Modular frames with custom_fixed_tree and zero tokens (jxl_from_tree): if
  // not null, the smallest and largest value of any modular buffer when
  // decoding (the samples, and the intermediate values while undoing the
  // transforms) are merged into [0] and [1], so that the caller can tell
  // whether 16-bit buffers suffice.
  int64_t* modular_range_out = nullptr;
  // The same frames: if not null, the smallest and largest decoded value of
  // each channel in the final layout (colour channels, if modular, then the
  // extra channels), merged into the entries (resized as needed).
  std::vector<std::pair<int64_t, int64_t>>* modular_channel_ranges_out =
      nullptr;
  // VarDCT frames: the LF image and the HF metadata (chroma-from-luma maps, AC
  // strategy, quantization field, EPF sharpness) are what custom_fixed_tree
  // gives with all residuals zero, and all HF coefficients are zero. Used in
  // jxl_from_tree tool.
  bool vardct_from_tree = false;
  // With vardct_from_tree: optionally a custom block context map, and (if not
  // empty) one fixed token per HF context (block_ctx_map.NumACContexts() of
  // them): every HF symbol in a context is that token, so HF costs no bits.
  // With vardct_from_tree and XYB: the frame header's x_qm_scale and
  // b_qm_scale (0..7; -1 = the encoder's choice). A decoder multiplies the HF
  // steps of X and B by 0.8^(scale - 2).
  int vardct_x_qm_scale = -1;
  int vardct_b_qm_scale = -1;
  bool use_custom_block_ctx_map = false;
  BlockCtxMap custom_block_ctx_map;
  std::vector<uint32_t> custom_hf_tokens;
  // With vardct_from_tree: if not empty (3 * kNumOrders entries, for order
  // class o and channel c (0 X, 1 Y, 2 B) at 3 * o + c), custom coefficient
  // orders. A non-empty entry lists the positions (row * columns + column, in
  // the coefficient layout of the order class, which has at least as many
  // columns as rows) that come right after the LLF coefficients, in that order;
  // the other positions keep their default order. The orders of the classes
  // with a non-empty entry are signaled.
  std::vector<std::vector<uint32_t>> custom_coeff_orders;
  // With vardct_from_tree: the quantizer's global scale and LF quantization
  // (quant_dc), 0 for the default ones (1024 and 64), and if not empty, the
  // inverse LF quantization steps of the channels X, Y and B (by default 4096,
  // 512 and 256). The LF step of channel c is
  // (65536 / global scale) / quant_dc / vardct_lf_inv_quant[c], the HF step is
  // (65536 / global scale) / quant field * the dequantization matrix.
  uint32_t vardct_global_scale = 0;
  uint32_t vardct_quant_dc = 0;
  std::vector<float> vardct_lf_inv_quant;
  // With vardct_from_tree: the chroma from luma factors of the LF (-128..127,
  // in units of 1 / color factor, 84, on top of the base correlations 0 for
  // X and 1 for B): the LF X is X + ytox * Y, the LF B is B + ytob * Y.
  int32_t vardct_ytox_dc = 0;
  int32_t vardct_ytob_dc = 0;
  // With vardct_from_tree: a custom dequantization matrix (quantization table
  // of lib/jxl/quant_weights.h). The dequantization step of a coefficient is
  // the inverse of its quantization weight.
  struct CustomDequantTable {
    // If not 0, a parametric table (QuantEncoding::DCT) with this many (1..17)
    // distance bands: band_steps[c][i] is the step of channel c (0 X, 1 Y,
    // 2 B) at distance band i, from the top-left coefficient (i = 0) to the
    // opposite corner, with geometric interpolation in between.
    size_t num_bands = 0;
    std::array<std::array<float, 17>, 3> band_steps = {};
    // If not 0, a RAW table (QuantEncoding::RAW): the integers (> 0) that
    // custom_fixed_tree gives with zero residuals for the stream of this
    // quantization table; the step of a coefficient is raw_den times its
    // integer.
    float raw_den = 0;
  };
  // If not empty (kNumQuantTables entries, by QuantTable index), custom
  // dequantization matrices; the entries with neither num_bands nor raw_den
  // keep the default table.
  std::vector<CustomDequantTable> vardct_dequant;
  // If not empty, these custom splines will be used instead of the computed
  // ones. Used in jxl_from_tee tool.
  SplineDataView custom_splines{};
  // A patch placement for custom_patches: the rectangle (x0, y0, xsize, ysize)
  // of reference frame `ref` is blended onto this frame at (x, y), with
  // `blend_mode` (a PatchBlendMode value) for the color channels and
  // `ec_blend_mode` for every extra channel.
  struct CustomPatch {
    size_t ref, x0, y0, xsize, ysize, x, y;
    uint8_t blend_mode, ec_blend_mode;
    bool clamp;
  };
  // If not empty, these patches will be used instead of computed ones (modular
  // mode only). Used in jxl_from_tree tool.
  std::vector<CustomPatch> custom_patches;
  // If not null, overrides progressive mode settings. Used in decode_test.
  const ProgressiveMode* custom_progressive_mode = nullptr;

  JxlDebugImageCallback debug_image = nullptr;
  void* debug_image_opaque;
};

static constexpr float kMinButteraugliForDynamicAR = 0.5f;
static constexpr float kMinButteraugliForDots = 3.0f;
static constexpr float kMinButteraugliToSubtractOriginalPatches = 3.0f;

// Always off
static constexpr float kMinButteraugliForNoise = 99.0f;

// Minimum butteraugli distance the encoder accepts.
// Below d0.05 is not useful and risks going outside Level 5 limits
// (in particular modular_16bit_buffers becomes an issue for DC)
static constexpr float kMinButteraugliDistance = 0.05f;

// Tile size for encoder-side processing. Must be equal to color tile dim in the
// current implementation.
static constexpr size_t kEncTileDim = 64;
static constexpr size_t kEncTileDimInBlocks = kEncTileDim / kBlockDim;

}  // namespace jxl

#endif  // LIB_JXL_ENC_PARAMS_H_
