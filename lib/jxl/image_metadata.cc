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

#include "lib/jxl/image_metadata.h"

#include <limits>
#include <utility>

#include "lib/jxl/alpha.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/profiler.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/fields.h"

namespace jxl {
BitDepth::BitDepth() { Bundle::Init(this); }
Status BitDepth::VisitFields(Visitor* JXL_RESTRICT visitor) {
  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &floating_point_sample));
  // The same fields (bits_per_sample and exponent_bits_per_sample) are read
  // in a different way depending on floating_point_sample's value. It's still
  // default-initialized correctly so using visitor->Conditional is not
  // required.
  if (!floating_point_sample) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
        Val(8), Val(10), Val(12), BitsOffset(6, 1), 8, &bits_per_sample));
    exponent_bits_per_sample = 0;
  } else {
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
        Val(32), Val(16), Val(24), BitsOffset(6, 1), 32, &bits_per_sample));
    // The encoded value is exponent_bits_per_sample - 1, encoded in 3 bits
    // so the value can be in range [1, 8].
    const uint32_t offset = 1;
    exponent_bits_per_sample -= offset;
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->Bits(4, 8 - offset, &exponent_bits_per_sample));
    exponent_bits_per_sample += offset;
  }

  // Error-checking for floating point ranges.
  if (floating_point_sample) {
    if (exponent_bits_per_sample < 2 || exponent_bits_per_sample > 8) {
      return JXL_FAILURE("Invalid exponent_bits_per_sample: %u",
                         exponent_bits_per_sample);
    }
    int mantissa_bits =
        static_cast<int>(bits_per_sample) - exponent_bits_per_sample - 1;
    if (mantissa_bits < 2 || mantissa_bits > 23) {
      return JXL_FAILURE("Invalid bits_per_sample: %u", bits_per_sample);
    }
  } else {
    if (bits_per_sample > 31) {
      return JXL_FAILURE("Invalid bits_per_sample: %u", bits_per_sample);
    }
  }
  return true;
}

ExtraChannelInfo::ExtraChannelInfo() { Bundle::Init(this); }
Status ExtraChannelInfo::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->AllDefault(*this, &all_default)) {
    // Overwrite all serialized fields, but not any nonserialized_*.
    visitor->SetDefault(this);
    return true;
  }

  // General
  JXL_QUIET_RETURN_IF_ERROR(visitor->Enum(ExtraChannel::kAlpha, &type));

  JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&bit_depth));

  JXL_QUIET_RETURN_IF_ERROR(
      visitor->U32(Val(0), Val(3), Val(4), BitsOffset(3, 1), 0, &dim_shift));
  if ((1U << dim_shift) > kGroupDim) {
    return JXL_FAILURE("dim_shift %u too large", dim_shift);
  }

  JXL_QUIET_RETURN_IF_ERROR(VisitNameString(visitor, &name));

  // Conditional
  if (visitor->Conditional(type == ExtraChannel::kAlpha)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &alpha_associated));
  }
  if (visitor->Conditional(type == ExtraChannel::kSpotColor)) {
    for (float& c : spot_color) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->F16(0, &c));
    }
  }
  if (visitor->Conditional(type == ExtraChannel::kCFA)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(1), Bits(2), BitsOffset(4, 3),
                                           BitsOffset(8, 19), 1, &cfa_channel));
  }
  return true;
}

ImageMetadata::ImageMetadata() { Bundle::Init(this); }
Status ImageMetadata::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->AllDefault(*this, &all_default)) {
    // Overwrite all serialized fields, but not any nonserialized_*.
    visitor->SetDefault(this);
    return true;
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&bit_depth));
  JXL_QUIET_RETURN_IF_ERROR(
      visitor->Bool(true, &modular_16_bit_buffer_sufficient));

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(true, &xyb_encoded));
  JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&color_encoding));

  m2.nonserialized_xyb_encoded = xyb_encoded;
  JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&m2));

  return true;
}

ImageMetadata2::ImageMetadata2() { Bundle::Init(this); }
Status ImageMetadata2::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->AllDefault(*this, &all_default)) {
    // Overwrite all serialized fields, but not any nonserialized_*.
    visitor->SetDefault(this);
    return true;
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &have_preview));
  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &have_animation));

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &have_intrinsic_size));
  if (visitor->Conditional(have_intrinsic_size)) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&intrinsic_size));
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bits(3, 0, &orientation_minus_1));
  // (No need for bounds checking because we read exactly 3 bits)

  JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&tone_mapping));

  num_extra_channels = extra_channel_info.size();
  JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(0), Bits(4), BitsOffset(8, 16),
                                         BitsOffset(12, 1), 0,
                                         &num_extra_channels));

  if (visitor->Conditional(num_extra_channels != 0)) {
    if (visitor->IsReading()) {
      extra_channel_info.resize(num_extra_channels);
    }
    for (ExtraChannelInfo& eci : extra_channel_info) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&eci));
    }
  }

  // Treat as if only the fields up to extra channels exist.
  if (visitor->IsReading() && nonserialized_only_parse_basic_info) {
    return true;
  }

  if (nonserialized_xyb_encoded) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&opsin_inverse_matrix));
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->BeginExtensions(&extensions));
  // Extensions: in chronological order of being added to the format.
  return visitor->EndExtensions();
}

OpsinInverseMatrix::OpsinInverseMatrix() { Bundle::Init(this); }
Status OpsinInverseMatrix::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->AllDefault(*this, &all_default)) {
    // Overwrite all serialized fields, but not any nonserialized_*.
    visitor->SetDefault(this);
    return true;
  }
  for (int i = 0; i < 9; ++i) {
    JXL_QUIET_RETURN_IF_ERROR(visitor->F16(
        DefaultInverseOpsinAbsorbanceMatrix()[i], &inverse_matrix[i]));
  }
  for (int i = 0; i < 3; ++i) {
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->F16(kNegOpsinAbsorbanceBiasRGB[i], &opsin_biases[i]));
  }
  for (int i = 0; i < 4; ++i) {
    JXL_QUIET_RETURN_IF_ERROR(
        visitor->F16(kDefaultQuantBias[i], &quant_biases[i]));
  }
  return true;
}

ToneMapping::ToneMapping() { Bundle::Init(this); }
Status ToneMapping::VisitFields(Visitor* JXL_RESTRICT visitor) {
  if (visitor->AllDefault(*this, &all_default)) {
    // Overwrite all serialized fields, but not any nonserialized_*.
    visitor->SetDefault(this);
    return true;
  }

  JXL_QUIET_RETURN_IF_ERROR(
      visitor->F16(kDefaultIntensityTarget, &intensity_target));
  if (intensity_target <= 0.f) {
    return JXL_FAILURE("invalid intensity target");
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->F16(0.0f, &min_nits));
  if (min_nits < 0.f || min_nits > intensity_target) {
    return JXL_FAILURE("invalid min %f vs max %f", min_nits, intensity_target);
  }

  JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &relative_to_max_display));

  JXL_QUIET_RETURN_IF_ERROR(visitor->F16(0.0f, &linear_below));
  if (linear_below < 0 || (relative_to_max_display && linear_below > 1.0f)) {
    return JXL_FAILURE("invalid linear_below %f (%s)", linear_below,
                       relative_to_max_display ? "relative" : "absolute");
  }

  return true;
}

Status ReadImageMetadata(BitReader* JXL_RESTRICT reader,
                         ImageMetadata* JXL_RESTRICT metadata) {
  return Bundle::Read(reader, metadata);
}

Status WriteImageMetadata(const ImageMetadata& metadata,
                          BitWriter* JXL_RESTRICT writer, size_t layer,
                          AuxOut* aux_out) {
  return Bundle::Write(metadata, writer, layer, aux_out);
}

void ImageMetadata::SetAlphaBits(uint32_t bits) {
  std::vector<ExtraChannelInfo>& eciv = m2.extra_channel_info;
  ExtraChannelInfo* alpha = m2.Find(ExtraChannel::kAlpha);
  if (bits == 0) {
    if (alpha != nullptr) {
      // Remove the alpha channel from the extra channel info. It's
      // theoretically possible that there are multiple, remove all in that
      // case. This ensure a next HasAlpha() will return false.
      const auto is_alpha = [](const ExtraChannelInfo& eci) {
        return eci.type == ExtraChannel::kAlpha;
      };
      eciv.erase(std::remove_if(eciv.begin(), eciv.end(), is_alpha),
                 eciv.end());
    }
  } else {
    if (alpha == nullptr) {
      ExtraChannelInfo info;
      info.type = ExtraChannel::kAlpha;
      info.bit_depth.bits_per_sample = bits;
      info.dim_shift = 0;
      info.alpha_associated = false;  // may be set by SetAlpha() later
      // Prepend rather than append: in case there already are other extra
      // channels, prefer alpha channel to be listed first.
      eciv.insert(eciv.begin(), info);
    } else {
      // Ignores potential extra alpha channels, only sets to first one.
      alpha->bit_depth.bits_per_sample = bits;
      alpha->bit_depth.floating_point_sample = false;
      alpha->bit_depth.exponent_bits_per_sample = 0;
    }
  }
}
}  // namespace jxl
