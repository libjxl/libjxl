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

#include "jxl/image_bundle.h"

#include <utility>

#include "jxl/alpha.h"
#include "jxl/base/byte_order.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/profiler.h"
#include "jxl/codec_in_out.h"
#include "jxl/color_management.h"
#include "jxl/external_image.h"
#include "jxl/fields.h"

namespace jxl {
namespace {

Status FromSRGB(const size_t xsize, const size_t ysize, const bool is_gray,
                const bool has_alpha, const bool alpha_is_premultiplied,
                const bool is_16bit, const bool big_endian,
                const uint8_t* pixels, const uint8_t* end, ThreadPool* pool,
                ImageBundle* ib) {
  const ColorEncoding& c = ColorEncoding::SRGB(is_gray);
  const size_t bits_per_sample = (is_16bit ? 2 : 1) * kBitsPerByte;
  const PackedImage desc(xsize, ysize, c, has_alpha, alpha_is_premultiplied,
                         /*bits_per_alpha=*/bits_per_sample, bits_per_sample,
                         big_endian, /*flipped_y=*/false);
  const Span<const uint8_t> span(pixels, end - pixels);
  return CopyTo(desc, span, pool, ib);
}

// Copies interleaved external color; skips any alpha. Caller ensures
// bits_per_sample matches T, and byte order=native.
template <typename T>
void AllocateAndFill(const ExternalImage& external, Image3<T>* out) {
  JXL_ASSERT(external.IsHealthy());  // Callers must check beforehand.

  // Here we just copy bytes for simplicity; for conversion/byte swapping, use
  // ExternalImage::CopyTo instead.
  JXL_CHECK(external.BitsPerSample() == sizeof(T) * kBitsPerByte);
  JXL_CHECK(external.BigEndian() == !IsLittleEndian());

  const size_t xsize = external.xsize();
  const size_t ysize = external.ysize();
  *out = Image3<T>(xsize, ysize);
  if (external.IsGray()) {
    if (external.HasAlpha()) {
      for (size_t y = 0; y < ysize; ++y) {
        const T* JXL_RESTRICT row =
            reinterpret_cast<const T*>(external.ConstRow(y));
        T* JXL_RESTRICT row0 = out->PlaneRow(0, y);
        T* JXL_RESTRICT row1 = out->PlaneRow(1, y);
        T* JXL_RESTRICT row2 = out->PlaneRow(2, y);
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row[2 * x + 0];
          row1[x] = row[2 * x + 0];
          row2[x] = row[2 * x + 0];
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        const T* JXL_RESTRICT row =
            reinterpret_cast<const T*>(external.ConstRow(y));
        T* JXL_RESTRICT row0 = out->PlaneRow(0, y);
        T* JXL_RESTRICT row1 = out->PlaneRow(1, y);
        T* JXL_RESTRICT row2 = out->PlaneRow(2, y);
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row[x];
          row1[x] = row[x];
          row2[x] = row[x];
        }
      }
    }
  } else {
    if (external.HasAlpha()) {
      for (size_t y = 0; y < ysize; ++y) {
        const T* JXL_RESTRICT row =
            reinterpret_cast<const T*>(external.ConstRow(y));
        T* JXL_RESTRICT row0 = out->PlaneRow(0, y);
        T* JXL_RESTRICT row1 = out->PlaneRow(1, y);
        T* JXL_RESTRICT row2 = out->PlaneRow(2, y);

        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row[4 * x + 0];
          row1[x] = row[4 * x + 1];
          row2[x] = row[4 * x + 2];
        }
      }
    } else {
      for (size_t y = 0; y < ysize; ++y) {
        const T* JXL_RESTRICT row =
            reinterpret_cast<const T*>(external.ConstRow(y));
        T* JXL_RESTRICT row0 = out->PlaneRow(0, y);
        T* JXL_RESTRICT row1 = out->PlaneRow(1, y);
        T* JXL_RESTRICT row2 = out->PlaneRow(2, y);
        for (size_t x = 0; x < xsize; ++x) {
          row0[x] = row[3 * x + 0];
          row1[x] = row[3 * x + 1];
          row2[x] = row[3 * x + 2];
        }
      }
    }
  }
}

// Copies ib:rect, converts, and copies into out.
template <typename T>
Status CopyToT(const ImageMetadata* metadata, const ImageBundle* ib,
               const Rect& rect, const ColorEncoding& c_desired,
               ThreadPool* pool, Image3<T>* out) {
  PROFILER_FUNC;
  // Changing IsGray is probably a bug.
  JXL_CHECK(ib->IsGray() == c_desired.IsGray());

  const ImageU* alpha = ib->HasAlpha() ? &ib->alpha() : nullptr;
  const size_t bits_per_sample = sizeof(T) * kBitsPerByte;
  const bool big_endian = !IsLittleEndian();
  CodecIntervals* temp_intervals = nullptr;  // Don't need min/max.
  const ExternalImage external(
      pool, ib->color(), rect, ib->c_current(), c_desired, ib->HasAlpha(),
      ib->AlphaIsPremultiplied(), alpha, metadata->GetAlphaBits(),
      bits_per_sample, big_endian, temp_intervals);
  JXL_RETURN_IF_ERROR(external.IsHealthy());
  AllocateAndFill(external, out);
  return true;
}

}  // namespace

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

  JXL_QUIET_RETURN_IF_ERROR(VisitNewBase(visitor, &new_base));

  JXL_QUIET_RETURN_IF_ERROR(VisitBlendMode(visitor, &blend_mode));
  if (blend_mode == BlendMode::kBlend && type == ExtraChannel::kAlpha) {
    return JXL_FAILURE("Cannot blend alpha");
  }

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
      // leave new_base default to save space, it is ignored anyway.
      info.blend_mode = BlendMode::kReplace;
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

void ImageBundle::ShrinkTo(size_t xsize, size_t ysize) {
  color_.ShrinkTo(xsize, ysize);
  for (ImageU& plane : extra_channels_) {
    plane.ShrinkTo(xsize, ysize);
  }
}

// Called by all other SetFrom*.
void ImageBundle::SetFromImage(Image3F&& color,
                               const ColorEncoding& c_current) {
  JXL_CHECK(color.xsize() != 0 && color.ysize() != 0);
  JXL_CHECK(metadata_->color_encoding.IsGray() == c_current.IsGray());
  color_ = std::move(color);
  c_current_ = c_current;
  VerifySizes();
}

Status ImageBundle::SetFromSRGB(size_t xsize, size_t ysize, bool is_gray,
                                bool has_alpha, bool alpha_is_premultiplied,
                                const uint8_t* pixels, const uint8_t* end,
                                ThreadPool* pool) {
  const bool big_endian = false;  // don't care since each sample is a byte
  return FromSRGB(xsize, ysize, is_gray, has_alpha, alpha_is_premultiplied,
                  /*is_16bit=*/false, big_endian, pixels, end, pool, this);
}

Status ImageBundle::SetFromSRGB(size_t xsize, size_t ysize, bool is_gray,
                                bool has_alpha, bool alpha_is_premultiplied,
                                const uint16_t* pixels, const uint16_t* end,
                                ThreadPool* pool) {
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(pixels);
  const uint8_t* bytes_end = reinterpret_cast<const uint8_t*>(end);
  // Given as uint16_t, so is in native order.
  const bool big_endian = !IsLittleEndian();
  return FromSRGB(xsize, ysize, is_gray, has_alpha, alpha_is_premultiplied,
                  /*is_16bit=*/true, big_endian, bytes, bytes_end, pool, this);
}

Status ImageBundle::SetFromSRGB(size_t xsize, size_t ysize, bool is_gray,
                                bool has_alpha, bool alpha_is_premultiplied,
                                bool is_16bit, bool big_endian,
                                const uint8_t* pixels, const uint8_t* end,
                                ThreadPool* pool) {
  return FromSRGB(xsize, ysize, is_gray, has_alpha, alpha_is_premultiplied,
                  is_16bit, big_endian, pixels, end, pool, this);
}

Status ImageBundle::TransformTo(const ColorEncoding& c_desired,
                                ThreadPool* pool) {
  PROFILER_FUNC;
  // Changing IsGray is probably a bug.
  JXL_CHECK(IsGray() == c_desired.IsGray());

  const ImageU* alpha_ptr = HasAlpha() ? &alpha() : nullptr;
  const size_t alpha_bits = HasAlpha() ? metadata_->GetAlphaBits() : 0;
  const bool big_endian = !IsLittleEndian();
  CodecIntervals temp_intervals;
  const ExternalImage external(pool, color_, Rect(color_), c_current_,
                               c_desired, HasAlpha(), AlphaIsPremultiplied(),
                               alpha_ptr, alpha_bits, 32, big_endian,
                               &temp_intervals);
  return external.IsHealthy() && external.CopyTo(&temp_intervals, pool, this);
}

Status ImageBundle::CopyTo(const Rect& rect, const ColorEncoding& c_desired,
                           Image3B* out, ThreadPool* pool) const {
  return CopyToT(metadata_, this, rect, c_desired, pool, out);
}
Status ImageBundle::CopyTo(const Rect& rect, const ColorEncoding& c_desired,
                           Image3U* out, ThreadPool* pool) const {
  return CopyToT(metadata_, this, rect, c_desired, pool, out);
}
Status ImageBundle::CopyTo(const Rect& rect, const ColorEncoding& c_desired,
                           Image3F* out, ThreadPool* pool) const {
  return CopyToT(metadata_, this, rect, c_desired, pool, out);
}

Status ImageBundle::CopyToSRGB(const Rect& rect, Image3B* out,
                               ThreadPool* pool) const {
  return CopyTo(rect, ColorEncoding::SRGB(IsGray()), out, pool);
}

void ImageBundle::VerifyMetadata() const {
  JXL_CHECK(!c_current_.ICC().empty());
  JXL_CHECK(metadata_->color_encoding.IsGray() == IsGray());

  if (metadata_->HasAlpha() && alpha().xsize() == 0) {
    JXL_ABORT("MD alpha_bits %u IB alpha %zu x %zu\n",
              metadata_->GetAlphaBits(), alpha().xsize(), alpha().ysize());
  }
  const uint32_t alpha_bits = metadata_->GetAlphaBits();
  JXL_CHECK(alpha_bits <= 16);

  // metadata_->m2.num_extra_channels may temporarily differ from
  // extra_channels_.size(), e.g. after SetAlpha. They are synced by the next
  // call to VisitFields.
}

void ImageBundle::VerifySizes() const {
  const size_t xs = xsize();
  const size_t ys = ysize();

  if (HasExtraChannels()) {
    JXL_CHECK(xs != 0 && ys != 0);
    for (size_t ec = 0; ec < metadata_->m2.extra_channel_info.size(); ++ec) {
      const ExtraChannelInfo& eci = metadata_->m2.extra_channel_info[ec];
      JXL_CHECK(extra_channels_[ec].xsize() == eci.Size(xs));
      JXL_CHECK(extra_channels_[ec].ysize() == eci.Size(ys));
    }
  }
}

size_t ImageBundle::DetectRealBitdepth() const {
  return metadata_->bit_depth.bits_per_sample;

  // TODO(lode): let this function return lower bit depth if possible, e.g.
  // return 8 bits in case the original image came from a 16-bit PNG that
  // was in fact representable as 8-bit PNG. Ensure that the implementation
  // returns 16 if e.g. two consecutive 16-bit values appeared in the original
  // image (such as 32768 and 32769), take into account that e.g. the values
  // 3-bit can represent is not a superset of the values 2-bit can represent,
  // and there may be slight imprecisions in the nits-scaled or 255-scaled
  // floating point image.
}

const ImageU& ImageBundle::alpha() const {
  JXL_ASSERT(HasAlpha());
  const size_t ec = metadata_->m2.Find(ExtraChannel::kAlpha) -
                    metadata_->m2.extra_channel_info.data();
  JXL_ASSERT(ec < extra_channels_.size());
  return extra_channels_[ec];
}

const ImageU& ImageBundle::depth() const {
  JXL_ASSERT(HasDepth());
  const size_t ec = metadata_->m2.Find(ExtraChannel::kDepth) -
                    metadata_->m2.extra_channel_info.data();
  JXL_ASSERT(ec < extra_channels_.size());
  return extra_channels_[ec];
}

void ImageBundle::RemoveAlpha() {
  const ExtraChannelInfo* eci = metadata_->m2.Find(ExtraChannel::kAlpha);
  JXL_ASSERT(eci != nullptr);
  const size_t ec = eci - metadata_->m2.extra_channel_info.data();
  JXL_ASSERT(ec < extra_channels_.size());
  extra_channels_.erase(extra_channels_.begin() + ec);
  metadata_->SetAlphaBits(0);  // maintain invariant for VerifyMetadata
  JXL_ASSERT(!HasAlpha());
}

void ImageBundle::SetAlpha(ImageU&& alpha, bool alpha_is_premultiplied) {
  ExtraChannelInfo* eci = metadata_->m2.Find(ExtraChannel::kAlpha);
  // Must call SetAlphaBits first, otherwise we don't know which channel index
  JXL_CHECK(eci != nullptr);
  JXL_CHECK(alpha.xsize() != 0 && alpha.ysize() != 0);
  eci->alpha_associated = alpha_is_premultiplied;
  extra_channels_.insert(
      extra_channels_.begin() + (eci - metadata_->m2.extra_channel_info.data()),
      std::move(alpha));
  // num_extra_channels is automatically set in visitor
  VerifySizes();
}

void ImageBundle::PremultiplyAlphaIfNeeded(ThreadPool* pool) {
  ExtraChannelInfo* eci = metadata_->m2.Find(ExtraChannel::kAlpha);
  JXL_ASSERT(eci != nullptr);
  const ImageU& alpha_ =
      extra_channels()[eci - metadata_->m2.extra_channel_info.data()];
  JXL_CHECK(alpha_.xsize() == color_.xsize() &&
            alpha_.ysize() == color_.ysize());
  if (eci->alpha_associated) return;
  eci->alpha_associated = true;
  RunOnPool(
      pool, 0, alpha_.ysize(), ThreadPool::SkipInit(),
      [this, &alpha_](const int y, const int /*thread*/) {
        PremultiplyAlpha(color_.PlaneRow(0, y), color_.PlaneRow(1, y),
                         color_.PlaneRow(2, y), alpha_.ConstRow(y),
                         metadata_->GetAlphaBits(), color_.xsize());
      },
      "premultiply alpha");
}

void ImageBundle::SetDepth(ImageU&& depth) {
  JXL_CHECK(depth.xsize() != 0 && depth.ysize() != 0);
  const ExtraChannelInfo* eci = metadata_->m2.Find(ExtraChannel::kDepth);
  JXL_CHECK(eci != nullptr);
  const size_t ec = eci - metadata_->m2.extra_channel_info.data();
  JXL_ASSERT(ec < extra_channels_.size());
  extra_channels_[ec] = std::move(depth);
  VerifySizes();
}

void ImageBundle::SetExtraChannels(std::vector<ImageU>&& extra_channels) {
  JXL_CHECK(!extra_channels.empty());
  for (const ImageU& plane : extra_channels) {
    JXL_CHECK(plane.xsize() != 0 && plane.ysize() != 0);
  }
  extra_channels_ = std::move(extra_channels);
  VerifySizes();
}

Status TransformIfNeeded(const ImageBundle& in, const ColorEncoding& c_desired,
                         ThreadPool* pool, ImageBundle* store,
                         const ImageBundle** out) {
  if (in.c_current().SameColorEncoding(c_desired)) {
    *out = &in;
    return true;
  }
  // TODO(janwas): avoid copying via createExternal+copyBackToIO
  // instead of copy+createExternal+copyBackToIO
  store->SetFromImage(CopyImage(in.color()), in.c_current());

  // Must at least copy the alpha channel for use by external_image.
  if (in.HasExtraChannels()) {
    std::vector<ImageU> extra_channels;
    for (const ImageU& extra_channel : in.extra_channels()) {
      extra_channels.emplace_back(CopyImage(extra_channel));
    }
    store->SetExtraChannels(std::move(extra_channels));
  }

  if (!store->TransformTo(c_desired, pool)) {
    return false;
  }
  *out = store;
  return true;
}

}  // namespace jxl
