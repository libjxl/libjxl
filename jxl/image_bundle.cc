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
      ib->AlphaIsPremultiplied(), alpha, metadata->alpha_bits, bits_per_sample,
      big_endian, temp_intervals);
  JXL_RETURN_IF_ERROR(external.IsHealthy());
  AllocateAndFill(external, out);
  return true;
}

}  // namespace

ImageMetadata::ImageMetadata() { Bundle::Init(this); }
ImageMetadata2::ImageMetadata2() { Bundle::Init(this); }
OpsinInverseMatrix::OpsinInverseMatrix() { Bundle::Init(this); }
IntensityTargetInfo::IntensityTargetInfo() { Bundle::Init(this); }

Status ReadImageMetadata(BitReader* JXL_RESTRICT reader,
                         ImageMetadata* JXL_RESTRICT metadata) {
  return Bundle::Read(reader, metadata);
}

Status WriteImageMetadata(const ImageMetadata& metadata,
                          BitWriter* JXL_RESTRICT writer, size_t layer,
                          AuxOut* aux_out) {
  return Bundle::Write(metadata, writer, layer, aux_out);
}

void ImageBundle::ShrinkTo(size_t xsize, size_t ysize) {
  color_.ShrinkTo(xsize, ysize);
  if (HasAlpha()) alpha_.ShrinkTo(xsize, ysize);
  if (HasDepth()) {
    depth_.ShrinkTo(DepthSize(xsize), DepthSize(ysize));
  }
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

  const ImageU* alpha = HasAlpha() ? &alpha_ : nullptr;
  const size_t alpha_bits = HasAlpha() ? metadata_->alpha_bits : 0;
  const bool big_endian = !IsLittleEndian();
  CodecIntervals temp_intervals;
  const ExternalImage external(pool, color_, Rect(color_), c_current_,
                               c_desired, HasAlpha(), AlphaIsPremultiplied(),
                               alpha, alpha_bits, 32, big_endian,
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

  if (metadata_->HasAlpha() != HasAlpha()) {
    JXL_ABORT("MD alpha_bits %u IB alpha %zu x %zu\n", metadata_->alpha_bits,
              alpha_.xsize(), alpha_.ysize());
  }

  const uint32_t alpha_bits = metadata_->alpha_bits;
  JXL_CHECK(alpha_bits <= 16);

  JXL_CHECK(metadata_->m2.HasDepth() == HasDepth());

  JXL_CHECK(metadata_->m2.num_extra_channels == extra_channels_.size());
}

void ImageBundle::VerifySizes() const {
  const size_t xs = xsize();
  const size_t ys = ysize();
  // Can only verify alpha/depth after SetFrom* or SetPlanes is called.
  if (xs != 0 && ys != 0) {
    if (HasAlpha()) {
      JXL_CHECK(DivCeil(alpha_.xsize(), kBlockDim) == DivCeil(xs, kBlockDim));
      JXL_CHECK(DivCeil(alpha_.ysize(), kBlockDim) == DivCeil(ys, kBlockDim));
    }

    if (HasDepth()) {
      JXL_CHECK(depth_.xsize() == DepthSize(xs) &&
                depth_.ysize() == DepthSize(ys));
    }
  }

  if (HasExtraChannels()) {
    JXL_CHECK(xs != 0 && ys != 0);
    for (const ImageU& plane : extra_channels_) {
      JXL_CHECK(plane.xsize() == xs && plane.ysize() == ys);
    }
  }
}

size_t ImageBundle::DetectRealBitdepth() const {
  JXL_CHECK(metadata_->floating_point_sample == false);
  const size_t orig_d = metadata_->bits_per_sample;
  const size_t maxval = (1 << orig_d) - 1;
  const double factor = maxval / 255.;
  size_t real_d = 1;
  double f = maxval;

  // Start with assuming that 1-bit is enough, then go over all pixels and check
  // if the original value can be reconstructed (within a margin of 0.6) at that
  // bit depth (it won't, unless it's a pure black & white image). As soon as
  // you find a pixel that needs more bit depth, you increment the assumed
  // bit-depth and try again. You don't need to revisit the pixels that were
  // already OK at the lower bit depth (you can assume they will still be OK at
  // the higher bit depth), but you do need to revisit the current pixel.
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < ysize(); y++) {
      const float* JXL_RESTRICT row = color_.PlaneRow(c, y);
      for (size_t x = 0; x < xsize(); x++) {
        int32_t po = row[x] * factor + 0.5;
        // allow a rescaling error of +- 0.6
        // if you set this to 1.0 or more, the 'real' bit depth will be too low
        // (e.g. 7-bit becomes enough to encode 8-bit) if you set this to 0.5 or
        // less, it will fail to find the real bit depth
        if (std::abs(row[x] * factor - (po >> (orig_d - real_d)) * f) > 0.6) {
          real_d++;
          if (real_d == orig_d) return orig_d;
          f = ((double)maxval) / ((1 << real_d) - 1);
          x--;
        }
      }
    }
  }
  JXL_WARNING("Interpreting input as %zu-bit (nominally it is %zu-bit)", real_d,
              orig_d);
  return real_d;
}

void ImageBundle::RemoveAlpha() {
  alpha_ = ImageU();
  JXL_ASSERT(!HasAlpha());
  metadata_->alpha_bits = 0;  // maintain invariant for VerifyMetadata
}

void ImageBundle::SetAlpha(ImageU&& alpha, bool alpha_is_premultiplied) {
  JXL_CHECK(alpha.xsize() != 0 && alpha.ysize() != 0);
  JXL_CHECK(metadata_->alpha_bits != 0);
  alpha_ = std::move(alpha);
  alpha_is_premultiplied_ = alpha_is_premultiplied;
  VerifySizes();
}

void ImageBundle::PremultiplyAlphaIfNeeded(ThreadPool* pool) {
  JXL_ASSERT(HasAlpha());
  JXL_CHECK(alpha_.xsize() == color_.xsize() &&
            alpha_.ysize() == color_.ysize());
  if (alpha_is_premultiplied_) return;
  alpha_is_premultiplied_ = true;
  RunOnPool(
      pool, 0, alpha_.ysize(), ThreadPool::SkipInit(),
      [this](const int y, const int /*thread*/) {
        PremultiplyAlpha(color_.PlaneRow(0, y), color_.PlaneRow(1, y),
                         color_.PlaneRow(2, y), alpha_.ConstRow(y),
                         metadata_->alpha_bits, color_.xsize());
      },
      "premultiply alpha");
}

void ImageBundle::SetDepth(ImageU&& depth) {
  JXL_CHECK(depth.xsize() != 0 && depth.ysize() != 0);
  JXL_CHECK(metadata_->m2.depth_bits != 0);
  depth_ = std::move(depth);
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
  if (!store->TransformTo(c_desired, pool)) {
    return false;
  }
  *out = store;
  return true;
}

}  // namespace jxl
