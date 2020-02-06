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

#include "jxl/extras/codec_exr.h"

#include <ImfChromaticitiesAttribute.h>
#include <ImfIO.h>
#include <ImfRgbaFile.h>
#include <ImfStandardAttributes.h>

#include <vector>

#include "jxl/color_encoding.h"
#include "jxl/color_management.h"

namespace jxl {

namespace {

namespace OpenEXR = OPENEXR_IMF_NAMESPACE;
namespace Imath = IMATH_NAMESPACE;

class InMemoryIStream : public OpenEXR::IStream {
 public:
  // The data pointed to by `bytes` must outlive the InMemoryIStream.
  explicit InMemoryIStream(const Span<const uint8_t> bytes)
      : IStream(/*fileName=*/""), bytes_(bytes) {}

  bool isMemoryMapped() const override { return true; }
  char* readMemoryMapped(const int n) override {
    JXL_ASSERT(pos_ + n <= bytes_.size());
    char* const result =
        const_cast<char*>(reinterpret_cast<const char*>(bytes_.data() + pos_));
    pos_ += n;
    return result;
  }
  bool read(char c[], const int n) override {
    std::copy_n(readMemoryMapped(n), n, c);
    return pos_ < bytes_.size();
  }

  OpenEXR::Int64 tellg() override { return pos_; }
  void seekg(const OpenEXR::Int64 pos) override {
    JXL_ASSERT(pos + 1 <= bytes_.size());
    pos_ = pos;
  }

 private:
  const Span<const uint8_t> bytes_;
  size_t pos_ = 0;
};

class InMemoryOStream : public OpenEXR::OStream {
 public:
  // `bytes` must outlive the InMemoryOStream.
  explicit InMemoryOStream(PaddedBytes* const bytes)
      : OStream(/*fileName=*/""), bytes_(*bytes) {}

  void write(const char c[], const int n) override {
    if (bytes_.size() < pos_ + n) {
      bytes_.resize(pos_ + n);
    }
    std::copy_n(c, n, bytes_.begin() + pos_);
    pos_ += n;
  }

  OpenEXR::Int64 tellp() override { return pos_; }
  void seekp(const OpenEXR::Int64 pos) override {
    if (bytes_.size() + 1 < pos) {
      bytes_.resize(pos - 1);
    }
    pos_ = pos;
  }

 private:
  PaddedBytes& bytes_;
  size_t pos_ = 0;
};

}  // namespace

Status DecodeImageEXR(Span<const uint8_t> bytes, CodecInOut* io) {
  InMemoryIStream is(bytes);
  OpenEXR::RgbaInputFile input(is);

  if ((input.channels() & OpenEXR::RgbaChannels::WRITE_RGB) !=
      OpenEXR::RgbaChannels::WRITE_RGB) {
    return JXL_FAILURE("only RGB OpenEXR files are supported");
  }
  const bool has_alpha = (input.channels() & OpenEXR::RgbaChannels::WRITE_A) ==
                         OpenEXR::RgbaChannels::WRITE_A;

  const float brightness_multiplier =
      OpenEXR::hasWhiteLuminance(input.header())
          ? OpenEXR::whiteLuminance(input.header()) / kDefaultIntensityTarget
          : 1.f;

  auto image_size = input.displayWindow().size();
  // Size is computed as max - min, but both bounds are inclusive.
  ++image_size.x;
  ++image_size.y;
  Image3F image(image_size.x, image_size.y);
  ImageU alpha;
  if (has_alpha) {
    alpha = ImageU(image_size.x, image_size.y);
  }

  std::vector<OpenEXR::Rgba> row_data(input.dataWindow().size().x + 1);
  input.setFrameBuffer(row_data.data(), /*xStride=*/1, /*yStride=*/0);
  for (int y = 0; y < image_size.y; ++y) {
    input.readPixels(y + input.displayWindow().min.y);
    const OpenEXR::Rgba* const JXL_RESTRICT input_row =
        row_data.data() + input.displayWindow().min.x;
    float* const JXL_RESTRICT rows[] = {
        image.PlaneRow(0, y),
        image.PlaneRow(1, y),
        image.PlaneRow(2, y),
    };
    uint16_t* const JXL_RESTRICT alpha_row = has_alpha ? alpha.Row(y) : nullptr;
    for (int x = 0; x < image_size.x; ++x) {
      const float multiplier = brightness_multiplier * 255.f;
      rows[0][x] = multiplier * input_row[x].r;
      rows[1][x] = multiplier * input_row[x].g;
      rows[2][x] = multiplier * input_row[x].b;
      if (has_alpha) {
        alpha_row[x] = std::numeric_limits<uint16_t>::max() * input_row[x].a;
      }
    }
  }

  ColorEncoding color_encoding;
  color_encoding.tf.SetTransferFunction(TransferFunction::kLinear);
  color_encoding.SetColorSpace(ColorSpace::kRGB);
  PrimariesCIExy primaries = ColorEncoding::SRGB().GetPrimaries();
  CIExy white_point = ColorEncoding::SRGB().GetWhitePoint();
  if (OpenEXR::hasChromaticities(input.header())) {
    const auto& chromaticities = OpenEXR::chromaticities(input.header());
    primaries.r.x = chromaticities.red.x;
    primaries.r.y = chromaticities.red.y;
    primaries.g.x = chromaticities.green.x;
    primaries.g.y = chromaticities.green.y;
    primaries.b.x = chromaticities.blue.x;
    primaries.b.y = chromaticities.blue.y;
    white_point.x = chromaticities.white.x;
    white_point.y = chromaticities.white.y;
  }
  JXL_RETURN_IF_ERROR(color_encoding.SetPrimaries(primaries));
  JXL_RETURN_IF_ERROR(color_encoding.SetWhitePoint(white_point));
  JXL_RETURN_IF_ERROR(color_encoding.CreateICC());

  io->metadata.bits_per_sample = 32;
  io->SetFromImage(std::move(image), color_encoding);
  io->metadata.color_encoding = color_encoding;
  if (has_alpha) {
    io->metadata.alpha_bits = 16;
    io->Main().SetAlpha(std::move(alpha), /*alpha_is_premultiplied=*/true);
  }
  return true;
}

Status EncodeImageEXR(const CodecInOut* io, const ColorEncoding& c_desired,
                      ThreadPool* pool, PaddedBytes* bytes) {
  ColorEncoding c_linear = c_desired;
  c_linear.tf.SetTransferFunction(TransferFunction::kLinear);
  JXL_RETURN_IF_ERROR(c_linear.CreateICC());
  ImageMetadata metadata = io->metadata;
  ImageBundle store(&metadata);
  const ImageBundle* linear;
  JXL_RETURN_IF_ERROR(
      TransformIfNeeded(io->Main(), c_linear, pool, &store, &linear));

  const bool has_alpha = io->Main().HasAlpha();
  const bool alpha_is_premultiplied = io->Main().AlphaIsPremultiplied();

  InMemoryOStream os(bytes);
  OpenEXR::Header header(io->xsize(), io->ysize());
  const PrimariesCIExy& primaries =
      c_linear.HasPrimaries() ? c_linear.GetPrimaries()
                              : ColorEncoding::SRGB().GetPrimaries();
  OpenEXR::Chromaticities chromaticities;
  chromaticities.red = Imath::V2f(primaries.r.x, primaries.r.y);
  chromaticities.green = Imath::V2f(primaries.g.x, primaries.g.y);
  chromaticities.blue = Imath::V2f(primaries.b.x, primaries.b.y);
  chromaticities.white =
      Imath::V2f(c_linear.GetWhitePoint().x, c_linear.GetWhitePoint().y);
  OpenEXR::addChromaticities(header, chromaticities);
  OpenEXR::addWhiteLuminance(header, kDefaultIntensityTarget);

  OpenEXR::RgbaOutputFile output(
      os, header, has_alpha ? OpenEXR::WRITE_RGBA : OpenEXR::WRITE_RGB);
  std::vector<OpenEXR::Rgba> row_data(io->xsize());
  output.setFrameBuffer(row_data.data(), /*xStride=*/1, /*yStride=*/0);

  const float multiplier =
      io->metadata.IntensityTarget() * kIntensityMultiplier / 255.f;
  const float alpha_normalizer =
      has_alpha ? 1.f / ((1 << io->metadata.alpha_bits) - 1) : 0;

  for (size_t y = 0; y < io->ysize(); ++y) {
    const float* const JXL_RESTRICT input_rows[] = {
        linear->color().ConstPlaneRow(0, y),
        linear->color().ConstPlaneRow(1, y),
        linear->color().ConstPlaneRow(2, y),
    };
    if (has_alpha) {
      const uint16_t* const JXL_RESTRICT alpha_row =
          io->Main().alpha().ConstRow(y);
      if (alpha_is_premultiplied) {
        for (size_t x = 0; x < io->xsize(); ++x) {
          const float alpha = alpha_normalizer * alpha_row[x];
          row_data[x] = OpenEXR::Rgba(multiplier * input_rows[0][x],
                                      multiplier * input_rows[1][x],
                                      multiplier * input_rows[2][x], alpha);
        }
      } else {
        for (size_t x = 0; x < io->xsize(); ++x) {
          const float alpha = alpha_normalizer * alpha_row[x];
          row_data[x] =
              OpenEXR::Rgba(multiplier * alpha * input_rows[0][x],
                            multiplier * alpha * input_rows[1][x],
                            multiplier * alpha * input_rows[2][x], alpha);
        }
      }
    } else {
      for (size_t x = 0; x < io->xsize(); ++x) {
        row_data[x] = OpenEXR::Rgba(multiplier * input_rows[0][x],
                                    multiplier * input_rows[1][x],
                                    multiplier * input_rows[2][x], 1.f);
      }
    }
    output.writePixels(/*numScanLines=*/1);
  }
  return true;
}

}  // namespace jxl
