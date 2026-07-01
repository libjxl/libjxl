// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/exr.h"

#include <jxl/codestream_header.h>  // JXL_CHANNEL_OPTIONAL

#include <cstdint>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/packed_image.h"
#include "lib/extras/size_constraints.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"

#if !JPEGXL_ENABLE_EXR

namespace jxl {
namespace extras {
bool CanDecodeEXR() { return false; }

Status DecodeImageEXR(Span<const uint8_t> bytes, const ColorHints& color_hints,
                      PackedPixelFile* ppf,
                      const SizeConstraints* constraints) {
  (void)bytes;
  (void)color_hints;
  (void)ppf;
  (void)constraints;
  return JXL_FAILURE("EXR is not supported");
}
}  // namespace extras
}  // namespace jxl

#else  // JPEGXL_ENABLE_EXR

#include <ImfChannelList.h>
#include <ImfFrameBuffer.h>
#include <ImfIO.h>
#include <ImfInputFile.h>
#include <ImfStandardAttributes.h>
#include <OpenEXRConfig.h>
#include <jxl/color_encoding.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "lib/jxl/base/common.h"
#include "lib/jxl/base/compiler_specific.h"

#ifdef __EXCEPTIONS
#include <IexBaseExc.h>
#define JXL_EXR_THROW_LENGTH_ERROR(M) throw Iex::InputExc(M);
#else  // __EXCEPTIONS
#define JXL_EXR_THROW_LENGTH_ERROR(M) JXL_CRASH()
#endif  // __EXCEPTIONS

namespace jxl {
namespace extras {

namespace {

namespace OpenEXR = OPENEXR_IMF_NAMESPACE;

// OpenEXR::Int64 is deprecated in favor of using uint64_t directly, but using
// uint64_t as recommended causes build failures with previous OpenEXR versions
// on macOS, where the definition for OpenEXR::Int64 was actually not equivalent
// to uint64_t. This alternative should work in all cases.
using ExrInt64 = decltype(std::declval<OpenEXR::IStream>().tellg());

class InMemoryIStream : public OpenEXR::IStream {
 public:
  // The data pointed to by `bytes` must outlive the InMemoryIStream.
  explicit InMemoryIStream(const Span<const uint8_t> bytes)
      : IStream(/*fileName=*/""), bytes_(bytes) {}

  bool isMemoryMapped() const override { return true; }
  char* readMemoryMapped(const int n) override {
    if (pos_ + n < pos_) {
      JXL_EXR_THROW_LENGTH_ERROR("Overflow");
    }
    if (pos_ + n > bytes_.size()) {
      JXL_EXR_THROW_LENGTH_ERROR("Read past end of file");
    }
    char* const result =
        const_cast<char*>(reinterpret_cast<const char*>(bytes_.data() + pos_));
    pos_ += n;
    return result;
  }
  bool read(char c[/*n*/], int n) override {
    // That is not stated in documentation, but the OpenEXR code expects that
    // when requested amount is not accessible and exception is thrown, all
    // the accessible data is read.
    if (pos_ + n < pos_) {
      JXL_EXR_THROW_LENGTH_ERROR("Overflow");
    }
    if (pos_ + n > bytes_.size()) {
      int can_read = static_cast<int>(bytes_.size() - pos_);
      std::copy_n(readMemoryMapped(can_read), can_read, c);
      JXL_EXR_THROW_LENGTH_ERROR("Read past end of file");
    } else {
      std::copy_n(readMemoryMapped(n), n, c);
    }
    return pos_ < bytes_.size();
  }

  ExrInt64 tellg() override { return pos_; }
  void seekg(const ExrInt64 pos) override {
    if (pos >= bytes_.size()) {
      JXL_EXR_THROW_LENGTH_ERROR("Seeks past end of file");
    }
    pos_ = pos;
  }

 private:
  const Span<const uint8_t> bytes_;
  size_t pos_ = 0;
};

}  // namespace

bool CanDecodeEXR() { return true; }

static std::string FindColorLayerPrefix(const OpenEXR::ChannelList& channels) {
  // check if just R,G,B channels exist; use those if present
  if (channels.findChannel("R") != nullptr &&
      channels.findChannel("G") != nullptr &&
      channels.findChannel("B") != nullptr) {
    return "";
  }

  // check channels for presence of R,G,B with common prefix ("layer name"),
  // use the first one that is present
  for (OpenEXR::ChannelList::ConstIterator it = channels.begin();
       it != channels.end(); ++it) {
    std::string name = it.name();
    size_t dotIndex = name.rfind('.');
    if (dotIndex == std::string::npos) continue;
    std::string suffix = name.substr(dotIndex + 1);
    if (suffix != "R" && suffix != "G" && suffix != "B") continue;
    std::string prefix = name.substr(0, dotIndex + 1);
    if (channels.findChannel((prefix + "R").c_str()) != nullptr &&
        channels.findChannel((prefix + "G").c_str()) != nullptr &&
        channels.findChannel((prefix + "B").c_str()) != nullptr) {
      return prefix;
    }
  }

  return "";
}

Status DecodeImageEXR(Span<const uint8_t> bytes, const ColorHints& color_hints,
                      PackedPixelFile* ppf,
                      const SizeConstraints* constraints) {
  InMemoryIStream is(bytes);

#ifdef __EXCEPTIONS
  std::unique_ptr<OpenEXR::InputFile> input_ptr;
  try {
    input_ptr = jxl::make_unique<OpenEXR::InputFile>(is);
  } catch (...) {
    // silently return false if it is not an EXR file
    return false;
  }
  OpenEXR::InputFile& input = *input_ptr;
#else
  OpenEXR::InputFile input(is);
#endif

  const OpenEXR::Header& header = input.header();
  const OpenEXR::ChannelList& channels = header.channels();

  // we don't support subsampled or UINT channels yet
  // TODO: support common cases of subsampling (2x, 4x)
  for (OpenEXR::ChannelList::ConstIterator it = channels.begin();
       it != channels.end(); ++it) {
    const OpenEXR::Channel& ch = it.channel();
    if (ch.type == OpenEXR::UINT) {
      return JXL_FAILURE("OpenEXR files with UINT channels are not supported");
    }
    if (ch.xSampling != 1 || ch.ySampling != 1) {
      return JXL_FAILURE(
          "OpenEXR files sub-sampled channels are not supported");
    }
  }

  const std::string color_prefix = FindColorLayerPrefix(channels);
  const std::string ch_name_r = color_prefix + "R";
  const OpenEXR::Channel* ch_r = channels.findChannel(ch_name_r.c_str());
  const std::string ch_name_g = color_prefix + "G";
  const OpenEXR::Channel* ch_g = channels.findChannel(ch_name_g.c_str());
  const std::string ch_name_b = color_prefix + "B";
  const OpenEXR::Channel* ch_b = channels.findChannel(ch_name_b.c_str());
  const std::string ch_name_a = color_prefix + "A";
  const OpenEXR::Channel* ch_a = channels.findChannel(ch_name_a.c_str());
  // If we don't have RGB (same type) channels, we'll treat the first
  // channel as grayscale.
  bool has_rgb = (ch_r != nullptr) && (ch_g != nullptr) && (ch_b != nullptr);
  bool is_gray = !has_rgb;
  const OpenEXR::Channel* ch_gray =
      is_gray ? &channels.begin().channel() : nullptr;
  const std::string ch_name_gray = is_gray ? channels.begin().name() : "";

  const auto color_type = (is_gray ? ch_gray : ch_r)->type;
  if (has_rgb) {
    if (ch_g->type != color_type || ch_b->type != color_type) {
      return JXL_FAILURE(
          "OpenEXR color channels with different types are not supported yet");
    }
  }

  bool has_alpha = (ch_a != nullptr) && (ch_a != ch_gray);
  if (has_alpha) {
    if (ch_a->type != color_type) {
      return JXL_FAILURE(
          "OpenEXR color channels with different types are not supported yet");
    }
  }

  const float intensity_target =
      OpenEXR::hasWhiteLuminance(header) ? OpenEXR::whiteLuminance(header) : 0;

  const Imath::Box2i display_window = header.displayWindow();
  const Imath::Box2i data_window = header.dataWindow();
  // display_window / data_window bounds come straight from the EXR file header
  // and are not otherwise constrained: an arbitrary range of ints is legal in
  // the file format. Reject coordinates outside +/-2^30 to keep the pointer
  // arithmetic below sane. JXL level 10 already caps image sizes at 2^30, so
  // this is not a real-world limitation. The inclusive window sizes are still
  // computed in 64 bits, because the span of two in-range coordinates can
  // reach 2^31, which does not fit in `int`.
  constexpr int kEXRCoordBound = 1 << 30;
  auto OutOfRange = [](int v) {
    return v < -kEXRCoordBound || v > kEXRCoordBound;
  };
  if (OutOfRange(display_window.min.x) || OutOfRange(display_window.max.x) ||
      OutOfRange(display_window.min.y) || OutOfRange(display_window.max.y) ||
      OutOfRange(data_window.min.x) || OutOfRange(data_window.max.x) ||
      OutOfRange(data_window.min.y) || OutOfRange(data_window.max.y)) {
    return JXL_FAILURE("EXR: window coordinates out of range");
  }
  // TODO(eustas): empty data_window could be valid use case.
  if (display_window.isEmpty() || data_window.isEmpty()) {
    return JXL_FAILURE("EXR: empty window");
  }
  // Size is computed as max - min, but both bounds are inclusive. Compute in
  // 64 bits: `Box2i::size()` subtracts two `int` coordinates, which overflows
  // when the (in-range) span reaches 2^31.
  const int64_t image_width =
      static_cast<int64_t>(display_window.max.x) - display_window.min.x + 1;
  const int64_t image_height =
      static_cast<int64_t>(display_window.max.y) - display_window.min.y + 1;

  if (!VerifyDimensions<uint32_t>(constraints, image_width, image_height)) {
    return JXL_FAILURE("image too big");
  }

  // Apply the same constraints to data_window, since its dimensions drive
  // input buffer allocations and per-row pointer arithmetic below.
  const int64_t data_width =
      static_cast<int64_t>(data_window.max.x) - data_window.min.x + 1;
  const int64_t data_height =
      static_cast<int64_t>(data_window.max.y) - data_window.min.y + 1;
  if (!VerifyDimensions<uint32_t>(constraints, data_width, data_height)) {
    return JXL_FAILURE("EXR: data_window too big");
  }

  // https://www.openexr.com/documentation/ReadingAndWritingImageFiles.pdf
  // recommends reading the whole file at once.
  size_t num_pixels;
  // Width must correspond to data scanlines in file.
  // TODO(eustas): if projection of data_window on X axis is inside of
  // projection of display_window, then we can avoid extra memcpy.
  if (!SafeMul(image_height, data_width, num_pixels)) {
    return JXL_FAILURE("EXR: image too big");
  }

  // Intersect data and display window.
  const int x1 = std::max(data_window.min.x, display_window.min.x);
  const int x2 = std::min(data_window.max.x, display_window.max.x);
  const int y1 = std::max(data_window.min.y, display_window.min.y);
  const int y2 = std::min(data_window.max.y, display_window.max.y);
  const int x_span = x2 - x1 + 1;
  const int y_span = y2 - y1 + 1;

  const auto get_pixel_type = [](OpenEXR::PixelType t) -> JxlDataType {
    return (t == OpenEXR::HALF) ? JXL_TYPE_FLOAT16 : JXL_TYPE_FLOAT;
  };

  const auto get_pixel_stride = [](OpenEXR::PixelType t) -> size_t {
    return (t == OpenEXR::HALF) ? 2 : 4;
  };

  const auto get_bpp = [](OpenEXR::PixelType t) -> size_t {
    return (t == OpenEXR::HALF) ? 16 : 32;
  };

  const auto get_exponent_bits = [](OpenEXR::PixelType t) -> size_t {
    return (t == OpenEXR::HALF) ? 5 : 8;
  };

  uint32_t num_color_channels = is_gray ? 1 : 3;

  ppf->info.xsize = image_width;
  ppf->info.ysize = image_height;
  ppf->info.num_color_channels = num_color_channels;

  const JxlPixelFormat format{
      /*num_channels=*/num_color_channels + (has_alpha ? 1u : 0u),
      /*data_type=*/get_pixel_type(color_type),
      /*endianness=*/JXL_NATIVE_ENDIAN,
      /*align=*/0,
  };
  ppf->frames.clear();
  // Allocates the frame buffer.
  {
    JXL_ASSIGN_OR_RETURN(
        PackedFrame frame,
        PackedFrame::Create(image_width, image_height, format));
    ppf->frames.emplace_back(std::move(frame));
  }
  auto& frame = ppf->frames.back();

  // Allocate extra channel images.
  std::set<const OpenEXR::Channel*> ec_set;
  std::vector<std::vector<char> > ec_data;
  for (OpenEXR::ChannelList::ConstIterator it = channels.begin();
       it != channels.end(); ++it) {
    const std::string name = it.name();
    // Skip {RGB|Gray}(A)
    if (is_gray && (name == ch_name_gray)) continue;
    if (has_alpha && (name == ch_name_a)) continue;
    if (has_rgb) {
      if ((name == ch_name_r) || (name == ch_name_g) || (name == ch_name_b)) {
        continue;
      }
    }
    const OpenEXR::Channel* ch = &it.channel();
    OpenEXR::PixelType t = ch->type;

    ec_set.insert(ch);
    size_t pixel_stride = get_pixel_stride(t);
    size_t volume;
    if (!SafeMul(pixel_stride, num_pixels, volume)) {
      return JXL_FAILURE("EXR: image too big");
    }
    std::vector<char> storage(volume);
    ec_data.emplace_back(std::move(storage));

    const JxlPixelFormat ec_format{/*num_channels=*/1,
                                   /*data_type=*/get_pixel_type(t),
                                   /*endianness=*/JXL_NATIVE_ENDIAN,
                                   /*align=*/0};
    JXL_ASSIGN_OR_RETURN(
        PackedImage ec,
        PackedImage::Create(image_width, image_height, ec_format));
    frame.extra_channels.emplace_back(std::move(ec));
    JXL_DASSERT(frame.extra_channels.back().pixel_stride() == pixel_stride);

    PackedExtraChannel pec = {};
    pec.ec_info.bits_per_sample = get_bpp(t);
    pec.ec_info.exponent_bits_per_sample = get_exponent_bits(t);
    // TODO: detect channel types (depth etc.) based on naming convention
    pec.ec_info.type = JXL_CHANNEL_OPTIONAL;
    pec.name = name;
    ppf->extra_channels_info.emplace_back(std::move(pec));
  }
  ppf->info.num_extra_channels =
      (has_alpha ? 1 : 0) + frame.extra_channels.size();

  const size_t color_channel_bytes = get_pixel_stride(color_type);
  const size_t color_pixel_bytes = color_channel_bytes * format.num_channels;
  size_t color_data_size;
  if (!SafeMul(color_pixel_bytes, num_pixels, color_data_size)) {
    return JXL_FAILURE("EXR: image too big");
  }
  // Interleaved RGB{A} / Gray{A}
  std::vector<char> color_data(color_data_size);

  // If intersection is empty, then image is just zeroes.
  if (x_span > 0 && y_span > 0) {
    // Setup framebuffer: color/grayscale
    OpenEXR::FrameBuffer fb;
    size_t x_stride = color_pixel_bytes;
    size_t y_stride = x_stride * static_cast<size_t>(data_width);
    char* virtual_image_origin = color_data.data();
    // Offset to match output to allocation start; when EXR puts pixel at
    // (data_window.min.x, y1) it goes to 0-th element.
    virtual_image_origin -=
        static_cast<ptrdiff_t>(y1) * static_cast<ptrdiff_t>(y_stride);
    virtual_image_origin -= static_cast<ptrdiff_t>(data_window.min.x) *
                            static_cast<ptrdiff_t>(x_stride);
    if (has_rgb) {
      fb.insert(ch_name_r.c_str(),
                OpenEXR::Slice(color_type,
                               virtual_image_origin + color_channel_bytes * 0,
                               x_stride, y_stride));
      fb.insert(ch_name_g.c_str(),
                OpenEXR::Slice(color_type,
                               virtual_image_origin + color_channel_bytes * 1,
                               x_stride, y_stride));
      fb.insert(ch_name_b.c_str(),
                OpenEXR::Slice(color_type,
                               virtual_image_origin + color_channel_bytes * 2,
                               x_stride, y_stride));
    } else {
      fb.insert(ch_name_gray.c_str(),
                OpenEXR::Slice(color_type,
                               virtual_image_origin + color_channel_bytes * 0,
                               x_stride, y_stride));
    }

    // Setup framebuffer: alpha
    if (has_alpha) {
      fb.insert(ch_name_a.c_str(),
                OpenEXR::Slice(color_type,
                               virtual_image_origin +
                                   color_channel_bytes * (has_rgb ? 3 : 1),
                               x_stride, y_stride));
    }

    // Setup framebuffer: extra channels
    size_t ec_data_idx = 0;
    for (OpenEXR::ChannelList::ConstIterator it = channels.begin();
         it != channels.end(); ++it) {
      const OpenEXR::Channel* ch = &it.channel();
      if (ec_set.find(ch) == ec_set.end()) {
        continue;
      }
      auto& ec = ec_data[ec_data_idx++];
      size_t ec_x_stride = get_pixel_stride(ch->type);
      size_t ec_y_stride = ec_x_stride * static_cast<size_t>(data_width);

      char* ec_virtual_image_origin = ec.data();
      // Offset to match output to allocation start; when EXR puts pixel at
      // (data_window.min.x, y1) it goes to 0-th element.
      ec_virtual_image_origin -=
          static_cast<ptrdiff_t>(y1) * static_cast<ptrdiff_t>(ec_y_stride);
      ec_virtual_image_origin -= static_cast<ptrdiff_t>(data_window.min.x) *
                                 static_cast<ptrdiff_t>(ec_x_stride);

      fb.insert(it.name(), OpenEXR::Slice(ch->type, ec_virtual_image_origin,
                                          ec_x_stride, ec_y_stride));
    }

    // Read EXR data
    input.setFrameBuffer(fb);
    input.readPixels(y1, y2);

    const int x_data = x1 - data_window.min.x;
    const int x_out = x1 - display_window.min.x;
    JXL_DASSERT(x_out >= 0);
    JXL_DASSERT(x_out + x_span <= image_width);

    // Copy read data into the result image.
    // TODO(eustas): should we deal with unpopulated pixels?
    for (int y = y1; y <= y2; ++y) {  // Scanline index
      const int y_data = y - y1;
      const int y_out = y - display_window.min.y;
      JXL_DASSERT(y_out >= 0);
      JXL_DASSERT(y_out < image_height);
      const char* const JXL_RESTRICT data_ptr =
          &color_data[x_data * color_pixel_bytes + y_data * y_stride];
      uint8_t* pixels = static_cast<uint8_t*>(frame.color.pixels());
      uint8_t* image_ptr =
          pixels + x_out * color_pixel_bytes + y_out * frame.color.stride;
      memcpy(image_ptr, data_ptr, x_span * color_pixel_bytes);

      for (size_t ec_idx = 0; ec_idx < frame.extra_channels.size(); ++ec_idx) {
        PackedImage& ec = frame.extra_channels[ec_idx];
        auto& data = ec_data[ec_idx];
        size_t ec_x_stride = ec.pixel_stride();
        size_t ec_y_stride = ec_x_stride * static_cast<size_t>(data_width);
        const char* const JXL_RESTRICT ec_data_ptr =
            &data[x_data * ec_x_stride + y_data * ec_y_stride];
        uint8_t* ec_pixels = static_cast<uint8_t*>(ec.pixels());
        uint8_t* ec_image_ptr =
            ec_pixels + x_out * ec_x_stride + y_out * ec.stride;
        memcpy(ec_image_ptr, ec_data_ptr, x_span * ec_x_stride);
      }
    }
  }

  ppf->color_encoding.transfer_function = JXL_TRANSFER_FUNCTION_LINEAR;
  ppf->color_encoding.color_space =
      has_rgb ? JXL_COLOR_SPACE_RGB : JXL_COLOR_SPACE_GRAY;
  ppf->color_encoding.primaries = JXL_PRIMARIES_SRGB;
  ppf->color_encoding.white_point = JXL_WHITE_POINT_D65;
  if (OpenEXR::hasChromaticities(header)) {
    ppf->color_encoding.primaries = JXL_PRIMARIES_CUSTOM;
    ppf->color_encoding.white_point = JXL_WHITE_POINT_CUSTOM;
    const auto& chromaticities = OpenEXR::chromaticities(header);
    ppf->color_encoding.primaries_red_xy[0] = chromaticities.red.x;
    ppf->color_encoding.primaries_red_xy[1] = chromaticities.red.y;
    ppf->color_encoding.primaries_green_xy[0] = chromaticities.green.x;
    ppf->color_encoding.primaries_green_xy[1] = chromaticities.green.y;
    ppf->color_encoding.primaries_blue_xy[0] = chromaticities.blue.x;
    ppf->color_encoding.primaries_blue_xy[1] = chromaticities.blue.y;
    ppf->color_encoding.white_point_xy[0] = chromaticities.white.x;
    ppf->color_encoding.white_point_xy[1] = chromaticities.white.y;
  }

  // EXR uses binary16 or binary32 floating point format.
  ppf->info.bits_per_sample = get_bpp(color_type);
  ppf->info.exponent_bits_per_sample = get_exponent_bits(color_type);
  if (has_alpha) {
    ppf->info.alpha_bits = ppf->info.bits_per_sample;
    ppf->info.alpha_exponent_bits = ppf->info.exponent_bits_per_sample;
    ppf->info.alpha_premultiplied = JXL_TRUE;
  }
  ppf->info.intensity_target = intensity_target;
  return true;
}

}  // namespace extras
}  // namespace jxl

#endif  // JPEGXL_ENABLE_EXR
