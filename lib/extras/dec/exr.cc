// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/exr.h"

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

  const std::string chPrefix = FindColorLayerPrefix(channels);
  const std::string chNameR = chPrefix + "R";
  const std::string chNameG = chPrefix + "G";
  const std::string chNameB = chPrefix + "B";
  const std::string chNameA = chPrefix + "A";
  const OpenEXR::Channel* chR = channels.findChannel(chNameR.c_str());
  const OpenEXR::Channel* chG = channels.findChannel(chNameG.c_str());
  const OpenEXR::Channel* chB = channels.findChannel(chNameB.c_str());
  const OpenEXR::Channel* chA = channels.findChannel(chNameA.c_str());
  // If we don't have RGB (same type) channels, we'll treat the first
  // channel as grayscale.
  const bool has_rgb = chR != nullptr && chG != nullptr && chB != nullptr &&
                       chR->type == chG->type && chR->type == chB->type;
  const OpenEXR::Channel* chBase = has_rgb ? chR : &channels.begin().channel();
  const std::string chNameBase = has_rgb ? chNameR : channels.begin().name();

  const bool has_alpha =
      chA != nullptr && chA != chBase && chA->type == chBase->type;

  const float intensity_target =
      OpenEXR::hasWhiteLuminance(header) ? OpenEXR::whiteLuminance(header) : 0;

  const Imath::Box2i displayWindow = header.displayWindow();
  const Imath::Box2i dataWindow = header.dataWindow();
  // Size is computed as max - min, but both bounds are inclusive.
  const int imageWidth = displayWindow.max.x - displayWindow.min.x + 1;
  const int imageHeight = displayWindow.max.y - displayWindow.min.y + 1;

  if (!VerifyDimensions<uint32_t>(constraints, imageWidth, imageHeight)) {
    return JXL_FAILURE("image too big");
  }

  ppf->info.xsize = imageWidth;
  ppf->info.ysize = imageHeight;
  ppf->info.num_color_channels = has_rgb ? 3 : 1;

  const JxlDataType data_type =
      chBase->type == OpenEXR::HALF ? JXL_TYPE_FLOAT16 : JXL_TYPE_FLOAT;
  const JxlPixelFormat format{
      /*num_channels=*/ppf->info.num_color_channels + (has_alpha ? 1u : 0u),
      /*data_type=*/data_type,
      /*endianness=*/JXL_NATIVE_ENDIAN,
      /*align=*/0,
  };
  ppf->frames.clear();
  // Allocates the frame buffer.
  {
    JXL_ASSIGN_OR_RETURN(PackedFrame frame,
                         PackedFrame::Create(imageWidth, imageHeight, format));
    ppf->frames.emplace_back(std::move(frame));
  }
  auto& frame = ppf->frames.back();

  // Allocate extra channel images
  uint32_t extraCount = 0;
  size_t extraPixelBytes = 0;
  std::set<const OpenEXR::Channel*> extraChannels;
  for (OpenEXR::ChannelList::ConstIterator it = channels.begin();
       it != channels.end(); ++it) {
    const std::string name = it.name();
    if (has_rgb && (name == chNameR || name == chNameG || name == chNameB))
      continue;
    if (has_alpha && name == chNameA) continue;
    if (name == chNameBase) continue;

    const bool fp16 = it.channel().type == OpenEXR::HALF;
    ++extraCount;
    extraPixelBytes += fp16 ? 2 : 4;
    extraChannels.insert(&it.channel());

    const JxlPixelFormat ec_format{1, fp16 ? JXL_TYPE_FLOAT16 : JXL_TYPE_FLOAT,
                                   JXL_NATIVE_ENDIAN, 0};
    JXL_ASSIGN_OR_RETURN(
        PackedImage ec,
        PackedImage::Create(imageWidth, imageHeight, ec_format));
    frame.extra_channels.emplace_back(std::move(ec));

    PackedExtraChannel pec = {};
    pec.ec_info.bits_per_sample = fp16 ? 16 : 32;
    pec.ec_info.exponent_bits_per_sample = fp16 ? 5 : 8;
    // TODO: detect channel types (depth etc.) based on naming convention
    pec.ec_info.type = JXL_CHANNEL_OPTIONAL;
    pec.name = name;
    ppf->extra_channels_info.emplace_back(std::move(pec));
  }
  ppf->info.num_extra_channels = (has_alpha ? 1 : 0) + extraCount;

  const int row_size = dataWindow.size().x + 1;
  // Number of rows to read at a time.
  // https://www.openexr.com/documentation/ReadingAndWritingImageFiles.pdf
  // recommends reading the whole file at once.
  const int y_chunk_size = displayWindow.size().y + 1;

  const size_t colorChannelBytes = chBase->type == OpenEXR::HALF ? 2 : 4;
  const size_t colorPixelBytes = colorChannelBytes * format.num_channels;
  std::vector<char> input_rows(colorPixelBytes * row_size * y_chunk_size);
  std::vector<char> input_extra_rows(extraPixelBytes * row_size * y_chunk_size);
  for (int start_y = std::max(dataWindow.min.y, displayWindow.min.y);
       start_y <= std::min(dataWindow.max.y, displayWindow.max.y);
       start_y += y_chunk_size) {
    // Inclusive.
    const int end_y = std::min(start_y + y_chunk_size - 1,
                               std::min(dataWindow.max.y, displayWindow.max.y));

    // Setup framebuffer: color/grayscale
    OpenEXR::FrameBuffer fb;
    char* input_rows_ptr =
        input_rows.data() -
        (dataWindow.min.x + start_y * row_size) * colorPixelBytes;
    if (has_rgb) {
      fb.insert(
          chNameR.c_str(),
          OpenEXR::Slice(chR->type, input_rows_ptr + colorChannelBytes * 0,
                         colorPixelBytes, colorPixelBytes * row_size));
      fb.insert(
          chNameG.c_str(),
          OpenEXR::Slice(chG->type, input_rows_ptr + colorChannelBytes * 1,
                         colorPixelBytes, colorPixelBytes * row_size));
      fb.insert(
          chNameB.c_str(),
          OpenEXR::Slice(chB->type, input_rows_ptr + colorChannelBytes * 2,
                         colorPixelBytes, colorPixelBytes * row_size));
    } else {
      fb.insert(
          chNameBase.c_str(),
          OpenEXR::Slice(chBase->type, input_rows_ptr + colorChannelBytes * 0,
                         colorPixelBytes, colorPixelBytes * row_size));
    }

    // Setup framebuffer: alpha
    if (has_alpha)
      fb.insert(
          chNameA.c_str(),
          OpenEXR::Slice(chA->type,
                         input_rows_ptr + colorChannelBytes * (has_rgb ? 3 : 1),
                         colorPixelBytes, colorPixelBytes * row_size));

    // Setup framebuffer: extra channels
    char* extra_rows_ptr =
        input_extra_rows.data() -
        (dataWindow.min.x + start_y * row_size) * extraPixelBytes;
    for (OpenEXR::ChannelList::ConstIterator it = channels.begin();
         it != channels.end(); ++it) {
      if (extraChannels.find(&it.channel()) == extraChannels.end()) {
        continue;
      }
      const size_t size = it.channel().type == OpenEXR::HALF ? 2 : 4;
      fb.insert(it.name(), OpenEXR::Slice(it.channel().type, extra_rows_ptr,
                                          size, size * row_size));
      extra_rows_ptr += size * row_size * (end_y - start_y + 1);
    }

    // Read EXR data
    input.setFrameBuffer(fb);
    input.readPixels(start_y, end_y);

    // Copy read data into the result image
    for (int exr_y = start_y; exr_y <= end_y; ++exr_y) {
      const int image_y = exr_y - displayWindow.min.y;
      const char* const JXL_RESTRICT input_row =
          &input_rows[(exr_y - start_y) * colorPixelBytes * row_size];
      uint8_t* row = static_cast<uint8_t*>(frame.color.pixels()) +
                     frame.color.stride * image_y;

      const int exr_x1 = std::max(dataWindow.min.x, displayWindow.min.x);
      const int exr_x2 = std::min(dataWindow.max.x, displayWindow.max.x);

      const char* exr_ptr =
          input_row + (exr_x1 - dataWindow.min.x) * colorPixelBytes;
      uint8_t* image_ptr =
          row + (exr_x1 - displayWindow.min.x) * colorPixelBytes;
      memcpy(image_ptr, exr_ptr, (exr_x2 - exr_x1 + 1) * colorPixelBytes);

      const char* JXL_RESTRICT input_ec_slice = input_extra_rows.data();
      for (PackedImage& ec : frame.extra_channels) {
        const char* const JXL_RESTRICT input_ec_row =
            input_ec_slice + (exr_y - start_y) * ec.stride;
        uint8_t* ec_row =
            static_cast<uint8_t*>(ec.pixels()) + ec.stride * image_y;
        input_ec_slice += ec.stride * (end_y - start_y + 1);

        const char* exr_ec_ptr =
            input_ec_row + (exr_x1 - dataWindow.min.x) * ec.pixel_stride();
        uint8_t* image_ec_ptr =
            ec_row + (exr_x1 - displayWindow.min.x) * ec.pixel_stride();
        memcpy(image_ec_ptr, exr_ec_ptr,
               (exr_x2 - exr_x1 + 1) * ec.pixel_stride());
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
  ppf->info.bits_per_sample = chBase->type == OpenEXR::HALF ? 16 : 32;
  ppf->info.exponent_bits_per_sample = chBase->type == OpenEXR::HALF ? 5 : 8;
  if (has_alpha) {
    ppf->info.alpha_bits = chA->type == OpenEXR::HALF ? 16 : 32;
    ppf->info.alpha_exponent_bits = chA->type == OpenEXR::HALF ? 5 : 8;
    ppf->info.alpha_premultiplied = JXL_TRUE;
  }
  ppf->info.intensity_target = intensity_target;
  return true;
}

}  // namespace extras
}  // namespace jxl

#endif  // JPEGXL_ENABLE_EXR
