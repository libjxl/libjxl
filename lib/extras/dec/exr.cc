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
#include <ImfCompression.h>
#include <ImfFrameBuffer.h>
#include <ImfIO.h>
#include <ImfInputFile.h>
#include <ImfStandardAttributes.h>
#include <OpenEXRConfig.h>
#include <jxl/color_encoding.h>
#include <jxl/types.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
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

  // Do not advertise memory mapping. When an OpenEXR::IStream reports itself
  // as memory-mapped, OpenEXR reads a scanline's pixel data straight from the
  // pointer returned by readMemoryMapped(), using the size derived from the
  // image header rather than the size of the on-disk chunk. A file whose chunk
  // declares fewer bytes than a full scanline then makes OpenEXR read past the
  // end of our input buffer (a heap out-of-bounds read reachable from cjxl on
  // an untrusted .exr file). The copying read() path below hands OpenEXR a
  // buffer that it sizes itself, so an over-read stays within OpenEXR's own
  // allocation and is caught by its chunk-size validation instead. The data is
  // already in memory, so this costs one extra copy per scanline and no I/O.
  bool isMemoryMapped() const override { return false; }
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
      MarkRead(pos_, pos_ + can_read);
      std::copy_n(readMemoryMapped(can_read), can_read, c);
      JXL_EXR_THROW_LENGTH_ERROR("Read past end of file");
    } else {
      MarkRead(pos_, pos_ + n);
      std::copy_n(readMemoryMapped(n), n, c);
    }
    return pos_ < bytes_.size();
  }

  // Number of distinct input bytes handed out by read() so far. Bytes that are
  // read again, e.g. by two chunks whose offsets overlap, are counted once.
  uint64_t distinct_bytes_read() const { return distinct_bytes_read_; }

  ExrInt64 tellg() override { return pos_; }
  void seekg(const ExrInt64 pos) override {
    if (pos >= bytes_.size()) {
      JXL_EXR_THROW_LENGTH_ERROR("Seeks past end of file");
    }
    pos_ = pos;
  }

 private:
  // Records that [begin, end) was read. read_ holds disjoint, non-adjacent
  // ranges, keyed by their first byte and mapped to one past their last byte.
  void MarkRead(size_t begin, size_t end) {
    if (begin >= end) return;
    uint64_t fresh = end - begin;
    size_t merged_begin = begin;
    size_t merged_end = end;
    auto it = read_.upper_bound(begin);
    if (it != read_.begin() && std::prev(it)->second >= begin) --it;
    while (it != read_.end() && it->first <= merged_end) {
      const size_t overlap_begin = std::max(it->first, begin);
      const size_t overlap_end = std::min(it->second, end);
      if (overlap_end > overlap_begin) fresh -= overlap_end - overlap_begin;
      merged_begin = std::min(merged_begin, it->first);
      merged_end = std::max(merged_end, it->second);
      it = read_.erase(it);
    }
    read_.emplace(merged_begin, merged_end);
    distinct_bytes_read_ += fresh;
  }

  const Span<const uint8_t> bytes_;
  size_t pos_ = 0;
  std::map<size_t, size_t> read_;
  uint64_t distinct_bytes_read_ = 0;
};

// Number of scan lines in one chunk of the file: the scan lines that OpenEXR
// stores, and decodes, as a unit. For tiled files, that is the height of a
// row of tiles. Unknown compression methods get the largest scan line count
// of the known ones; reading a band then covers several chunks or decodes one
// chunk more than once, which costs time but not correctness.
int64_t ScanLinesPerChunk(const OpenEXR::Header& header) {
  if (header.hasTileDescription()) {
    return std::max<int64_t>(1, header.tileDescription().ySize);
  }
  switch (header.compression()) {
    case OpenEXR::NO_COMPRESSION:
    case OpenEXR::RLE_COMPRESSION:
    case OpenEXR::ZIPS_COMPRESSION:
      return 1;
    case OpenEXR::ZIP_COMPRESSION:
    case OpenEXR::PXR24_COMPRESSION:
      return 16;
    case OpenEXR::PIZ_COMPRESSION:
    case OpenEXR::B44_COMPRESSION:
    case OpenEXR::B44A_COMPRESSION:
    case OpenEXR::DWAA_COMPRESSION:
      return 32;
    default:
      return 256;
  }
}

// How many bytes of scan lines a valid file can decode from one byte of
// chunk data, with a margin of at least two over what the compression method
// reaches on black frames: NONE stores samples as they are, RLE stores at most
// 128 bytes in 2 (64:1), ZIP and ZIPS are limited by deflate (1032:1), PXR24
// packs FLOAT samples into 24 bits before deflate (1376:1), B44A stores a flat
// 4x4 block of HALF samples in 3 bytes (32:3), and PIZ reaches about 400:1.
// DWAA and DWAB store the channels that they do not compress with a DCT (such
// as alpha layers) with RLE and zlib, so black frames with many such channels
// reach about 60000:1. Methods unknown to this code get the DWA bound.
uint64_t MaxDecodedBytesPerInputByte(const OpenEXR::Header& header) {
  switch (header.compression()) {
    case OpenEXR::NO_COMPRESSION:
      return 2;
    case OpenEXR::RLE_COMPRESSION:
      return 128;
    case OpenEXR::B44_COMPRESSION:
    case OpenEXR::B44A_COMPRESSION:
      return 32;
    case OpenEXR::ZIPS_COMPRESSION:
    case OpenEXR::ZIP_COMPRESSION:
      return 2048;
    case OpenEXR::PXR24_COMPRESSION:
    case OpenEXR::PIZ_COMPRESSION:
      return 4096;
    default:
      return uint64_t{1} << 17;
  }
}

struct FreeDeleter {
  void operator()(void* p) const { free(p); }
};
using ScanlineBuffer = std::unique_ptr<char, FreeDeleter>;

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

  // Bytes of the input that follow the header and its chunk offset table.
  // Pixel data can only come from there, so no number of bands can be paid for
  // with more input than this.
  uint64_t input_after_header = 0;
  {
    const ExrInt64 header_end = is.tellg();
    if (header_end > 0 && static_cast<uint64_t>(header_end) <= bytes.size()) {
      input_after_header = static_cast<uint64_t>(bytes.size()) -
                           static_cast<uint64_t>(header_end);
    }
  }

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

    const JxlPixelFormat ec_format{/*num_channels=*/1,
                                   /*data_type=*/get_pixel_type(t),
                                   /*endianness=*/JXL_NATIVE_ENDIAN,
                                   /*align=*/0};
    JXL_ASSIGN_OR_RETURN(
        PackedImage ec,
        PackedImage::Create(image_width, image_height, ec_format));
    frame.extra_channels.emplace_back(std::move(ec));
    JXL_DASSERT(frame.extra_channels.back().pixel_stride() ==
                get_pixel_stride(t));

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

  // If intersection is empty, then image is just zeroes.
  if (x_span > 0 && y_span > 0) {
    // The scan lines are read one band at a time, into scanline storage that
    // holds a single band, and each band is copied into the output image
    // before the next one is read. A band is the part of one chunk of the file
    // (see ScanLinesPerChunk) that lies inside the display window, or of as
    // many whole chunks as fit in kMinBandBytes, so that narrow images are not
    // read a few scan lines at a time. OpenEXR only decodes a chunk when its
    // scan lines are read, so reading the whole image at once would allocate
    // the scanline storage of every channel that the header declares, for the
    // whole data window, before finding out whether the file holds any pixel
    // data at all. A file whose chunks are missing or point to the wrong place
    // now fails while reading the first band that needs them.
    //
    // OpenEXR 3.1 also accepts a chunk whose payload is shorter than its scan
    // lines need, and fills the rest from stale buffer contents, so a chunk of
    // a few bytes can stand for any number of channels. Decoded scan lines
    // therefore have to be paid for with input: the bands read so far may
    // decode to at most MaxDecodedBytesPerInputByte() times the distinct input
    // bytes that OpenEXR read for them, plus kDecodedBytesAllowance. This is
    // checked before a band is copied into the output image.
    const uint64_t max_decoded_per_input_byte =
        MaxDecodedBytesPerInputByte(header);
    constexpr uint64_t kDecodedBytesAllowance = uint64_t{16} << 20;
    constexpr size_t kMinBandBytes = size_t{1} << 20;

    const size_t width = static_cast<size_t>(data_width);
    // Bytes of scanline storage per scan line, for all channels.
    size_t line_pixel_bytes = color_pixel_bytes;
    for (const PackedImage& ec : frame.extra_channels) {
      line_pixel_bytes += ec.pixel_stride();
    }
    size_t line_bytes;
    if (!SafeMul(line_pixel_bytes, width, line_bytes)) {
      return JXL_FAILURE("EXR: image too big");
    }
    const int64_t chunk_lines = ScanLinesPerChunk(header);
    int64_t band_chunks = 1;
    if (line_bytes < kMinBandBytes / static_cast<size_t>(chunk_lines)) {
      band_chunks = static_cast<int64_t>(
          kMinBandBytes / (static_cast<size_t>(chunk_lines) * line_bytes));
    }
    const int64_t chunk_lines_per_band = chunk_lines * band_chunks;
    const size_t band_lines =
        static_cast<size_t>(std::min<int64_t>(chunk_lines_per_band, y_span));

    // Total scanline storage of one full band, for all channels.
    size_t band_bytes;
    if (!SafeMul(line_bytes, band_lines, band_bytes)) {
      return JXL_FAILURE("EXR: image too big");
    }

    // A band has to be paid for with input: the bands read so far, plus this
    // one, may decode to at most MaxDecodedBytesPerInputByte() times the input
    // that follows the header, plus kDecodedBytesAllowance. The check after
    // readPixels() below applies the same rule to the bytes that were actually
    // read; this one runs before the band's scanline storage is allocated and
    // before readPixels() touches it. A header can declare a band of gigabytes
    // while the file holds a few hundred bytes, and OpenEXR 3.1 fills a band
    // from chunks that are shorter than their scan lines need without
    // complaining, so such a band has to be refused on paper first.
    const auto band_is_paid_for = [&](uint64_t decoded,
                                      size_t bytes_of_band) -> bool {
      if (bytes_of_band > UINT64_MAX - decoded) return false;
      const uint64_t needed = decoded + bytes_of_band;
      if (needed <= kDecodedBytesAllowance) return true;
      return (needed - kDecodedBytesAllowance - 1) /
                 max_decoded_per_input_byte <
             input_after_header;
    };
    if (!band_is_paid_for(0, band_bytes)) {
      return JXL_FAILURE("EXR: too little pixel data for the scan lines");
    }

    // Scanline storage of one band. It is not initialized: readPixels() writes
    // every sample of the band before it is copied out.
    const auto allocate_band = [&](size_t pixel_bytes,
                                   ScanlineBuffer* buffer) -> Status {
      size_t size;
      if (!SafeMul(pixel_bytes, width, size) ||
          !SafeMul(size, band_lines, size)) {
        return JXL_FAILURE("EXR: image too big");
      }
      buffer->reset(static_cast<char*>(malloc(std::max<size_t>(1, size))));
      if (!*buffer) return JXL_FAILURE("EXR: out of memory");
      return true;
    };
    ScanlineBuffer color_data;
    JXL_RETURN_IF_ERROR(allocate_band(color_pixel_bytes, &color_data));
    std::vector<ScanlineBuffer> ec_data(frame.extra_channels.size());
    for (size_t ec_idx = 0; ec_idx < ec_data.size(); ++ec_idx) {
      JXL_RETURN_IF_ERROR(allocate_band(
          frame.extra_channels[ec_idx].pixel_stride(), &ec_data[ec_idx]));
    }

    const size_t x_stride = color_pixel_bytes;
    const size_t y_stride = x_stride * width;
    const int x_data = x1 - data_window.min.x;
    const int x_out = x1 - display_window.min.x;
    JXL_DASSERT(x_out >= 0);
    JXL_DASSERT(x_out + x_span <= image_width);

    uint64_t decoded_bytes = 0;
    const uint64_t read_before_pixels = is.distinct_bytes_read();
    for (int64_t band_begin = y1; band_begin <= y2;) {
      const int64_t chunk = (band_begin - data_window.min.y) / chunk_lines;
      const int64_t band_end = std::min<int64_t>(
          y2, data_window.min.y + (chunk + band_chunks) * chunk_lines - 1);

      // Offset to match output to allocation start; when EXR puts pixel at
      // (data_window.min.x, band_begin) it goes to 0-th element.
      const auto virtual_origin = [&](char* data, size_t pixel_bytes) {
        return data -
               static_cast<ptrdiff_t>(band_begin) *
                   static_cast<ptrdiff_t>(pixel_bytes * width) -
               static_cast<ptrdiff_t>(data_window.min.x) *
                   static_cast<ptrdiff_t>(pixel_bytes);
      };

      // Setup framebuffer: color/grayscale
      OpenEXR::FrameBuffer fb;
      char* virtual_image_origin = virtual_origin(color_data.get(), x_stride);
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
        const size_t ec_x_stride = get_pixel_stride(ch->type);
        const size_t ec_y_stride = ec_x_stride * width;
        char* ec_virtual_image_origin =
            virtual_origin(ec_data[ec_data_idx++].get(), ec_x_stride);
        fb.insert(it.name(), OpenEXR::Slice(ch->type, ec_virtual_image_origin,
                                            ec_x_stride, ec_y_stride));
      }

      const size_t num_lines = static_cast<size_t>(band_end - band_begin + 1);
      if (!band_is_paid_for(decoded_bytes, line_bytes * num_lines)) {
        return JXL_FAILURE("EXR: too little pixel data for the scan lines");
      }

      // Read EXR data
      input.setFrameBuffer(fb);
      input.readPixels(static_cast<int>(band_begin),
                       static_cast<int>(band_end));

      decoded_bytes += static_cast<uint64_t>(line_bytes) * num_lines;
      const uint64_t read_bytes = is.distinct_bytes_read() - read_before_pixels;
      if (decoded_bytes > kDecodedBytesAllowance &&
          (decoded_bytes - kDecodedBytesAllowance - 1) /
                  max_decoded_per_input_byte >=
              read_bytes) {
        return JXL_FAILURE("EXR: too little pixel data for the scan lines");
      }

      // Copy read data into the result image.
      // TODO(eustas): should we deal with unpopulated pixels?
      for (int64_t y = band_begin; y <= band_end; ++y) {  // Scanline index
        const size_t y_data = static_cast<size_t>(y - band_begin);
        const int64_t y_out = y - display_window.min.y;
        JXL_DASSERT(y_out >= 0);
        JXL_DASSERT(y_out < image_height);
        const char* const JXL_RESTRICT data_ptr =
            color_data.get() + x_data * color_pixel_bytes + y_data * y_stride;
        uint8_t* pixels = static_cast<uint8_t*>(frame.color.pixels());
        uint8_t* image_ptr =
            pixels + x_out * color_pixel_bytes + y_out * frame.color.stride;
        memcpy(image_ptr, data_ptr, x_span * color_pixel_bytes);

        for (size_t ec_idx = 0; ec_idx < frame.extra_channels.size();
             ++ec_idx) {
          PackedImage& ec = frame.extra_channels[ec_idx];
          size_t ec_x_stride = ec.pixel_stride();
          size_t ec_y_stride = ec_x_stride * width;
          const char* const JXL_RESTRICT ec_data_ptr = ec_data[ec_idx].get() +
                                                       x_data * ec_x_stride +
                                                       y_data * ec_y_stride;
          uint8_t* ec_pixels = static_cast<uint8_t*>(ec.pixels());
          uint8_t* ec_image_ptr =
              ec_pixels + x_out * ec_x_stride + y_out * ec.stride;
          memcpy(ec_image_ptr, ec_data_ptr, x_span * ec_x_stride);
        }
      }
      band_begin = band_end + 1;
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
