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

#include "jpegxl/decode.h"

#include "jxl/base/byte_order.h"
#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/brunsli.h"
#include "jxl/fields.h"
#include "jxl/headers.h"
#include "jxl/memory_manager_internal.h"

namespace {
// Checks if a + b > size, taking possible integer overflow into account.
bool OutOfBounds(size_t a, size_t b, size_t size) {
  size_t pos = a + b;
  if (pos > size) return true;
  if (pos < a) return true;  // overflow happened
  return false;
}

JXL_INLINE size_t InitialBasicInfoSizeHint() {
  // Amount of bytes before the start of the codestream in the container format,
  // assuming that the codestream is the first box after the signature and
  // filetype boxes. 12 bytes signature box + 20 bytes filetype box + 16 bytes
  // codestream box length + name + optional XLBox length.
  const size_t container_header_size = 48;

  // Worst-case amount of bytes for basic info of the JPEG XL codestream header,
  // that is all information up to and including extra_channel_bits. Up to
  // around 2 bytes signature + 8 bytes SizeHeader + 31 bytes ColourEncoding + 4
  // bytes rest of ImageMetadata + 5 bytes part of ImageMetadata2.
  // TODO(lode): recompute and update this value when alpha_bits is moved to
  // extra channels info.
  const size_t max_codestream_basic_info_size = 50;

  return container_header_size + max_codestream_basic_info_size;
}

// Debug-printing failure macro similar to JXL_FAILURE, but for the status code
// JXL_DEC_ERROR
#ifdef JXL_CRASH_ON_ERROR
#define JXL_API_ERROR(format, ...)                                           \
  (::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, ##__VA_ARGS__), \
   ::jxl::Abort(), JPEGXL_DEC_ERROR)
#else  // JXL_CRASH_ON_ERROR
#define JXL_API_ERROR(format, ...)                                             \
  (((JXL_DEBUG_ON_ERROR) &&                                                    \
    ::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, ##__VA_ARGS__)), \
   JPEGXL_DEC_ERROR)
#endif  // JXL_CRASH_ON_ERROR

}  // namespace

uint32_t JpegxlDecoderVersion(void) {
  return JPEGXL_MAJOR_VERSION * 1000000 + JPEGXL_MINOR_VERSION * 1000 +
         JPEGXL_PATCH_VERSION;
}

enum JpegxlSignature JpegxlSignatureCheck(const uint8_t* buf, size_t len) {
  if (len == 0) return JPEGXL_SIG_NOT_ENOUGH_BYTES;

  // Transcoded JPEG
  if (len >= 1 && buf[0] == 0x0A) {
    jxl::BrunsliFileSignature brn =
        IsBrunsliFile(jxl::Span<const uint8_t>(buf, len));
    if (brn == jxl::BrunsliFileSignature::kBrunsli) {
      return JPEGXL_SIG_VALID;
    } else if (brn == jxl::BrunsliFileSignature::kNotEnoughData) {
      return JPEGXL_SIG_NOT_ENOUGH_BYTES;
    }
  }

  // Marker: JPEG1 or JPEG XL
  if (len >= 1 && buf[0] == 0xff) {
    if (len < 2) {
      return JPEGXL_SIG_NOT_ENOUGH_BYTES;
    } else if (buf[1] == jxl::kCodestreamMarker || buf[1] == 0xD8) {
      return JPEGXL_SIG_VALID;
    }
  }

  // JPEG XL container
  if (len >= 1 && buf[0] == 0) {
    if (len < 12) {
      return JPEGXL_SIG_NOT_ENOUGH_BYTES;
    } else {
      if (buf[1] == 0 && buf[2] == 0 && buf[3] == 0xC && buf[4] == 'J' &&
          buf[5] == 'X' && buf[6] == 'L' && buf[7] == ' ' && buf[8] == 0xD &&
          buf[9] == 0xA && buf[10] == 0x87 && buf[11] == 0xA) {
        return JPEGXL_SIG_VALID;
      }
    }
  }

  return JPEGXL_SIG_INVALID;
}

struct JpegxlDecoderStruct {
  JpegxlMemoryManager memory_manager;

  // Fields for reading the basic info from the header.
  size_t basic_info_size_hint = InitialBasicInfoSizeHint();
  bool basic_info_available = false;
  size_t xsize = 0;
  size_t ysize = 0;
  int have_container = 0;
  JpegxlSignatureType signature_type = JPEGXL_SIG_TYPE_JPEGXL;
  jxl::ImageMetadata metadata;
};

JpegxlDecoder* JpegxlDecoderCreate(const JpegxlMemoryManager* memory_manager) {
  JpegxlMemoryManager local_memory_manager;
  if (!jxl::MemoryManagerInit(&local_memory_manager, memory_manager))
    return nullptr;

  void* alloc =
      jxl::MemoryManagerAlloc(&local_memory_manager, sizeof(JpegxlDecoder));
  if (!alloc) return nullptr;
  // Placement new constructor on allocated memory
  JpegxlDecoder* dec = new (alloc) JpegxlDecoder();
  dec->memory_manager = local_memory_manager;

  return dec;
}

void JpegxlDecoderDestroy(JpegxlDecoder* dec) {
  if (dec) {
    // Call destructor directly since custom free function is used.
    dec->~JpegxlDecoder();
    jxl::MemoryManagerFree(&dec->memory_manager, dec);
  }
}

size_t JpegxlDecoderSizeHintBasicInfo(const JpegxlDecoder* dec) {
  return dec->basic_info_size_hint;
}

namespace jxl {

template <class T>
bool CanRead(Span<const uint8_t> data, BitReader* reader, T* JXL_RESTRICT t) {
  BitReader reader2(data);
  reader2.SkipBits(reader->TotalBitsConsumed());
  bool result = Bundle::CanRead(&reader2, t);
  JXL_ASSERT(reader2.Close());
  return result;
}

JpegxlDecoderStatus JpegxlDecoderReadBasicInput(JpegxlDecoder* dec,
                                                const uint8_t* in,
                                                size_t size) {
  JpegxlSignature sig = JpegxlSignatureCheck(in, size);
  if (sig == JPEGXL_SIG_INVALID) return JXL_API_ERROR("invalid signature");
  if (sig == JPEGXL_SIG_NOT_ENOUGH_BYTES) return JPEGXL_DEC_NEED_MORE_INPUT;

  // Signature guaranteed correct, now it's possible to tell which one from
  // just the first few bytes.
  if (in[0] == 0) {
    // container
    dec->have_container = 1;
    // signature_type can only be read in the codestream box
  }

  size_t pos = 0;

  // Find the codestream box
  if (dec->have_container) {
    for (;;) {
      if (OutOfBounds(pos, 8, size)) return JPEGXL_DEC_NEED_MORE_INPUT;
      size_t box_start = pos;
      uint64_t box_size = LoadBE32(in + pos);
      char type[5] = {0};
      memcpy(type, in + pos + 4, 4);
      pos += 8;
      if (box_size == 1) {
        if (OutOfBounds(pos, 8, size)) return JPEGXL_DEC_NEED_MORE_INPUT;
        box_size = LoadBE64(in + pos);
        pos += 8;
      }
      if (box_size < 8) {
        return JPEGXL_DEC_ERROR;
      }
      // TODO(lode): support the case where the header is split across multiple
      // codestreaam boxes
      if (strcmp(type, "jxlc") == 0 || strcmp(type, "jxlp") == 0) {
        // Check signature again, for the codestream this time
        JpegxlSignature sig = JpegxlSignatureCheck(in + pos, size - pos);
        if (sig == JPEGXL_SIG_INVALID)
          return JXL_API_ERROR("invalid signature");
        if (sig == JPEGXL_SIG_NOT_ENOUGH_BYTES) {
          return JPEGXL_DEC_NEED_MORE_INPUT;
        }

        // pos is now at the start of the first codestream
        break;
      } else {
        if (OutOfBounds(pos, box_size, size)) {
          // Indicate how many more bytes needed starting from *next_in, which
          // is from the start of the file since the current implementation does
          // not increment *next_in.
          dec->basic_info_size_hint += box_size;
          return JPEGXL_DEC_NEED_MORE_INPUT;
        }
        pos = box_start + box_size;
      }
    }
  }

  // Signature guaranteed correct, now it's possible to tell which one from
  // just the first few bytes.
  if (in[pos] == 0) {
    return JXL_API_ERROR("container nested in codestream not supported");
  } else if (in[pos] == 0xff && in[pos + 1] == 0x0a) {
    dec->signature_type = JPEGXL_SIG_TYPE_JPEGXL;
    pos += 2;
  } else if (in[pos] == 0xff && in[pos + 1] == 0xd8) {
    dec->signature_type = JPEGXL_SIG_TYPE_JPEG;
    pos += 2;
  } else if (in[pos] == 0x0a) {
    dec->signature_type = JPEGXL_SIG_TYPE_TRANSCODED_JPEG;
    pos += 7;
  }

  // TODO(lode): support reading basic info from JPEG and Recompressed JPEG.
  if (dec->signature_type != JPEGXL_SIG_TYPE_JPEGXL) {
    return JXL_API_ERROR("reading non-jxl header not yet supported");
  }

  Span<const uint8_t> span(in + pos, size - pos);
  BitReader reader(span);
  struct ReaderCloser {
    ~ReaderCloser() {
      if (reader->Close()) { /* ignore */
      };
    }
    BitReader* reader;
  } reader_closer = {&reader};

  SizeHeader size_header;
  if (!CanRead(span, &reader, &size_header)) {
    return JPEGXL_DEC_NEED_MORE_INPUT;
  }
  if (!Bundle::Read(&reader, &size_header)) {
    return JXL_API_ERROR("invalid SizeHeader");
  }

  dec->xsize = size_header.xsize();
  dec->ysize = size_header.ysize();

  if (!CanRead(span, &reader, &dec->metadata)) {
    return JPEGXL_DEC_NEED_MORE_INPUT;
  }
  if (!Bundle::Read(&reader, &dec->metadata)) {
    return JXL_API_ERROR("invalid ImageMetadata");
  }
  dec->basic_info_available = true;
  dec->basic_info_size_hint = 0;

  return JPEGXL_DEC_NEED_MORE_INPUT;
}
}  // namespace jxl

JpegxlDecoderStatus JpegxlDecoderProcessInput(JpegxlDecoder* dec,
                                              const uint8_t** next_in,
                                              size_t* avail_in) {
  return jxl::JpegxlDecoderReadBasicInput(dec, *next_in, *avail_in);
}

int JpegxlDecoderGetBasicInfo(const JpegxlDecoder* dec, JpegxlBasicInfo* info) {
  if (!dec->basic_info_available) return 1;  // indicate not available

  if (info) {
    info->have_container = dec->have_container;
    info->signature_type = dec->signature_type;
    info->xsize = dec->xsize;
    info->ysize = dec->ysize;

    const jxl::ImageMetadata meta = dec->metadata;
    info->bits_per_sample = meta.bit_depth.bits_per_sample;
    info->exponent_bits_per_sample = meta.bit_depth.exponent_bits_per_sample;
    info->have_icc = meta.color_encoding.WantICC();

    info->have_preview = meta.m2.have_preview;
    info->have_animation = meta.m2.have_animation;
    // TODO(janwas): intrinsic_size
    info->orientation =
        static_cast<JpegxlOrientation>(meta.m2.orientation_minus_1 + 1);

    info->intensity_target = meta.IntensityTarget();
    info->min_nits = meta.m2.tone_mapping.min_nits;
    info->relative_to_max_display =
        meta.m2.tone_mapping.relative_to_max_display;
    info->linear_below = meta.m2.tone_mapping.linear_below;

    const jxl::ExtraChannelInfo* alpha =
        meta.m2.Find(jxl::ExtraChannel::kAlpha);
    if (alpha != nullptr) {
      info->alpha_bits = alpha->bit_depth.bits_per_sample;
      info->alpha_exponent_bits = alpha->bit_depth.exponent_bits_per_sample;
      info->alpha_premultiplied = alpha->alpha_associated;
    } else {
      info->alpha_bits = 0;
      info->alpha_exponent_bits = 0;
      info->alpha_premultiplied = 0;
    }

    info->num_extra_channels = meta.m2.num_extra_channels;
  }

  return 0;
}

int JpegxlDecoderGetExtraChannelInfo(const JpegxlDecoder* dec, size_t index,
                                     JpegxlExtraChannelInfo* info) {
  if (!dec->basic_info_available) return 1;

  const std::vector<jxl::ExtraChannelInfo>& channels =
      dec->metadata.m2.extra_channel_info;

  if (index >= channels.size()) return 1;  // out of bounds
  const jxl::ExtraChannelInfo& channel = channels[index];

  info->type = static_cast<JpegxlExtraChannelType>(channel.type);
  info->next_frame_base = static_cast<JpegxlFrameBase>(channel.new_base);
  info->blend_mode = static_cast<JpegxlBlendMode>(channel.blend_mode);
  info->bits_per_sample = channel.bit_depth.bits_per_sample;
  info->exponent_bits_per_sample =
      channel.bit_depth.floating_point_sample
          ? channel.bit_depth.exponent_bits_per_sample
          : 0;
  info->dim_shift = channel.dim_shift;
  info->name_length = channel.name.size();
  info->alpha_associated = channel.alpha_associated;
  info->spot_color[0] = channel.spot_color[0];
  info->spot_color[1] = channel.spot_color[1];
  info->spot_color[2] = channel.spot_color[2];
  info->spot_color[3] = channel.spot_color[3];
  info->cfa_channel = channel.cfa_channel;

  return 0;
}

int JpegxlDecoderGetExtraChannelName(const JpegxlDecoder* dec, size_t index,
                                     size_t n, char* name) {
  if (!dec->basic_info_available) return 1;

  const std::vector<jxl::ExtraChannelInfo>& channels =
      dec->metadata.m2.extra_channel_info;

  if (index >= channels.size()) return 1;  // out of bounds
  const jxl::ExtraChannelInfo& channel = channels[index];

  // Also need null-termination character
  if (channel.name.size() + 1 > n) return 1;

  memcpy(name, channel.name.c_str(), channel.name.size() + 1);

  return 0;
}
