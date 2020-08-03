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
#include "jxl/icc_codec.h"
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

enum class DecoderStage : uint32_t {
  kInited,
  kStarted,
  kFinished,
  kError,
};

struct JpegxlDecoderStruct {
  JpegxlMemoryManager memory_manager;
  std::unique_ptr<jxl::ThreadPool> thread_pool;

  DecoderStage stage = DecoderStage::kInited;

  // Status of progression, internal.
  bool got_basic_info = false;
  bool got_all_headers = false;

  // Bitfield, for which informative events (JPEGXL_DEC_BASIC_INFO, etc...) the
  // decoder returns a status. By default, do not return for any of the events,
  // only return when the decoder cannot continue becasue it needs mor input or
  // output data.
  int events_wanted = 0;

  // Fields for reading the basic info from the header.
  size_t basic_info_size_hint = InitialBasicInfoSizeHint();
  size_t xsize = 0;
  size_t ysize = 0;
  bool have_container = 0;
  size_t codestream_pos = 0;  // if have_container, where the codestream starts
  JpegxlSignatureType signature_type = JPEGXL_SIG_TYPE_JPEGXL;

  jxl::CodecInOut io;
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

JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderSetParallelRunner(
    JpegxlDecoder* dec, JpegxlParallelRunner parallel_runner,
    void* parallel_runner_opaque) {
  if (dec->thread_pool) return JXL_API_ERROR("parallel runner already set");
  dec->thread_pool.reset(
      new jxl::ThreadPool(parallel_runner, parallel_runner_opaque));
  return JPEGXL_DEC_SUCCESS;
}

size_t JpegxlDecoderSizeHintBasicInfo(const JpegxlDecoder* dec) {
  return dec->basic_info_size_hint;
}

JpegxlDecoderStatus JpegxlDecoderSubscribeEvents(JpegxlDecoder* dec,
                                                 int events_wanted) {
  if (dec->stage != DecoderStage::kInited) {
    return JXL_API_ERROR("Must subscribe to events before starting");
  }
  if (events_wanted & 63) {
    return JXL_API_ERROR("Can only subscribe to informative events.");
  }
  dec->events_wanted = events_wanted;
  return JPEGXL_DEC_SUCCESS;
}

namespace jxl {
template <class T>
bool CanRead(Span<const uint8_t> data, BitReader* reader, T* JXL_RESTRICT t) {
  // Use a copy of the bit reader because CanRead advances bits.
  BitReader reader2(data);
  reader2.SkipBits(reader->TotalBitsConsumed());
  bool result = Bundle::CanRead(&reader2, t);
  JXL_ASSERT(reader2.Close());
  return result;
}

// Returns JPEGXL_DEC_SUCCESS if the full bundle was successfully read, status
// indicating either error or need more input otherwise.
template <class T>
JpegxlDecoderStatus ReadBundle(Span<const uint8_t> data, BitReader* reader,
                               T* JXL_RESTRICT t) {
  if (!CanRead(data, reader, t)) {
    return JPEGXL_DEC_NEED_MORE_INPUT;
  }
  if (!Bundle::Read(reader, t)) {
    return JPEGXL_DEC_ERROR;
  }
  return JPEGXL_DEC_SUCCESS;
}

#define JXL_API_RETURN_IF_ERROR(expr)                \
  {                                                  \
    JpegxlDecoderStatus status = expr;               \
    if (status != JPEGXL_DEC_SUCCESS) return status; \
  }

std::unique_ptr<BitReader, std::function<void(BitReader*)>> GetBitReader(
    Span<const uint8_t> span) {
  BitReader* reader = new BitReader(span);
  return std::unique_ptr<BitReader, std::function<void(BitReader*)>>(
      reader, [](BitReader* reader) {
        JXL_CHECK(reader->Close());
        delete reader;
      });
}

JpegxlDecoderStatus JpegxlDecoderReadBasicInfo(JpegxlDecoder* dec,
                                               const uint8_t* in, size_t size) {
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
        dec->codestream_pos = pos;
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
  auto reader = GetBitReader(span);
  SizeHeader size_header;
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &size_header));

  dec->xsize = size_header.xsize();
  dec->ysize = size_header.ysize();

  dec->io.metadata.m2.nonserialized_only_parse_basic_info = true;
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dec->io.metadata));
  dec->got_basic_info = true;
  dec->basic_info_size_hint = 0;

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderReadAllHeaders(JpegxlDecoder* dec,
                                                const uint8_t* in,
                                                size_t size) {
  size_t pos = dec->codestream_pos;

  Span<const uint8_t> span(in + pos, size - pos);
  auto reader = GetBitReader(span);

  // True streaming decoding and remembering state is not yet supported in the
  // C++ decoder implementation.
  // Therefore, we read the file from the start including basic info header
  // again, rather than continue at the part where JpegxlDecoderReadBasicInfo
  // finished.
  SizeHeader size_header;
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &size_header));

  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dec->io.metadata));

  if (dec->io.metadata.m2.have_preview) {
    JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dec->io.preview));
  }

  if (dec->io.metadata.m2.have_animation) {
    JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dec->io.animation));
  }

  if (dec->io.metadata.color_encoding.WantICC()) {
    // Parse the ICC size first.
    if (!reader->JumpToByteBoundary()) return JPEGXL_DEC_ERROR;
    PaddedBytes icc;
    jxl::Status status = ReadICC(reader.get(), &icc);
    if (!status) {
      if (status.code() == StatusCode::kNotEnoughBytes) {
        return JPEGXL_DEC_NEED_MORE_INPUT;
      }
      // Other non-successful status is an error
      return JPEGXL_DEC_ERROR;
    }
    if (!dec->io.metadata.color_encoding.SetICC(std::move(icc))) {
      return JPEGXL_DEC_ERROR;
    }
  }

  dec->got_all_headers = true;

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderProcessInternal(JpegxlDecoder* dec,
                                                 const uint8_t* in,
                                                 size_t size) {
  // If no parallel runner is set, use the default
  // TODO(lode): move this initialization to an appropriate location once the
  // runner is used to decode pixels.
  if (!dec->thread_pool) {
    dec->thread_pool.reset(new jxl::ThreadPool(nullptr, nullptr));
  }
  if (dec->stage == DecoderStage::kInited) {
    dec->stage = DecoderStage::kStarted;
  }
  if (dec->stage == DecoderStage::kError) {
    return JXL_API_ERROR("Cannot use decoder after it encountered error");
  }
  if (dec->stage == DecoderStage::kFinished) {
    return JXL_API_ERROR("Cannot reuse decoder after it finished");
  }

  if (!dec->got_basic_info) {
    JpegxlDecoderStatus status = JpegxlDecoderReadBasicInfo(dec, in, size);
    if (status != JPEGXL_DEC_SUCCESS) return status;
  }

  if (dec->events_wanted & JPEGXL_DEC_BASIC_INFO) {
    dec->events_wanted &= ~JPEGXL_DEC_BASIC_INFO;
    return JPEGXL_DEC_BASIC_INFO;
  }

  if (!dec->got_all_headers) {
    JpegxlDecoderStatus status = JpegxlDecoderReadAllHeaders(dec, in, size);
    if (status != JPEGXL_DEC_SUCCESS) return status;
  }

  if (dec->events_wanted & JPEGXL_DEC_EXTENSIONS) {
    dec->events_wanted &= ~JPEGXL_DEC_EXTENSIONS;
    if (dec->io.metadata.m2.extensions != 0) {
      return JPEGXL_DEC_EXTENSIONS;
    }
  }

  if (dec->events_wanted & JPEGXL_DEC_PREVIEW_HEADER) {
    dec->events_wanted &= ~JPEGXL_DEC_PREVIEW_HEADER;
    if (dec->io.metadata.m2.have_preview) {
      return JPEGXL_DEC_PREVIEW_HEADER;
    }
  }

  if (dec->events_wanted & JPEGXL_DEC_ANIMATION_HEADER) {
    dec->events_wanted &= ~JPEGXL_DEC_ANIMATION_HEADER;
    if (dec->io.metadata.m2.have_animation) {
      return JPEGXL_DEC_ANIMATION_HEADER;
    }
  }

  if (dec->events_wanted & JPEGXL_DEC_COLOR_ENCODING) {
    dec->events_wanted &= ~JPEGXL_DEC_COLOR_ENCODING;
    return JPEGXL_DEC_COLOR_ENCODING;
  }

  // Outputting pixels not yet implemented, indicate error.
  dec->stage = DecoderStage::kError;
  return JPEGXL_DEC_ERROR;
}

}  // namespace jxl

JpegxlDecoderStatus JpegxlDecoderProcessInput(JpegxlDecoder* dec,
                                              const uint8_t** next_in,
                                              size_t* avail_in) {
  return jxl::JpegxlDecoderProcessInternal(dec, *next_in, *avail_in);
}

JpegxlDecoderStatus JpegxlDecoderGetBasicInfo(const JpegxlDecoder* dec,
                                              JpegxlBasicInfo* info) {
  if (!dec->got_basic_info) return JPEGXL_DEC_NEED_MORE_INPUT;

  if (info) {
    info->have_container = dec->have_container;
    info->signature_type = dec->signature_type;
    info->xsize = dec->xsize;
    info->ysize = dec->ysize;

    const jxl::ImageMetadata& meta = dec->io.metadata;
    info->bits_per_sample = meta.bit_depth.bits_per_sample;
    info->exponent_bits_per_sample = meta.bit_depth.exponent_bits_per_sample;

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

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderGetExtraChannelInfo(
    const JpegxlDecoder* dec, size_t index, JpegxlExtraChannelInfo* info) {
  if (!dec->got_basic_info) return JPEGXL_DEC_NEED_MORE_INPUT;

  const std::vector<jxl::ExtraChannelInfo>& channels =
      dec->io.metadata.m2.extra_channel_info;

  if (index >= channels.size()) return JPEGXL_DEC_ERROR;  // out of bounds
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

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderGetExtraChannelName(const JpegxlDecoder* dec,
                                                     size_t index, char* name,
                                                     size_t size) {
  if (!dec->got_basic_info) return JPEGXL_DEC_NEED_MORE_INPUT;

  const std::vector<jxl::ExtraChannelInfo>& channels =
      dec->io.metadata.m2.extra_channel_info;

  if (index >= channels.size()) return JPEGXL_DEC_ERROR;  // out of bounds
  const jxl::ExtraChannelInfo& channel = channels[index];

  // Also need null-termination character
  if (channel.name.size() + 1 > size) return JPEGXL_DEC_ERROR;

  memcpy(name, channel.name.c_str(), channel.name.size() + 1);

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderGetPreviewHeader(
    const JpegxlDecoder* dec, JpegxlPreviewHeader* preview_header) {
  if (!dec->got_all_headers) return JPEGXL_DEC_NEED_MORE_INPUT;

  if (!dec->io.metadata.m2.have_preview) return JPEGXL_DEC_ERROR;

  preview_header->xsize = dec->io.preview.xsize();
  preview_header->ysize = dec->io.preview.ysize();

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderGetAnimationHeader(
    const JpegxlDecoder* dec, JpegxlAnimationHeader* animation_header) {
  if (!dec->got_all_headers) return JPEGXL_DEC_NEED_MORE_INPUT;

  if (!dec->io.metadata.m2.have_animation) return JPEGXL_DEC_ERROR;

  animation_header->composite_still = dec->io.animation.composite_still;
  animation_header->tps_numerator = dec->io.animation.tps_numerator;
  animation_header->tps_denominator = dec->io.animation.tps_denominator;
  animation_header->num_loops = dec->io.animation.num_loops;
  animation_header->have_timecodes = dec->io.animation.have_timecodes;

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderGetColorProfileSource(
    const JpegxlDecoder* dec, JpegxlColorProfileSource* color_info) {
  if (!dec->got_all_headers) return JPEGXL_DEC_NEED_MORE_INPUT;

  if (dec->io.metadata.color_encoding.WantICC()) {
    color_info->icc_profile_accurate = 1;
    color_info->color_encoding_valid = 0;
  } else {
    // TODO(lode): there are cases where the ICC profile is accurate. It is not
    // accurate for PQ or HLG transfer functions (possibly others such as sRGB
    // too), but can be accurate for linear, gamma, ... In that case, this
    // could be set to 1 so the user has the option of using that rather than
    // deal with all the JpegxlColorEncoding fields.
    color_info->icc_profile_accurate = 0;
    color_info->color_encoding_valid = 1;
  }

  color_info->icc_profile_size = dec->io.metadata.color_encoding.ICC().size();

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderGetColorEncoding(
    const JpegxlDecoder* dec, JpegxlColorEncoding* color_encoding) {
  if (!dec->got_all_headers) return JPEGXL_DEC_NEED_MORE_INPUT;

  if (dec->io.metadata.color_encoding.WantICC()) {
    return JXL_API_ERROR("Trying to get color encoding but is invalid");
  }

  color_encoding->color_space = static_cast<JpegxlColorSpace>(
      dec->io.metadata.color_encoding.GetColorSpace());

  color_encoding->white_point = static_cast<JpegxlWhitePoint>(
      dec->io.metadata.color_encoding.white_point);

  jxl::CIExy whitepoint = dec->io.metadata.color_encoding.GetWhitePoint();
  color_encoding->white_point_xy[0] = whitepoint.x;
  color_encoding->white_point_xy[1] = whitepoint.y;

  color_encoding->primaries =
      static_cast<JpegxlPrimaries>(dec->io.metadata.color_encoding.primaries);
  jxl::PrimariesCIExy primaries =
      dec->io.metadata.color_encoding.GetPrimaries();
  color_encoding->primaries_red_xy[0] = primaries.r.x;
  color_encoding->primaries_red_xy[1] = primaries.r.y;
  color_encoding->primaries_green_xy[0] = primaries.g.x;
  color_encoding->primaries_green_xy[1] = primaries.g.y;
  color_encoding->primaries_blue_xy[0] = primaries.b.x;
  color_encoding->primaries_blue_xy[1] = primaries.b.y;

  if (dec->io.metadata.color_encoding.tf.IsGamma()) {
    color_encoding->transfer_function = JPEGXL_TRANSFER_FUNCTION_GAMMA;
    color_encoding->gamma = dec->io.metadata.color_encoding.tf.GetGamma();
  } else {
    color_encoding->transfer_function = static_cast<JpegxlTransferFunction>(
        dec->io.metadata.color_encoding.tf.GetTransferFunction());
    color_encoding->gamma = 0;
  }

  color_encoding->rendering_intent = static_cast<JpegxlRenderingIntent>(
      dec->io.metadata.color_encoding.rendering_intent);

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderGetInverseOpsinMatrix(
    const JpegxlDecoder* dec, JpegxlInverseOpsinMatrix* matrix) {
  memcpy(matrix->opsin_inv_matrix,
         dec->io.metadata.m2.opsin_inverse_matrix.inverse_matrix,
         sizeof(matrix->opsin_inv_matrix));
  memcpy(matrix->opsin_biases,
         dec->io.metadata.m2.opsin_inverse_matrix.opsin_biases,
         sizeof(matrix->opsin_biases));
  memcpy(matrix->quant_biases,
         dec->io.metadata.m2.opsin_inverse_matrix.quant_biases,
         sizeof(matrix->quant_biases));

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderGetICCProfile(const JpegxlDecoder* dec,
                                               uint8_t* icc_profile,
                                               size_t size) {
  if (!dec->got_all_headers) return JPEGXL_DEC_NEED_MORE_INPUT;
  if (size < dec->io.metadata.color_encoding.ICC().size())
    return JPEGXL_DEC_ERROR;

  memcpy(icc_profile, dec->io.metadata.color_encoding.ICC().data(),
         dec->io.metadata.color_encoding.ICC().size());

  return JPEGXL_DEC_SUCCESS;
}
