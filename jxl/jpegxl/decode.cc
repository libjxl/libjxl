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
#include "jxl/dec_file.h"
#include "jxl/dec_frame.h"
#include "jxl/external_image.h"
#include "jxl/fields.h"
#include "jxl/headers.h"
#include "jxl/icc_codec.h"
#include "jxl/loop_filter.h"
#include "jxl/memory_manager_internal.h"
#include "jxl/toc.h"

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

JpegxlDecoderStatus ConvertStatus(JpegxlDecoderStatus status) { return status; }

JpegxlDecoderStatus ConvertStatus(jxl::Status status) {
  return status ? JPEGXL_DEC_SUCCESS : JPEGXL_DEC_ERROR;
}

// Stores a float in big endian
void StoreBEFloat(float value, uint8_t* p) {
  uint32_t u;
  memcpy(&u, &value, 4);
  StoreBE32(u, p);
}

// Stores a float in little endian
void StoreLEFloat(float value, uint8_t* p) {
  uint32_t u;
  memcpy(&u, &value, 4);
  StoreLE32(u, p);
}

JpegxlDecoderStatus ReadCodestreamSignature(const uint8_t* buf, size_t len,
                                            size_t* pos,
                                            JpegxlSignatureType* sig) {
  if (*pos >= len) return JXL_API_ERROR("signature check out of bounds");
  if (buf[*pos] == 0) {
    // We're reading a codestream signature, if it's a container, that must
    // already have been handled by JpegxlSignatureCheck. If the resulting
    // codestream has a container signature again, it's nested, which is not
    // supported.
    return JXL_API_ERROR("container nested in codestream not supported");
  } else if (buf[*pos] == 0xff) {
    if ((*pos) + 2 > len) return JXL_API_ERROR("signature check out of bounds");
    if (buf[*pos + 1] == 0x0a) {
      *sig = JPEGXL_SIG_TYPE_JPEGXL;
    } else if (buf[*pos + 1] == 0xd8) {
      *sig = JPEGXL_SIG_TYPE_JPEG;
    } else {
      return JXL_API_ERROR("invalid codestream signature");
    }
    (*pos) += 2;
  } else if (buf[*pos] == 0x0a) {
    // The other bytes are guaranteed correct if this is used after
    // JpegxlSignatureCheck
    if ((*pos) + 7 > len) return JXL_API_ERROR("signature check out of bounds");
    *sig = JPEGXL_SIG_TYPE_TRANSCODED_JPEG;
    (*pos) += 7;
  }
  return JPEGXL_DEC_SUCCESS;
}

}  // namespace

uint32_t JpegxlDecoderVersion(void) {
  return JPEGXL_MAJOR_VERSION * 1000000 + JPEGXL_MINOR_VERSION * 1000 +
         JPEGXL_PATCH_VERSION;
}

JpegxlSignature JpegxlSignatureCheck(const uint8_t* buf, size_t len) {
  if (len == 0) return JPEGXL_SIG_NOT_ENOUGH_BYTES;

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

size_t BitsPerChannel(JpegxlDataType data_type) {
  switch (data_type) {
    case JPEGXL_TYPE_BOOLEAN:
      return 1;
    case JPEGXL_TYPE_UINT8:
      return 8;
    case JPEGXL_TYPE_UINT16:
      return 16;
    case JPEGXL_TYPE_UINT32:
      return 32;
    case JPEGXL_TYPE_FLOAT:
      return 32;
      // No default, give compiler error if new type not handled.
  }
  return 0;  // Indicate invalid data type.
}

enum class DecoderStage : uint32_t {
  kInited,    // Decoder created, no JpegxlDecoderProcessInput called yet
  kStarted,   // Running JpegxlDecoderProcessInput calls
  kFinished,  // Everything done, nothing left to process
  kError,     // Error occured, decoder object no longer useable
};

struct JpegxlDecoderStruct {
  JpegxlMemoryManager memory_manager;
  std::unique_ptr<jxl::ThreadPool> thread_pool;

  DecoderStage stage = DecoderStage::kInited;

  // Status of progression, internal.
  bool got_basic_info = false;
  bool got_all_headers = false;
  // For current frame
  bool got_toc = false;
  // This means either we actually got the DC image, or determined we cannot
  // get it.
  bool got_dc_image = false;
  bool got_full_image = false;

  // Bit position of next frame, after the codestream headers, relative to
  // beginning of file.
  // TODO(lode): express in bytes instead of bits since frames should start
  // at a byte boundary
  size_t next_frame_bitpos = 0;

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

  // Owned by the caller, buffers for DC image (8x8 downscaled)
  void* dc_out_buffer = nullptr;
  size_t dc_out_size = 0;
  JpegxlPixelFormat dc_out_format;

  // Owned by the caller, buffer for full resolution image.
  void* image_out_buffer = nullptr;
  size_t image_out_size = 0;
  JpegxlPixelFormat image_out_format;

  jxl::CodecInOut io;

  // headers and TOC for the current frame
  jxl::FrameHeader frame_header;
  jxl::FrameDimensions frame_dim;
  std::vector<uint64_t> group_offsets;
  std::vector<uint32_t> group_sizes;
  size_t frame_start;
  size_t frame_end;

  // User input data is stored here, when the decoder takes in and stores the
  // user input bytes. If the decoder does not do that, this field is unused.
  std::vector<uint8_t> input;
};

// TODO(zond): Make this depend on the data loaded into the decoder.
JpegxlDecoderStatus JpegxlDecoderDefaultPixelFormat(const JpegxlDecoder* dec,
                                                    JpegxlPixelFormat* format) {
  if (!dec->got_basic_info) return JPEGXL_DEC_NEED_MORE_INPUT;
  *format = {4, JPEGXL_LITTLE_ENDIAN, JPEGXL_TYPE_FLOAT};
  return JPEGXL_DEC_SUCCESS;
}

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

#define JXL_API_RETURN_IF_ERROR(expr)                 \
  {                                                   \
    JpegxlDecoderStatus status = ConvertStatus(expr); \
    if (status != JPEGXL_DEC_SUCCESS) return status;  \
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
  JXL_API_RETURN_IF_ERROR(
      ReadCodestreamSignature(in, size, &pos, &dec->signature_type));

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
  dec->io.metadata.m2.nonserialized_only_parse_basic_info = false;
  dec->got_basic_info = true;
  dec->basic_info_size_hint = 0;

  return JPEGXL_DEC_SUCCESS;
}

// Reads all codestream headers (but not frame headers)
JpegxlDecoderStatus JpegxlDecoderReadAllHeaders(JpegxlDecoder* dec,
                                                const uint8_t* in,
                                                size_t size) {
  size_t pos = dec->codestream_pos;

  // True streaming decoding and remembering state is not yet supported in the
  // C++ decoder implementation.
  // Therefore, we read the file from the start including signature again,
  // rather than continue at the part where JpegxlDecoderReadBasicInfo finished.
  JXL_API_RETURN_IF_ERROR(
      ReadCodestreamSignature(in, size, &pos, &dec->signature_type));

  Span<const uint8_t> span(in + pos, size - pos);
  auto reader = GetBitReader(span);
  SizeHeader dummy_size_header;
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dummy_size_header));

  // We already decoded the metadata to dec->io.metadata, no reason to overwrite
  // it, use a dummy metadata instead.
  ImageMetadata dummy_metadata;
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dummy_metadata));

  if (dec->io.metadata.m2.have_preview) {
    JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dec->io.preview));
  }

  if (dec->io.metadata.m2.have_animation) {
    JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dec->io.animation));
  }

  if (dec->io.metadata.color_encoding.WantICC()) {
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
  dec->next_frame_bitpos =
      pos * jxl::kBitsPerByte + reader->TotalBitsConsumed();

  return JPEGXL_DEC_SUCCESS;
}

static void ConvertAlpha(size_t bits_in, const jxl::ImageU& in, size_t bits_out,
                         jxl::ImageU* out) {
  size_t xsize = in.xsize();
  size_t ysize = in.ysize();

  // Error checked elsewhere, but ensure clang-tidy does not report division
  // through zero.
  if (bits_in == 0 || bits_out == 0) return;

  if (bits_in < bits_out) {
    // Multiplier such that bits are duplicated, e.g. when going from 4 bits
    // to 16 bits, converts 0x5 into 0x5555.
    uint16_t mul = ((1ull << bits_out) - 1ull) / ((1ull << bits_in) - 1ull);
    for (size_t y = 0; y < ysize; ++y) {
      const uint16_t* JXL_RESTRICT row_in = in.Row(y);
      uint16_t* JXL_RESTRICT row_out = out->Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = row_in[x] * mul;
      }
    }
  } else {
    // E.g. divide through 257 when converting 16-bit to 8-bit
    uint16_t div = ((1ull << bits_in) - 1ull) / ((1ull << bits_out) - 1ull);
    // Add for round to nearest division.
    uint16_t add = 1 << (bits_out - 1);
    for (size_t y = 0; y < ysize; ++y) {
      const uint16_t* JXL_RESTRICT row_in = in.Row(y);
      uint16_t* JXL_RESTRICT row_out = out->Row(y);
      for (size_t x = 0; x < xsize; ++x) {
        row_out[x] = (row_in[x] + add) / div;
      }
    }
  }
}

static JpegxlDecoderStatus ConvertImage(const jxl::CodecInOut& io,
                                        const JpegxlPixelFormat& format,
                                        jxl::ThreadPool* thread_pool,
                                        void* out_image, size_t out_size) {
  size_t xsize = io.xsize();
  size_t ysize = io.ysize();

  const ColorEncoding& color = io.metadata.color_encoding;
  bool want_alpha = format.num_channels == 2 || format.num_channels == 4;
  bool alpha_premultiplied = false;
  size_t bits_per_sample = BitsPerChannel(format.data_type);
  bool float_out = format.data_type == JPEGXL_TYPE_FLOAT;
  size_t alpha_bits = 0;
  bool big_endian = (format.endianness == JPEGXL_BIG_ENDIAN);
  const jxl::ImageU* alpha = nullptr;
  jxl::ImageU alpha_temp;
  if (want_alpha) {
    if (io.frames[0].HasAlpha() && !float_out) {
      alpha = &io.frames[0].alpha();
      alpha_bits = io.metadata.GetAlphaBits();
      if (alpha_bits == 0 || bits_per_sample == 0) {
        return JXL_API_ERROR("invalid bit depth");
      }
      if (alpha_bits != bits_per_sample) {
        alpha_temp = jxl::ImageU(xsize, ysize);
        // Converting alpha is not (yet) implemented in ExternalImage, it
        // expects the alpha channel values to already be in the output
        // range. Convert here instead.
        ConvertAlpha(alpha_bits, io.frames[0].alpha(), bits_per_sample,
                     &alpha_temp);
        alpha_bits = bits_per_sample;
        alpha = &alpha_temp;
      }
    } else {
      alpha_temp = jxl::ImageU(xsize, ysize);
      for (size_t y = 0; y < ysize; ++y) {
        uint16_t* JXL_RESTRICT row = alpha_temp.Row(y);
        for (size_t x = 0; x < xsize; ++x) {
          row[x] = 255;
        }
      }
      alpha = &alpha_temp;
      alpha_bits = 8;
    }
  }

  jxl::CodecIntervals* intervals = nullptr;
  const jxl::ExternalImage external(thread_pool, io.frames[0].color(),
                                    jxl::Rect(io), color, color, want_alpha,
                                    alpha_premultiplied, alpha, alpha_bits,
                                    bits_per_sample, big_endian, intervals);
  if (out_size < external.Bytes().size()) {
    return JXL_API_ERROR("output buffer too small");
  }

  memcpy(out_image, external.Bytes().data(), external.Bytes().size());

  // For floating point output, manually convert alpha at the end. Unlike the
  // integer case, ExternalImage doesn't already output the correct values
  // in the final buffer so post processing is used in this case.
  if (want_alpha && float_out && io.frames[0].HasAlpha()) {
    // Multiplier for 0.0-1.0 nominal range.
    float mul = 1.0 / ((1ull << io.metadata.GetAlphaBits()) - 1ull);
    size_t i = 0;
    uint8_t* out = reinterpret_cast<uint8_t*>(out_image);
    for (size_t y = 0; y < ysize; ++y) {
      const uint16_t* JXL_RESTRICT row_in = io.frames[0].alpha().Row(y);
      if (format.endianness == JPEGXL_BIG_ENDIAN) {
        for (size_t x = 0; x < xsize; ++x) {
          float alpha = row_in[x] * mul;
          StoreBEFloat(alpha, out + i * 16 + 12);
          i++;
        }
      } else {
        for (size_t x = 0; x < xsize; ++x) {
          float alpha = row_in[x] * mul;
          StoreLEFloat(alpha, out + i * 16 + 12);
          i++;
        }
      }
    }
  }

  // Use 0-1 nominal range for RGB floating point output
  // TODO(lode): support this multiplier in ExternalImage instead to avoid
  // extra pass over the data.
  if (float_out) {
    size_t i = 0;
    uint8_t* out = reinterpret_cast<uint8_t*>(out_image);
    float mul = 1.0 / 255.0;
    for (size_t y = 0; y < ysize; ++y) {
      if (format.endianness == JPEGXL_BIG_ENDIAN) {
        for (size_t x = 0; x < xsize; ++x) {
          for (size_t c = 0; c < 3; ++c) {
            uint32_t u = LoadBE32(out + i);
            float value;
            memcpy(&value, &u, 4);
            value *= mul;
            StoreBEFloat(value, out + i);
            i += 4;
          }
          if (want_alpha) i += 4;  // Skip alpha channel
        }
      } else {
        for (size_t x = 0; x < xsize; ++x) {
          for (size_t c = 0; c < 3; ++c) {
            uint32_t u = LoadLE32(out + i);
            float value;
            memcpy(&value, &u, 4);
            value *= mul;
            StoreLEFloat(value, out + i);
            i += 4;
          }
          if (want_alpha) i += 4;  // Skip alpha channel
        }
      }
    }
  }

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

  // No matter what events are wanted, the basic info is always required.
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

  // Read TOC to find required filesize for DC and full frame
  if (!dec->got_toc &&
      (dec->events_wanted & (JPEGXL_DEC_FULL_IMAGE | JPEGXL_DEC_DC))) {
    size_t pos = (dec->next_frame_bitpos >> 3);
    Span<const uint8_t> span(in + pos, size - pos);
    auto reader = GetBitReader(span);
    reader->SkipBits(dec->next_frame_bitpos - pos * jxl::kBitsPerByte);
    JXL_API_RETURN_IF_ERROR(reader->JumpToByteBoundary());

    dec->frame_dim.Set(dec->xsize, dec->ysize, /*group_size_shift=*/1,
                       /*max_hshift=*/0,
                       /*max_vshift=*/0);

    LoopFilter loop_filter;
    jxl::Status status =
        DecodeFrameHeader(nullptr, reader.get(), &dec->frame_header,
                          &dec->frame_dim, &loop_filter);

    if (status.code() == StatusCode::kNotEnoughBytes) {
      return JPEGXL_DEC_NEED_MORE_INPUT;
    } else if (!status) {
      return JXL_API_ERROR("invalid frame header");
    }

    // Read TOC.
    uint64_t groups_total_size;
    const bool has_ac_global = true;
    const size_t toc_entries =
        NumTocEntries(dec->frame_dim.num_groups, dec->frame_dim.num_dc_groups,
                      dec->frame_header.passes.num_passes, has_ac_global);
    status = ReadGroupOffsets(toc_entries, reader.get(), &dec->group_offsets,
                              &dec->group_sizes, &groups_total_size);

    // TODO(lode): we're actually relying on AllReadsWithinBounds() here
    // instead of on status.code(), change the internal TOC C++ code to
    // correctly set the status.code() instead so we can rely on that one.
    if (!reader->AllReadsWithinBounds() ||
        status.code() == StatusCode::kNotEnoughBytes) {
      return JPEGXL_DEC_NEED_MORE_INPUT;
    } else if (!status) {
      return JXL_API_ERROR("invalid toc entries");
    }

    JXL_API_RETURN_IF_ERROR(reader->JumpToByteBoundary());
    dec->frame_start = pos + (reader->TotalBitsConsumed() >> 3);
    dec->frame_end = dec->frame_start + groups_total_size;

    dec->got_toc = true;
  }

  // Decode to pixels, only if required for the events the user wants.
  if (!dec->got_dc_image && (dec->events_wanted & JPEGXL_DEC_DC)) {
    // Compute amount of bytes for the DC image only from the TOC. That is the
    // bytes of all DC groups.

    // If there is one pass and one group, the TOC only has one entry and
    // doesn't allow to distinguish the DC size, so it's not easy to tell
    // whether we got all DC bytes or not. This will happen for very small
    // images only. Instead, do not return the DC at all, mark DC as done but do
    // not return it: subscribing to the JPEGXL_DEC_DC event is no guarantee of
    // receiving it for all images.
    if (dec->frame_header.passes.num_passes == 1 &&
        dec->frame_dim.num_groups == 1) {
      // Mark DC done and fall through to next decoding stage.
      dec->got_dc_image = true;
    } else {
      size_t dc_size = 0;
      // one DcGlobal entry, N dc group entries.
      size_t num_dc_toc_entries = 1 + dec->frame_dim.num_dc_groups;
      const std::vector<uint32_t>& sizes = dec->group_sizes;
      if (sizes.size() < num_dc_toc_entries)
        return JXL_API_ERROR("too small TOC");
      for (size_t i = 0; i < num_dc_toc_entries; i++) {
        dc_size += sizes[i];
      }
      size_t dc_end = dec->frame_start + dc_size;
      // Not yet enough bytes to decode the DC.
      if (dc_end > size) return JPEGXL_DEC_NEED_MORE_INPUT;

      // Decoding of the DC itself not yet implemented.
      return JXL_API_ERROR("not implemented");
    }
  }

  // Decode to pixels, only if required for the events the user wants.
  if (!dec->got_full_image && (dec->events_wanted & JPEGXL_DEC_FULL_IMAGE)) {
    if (dec->frame_end > size) return JPEGXL_DEC_NEED_MORE_INPUT;

    jxl::DecompressParams dparams;
    jxl::Span<const uint8_t> compressed(in, size);
    jxl::Status status =
        jxl::DecodeFile(dparams, compressed, &dec->io, nullptr, nullptr);
    if (!status) {
      return JXL_API_ERROR("decoding file failed");
    }
    dec->got_full_image = true;
  }

  if (dec->events_wanted & JPEGXL_DEC_FULL_IMAGE) {
    dec->events_wanted &= ~JPEGXL_DEC_FULL_IMAGE;

    // Copy pixels to output buffer if desired. If no output buffer was set, we
    // merely return the JPEGXL_DEC_FULL_IMAGE status without outputting pixels.
    if (dec->image_out_buffer) {
      JpegxlDecoderStatus status =
          ConvertImage(dec->io, dec->image_out_format, dec->thread_pool.get(),
                       dec->image_out_buffer, dec->image_out_size);
      if (status != JPEGXL_DEC_SUCCESS) return status;
    }

    return JPEGXL_DEC_FULL_IMAGE;
  }

  dec->stage = DecoderStage::kFinished;
  // Return success, this means there is nothing more to do.
  return JPEGXL_DEC_SUCCESS;
}

}  // namespace jxl

JpegxlDecoderStatus JpegxlDecoderProcessInput(JpegxlDecoder* dec,
                                              const uint8_t** next_in,
                                              size_t* avail_in) {
  // Two possible ways of consuming the next bytes are implemented here:
  // Either never consume them (next_in and avail_in are not changed, user
  // keeps all bytes), or always consume all (bytes copied into internal
  // vector).
  // Eventually, a middle ground needs to be implemented instead, where the
  // decoder consumes as many bytes as it can and leaves the remaining bytes to
  // the user, e.g. as per frame or per group. Since per frame and per group
  // parsing are not yet implemented, that is not yet possible now. The C++
  // decoder can only process the AC image in one go currently, so cannot yet
  // consume and remember bytes in smaller chunks. Note that it does not matter
  // which of the two strategies here is used, since the user must be able to
  // cope with both behaviours and everything in-between, both are implemented
  // currently for testing. Not consuming and copying bytes may have a slight
  // performance benefit since the user can control how to copy.
#ifdef JPEGXL_API_DEC_CONSUME_BYTES
  // TODO(lode): if dec->input is empty, we can call
  // jxl::JpegxlDecoderProcessInternal directly on next_in without copying. If
  // the processing finishes successfully when the user already gave all bytes
  // (one shot), we may never need to copy the bytes. However since process will
  // typically return multiple status codes (so require multiple calls), the
  // copy would happen anyway, unless the decoder marks that it no longer needs
  // to read them.
  dec->input.insert(dec->input.end(), *next_in, *next_in + *avail_in);
  JpegxlDecoderStatus status = jxl::JpegxlDecoderProcessInternal(
      dec, dec->input.data(), dec->input.size());
  *next_in += *avail_in;
  *avail_in = 0;
  return status;
#else   // JPEGXL_API_DEC_CONSUME_BYTES
  return jxl::JpegxlDecoderProcessInternal(dec, *next_in, *avail_in);
#endif  // JPEGXL_API_DEC_CONSUME_BYTES
}

JpegxlDecoderStatus JpegxlDecoderGetBasicInfo(const JpegxlDecoder* dec,
                                              JpegxlBasicInfo* info) {
  if (!dec->got_basic_info) return JPEGXL_DEC_NEED_MORE_INPUT;

  if (info) {
    const jxl::ImageMetadata& meta = dec->io.metadata;

    info->have_container = dec->have_container;
    info->signature_type = dec->signature_type;
    info->xsize = dec->xsize;
    info->ysize = dec->ysize;
    info->uses_original_profile = !meta.xyb_encoded;

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

namespace {
// Gets the jxl::ColorEncoding for the desired target, and checks errors.
// Returns the object regardless of whether the actual color space is in ICC,
// but ensures that if the color encoding is not the encoding from the
// codestream header metadata, it cannot require ICC profile.
JpegxlDecoderStatus GetColorEncodingForTarget(
    const JpegxlDecoder* dec, JpegxlColorProfileTarget target,
    const jxl::ColorEncoding** encoding) {
  if (!dec->got_all_headers) return JPEGXL_DEC_NEED_MORE_INPUT;

  *encoding = nullptr;
  if (target == JPEGXL_COLOR_PROFILE_TARGET_DATA &&
      dec->io.metadata.xyb_encoded) {
    // In this case, the color profile of the pixels differs from that of the
    // metadata so get the current pixel profile.
    *encoding = &dec->io.Main().c_current();
    if ((*encoding)->WantICC()) {
      // The built-in color spaces supported by the JXL decoder always use a
      // struct, not a ICC profile bytestream, so it would be suspicious if
      // WantICC is true. Of course, it is still possible to get it as an ICC
      // profile by generating it, but there cannot be an encoded ICC profile
      // bytestream because the only ICC profile bytestream that can be present
      // in the JPEG XL codestream header metadata is the original one.
      return JXL_API_ERROR(
          "XYB/absolute color space encoding shoulnd't use ICC");
    }
  } else {
    *encoding = &dec->io.metadata.color_encoding;
  }
  return JPEGXL_DEC_SUCCESS;
}
}  // namespace

JpegxlDecoderStatus JpegxlDecoderGetColorAsEncodedProfile(
    const JpegxlDecoder* dec, JpegxlColorProfileTarget target,
    JpegxlColorEncoding* color_encoding) {
  const jxl::ColorEncoding* jxl_color_encoding = nullptr;
  JpegxlDecoderStatus status =
      GetColorEncodingForTarget(dec, target, &jxl_color_encoding);
  if (status) return status;

  if (jxl_color_encoding->WantICC())
    return JPEGXL_DEC_ERROR;  // Indicate no encoded profile available.

  if (color_encoding) {
    color_encoding->color_space =
        static_cast<JpegxlColorSpace>(jxl_color_encoding->GetColorSpace());

    color_encoding->white_point =
        static_cast<JpegxlWhitePoint>(jxl_color_encoding->white_point);

    jxl::CIExy whitepoint = jxl_color_encoding->GetWhitePoint();
    color_encoding->white_point_xy[0] = whitepoint.x;
    color_encoding->white_point_xy[1] = whitepoint.y;

    color_encoding->primaries =
        static_cast<JpegxlPrimaries>(jxl_color_encoding->primaries);
    jxl::PrimariesCIExy primaries = jxl_color_encoding->GetPrimaries();
    color_encoding->primaries_red_xy[0] = primaries.r.x;
    color_encoding->primaries_red_xy[1] = primaries.r.y;
    color_encoding->primaries_green_xy[0] = primaries.g.x;
    color_encoding->primaries_green_xy[1] = primaries.g.y;
    color_encoding->primaries_blue_xy[0] = primaries.b.x;
    color_encoding->primaries_blue_xy[1] = primaries.b.y;

    if (jxl_color_encoding->tf.IsGamma()) {
      color_encoding->transfer_function = JPEGXL_TRANSFER_FUNCTION_GAMMA;
      color_encoding->gamma = jxl_color_encoding->tf.GetGamma();
    } else {
      color_encoding->transfer_function = static_cast<JpegxlTransferFunction>(
          jxl_color_encoding->tf.GetTransferFunction());
      color_encoding->gamma = 0;
    }

    color_encoding->rendering_intent = static_cast<JpegxlRenderingIntent>(
        jxl_color_encoding->rendering_intent);
  }

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderGetICCProfileSize(
    const JpegxlDecoder* dec, JpegxlColorProfileTarget target, size_t* size) {
  const jxl::ColorEncoding* jxl_color_encoding = nullptr;
  JpegxlDecoderStatus status =
      GetColorEncodingForTarget(dec, target, &jxl_color_encoding);
  if (status != JPEGXL_DEC_SUCCESS) return status;

  if (jxl_color_encoding->WantICC()) {
    jxl::ColorSpace color_space =
        dec->io.metadata.color_encoding.GetColorSpace();
    if (color_space == jxl::ColorSpace::kUnknown ||
        color_space == jxl::ColorSpace::kXYB) {
      // This indicates there's no ICC profile available
      // TODO(lode): for the XYB case, do we want to craft an ICC profile that
      // represents XYB as an RGB profile? It may be possible, but not with
      // only 1D transfer functions.
      return JPEGXL_DEC_ERROR;
    }
  }

  if (size) {
    *size = jxl_color_encoding->ICC().size();
  }

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderGetColorAsICCProfile(
    const JpegxlDecoder* dec, JpegxlColorProfileTarget target,
    uint8_t* icc_profile, size_t size) {
  size_t wanted_size;
  // This also checks the NEED_MORE_INPUT and the unknown/xyb cases
  JpegxlDecoderStatus status =
      JpegxlDecoderGetICCProfileSize(dec, target, &wanted_size);
  if (status != JPEGXL_DEC_SUCCESS) return status;
  if (size < wanted_size) return JXL_API_ERROR("ICC profile output too small");

  const jxl::ColorEncoding* jxl_color_encoding = nullptr;
  status = GetColorEncodingForTarget(dec, target, &jxl_color_encoding);
  if (status != JPEGXL_DEC_SUCCESS) return status;

  memcpy(icc_profile, jxl_color_encoding->ICC().data(),
         jxl_color_encoding->ICC().size());

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

namespace {
// Returns the amount of bits needed for getting memory buffer size, and does
// all error checking required for size checking and format validity.
JpegxlDecoderStatus PrepareSizeCheck(const JpegxlDecoder* dec,
                                     const JpegxlPixelFormat* format,
                                     size_t* bits) {
  if (!dec->got_basic_info) {
    // Don't know image dimensions yet, cannot check for valid size.
    return JPEGXL_DEC_NEED_MORE_INPUT;
  }
  if (format->num_channels > 4) {
    return JXL_API_ERROR("More than 4 channels not supported");
  }
  if (format->num_channels < 3) {
    return JXL_API_ERROR("Grayscale not yet supported");
  }

  *bits = BitsPerChannel(format->data_type);

  if (*bits == 0) {
    return JXL_API_ERROR("Invalid data type");
  }

  return JPEGXL_DEC_SUCCESS;
}
}  // namespace

JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderDCOutBufferSize(
    const JpegxlDecoder* dec, const JpegxlPixelFormat* format, size_t* size) {
  size_t bits;
  JpegxlDecoderStatus status = PrepareSizeCheck(dec, format, &bits);
  if (status != JPEGXL_DEC_SUCCESS) return status;

  size_t xsize = (dec->xsize + jxl::kBlockDim - 1) / jxl::kBlockDim;
  size_t ysize = (dec->ysize + jxl::kBlockDim - 1) / jxl::kBlockDim;

  size_t row_size =
      (xsize * format->num_channels * bits + jxl::kBitsPerByte - 1) /
      jxl::kBitsPerByte;
  *size = row_size * ysize;
  return JPEGXL_DEC_SUCCESS;
}

JPEGXL_EXPORT JpegxlDecoderStatus
JpegxlDecoderSetDCOutBuffer(JpegxlDecoder* dec, const JpegxlPixelFormat* format,
                            void* buffer, size_t size) {
  size_t min_size;
  // This also checks whether the format is valid and supported and basic info
  // is available.
  JpegxlDecoderStatus status =
      JpegxlDecoderImageOutBufferSize(dec, format, &min_size);
  if (status != JPEGXL_DEC_SUCCESS) return status;

  if (size < min_size) return JPEGXL_DEC_ERROR;

  dec->dc_out_buffer = buffer;
  dec->dc_out_size = size;
  dec->dc_out_format = *format;

  return JPEGXL_DEC_SUCCESS;
}

JPEGXL_EXPORT JpegxlDecoderStatus JpegxlDecoderImageOutBufferSize(
    const JpegxlDecoder* dec, const JpegxlPixelFormat* format, size_t* size) {
  size_t bits;
  JpegxlDecoderStatus status = PrepareSizeCheck(dec, format, &bits);
  if (status != JPEGXL_DEC_SUCCESS) return status;

  size_t row_size =
      (dec->xsize * format->num_channels * bits + jxl::kBitsPerByte - 1) /
      jxl::kBitsPerByte;
  *size = row_size * dec->ysize;

  return JPEGXL_DEC_SUCCESS;
}

JpegxlDecoderStatus JpegxlDecoderSetImageOutBuffer(
    JpegxlDecoder* dec, const JpegxlPixelFormat* format, void* buffer,
    size_t size) {
  size_t min_size;
  // This also checks whether the format is valid and supported and basic info
  // is available.
  JpegxlDecoderStatus status =
      JpegxlDecoderImageOutBufferSize(dec, format, &min_size);
  if (status != JPEGXL_DEC_SUCCESS) return status;

  if (size < min_size) return JPEGXL_DEC_ERROR;

  dec->image_out_buffer = buffer;
  dec->image_out_size = size;
  dec->image_out_format = *format;

  return JPEGXL_DEC_SUCCESS;
}
