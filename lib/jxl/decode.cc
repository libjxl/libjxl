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

#include "jxl/decode.h"

#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/dec_frame.h"
#include "lib/jxl/dec_modular.h"
#include "lib/jxl/external_image.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/icc_codec.h"
#include "lib/jxl/loop_filter.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/toc.h"

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
  // around 2 bytes signature + 8 bytes SizeHeader + 31 bytes ColorEncoding + 4
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
   ::jxl::Abort(), JXL_DEC_ERROR)
#else  // JXL_CRASH_ON_ERROR
#define JXL_API_ERROR(format, ...)                                             \
  (((JXL_DEBUG_ON_ERROR) &&                                                    \
    ::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, ##__VA_ARGS__)), \
   JXL_DEC_ERROR)
#endif  // JXL_CRASH_ON_ERROR

JxlDecoderStatus ConvertStatus(JxlDecoderStatus status) { return status; }

JxlDecoderStatus ConvertStatus(jxl::Status status) {
  return status ? JXL_DEC_SUCCESS : JXL_DEC_ERROR;
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

JxlSignature ReadSignature(const uint8_t* buf, size_t len, size_t* pos) {
  if (*pos >= len) return JXL_SIG_NOT_ENOUGH_BYTES;

  buf += *pos;
  len -= *pos;

  // JPEG XL codestream
  if (len >= 1 && buf[0] == 0xff) {
    if (len < 2) {
      return JXL_SIG_NOT_ENOUGH_BYTES;
    } else if (buf[1] == jxl::kCodestreamMarker) {
      *pos += 2;
      return JXL_SIG_CODESTREAM;
    } else {
      return JXL_SIG_INVALID;
    }
  }

  // JPEG XL container
  if (len >= 1 && buf[0] == 0) {
    if (len < 12) {
      return JXL_SIG_NOT_ENOUGH_BYTES;
    } else if (buf[1] == 0 && buf[2] == 0 && buf[3] == 0xC && buf[4] == 'J' &&
               buf[5] == 'X' && buf[6] == 'L' && buf[7] == ' ' &&
               buf[8] == 0xD && buf[9] == 0xA && buf[10] == 0x87 &&
               buf[11] == 0xA) {
      *pos += 12;
      return JXL_SIG_CONTAINER;
    } else {
      return JXL_SIG_INVALID;
    }
  }

  return JXL_SIG_INVALID;
}

}  // namespace

uint32_t JxlDecoderVersion(void) {
  return JPEGXL_MAJOR_VERSION * 1000000 + JPEGXL_MINOR_VERSION * 1000 +
         JPEGXL_PATCH_VERSION;
}

JxlSignature JxlSignatureCheck(const uint8_t* buf, size_t len) {
  size_t pos = 0;
  return ReadSignature(buf, len, &pos);
}

size_t BitsPerChannel(JxlDataType data_type) {
  switch (data_type) {
    case JXL_TYPE_BOOLEAN:
      return 1;
    case JXL_TYPE_UINT8:
      return 8;
    case JXL_TYPE_UINT16:
      return 16;
    case JXL_TYPE_UINT32:
      return 32;
    case JXL_TYPE_FLOAT:
      return 32;
      // No default, give compiler error if new type not handled.
  }
  return 0;  // Indicate invalid data type.
}

enum class DecoderStage : uint32_t {
  kInited,    // Decoder created, no JxlDecoderProcessInput called yet
  kStarted,   // Running JxlDecoderProcessInput calls
  kFinished,  // Everything done, nothing left to process
  kError,     // Error occured, decoder object no longer useable
};

struct JxlDecoderStruct {
  JxlDecoderStruct() {}

  JxlMemoryManager memory_manager;
  std::unique_ptr<jxl::ThreadPool> thread_pool;

  DecoderStage stage;

  // Status of progression, internal.
  bool got_basic_info;
  bool got_all_headers;
  // For current frame
  bool got_toc;
  // This means either we actually got the DC image, or determined we cannot
  // get it.
  bool got_dc_image;
  bool got_full_image;

  // Settings
  bool keep_orientation = false;

  // Start position of the next frame in bytes, including its headers. This is
  // the next frame to parse headers from. The current implementation only uses
  // this to find the size of frames until the last one is reached, and then
  // decodes them all at once with the C++ implementation, for composite stills
  // only.
  size_t next_frame_pos;
  // Start position of the first un-processed frame. This variable is currently
  // unused, but will be needed for animation support: a dispalyed animation
  // frame can exist out of multiple frames, the first one starting at
  // first_frame_pos, its last one eventually at next_frame_pos.
  size_t first_frame_pos;

  // Bitfield, for which informative events (JXL_DEC_BASIC_INFO, etc...) the
  // decoder returns a status. By default, do not return for any of the events,
  // only return when the decoder cannot continue becasue it needs mor input or
  // output data.
  int events_wanted;

  // Fields for reading the basic info from the header.
  size_t basic_info_size_hint;
  size_t codestream_pos;  // if have_container, where the codestream starts
  bool have_container;

  // Whether the DC out buffer was set. It is possible for dc_out_buffer to
  // be nullptr and dc_out_buffer_set be true, indicating it was deliberately
  // set to nullptr.
  bool dc_out_buffer_set;
  // Idem for the image buffer.
  bool image_out_buffer_set;

  // Owned by the caller, buffers for DC image and full resolution images
  void* dc_out_buffer;
  void* image_out_buffer;

  size_t dc_out_size;
  size_t image_out_size;

  JxlPixelFormat dc_out_format;
  JxlPixelFormat image_out_format;

  std::unique_ptr<jxl::CodecInOut> io;

  // headers and TOC for the current frame
  std::unique_ptr<jxl::FrameHeader> frame_header;
  jxl::FrameDimensions frame_dim;
  std::vector<uint64_t> group_offsets;
  std::vector<uint32_t> group_sizes;
  // Start and end of all passes data of the frame, excluding TOC and headers.
  size_t frame_start;
  size_t frame_end;
  size_t frames_seen;

  // User input data is stored here, when the decoder takes in and stores the
  // user input bytes. If the decoder does not do that, this field is unused.
  std::vector<uint8_t> input;
};

// TODO(zond): Make this depend on the data loaded into the decoder.
JxlDecoderStatus JxlDecoderDefaultPixelFormat(const JxlDecoder* dec,
                                              JxlPixelFormat* format) {
  if (!dec->got_basic_info) return JXL_DEC_NEED_MORE_INPUT;
  *format = {4, JXL_TYPE_FLOAT, JXL_LITTLE_ENDIAN, 0};
  return JXL_DEC_SUCCESS;
}

void JxlDecoderReset(JxlDecoder* dec) {
  dec->thread_pool.reset();
  dec->stage = DecoderStage::kInited;
  dec->got_basic_info = false;
  dec->got_all_headers = false;
  dec->got_toc = false;
  dec->got_dc_image = false;
  dec->got_full_image = false;
  dec->next_frame_pos = 0;
  dec->first_frame_pos = 0;
  dec->events_wanted = 0;
  dec->basic_info_size_hint = InitialBasicInfoSizeHint();
  dec->codestream_pos = 0;  // if have_container, where the codestream starts
  dec->have_container = 0;
  dec->dc_out_buffer_set = false;
  dec->image_out_buffer_set = false;
  dec->dc_out_buffer = nullptr;
  dec->image_out_buffer = nullptr;
  dec->dc_out_size = 0;
  dec->image_out_size = 0;

  dec->io.reset(new jxl::CodecInOut());

  dec->frame_header.reset(new jxl::FrameHeader(&dec->io->metadata));
  dec->frame_dim = jxl::FrameDimensions();
  dec->group_offsets.clear();
  dec->group_sizes.clear();
  dec->frames_seen = 0;
  dec->input.clear();
}

JxlDecoder* JxlDecoderCreate(const JxlMemoryManager* memory_manager) {
  JxlMemoryManager local_memory_manager;
  if (!jxl::MemoryManagerInit(&local_memory_manager, memory_manager))
    return nullptr;

  void* alloc =
      jxl::MemoryManagerAlloc(&local_memory_manager, sizeof(JxlDecoder));
  if (!alloc) return nullptr;
  // Placement new constructor on allocated memory
  JxlDecoder* dec = new (alloc) JxlDecoder();
  dec->memory_manager = local_memory_manager;

  JxlDecoderReset(dec);

  return dec;
}

void JxlDecoderDestroy(JxlDecoder* dec) {
  if (dec) {
    // Call destructor directly since custom free function is used.
    dec->~JxlDecoder();
    jxl::MemoryManagerFree(&dec->memory_manager, dec);
  }
}

JXL_EXPORT JxlDecoderStatus
JxlDecoderSetParallelRunner(JxlDecoder* dec, JxlParallelRunner parallel_runner,
                            void* parallel_runner_opaque) {
  if (dec->thread_pool) return JXL_API_ERROR("parallel runner already set");
  dec->thread_pool.reset(
      new jxl::ThreadPool(parallel_runner, parallel_runner_opaque));
  return JXL_DEC_SUCCESS;
}

size_t JxlDecoderSizeHintBasicInfo(const JxlDecoder* dec) {
  return dec->basic_info_size_hint;
}

JxlDecoderStatus JxlDecoderSubscribeEvents(JxlDecoder* dec, int events_wanted) {
  if (dec->stage != DecoderStage::kInited) {
    return JXL_DEC_ERROR;  // Cannot subscribe to events after having started.
  }
  if (events_wanted & 63) {
    return JXL_DEC_ERROR;  // Can only subscribe to informative events.
  }
  dec->events_wanted = events_wanted;
  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetKeepOrientation(JxlDecoder* dec,
                                              JXL_BOOL keep_orientation) {
  if (dec->stage != DecoderStage::kInited) {
    return JXL_API_ERROR("Must set keep_orientation option before starting");
  }
  dec->keep_orientation = !!keep_orientation;
  return JXL_DEC_SUCCESS;
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

// Returns JXL_DEC_SUCCESS if the full bundle was successfully read, status
// indicating either error or need more input otherwise.
template <class T>
JxlDecoderStatus ReadBundle(Span<const uint8_t> data, BitReader* reader,
                            T* JXL_RESTRICT t) {
  if (!CanRead(data, reader, t)) {
    return JXL_DEC_NEED_MORE_INPUT;
  }
  if (!Bundle::Read(reader, t)) {
    return JXL_DEC_ERROR;
  }
  return JXL_DEC_SUCCESS;
}

#define JXL_API_RETURN_IF_ERROR(expr)              \
  {                                                \
    JxlDecoderStatus status = ConvertStatus(expr); \
    if (status != JXL_DEC_SUCCESS) return status;  \
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

JxlDecoderStatus JxlDecoderReadBasicInfo(JxlDecoder* dec, const uint8_t* in,
                                         size_t size) {
  JxlSignature sig = JxlSignatureCheck(in, size);
  if (sig == JXL_SIG_INVALID) return JXL_API_ERROR("invalid signature");
  if (sig == JXL_SIG_NOT_ENOUGH_BYTES) return JXL_DEC_NEED_MORE_INPUT;

  if (sig == JXL_SIG_CONTAINER) {
    dec->have_container = 1;
  }

  size_t pos = 0;

  // Find the codestream box
  if (dec->have_container) {
    for (;;) {
      if (OutOfBounds(pos, 8, size)) return JXL_DEC_NEED_MORE_INPUT;
      size_t box_start = pos;
      uint64_t box_size = LoadBE32(in + pos);
      char type[5] = {0};
      memcpy(type, in + pos + 4, 4);
      pos += 8;
      if (box_size == 1) {
        if (OutOfBounds(pos, 8, size)) return JXL_DEC_NEED_MORE_INPUT;
        box_size = LoadBE64(in + pos);
        pos += 8;
      }
      if (box_size < 8) {
        return JXL_DEC_ERROR;
      }
      // TODO(lode): support the case where the header is split across multiple
      // codestreaam boxes
      if (strcmp(type, "jxlc") == 0 || strcmp(type, "jxlp") == 0) {
        // Check signature again, for the codestream this time
        JxlSignature sig = JxlSignatureCheck(in + pos, size - pos);
        if (sig == JXL_SIG_INVALID) return JXL_API_ERROR("invalid signature");
        if (sig == JXL_SIG_NOT_ENOUGH_BYTES) {
          return JXL_DEC_NEED_MORE_INPUT;
        }

        // pos is now at the start of the first codestream
        dec->codestream_pos = pos;
        break;
      } else {
        if (OutOfBounds(pos, box_size, size)) {
          // Indicate how many more bytes needed starting from *next_in, which
          // is from the start of the file since the current implementation does
          // not increment *next_in.
          dec->basic_info_size_hint += box_size;
          return JXL_DEC_NEED_MORE_INPUT;
        }
        pos = box_start + box_size;
      }
    }
  }

  // Check and skip the codestream signature
  JxlSignature signature = ReadSignature(in, size, &pos);
  if (signature == JXL_SIG_CONTAINER) {
    return JXL_API_ERROR("invalid: nested container");
  }
  if (signature != JXL_SIG_CODESTREAM) {
    return JXL_API_ERROR("invalid signature");
  }

  Span<const uint8_t> span(in + pos, size - pos);
  auto reader = GetBitReader(span);
  JXL_API_RETURN_IF_ERROR(
      ReadBundle(span, reader.get(), &dec->io->metadata.size));

  dec->io->metadata.m.nonserialized_only_parse_basic_info = true;
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dec->io->metadata.m));
  dec->io->metadata.m.nonserialized_only_parse_basic_info = false;
  dec->got_basic_info = true;
  dec->basic_info_size_hint = 0;

  return JXL_DEC_SUCCESS;
}

// Reads all codestream headers (but not frame headers)
JxlDecoderStatus JxlDecoderReadAllHeaders(JxlDecoder* dec, const uint8_t* in,
                                          size_t size) {
  size_t pos = dec->codestream_pos;

  // Check and skip the codestream signature
  JxlSignature signature = ReadSignature(in, size, &pos);
  if (signature == JXL_SIG_CONTAINER) {
    return JXL_API_ERROR("invalid: nested container");
  }
  if (signature != JXL_SIG_CODESTREAM) {
    return JXL_API_ERROR("invalid signature");
  }

  Span<const uint8_t> span(in + pos, size - pos);
  auto reader = GetBitReader(span);
  SizeHeader dummy_size_header;
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dummy_size_header));

  // We already decoded the metadata to dec->io->metadata.m, no reason to
  // overwrite it, use a dummy metadata instead.
  ImageMetadata dummy_metadata;
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dummy_metadata));

  JXL_API_RETURN_IF_ERROR(
      ReadBundle(span, reader.get(), &dec->io->metadata.transform_data));

  if (dec->io->metadata.m.color_encoding.WantICC()) {
    PaddedBytes icc;
    jxl::Status status = ReadICC(reader.get(), &icc);
    // Always check AllReadsWithinBounds, not all the C++ decoder implementation
    // handles reader out of bounds correctly  yet (e.g. context map). Not
    // checking AllReadsWithinBounds can cause reader->Close() to trigger an
    // assert, but we don't want library to quit program for invalid codestream.
    if (!reader->AllReadsWithinBounds()) {
      return JXL_DEC_NEED_MORE_INPUT;
    }
    if (!status) {
      if (status.code() == StatusCode::kNotEnoughBytes) {
        return JXL_DEC_NEED_MORE_INPUT;
      }
      // Other non-successful status is an error
      return JXL_DEC_ERROR;
    }
    if (!dec->io->metadata.m.color_encoding.SetICC(std::move(icc))) {
      return JXL_DEC_ERROR;
    }
  }

  dec->got_all_headers = true;
  JXL_API_RETURN_IF_ERROR(reader->JumpToByteBoundary());
  dec->first_frame_pos = pos + reader->TotalBitsConsumed() / jxl::kBitsPerByte;
  dec->next_frame_pos = dec->first_frame_pos;

  return JXL_DEC_SUCCESS;
}

// Returns whether the JxlEndianness value indicates little endian. If not,
// then big endian is assumed.
static bool IsLittleEndian(const JxlEndianness& endianness) {
  switch (endianness) {
    case JXL_LITTLE_ENDIAN:
      return true;
    case JXL_BIG_ENDIAN:
      return false;
    case JXL_NATIVE_ENDIAN: {
      // JXL_BYTE_ORDER_LITTLE from byte_order.h cannot be used because it only
      // distinguishes between little endian and unknown.
      uint32_t u = 1;
      char c[4];
      memcpy(c, &u, 4);
      return c[0] == 1;
    }
  }

  JXL_ASSERT(false);
  return false;
}

static JxlDecoderStatus ConvertImage(const JxlDecoder* dec,
                                     const jxl::CodecInOut& io,
                                     const JxlPixelFormat& format,
                                     void* out_image, size_t out_size) {
  // TODO(lode): handle mismatch of RGB/grayscale color profiles and pixel data
  // color/grayscale format

  size_t stride = io.xsize() * (BitsPerChannel(format.data_type) *
                                format.num_channels / jxl::kBitsPerByte);

  bool apply_srgb_tf = false;
  if (io.metadata.m.xyb_encoded) {
    if (!io.Main().c_current().IsLinearSRGB()) {
      return JXL_API_ERROR(
          "Error, the implementation expects that ImageBundle is in linear "
          "sRGB when the image was xyb_encoded");
    }
    if (format.data_type != JXL_TYPE_FLOAT) {
      // Convert to nonlinear sRGB for integer pixels.
      apply_srgb_tf = true;
    }
  }
  jxl::Orientation undo_orientation = dec->keep_orientation
                                          ? io.metadata.m.GetOrientation()
                                          : jxl::Orientation::kIdentity;
  jxl::Status status = jxl::ConvertImage(
      io.Main(), BitsPerChannel(format.data_type),
      format.data_type == JXL_TYPE_FLOAT, false, apply_srgb_tf,
      format.num_channels, IsLittleEndian(format.endianness), stride,
      dec->thread_pool.get(), out_image, out_size, undo_orientation);

  return status ? JXL_DEC_SUCCESS : JXL_DEC_ERROR;
}

jxl::Status DecodeDC(JxlDecoder* dec, const uint8_t* in, size_t size) {
  FrameHeader& frame_header = *dec->frame_header;
  FrameDimensions& frame_dim = dec->frame_dim;
  PassesDecoderState dec_state;
  ModularFrameDecoder modular_frame_decoder(frame_dim);
  const std::vector<uint32_t>& group_sizes = dec->group_sizes;
  const std::vector<uint64_t>& group_offsets = dec->group_offsets;
  ThreadPool* pool = nullptr;
  std::vector<AuxOut>* aux_outs = nullptr;
  AuxOut* JXL_RESTRICT aux_out = nullptr;

  Span<const uint8_t> span(in + dec->frame_start, size - dec->frame_start);
  auto reader = GetBitReader(span);

  JXL_RETURN_IF_ERROR(
      InitializePassesSharedState(frame_header, &dec_state.shared_storage));

  {
    PassesSharedState& shared = dec_state.shared_storage;

    {
      if (shared.frame_header.flags & FrameHeader::kPatches) {
        JXL_RETURN_IF_ERROR(shared.image_features.patches.Decode(
            reader.get(), shared.frame_dim.xsize_padded,
            shared.frame_dim.ysize_padded));
      }
      if (shared.frame_header.flags & FrameHeader::kSplines) {
        JXL_RETURN_IF_ERROR(shared.image_features.splines.Decode(reader.get()));
      }
      if (shared.frame_header.flags & FrameHeader::kNoise) {
        JXL_RETURN_IF_ERROR(
            DecodeNoise(reader.get(), &shared.image_features.noise_params));
      }
    }

    // TODO(lode): Share more code between here and dec_frame.cc
    // TODO(lode): support non-lossy, grayscale and/or non-xyb as well.
    JXL_RETURN_IF_ERROR(shared.matrices.DecodeDC(reader.get()));
    if (frame_header.encoding == FrameEncoding::kVarDCT) {
      JXL_RETURN_IF_ERROR(jxl::DecodeGlobalDCInfo(
          0 /*downsampling*/, reader.get(), &dec->io->Main(), &dec_state,
          dec->thread_pool.get()));
    } else if (frame_header.encoding == FrameEncoding::kModular) {
      dec_state.Init(dec->thread_pool.get());
    }
    JXL_RETURN_IF_ERROR(modular_frame_decoder.DecodeGlobalInfo(
        reader.get(), frame_header, &dec->io->Main(),
        (frame_header.encoding == FrameEncoding::kModular), dec->io->xsize(),
        dec->io->ysize()));
  }

  // span and reader begin at dec->frame_start, so group_codes_begin can be 0.
  size_t group_codes_begin = 0;

  JXL_RETURN_IF_ERROR(jxl::DecodeDC(
      frame_header, &dec_state, modular_frame_decoder, group_codes_begin,
      group_offsets, group_sizes, pool, reader.get(), aux_outs, aux_out));

  // Copy pixels to output buffer if desired. If no output buffer was set, we
  // merely return the JXL_DEC_FULL_IMAGE status without outputting pixels.
  if (dec->dc_out_buffer) {
    PassesSharedState& shared = dec_state.shared_storage;
    Image3F dc(shared.dc_storage.xsize(), shared.dc_storage.ysize());
    OpsinToLinear(
        shared.dc_storage, Rect(dc), dec->thread_pool.get(), &dc,
        dec->io->metadata.transform_data.opsin_inverse_matrix.ToOpsinParams(
            dec->io->metadata.m.IntensityTarget()));
    CodecInOut dc_io;
    dc_io.SetFromImage(
        std::move(dc),
        ColorEncoding::LinearSRGB(dec->io->metadata.m.color_encoding.IsGray()));
    JXL_API_RETURN_IF_ERROR(ConvertImage(dec, dc_io, dec->dc_out_format,
                                         dec->dc_out_buffer, dec->dc_out_size));
  }

  return true;
}

JxlDecoderStatus JxlDecoderProcessInternal(JxlDecoder* dec, const uint8_t* in,
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
    JxlDecoderStatus status = JxlDecoderReadBasicInfo(dec, in, size);
    if (status != JXL_DEC_SUCCESS) return status;
  }

  if (dec->events_wanted & JXL_DEC_BASIC_INFO) {
    dec->events_wanted &= ~JXL_DEC_BASIC_INFO;
    return JXL_DEC_BASIC_INFO;
  }

  if (!dec->got_all_headers) {
    JxlDecoderStatus status = JxlDecoderReadAllHeaders(dec, in, size);
    if (status != JXL_DEC_SUCCESS) return status;
  }

  if (dec->events_wanted & JXL_DEC_EXTENSIONS) {
    dec->events_wanted &= ~JXL_DEC_EXTENSIONS;
    if (dec->io->metadata.m.extensions != 0) {
      return JXL_DEC_EXTENSIONS;
    }
  }

  if (dec->events_wanted & JXL_DEC_COLOR_ENCODING) {
    dec->events_wanted &= ~JXL_DEC_COLOR_ENCODING;
    return JXL_DEC_COLOR_ENCODING;
  }

  // Read TOC to find required filesize for DC and full frame
  if (!dec->got_toc &&
      (dec->events_wanted & (JXL_DEC_FULL_IMAGE | JXL_DEC_DC_IMAGE))) {
    for (;;) {
      bool is_preview =
          dec->frames_seen == 0 && dec->io->metadata.m.have_preview;
      size_t pos = dec->next_frame_pos;
      if (pos >= size) {
        return JXL_DEC_NEED_MORE_INPUT;
      }
      Span<const uint8_t> span(in + pos, size - pos);
      auto reader = GetBitReader(span);

      dec->frame_header.reset(new FrameHeader(&dec->io->metadata));
      dec->frame_header->nonserialized_is_preview = is_preview;
      jxl::Status status =
          DecodeFrameHeader(reader.get(), dec->frame_header.get());
      dec->frame_dim = dec->frame_header->ToFrameDimensions();

      if (status.code() == StatusCode::kNotEnoughBytes) {
        // TODO(lode): prevent asking for way too much input bytes in case of
        // invalid header that the decoder thinks is a very long user extension
        // instead. Example: fields can currently print something like this:
        // "../lib/jxl/fields.cc:416: Skipping 71467322-bit extension(s)"
        // Maybe fields.cc should return error in the above case rather than
        // print a message.
        return JXL_DEC_NEED_MORE_INPUT;
      } else if (!status) {
        return JXL_API_ERROR("invalid frame header");
      }

      // Read TOC.
      uint64_t groups_total_size;
      const bool has_ac_global = true;
      const size_t toc_entries =
          NumTocEntries(dec->frame_dim.num_groups, dec->frame_dim.num_dc_groups,
                        dec->frame_header->passes.num_passes, has_ac_global);
      status = ReadGroupOffsets(toc_entries, reader.get(), &dec->group_offsets,
                                &dec->group_sizes, &groups_total_size);

      // TODO(lode): we're actually relying on AllReadsWithinBounds() here
      // instead of on status.code(), change the internal TOC C++ code to
      // correctly set the status.code() instead so we can rely on that one.
      if (!reader->AllReadsWithinBounds() ||
          status.code() == StatusCode::kNotEnoughBytes) {
        return JXL_DEC_NEED_MORE_INPUT;
      } else if (!status) {
        return JXL_API_ERROR("invalid toc entries");
      }

      JXL_DASSERT((reader->TotalBitsConsumed() % kBitsPerByte) == 0);
      JXL_API_RETURN_IF_ERROR(reader->JumpToByteBoundary());
      dec->frame_start = pos + (reader->TotalBitsConsumed() >> 3);
      dec->frame_end = dec->frame_start + groups_total_size;
      if (dec->frame_header->animation_frame.duration > 0) {
        return JXL_API_ERROR("Animation frames not yet implemented");
      }

      dec->frames_seen++;
      bool last = dec->frame_header->is_last;
      // Decoding preview not yet supported.
      if (is_preview) last = false;
      // dc_level frames don't have animation_frame at all_default (which sets
      // is_last to true), but are not actually last.
      if (dec->frame_header->dc_level != 0) last = false;
      if (!last) {
        dec->next_frame_pos = dec->frame_end;
        continue;
      }

      dec->got_toc = true;
      break;
    }
  }

  // Decode to pixels, only if required for the events the user wants.
  if (!dec->got_dc_image && (dec->events_wanted & JXL_DEC_DC_IMAGE)) {
    // Compute amount of bytes for the DC image only from the TOC. That is the
    // bytes of all DC groups.

    bool get_dc = true;

    if (dec->frame_header->passes.num_passes == 1 &&
        dec->frame_dim.num_groups == 1) {
      // If there is one pass and one group, the TOC only has one entry and
      // doesn't allow to distinguish the DC size, so it's not easy to tell
      // whether we got all DC bytes or not. This will happen for very small
      // images only.
      get_dc = false;
    } else if (dec->frame_header->color_transform != ColorTransform::kXYB) {
      // The implementation here for now only supports getting DC in XYB case.
      get_dc = false;
    } else if (dec->frame_header->encoding != FrameEncoding::kVarDCT) {
      // The implementation here for now only supports getting DC in the
      // lossy VarDCT case.
      get_dc = false;
    } else if (dec->io->metadata.m.color_encoding.IsGray()) {
      // The implementation here does not yet support grayscale for now.
      get_dc = false;
    }

    if (!get_dc) {
      // Mark DC done and fall through to next decoding stage.
      // The DC is never actually returned in this case, got_dc_image simply
      // means this stage is done.
      // Subscribing to the JXL_DEC_DC_IMAGE event is no guarantee of
      // receiving it for all images.
      dec->got_dc_image = true;
    } else {
      size_t dc_size = 0;
      // one DcGlobal entry, N dc group entries.
      size_t num_dc_toc_entries = 1 + dec->frame_dim.num_dc_groups;
      const std::vector<uint32_t>& group_sizes = dec->group_sizes;
      if (group_sizes.size() < num_dc_toc_entries) {
        return JXL_API_ERROR("too small TOC");
      }
      for (size_t i = 0; i < num_dc_toc_entries; i++) {
        dc_size += group_sizes[i];
      }
      size_t dc_end = dec->frame_start + dc_size;
      // Not yet enough bytes to decode the DC.
      if (dc_end > size) return JXL_DEC_NEED_MORE_INPUT;

      if (!dec->dc_out_buffer_set) {
        return JXL_DEC_NEED_DC_OUT_BUFFER;
      }

      jxl::Status status = DecodeDC(dec, in, size);
      if (!status) {
        return JXL_API_ERROR("decoding dc failed");
      }

      dec->got_dc_image = true;

      return JXL_DEC_DC_IMAGE;
    }
  }

  // Decode to pixels, only if required for the events the user wants.
  if (!dec->got_full_image && (dec->events_wanted & JXL_DEC_FULL_IMAGE)) {
    if (dec->frame_end > size) return JXL_DEC_NEED_MORE_INPUT;

    jxl::DecompressParams dparams;
    jxl::Span<const uint8_t> compressed(in, size);
    jxl::Status status =
        jxl::DecodeFile(dparams, compressed, dec->io.get(), nullptr, nullptr);
    if (!status) {
      return JXL_API_ERROR("decoding file failed");
    }
    dec->got_full_image = true;
  }

  if (dec->events_wanted & JXL_DEC_FULL_IMAGE) {
    if (!dec->image_out_buffer_set) {
      return JXL_DEC_NEED_IMAGE_OUT_BUFFER;
    }

    // TODO(lode): resetting events_wanted for a single frame will not work
    // correctly with multi-frame images.
    dec->events_wanted &= ~JXL_DEC_FULL_IMAGE;

    // Copy pixels to output buffer if desired. If no output buffer was set, we
    // merely return the JXL_DEC_FULL_IMAGE status without outputting pixels.
    if (dec->image_out_buffer) {
      JxlDecoderStatus status =
          ConvertImage(dec, *dec->io.get(), dec->image_out_format,
                       dec->image_out_buffer, dec->image_out_size);
      if (status != JXL_DEC_SUCCESS) return status;
    }

    return JXL_DEC_FULL_IMAGE;
  }

  dec->stage = DecoderStage::kFinished;
  // Return success, this means there is nothing more to do.
  return JXL_DEC_SUCCESS;
}

}  // namespace jxl

JxlDecoderStatus JxlDecoderProcessInput(JxlDecoder* dec,
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
#ifdef JXL_API_DEC_CONSUME_BYTES
  // TODO(lode): if dec->input is empty, we can call
  // jxl::JxlDecoderProcessInternal directly on next_in without copying. If
  // the processing finishes successfully when the user already gave all bytes
  // (one shot), we may never need to copy the bytes. However since process will
  // typically return multiple status codes (so require multiple calls), the
  // copy would happen anyway, unless the decoder marks that it no longer needs
  // to read them.
  dec->input.insert(dec->input.end(), *next_in, *next_in + *avail_in);
  JxlDecoderStatus status =
      jxl::JxlDecoderProcessInternal(dec, dec->input.data(), dec->input.size());
  *next_in += *avail_in;
  *avail_in = 0;
  return status;
#else   // JXL_API_DEC_CONSUME_BYTES
  return jxl::JxlDecoderProcessInternal(dec, *next_in, *avail_in);
#endif  // JXL_API_DEC_CONSUME_BYTES
}

JxlDecoderStatus JxlDecoderGetBasicInfo(const JxlDecoder* dec,
                                        JxlBasicInfo* info) {
  if (!dec->got_basic_info) return JXL_DEC_NEED_MORE_INPUT;

  if (info) {
    const jxl::ImageMetadata& meta = dec->io->metadata.m;

    info->have_container = dec->have_container;
    info->xsize = dec->io->xsize();
    info->ysize = dec->io->ysize();
    info->uses_original_profile = !meta.xyb_encoded;

    info->bits_per_sample = meta.bit_depth.bits_per_sample;
    info->exponent_bits_per_sample = meta.bit_depth.exponent_bits_per_sample;

    info->have_preview = meta.have_preview;
    info->have_animation = meta.have_animation;
    // TODO(janwas): intrinsic_size
    info->orientation = static_cast<JxlOrientation>(meta.orientation);

    if (!dec->keep_orientation) {
      if (info->orientation >= JXL_ORIENT_TRANSPOSE) {
        std::swap(info->xsize, info->ysize);
      }
      info->orientation = JXL_ORIENT_IDENTITY;
    }

    info->intensity_target = meta.IntensityTarget();
    info->min_nits = meta.tone_mapping.min_nits;
    info->relative_to_max_display = meta.tone_mapping.relative_to_max_display;
    info->linear_below = meta.tone_mapping.linear_below;

    const jxl::ExtraChannelInfo* alpha = meta.Find(jxl::ExtraChannel::kAlpha);
    if (alpha != nullptr) {
      info->alpha_bits = alpha->bit_depth.bits_per_sample;
      info->alpha_exponent_bits = alpha->bit_depth.exponent_bits_per_sample;
      info->alpha_premultiplied = alpha->alpha_associated;
    } else {
      info->alpha_bits = 0;
      info->alpha_exponent_bits = 0;
      info->alpha_premultiplied = 0;
    }

    info->num_extra_channels = meta.num_extra_channels;

    if (info->have_preview) {
      info->preview.xsize = dec->io->metadata.m.preview_size.xsize();
      info->preview.ysize = dec->io->metadata.m.preview_size.ysize();
    }

    if (info->have_animation) {
      info->animation.tps_numerator =
          dec->io->metadata.m.animation.tps_numerator;
      info->animation.tps_denominator =
          dec->io->metadata.m.animation.tps_denominator;
      info->animation.num_loops = dec->io->metadata.m.animation.num_loops;
      info->animation.have_timecodes =
          dec->io->metadata.m.animation.have_timecodes;
    }
  }

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetExtraChannelInfo(const JxlDecoder* dec,
                                               size_t index,
                                               JxlExtraChannelInfo* info) {
  if (!dec->got_basic_info) return JXL_DEC_NEED_MORE_INPUT;

  const std::vector<jxl::ExtraChannelInfo>& channels =
      dec->io->metadata.m.extra_channel_info;

  if (index >= channels.size()) return JXL_DEC_ERROR;  // out of bounds
  const jxl::ExtraChannelInfo& channel = channels[index];

  info->type = static_cast<JxlExtraChannelType>(channel.type);
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

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetExtraChannelName(const JxlDecoder* dec,
                                               size_t index, char* name,
                                               size_t size) {
  if (!dec->got_basic_info) return JXL_DEC_NEED_MORE_INPUT;

  const std::vector<jxl::ExtraChannelInfo>& channels =
      dec->io->metadata.m.extra_channel_info;

  if (index >= channels.size()) return JXL_DEC_ERROR;  // out of bounds
  const jxl::ExtraChannelInfo& channel = channels[index];

  // Also need null-termination character
  if (channel.name.size() + 1 > size) return JXL_DEC_ERROR;

  memcpy(name, channel.name.c_str(), channel.name.size() + 1);

  return JXL_DEC_SUCCESS;
}

namespace {

// Gets the jxl::ColorEncoding for the desired target, and checks errors.
// Returns the object regardless of whether the actual color space is in ICC,
// but ensures that if the color encoding is not the encoding from the
// codestream header metadata, it cannot require ICC profile.
JxlDecoderStatus GetColorEncodingForTarget(
    const JxlDecoder* dec, const JxlPixelFormat* format,
    JxlColorProfileTarget target, const jxl::ColorEncoding** encoding) {
  if (!dec->got_all_headers) return JXL_DEC_NEED_MORE_INPUT;

  *encoding = nullptr;
  if (target == JXL_COLOR_PROFILE_TARGET_DATA &&
      dec->io->metadata.m.xyb_encoded) {
    // The profile of the pixels matches dec->io->Main().c_current(). However,
    // c_current in the ImageBundle is not yet filled in correctly at this point
    // since the pixels have not been decoded yet.
    // Instead, output the profile that the API specifies it uses for this case:
    // linear sRGB for floating point output, and nonlinear sRGB for integer
    // output, grayscale or color depending on the image header.
    bool grayscale = dec->io->metadata.m.color_encoding.IsGray();
    if (!format) {
      return JXL_API_ERROR("Must provide pixel format for data color profile");
    }
    if (format->data_type == JXL_TYPE_FLOAT) {
      *encoding = &jxl::ColorEncoding::LinearSRGB(grayscale);
    } else {
      *encoding = &jxl::ColorEncoding::SRGB(grayscale);
    }
  } else {
    *encoding = &dec->io->metadata.m.color_encoding;
  }
  return JXL_DEC_SUCCESS;
}
}  // namespace

JxlDecoderStatus JxlDecoderGetColorAsEncodedProfile(
    const JxlDecoder* dec, const JxlPixelFormat* format,
    JxlColorProfileTarget target, JxlColorEncoding* color_encoding) {
  const jxl::ColorEncoding* jxl_color_encoding = nullptr;
  JxlDecoderStatus status =
      GetColorEncodingForTarget(dec, format, target, &jxl_color_encoding);
  if (status) return status;

  if (jxl_color_encoding->WantICC())
    return JXL_DEC_ERROR;  // Indicate no encoded profile available.

  if (color_encoding) {
    color_encoding->color_space =
        static_cast<JxlColorSpace>(jxl_color_encoding->GetColorSpace());

    color_encoding->white_point =
        static_cast<JxlWhitePoint>(jxl_color_encoding->white_point);

    jxl::CIExy whitepoint = jxl_color_encoding->GetWhitePoint();
    color_encoding->white_point_xy[0] = whitepoint.x;
    color_encoding->white_point_xy[1] = whitepoint.y;

    color_encoding->primaries =
        static_cast<JxlPrimaries>(jxl_color_encoding->primaries);
    jxl::PrimariesCIExy primaries = jxl_color_encoding->GetPrimaries();
    color_encoding->primaries_red_xy[0] = primaries.r.x;
    color_encoding->primaries_red_xy[1] = primaries.r.y;
    color_encoding->primaries_green_xy[0] = primaries.g.x;
    color_encoding->primaries_green_xy[1] = primaries.g.y;
    color_encoding->primaries_blue_xy[0] = primaries.b.x;
    color_encoding->primaries_blue_xy[1] = primaries.b.y;

    if (jxl_color_encoding->tf.IsGamma()) {
      color_encoding->transfer_function = JXL_TRANSFER_FUNCTION_GAMMA;
      color_encoding->gamma = jxl_color_encoding->tf.GetGamma();
    } else {
      color_encoding->transfer_function = static_cast<JxlTransferFunction>(
          jxl_color_encoding->tf.GetTransferFunction());
      color_encoding->gamma = 0;
    }

    color_encoding->rendering_intent =
        static_cast<JxlRenderingIntent>(jxl_color_encoding->rendering_intent);
  }

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetICCProfileSize(const JxlDecoder* dec,
                                             const JxlPixelFormat* format,
                                             JxlColorProfileTarget target,
                                             size_t* size) {
  const jxl::ColorEncoding* jxl_color_encoding = nullptr;
  JxlDecoderStatus status =
      GetColorEncodingForTarget(dec, format, target, &jxl_color_encoding);
  if (status != JXL_DEC_SUCCESS) return status;

  if (jxl_color_encoding->WantICC()) {
    jxl::ColorSpace color_space =
        dec->io->metadata.m.color_encoding.GetColorSpace();
    if (color_space == jxl::ColorSpace::kUnknown ||
        color_space == jxl::ColorSpace::kXYB) {
      // This indicates there's no ICC profile available
      // TODO(lode): for the XYB case, do we want to craft an ICC profile that
      // represents XYB as an RGB profile? It may be possible, but not with
      // only 1D transfer functions.
      return JXL_DEC_ERROR;
    }
  }

  if (size) {
    *size = jxl_color_encoding->ICC().size();
  }

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetColorAsICCProfile(const JxlDecoder* dec,
                                                const JxlPixelFormat* format,
                                                JxlColorProfileTarget target,
                                                uint8_t* icc_profile,
                                                size_t size) {
  size_t wanted_size;
  // This also checks the NEED_MORE_INPUT and the unknown/xyb cases
  JxlDecoderStatus status =
      JxlDecoderGetICCProfileSize(dec, format, target, &wanted_size);
  if (status != JXL_DEC_SUCCESS) return status;
  if (size < wanted_size) return JXL_API_ERROR("ICC profile output too small");

  const jxl::ColorEncoding* jxl_color_encoding = nullptr;
  status = GetColorEncodingForTarget(dec, format, target, &jxl_color_encoding);
  if (status != JXL_DEC_SUCCESS) return status;

  memcpy(icc_profile, jxl_color_encoding->ICC().data(),
         jxl_color_encoding->ICC().size());

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetInverseOpsinMatrix(
    const JxlDecoder* dec, JxlInverseOpsinMatrix* matrix) {
  memcpy(matrix->opsin_inv_matrix,
         dec->io->metadata.transform_data.opsin_inverse_matrix.inverse_matrix,
         sizeof(matrix->opsin_inv_matrix));
  memcpy(matrix->opsin_biases,
         dec->io->metadata.transform_data.opsin_inverse_matrix.opsin_biases,
         sizeof(matrix->opsin_biases));
  memcpy(matrix->quant_biases,
         dec->io->metadata.transform_data.opsin_inverse_matrix.quant_biases,
         sizeof(matrix->quant_biases));

  return JXL_DEC_SUCCESS;
}

namespace {
// Returns the amount of bits needed for getting memory buffer size, and does
// all error checking required for size checking and format validity.
JxlDecoderStatus PrepareSizeCheck(const JxlDecoder* dec,
                                  const JxlPixelFormat* format, size_t* bits) {
  if (format->align > 1) {
    return JXL_API_ERROR("align in JxlPixelFormat is not yet supported");
  }
  if (!dec->got_basic_info) {
    // Don't know image dimensions yet, cannot check for valid size.
    return JXL_DEC_NEED_MORE_INPUT;
  }
  if (format->num_channels > 4) {
    return JXL_API_ERROR("More than 4 channels not supported");
  }
  if (format->num_channels < 3 &&
      !dec->io->metadata.m.color_encoding.IsGray()) {
    return JXL_API_ERROR("Grayscale output not possible for color image");
  }

  *bits = BitsPerChannel(format->data_type);

  if (*bits == 0) {
    return JXL_API_ERROR("Invalid data type");
  }

  return JXL_DEC_SUCCESS;
}
}  // namespace

JXL_EXPORT JxlDecoderStatus JxlDecoderDCOutBufferSize(
    const JxlDecoder* dec, const JxlPixelFormat* format, size_t* size) {
  size_t bits;
  JxlDecoderStatus status = PrepareSizeCheck(dec, format, &bits);
  if (status != JXL_DEC_SUCCESS) return status;

  size_t xsize = jxl::DivCeil(dec->io->xsize(), jxl::kBlockDim);
  size_t ysize = jxl::DivCeil(dec->io->ysize(), jxl::kBlockDim);

  size_t row_size =
      jxl::DivCeil(xsize * format->num_channels * bits, jxl::kBitsPerByte);
  *size = row_size * ysize;
  return JXL_DEC_SUCCESS;
}

JXL_EXPORT JxlDecoderStatus JxlDecoderSetDCOutBuffer(
    JxlDecoder* dec, const JxlPixelFormat* format, void* buffer, size_t size) {
  size_t min_size;
  // This also checks whether the format is valid and supported and basic info
  // is available.
  JxlDecoderStatus status = JxlDecoderDCOutBufferSize(dec, format, &min_size);
  if (status != JXL_DEC_SUCCESS) return status;

  if (size < min_size) return JXL_DEC_ERROR;

  dec->dc_out_buffer_set = true;
  dec->dc_out_buffer = buffer;
  dec->dc_out_size = size;
  dec->dc_out_format = *format;

  return JXL_DEC_SUCCESS;
}

JXL_EXPORT JxlDecoderStatus JxlDecoderImageOutBufferSize(
    const JxlDecoder* dec, const JxlPixelFormat* format, size_t* size) {
  size_t bits;
  JxlDecoderStatus status = PrepareSizeCheck(dec, format, &bits);
  if (status != JXL_DEC_SUCCESS) return status;

  size_t row_size = jxl::DivCeil(dec->io->xsize() * format->num_channels * bits,
                                 jxl::kBitsPerByte);
  *size = row_size * dec->io->ysize();

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetImageOutBuffer(JxlDecoder* dec,
                                             const JxlPixelFormat* format,
                                             void* buffer, size_t size) {
  size_t min_size;
  // This also checks whether the format is valid and supported and basic info
  // is available.
  JxlDecoderStatus status =
      JxlDecoderImageOutBufferSize(dec, format, &min_size);
  if (status != JXL_DEC_SUCCESS) return status;

  if (size < min_size) return JXL_DEC_ERROR;

  dec->image_out_buffer_set = true;
  dec->image_out_buffer = buffer;
  dec->image_out_size = size;
  dec->image_out_format = *format;

  return JXL_DEC_SUCCESS;
}
