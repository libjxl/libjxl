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

  // JPEG XL codestream: 0xff 0x0a
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
  JxlDecoderStruct() = default;

  JxlMemoryManager memory_manager;
  std::unique_ptr<jxl::ThreadPool> thread_pool;

  DecoderStage stage;

  // Status of progression, internal.
  bool got_signature;
  bool first_codestream_seen;
  // Indicates we know that we've seen the last codestream, however this is not
  // guaranteed to be true for the last box because a jxl file may have multiple
  // "jxlp" boxes and it is possible (and permitted) that the last one is not a
  // final box that uses size 0 to indicate the end.
  bool last_codestream_seen;
  bool got_basic_info;
  bool got_all_headers;  // Codestream metadata headers

  // This means either we actually got the preview image, or determined we
  // cannot get it or there is none.
  bool got_preview_image;

  // For current frame
  // Got these steps of the current frame. Reset once started on a next frame.
  bool got_toc;
  bool got_dc_image;
  bool got_full_image;
  // Processed the last frame, so got_toc, got_dc_image, and so on false no
  // longer mean there is more work to do.
  bool last_frame_reached;

  // Position of next_in in the original file including box format if present
  // (as opposed to positiion in the codestream)
  size_t file_pos;
  // Begin and end of the content of the current codestream box. This could be
  // a partial codestream box.
  // codestream_begin 0 is used to indicate the begin is not yet known.
  // codestream_end 0 is used to indicate uncapped (until end of file, for the
  // last box if this box doesn't indicate its actual size).
  // Not used if the file is a direct codestream.
  size_t codestream_begin;
  size_t codestream_end;

  // Settings
  bool keep_orientation;

  // Bitfield, for which informative events (JXL_DEC_BASIC_INFO, etc...) the
  // decoder returns a status. By default, do not return for any of the events,
  // only return when the decoder cannot continue becasue it needs mor input or
  // output data.
  int events_wanted;
  int orig_events_wanted;

  // Fields for reading the basic info from the header.
  size_t basic_info_size_hint;
  bool have_container;

  // Whether the DC out buffer was set. It is possible for dc_out_buffer to
  // be nullptr and dc_out_buffer_set be true, indicating it was deliberately
  // set to nullptr.
  bool preview_out_buffer_set;
  bool dc_out_buffer_set;
  // Idem for the image buffer.
  bool image_out_buffer_set;

  bool need_preview_out_buffer;
  bool need_dc_out_buffer;
  bool need_image_out_buffer;

  // Owned by the caller, buffers for DC image and full resolution images
  void* preview_out_buffer;
  void* dc_out_buffer;
  void* image_out_buffer;

  size_t preview_out_size;
  size_t dc_out_size;
  size_t image_out_size;

  // TODO(lode): merge these?
  JxlPixelFormat preview_out_format;
  JxlPixelFormat dc_out_format;
  JxlPixelFormat image_out_format;

  jxl::CodecMetadata metadata;
  std::unique_ptr<jxl::ImageBundle> ib;

  std::unique_ptr<jxl::PassesDecoderState> passes_state;

  // headers and TOC for the current frame. When got_toc is true, this is
  // always the frame header of the last frame of the current still series,
  // that is, the displayed frame.
  std::unique_ptr<jxl::FrameHeader> frame_header;
  jxl::FrameDimensions frame_dim;

  // Start of the preview frame, in codestream bytes, or 0 if there is no
  // preview.
  // size_t preview_frame_start;
  // Start of the first frame in the JXL file. If there is a preview, this will
  // be the preview frame. Otherwise it's the first true frame.
  size_t first_frame_start;
  // Start of the current composite still being processed, in codestream bytes.
  // A composite still is a group of 1 or more frames that are dispalyed
  // together during one animation tick, or if there is no animation, a single
  // still image made by blending multiple frames, or just a single frame. If
  // this is equal to preview start, then we're processing the preview frame.
  size_t still_start;
  size_t still_end;
  // Start of the current frame being processed, in the current still series.
  // In the case of a simple single-frame JXL image, or an animation where the
  // animation frame is made out of just a single JXL frame, this is equal to
  // still_start. In case of a composite still, frame_start increments to next
  // frames while still_start does not until a frame with duration is
  // encountered.
  size_t frame_start;

  // Codestream input data is stored here, when the decoder takes in and stores
  // the user input bytes. If the decoder does not do that (e.g. in one-shot
  // case), this field is unused.
  // TODO(lode): avoid needing this field once the C++ decoder doesn't need
  // all bytes at once, to save memory. Find alternative to std::vector doubling
  // strategy to prevent some memory usage.
  std::vector<uint8_t> codestream;

  // Position in the actual codestream, which codestream.begin() points to.
  // Non-zero once earlier parts of the codestream vector have been erased.
  size_t codestream_pos;

  // Statistics which CodecInOut can keep
  uint64_t dec_pixels;

  const uint8_t* next_in;
  size_t avail_in;
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
  dec->got_signature = false;
  dec->first_codestream_seen = false;
  dec->last_codestream_seen = false;
  dec->got_basic_info = false;
  dec->got_all_headers = false;
  dec->got_toc = false;
  dec->got_preview_image = false;
  dec->got_dc_image = false;
  dec->got_full_image = false;
  dec->last_frame_reached = false;
  dec->file_pos = 0;
  dec->codestream_pos = 0;
  dec->codestream_begin = 0;
  dec->codestream_end = 0;
  dec->keep_orientation = false;
  dec->events_wanted = 0;
  dec->orig_events_wanted = 0;
  dec->basic_info_size_hint = InitialBasicInfoSizeHint();
  dec->have_container = 0;
  dec->preview_out_buffer_set = false;
  dec->dc_out_buffer_set = false;
  dec->image_out_buffer_set = false;
  dec->need_preview_out_buffer = false;
  dec->need_dc_out_buffer = false;
  dec->need_image_out_buffer = false;
  dec->preview_out_buffer = nullptr;
  dec->dc_out_buffer = nullptr;
  dec->image_out_buffer = nullptr;
  dec->preview_out_size = 0;
  dec->dc_out_size = 0;
  dec->image_out_size = 0;
  dec->dec_pixels = 0;
  dec->next_in = 0;
  dec->avail_in = 0;

  dec->passes_state.reset(nullptr);

  dec->frame_header.reset(new jxl::FrameHeader(&dec->metadata));
  dec->frame_dim = jxl::FrameDimensions();
  dec->codestream.clear();
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
  if (dec->got_basic_info) return 0;
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
  dec->orig_events_wanted = events_wanted;
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
namespace {

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
  size_t pos = 0;

  // Check and skip the codestream signature
  JxlSignature signature = ReadSignature(in, size, &pos);
  if (signature == JXL_SIG_NOT_ENOUGH_BYTES) {
    return JXL_DEC_NEED_MORE_INPUT;
  }
  if (signature == JXL_SIG_CONTAINER) {
    // There is a container signature where we expect a codestream, container
    // is handled at a higher level already.
    return JXL_API_ERROR("invalid: nested container");
  }
  if (signature != JXL_SIG_CODESTREAM) {
    return JXL_API_ERROR("invalid signature");
  }

  Span<const uint8_t> span(in + pos, size - pos);
  auto reader = GetBitReader(span);
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dec->metadata.size));

  dec->metadata.m.nonserialized_only_parse_basic_info = true;
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dec->metadata.m));
  dec->metadata.m.nonserialized_only_parse_basic_info = false;
  dec->got_basic_info = true;
  dec->basic_info_size_hint = 0;

  return JXL_DEC_SUCCESS;
}

// Reads all codestream headers (but not frame headers)
JxlDecoderStatus JxlDecoderReadAllHeaders(JxlDecoder* dec, const uint8_t* in,
                                          size_t size) {
  size_t pos = 0;

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

  // We already decoded the metadata to dec->metadata.m, no reason to
  // overwrite it, use a dummy metadata instead.
  ImageMetadata dummy_metadata;
  JXL_API_RETURN_IF_ERROR(ReadBundle(span, reader.get(), &dummy_metadata));

  JXL_API_RETURN_IF_ERROR(
      ReadBundle(span, reader.get(), &dec->metadata.transform_data));

  if (dec->metadata.m.color_encoding.WantICC()) {
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
    if (!dec->metadata.m.color_encoding.SetICC(std::move(icc))) {
      return JXL_DEC_ERROR;
    }
  }

  dec->got_all_headers = true;
  JXL_API_RETURN_IF_ERROR(reader->JumpToByteBoundary());

  dec->first_frame_start =
      pos + reader->TotalBitsConsumed() / jxl::kBitsPerByte;
  dec->frame_start = dec->first_frame_start;
  dec->still_start = dec->first_frame_start;

  return JXL_DEC_SUCCESS;
}

static JxlDecoderStatus ConvertImage(const JxlDecoder* dec,
                                     const jxl::ImageBundle& frame,
                                     const JxlPixelFormat& format,
                                     void* out_image, size_t out_size) {
  // TODO(lode): handle mismatch of RGB/grayscale color profiles and pixel data
  // color/grayscale format
  const auto& metadata = dec->metadata.m;

  size_t stride = frame.xsize() * (BitsPerChannel(format.data_type) *
                                   format.num_channels / jxl::kBitsPerByte);
  if (format.align > 1) {
    stride = jxl::DivCeil(stride, format.align) * format.align;
  }

  bool apply_srgb_tf = false;
  if (metadata.xyb_encoded) {
    if (!frame.c_current().IsLinearSRGB() && !frame.c_current().IsSRGB()) {
      return JXL_API_ERROR(
          "Error, the implementation expects that ImageBundle is in linear "
          "or nonlinear sRGB when the image was xyb_encoded");
    }
    if (format.data_type != JXL_TYPE_FLOAT &&
        frame.c_current().IsLinearSRGB()) {
      // Convert to nonlinear sRGB for integer pixels.
      apply_srgb_tf = true;
    }
  }
  jxl::Orientation undo_orientation = dec->keep_orientation
                                          ? metadata.GetOrientation()
                                          : jxl::Orientation::kIdentity;
  jxl::Status status = jxl::ConvertImage(
      frame, BitsPerChannel(format.data_type),
      format.data_type == JXL_TYPE_FLOAT, apply_srgb_tf, format.num_channels,
      format.endianness, stride, dec->thread_pool.get(), out_image, out_size,
      undo_orientation);

  return status ? JXL_DEC_SUCCESS : JXL_DEC_ERROR;
}

// Reads all frame headers and computes the total size in bytes of the frame.
// Stores information in dec->frame_header and dec->frame_dim.
// Outputs optional variables, unless set to nullptr:
// frame_size: total frame size
// header_size: size of the frame header and TOC within the frame
// dc_size: size of DC groups within the frame, or 0 if there's no DC or we're
// unable to compute its size.
// group_offsets and group_sizes: information for groups and passes
// Can finish successfully if reader has headers and TOC available, does not
// read groups themselves.
JxlDecoderStatus ParseFrameHeader(JxlDecoder* dec,
                                  jxl::FrameHeader* frame_header,
                                  const uint8_t* in, size_t size, size_t pos,
                                  bool is_preview, size_t* frame_size,
                                  size_t* header_size, size_t* dc_size,
                                  std::vector<uint64_t>* group_offsets,
                                  std::vector<uint32_t>* group_sizes) {
  Span<const uint8_t> span(in + pos, size - pos);
  auto reader = GetBitReader(span);

  frame_header->nonserialized_is_preview = is_preview;
  jxl::Status status = DecodeFrameHeader(reader.get(), frame_header);
  dec->frame_dim = frame_header->ToFrameDimensions();

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
                    frame_header->passes.num_passes, has_ac_global);

  std::vector<uint64_t> group_offsets_;
  std::vector<uint32_t> group_sizes_;
  status = ReadGroupOffsets(toc_entries, reader.get(), &group_offsets_,
                            &group_sizes_, &groups_total_size);
  if (dc_size) {
    bool can_get_dc = true;
    if (frame_header->passes.num_passes == 1 &&
        dec->frame_dim.num_groups == 1) {
      // If there is one pass and one group, the TOC only has one entry and
      // doesn't allow to distinguish the DC size, so it's not easy to tell
      // whether we got all DC bytes or not. This will happen for very small
      // images only.
      can_get_dc = false;
    }

    *dc_size = 0;
    if (can_get_dc) {
      // one DcGlobal entry, N dc group entries.
      size_t num_dc_toc_entries = 1 + dec->frame_dim.num_dc_groups;
      if (group_sizes_.size() < num_dc_toc_entries) {
        JXL_ABORT("too small TOC");
      }
      for (size_t i = 0; i < num_dc_toc_entries; i++) {
        *dc_size =
            std::max<size_t>(*dc_size, group_sizes_[i] + group_offsets_[i]);
      }
    }
  }

  if (group_offsets) group_offsets->swap(group_offsets_);
  if (group_sizes) group_sizes->swap(group_sizes_);

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
  size_t header_size_ = (reader->TotalBitsConsumed() >> 3);
  if (header_size) *header_size = header_size_;
  *frame_size = header_size_ + groups_total_size;

  return JXL_DEC_SUCCESS;
}

// TODO(lode): share more of this code with the C++ decoder implementation in
// dec_frame.cc.
jxl::Status DecodeDC(JxlDecoder* dec, const uint8_t* in, size_t size) {
  size_t pos = 0;
  FrameHeader frame_header(&dec->metadata);
  std::vector<uint64_t> group_offsets;
  std::vector<uint32_t> group_sizes;
  size_t frame_size, header_size, dc_size;
  JxlDecoderStatus status =
      ParseFrameHeader(dec, &frame_header, in, size, pos, false, &frame_size,
                       &header_size, &dc_size, &group_offsets, &group_sizes);
  if (status != JXL_DEC_SUCCESS) return status;
  jxl::FrameDimensions frame_dim = frame_header.ToFrameDimensions();

  PassesDecoderState dec_state;
  ModularFrameDecoder modular_frame_decoder;
  modular_frame_decoder.Init(frame_dim);
  ThreadPool* pool = nullptr;
  std::vector<AuxOut>* aux_outs = nullptr;
  AuxOut* JXL_RESTRICT aux_out = nullptr;
  JXL_RETURN_IF_ERROR(
      InitializePassesSharedState(frame_header, &dec_state.shared_storage));

  Span<const uint8_t> span(in + header_size, size - header_size);
  auto reader = GetBitReader(span);

  {
    PassesSharedState& shared = dec_state.shared_storage;

    {
      if (shared.frame_header.flags & FrameHeader::kPatches) {
        JXL_RETURN_IF_ERROR(shared.image_features.patches.Decode(
            reader.get(), shared.frame_dim.xsize_padded,
            shared.frame_dim.ysize_padded));
      }
      if (shared.frame_header.flags & FrameHeader::kSplines) {
        JXL_RETURN_IF_ERROR(shared.image_features.splines.Decode(
            reader.get(), shared.frame_dim.xsize * shared.frame_dim.ysize));
      }
      if (shared.frame_header.flags & FrameHeader::kNoise) {
        JXL_RETURN_IF_ERROR(
            DecodeNoise(reader.get(), &shared.image_features.noise_params));
      }
    }

    // TODO(lode): support non-lossy, grayscale and/or non-xyb as well.
    JXL_RETURN_IF_ERROR(shared.matrices.DecodeDC(reader.get()));
    if (frame_header.encoding == FrameEncoding::kVarDCT) {
      JXL_RETURN_IF_ERROR(jxl::DecodeGlobalDCInfo(
          reader.get(), /*is_jpeg=*/false, &dec_state, dec->thread_pool.get()));
    } else if (frame_header.encoding == FrameEncoding::kModular) {
      dec_state.Init(dec->thread_pool.get());
    }
    JXL_RETURN_IF_ERROR(
        modular_frame_decoder.DecodeGlobalInfo(reader.get(), frame_header));
  }

  // span and reader begin at groups start, so group_codes_begin can be 0.
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
        dec->metadata.transform_data.opsin_inverse_matrix.ToOpsinParams(
            dec->metadata.m.IntensityTarget()));
    // TODO(lode): use the real metadata instead, this requires matching all
    // the extra channels. Support DC with alpha too.
    jxl::ImageMetadata dummy;
    ImageBundle dc_bundle(&dummy);
    dc_bundle.SetFromImage(
        std::move(dc),
        ColorEncoding::LinearSRGB(dec->metadata.m.color_encoding.IsGray()));
    JXL_API_RETURN_IF_ERROR(ConvertImage(dec, dc_bundle, dec->dc_out_format,
                                         dec->dc_out_buffer, dec->dc_out_size));
  }

  return true;
}

// TODO(eustas): no CodecInOut -> no image size reinforcement -> possible OOM.
JxlDecoderStatus JxlDecoderProcessInternal(JxlDecoder* dec, const uint8_t* in,
                                           size_t size) {
  // If no parallel runner is set, use the default
  // TODO(lode): move this initialization to an appropriate location once the
  // runner is used to decode pixels.
  if (!dec->thread_pool) {
    dec->thread_pool.reset(new jxl::ThreadPool(nullptr, nullptr));
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
    if (dec->metadata.m.extensions != 0) {
      return JXL_DEC_EXTENSIONS;
    }
  }

  if (dec->events_wanted & JXL_DEC_COLOR_ENCODING) {
    dec->events_wanted &= ~JXL_DEC_COLOR_ENCODING;
    return JXL_DEC_COLOR_ENCODING;
  }

  // Decode to pixels, only if required for the events the user wants.
  if (!dec->got_preview_image && (dec->events_wanted & JXL_DEC_PREVIEW_IMAGE)) {
    if (!dec->metadata.m.have_preview) {
      // There is no preview, mark this as done and go to next step
      dec->got_preview_image = true;
    } else {
      size_t frame_size;
      size_t pos = dec->first_frame_start;
      dec->frame_header.reset(new FrameHeader(&dec->metadata));
      JxlDecoderStatus status =
          ParseFrameHeader(dec, dec->frame_header.get(), in, size, pos, true,
                           &frame_size, nullptr, nullptr, nullptr, nullptr);
      if (status != JXL_DEC_SUCCESS) return status;
      if (OutOfBounds(pos, frame_size, size)) {
        return JXL_DEC_NEED_MORE_INPUT;
      }

      if (!dec->preview_out_buffer_set) {
        dec->need_preview_out_buffer = true;
        return JXL_DEC_NEED_PREVIEW_OUT_BUFFER;
      }

      jxl::Span<const uint8_t> compressed(in + dec->first_frame_start,
                                          size - dec->first_frame_start);
      auto reader = GetBitReader(compressed);
      jxl::DecompressParams dparams;
      jxl::ImageBundle ib(&dec->metadata.m);
      if (!DecodePreview(dparams, dec->metadata, reader.get(),
                         /*aux_out=*/nullptr, dec->thread_pool.get(), &ib,
                         &dec->dec_pixels, /*constraints=*/nullptr)) {
        return JXL_API_ERROR("decoding preview failed");
      }

      if (dec->preview_out_buffer) {
        JxlDecoderStatus status =
            ConvertImage(dec, ib, dec->preview_out_format,
                         dec->preview_out_buffer, dec->preview_out_size);
        if (status != JXL_DEC_SUCCESS) return status;
      }
      dec->got_preview_image = true;
      return JXL_DEC_PREVIEW_IMAGE;
    }
  }

  std::vector<uint64_t> group_offsets;
  std::vector<uint32_t> group_sizes;

  // Handle frames
  for (;;) {
    if (!(dec->events_wanted &
          (JXL_DEC_FULL_IMAGE | JXL_DEC_DC_IMAGE | JXL_DEC_FRAME))) {
      break;
    }
    // Read TOC to find required filesize for DC and full frame, or all
    // composite still frames in a series.
    if (!dec->got_toc) {
      if (!dec->codestream.empty() && dec->codestream_pos < dec->frame_start &&
          dec->frame_start - dec->codestream_pos <= dec->codestream.size()) {
        // Remove earlier bytes from the codestream vector, if the input comes
        // from there.
        // TODO(lode): it would be better if this is indicated using next_in and
        // avail_in instead, currently this relies on JxlDecoderProcess using
        // dec->codestream as soon as it's not empty.
        size_t diff = dec->frame_start - dec->codestream_pos;
        dec->codestream.erase(dec->codestream.begin(),
                              dec->codestream.begin() + diff);
        dec->codestream_pos = dec->frame_start;
        size -= diff;
        in = dec->codestream.data();
      }
      // First get size of all frames belonging to the current still
      for (;;) {
        size_t pos = dec->frame_start - dec->codestream_pos;
        if (pos >= size) {
          return JXL_DEC_NEED_MORE_INPUT;
        }
        size_t frame_size, header_size;
        bool is_preview = (dec->frame_start == dec->first_frame_start) &&
                          dec->metadata.m.have_preview;
        dec->frame_header.reset(new FrameHeader(&dec->metadata));
        JxlDecoderStatus status = ParseFrameHeader(
            dec, dec->frame_header.get(), in, size, pos, is_preview,
            &frame_size, &header_size, nullptr, &group_offsets, &group_sizes);
        if (status != JXL_DEC_SUCCESS) return status;

        // last of the current still frame series. That means it's the last if
        // it has a duration or if it's the last frame of the entire codestream.
        bool last_of_this_series = false;
        if (dec->frame_header->is_last) {
          last_of_this_series = true;
        }
        if (dec->frame_header->animation_frame.duration > 0) {
          last_of_this_series = true;
        }
        // The preview is not part of the current still and should be skipped.
        if (is_preview) {
          dec->still_start = dec->frame_start + frame_size;
          // The preview is not the last frame, no matter what its header says.
          last_of_this_series = false;
        }

        dec->frame_start += frame_size;

        if (!last_of_this_series) continue;

        dec->got_toc = true;
        // frame_start has already been incremented to the next frame
        dec->still_end = dec->frame_start;
        break;
      }
    }

    if (dec->events_wanted & JXL_DEC_FRAME) {
      dec->events_wanted &= ~JXL_DEC_FRAME;
      return JXL_DEC_FRAME;
    }

    // Decode to pixels, only if required for the events the user wants.
    if (!dec->got_dc_image && (dec->events_wanted & JXL_DEC_DC_IMAGE)) {
      size_t pos = dec->still_start - dec->codestream_pos;
      // Compute amount of bytes for the DC image only from the TOC. That is the
      // bytes of all DC groups.
      size_t frame_size, header_size, dc_size;
      FrameHeader frame_header(&dec->metadata);
      JxlDecoderStatus status = ParseFrameHeader(
          dec, &frame_header, in, size, pos, false, &frame_size, &header_size,
          &dc_size, &group_offsets, &group_sizes);
      if (status != JXL_DEC_SUCCESS) return status;

      bool get_dc = true;

      if (dec->frame_header->color_transform != ColorTransform::kXYB) {
        // The implementation here for now only supports getting DC in XYB case.
        get_dc = false;
      } else if (dec->frame_header->encoding != FrameEncoding::kVarDCT) {
        // The implementation here for now only supports getting DC in the
        // lossy VarDCT case.
        get_dc = false;
      } else if (dec->metadata.m.color_encoding.IsGray()) {
        // The implementation here does not yet support grayscale for now.
        get_dc = false;
      } else if (dec->metadata.m.HasAlpha()) {
        // Getting alpha from DC not yet supported.
        // TODO(lode): Support DC with alpha. Note that it may not be possible
        // if there's no progressive rendering for the modular alpha channel.
        // The only way to find that out is from the DcGlobal section, but we
        // could let encoder indicate this with kSkipProgressive instead.
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
        // Not yet enough bytes to decode the DC.
        if (dc_size > size) return JXL_DEC_NEED_MORE_INPUT;
        if (!dec->dc_out_buffer_set) {
          dec->need_dc_out_buffer = true;
          return JXL_DEC_NEED_DC_OUT_BUFFER;
        }
        jxl::Status status =
            DecodeDC(dec, in + (dec->still_start - dec->codestream_pos),
                     size - (dec->still_start - dec->codestream_pos));
        if (!status) {
          return JXL_API_ERROR("decoding dc failed");
        }
        dec->got_dc_image = true;

        return JXL_DEC_DC_IMAGE;
      }
    }

    // Decode to pixels, only if required for the events the user wants.
    if (!dec->got_full_image && (dec->events_wanted & JXL_DEC_FULL_IMAGE)) {
      if (dec->still_end - dec->codestream_pos > size)
        return JXL_DEC_NEED_MORE_INPUT;
      // If we got here, we know for sure there are enough bytes in the input
      // for all frames, this was checked during the TOC parsing
      // TODO(lode): allow to customize dparams through API settings, and share
      // these params for DC and preview too
      jxl::DecompressParams dparams;
      jxl::Span<const uint8_t> compressed(
          in + (dec->still_start - dec->codestream_pos),
          size - (dec->still_start - dec->codestream_pos));
      auto reader = GetBitReader(compressed);
      dec->ib.reset(new jxl::ImageBundle(&dec->metadata.m));

      if (!dec->passes_state) {
        dec->passes_state.reset(new jxl::PassesDecoderState());
      }
      const jxl::FrameHeader& frame_header =
          dec->passes_state->shared->frame_header;

      bool done = false;
      while (!done) {
        // Skip frames that are not displayed.
        do {
          if (!DecodeFrame(dparams, dec->passes_state.get(),
                           dec->thread_pool.get(), reader.get(),
                           /*aux_out=*/nullptr, dec->ib.get(), dec->metadata,
                           /*constraints=*/nullptr)) {
            return JXL_API_ERROR("decoding frame failed");
          }
          if (frame_header.is_last) {
            dec->last_frame_reached = true;
            done = true;
          }
          if (dec->frame_header->animation_frame.duration > 0) {
            done = true;
          }
        } while (frame_header.frame_type != FrameType::kRegularFrame);
        dec->dec_pixels += dec->ib->xsize() * dec->ib->ysize();
      }
      dec->got_full_image = true;
    }

    if (dec->last_frame_reached) {
      // No more reason to keep the passes state in memory
      dec->passes_state.reset(nullptr);
    }

    bool return_full_image = false;

    if (dec->events_wanted & JXL_DEC_FULL_IMAGE) {
      if (!dec->image_out_buffer_set) {
        dec->need_image_out_buffer = true;
        return JXL_DEC_NEED_IMAGE_OUT_BUFFER;
      }
      dec->events_wanted &= ~JXL_DEC_FULL_IMAGE;
      return_full_image = true;
    }

    if (!dec->last_frame_reached) {
      dec->got_toc = false;
      dec->got_dc_image = false;
      dec->got_full_image = false;
      dec->still_start = dec->frame_start;

      dec->events_wanted =
          dec->orig_events_wanted &
          (JXL_DEC_FULL_IMAGE | JXL_DEC_DC_IMAGE | JXL_DEC_FRAME);
    }

    // Copy pixels to output buffer if desired. If no output buffer was set,
    // we merely return the JXL_DEC_FULL_IMAGE status without outputting
    // pixels.
    if (return_full_image && dec->image_out_buffer_set) {
      JxlDecoderStatus status =
          ConvertImage(dec, *dec->ib, dec->image_out_format,
                       dec->image_out_buffer, dec->image_out_size);
      if (status != JXL_DEC_SUCCESS) return status;
      dec->image_out_buffer_set = false;
    }

    // The pixels have been output or are not needed, do not keep them in
    // memory here.
    dec->ib.reset();

    if (return_full_image) {
      return JXL_DEC_FULL_IMAGE;
    }

    if (dec->last_frame_reached) break;
  }

  dec->stage = DecoderStage::kFinished;
  // Return success, this means there is nothing more to do.
  return JXL_DEC_SUCCESS;
}

}  // namespace
}  // namespace jxl

JxlDecoderStatus JxlDecoderSetInput(JxlDecoder* dec, const uint8_t* data,
                                    size_t size) {
  if (dec->next_in) return JXL_DEC_ERROR;

  dec->next_in = data;
  dec->avail_in = size;
  return JXL_DEC_SUCCESS;
}

size_t JxlDecoderReleaseInput(JxlDecoder* dec) {
  size_t result = dec->avail_in;
  dec->next_in = nullptr;
  dec->avail_in = 0;
  return result;
}

JxlDecoderStatus JxlDecoderProcessInput(JxlDecoder* dec) {
  const uint8_t** next_in = &dec->next_in;
  size_t* avail_in = &dec->avail_in;
  if (dec->stage == DecoderStage::kInited) {
    dec->stage = DecoderStage::kStarted;
  }
  if (dec->stage == DecoderStage::kError) {
    return JXL_API_ERROR(
        "Cannot keep using decoder after it encountered an error, use "
        "JxlDecoderReset to reset it");
  }
  if (dec->stage == DecoderStage::kFinished) {
    return JXL_API_ERROR(
        "Cannot keep using decoder after it finished, use JxlDecoderReset to "
        "reset it");
  }

  if (!dec->got_signature) {
    JxlSignature sig = JxlSignatureCheck(*next_in, *avail_in);
    if (sig == JXL_SIG_INVALID) return JXL_API_ERROR("invalid signature");
    if (sig == JXL_SIG_NOT_ENOUGH_BYTES) return JXL_DEC_NEED_MORE_INPUT;

    dec->got_signature = true;

    if (sig == JXL_SIG_CONTAINER) {
      dec->have_container = 1;
    }
  }

  if (dec->have_container) {
    /*
    Process bytes as follows:
    *) find the box(es) containing the codestream
    *) support codestream split over multiple partial boxes
    *) avoid copying bytes to the codestream vector if the decoding will be
     one-shot, when the user already provided everything contiguously in
     memory
    *) copy to codestream vector, and update next_in so user can delete the data
    on their side, once we know it's not oneshot. This relieves the user from
    continuing to store the data.
    *) also copy to codestream if one-shot but the codestream is split across
    multiple boxes: this copying can be avoided in the future if the C++
    decoder is updated for streaming, but for now it requires all consecutive
    data at once.
    */

    if (dec->first_codestream_seen && !dec->last_codestream_seen &&
        dec->codestream_end != 0 && dec->file_pos < dec->codestream_end &&
        dec->file_pos + *avail_in > dec->codestream_end) {
      // dec->file_pos in a codestream, not in surrounding box format bytes, but
      // the end of the current codestream part is in the current input, and
      // boxes that can contain a next part of the codestream could be present.
      // Therefore, store the known codestream part, and ensure processing of
      // boxes below will trigger.

      if (dec->codestream.empty()) {
        JXL_ABORT("impossible to get in this situation");
      } else {
        // Size of the codestream, excluding potential boxes that come after it.
        size_t csize = *avail_in;
        if (dec->codestream_end &&
            csize > dec->codestream_end - dec->file_pos) {
          csize = dec->codestream_end - dec->file_pos;
        }
        dec->codestream.insert(dec->codestream.end(), *next_in,
                               *next_in + csize);
        dec->file_pos += csize;
        *next_in += csize;
        *avail_in -= csize;
      }
    }

    // if dec->codestream_begin > 0 && dec->codestream_end == 0, then all
    // remaining bytes are codestream, no further box processing needed
    if (!dec->last_codestream_seen &&
        (dec->codestream_begin == 0 ||
         (dec->codestream_end != 0 && dec->file_pos >= dec->codestream_end))) {
      size_t pos = 0;
      // after this for loop, either we should be in a part of the data that is
      // codestream (not boxes), or have returned that we need more input.
      for (;;) {
        const uint8_t* in = *next_in;
        size_t size = *avail_in;
        if (size == 0) {
          // If the remaining size is 0, we are exactly after a full box. We
          // can't know for sure if this is the last box or not since more bytes
          // can follow, but do not return NEED_MORE_INPUT, instead break and
          // let the codestream-handling code determine if we need more.
          break;
        }
        if (OutOfBounds(pos, 8, size)) {
          dec->basic_info_size_hint =
              InitialBasicInfoSizeHint() + pos + 8 - dec->file_pos;
          return JXL_DEC_NEED_MORE_INPUT;
        }
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
        size_t header_size = pos - box_start;
        if (box_size > 0 && box_size < header_size) {
          return JXL_API_ERROR("invalid box size");
        }
        size_t min_contents_size =
            (box_size == 0)
                ? (size - pos)
                : std::min<size_t>(size - pos, box_size - pos + box_start);
        size_t contents_size =
            (box_size == 0) ? 0 : (box_size - pos + box_start);
        // TODO(lode): support the case where the header is split across
        // multiple codestream boxes
        if (strcmp(type, "jxlc") == 0 || strcmp(type, "jxlp") == 0) {
          bool last_codestream = (strcmp(type, "jxlc") == 0) || (box_size == 0);
          dec->first_codestream_seen = true;
          if (last_codestream) dec->last_codestream_seen = true;
          if (dec->codestream_begin != 0 && dec->codestream.empty()) {
            // We've already seen a codestream part, so it's a stream spanning
            // multiple boxes.
            // We have no choice but to copy contents to the codestream
            // vector to make it a contiguous stream for the C++ decoder.
            if (dec->codestream_begin < dec->file_pos) {
              return JXL_API_ERROR("earlier codestream box out of range");
            }
            size_t begin = dec->codestream_begin - dec->file_pos;
            size_t end = dec->codestream_end - dec->file_pos;
            dec->codestream.insert(dec->codestream.end(), *next_in + begin,
                                   *next_in + end);
          }
          dec->codestream_begin = dec->file_pos + pos;
          dec->codestream_end =
              (box_size == 0) ? 0 : (dec->codestream_begin + contents_size);
          // If already appending codestream, append what we have here too
          if (!dec->codestream.empty()) {
            size_t begin = pos;
            size_t end = std::min<size_t>(*avail_in, begin + min_contents_size);
            dec->codestream.insert(dec->codestream.end(), *next_in + begin,
                                   *next_in + end);
            pos += (end - begin);
            dec->file_pos += pos;
            *next_in += pos;
            *avail_in -= pos;
            pos = 0;
            if (*avail_in == 0) break;
          } else {
            // skip only the header, so next_in points to the start of this new
            // codestream part
            dec->file_pos += pos;
            *next_in += pos;
            *avail_in -= pos;
            pos = 0;
            // Update pos to be after the box contents with codestream
            if (min_contents_size == *avail_in) {
              break;  // the rest is codestream, this loop is done
            }
            pos += min_contents_size;
          }
        } else {
          if (box_size == 0) {
            // Final box with unknown size, but it's not a codestream box, so
            // nothing more to do.
            if (!dec->last_codestream_seen) {
              return JXL_API_ERROR("didn't find any codestream box");
            }
            break;
          }
          if (OutOfBounds(pos, contents_size, size)) {
            // Indicate how many more bytes needed starting from *next_in.
            dec->basic_info_size_hint = InitialBasicInfoSizeHint() + pos +
                                        contents_size - dec->file_pos;
            return JXL_DEC_NEED_MORE_INPUT;
          }
          pos += contents_size;
          if (!(dec->codestream.empty() && dec->first_codestream_seen)) {
            if (box_size == 0) break;  // last box, nothing to do anymore
            // Last box no longer needed, remove from input.
            dec->file_pos += pos;
            *next_in += pos;
            *avail_in -= pos;
            pos = 0;
          }
        }
      }
    }

    // Size of the codestream, excluding potential boxes that come after it.
    size_t csize = *avail_in;
    if (dec->codestream_end && csize > dec->codestream_end - dec->file_pos) {
      csize = dec->codestream_end - dec->file_pos;
    }

    if (!dec->codestream.empty()) {
      dec->codestream.insert(dec->codestream.end(), *next_in, *next_in + csize);
      dec->file_pos += csize;
      *next_in += csize;
      *avail_in -= csize;
    }

    JxlDecoderStatus result;
    if (dec->codestream.empty()) {
      // No data copied to codestream buffer yet, the user input contains the
      // full codestream.
      result = jxl::JxlDecoderProcessInternal(dec, *next_in, csize);

      // Copy the user's input bytes to the codestream once we are able to and
      // it is needed. Before we got the basic info, we're still parsing the box
      // format instead. If the result is not JXL_DEC_NEED_MORE_INPUT, then
      // there is no reason yet to copy since the user may have a full buffer
      // allowing one-shot. Once JXL_DEC_NEED_MORE_INPUT occured at least once,
      // start copying over the codestream bytes and allow user to free them
      // instead.
      if (dec->got_basic_info && result == JXL_DEC_NEED_MORE_INPUT) {
        dec->codestream.insert(dec->codestream.end(), *next_in,
                               *next_in + csize);
        dec->file_pos += csize;
        *next_in += csize;
        *avail_in -= csize;
      }

    } else {
      result = jxl::JxlDecoderProcessInternal(dec, dec->codestream.data(),
                                              dec->codestream.size());
    }

    return result;
  } else {
    if (!dec->codestream.empty()) {
      dec->codestream.insert(dec->codestream.end(), *next_in,
                             *next_in + *avail_in);
      dec->file_pos += *avail_in;
      *next_in += *avail_in;
      *avail_in = 0;
    }

    JxlDecoderStatus result;
    if (dec->codestream.empty()) {
      // No data copied to codestream buffer yet, the user input contains the
      // full codestream.
      result = jxl::JxlDecoderProcessInternal(dec, *next_in, *avail_in);
      // Copy the user's input bytes to the codestream once we are able to and
      // it is needed. Before we got the basic info, we're still parsing the box
      // format instead. If the result is not JXL_DEC_NEED_MORE_INPUT, then
      // there is no reason yet to copy since the user may have a full buffer
      // allowing one-shot. Once JXL_DEC_NEED_MORE_INPUT occured at least once,
      // start copying over the codestream bytes and allow user to free them
      // instead.
      if ((dec->got_basic_info && result == JXL_DEC_NEED_MORE_INPUT) ||
          !dec->codestream.empty()) {
        dec->codestream.insert(dec->codestream.end(), *next_in,
                               *next_in + *avail_in);
        dec->file_pos += *avail_in;
        *next_in += *avail_in;
        *avail_in = 0;
      }
    } else {
      result = jxl::JxlDecoderProcessInternal(dec, dec->codestream.data(),
                                              dec->codestream.size());
    }

    return result;
  }

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetBasicInfo(const JxlDecoder* dec,
                                        JxlBasicInfo* info) {
  if (!dec->got_basic_info) return JXL_DEC_NEED_MORE_INPUT;

  if (info) {
    const jxl::ImageMetadata& meta = dec->metadata.m;

    info->have_container = dec->have_container;
    info->xsize = dec->metadata.size.xsize();
    info->ysize = dec->metadata.size.ysize();
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
      info->preview.xsize = dec->metadata.m.preview_size.xsize();
      info->preview.ysize = dec->metadata.m.preview_size.ysize();
    }

    if (info->have_animation) {
      info->animation.tps_numerator = dec->metadata.m.animation.tps_numerator;
      info->animation.tps_denominator =
          dec->metadata.m.animation.tps_denominator;
      info->animation.num_loops = dec->metadata.m.animation.num_loops;
      info->animation.have_timecodes = dec->metadata.m.animation.have_timecodes;
    }
  }

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetExtraChannelInfo(const JxlDecoder* dec,
                                               size_t index,
                                               JxlExtraChannelInfo* info) {
  if (!dec->got_basic_info) return JXL_DEC_NEED_MORE_INPUT;

  const std::vector<jxl::ExtraChannelInfo>& channels =
      dec->metadata.m.extra_channel_info;

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
      dec->metadata.m.extra_channel_info;

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
  if (target == JXL_COLOR_PROFILE_TARGET_DATA && dec->metadata.m.xyb_encoded) {
    // The profile of the pixels matches dec->ib->c_current(). However,
    // c_current in the ImageBundle is not yet filled in correctly at this point
    // since the pixels have not been decoded yet.
    // Instead, output the profile that the API specifies it uses for this case:
    // linear sRGB for floating point output, and nonlinear sRGB for integer
    // output, grayscale or color depending on the image header.
    bool grayscale = dec->metadata.m.color_encoding.IsGray();
    if (!format) {
      return JXL_API_ERROR("Must provide pixel format for data color profile");
    }
    if (format->data_type == JXL_TYPE_FLOAT &&
        !dec->metadata.m.color_encoding.IsSRGB()) {
      *encoding = &jxl::ColorEncoding::LinearSRGB(grayscale);
    } else {
      *encoding = &jxl::ColorEncoding::SRGB(grayscale);
    }
  } else {
    *encoding = &dec->metadata.m.color_encoding;
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

    if (color_encoding->color_space == JXL_COLOR_SPACE_RGB ||
        color_encoding->color_space == JXL_COLOR_SPACE_UNKNOWN) {
      color_encoding->primaries =
          static_cast<JxlPrimaries>(jxl_color_encoding->primaries);
      jxl::PrimariesCIExy primaries = jxl_color_encoding->GetPrimaries();
      color_encoding->primaries_red_xy[0] = primaries.r.x;
      color_encoding->primaries_red_xy[1] = primaries.r.y;
      color_encoding->primaries_green_xy[0] = primaries.g.x;
      color_encoding->primaries_green_xy[1] = primaries.g.y;
      color_encoding->primaries_blue_xy[0] = primaries.b.x;
      color_encoding->primaries_blue_xy[1] = primaries.b.y;
    }

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
        dec->metadata.m.color_encoding.GetColorSpace();
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
         dec->metadata.transform_data.opsin_inverse_matrix.inverse_matrix,
         sizeof(matrix->opsin_inv_matrix));
  memcpy(matrix->opsin_biases,
         dec->metadata.transform_data.opsin_inverse_matrix.opsin_biases,
         sizeof(matrix->opsin_biases));
  memcpy(matrix->quant_biases,
         dec->metadata.transform_data.opsin_inverse_matrix.quant_biases,
         sizeof(matrix->quant_biases));

  return JXL_DEC_SUCCESS;
}

namespace {
// Returns the amount of bits needed for getting memory buffer size, and does
// all error checking required for size checking and format validity.
JxlDecoderStatus PrepareSizeCheck(const JxlDecoder* dec,
                                  const JxlPixelFormat* format, size_t* bits) {
  if (!dec->got_basic_info) {
    // Don't know image dimensions yet, cannot check for valid size.
    return JXL_DEC_NEED_MORE_INPUT;
  }
  if (format->num_channels > 4) {
    return JXL_API_ERROR("More than 4 channels not supported");
  }
  if (format->num_channels < 3 && !dec->metadata.m.color_encoding.IsGray()) {
    return JXL_API_ERROR("Grayscale output not possible for color image");
  }
  if (format->data_type == JXL_TYPE_BOOLEAN) {
    return JXL_API_ERROR("Boolean data type not yet supported");
  }
  if (format->data_type == JXL_TYPE_UINT32) {
    return JXL_API_ERROR("uint32 data type not yet supported");
  }

  *bits = BitsPerChannel(format->data_type);

  if (*bits == 0) {
    return JXL_API_ERROR("Invalid data type");
  }

  return JXL_DEC_SUCCESS;
}
}  // namespace

JXL_EXPORT JxlDecoderStatus JxlDecoderPreviewOutBufferSize(
    const JxlDecoder* dec, const JxlPixelFormat* format, size_t* size) {
  size_t bits;
  JxlDecoderStatus status = PrepareSizeCheck(dec, format, &bits);
  if (status != JXL_DEC_SUCCESS) return status;

  const auto& metadata = dec->metadata.m;
  size_t xsize = metadata.preview_size.xsize();
  size_t ysize = metadata.preview_size.ysize();

  size_t row_size =
      jxl::DivCeil(xsize * format->num_channels * bits, jxl::kBitsPerByte);
  if (format->align > 1) {
    row_size = jxl::DivCeil(row_size, format->align) * format->align;
  }
  *size = row_size * ysize;
  return JXL_DEC_SUCCESS;
}

JXL_EXPORT JxlDecoderStatus JxlDecoderSetPreviewOutBuffer(
    JxlDecoder* dec, const JxlPixelFormat* format, void* buffer, size_t size) {
  if (!dec->need_preview_out_buffer) {
    return JXL_API_ERROR("No preview out buffer needed at this time");
  }

  size_t min_size;
  // This also checks whether the format is valid and supported and basic info
  // is available.
  JxlDecoderStatus status =
      JxlDecoderPreviewOutBufferSize(dec, format, &min_size);
  if (status != JXL_DEC_SUCCESS) return status;

  if (size < min_size) return JXL_DEC_ERROR;

  dec->need_preview_out_buffer = false;
  dec->preview_out_buffer_set = true;
  dec->preview_out_buffer = buffer;
  dec->preview_out_size = size;
  dec->preview_out_format = *format;

  return JXL_DEC_SUCCESS;
}

JXL_EXPORT JxlDecoderStatus JxlDecoderDCOutBufferSize(
    const JxlDecoder* dec, const JxlPixelFormat* format, size_t* size) {
  size_t bits;
  JxlDecoderStatus status = PrepareSizeCheck(dec, format, &bits);
  if (status != JXL_DEC_SUCCESS) return status;

  size_t xsize = jxl::DivCeil(dec->metadata.size.xsize(), jxl::kBlockDim);
  size_t ysize = jxl::DivCeil(dec->metadata.size.ysize(), jxl::kBlockDim);

  size_t row_size =
      jxl::DivCeil(xsize * format->num_channels * bits, jxl::kBitsPerByte);
  if (format->align > 1) {
    row_size = jxl::DivCeil(row_size, format->align) * format->align;
  }
  *size = row_size * ysize;
  return JXL_DEC_SUCCESS;
}

JXL_EXPORT JxlDecoderStatus JxlDecoderSetDCOutBuffer(
    JxlDecoder* dec, const JxlPixelFormat* format, void* buffer, size_t size) {
  if (!dec->need_dc_out_buffer) {
    return JXL_API_ERROR("No dc out buffer needed at this time");
  }
  size_t min_size;
  // This also checks whether the format is valid and supported and basic info
  // is available.
  JxlDecoderStatus status = JxlDecoderDCOutBufferSize(dec, format, &min_size);
  if (status != JXL_DEC_SUCCESS) return status;

  if (size < min_size) return JXL_DEC_ERROR;

  dec->need_dc_out_buffer = false;
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

  size_t row_size =
      jxl::DivCeil(dec->metadata.size.xsize() * format->num_channels * bits,
                   jxl::kBitsPerByte);
  if (format->align > 1) {
    row_size = jxl::DivCeil(row_size, format->align) * format->align;
  }
  *size = row_size * dec->metadata.size.ysize();

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderSetImageOutBuffer(JxlDecoder* dec,
                                             const JxlPixelFormat* format,
                                             void* buffer, size_t size) {
  if (!dec->need_image_out_buffer) {
    return JXL_API_ERROR("No image out buffer needed at this time");
  }
  size_t min_size;
  // This also checks whether the format is valid and supported and basic info
  // is available.
  JxlDecoderStatus status =
      JxlDecoderImageOutBufferSize(dec, format, &min_size);
  if (status != JXL_DEC_SUCCESS) return status;

  if (size < min_size) return JXL_DEC_ERROR;

  dec->need_image_out_buffer = false;
  dec->image_out_buffer_set = true;
  dec->image_out_buffer = buffer;
  dec->image_out_size = size;
  dec->image_out_format = *format;

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetFrameHeader(const JxlDecoder* dec,
                                          JxlFrameHeader* header) {
  if (!dec->frame_header || !dec->got_toc) {
    return JXL_API_ERROR("no frame header available");
  }
  const auto& metadata = dec->metadata.m;
  if (metadata.have_animation) {
    header->duration = dec->frame_header->animation_frame.duration;
    if (metadata.animation.have_timecodes) {
      header->timecode = dec->frame_header->animation_frame.timecode;
    }
  }
  header->name_length = dec->frame_header->name.size();

  return JXL_DEC_SUCCESS;
}

JxlDecoderStatus JxlDecoderGetFrameName(const JxlDecoder* dec, char* name,
                                        size_t size) {
  if (!dec->frame_header || !dec->got_toc) {
    return JXL_API_ERROR("no frame header available");
  }
  if (size < dec->frame_header->name.size() + 1) {
    return JXL_API_ERROR("too small frame name output buffer");
  }
  memcpy(name, dec->frame_header->name.c_str(),
         dec->frame_header->name.size() + 1);

  return JXL_DEC_SUCCESS;
}
