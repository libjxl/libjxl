// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/box/box.h"

#include "lib/jxl/base/byte_order.h"  // for GetMaximumBrunsliEncodedSize
#include "lib/jxl/jpeg/dec_jpeg_data.h"
#include "lib/jxl/jpeg/jpeg_data.h"

namespace jpegxl {
namespace tools {

namespace {
// Checks if a + b > size, taking possible integer overflow into account.
bool OutOfBounds(size_t a, size_t b, size_t size) {
  size_t pos = a + b;
  if (pos > size) return true;
  if (pos < a) return true;  // overflow happened
  return false;
}
}  // namespace

// Parses the header of a BMFF box. Returns the result in a Box struct.
// Sets the position to the end of the box header after parsing. The data size
// is output if known, or must be handled by the caller and runs until the end
// of the container file if not known.
jxl::Status ParseBoxHeader(const uint8_t** next_in, size_t* available_in,
                           Box* box) {
  size_t pos = 0;
  size_t size = *available_in;
  const uint8_t* in = *next_in;

  if (OutOfBounds(pos, 8, size)) return JXL_FAILURE("out of bounds");

  const size_t initial_pos = pos;

  // Total box_size including this header itself.
  uint64_t box_size = LoadBE32(in + pos);
  memcpy(box->type, in + pos + 4, 4);

  pos += 8;

  if (box_size == 1) {
    // If the size is 1, it indicates extended size read from 64-bit integer.
    if (OutOfBounds(pos, 8, size)) return JXL_FAILURE("out of bounds");
    box_size = LoadBE64(in + pos);
    pos += 8;
  }

  if (!memcmp("uuid", box->type, 4)) {
    if (OutOfBounds(pos, 16, size)) return JXL_FAILURE("out of bounds");
    memcpy(box->extended_type, in + pos, 16);
    pos += 16;
  }

  // This is the end of the box header, the box data begins here. Handle
  // the data size now.
  const size_t data_pos = pos;
  const size_t header_size = data_pos - initial_pos;

  if (box_size != 0) {
    if (box_size < header_size) {
      return JXL_FAILURE("invalid box size");
    }
    box->data_size_given = true;
    box->data_size = box_size - header_size;
  } else {
    // The size extends to the end of the file. We don't necessarily know the
    // end of the file here, since the input size may be only part of the full
    // container file. Indicate the size is not given, the caller must handle
    // this.
    box->data_size_given = false;
    box->data_size = 0;
  }

  // The remaining bytes are the data. If the box is a full box, the first
  // bytes of the data have a certain structure but this is to be handled by
  // the caller for the appropriate box type.
  *next_in += pos;
  *available_in -= pos;

  return true;
}

jxl::Status AppendBoxHeader(const Box& box, jxl::PaddedBytes* out) {
  bool use_extended = !memcmp("uuid", box.type, 4);

  uint64_t box_size = 0;
  bool large_size = false;
  if (box.data_size_given) {
    box_size = box.data_size + 8 + (use_extended ? 16 : 0);
    if (box_size >= 0x100000000ull) {
      large_size = true;
    }
  }

  out->resize(out->size() + 4);
  StoreBE32(large_size ? 1 : box_size, &out->back() - 4 + 1);

  out->resize(out->size() + 4);
  memcpy(&out->back() - 4 + 1, box.type, 4);

  if (large_size) {
    out->resize(out->size() + 8);
    StoreBE64(box_size, &out->back() - 8 + 1);
  }

  if (use_extended) {
    out->resize(out->size() + 16);
    memcpy(&out->back() - 16 + 1, box.extended_type, 16);
  }

  return true;
}

bool IsContainerHeader(const uint8_t* data, size_t size) {
  const uint8_t box_header[] = {0,   0,   0,   0xc, 'J',  'X',
                                'L', ' ', 0xd, 0xa, 0x87, 0xa};
  if (size < sizeof(box_header)) return false;
  return memcmp(box_header, data, sizeof(box_header)) == 0;
}

jxl::Status DecodeJpegXlContainerOneShot(const uint8_t* data, size_t size,
                                         JpegXlContainer* container) {
  const uint8_t* in = data;
  size_t available_in = size;

  container->blobs.clear();
  container->codestream = nullptr;
  container->codestream_size = 0;
  container->jpeg_reconstruction = nullptr;
  container->jpeg_reconstruction_size = 0;

  size_t box_index = 0;

  while (available_in != 0) {
    Box box;
    if (!ParseBoxHeader(&in, &available_in, &box)) {
      return JXL_FAILURE("Invalid box header");
    }

    size_t data_size = box.data_size_given ? box.data_size : available_in;

    if (box.data_size > available_in) {
      return JXL_FAILURE("Unexpected end of file");
    }

    if (box_index == 0) {
      // TODO(lode): leave out magic signature box?
      // Must be magic signature box.
      if (memcmp("JXL ", box.type, 4) != 0) {
        return JXL_FAILURE("Invalid magic signature");
      }
      if (box.data_size != 4) return JXL_FAILURE("Invalid magic signature");
      if (in[0] != 0xd || in[1] != 0xa || in[2] != 0x87 || in[3] != 0xa) {
        return JXL_FAILURE("Invalid magic signature");
      }
    } else if (box_index == 1) {
      // Must be ftyp box.
      if (memcmp("ftyp", box.type, 4) != 0) {
        return JXL_FAILURE("Invalid ftyp");
      }
      if (box.data_size != 12) return JXL_FAILURE("Invalid ftyp");
      const char* expected = "jxl \0\0\0\0jxl ";
      if (memcmp(expected, in, 12) != 0) return JXL_FAILURE("Invalid ftyp");
    } else if (!memcmp("jxli", box.type, 4)) {
      // TODO(lode): parse JXL frame index box
      if (container->codestream) {
        return JXL_FAILURE("frame index must come before codestream");
      }
    } else if (!memcmp("jxlc", box.type, 4)) {
      container->codestream = in;
      container->codestream_size = data_size;
    } else if (!memcmp("Exif", box.type, 4) || !memcmp("xml ", box.type, 4) ||
               !memcmp("jumb", box.type, 4)) {
      BrobBlob blob;
      memcpy(blob.type, box.type, 4);
      blob.udata = in;
      blob.udata_size = data_size;
      container->blobs.push_back(blob);
    } else if (!memcmp("brob", box.type, 4)) {
      if (data_size < 4) return JXL_FAILURE("Invalid brob");
      BrobBlob blob;
      memcpy(blob.type, in, 4);
      blob.cdata = in + 4;
      blob.cdata_size = data_size - 4;
      container->blobs.emplace_back(blob);
    } else if (!memcmp("jbrd", box.type, 4)) {
      container->jpeg_reconstruction = in;
      container->jpeg_reconstruction_size = data_size;
    } else {
      // Do nothing: box not recognized here but may be recognizable by
      // other software.
    }

    in += data_size;
    available_in -= data_size;
    box_index++;
  }

  return true;
}

static jxl::Status AppendBoxAndData(const char type[4], const uint8_t* data,
                                    size_t data_size, jxl::PaddedBytes* out,
                                    const uint8_t* brobtype = nullptr) {
  Box box;
  memcpy(box.type, type, 4);
  box.data_size = data_size + (brobtype != nullptr ? 4 : 0);
  box.data_size_given = true;
  JXL_RETURN_IF_ERROR(AppendBoxHeader(box, out));
  if (brobtype != nullptr) out->append(brobtype, brobtype + 4);

  out->append(data, data + data_size);
  return true;
}

jxl::Status EncodeJpegXlContainerOneShot(const JpegXlContainer& container,
                                         jxl::PaddedBytes* out) {
  const unsigned char header[] = {0,   0,   0,    0xc, 'J', 'X', 'L', ' ',
                                  0xd, 0xa, 0x87, 0xa, 0,   0,   0,   0x14,
                                  'f', 't', 'y',  'p', 'j', 'x', 'l', ' ',
                                  0,   0,   0,    0,   'j', 'x', 'l', ' '};
  size_t header_size = sizeof(header);
  out->append(header, header + header_size);

  for (auto& blob : container.blobs) {
    if (blob.cdata_size > 0) {
      JXL_RETURN_IF_ERROR(AppendBoxAndData("brob", blob.cdata, blob.cdata_size,
                                           out, blob.type));
    } else if (blob.udata_size > 0) {
      char type[4];
      memcpy(type, blob.type, 4);
      JXL_RETURN_IF_ERROR(
          AppendBoxAndData(type, blob.udata, blob.udata_size, out));
    } else
      return JXL_FAILURE("Empty blob");
  }
  if (container.jpeg_reconstruction) {
    JXL_RETURN_IF_ERROR(AppendBoxAndData("jbrd", container.jpeg_reconstruction,
                                         container.jpeg_reconstruction_size,
                                         out));
  }

  if (container.codestream) {
    JXL_RETURN_IF_ERROR(AppendBoxAndData("jxlc", container.codestream,
                                         container.codestream_size, out));
  } else {
    return JXL_FAILURE("must have primary image frame");
  }
  return true;
}

// TODO(veluca): the format defined here encode some things multiple times. Fix
// that.
jxl::Status DecodeJpegXlToJpeg(jxl::DecompressParams params,
                               const JpegXlContainer& container,
                               jxl::CodecInOut* io, jxl::ThreadPool* pool) {
  params.keep_dct = true;
  if (container.jpeg_reconstruction == nullptr) {
    return JXL_FAILURE(
        "Cannot decode to JPEG without a JPEG reconstruction box");
  }

  io->Main().jpeg_data = jxl::make_unique<jxl::jpeg::JPEGData>();

  JXL_RETURN_IF_ERROR(DecodeJPEGData(
      jxl::Span<const uint8_t>(container.jpeg_reconstruction,
                               container.jpeg_reconstruction_size),
      io->Main().jpeg_data.get()));

  auto& jpeg_data = io->Main().jpeg_data;
  bool have_exif = false, have_xmp = false;
  for (size_t i = 0; i < jpeg_data->app_data.size(); i++) {
    if (jpeg_data->app_marker_type[i] == jxl::jpeg::AppMarkerType::kExif) {
      if (have_exif)
        return JXL_FAILURE("Unexpected: more than one Exif box required?");
      const BrobBlob* exif = container.getBlob("Exif");
      if (exif == nullptr) return JXL_FAILURE("Unexpected: No Exif box found");
      if (jpeg_data->app_data[i].size() != exif->udata_size - 4 + 9) {
        return JXL_FAILURE(
            "Exif box size (%zu) does not match JPEG reconstruction data Exif "
            "size (%zu)",
            exif->udata_size - 4, jpeg_data->app_data[i].size() - 9);
      }
      have_exif = true;
      memcpy(&jpeg_data->app_data[i][3 + 6], exif->udata + 4,
             exif->udata_size - 4);
    }
    if (jpeg_data->app_marker_type[i] == jxl::jpeg::AppMarkerType::kXMP) {
      if (have_xmp)
        return JXL_FAILURE("Unexpected: more than one XMP box required?");
      const BrobBlob* xmp = container.getBlob("xml ");
      if (xmp == nullptr) return JXL_FAILURE("Unexpected: No XMP box found");
      if (jpeg_data->app_data[i].size() != xmp->udata_size + 32) {
        return JXL_FAILURE(
            "XMP box size does not match JPEG reconstruction data");
      }
      have_xmp = true;
      memcpy(&jpeg_data->app_data[i][3 + 29], xmp->udata, xmp->udata_size);
    }
  }

  JXL_RETURN_IF_ERROR(DecodeFile(
      params,
      jxl::Span<const uint8_t>(container.codestream, container.codestream_size),
      io, pool));
  return true;
}

}  // namespace tools
}  // namespace jpegxl
