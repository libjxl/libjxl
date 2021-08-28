// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Tools for reading from / writing to ISOBMFF format for JPEG XL.

#ifndef TOOLS_BOX_BOX_H_
#define TOOLS_BOX_BOX_H_

#include <brotli/decode.h>
#include <brotli/encode.h>

#include <string>
#include <vector>

#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/enc_file.h"

namespace jpegxl {
namespace tools {

// A top-level box in the box format.
struct Box {
  // The type of the box.
  // If "uuid", use extended_type instead
  char type[4];

  // The extended_type is only used when type == "uuid".
  // Extended types are not used in JXL. However, the box format itself
  // supports this so they are handled correctly.
  char extended_type[16];

  // Size of the data, excluding box header. The box ends, and next box
  // begins, at data + size. May not be used if data_size_given is false.
  uint64_t data_size;

  // If the size is not given, the datasize extends to the end of the file.
  // If this field is false, the size field may not be used.
  bool data_size_given;
};

// Parses the header of a BMFF box. Returns the result in a Box struct.
// Updates next_in and available_in to point at the data in the box, directly
// after the header.
// Sets the data_size if known, or must be handled by the caller and runs until
// the end of the container file if not known.
// NOTE: available_in should be at least 8 up to 32 bytes to parse the
// header without error.
jxl::Status ParseBoxHeader(const uint8_t** next_in, size_t* available_in,
                           Box* box);

// TODO(lode): streaming C API
jxl::Status AppendBoxHeader(const Box& box, jxl::PaddedBytes* out);

/* Known/defined types:
  "Exif"
     The exif data has the format of 'Exif block' as defined in
     ISO/IEC23008-12:2017 Clause A.2.1
     Note: it starts with a tiff header offset of 4 bytes (usually zeroes),
     and the actual Exif data (starting with the tiff header MM or II) is
     located at that offset.
     // TODO(lode): support the theoretical case of multiple exif boxes
  "xml "
     Contains XML data, typically used for XMP.
  "jumb"
     JUMBF superbox data.
     The parsing of the nested boxes inside is not handled here.

  Any of these boxes (and possibly others) can be stored in a compressed way
  in a "brob" box, which starts with 4 uncompressed bytes specifying the
  actual box type, followed by a Brotli stream.

  To facilitate transparent handling of boxes, the BrobBlob struct is used.
  It has pointers to both the uncompressed data and the compressed data.
  When decoding:
    - when reading, either udata* or cdata* will be set
    - in the uncompressed case, only udata* is used.
    - in the compressed case, cdata* points into the input stream, and
      when accessing the box with getBlob(), pbytes will store the
      decompressed data, and udata* will point to it.
  When encoding:
    - in the uncompressed case, only udata* is used.
    - in the compressed case, udata* points to the uncompressed input, and
      when adding the box with addBlob(), pbytes will store the compressed
      data, and cdata* will point to it.
    - when writing, a "brob" box will be written if cdata* is nonzero,
      otherwise an uncompressed box will be written.
*/
struct BrobBlob {
  uint8_t type[4];
  // uncompressed data (mutable: decompressed data can be added or removed)
  mutable const uint8_t* udata = nullptr;  // Not owned
  mutable size_t udata_size = 0;
  // compressed data
  const uint8_t* cdata = nullptr;  // Not owned
  size_t cdata_size = 0;
  // a buffer to store the (de)compressed data
  mutable jxl::PaddedBytes pbytes;  // Owned
};

// NOTE: after DecodeJpegXlContainerOneShot, the exif etc. pointers point to
// regions within the input data passed to that function.
struct JpegXlContainer {
  std::vector<BrobBlob> blobs;

  // Returns a pointer to a BrobBlob with uncompressed data (or nullptr if not
  // found). In case there are multiple of that type, return the index-th one.
  const BrobBlob* getBlob(const char* type, size_t index = 0) const {
    for (auto& b : blobs) {
      if (!memcmp(type, b.type, 4)) {
        if (index > 0) {
          index--;
          continue;
        }
        if (b.udata_size == 0) {
          BrotliDecoderState* brotli_dec =
              BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
          struct BrotliDecDeleter {
            BrotliDecoderState* brotli_dec;
            ~BrotliDecDeleter() { BrotliDecoderDestroyInstance(brotli_dec); }
          } brotli_dec_deleter{brotli_dec};
          size_t available_in = b.cdata_size;
          b.pbytes.resize(available_in);
          size_t available_out = b.pbytes.size();
          const uint8_t* in = b.cdata;
          uint8_t* out = b.pbytes.data();
          size_t tot_dec = 0;
          while (available_in > 0 || !BrotliDecoderIsFinished(brotli_dec)) {
            BrotliDecoderResult result = BrotliDecoderDecompressStream(
                brotli_dec, &available_in, &in, &available_out, &out, &tot_dec);
            if (result ==
                BrotliDecoderResult::BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) {
              b.pbytes.resize(b.pbytes.size() + 10240);
              out = b.pbytes.data() + tot_dec;
              available_out = b.pbytes.size() - tot_dec;
            } else if (result !=
                       BrotliDecoderResult::BROTLI_DECODER_RESULT_SUCCESS) {
              JXL_WARNING("Brotli decoding error: %s\n",
                          BrotliDecoderErrorString(
                              BrotliDecoderGetErrorCode(brotli_dec)));
              return nullptr;
            }
          }
          b.udata = b.pbytes.data();
          b.udata_size = tot_dec;
        }
        return &b;
      }
    }
    return nullptr;
  }

  void addBlob(const char* type, const uint8_t* data, size_t data_size,
               bool compress = true) {
    BrobBlob b;
    memcpy(b.type, type, 4);
    b.udata = data;
    b.udata_size = data_size;
    if (compress) {
      b.pbytes.resize(b.udata_size);
      b.cdata_size = b.udata_size;
      if (!BrotliEncoderCompress(BROTLI_DEFAULT_QUALITY, BROTLI_DEFAULT_WINDOW,
                                 BROTLI_DEFAULT_MODE, b.udata_size, b.udata,
                                 &b.cdata_size, b.pbytes.data())) {
        JXL_WARNING("Could not brotli compress brob box");
      }
      b.cdata = b.pbytes.data();
    }
    blobs.emplace_back(std::move(b));
  }
  // TODO(lode): add frame index data

  // JPEG reconstruction data, or null if not present in the container.
  const uint8_t* jpeg_reconstruction = nullptr;
  size_t jpeg_reconstruction_size = 0;

  // The main JPEG XL codestream, of which there must be 1 in the container.
  // TODO(lode): support split codestream: there may be multiple jxlp boxes.
  const uint8_t* codestream = nullptr;  // Not owned
  size_t codestream_size = 0;
};

// Returns whether `data` starts with a container header; definitely returns
// false if `size` is less than 12 bytes.
bool IsContainerHeader(const uint8_t* data, size_t size);

// NOTE: the input data must remain valid as long as `container` is used,
// because its exif etc. pointers point to that data.
jxl::Status DecodeJpegXlContainerOneShot(const uint8_t* data, size_t size,
                                         JpegXlContainer* container);

// TODO(lode): streaming C API
jxl::Status EncodeJpegXlContainerOneShot(const JpegXlContainer& container,
                                         jxl::PaddedBytes* out);

// TODO(veluca): this doesn't really belong here.
jxl::Status DecodeJpegXlToJpeg(jxl::DecompressParams params,
                               const JpegXlContainer& container,
                               jxl::CodecInOut* io,
                               jxl::ThreadPool* pool = nullptr);

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_BOX_BOX_H_
