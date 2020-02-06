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

// Tools for reading from / writing to ISOBMFF format for JPEG XL.

#ifndef TOOLS_BOX_BOX_H_
#define TOOLS_BOX_BOX_H_

#include <string>

#include "jxl/base/padded_bytes.h"
#include "jxl/base/status.h"

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

struct JpegXlContainer {
  // Exif metadata, or null if not present in the container.
  // The exif data has the format of 'Exif block' as defined in
  // ISO/IEC23008-12:2017 Clause A.2.1
  // TODO(lode): That means it first has 4 bytes exif_tiff_header_offset,
  // followed by the payload (EXIF and TIFF) in the next bytes. Offer the offset
  // adn payload as separate fields in the API here instead?
  const uint8_t* exif = nullptr;  // Not owned
  size_t exif_size = 0;

  // TODO(lode): add support for "xml " box, which can be used for XMP. There
  // may be multiple.

  // JUMBF superbox data, or null if not present in the container.
  // The parsing of the nested boxes inside is not handled here.
  const uint8_t* jumb = nullptr;  // Not owned
  size_t jumb_size = 0;

  // TODO(lode): add frame index data

  // The main JPEG XL codestream, of which there must be 1 in the container.
  const uint8_t* codestream = nullptr;  // Not owned
  size_t codestream_size = 0;
};

jxl::Status DecodeJpegXlContainerOneShot(const uint8_t* data, size_t size,
                                         JpegXlContainer* container);

// TODO(lode): streaming C API
jxl::Status EncodeJpegXlContainerOneShot(const JpegXlContainer& container,
                                         jxl::PaddedBytes* out);

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_BOX_BOX_H_
