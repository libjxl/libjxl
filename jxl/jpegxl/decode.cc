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

#include "jxl/base/span.h"
#include "jxl/base/status.h"
#include "jxl/brunsli.h"
#include "jxl/headers.h"
#include "jxl/memory_manager_internal.h"

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
};

JpegxlDecoder* JpegxlDecoderCreate(const JpegxlMemoryManager* memory_manager) {
  JpegxlMemoryManager local_memory_manager;
  if (!jxl::MemoryManagerInit(&local_memory_manager, memory_manager))
    return nullptr;

  JpegxlDecoder* dec = static_cast<JpegxlDecoder*>(
      jxl::MemoryManagerAlloc(&local_memory_manager, sizeof(JpegxlDecoder)));
  if (!dec) return nullptr;
  dec->memory_manager = local_memory_manager;

  return dec;
}

void JpegxlDecoderDestroy(JpegxlDecoder* dec) {
  if (dec) {
    jxl::MemoryManagerFree(&dec->memory_manager, dec);
  }
}
