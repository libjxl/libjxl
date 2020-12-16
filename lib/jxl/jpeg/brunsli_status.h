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

#ifndef LIB_JXL_JPEG_BRUNSLI_STATUS_H_
#define LIB_JXL_JPEG_BRUNSLI_STATUS_H_

namespace jxl {
namespace jpeg {

typedef enum {
  BRUNSLI_OK = 0,

  // Used if the input is not representable in the compressed brunsli format,
  // either because it is not a valid JPEG file or if some other limitation
  // is exceeded (e.g. absolute value of coefficients or number of Huffman
  // codes).
  BRUNSLI_NON_REPRESENTABLE,

  BRUNSLI_MEMORY_ERROR,
  BRUNSLI_INVALID_PARAM,

  BRUNSLI_COMPRESSION_ERROR,

  BRUNSLI_INVALID_BRN,
  BRUNSLI_DECOMPRESSION_ERROR,

  BRUNSLI_NOT_ENOUGH_DATA,
} BrunsliStatus;

}  // namespace jpeg
}  // namespace jxl

#endif  // LIB_JXL_JPEG_BRUNSLI_STATUS_H_
