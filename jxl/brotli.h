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

#ifndef JXL_BROTLI_H_
#define JXL_BROTLI_H_

// Convenience functions for Brotli compression/decompression.

#include <stddef.h>
#include <stdint.h>

#include "jxl/base/compiler_specific.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/span.h"
#include "jxl/base/status.h"

namespace jxl {

// Appends to out.
Status BrotliCompress(int quality, const PaddedBytes& in,
                      PaddedBytes* JXL_RESTRICT out);

// Appends to out + *total_out_size.
Status BrotliCompress(int quality, const uint8_t* in, const size_t in_size,
                      uint8_t* JXL_RESTRICT out,
                      size_t* JXL_RESTRICT total_out_size);

// Appends to out and ADDS to "bytes_read", which must be pre-initialized.
Status BrotliDecompress(const Span<const uint8_t> in, size_t max_output_size,
                        size_t* JXL_RESTRICT bytes_read,
                        PaddedBytes* JXL_RESTRICT out);

}  // namespace jxl

#endif  // JXL_BROTLI_H_
