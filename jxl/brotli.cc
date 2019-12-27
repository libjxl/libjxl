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

#include "jxl/brotli.h"

#include <string.h>  // memcpy

#include <memory>

#include "brotli/decode.h"
#include "brotli/encode.h"
#include "jxl/base/status.h"

namespace jxl {

Status BrotliCompress(int quality, const uint8_t* in, const size_t in_size,
                      PaddedBytes* JXL_RESTRICT out) {
  std::unique_ptr<BrotliEncoderState, decltype(BrotliEncoderDestroyInstance)*>
      enc(BrotliEncoderCreateInstance(nullptr, nullptr, nullptr),
          BrotliEncoderDestroyInstance);
  if (!enc) return JXL_FAILURE("BrotliEncoderCreateInstance failed");

  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_QUALITY, quality);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_LGWIN, 24);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_LGBLOCK, 0);
  BrotliEncoderSetParameter(enc.get(), BROTLI_PARAM_MODE, BROTLI_MODE_FONT);

  const size_t kBufferSize = 128 * 1024;
  PaddedBytes temp_buffer(kBufferSize);

  size_t avail_in = in_size;
  const uint8_t* next_in = in;

  size_t total_out = 0;

  while (true) {
    size_t out_size;
    size_t avail_out = kBufferSize;
    uint8_t* next_out = temp_buffer.data();
    if (!BrotliEncoderCompressStream(enc.get(), BROTLI_OPERATION_FINISH,
                                     &avail_in, &next_in, &avail_out, &next_out,
                                     &total_out)) {
      return JXL_FAILURE("Brotli compression failed");
    }
    out_size = next_out - temp_buffer.data();
    out->resize(out->size() + out_size);
    memcpy(out->data() + out->size() - out_size, temp_buffer.data(), out_size);
    if (BrotliEncoderIsFinished(enc.get())) break;
  }

  return true;
}

Status BrotliCompress(int quality, const uint8_t* in, const size_t in_size,
                      uint8_t* JXL_RESTRICT out,
                      size_t* JXL_RESTRICT total_out_size) {
  PaddedBytes temp_buffer;
  if (!BrotliCompress(quality, in, in_size, &temp_buffer)) {
    return false;
  }
  *total_out_size = temp_buffer.size();
  memcpy(out, temp_buffer.data(), temp_buffer.size());
  return true;
}

Status BrotliCompress(int quality, const PaddedBytes& in,
                      PaddedBytes* JXL_RESTRICT out) {
  return BrotliCompress(quality, in.data(), in.size(), out);
}

Status BrotliDecompress(const Span<const uint8_t> in, size_t max_output_size,
                        size_t* JXL_RESTRICT bytes_read,
                        PaddedBytes* JXL_RESTRICT out) {
  std::unique_ptr<BrotliDecoderState, decltype(BrotliDecoderDestroyInstance)*>
      s(BrotliDecoderCreateInstance(nullptr, nullptr, nullptr),
        BrotliDecoderDestroyInstance);
  if (!s) return JXL_FAILURE("BrotliDecoderCreateInstance failed");

  const size_t kBufferSize = 128 * 1024;
  PaddedBytes temp_buffer(kBufferSize);

  size_t avail_in = in.size();
  if (in.size() == 0) return false;
  const uint8_t* next_in = in.data();
  BrotliDecoderResult code;

  while (true) {
    size_t out_size;
    size_t avail_out = kBufferSize;
    uint8_t* next_out = temp_buffer.data();
    code = BrotliDecoderDecompressStream(s.get(), &avail_in, &next_in,
                                         &avail_out, &next_out, nullptr);
    out_size = next_out - temp_buffer.data();
    out->resize(out->size() + out_size);
    if (out->size() > max_output_size)
      return JXL_FAILURE("Brotli output too large");
    memcpy(out->data() + out->size() - out_size, temp_buffer.data(), out_size);
    if (code != BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) break;
  }
  if (code != BROTLI_DECODER_RESULT_SUCCESS)
    return JXL_FAILURE("Brotli decompression failed");
  *bytes_read += (in.size() - avail_in);
  return true;
}

}  // namespace jxl
