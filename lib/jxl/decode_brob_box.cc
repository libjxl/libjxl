// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/decode_brob_box.h"

#include "lib/jxl/sanitizers.h"

namespace jxl {

JxlBrobBoxDecoder::JxlBrobBoxDecoder() {}

JxlBrobBoxDecoder::~JxlBrobBoxDecoder() {
  if (brotli_dec) {
    BrotliDecoderDestroyInstance(brotli_dec);
  }
}

void JxlBrobBoxDecoder::StartBox() {
  if (brotli_dec) {
    BrotliDecoderDestroyInstance(brotli_dec);
  }
  brotli_dec = BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
}

JxlDecoderStatus JxlBrobBoxDecoder::Process(const uint8_t** next_in,
                                            size_t* avail_in,
                                            uint8_t** next_out,
                                            size_t* avail_out) {
  uint8_t* next_out_before = *next_out;
  size_t avail_out_before = *avail_out;
  msan::MemoryIsInitialized(*next_in, *avail_in);
  BrotliDecoderResult res = BrotliDecoderDecompressStream(
      brotli_dec, avail_in, next_in, avail_out, next_out, nullptr);
  if (res == BROTLI_DECODER_RESULT_ERROR) {
    return JXL_DEC_ERROR;
  }
  msan::UnpoisonMemory(next_out_before, avail_out_before - *avail_out);
  if (res == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT) {
    return JXL_DEC_NEED_MORE_INPUT;
  }
  if (res == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) {
    return JXL_DEC_BOX_NEED_MORE_OUTPUT;
  }
  if (res == BROTLI_DECODER_RESULT_SUCCESS) {
    return JXL_DEC_SUCCESS;
  }
  // unknown result
  return JXL_DEC_ERROR;

  return JXL_DEC_NEED_MORE_INPUT;
}

}  // namespace jxl
