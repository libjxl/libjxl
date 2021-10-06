// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_DECODE_BROB_BOX_H_
#define LIB_JXL_DECODE_BROB_BOX_H_

#include <brotli/decode.h>
#include <stdint.h>
#include <stdlib.h>

#include <memory>
#include <vector>

#include "jxl/decode.h"

namespace jxl {

class JxlBrobBoxDecoder {
 public:
  JxlBrobBoxDecoder();
  ~JxlBrobBoxDecoder();

  void StartBox();

  // Decode brob box bytes with brotli, advancing next_in and next_out the same
  // way brotli does, and returning the brotli status codes converted to JXL
  // status codes.
  JxlDecoderStatus Process(const uint8_t** next_in, size_t* avail_in,
                           uint8_t** next_out, size_t* avail_out);

 private:
  BrotliDecoderState* brotli_dec;
};

}  // namespace jxl

#endif  // LIB_JXL_DECODE_BROB_BOX_H_
