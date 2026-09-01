// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/pnm.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "lib/extras/dec/color_hints.h"
#include "lib/extras/packed_image.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace extras {
namespace {

Span<const uint8_t> MakeSpan(const char* str) {
  return Bytes(reinterpret_cast<const uint8_t*>(str), strlen(str));
}

// The two-byte "P<type>" signature was read before any range check, relying on
// a minimum input size enforced in a different translation unit
// (extras/dec/decode.cc). DecodeImagePNM does not enforce that itself, so a
// caller reaching it directly - as these tests do - read past the end of the
// buffer. The out-of-bounds byte then selected the header variant in the switch
// that follows. ParseHeader must enforce its own precondition.
//
// The buffers are heap-allocated at exactly the tested length so sanitizer
// builds observe any overread; a string literal would hide it inside the
// terminating NUL. Length 1 is checked first because it is the case a
// sanitizer reports precisely - for length 0 an empty vector may yield a null
// data() pointer, whose behaviour is not guaranteed across standard libraries.
TEST(CodecPNMTest, RejectsInputShorterThanSignature) {
  PackedPixelFile ppf;
  for (const size_t len : {static_cast<size_t>(1), static_cast<size_t>(0)}) {
    const std::vector<uint8_t> bytes(len, static_cast<uint8_t>('P'));
    EXPECT_FALSE(
        DecodeImagePNM(Bytes(bytes.data(), bytes.size()), ColorHints(), &ppf));
  }
}

// A complete signature with nothing after it must still be rejected, and must
// not read beyond the two bytes it consumed.
TEST(CodecPNMTest, RejectsSignatureOnly) {
  PackedPixelFile ppf;
  const std::string pnm = "P6";
  EXPECT_FALSE(DecodeImagePNM(MakeSpan(pnm.c_str()), ColorHints(), &ppf));
}

}  // namespace
}  // namespace extras
}  // namespace jxl
