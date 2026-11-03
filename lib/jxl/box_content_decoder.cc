// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/box_content_decoder.h"

#include <brotli/decode.h>
#include <jxl/decode.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "lib/jxl/base/sanitizers.h"
#include "lib/jxl/base/span.h"

// All accesses to the box contents and to the output buffer in this file go
// through the bounds-carrying Span; keep it free of raw pointer arithmetic.
#if defined(__clang__) && defined(__has_warning)
#if __has_warning("-Wunsafe-buffer-usage")
#pragma clang diagnostic error "-Wunsafe-buffer-usage"
#endif
#endif

namespace jxl {

JxlBoxContentDecoder::JxlBoxContentDecoder() = default;

JxlBoxContentDecoder::~JxlBoxContentDecoder() {
  if (brotli_dec) {
    BrotliDecoderDestroyInstance(brotli_dec);
  }
}

void JxlBoxContentDecoder::StartBox(bool brob_decode, bool box_until_eof,
                                    size_t contents_size) {
  if (brotli_dec) {
    BrotliDecoderDestroyInstance(brotli_dec);
    brotli_dec = nullptr;
  }
  header_done_ = false;
  brob_decode_ = brob_decode;
  box_until_eof_ = box_until_eof;
  remaining_ = box_until_eof ? 0 : contents_size;
  pos_ = 0;
}

JxlDecoderStatus JxlBoxContentDecoder::Process(Span<const uint8_t> in,
                                               size_t box_pos,
                                               Span<uint8_t> out,
                                               size_t* produced) {
  *produced = 0;
  // The caller provides `box_pos` as the current position (in bytes) within
  // the box contents corresponding to the start of `in`. Our internal `pos_`
  // tracks how many bytes of box contents have been consumed/processed so far.
  // Skip the bytes of `in` that were already consumed. Inconsistent positions
  // are reported as an error here rather than left to remove_prefix(), which
  // aborts in debug builds.
  if (box_pos > pos_) return JXL_DEC_ERROR;
  const size_t offset = pos_ - box_pos;
  if (offset > in.size()) return JXL_DEC_ERROR;
  if (!in.remove_prefix(offset)) return JXL_DEC_ERROR;

  if (brob_decode_) {
    if (!header_done_) {
      if (in.size() < 4) return JXL_DEC_NEED_MORE_INPUT;
      if (!box_until_eof_) {
        if (remaining_ < 4) return JXL_DEC_ERROR;
        remaining_ -= 4;
      }
      if (!in.remove_prefix(4)) return JXL_DEC_ERROR;
      pos_ += 4;
      header_done_ = true;
    }

    if (!brotli_dec) {
      brotli_dec = BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
    }

    size_t avail_in = in.size();
    if (!box_until_eof_) avail_in = std::min<size_t>(avail_in, remaining_);
    const size_t avail_in_before = avail_in;
    const uint8_t* next_in = in.data();
    size_t avail_out = out.size();
    uint8_t* next_out = out.data();
    msan::MemoryIsInitialized(next_in, avail_in);
    BrotliDecoderResult res = BrotliDecoderDecompressStream(
        brotli_dec, &avail_in, &next_in, &avail_out, &next_out, nullptr);
    // Brotli never consumes or produces more than it was given room for, so
    // these can be derived from the remaining sizes.
    size_t consumed = avail_in_before - avail_in;
    *produced = out.size() - avail_out;
    if (res == BROTLI_DECODER_RESULT_ERROR) {
      return JXL_DEC_ERROR;
    }
    msan::UnpoisonMemory(out.data(), *produced);
    pos_ += consumed;
    if (!box_until_eof_) {
      remaining_ -= consumed;
      if (res == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT && remaining_ == 0) {
        return JXL_DEC_ERROR;
      }
    }
    if (res == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT) {
      return JXL_DEC_NEED_MORE_INPUT;
    }
    if (res == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT) {
      return JXL_DEC_BOX_NEED_MORE_OUTPUT;
    }
    if (res == BROTLI_DECODER_RESULT_SUCCESS) {
      if (!box_until_eof_ && remaining_ != 0) return JXL_DEC_ERROR;
      return JXL_DEC_BOX_COMPLETE;
    }
    // unknown Brotli result
    return JXL_DEC_ERROR;
  } else {
    // remaining box bytes as seen from dec->file_pos
    size_t can_read = in.size();
    if (!box_until_eof_) can_read = std::min<size_t>(can_read, remaining_);
    size_t to_write = std::min<size_t>(can_read, out.size());
    std::copy_n(in.begin(), to_write, out.begin());

    *produced = to_write;
    if (!box_until_eof_) remaining_ -= to_write;
    pos_ += to_write;

    if (to_write < can_read) return JXL_DEC_BOX_NEED_MORE_OUTPUT;

    if (!box_until_eof_ && remaining_ > 0) return JXL_DEC_NEED_MORE_INPUT;

    return JXL_DEC_BOX_COMPLETE;
  }
}
}  // namespace jxl