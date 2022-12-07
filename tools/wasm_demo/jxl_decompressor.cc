// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/wasm_demo/jxl_decompressor.h"

#include <cstring>
#include <memory>

#include "jxl/thread_parallel_runner_cxx.h"
#include "lib/extras/dec/jxl.h"
#include "tools/wasm_demo/no_png.h"

extern "C" {

namespace {

struct DecompressorOutputPrivate {
  // Due to "Standard Layout" rules it is guaranteed that address of the entity
  // and its first non-static member are the same.
  DecompressorOutput output;
};

}  // namespace
DecompressorOutput* jxlDecompress(const uint8_t* input, size_t input_size) {
  DecompressorOutputPrivate* self = new DecompressorOutputPrivate();

  if (!self) {
    return nullptr;
  }

  auto report_error = [&](uint32_t code, const char* text) {
    fprintf(stderr, "%s\n", text);
    delete self;
    return reinterpret_cast<DecompressorOutput*>(code);
  };

  auto thread_pool = JxlThreadParallelRunnerMake(nullptr, 4);
  void* runner = thread_pool.get();

  jxl::extras::JXLDecompressParams dparams;
  dparams.accepted_formats.push_back(
      {/* num_channels */ 3, JXL_TYPE_UINT8, JXL_LITTLE_ENDIAN, /* align */ 0});
  dparams.runner = JxlThreadParallelRunner;
  dparams.runner_opaque = runner;
  jxl::extras::PackedPixelFile ppf;

  if (!jxl::extras::DecodeImageJXL(input, input_size, dparams, nullptr, &ppf)) {
    return report_error(1, "failed to decode jxl");
  }

  // Just 1-st frame.
  const auto& image = ppf.frames[0].color;
  self->output.data = WrapPixelsToPng(
      image.xsize, image.ysize, /* bit_depth */ 8,
      /* has_alpha */ false, reinterpret_cast<const uint8_t*>(image.pixels()),
      &self->output.size);
  if (!self->output.data) {
    return report_error(2, "failed to encode png");
  }

  // jxl::extras::EncodedImage encoded_image;
  // if (!encoder->Encode(ppf, &encoded_image)) {
  //   return report_error(2, "failed to encode png");
  // }
  //  self->bitstream.swap(encoded_image.bitstreams[0]);
  //  self->output.size = self->bitstream.size();
  //  self->output.data = self->bitstream.data();

  return &self->output;
}

void jxlCleanup(DecompressorOutput* output) {
  if (output == nullptr) return;
  DecompressorOutputPrivate* self =
      reinterpret_cast<DecompressorOutputPrivate*>(output);
  if (self->output.data) {
    free(self->output.data);
  }
  delete self;
}

}  // extern "C"
