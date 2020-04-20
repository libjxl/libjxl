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

#include <brunsli/brunsli_decode.h>
#include <jxl/base/span.h>
#include <jxl/brunsli.h>
#include <jxl/color_management.h>
#include <jxl/common.h>
#include <jxl/external_image.h>
#include <jxl/image_ops.h>

#include <cstdio>
#include <cstring>

using namespace jxl;

extern "C" {

ExternalImage* decompress(const uint8_t* data, size_t size) {
  ThreadPool* pool = nullptr;
  Span<const uint8_t> compressed(data, size);
  std::unique_ptr<ExternalImage> result;

  CodecInOut io;
  BrunsliDecoderOptions options;
  BrunsliDecoderMeta metadata;

  io.enc_size = compressed.size();

  brunsli::JPEGData jpg;
  brunsli::BrunsliStatus status =
      brunsli::BrunsliDecodeJpeg(compressed.data(), compressed.size(), &jpg);
  if (status != brunsli::BRUNSLI_OK) {
    printf("Failed to parse Brunsli input.");
    return nullptr;
  }

  if (!BrunsliToPixels(jpg, &io, options, &metadata, pool)) {
    printf("Failed to decompress.\n");
    return nullptr;
  }

  if (!metadata.hdr_orig_colorspace.empty()) {
    printf("Original colorspace: %s\n", metadata.hdr_orig_colorspace.c_str());
    // Hopefully, that is something like Chrome / rec2020.
    metadata.hdr_orig_colorspace = "RGB_D65_202_Rel_Lin";
    printf("Output colorspace: %s\n", metadata.hdr_orig_colorspace.c_str());

    ColorEncoding c;
    if (!ParseDescription(metadata.hdr_orig_colorspace, &c)) {
      printf("Failed to parse color profile description.\n");
      return nullptr;
    }
    if (!c.CreateICC()) {
      printf("Failed to create color profile.\n");
      return nullptr;
    }
    if (!io.Main().TransformTo(c, pool)) {
      printf("Failed to transform colorspace.\n");
      return nullptr;
    }
    io.metadata.color_encoding = c;
  }

  const ImageBundle& ib = io.Main();
  const ColorEncoding& c_desired = io.metadata.color_encoding;
  const bool has_alpha = true;
  const bool alpha_is_premultiplied = false;
  ImageU alpha(ib.color().xsize(), ib.color().ysize());
  const size_t alpha_bits = 8;
  size_t bits_per_sample = 32;
  const bool big_endian = false;
  CodecIntervals* temp_intervals = nullptr;
  Rect rect = Rect(ib);

  result = make_unique<ExternalImage>(
      pool, ib.color(), rect, ib.c_current(), c_desired, has_alpha,
      alpha_is_premultiplied, &alpha, alpha_bits, bits_per_sample, big_endian,
      temp_intervals);

  if (!result->IsHealthy()) {
    printf("ExternalImage is unhealthy.\n");
    return nullptr;
  }

  size_t w = result->xsize();
  size_t h = result->ysize();
  float* pixels = const_cast<float*>(reinterpret_cast<const float*>(result->Bytes().data()));
  for (size_t i = 0; i < w * h; ++i) {
    pixels[i * 4 + 3] = 1.0f;
  }

  return result.release();
}

void freeImage(ExternalImage* img) {
  delete img;
}

int getImageWidth(ExternalImage* img) {
  return img->xsize();
}

int getImageHeight(ExternalImage* img) {
  return img->ysize();
}

const void* getImagePixels(ExternalImage* img) {
  return img->Bytes().data();
}

}  // extern "C"

/*
// Chrome must be launched with "--enable-blink-features=CanvasColorManagement" option.

function showImage(bytes, amp) {
  if (!amp) amp = 1.0 / 256;
  console.log("Encoded size: " + bytes.length);
  var buf = Module._malloc(bytes.length);
  Module.HEAPU8.set(bytes, buf);
  var img = Module._decompress(buf, bytes.length);
  Module._free(buf);
  if (!img) return;
  var w = Module._getImageWidth(img);
  var h = Module._getImageHeight(img);
  var pixelsPtr = Module._getImagePixels(img);
  var pixels = new Float32Array(Module.HEAPF32.subarray(pixelsPtr >> 2, (pixelsPtr >> 2) + w * h * 4));
  for (var i = 0; i < w * h * 4; ++i) if ((i & 3) != 3) pixels[i] *= amp;
  var canvas = document.getElementById("canvas");
  canvas.width = w;
  canvas.height = h;
  var ctx = canvas.getContext("2d", {"colorSpace": "rec2020", "pixelFormat": "float16"});
  // Is there a way to create F16 ImageData?
  var imageData = ctx.getImageData(0, 0, w, h);
  imageData.dataUnion.set(pixels);
  ctx.putImageData(imageData, 0, 0);
  Module._freeImage(img);
}

function loadAndShow(path, amp) {
  var xhr = new XMLHttpRequest();
  xhr.open("GET", path, true);
  xhr.responseType = "arraybuffer";
  xhr.onload = (e) => { showImage(new Uint8Array(xhr.response), amp); }
  xhr.send(null);
}

*/
