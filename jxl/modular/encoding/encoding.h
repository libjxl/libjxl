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

#ifndef JXL_MODULAR_ENCODING_ENCODING_H_
#define JXL_MODULAR_ENCODING_ENCODING_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/span.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/image.h"
#include "jxl/modular/config.h"
#include "jxl/modular/image/image.h"
#include "jxl/modular/ma/compound.h"
#include "jxl/modular/memio.h"

namespace jxl {

struct modular_options {
  // decoding options
  bool identify;  // don't decode image data, just decode header

  // used in both encode and decode
  int nb_channels;   // if full_header==false, need to specify how many channels
                     // to expect
  int skipchannels;  // the first <skipchannels> channels will not be
                     // encoded/decoded
  size_t max_chan_size;  // stop encoding/decoding when reaching a (non-meta)
                         // channel that has a dimension bigger than this

  // encoding options (some of which are needed during decoding too)
  int entropy_coder;  // 0 = MABEGABRAC, 1 = MABrotli, 2 = MARANS

  // MA options
  float nb_repeats;    // number of iterations to do to learn a MA tree (does
                       // not have to be an integer; if zero there is no MA
                       // context model)
  int max_properties;  // maximum number of (previous channel) properties to use
                       // in the MA trees
  float ctx_threshold;  // number of bits to be saved to justify adding
                        // another node to the MA tree (lower value = bigger
                        // context model)

  // Brotli options
  int brotli_effort;  // 0..11

  std::vector<int> predictor;  // predictor to use for each channel. last one
                               // gets repeated if needed

  int nb_wp_modes;

  // deprecated
  bool debug;      // produce debug images, including (for MABEGABRAC only) a
                   // compression heatmap
  Image *heatmap;  // produced if debug==true
};
void set_default_modular_options(struct modular_options &o);

#ifdef HAS_ENCODER
void modular_prepare_encode(Image &image, modular_options &options);

bool modular_generic_compress(Image &image, PaddedBytes *bytes,
                              modular_options *opts = nullptr, int loss = 1,
                              bool try_transforms = true);

// undo_transforms == N > 0: undo all transforms except the first N
//                           (e.g. to represent YCbCr420 losslessly)
// undo_transforms == 0: undo all transforms
// undo_transforms == -1: undo all transforms but don't clamp to range
// undo_transforms == -2: don't undo any transform
bool modular_generic_decompress(const Span<const uint8_t> bytes, size_t *pos,
                                Image &image, modular_options &options,
                                size_t bytes_to_load = 0,
                                int undo_transforms = -1);

// encode a rect from an arbitrary plane; dimensions etc are implicit
template <typename T>
bool modular_rect_compress_1(const Plane<T> &img, const Rect &rect,
                             PaddedBytes *bytes, modular_options *opts,
                             int loss = 1);
extern template bool modular_rect_compress_1<uint8_t>(const ImageB &img,
                                                      const Rect &rect,
                                                      PaddedBytes *bytes,
                                                      modular_options *opts,
                                                      int loss = 1);
extern template bool modular_rect_compress_1<uint16_t>(const ImageU &img,
                                                       const Rect &rect,
                                                       PaddedBytes *bytes,
                                                       modular_options *opts,
                                                       int loss = 1);
extern template bool modular_rect_compress_1<int16_t>(const ImageS &img,
                                                      const Rect &rect,
                                                      PaddedBytes *bytes,
                                                      modular_options *opts,
                                                      int loss = 1);

// these two planes are supposed to be single-row with the same width
// (used specifically to encode the ac_strategy + quant_field)
// width is not implicit
template <typename T>
bool modular_compress_2(const Plane<T> &img1, const Plane<T> &img2,
                        PaddedBytes *bytes, modular_options *opts);

template <typename T>
bool modular_rect_compress_2(const Plane<T> &img1, const Plane<T> &img2,
                             const Rect &rect, PaddedBytes *bytes,
                             modular_options *opts, pixel_type offset1 = 0,
                             pixel_type offset2 = 0);
extern template bool modular_rect_compress_2<uint8_t>(
    const ImageB &img1, const ImageB &img2, const Rect &rect,
    PaddedBytes *bytes, modular_options *opts, pixel_type offset1 = 0,
    pixel_type offset2 = 0);

// encode a rect from an arbitrary Image3; dimensions etc are implicit
template <typename T>
bool modular_rect_compress_3(const Image3<T> &img, const Rect &rect,
                             PaddedBytes *bytes, modular_options *opts,
                             int loss = 1);
extern template bool modular_rect_compress_3<int16_t>(const Image3S &img,
                                                      const Rect &rect,
                                                      PaddedBytes *bytes,
                                                      modular_options *opts,
                                                      int loss = 1);

extern template bool modular_rect_compress_3<int32_t>(const Image3I &img,
                                                      const Rect &rect,
                                                      PaddedBytes *bytes,
                                                      modular_options *opts,
                                                      int loss = 1);

#endif

template <typename T>
bool modular_rect_decompress_1(const Span<const uint8_t> bytes, size_t *pos,
                               const Plane<T> *JXL_RESTRICT result,
                               const Rect &rect);
extern template bool modular_rect_decompress_1<uint8_t>(
    const Span<const uint8_t> bytes, size_t *pos,
    const ImageB *JXL_RESTRICT result, const Rect &rect);
extern template bool modular_rect_decompress_1<uint16_t>(
    const Span<const uint8_t> bytes, size_t *pos,
    const ImageU *JXL_RESTRICT result, const Rect &rect);
extern template bool modular_rect_decompress_1<int16_t>(
    const Span<const uint8_t> bytes, size_t *pos,
    const ImageS *JXL_RESTRICT result, const Rect &rect);

template <typename T>
bool modular_rect_decompress_2(const Span<const uint8_t> bytes, size_t *pos,
                               const Plane<T> *JXL_RESTRICT result1,
                               const Plane<T> *JXL_RESTRICT result2,
                               const Rect &rect, pixel_type offset1 = 0,
                               pixel_type offset2 = 0);
extern template bool modular_rect_decompress_2<uint8_t>(
    const Span<const uint8_t> bytes, size_t *pos,
    const ImageB *JXL_RESTRICT result1, const ImageB *JXL_RESTRICT result2,
    const Rect &rect, pixel_type offset1 = 0, pixel_type offset2 = 0);

template <typename T>
bool modular_rect_decompress_3(const Span<const uint8_t> bytes, size_t *pos,
                               const Image3<T> *JXL_RESTRICT result,
                               const Rect &rect);
extern template bool modular_rect_decompress_3<int16_t>(
    const Span<const uint8_t> bytes, size_t *pos,
    const Image3S *JXL_RESTRICT result, const Rect &rect);
extern template bool modular_rect_decompress_3<int32_t>(
    const Span<const uint8_t> bytes, size_t *pos,
    const Image3I *JXL_RESTRICT result, const Rect &rect);

template <typename IO>
bool modular_decode(IO &io, Image &image, modular_options &options,
                    size_t bytes_to_load = 0);

}  // namespace jxl

#endif  // JXL_MODULAR_ENCODING_ENCODING_H_
