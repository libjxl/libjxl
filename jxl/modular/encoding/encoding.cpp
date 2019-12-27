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

#include "jxl/modular/encoding/encoding.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdlib.h>
#include <random>

#include "jxl/base/status.h"
#include "jxl/brotli.h"
#include "jxl/dec_ans.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/entropy_coder.h"
#include "jxl/modular/encoding/context_predict.h"
#include "jxl/modular/encoding/weighted_predict.h"
#include "jxl/modular/ma/compound.h"
#include "jxl/modular/ma/rac.h"
#include "jxl/modular/ma/rac_enc.h"
#include "jxl/modular/ma/util.h"
#include "jxl/modular/memio.h"
#include "jxl/modular/transform/transform.h"
#include "jxl/modular/util.h"
#include "jxl/toc.h"

namespace jxl {

void set_default_modular_options(struct modular_options &o) {
  o.identify = false;

  o.entropy_coder = 0;  // MABEGABRAC
  o.nb_channels = 1;
  o.skipchannels = 0;
  o.max_chan_size = 0xFFFFFF;
  o.nb_repeats =
      0.5;  // learn MA tree by looking at 50% of the rows, in random order
  o.ctx_threshold = 16;  // 2 byte improvement needed to add another context
  o.max_properties = 0;  // no previous channels
  o.brotli_effort = 11;
  o.nb_wp_modes = 1;
  o.debug = false;
  o.heatmap = nullptr;
};

template <typename IO>
void write_big_endian_varint(IO &io, size_t number, bool done = true) {
  if (number < 128) {
    if (done)
      io.fputc(number);
    else
      io.fputc(number + 128);
  } else {
    size_t lsb = (number & 127);
    number >>= 7;
    write_big_endian_varint(io, number, false);
    write_big_endian_varint(io, lsb, done);
  }
}

template <typename IO>
int read_big_endian_varint(IO &io) {
  int result = 0;
  int bytes_read = 0;
  while (bytes_read++ < 10) {
    int number = io.get_c();
    if (number < 0) break;
    if (number < 128) return result + number;
    number -= 128;
    result += number;
    result <<= 7;
  }
  if (!io.isEOF()) JXL_FAILURE("Invalid number encountered!");
  return -1;
}

bool check_bit_depth(pixel_type minv, pixel_type maxv) {
  uint32_t maxav = abs(maxv - minv);
  if (ilog2(maxav) + 1 > MAX_BIT_DEPTH) {
    return JXL_FAILURE(
        "Error: compiled for a maximum bit depth of %i, while %i "
        "bits are needed to encode this channel (range=%i..%i)",
        MAX_BIT_DEPTH, ilog2(maxav) + 1, minv, maxv);
  }
  return true;
}

#define ENTROPY_CODER_NAME(e) \
  ((e == 0 ? "BEGABRAC" : (e == 1 ? "Brotli" : "ANS")))

#ifdef HAS_ENCODER
Predictor find_best(const Channel &channel, const pixel_type *JXL_RESTRICT p,
                    intptr_t onerow, size_t y, Predictor prev_predictor) {
  uint64_t best = 0;
  uint64_t prev = -1;
  int best_predictor = 0;
  for (uint32_t i = 0; i < 6; i++) {
    uint64_t sum_of_abs_residuals = 0;
    for (size_t x = 0; x < channel.w; x++) {
      pixel_type guess = predict(channel, p + x, onerow, x, y, (Predictor)i);
      sum_of_abs_residuals += abs(p[x] - guess);
    }
    if (i == 0 || sum_of_abs_residuals < best) {
      best = sum_of_abs_residuals;
      best_predictor = i;
    }
    if ((Predictor)i == prev_predictor) prev = sum_of_abs_residuals;
  }
  // only change predictor if residuals are 10% smaller
  if (prev < best * 1.1) return prev_predictor;
  return (Predictor)best_predictor;
}

template <typename IO, typename Rac, typename Coder, bool learn>
bool modular_encode_channels(IO &io, Tree &tree, modular_options &options,
                             Predictor predictor, int beginc, int endc,
                             const Image &image, size_t &header_pos,
                             int entropy_coder) {
  const Channel &channel = image.channel[beginc];
  if (channel.w == 0 || channel.h == 0) return true;
  // BEGABRAC with left predictor works best for palette data
  if (channel.hshift < 0) {
    entropy_coder = 0;
    predictor = Predictor::Left;
  }
  int predictability = 2048;  // 50%
  {
    if (predictor == Predictor::Zero && entropy_coder == 0 && learn) {
      uint64_t zeroes = 0;
      uint64_t pixels = channel.h * channel.w;
      for (size_t y = 0; y < channel.h; y++) {
        const pixel_type *JXL_RESTRICT p = channel.Row(y);
        for (size_t x = 0; x < channel.w; x++)
          if (p[x] == 0) zeroes++;
      }
      predictability = CLAMP(zeroes * 4096 / pixels, 1, 4095);
      JXL_DEBUG_V(6,
                  "Found %" PRIu64 " zeroes in %" PRIu64
                  " pixels (zero chance=%i/4096)",
                  zeroes, pixels, predictability);
    }
  }

  JXL_ASSERT(endc == beginc);

  pixel_type minv = channel.minval;
  pixel_type maxv = channel.maxval;
  if (minv == 0 && maxv == 0) {
    write_big_endian_varint(io, 0);
  } else {
    if (minv <= 0) {
      write_big_endian_varint(io, 2 - minv);
    } else {
      write_big_endian_varint(io, 1);
      write_big_endian_varint(io, minv - 1);
    }
    if (channel.w > 1 || channel.h > 1)
      write_big_endian_varint(io, maxv - minv);
  }
  if (entropy_coder == 0 && !check_bit_depth(minv, maxv)) return false;

  if (!learn) {
    JXL_DEBUG_V(6,
                "Encoding %zux%zu channel %i with range %i..%i (predictor %i), "
                "(shift=%i,%i, cshift=%i,%i)",
                channel.w, channel.h, beginc, minv, maxv, predictor,
                channel.hshift, channel.vshift, channel.hcshift,
                channel.vcshift);
  } else {
    JXL_DEBUG_V(7,
                "Learning %zux%zu channel %i with range %i..%i (predictor %i)",
                channel.w, channel.h, beginc, minv, maxv, predictor);
  }
  channel.setzero();

  header_pos = io.ftell();
  if (minv == maxv) return true;

  write_big_endian_varint(io, ((int)predictor * 3) + entropy_coder);

  if (predictor == Predictor::Weighted) {
    if (learn) return true;
    const Channel &channel = image.channel[beginc];
    if (channel.minval == channel.maxval) return true;
    PaddedBytes bytes;
    if (!wp_compress(image.channel[beginc], &bytes, options.nb_wp_modes))
      return false;
    io.append(bytes);
    return true;
  }

  Predictor subpredictor = predictor;
  if (entropy_coder == 1) {
    if (learn) return true;
    size_t bytesperpixel = 1;
    pixel_type maxdiff = channel.maxval - channel.minval;
    pixel_type maxval = maxdiff;
    bool sign2lsb = true;
    if (predictor == Predictor::Zero &&
        (channel.maxval <= 0 || channel.minval >= 0))
      sign2lsb = false;

    if (sign2lsb) {
      maxval = 2 * maxdiff;
    }

    if (maxval > 0xff) bytesperpixel++;
    if (maxval > 0xffff) bytesperpixel++;
    size_t pixels = channel.w * channel.h;
    PaddedBytes buffer(bytesperpixel * pixels);
    PaddedBytes pbuffer;
    const intptr_t onerow = channel.plane.PixelsPerRow();
    size_t pos = 0;
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      if (predictor == Predictor::Variable) {
        subpredictor = find_best(channel, p, onerow, y, subpredictor);
        pbuffer.push_back((uint8_t)subpredictor);
      }
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type guess = predict(channel, p + x, onerow, x, y, subpredictor);
        pixel_type v;
        if (sign2lsb) {
          v = PackSigned(p[x] - guess);
        } else
          v = p[x] - channel.minval;
        buffer[pos] = (v & 0xff);
        if (bytesperpixel > 1) buffer[pos + pixels] = ((v >> 8) & 0xff);
        if (bytesperpixel > 2) buffer[pos + 2 * pixels] = ((v >> 16) & 0xff);
        pos++;
      }
    }
    pos = 0;
    pbuffer.append(buffer);
    PaddedBytes cbuffer;
    JXL_RETURN_IF_ERROR(
        BrotliCompress(options.brotli_effort, pbuffer, &cbuffer));
    pos = cbuffer.size();
    JXL_DEBUG_V(4,
                "Encoded %zu Brotli bytes from %zu uncompressed "
                "bytes (%f bpp)",
                pos, pixels * bytesperpixel, pos * 8.0 / pixels);
    io.append(std::move(cbuffer));
    return true;
  }

  Ranges propRanges;
  init_properties(propRanges, image, beginc, endc, options);
  size_t treebegin = io.ftell();
  Rac rac(io);
  if (!learn && entropy_coder != 1) {
    // encode tree here
    MetaPropertySymbolCoder<ModularBitChanceTree, Rac> metacoder(rac,
                                                                 propRanges);
    metacoder.write_tree(tree, entropy_coder == 0);
  }
  Properties properties(propRanges.size());

  if (!learn && entropy_coder == 2) {
    rac.flush();
    BlobReader tmp(io.ptr() + treebegin, io.ftell() - treebegin);
    RacIn<BlobReader> drac(tmp);
    // decode tree we just encoded, so it has the same node ordering as what the
    // decoder gets. Doesn't matter for MABEGABRAC, but does matter for MARANS
    // and MABrotli
    MetaPropertySymbolCoder<ModularBitChanceTree, RacIn<BlobReader>>
        metadecoder(drac, propRanges);
    metadecoder.read_tree(tree, false);
    Coder coder(rac, propRanges, tree);
    JXL_DEBUG_V(3, "Encoding using a MA tree with %zu nodes (%d contexts)",
                tree.size(), coder.nb_contexts());
    bool sign2lsb = true;
    if (predictor == Predictor::Zero &&
        (channel.maxval <= 0 || channel.minval >= 0))
      sign2lsb = false;

    int nbctx = coder.nb_contexts();
    if (predictor == Predictor::Variable) nbctx++;
    BitWriter writer;
    std::vector<std::vector<Token>> tokens(1);

    const intptr_t onerow = channel.plane.PixelsPerRow();
    Channel references(properties.size() - NB_NONREF_PROPERTIES, channel.w, 0,
                       0);
    for (size_t y = 0; y < channel.h; y++) {
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      precompute_references(channel, y, image, beginc, options, references);
      if (predictor == Predictor::Variable) {
        subpredictor = find_best(channel, p, onerow, y, subpredictor);
        TokenizeHybridUint(nbctx - 1, (uint32_t)subpredictor, &tokens[0]);
      }
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type guess =
            predict_and_compute_properties_with_precomputed_reference(
                properties, channel, p + x, onerow, x, y, subpredictor, image,
                beginc, options, references);
        int ctx = coder.context_id(properties);
        if (sign2lsb) {
          TokenizeHybridUint(ctx, PackSigned(p[x] - guess), &tokens[0]);
        } else {
          TokenizeHybridUint(ctx, p[x] - channel.minval, &tokens[0]);
        }
      }
    }
    EntropyEncodingData codes;
    std::vector<uint8_t> context_map;
    size_t pixels = channel.h * channel.w;
    BuildAndEncodeHistograms(HistogramParams(), nbctx, tokens, &codes,
                             &context_map, &writer, 0, nullptr);
    WriteTokens(tokens[0], codes, context_map, &writer, 0, nullptr);
    writer.ZeroPadToByte();
    size_t pos = writer.BitsWritten() / 8;
    JXL_DEBUG_V(9, "   Encoded %zu ANS bytes for %zu pixels (%f bpp)", pos,
                pixels, pos * 8.0 / pixels);
    io.append(std::move(writer).TakeBytes());
  } else {  // entropy_coder == 0  (or learning and pretending it is)
    Coder coder(rac, propRanges, tree, predictability,
                CONTEXT_TREE_SPLIT_THRESHOLD * options.nb_repeats * 2 *
                    options.ctx_threshold * (entropy_coder > 0 ? 2 : 1));
    SimpleSymbolCoder<SimpleBitChance, Rac, 3> pcoder(rac);
    int rowslearned = 0;
    std::mt19937_64 gen(1);  // deterministic learning (also between threads)
    std::uniform_int_distribution<> random_row(0, channel.h - 1);
    Channel references(properties.size() - NB_NONREF_PROPERTIES, channel.w, 0,
                       0);
    const intptr_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      if (learn) {
        if (++rowslearned > options.nb_repeats * channel.h) break;
      }
      // try random rows, to avoid giving priority to the top of the image
      // (because then the y property cannot be learned)
      if (learn && options.nb_repeats < 2) y = random_row(gen);
      const pixel_type *JXL_RESTRICT p = channel.Row(y);
      pixel_type *JXL_RESTRICT hp;
      if (!learn && options.debug) hp = options.heatmap->channel[beginc].Row(y);
      precompute_references(channel, y, image, beginc, options, references);
      if (predictor == Predictor::Variable) {
        subpredictor = find_best(channel, p, onerow, y, subpredictor);
        pcoder.write_int(0, 5, (int)subpredictor);
      }
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type guess =
            predict_and_compute_properties_with_precomputed_reference(
                properties, channel, p + x, onerow, x, y, subpredictor, image,
                beginc, options, references);
        pixel_type diff = p[x] - guess;
        if (!learn && options.debug) {
          int estimate =
              coder.estimate_int(properties, minv - guess, maxv - guess, diff);
          hp[x] = estimate;
        }
        coder.write_int(properties, minv - guess, maxv - guess, diff);
      }
      if (learn && y == channel.h - 1) {
        y = 0;
      }  // set to zero in case random row was channel.h-1
    }

    // collect stats for chance initialization
    if (learn) {
      Channel references(properties.size() - NB_NONREF_PROPERTIES, channel.w, 0,
                         0);
      const intptr_t onerow = channel.plane.PixelsPerRow();
      for (size_t y = 0; y < channel.h; y++) {
        const pixel_type *JXL_RESTRICT p = channel.Row(y);
        precompute_references(channel, y, image, beginc, options, references);
        for (size_t x = 0; x < channel.w; x++) {
          pixel_type guess =
              predict_and_compute_properties_with_precomputed_reference(
                  properties, channel, p + x, onerow, x, y, predictor, image,
                  beginc, options, references);
          pixel_type diff = p[x] - guess;
          coder.stats_write_int(properties, minv - guess, maxv - guess, diff);
        }
      }
    }

    coder.simplify();
    rac.flush();
  }
  return true;
}
#endif

template <typename IO>
bool corrupt_or_truncated(IO &io, Channel &channel, size_t bytes_to_load) {
  if (io.isEOF() ||
      (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load)) {
    JXL_DEBUG_V(3, "Premature end-of-file detected.");
    channel.resize();
    return true;
  } else {
    JXL_DEBUG_V(3, "Corruption detected.");
    return false;
  }
}

template <typename IO, typename Coder>
HWY_ATTR bool modular_decode_channel(IO &io, modular_options &options,
                                     size_t &beginc, Image &image,
                                     size_t bytes_to_load) {
  Channel &channel = image.channel[beginc];

  // zero pixel channel? could happen
  if (channel.w == 0 || channel.h == 0) return true;
  size_t filepos = io.ftell();
  if (io.isEOF() ||
      (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load))
    return true;
  pixel_type minv = 2 - read_big_endian_varint(io);
  if (io.isEOF() ||
      (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load))
    return true;
  pixel_type maxv;
  if (minv == 2) {
    minv = 0;
    maxv = 0;
  } else {
    if (minv == 1) minv = 1 + read_big_endian_varint(io);

    if (io.isEOF() ||
        (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load))
      return true;
    if (channel.w == 1 && channel.h == 1) {
      maxv = minv;
    } else {
      maxv = minv + read_big_endian_varint(io);
    }
  }
  if (io.isEOF() ||
      (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load))
    return true;

  channel.minval = minv;
  channel.maxval = maxv;
  if (channel.minval == channel.maxval) {
    channel.setzero();
    channel.plane = Plane<pixel_type>();
    channel.resize();  // fill it with the constant value
    JXL_DEBUG_V(
        4, "[File position %zu] Decoding channel %zu: %zux%zu, constant %i",
        filepos, beginc, channel.w, channel.h, channel.minval);
  }

  if (io.isEOF() ||
      (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load))
    return corrupt_or_truncated(io, channel, bytes_to_load);

  if (minv == maxv) return true;

  int firstbyte = read_big_endian_varint(io);
  if (io.isEOF() ||
      (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load))
    return true;
  int entropy_coder = (firstbyte % 3);
  Predictor predictor = (Predictor)(firstbyte / 3);
  JXL_DEBUG_V(4,
              "[File position %zu] %s-decoding channel %zu: %zux%zu with range "
              "%i..%i, predictor %i",
              filepos, ENTROPY_CODER_NAME(entropy_coder), beginc, channel.w,
              channel.h, channel.minval, channel.maxval, predictor);
  JXL_DEBUG_V(6, " (shift=%i,%i, cshift=%i,%i)", channel.hshift, channel.vshift,
              channel.hcshift, channel.vcshift);

  if (entropy_coder == 0 && !check_bit_depth(channel.minval, channel.maxval))
    return false;

  channel.setzero();
  channel.resize(channel.w, channel.h);

  if (predictor == Predictor::Weighted) {
    Channel &channel = image.channel[beginc];
    channel.setzero();
    channel.resize(channel.w, channel.h);
    if (channel.minval == channel.maxval) return true;
    size_t iopos = io.ftell();
    const Span<const uint8_t> bytes(io.ptr(), io.size());
    if (!wp_decompress(bytes, &iopos, channel)) return false;
    io.fseek(iopos, SEEK_SET);
    return true;
  }
  Predictor subpredictor = predictor;
  // Brotli decode
  if (entropy_coder == 1) {
    size_t bytesperpixel = 1;
    pixel_type maxdiff = channel.maxval - channel.minval;
    pixel_type maxval = maxdiff;
    bool sign2lsb = true;
    if (predictor == Predictor::Zero &&
        (channel.maxval <= 0 || channel.minval >= 0))
      sign2lsb = false;

    if (sign2lsb) maxval = 2 * maxdiff;
    if (maxval > 0xff) bytesperpixel++;
    if (maxval > 0xffff) bytesperpixel++;
    size_t pixels = channel.w * channel.h;
    PaddedBytes obuffer;
    size_t pos = 0;
    size_t iopos = io.ftell();
    bool decodestatus = BrotliDecompress(
        Span<const uint8_t>(io.ptr() + iopos, io.size() - iopos),
        pixels * bytesperpixel +
            (predictor == Predictor::Variable ? channel.h : 0),
        &pos, &obuffer);
    io.fseek(iopos + pos, SEEK_SET);
    JXL_DEBUG_V(4, "   Decoded %zu bytes for %zu pixels", pos, pixels);
    pos = (predictor == Predictor::Variable ? channel.h : 0);
    if (!decodestatus) {
      if (io.isEOF() ||
          (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load)) {
        JXL_DEBUG_V(3, "Premature end-of-file at channel %zu.", beginc);
        return true;
      }
      return JXL_FAILURE("Problem during Brotli decode");
    }

    if (predictor == Predictor::Zero && bytesperpixel <= 2) {
      // special optimized case: no predictor
      JXL_DEBUG_V(8, "Fast track.");
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        if (sign2lsb) {
          if (bytesperpixel == 1)
            for (size_t x = 0; x < channel.w; x++) {
              r[x] = UnpackSigned(obuffer[pos++]);
            }
          else
            for (size_t x = 0; x < channel.w; x++) {
              pixel_type v = obuffer[pos];
              v += obuffer[pixels + pos] << 8;
              pos++;
              r[x] = UnpackSigned(v);
            }
        } else {
          if (bytesperpixel == 1)
            for (size_t x = 0; x < channel.w; x++) {
              pixel_type v = obuffer[pos++];
              v += channel.minval;
              r[x] = v;
            }
          else
            for (size_t x = 0; x < channel.w; x++) {
              pixel_type v = obuffer[pos];
              v += obuffer[pixels + pos] << 8;
              pos++;
              v += channel.minval;
              r[x] = v;
            }
        }
      }
    } else {
      const intptr_t onerow = channel.plane.PixelsPerRow();
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        if (predictor == Predictor::Variable)
          subpredictor = (Predictor)obuffer[y];
        if (bytesperpixel == 1)
          for (size_t x = 0; x < channel.w; x++) {
            pixel_type g = predict(channel, r + x, onerow, x, y, subpredictor);
            pixel_type v = obuffer[pos++];
            r[x] = UnpackSigned(v) + g;
          }
        else if (bytesperpixel == 2)
          for (size_t x = 0; x < channel.w; x++) {
            pixel_type g = predict(channel, r + x, onerow, x, y, subpredictor);
            pixel_type v = obuffer[pos];
            v += obuffer[pixels + pos] << 8;
            pos++;
            r[x] = UnpackSigned(v) + g;
          }
        else if (bytesperpixel == 3)
          for (size_t x = 0; x < channel.w; x++) {
            pixel_type g = predict(channel, r + x, onerow, x, y, subpredictor);
            pixel_type v = obuffer[pos];
            v += obuffer[pixels + pos] << 8;
            v += obuffer[2 * pixels + pos] << 16;
            pos++;
            r[x] = UnpackSigned(v) + g;
          }
      }
    }

    return true;
  }
  Ranges propRanges;
  init_properties(propRanges, image, beginc, beginc, options);

  int predictability = 2048;

  RacIn<IO> rac(io);

  // decode trees
  Tree tree;
  MetaPropertySymbolCoder<ModularBitChanceTree, RacIn<IO>> metacoder(
      rac, propRanges);
  if (!metacoder.read_tree(tree, entropy_coder == 0))
    return corrupt_or_truncated(io, image.channel[beginc], bytes_to_load);

  Coder coder(rac, propRanges, tree, predictability);
  JXL_DEBUG_V(3, "Decoded MA tree with %zu nodes (%d contexts)", tree.size(),
              coder.nb_contexts());
  Properties properties(propRanges.size());

  // MAANS decode
  if (entropy_coder == 2) {
    bool sign2lsb = true;
    if (predictor == Predictor::Zero &&
        (channel.maxval <= 0 || channel.minval >= 0))
      sign2lsb = false;

    int nbctx = coder.nb_contexts();
    if (predictor == Predictor::Variable) nbctx++;

    size_t iopos = io.ftell();
    BitReader br(Span<const uint8_t>(io.ptr() + iopos, io.size() - iopos));
    std::vector<uint8_t> context_map;
    ANSCode code;
    JXL_RETURN_IF_ERROR(
        DecodeHistograms(&br, nbctx, ANS_MAX_ALPHA_SIZE, &code, &context_map));
    ANSSymbolReader reader(&code, &br);
    if (tree.size() == 1 && predictor == Predictor::Zero) {
      // special optimized case: no meta-adaptation, no predictor, so no need
      // to compute properties
      JXL_DEBUG_V(8, "Fast track.");
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        if (sign2lsb) {
          for (size_t x = 0; x < channel.w; x++) {
            uint32_t v = ReadHybridUint(0, &br, &reader, context_map);
            r[x] = UnpackSigned(v);
          }
        } else {
          for (size_t x = 0; x < channel.w; x++) {
            pixel_type v = ReadHybridUint(0, &br, &reader, context_map);
            v += channel.minval;
            r[x] = v;
          }
        }
      }
    } else if (tree.size() == 1) {
      // special optimized case: no meta-adaptation, so no need to compute
      // properties
      JXL_DEBUG_V(8, "Quite fast track.");
      const intptr_t onerow = channel.plane.PixelsPerRow();
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT r = channel.Row(y);
        if (predictor == Predictor::Variable)
          subpredictor =
              (Predictor)ReadHybridUint(nbctx - 1, &br, &reader, context_map);
        for (size_t x = 0; x < channel.w; x++) {
          pixel_type g = predict(channel, r + x, onerow, x, y, subpredictor);
          uint32_t v = ReadHybridUint(0, &br, &reader, context_map);
          r[x] = UnpackSigned(v) + g;
        }
      }
    } else {
      JXL_DEBUG_V(8, "Slow track.");
      const intptr_t onerow = channel.plane.PixelsPerRow();
      Channel references(properties.size() - NB_NONREF_PROPERTIES, channel.w, 0,
                         0);
      for (size_t y = 0; y < channel.h; y++) {
        pixel_type *JXL_RESTRICT p = channel.Row(y);
        precompute_references(channel, y, image, beginc, options, references);
        if (predictor == Predictor::Variable)
          subpredictor =
              (Predictor)ReadHybridUint(nbctx - 1, &br, &reader, context_map);
        for (size_t x = 0; x < channel.w; x++) {
          pixel_type guess =
              predict_and_compute_properties_with_precomputed_reference(
                  properties, channel, p + x, onerow, x, y, subpredictor, image,
                  beginc, options, references);
          int ctx = coder.context_id(properties);
          uint32_t v = ReadHybridUint(ctx, &br, &reader, context_map);
          if (sign2lsb) {
            p[x] = UnpackSigned(v) + guess;
          } else {
            p[x] = channel.minval + v;
          }
        }
      }
    }
    if (!reader.CheckANSFinalState()) {
      return JXL_FAILURE("ANS decode final state failed");
    }
    JXL_RETURN_IF_ERROR(br.JumpToByteBoundary());
    io.fseek(iopos + br.TotalBitsConsumed() / 8, SEEK_SET);
    JXL_RETURN_IF_ERROR(br.Close());
    return true;
  }

  // MABEGABRAC decode

  SimpleSymbolCoder<SimpleBitChance, RacIn<IO>, 3> pcoder(rac);
  if (tree.size() == 1 && predictor == Predictor::Zero && channel.zero == 0) {
    // special optimized case: no meta-adaptation, no predictor, so no need to
    // compute properties
    JXL_DEBUG_V(8, "Fast track.");
    for (size_t y = 0; y < channel.h; y++) {
      if (io.isEOF() ||
          (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load)) {
        JXL_DEBUG_V(3, "Premature end-of-file at row %zu of channel %zu.", y,
                    beginc);
        break;
      }
      pixel_type *JXL_RESTRICT r = channel.Row(y);
      for (size_t x = 0; x < channel.w; x++) {
        r[x] = coder.read_int(properties, channel.minval, channel.maxval);
      }
    }
  } else if (tree.size() == 1) {
    // special optimized case: no meta-adaptation, no predictor, so no need to
    // compute properties
    JXL_DEBUG_V(8, "Quite fast track.");
    const intptr_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      if (io.isEOF() ||
          (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load)) {
        JXL_DEBUG_V(3, "Premature end-of-file at row %zu of channel %zu.", y,
                    beginc);
        break;
      }
      pixel_type *JXL_RESTRICT r = channel.Row(y);
      if (predictor == Predictor::Variable)
        subpredictor = (Predictor)pcoder.read_int(0, 5);
      for (size_t x = 0; x < channel.w; x++) {
        pixel_type g = predict(channel, r + x, onerow, x, y, subpredictor);
        r[x] = g + coder.read_int(properties, channel.minval - g,
                                  channel.maxval - g);
      }
    }
  } else {
    JXL_DEBUG_V(8, "Slow track.");

    Channel references(properties.size() - NB_NONREF_PROPERTIES, channel.w, 0,
                       0);
    const intptr_t onerow = channel.plane.PixelsPerRow();
    for (size_t y = 0; y < channel.h; y++) {
      if (io.isEOF() ||
          (bytes_to_load && static_cast<size_t>(io.ftell()) >= bytes_to_load)) {
        JXL_DEBUG_V(3, "Premature end-of-file at row %zu of channel %zu.", y,
                    beginc);
        break;
      }
      precompute_references(channel, y, image, beginc, options, references);
      pixel_type *JXL_RESTRICT p = channel.Row(y);
      if (predictor == Predictor::Variable)
        subpredictor = (Predictor)pcoder.read_int(0, 5);
      if (y <= 1 || subpredictor != Predictor::Zero || channel.zero != 0) {
        for (size_t x = 0; x < channel.w; x++) {
          pixel_type guess;
          guess = predict_and_compute_properties_with_precomputed_reference(
              properties, channel, p + x, onerow, x, y, subpredictor, image,
              beginc, options, references);
          pixel_type diff = coder.read_int(properties, channel.minval - guess,
                                           channel.maxval - guess);
          p[x] = diff + guess;
        }
      } else {
        size_t x = 0;
        pixel_type guess, diff;
        for (; x < channel.w && x < 2; x++) {
          guess = predict_and_compute_properties_with_precomputed_reference(
              properties, channel, p + x, onerow, x, y, subpredictor, image,
              beginc, options, references);
          diff = coder.read_int(properties, channel.minval - guess,
                                channel.maxval - guess);
          p[x] = diff + guess;
        }
        for (; x < channel.w - 1; x++) {
          guess =
              predict_and_compute_properties_with_precomputed_reference_no_edge_case(
                  properties, p + x, onerow, x, y, references);
          diff = coder.read_int(properties, channel.minval - guess,
                                channel.maxval - guess);
          p[x] = diff + guess;
        }
        for (; x < channel.w; x++) {
          guess = predict_and_compute_properties_with_precomputed_reference(
              properties, channel, p + x, onerow, x, y, subpredictor, image,
              beginc, options, references);
          diff = coder.read_int(properties, channel.minval - guess,
                                channel.maxval - guess);
          p[x] = diff + guess;
        }
      }
    }
  }

  return true;
}

#ifdef HAS_ENCODER
template <typename IO>
bool modular_encode(IO &realio, const Image &image, modular_options &options) {
  if (image.error) return false;
  size_t nb_channels = image.real_nb_channels;
  int bit_depth = 1, maxval = 1;
  while (maxval < image.maxval) {
    bit_depth++;
    maxval = maxval * 2 + 1;
  }
  JXL_DEBUG_V(2, "Encoding %zu-channel, %i-bit, %zux%zu image.", nb_channels,
              bit_depth, image.w, image.h);

  BlobWriter io;

  if (nb_channels < 1)
    return true;  // is there any use for a zero-channel image?

  int max_properties_nb_transforms = (options.max_properties << 2);

  // encode transforms
  int nb_transforms = image.transform.size();
  max_properties_nb_transforms += (nb_transforms < 3 ? nb_transforms : 3);
  write_big_endian_varint(io, max_properties_nb_transforms);
  if (nb_transforms >= 3) write_big_endian_varint(io, nb_transforms - 3);
  JXL_DEBUG_V(5, "Image data underwent %i transformations: ", nb_transforms);
  for (int i = 0; i < nb_transforms; i++) {
    int id = image.transform[i].ID;
    int nb_params = image.transform[i].parameters.size();
    write_big_endian_varint(io, (nb_params << 4) + id);
    for (int j = 0; j < nb_params; j++)
      write_big_endian_varint(io, image.transform[i].parameters[j]);
  }

  nb_channels = image.channel.size();

  std::vector<int> responsive_offsets(6, -1);  //   LQIP, 1/16 1/8 1/4 1/2 full

  // encode channel data
  for (size_t i = options.skipchannels; i < nb_channels; i++) {
    if (!image.channel[i].w || !image.channel[i].h)
      continue;  // skip empty channels
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    Predictor predictor = Predictor::Zero;
    if (options.predictor.size() > i) {
      predictor = (Predictor)options.predictor[i];
    } else {
      if (options.predictor.size() > 0) {
        predictor = (Predictor)options.predictor.back();
      } else {
        // if nothing at all is specified, use Gradient,
        // seems to be the best general-purpose predictor
        predictor = Predictor::Gradient;
      }
    }
    size_t j = i;
    Tree tree;
    size_t header_pos = 0;
    size_t before = io.ftell();
    Predictor learn_predictor = predictor;
    if (predictor == Predictor::Best) learn_predictor = Predictor::Gradient;
    DummyWriter dummyio;
    if (!modular_encode_channels<
            DummyWriter, RacDummy<DummyWriter>,
            PropertySymbolCoder<ModularBitChancePass1, RacDummy<DummyWriter>,
                                MAX_BIT_DEPTH>,
            true>(dummyio, tree, options, learn_predictor, i, j, image,
                  header_pos, options.entropy_coder))
      return false;
    if (!modular_encode_channels<
            BlobWriter, RacOut<BlobWriter>,
            FinalPropertySymbolCoder<ModularBitChancePass2, RacOut<BlobWriter>,
                                     MAX_BIT_DEPTH>,
            false>(io, tree, options, learn_predictor, i, j, image, header_pos,
                   options.entropy_coder))
      return false;
    size_t after = io.ftell();

    if (predictor == Predictor::Best) {
      BlobWriter try_io;
      size_t dummy_header_pos;
      if (!modular_encode_channels<
              BlobWriter, RacOut<BlobWriter>,
              FinalPropertySymbolCoder<ModularBitChancePass2,
                                       RacOut<BlobWriter>, MAX_BIT_DEPTH>,
              false>(try_io, tree, options, Predictor::Weighted, i, j, image,
                     dummy_header_pos, 2))
        return false;
      if (after - before > try_io.ftell()) {
        io.fseek(before, SEEK_SET);
        io.append(try_io);
        after = io.ftell();
      }
    }
    float bits = (after - header_pos) * 8.0;
    float pixels = 0.0;
    float ubits = 0.0;
    for (size_t k = i; k <= j; k++) {
      float chpixels = image.channel[k].w * image.channel[k].h;
      float uncompressed_bpp =
          ilog2(image.channel[k].maxval - image.channel[k].minval) + 1;
      if (uncompressed_bpp < 8)
        uncompressed_bpp = 8;
      else if (uncompressed_bpp < 16)
        uncompressed_bpp = 16;
      else
        uncompressed_bpp = 24;

      pixels += chpixels;
      if (image.channel[k].maxval > image.channel[k].minval)
        ubits += chpixels * uncompressed_bpp;
      if (ubits > 0.0) ubits += 32;
    }
    float bpp = bits / pixels;
    float ubpp = ubits / pixels;

    JXL_DEBUG_V(
        4,
        "Encoded channel %zu (%zux%zu, range %i..%i), %zu+%zu bytes "
        "[%zu-%zu] (%f bpp; uncompressed estimate: %f bpp; %.2f%% reduction)",
        i, image.channel[i].w, image.channel[i].h, image.channel[i].minval,
        image.channel[i].maxval, after - header_pos, header_pos - before,
        before, after, bpp, ubpp, (ubpp ? 100.0 - 100.0 * bpp / ubpp : 0));

    if (bits > ubits && ubits > 0.0) {
      io.fseek(before, SEEK_SET);
      Tree notree;
      if (!modular_encode_channels<
              BlobWriter, RacOut<BlobWriter>,
              FinalPropertySymbolCoder<ModularBitChancePass2,
                                       RacOut<BlobWriter>, MAX_BIT_DEPTH>,
              false>(io, notree, options, Predictor::Zero, i, j, image,
                     header_pos, 1))
        return false;
      after = io.ftell();
      float bpp = (after - header_pos) * 8.0 / pixels;
      JXL_DEBUG_V(4,
                  "Rolled back. Encoded channel %zu uncompressed (%zux%zu, "
                  "range %i..%i), %zu+%zu bytes [%zu-%zu] (%f bpp)",
                  i, image.channel[i].w, image.channel[i].h,
                  image.channel[i].minval, image.channel[i].maxval,
                  after - header_pos, header_pos - before, before, after, bpp);
    }
    i = j;
  }

  if (options.debug) options.heatmap->recompute_minmax();

  realio.append(io);
  return true;
}
#endif

template <typename IO>
bool modular_decode(IO &io, Image &image, modular_options &options,
                    size_t bytes_to_load) {
  if (image.nb_channels < 1) return true;

  // decode transforms
  int max_properties_nb_transforms = read_big_endian_varint(io);
  options.max_properties = (max_properties_nb_transforms >> 2);
  JXL_DEBUG_V(4, "Global option: up to %i back-referencing MA properties.",
              options.max_properties);
  int nb_transforms = (max_properties_nb_transforms & 3);
  if (nb_transforms == 3) nb_transforms += read_big_endian_varint(io);
  JXL_DEBUG_V(3, "Image data underwent %i transformations: ", nb_transforms);
  for (int i = 0; i < nb_transforms; i++) {
    int id_and_nb_params = read_big_endian_varint(io);
    Transform t(id_and_nb_params & 0xf);
    int nb_params = (id_and_nb_params >> 4);
    for (int j = 0; j < nb_params; j++)
      t.parameters.push_back(read_big_endian_varint(io));
    if (!options.identify) {
      t.meta_apply(image);
      image.transform.push_back(t);
    }
    JXL_DEBUG_V(2, "%s", t.name());
    if (t.ID == TRANSFORM_PALETTE) {
      if (t.parameters[0] == t.parameters[1])
        JXL_DEBUG_V(3, "[Compact channel %i to ", t.parameters[0]);
      else
        JXL_DEBUG_V(3, "[channels %i-%i with ", t.parameters[0],
                    t.parameters[1]);
      JXL_DEBUG_V(3, "%i colors]", t.parameters[2]);
    }
  }
  JXL_DEBUG_V(5, "Header decoded. Read %ld bytes so far.", io.ftell());
  if (options.identify) return true;
  if (image.error) {
    return JXL_FAILURE("Corrupt file. Aborting.");
  }

  size_t nb_channels = image.channel.size();

  // decode channel data
  for (size_t i = options.skipchannels; i < nb_channels; i++) {
    if (i >= image.nb_meta_channels &&
        (image.channel[i].w > options.max_chan_size ||
         image.channel[i].h > options.max_chan_size)) {
      break;
    }
    if ((bytes_to_load == 0 ||
         static_cast<size_t>(io.ftell()) < bytes_to_load) &&
        !io.isEOF()) {
      if (!modular_decode_channel<
              IO, FinalPropertySymbolCoder<ModularBitChancePass2, RacIn<IO>,
                                           MAX_BIT_DEPTH>>(
              io, options, i, image, bytes_to_load))
        return false;
    } else {
      JXL_DEBUG_V(3, "Skipping decode of channels %zu-%zu.", i,
                  nb_channels - 1);
      break;
    }
  }
  JXL_DEBUG_V(3, "Done decoding. Read %lu bytes.", io.ftell());
  return true;
}

template bool modular_decode(BlobReader &io, Image &image,
                             modular_options &options, size_t bytes_to_load);

#ifdef HAS_ENCODER

template bool modular_encode(BlobWriter &io, const Image &image,
                             modular_options &options);

void modular_prepare_encode(Image &image, modular_options &options) {
  // ensure that the ranges are correct and tight
  image.recompute_minmax();
  if (options.debug) {
    if (!options.heatmap) options.heatmap = new Image();
    options.heatmap->w = image.w;
    options.heatmap->h = image.h;
    for (size_t i = 0; i < image.channel.size(); i++)
      options.heatmap->channel.emplace_back(
          Channel(image.channel[i].w, image.channel[i].h, 0, 255));
  }
}
#endif

#ifdef HAS_ENCODER
bool modular_generic_compress(Image &image, PaddedBytes *bytes,
                              modular_options *opts, int loss,
                              bool try_transforms) {
  if (image.w == 0 || image.h == 0) return true;
  modular_options options;
  if (opts == nullptr)
    set_default_modular_options(options);
  else
    options = *opts;

  if (!options.predictor.size())
    options.predictor.push_back(5);  // use predictor 5 by default

  if (try_transforms) {
    if (loss > 1) {
      // lossy DC
      image.do_transform(Transform(TRANSFORM_SQUEEZE));
      Transform quantize(TRANSFORM_QUANTIZE);
      for (size_t i = image.nb_meta_channels; i < image.channel.size(); i++) {
        Channel &ch = image.channel[i];
        int shift = ch.hcshift + ch.vcshift;  // number of pixel halvings
        int q;
        q = (loss >> shift);
        if (q < 1) q = 1;
        quantize.parameters.push_back(q);
      }
      image.do_transform(quantize);
    } else if (!options.skipchannels) {
      // simple heuristic: if less than 90 percent of the values in the range
      // actually occur, it is probably worth it to do a compaction
      // unless there are few pixels to encode, then the overhead is not worth
      // it
      image.recompute_minmax();

      for (size_t c = 0; c < image.nb_channels; c++) {
        Transform maybe_palette_1(TRANSFORM_PALETTE);
        maybe_palette_1.parameters.push_back(c + image.nb_meta_channels);
        maybe_palette_1.parameters.push_back(c + image.nb_meta_channels);
        int colors = image.channel[c + image.nb_meta_channels].maxval -
                     image.channel[c + image.nb_meta_channels].minval + 1;
        float factor = 1.0;
        if (image.w * image.h < 1000) {
          factor =
              static_cast<float>(image.w) * static_cast<float>(image.h) / 1000;
        }
        maybe_palette_1.parameters.push_back((int)(0.9 * factor * colors));
        if (image.do_transform(maybe_palette_1)) {
          options.predictor.insert(options.predictor.begin(),
                                   1);  // left predictor for palette encoding
        }
      }
    }
    image.recompute_minmax();
  }

  modular_prepare_encode(image, options);
  BlobWriter io;
  bool status = modular_encode(io, image, options);
  if (!status) return false;
  JXL_DEBUG_V(
      4, "Modular-encoded a %zux%zu maxval=%i nbchans=%zu image in %zu bytes",
      image.w, image.h, image.maxval, image.real_nb_channels, io.size());
  bytes->append(io.blob);
  return true;
}
template <typename T>
bool modular_rect_compress_1(const Plane<T> &img, const Rect &rect,
                             PaddedBytes *bytes, modular_options *opts,
                             int loss) {
  Image image(rect.xsize(), rect.ysize(), 255, 1);
  for (size_t y = 0; y < image.h; y++) {
    pixel_type *JXL_RESTRICT to = image.channel[0].Row(y);
    const T *JXL_RESTRICT from = rect.ConstRow(img, y);
    for (size_t x = 0; x < image.w; x++) to[x] = from[x];
  }
  return modular_generic_compress(image, bytes, opts, loss);
}

// Change RGB to GRB, or XYB to YXB  (luma-like channel first)
int swap01(int i) {
  if (i == 0) return 1;
  if (i == 1) return 0;
  return i;
}
template <typename T>
bool modular_rect_compress_3(const Image3<T> &img, const Rect &rect,
                             PaddedBytes *bytes, modular_options *opts,
                             int loss) {
  Image image(rect.xsize(), rect.ysize(), 255, 3);
  for (int c = 0; c < 3; c++)
    for (size_t y = 0; y < image.h; y++) {
      pixel_type *JXL_RESTRICT to = image.channel[c].Row(y);
      const T *JXL_RESTRICT from = rect.ConstPlaneRow(img, swap01(c), y);
      for (size_t x = 0; x < image.w; x++) to[x] = from[x];
    }

  return modular_generic_compress(image, bytes, opts, loss);
}

template <typename T>
bool modular_rect_compress_2(const Plane<T> &img1, const Plane<T> &img2,
                             const Rect &rect, PaddedBytes *bytes,
                             modular_options *opts, pixel_type offset1,
                             pixel_type offset2) {
  Image image(rect.xsize(), rect.ysize(), 255, 2);
  for (size_t y = 0; y < image.h; y++) {
    const T *JXL_RESTRICT from1 = img1.ConstRow(y);
    pixel_type *JXL_RESTRICT to1 = image.channel[0].Row(y);
    for (size_t x = 0; x < image.w; x++)
      to1[x] = (pixel_type)from1[x] - offset1;
    const T *JXL_RESTRICT from2 = img2.ConstRow(y);
    pixel_type *JXL_RESTRICT to2 = image.channel[1].Row(y);
    for (size_t x = 0; x < image.w; x++)
      to2[x] = (pixel_type)from2[x] - offset2;
  }
  return modular_generic_compress(image, bytes, opts);
}

template <typename T>
bool modular_compress_2(const Plane<T> &img1, const Plane<T> &img2,
                        PaddedBytes *bytes, modular_options *opts) {
  JXL_DASSERT(img1.xsize() > 0);
  Image image(img1.xsize(), img1.ysize(), 255, 2);
  JXL_DASSERT(img1.xsize() == img2.xsize());
  JXL_DASSERT(img1.ysize() == 1);
  JXL_DASSERT(img2.ysize() == 1);
  for (size_t y = 0; y < image.h; y++) {
    const T *JXL_RESTRICT from1 = img1.ConstRow(y);
    pixel_type *JXL_RESTRICT to1 = image.channel[0].Row(y);
    for (size_t x = 0; x < image.w; x++) to1[x] = from1[x];
    const T *JXL_RESTRICT from2 = img2.ConstRow(y);
    pixel_type *JXL_RESTRICT to2 = image.channel[1].Row(y);
    for (size_t x = 0; x < image.w; x++) to2[x] = from2[x];
  }
  image.w = -image.w;  // flip the sign, so the width gets signalled
  return modular_generic_compress(image, bytes, opts);
}

template bool modular_rect_compress_1(const Plane<uint8_t> &img,
                                      const Rect &rect, PaddedBytes *bytes,
                                      modular_options *opts, int loss);
template bool modular_rect_compress_1(const Plane<int16_t> &img,
                                      const Rect &rect, PaddedBytes *bytes,
                                      modular_options *opts, int loss);
template bool modular_rect_compress_1(const Plane<uint16_t> &img,
                                      const Rect &rect, PaddedBytes *bytes,
                                      modular_options *opts, int loss);
template bool modular_rect_compress_3(const Image3<int16_t> &img,
                                      const Rect &rect, PaddedBytes *bytes,
                                      modular_options *opts, int loss);
template bool modular_rect_compress_3(const Image3<int32_t> &img,
                                      const Rect &rect, PaddedBytes *bytes,
                                      modular_options *opts, int loss);
template bool modular_rect_compress_2(const Plane<uint8_t> &img1,
                                      const Plane<uint8_t> &img2,
                                      const Rect &rect, PaddedBytes *bytes,
                                      modular_options *opts, pixel_type offset1,
                                      pixel_type offset2);
template bool modular_compress_2(const ImageI &img, const ImageI &img2,
                                 PaddedBytes *bytes, modular_options *opts);
#endif

bool modular_generic_decompress(const Span<const uint8_t> bytes, size_t *pos,
                                Image &image, modular_options &options,
                                size_t bytes_to_load, int undo_transforms) {
  BlobReader io(bytes.data(), bytes.size());
  io.fseek(*pos, SEEK_SET);
  bool status = modular_decode(io, image, options, bytes_to_load);
  if (!status) return false;
  image.undo_transforms(undo_transforms);
  JXL_DEBUG_V(4, "Modular-decoded a %zux%zu nbchans=%zu image from %lu bytes",
              image.w, image.h, image.real_nb_channels, io.ftell() - *pos);
  *pos = io.ftell();
  return true;
}
template <typename T>
bool modular_rect_decompress_1(const Span<const uint8_t> bytes, size_t *pos,
                               const Plane<T> *JXL_RESTRICT result,
                               const Rect &rect) {
  Image image(rect.xsize(), rect.ysize(), 255, 1);
  if (image.w == 0 || image.h == 0) return true;
  modular_options options;
  set_default_modular_options(options);
  options.nb_channels = 1;
  if (!modular_generic_decompress(bytes, pos, image, options)) return false;
  for (size_t y = 0; y < image.h; y++) {
    const pixel_type *JXL_RESTRICT from = image.channel[0].Row(y);
    T *JXL_RESTRICT to = rect.MutableRow(result, y);
    for (size_t x = 0; x < image.w; x++) to[x] = from[x];
  }
  return true;
}
template <typename T>
bool modular_rect_decompress_2(const Span<const uint8_t> bytes, size_t *pos,
                               const Plane<T> *JXL_RESTRICT result1,
                               const Plane<T> *JXL_RESTRICT result2,
                               const Rect &rect, pixel_type offset1,
                               pixel_type offset2) {
  Image image(rect.xsize(), rect.ysize(), 255, 2);
  if (image.w == 0 || image.h == 0) return true;
  modular_options options;
  set_default_modular_options(options);
  options.nb_channels = 2;
  if (!modular_generic_decompress(bytes, pos, image, options)) return false;
  for (size_t y = 0; y < image.h; y++) {
    const pixel_type *JXL_RESTRICT from = image.channel[0].Row(y);
    T *JXL_RESTRICT to = result1->MutableRow(y);
    for (size_t x = 0; x < image.w; x++) to[x] = from[x] + offset1;
  }
  for (size_t y = 0; y < image.h; y++) {
    const pixel_type *JXL_RESTRICT from = image.channel[1].Row(y);
    T *JXL_RESTRICT to = result2->MutableRow(y);
    for (size_t x = 0; x < image.w; x++) to[x] = from[x] + offset2;
  }
  return true;
}
template <typename T>
bool modular_rect_decompress_3(const Span<const uint8_t> bytes, size_t *pos,
                               const Image3<T> *JXL_RESTRICT result,
                               const Rect &rect) {
  Image image(rect.xsize(), rect.ysize(), 255, 3);
  if (image.w == 0 || image.h == 0) return true;
  modular_options options;
  set_default_modular_options(options);
  options.nb_channels = 3;
  if (!modular_generic_decompress(bytes, pos, image, options)) return false;
  for (int c = 0; c < 3; c++)
    for (size_t y = 0; y < image.h; y++) {
      const pixel_type *JXL_RESTRICT from = image.channel[c].Row(y);
      T *JXL_RESTRICT to = const_cast<T *>(result->PlaneRow(swap01(c), y));
      for (size_t x = 0; x < image.w; x++) to[x] = from[x];
    }
  return true;
}

template bool modular_rect_decompress_1(
    const Span<const uint8_t> bytes, size_t *pos,
    const Plane<uint8_t> *JXL_RESTRICT result, const Rect &rect);
template bool modular_rect_decompress_1(
    const Span<const uint8_t> bytes, size_t *pos,
    const Plane<int16_t> *JXL_RESTRICT result, const Rect &rect);
template bool modular_rect_decompress_1(
    const Span<const uint8_t> bytes, size_t *pos,
    const Plane<uint16_t> *JXL_RESTRICT result, const Rect &rect);
template bool modular_rect_decompress_2(
    const Span<const uint8_t> bytes, size_t *pos,
    const Plane<uint8_t> *JXL_RESTRICT result1,
    const Plane<uint8_t> *JXL_RESTRICT result2, const Rect &rect,
    pixel_type offset1, pixel_type offset2);
template bool modular_rect_decompress_3(
    const Span<const uint8_t> bytes, size_t *pos,
    const Image3<int16_t> *JXL_RESTRICT result, const Rect &rect);
template bool modular_rect_decompress_3(
    const Span<const uint8_t> bytes, size_t *pos,
    const Image3<int32_t> *JXL_RESTRICT result, const Rect &rect);

}  // namespace jxl