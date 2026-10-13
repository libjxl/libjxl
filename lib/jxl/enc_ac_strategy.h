// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_AC_STRATEGY_H_
#define LIB_JXL_ENC_AC_STRATEGY_H_

#include <jxl/memory_manager.h>

#include <algorithm>
#include <cstddef>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/image.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/quant_weights.h"

// `FindBestAcStrategy` uses heuristics to choose which AC strategy should be
// used in each block, as well as the initial quantization field.

namespace jxl {

struct AuxOut;
class AcStrategyImage;

// AC strategy selection: utility struct.

// Keeps the images that the AC strategy search reads together with their
// dimensions, so that every read can be checked against them (in debug builds,
// which include the fuzzers). Coordinates are in the frame, which is the
// `src_rect` area of `src`.
struct ACSConfig {
  const DequantMatrices* JXL_RESTRICT dequant;
  // One value per block.
  const ImageF* quant_field;
  const ImageF* masking_field;
  // One value per pixel; empty unless computed for this speed tier.
  const ImageF* masking1x1_field;
  const Image3F* src;
  Rect src_rect;
  float info_loss_multiplier;
  float cost_delta;
  float zeros_mul;
  // First pixel of the xsize x ysize block of channel c at (x, y); rows are
  // PixelStride() floats apart.
  const float* PixelBlock(size_t c, size_t x, size_t y, size_t xsize,
                          size_t ysize) const {
    JXL_DASSERT(x + xsize <= src_rect.xsize());
    JXL_DASSERT(y + ysize <= src_rect.ysize());
    return src_rect.ConstPlaneRow(*src, c, y) + x;
  }
  size_t PixelStride() const { return src->PixelsPerRow(); }
  float Masking(size_t bx, size_t by) const {
    JXL_DASSERT(bx < masking_field->xsize());
    const float masking = masking_field->ConstRow(by)[bx];
    JXL_DASSERT(masking > 0);
    return masking;
  }
  // How many of the n columns (rows) starting at x (y) the 1x1 masking field
  // has.
  size_t Masking1x1Columns(size_t x, size_t n) const {
    const size_t xsize = masking1x1_field->xsize();
    return x < xsize ? std::min(n, xsize - x) : 0;
  }
  size_t Masking1x1Rows(size_t y, size_t n) const {
    const size_t ysize = masking1x1_field->ysize();
    return y < ysize ? std::min(n, ysize - y) : 0;
  }
  // The n values of row y of the 1x1 masking field starting at column x.
  const float* Masking1x1Row(size_t x, size_t y, size_t n) const {
    JXL_DASSERT(n > 0 && x + n <= masking1x1_field->xsize());
    const float* row = masking1x1_field->ConstRow(y) + x;
    JXL_DASSERT(row[0] > 0);
    return row;
  }
  float Quant(size_t bx, size_t by) const {
    JXL_DASSERT(bx < quant_field->xsize());
    const float quant = quant_field->ConstRow(by)[bx];
    JXL_DASSERT(quant > 0);
    return quant;
  }
};

struct AcStrategyHeuristics {
  explicit AcStrategyHeuristics(JxlMemoryManager* memory_manager,
                                const CompressParams& cparams)
      : memory_manager(memory_manager),
        cparams(cparams),
        mem_per_thread(0),
        qmem_per_thread(0) {}
  Status Init(const Image3F& src, const Rect& rect_in,
              const ImageF& quant_field, const ImageF& mask,
              const ImageF& mask1x1, DequantMatrices* matrices);
  Status PrepareForThreads(std::size_t num_threads);
  Status ProcessRect(const Rect& rect, const ColorCorrelationMap& cmap,
                     AcStrategyImage* ac_strategy, size_t thread);
  Status Finalize(const FrameDimensions& frame_dim,
                  const AcStrategyImage& ac_strategy, AuxOut* aux_out);
  JxlMemoryManager* memory_manager;
  const CompressParams& cparams;
  ACSConfig config = {};
  size_t mem_per_thread;
  AlignedMemory mem;
  size_t qmem_per_thread;
  AlignedMemory qmem;
};

}  // namespace jxl

#endif  // LIB_JXL_ENC_AC_STRATEGY_H_
