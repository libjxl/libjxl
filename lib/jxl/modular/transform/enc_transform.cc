// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/modular/transform/enc_transform.h"

#include "lib/jxl/modular/transform/enc_palette.h"
#include "lib/jxl/modular/transform/enc_rct.h"
#include "lib/jxl/modular/transform/enc_squeeze.h"

namespace jxl {

Status TransformForward(Transform &t, Image &input,
                        const weighted::Header &wp_header, ThreadPool *pool) {
  switch (t.id) {
    case TransformId::kRCT:
      return FwdRCT(input, t.begin_c, t.rct_type);
    case TransformId::kSqueeze:
      return FwdSqueeze(input, t.squeezes, pool);
    case TransformId::kPalette:
      return FwdPalette(input, t.begin_c, t.begin_c + t.num_c - 1, t.nb_colors,
                        t.ordered_palette, t.lossy_palette, t.predictor,
                        wp_header);
    default:
      return JXL_FAILURE("Unknown transformation (ID=%u)",
                         static_cast<unsigned int>(t.id));
  }
}

}  // namespace jxl
