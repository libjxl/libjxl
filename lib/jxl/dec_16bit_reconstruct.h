// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_cache.h"

namespace jxl {
void FinalizeImageRect16(Image3F* input_image, const Rect& input_rect,
                         ImageF* alpha, const Rect& alpha_rect,
                         PassesDecoderState* dec_state, size_t thread,
                         const Rect& frame_rect);
}
