// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_NONPHOTO_SEPARATION_H_
#define LIB_JXL_ENC_NONPHOTO_SEPARATION_H_

// Separates nonphotographic and photographic parts of an image,
// so VarDCT can be used for photographic parts and Modular for nonphoto

#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/common.h"
#include "lib/jxl/enc_cache.h"
#include "lib/jxl/enc_params.h"
#include "lib/jxl/image.h"

namespace jxl {

// Heuristic to split an image in a nonphoto and a photo part
// `separation` is 0 for nonphoto and nonzero for photo parts
// Returns fraction of the image that is nonphoto
float FindSeparation(const Image3F& opsin, ImageU* separation,
                     const CompressParams& cparams, ThreadPool* pool);

// Encode the nonphoto part as a separate frame, and subtract the
// decoded nonphoto frame from the remaining image
void EncodeAndSubtract(Image3F* JXL_RESTRICT opsin, const ImageU& separation,
                       const CompressParams& orig_cparams,
                       PassesEncoderState* JXL_RESTRICT state, ThreadPool* pool,
                       AuxOut* aux_out);

}  // namespace jxl

#endif  // LIB_JXL_ENC_NONPHOTO_SEPARATION_H_
