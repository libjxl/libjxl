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
#include "jxl/dec_ans.h"
#include "jxl/enc_ans.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/image.h"
#include "jxl/modular/image/image.h"
#include "jxl/modular/options.h"

namespace jxl {

bool ModularGenericCompress(
    Image &image, const ModularOptions &opts, BitWriter *writer,
    AuxOut *aux_out = nullptr, size_t layer = 0,
    const HybridUintConfig &uint_config = kHybridUint420Config);

// undo_transforms == N > 0: undo all transforms except the first N
//                           (e.g. to represent YCbCr420 losslessly)
// undo_transforms == 0: undo all transforms
// undo_transforms == -1: undo all transforms but don't clamp to range
// undo_transforms == -2: don't undo any transform
bool ModularGenericDecompress(BitReader *br, Image &image,
                              ModularOptions *options,
                              int undo_transforms = -1);
}  // namespace jxl

#endif  // JXL_MODULAR_ENCODING_ENCODING_H_
