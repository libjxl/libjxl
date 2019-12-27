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

#ifndef JXL_DOT_DICTIONARY_H_
#define JXL_DOT_DICTIONARY_H_

// Dots are stored in a dictionary to avoid storing similar dots multiple
// times.

#include <stddef.h>

#include <vector>

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/enc_params.h"
#include "jxl/image.h"
#include "jxl/patch_dictionary.h"

namespace jxl {

std::vector<PatchInfo> FindDotDictionary(const CompressParams& cparams,
                                         const Image3F& opsin,
                                         const ColorCorrelationMap& cmap,
                                         ThreadPool* pool);

}  // namespace jxl

#endif  // JXL_DOT_DICTIONARY_H_
