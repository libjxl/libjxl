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

#ifndef JXL_AR_CONTROL_FIELD_H_
#define JXL_AR_CONTROL_FIELD_H_

#include "jxl/ac_strategy.h"
#include "jxl/base/data_parallel.h"
#include "jxl/chroma_from_luma.h"
#include "jxl/common.h"
#include "jxl/enc_cache.h"
#include "jxl/enc_params.h"
#include "jxl/image.h"
#include "jxl/quant_weights.h"

namespace jxl {

void FindBestArControlField(const Image3F& opsin, PassesEncoderState* enc_state,
                            ThreadPool* pool);

}  // namespace jxl

#endif  // JXL_AR_CONTROL_FIELD_H_
