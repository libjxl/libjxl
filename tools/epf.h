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

#ifndef TOOLS_EPF_H_
#define TOOLS_EPF_H_

#include "lib/jxl/base/status.h"
#include "lib/jxl/codec_in_out.h"

namespace jpegxl {
namespace tools {

jxl::Status RunEPF(uint32_t epf_iters, float distance, int sharpness_parameter,
                   jxl::CodecInOut* io, jxl::ThreadPool* pool);

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_EPF_H_
