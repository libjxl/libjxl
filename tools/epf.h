// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
