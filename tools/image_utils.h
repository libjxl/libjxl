// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_IMAGE_UTILS_H_
#define TOOLS_IMAGE_UTILS_H_

#include <jxl/cms_interface.h>

#include "lib/jxl/base/status.h"

namespace jxl {
class CodecInOut;
struct ColorEncoding;
class ThreadPool;
}  // namespace jxl

namespace jpegxl {
namespace tools {

jxl::Status TransformCodecInOutTo(jxl::CodecInOut& io,
                                  const jxl::ColorEncoding& c_desired,
                                  const JxlCmsInterface& cms,
                                  jxl::ThreadPool* pool);

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_IMAGE_UTILS_H_
