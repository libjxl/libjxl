// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/image_utils.h"

#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/color_encoding_internal.h"

namespace jpegxl {
namespace tools {

using ::jxl::CodecInOut;
using ::jxl::ColorEncoding;
using ::jxl::ImageBundle;
using ::jxl::Status;
using ::jxl::ThreadPool;

jxl::Status TransformCodecInOutTo(CodecInOut& io,
                                  const ColorEncoding& c_desired,
                                  const JxlCmsInterface& cms,
                                  ThreadPool* pool) {
  if (io.metadata.m.have_preview) {
    JXL_RETURN_IF_ERROR(io.preview_frame.TransformTo(c_desired, cms, pool));
  }
  for (ImageBundle& ib : io.frames) {
    JXL_RETURN_IF_ERROR(ib.TransformTo(c_desired, cms, pool));
  }
  return true;
}

}  // namespace tools
}  // namespace jpegxl
