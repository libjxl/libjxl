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

jxl::Status TransformCodecInOutTo(jxl::CodecInOut& io,
                                  const jxl::ColorEncoding& c_desired,
                                  const JxlCmsInterface& cms,
                                  jxl::ThreadPool* pool) {
  if (io.metadata.m.have_preview) {
    JXL_RETURN_IF_ERROR(io.preview_frame.TransformTo(c_desired, cms, pool));
  }
  for (jxl::ImageBundle& ib : io.frames) {
    JXL_RETURN_IF_ERROR(ib.TransformTo(c_desired, cms, pool));
  }
  return true;
}

}  // namespace tools
}  // namespace jpegxl
