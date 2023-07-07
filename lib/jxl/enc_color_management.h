// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_COLOR_MANAGEMENT_H_
#define LIB_JXL_ENC_COLOR_MANAGEMENT_H_

// ICC profiles and color space conversions.

#include <jxl/cms_interface.h>
#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/common.h"
#include "lib/jxl/image.h"

namespace jxl {

const JxlCmsInterface& GetJxlCms();

}  // namespace jxl

#endif  // LIB_JXL_ENC_COLOR_MANAGEMENT_H_
