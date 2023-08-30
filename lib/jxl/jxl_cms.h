// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_JXL_CMS_H_
#define LIB_JXL_JXL_CMS_H_

// ICC profiles and color space conversions.

#include <jxl/cms_interface.h>
#include <jxl/jxl_cms_export.h>

extern "C" JXL_CMS_EXPORT const JxlCmsInterface* JxlGetDefaultCms();

#endif  // LIB_JXL_JXL_CMS_H_
