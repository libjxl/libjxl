// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_JXL_CMS_H_
#define LIB_JXL_JXL_CMS_H_

// ICC profiles and color space conversions.

#include <jxl/cms_interface.h>
#include <jxl/color_encoding.h>
#include <jxl/jxl_cms_export.h>
#include <jxl/types.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

JXL_CMS_EXPORT const JxlCmsInterface* JxlGetDefaultCms();
JXL_CMS_EXPORT JXL_BOOL JxlCmsCreateProfile(const JxlColorEncoding* c,
                                            uint8_t** out, size_t* out_size);
JXL_CMS_EXPORT JXL_BOOL JxlCmsCIEXYZFromWhiteCIExy(double wx, double wy,
                                                   float XYZ[3]);
JXL_CMS_EXPORT JXL_BOOL JxlCmsAdaptToXYZD50(float wx, float wy,
                                            float matrix[9]);
JXL_CMS_EXPORT JXL_BOOL JxlCmsPrimariesToXYZ(float rx, float ry, float gx,
                                             float gy, float bx, float by,
                                             float wx, float wy,
                                             float matrix[9]);
JXL_CMS_EXPORT JXL_BOOL JxlCmsPrimariesToXYZD50(float rx, float ry, float gx,
                                                float gy, float bx, float by,
                                                float wx, float wy,
                                                float matrix[9]);
JXL_CMS_EXPORT size_t JxlCmsColorEncodingDescription(const JxlColorEncoding* c,
                                                     char out[320]);

#ifdef __cplusplus
}
#endif

#endif  // LIB_JXL_JXL_CMS_H_
