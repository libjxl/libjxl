// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_BOUNDS_SAFETY_H_
#define LIB_JXL_BASE_BOUNDS_SAFETY_H_

// Portability macros for optional Clang -fbounds-safety annotations.
//
// When JPEGXL_SUPPORT_FBOUNDS_SAFETY is defined (typically via
// -DJPEGXL_SUPPORT_FBOUNDS_SAFETY and a Clang toolchain that implements
// -fbounds-safety), these macros expand to Clang bounds annotations.
// Otherwise they expand to nothing so default builds are unchanged.
//
// Pattern matches libwebp / libpng inert-macro adoption: annotations are
// inert unless explicitly enabled.

#ifdef JPEGXL_SUPPORT_FBOUNDS_SAFETY

#include <ptrcheck.h>
/* Prefer __counted_by_or_null for pointers that may be NULL while the
 * companion size field is zero (JxlDecoder next_in / avail_in pattern).
 */
#define JXL_COUNTED_BY(n) __counted_by(n)
#define JXL_COUNTED_BY_OR_NULL(n) __counted_by_or_null(n)

#else /* !JPEGXL_SUPPORT_FBOUNDS_SAFETY */

#define JXL_COUNTED_BY(n)
#define JXL_COUNTED_BY_OR_NULL(n)

#endif /* JPEGXL_SUPPORT_FBOUNDS_SAFETY */

#endif  // LIB_JXL_BASE_BOUNDS_SAFETY_H_
