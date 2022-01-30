/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

/** @addtogroup libjxl_common
 * @{
 * @file types.h
 * @brief Data types for the JPEG XL API, for both encoding and decoding.
 */

#ifndef JXL_TYPES_H_
#define JXL_TYPES_H_

// example of a new struct build system.
// this will allow for more advanced code generation.

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/**
 * A portable @c bool replacement.
 *
 * ::JXL_BOOL is a "documentation" type: actually it is @c int, but in API it
 * denotes a type, whose only values are ::JXL_TRUE and ::JXL_FALSE.
 */
#define JXL_BOOL int
/** Portable @c true replacement. */
#define JXL_TRUE 1
/** Portable @c false replacement. */
#define JXL_FALSE 0

// load the definitions from the underlying implementation
#include "internal/meta_types.h"

/** Data type holding the 4-character type name of an ISOBMFF box.
 */
typedef char JxlBoxType[4];

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JXL_TYPES_H_ */

/** @}*/
