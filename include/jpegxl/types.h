/* Copyright (c) the JPEG XL Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file jpegxl/types.h
 * @brief Data types for the JPEG XL API.
 */

#ifndef JPEGXL_TYPES_H_
#define JPEGXL_TYPES_H_

/**
 * A portable @c bool replacement.
 *
 * ::JPEGXL_BOOL is a "documentation" type: actually it is @c int, but in API it
 * denotes a type, whose only values are ::JPEGXL_TRUE and ::JPEGXL_FALSE.
 */
#define JPEGXL_BOOL int
/** Portable @c true replacement. */
#define JPEGXL_TRUE 1
/** Portable @c false replacement. */
#define JPEGXL_FALSE 0

#endif /* JPEGXL_TYPES_H_ */
