// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef JXL_C_INTEROP
#define JXL_C_INTEROP

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
#define CLINK extern "C"
#else
#define CLINK
#endif

// Input: byte array of length size.
// Returns a pointer to an owned chunk of memory, containing interleaved R, G,
// B, A channels (null there was a decoding error). If decoding was successful,
// writes to stride the stride of the returned buffer, to xsize and ysize the
// size of the image in pixels, and 0/1 to has_alpha if the image doesn't/has an
// alpha channel. Returned pointer should be freed with JxlFreePixels.
// TODO(firsching): remove this file and use the generic C API directly.
CLINK uint8_t *JxlMemoryToPixels(const uint8_t *data, size_t size,
                                 size_t *stride, size_t *xsize, size_t *ysize,
                                 int *has_alpha);

CLINK void JxlFreePixels(uint8_t *pixels);

#undef CLINK

#endif
