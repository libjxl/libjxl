// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_ENC_FAST_LOSSLESS_H_
#define LIB_JXL_ENC_FAST_LOSSLESS_H_
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// A FJxlParallelRunner must call fun(opaque, i) for all i from 0 to count. It
// may do so in parallel.
typedef void(FJxlParallelRunner)(void* runner_opaque, void* opaque,
                                 void fun(void*, size_t), size_t count);

// You may pass `nullptr` as a runner: encoding will be sequential.
size_t JxlFastLosslessEncode(const unsigned char* rgba, size_t width,
                             size_t row_stride, size_t height, size_t nb_chans,
                             size_t bitdepth, bool big_endian, int effort,
                             unsigned char** output, void* runner_opaque,
                             FJxlParallelRunner runner);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIB_JXL_ENC_FAST_LOSSLESS_H_
