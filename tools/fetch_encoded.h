/* Copyright (c) the JPEG XL Project Authors. All rights reserved.
 *
 * Use of this source code is governed by a BSD-style
 * license that can be found in the LICENSE file.
 */

#ifndef JXL_FETCH_ENCODED_H_
#define JXL_FETCH_ENCODED_H_


#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

#include <stdint.h>

#include "jxl/types.h"


JXL_BOOL fetch_jxl_encoded_image(JxlEncoder *jxl_encoder,
                                 uint8_t **compressed_out,
                                 size_t *compressed_size_out);

JXL_BOOL write_jxl_file(const uint8_t* bytes,
                        size_t size,
                        const char* filename);
  
#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JXL_FETCH_ENCODED_H_ */
