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

#ifndef JXL_MODULAR_CONFIG_H_
#define JXL_MODULAR_CONFIG_H_

namespace jxl {

#ifndef DECODER_ONLY
#define HAS_ENCODER
#endif

// MAX_BIT_DEPTH is the maximum bit depth of the absolute values of the numbers
// that actually get encoded Squeeze residuals plus YCoCg can result in 17-bit
// absolute values on 16-bit input, so 17 is needed to encode 16-bit input with
// default options Higher bit depth is needed when DCT is used on 16-bit input.

#define HDR

#ifndef HDR
#define MAX_BIT_DEPTH 14
#else
#define MAX_BIT_DEPTH 30
#endif

// The above compile-time constant only determines the size of the chance tables
// in the MA trees, and in any case the maximum bit depth is limited by the
// integer type used in the channel buffers

/**************************************************/
/* DANGER ZONE: OPTIONS THAT CHANGE THE BITSTREAM */
/* If you modify these, the bitstream format      */
/* changes, so it is no longer compatible!        */
/**************************************************/

// Default squeeze will ensure that the first 'scan' fits in a 8x8 rectangle
#define MAX_FIRST_PREVIEW_SIZE 8
// Round truncation offsets to a multiples of 1 byte (using less precise offsets
// requires a more careful implementation of partial decode)
#define TRUNCATION_OFFSET_RESOLUTION 1

#ifdef _MSC_VER
#define ATTRIBUTE_HOT
#else
#define ATTRIBUTE_HOT __attribute__((hot))
#endif

}  // namespace jxl

#endif  // JXL_MODULAR_CONFIG_H_
