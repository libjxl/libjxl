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

#include "jxl/dct.h"

namespace jxl {

// Definition of constexpr arrays.
constexpr float DCTResampleScales<1, 8>::kScales[];
constexpr float DCTResampleScales<2, 8>::kScales[];
constexpr float DCTResampleScales<2, 16>::kScales[];
constexpr float DCTResampleScales<4, 16>::kScales[];
constexpr float DCTResampleScales<4, 32>::kScales[];
constexpr float DCTResampleScales<8, 32>::kScales[];
constexpr float DCTResampleScales<8, 1>::kScales[];
constexpr float DCTResampleScales<8, 2>::kScales[];
constexpr float DCTResampleScales<16, 2>::kScales[];
constexpr float DCTResampleScales<16, 4>::kScales[];
constexpr float DCTResampleScales<32, 4>::kScales[];
constexpr float DCTResampleScales<32, 8>::kScales[];

}  // namespace jxl
