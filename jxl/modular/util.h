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

#ifndef JXL_MODULAR_UTIL_H_
#define JXL_MODULAR_UTIL_H_

namespace jxl {

#define CLAMP(x, l, u) (((x) < (l) ? (l) : ((x) > (u) ? (u) : (x))))
#define MIN(A, B) ((A) < (B) ? (A) : (B))
#define MAX(A, B) ((A) > (B) ? (A) : (B))

template <typename I>
I inline median3(I a, I b, I c) {
  if (a < b) {
    if (b < c) {
      return b;
    } else {
      return a < c ? c : a;
    }
  } else {
    if (a < c) {
      return a;
    } else {
      return b < c ? c : b;
    }
  }
}

}  // namespace jxl

#endif  // JXL_MODULAR_UTIL_H_
