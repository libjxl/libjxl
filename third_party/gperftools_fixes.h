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

#ifndef __THIRD_PARTY_GPERFTOOLS_FIXES_H__
#define __THIRD_PARTY_GPERFTOOLS_FIXES_H__

#include <stdarg.h>  // for thread_lister.h

#ifdef __cplusplus
#include <cstddef>  // std::size_t
namespace std {

#if __cplusplus < 201703L
enum class align_val_t : std::size_t {};
#endif

}  // namespace std
#endif  // __cplusplus

#endif  // __THIRD_PARTY_GPERFTOOLS_FIXES_H__
