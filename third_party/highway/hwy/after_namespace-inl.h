// Copyright 2020 Google LLC
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

// Preceded by end_target-inl.h and closing any namespaces (required on clang
// prior to version 9).

#if HWY_TARGET != HWY_SCALAR
#if HWY_COMPILER_CLANG
#pragma clang attribute pop
#elif HWY_COMPILER_GCC
#pragma GCC pop_options
#else
// MSVC doesn't require any attributes in order to use AVX2 etc.
#endif  // HWY_COMPILER_*
#endif  // HWY_TARGET != HWY_SCALAR
