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

// Preceded by the implementation section of the file that included us.

// To avoid warnings, only close namespace opened by begin_target-inl.h if
// compiling, or the IDE is parsing something that included this header.
#ifdef HWY_ALIGN
}  // namespace
}  // namespace N_$TARGET
#endif

// Undef to ensure no implementation functions come after this header. Also
// allows begin_target-inl.h to immediately redefine them.
#undef HWY_ALIGN
#undef HWY_LANES

#undef HWY_CAP_GATHER
#undef HWY_CAP_VARIABLE_SHIFT
#undef HWY_CAP_INT64
#undef HWY_CAP_CMP64
#undef HWY_CAP_DOUBLE
#undef HWY_CAP_GE256
#undef HWY_CAP_GE512

// Reset include guard of begin_target-inl.h so it is active again for the
// next target (if HWY_TARGET_INCLUDE is defined).
#undef HWY_BEGIN_TARGET_INL_H_
