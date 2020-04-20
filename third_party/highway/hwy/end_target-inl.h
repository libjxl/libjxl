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

// This was included from ops/*-inl.h: do nothing (keep namespace open until
// the user's end_target-inl.h)
#ifdef HWY_BEGIN_TARGET_NESTED
#undef HWY_BEGIN_TARGET_NESTED
#else

// To avoid warnings, only close namespace opened by begin_target-inl.h if
// compiling, or the IDE is parsing a translation unit (not just this header).
#ifdef HWY_ATTR
}  // namespace
}  // namespace N_$TARGET
#endif  // HWY_ATTR

// Undef to ensure no implementation functions come after this header. Also
// allows begin_target-inl.h to immediately redefine them.
#undef HWY_ATTR
#undef HWY_ALIGN
#undef HWY_LANES
#undef HWY_CAPS

// Reset include guard of begin_target-inl.h so it is active again for the
// next target (if using foreach_target.h).
#undef HWY_BEGIN_TARGET_INL_H_
#endif  // HWY_BEGIN_TARGET_NESTED
