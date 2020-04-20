// Copyright 2019 Google LLC
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

#ifndef HWY_INTERFACE_H_
#define HWY_INTERFACE_H_

// Definitions that may be useful for interfaces of modules using Highway.

#include <stddef.h>
#include <stdint.h>

namespace hwy {

#ifdef _MSC_VER
#define HWY_RESTRICT __restrict
#else
#define HWY_RESTRICT __restrict__
#endif

#if defined(__i386__) || defined(__x86_64__) || defined(_M_X64)
static constexpr size_t kMaxVectorSize = 64;  // AVX-512
#define HWY_ALIGN_MAX alignas(64)
#else
static constexpr size_t kMaxVectorSize = 16;
#define HWY_ALIGN_MAX alignas(16)
#endif

// 4 instances of a given literal value, useful as input to LoadDup128.
#define HWY_REP4(literal) literal, literal, literal, literal

// Returns (cached) bitfield of enabled targets that are supported on this CPU.
uint32_t SupportedTargets();

}  // namespace hwy

#endif  // HWY_INTERFACE_H_
