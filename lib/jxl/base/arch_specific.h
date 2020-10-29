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

#ifndef LIB_JXL_BASE_ARCH_SPECIFIC_H_
#define LIB_JXL_BASE_ARCH_SPECIFIC_H_

#include <stddef.h>
#include <stdint.h>

#include <hwy/base.h>  // kMaxVectorSize

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

#if defined(__x86_64__) || defined(_M_X64)
#define JXL_ARCH_X64 1
#else
#define JXL_ARCH_X64 0
#endif

#if defined(__powerpc64__) || defined(_M_PPC)
#define JXL_ARCH_PPC 1
#else
#define JXL_ARCH_PPC 0
#endif

#if defined(__aarch64__) || defined(__arm__)
#define JXL_ARCH_ARM 1
#else
#define JXL_ARCH_ARM 0
#endif

struct ProcessorTopology {
  size_t logical_per_core = 1;
  size_t cores_per_package = 1;
  size_t packages = 1;
};

// Relatively expensive, preferably only call once.
Status DetectProcessorTopology(ProcessorTopology* pt);

// Returns the nominal (without Turbo Boost) CPU clock rate [Hertz]. Useful for
// (roughly) characterizing the CPU speed.
double NominalClockRate();

// Returns tsc_timer frequency, useful for converting ticks to seconds. This is
// unaffected by CPU throttling ("invariant"). Thread-safe. Returns timebase
// frequency on PPC, NominalClockRate on X64, otherwise 1E9.
double InvariantTicksPerSecond();

}  // namespace jxl

#endif  // LIB_JXL_BASE_ARCH_SPECIFIC_H_
