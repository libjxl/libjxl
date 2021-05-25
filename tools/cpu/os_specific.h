// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_CPU_OS_SPECIFIC_H_
#define TOOLS_CPU_OS_SPECIFIC_H_

// OS-specific function to query the processor topology and thread affinity.

#include <stddef.h>

#include <string>
#include <vector>

#include "lib/jxl/base/status.h"

namespace jpegxl {
namespace tools {
namespace cpu {

// Called by arch_specific. Returns false if `pt` remains unchanged. Only
// implemented/needed on OSX.
struct ProcessorTopology;

jxl::Status GetProcessorTopologyFromOS(ProcessorTopology* pt);

// Returns logical processor numbers in [0, N), where N is the number of bits in
// the thread's initial affinity (unaffected by any SetThreadAffinity).
std::vector<int> AvailableCPUs();

// Opaque.
struct ThreadAffinity;

// Returns current affinity; useful for restoring the original value.
// Caller must free() the pointer - dynamic allocation is required because
// ThreadAffinity is an incomplete type.
ThreadAffinity* GetThreadAffinity();

// Restores a previous affinity returned by GetThreadAffinity.
jxl::Status SetThreadAffinity(ThreadAffinity* affinity);

// Ensures the thread is running on the specified cpu, and no others.
// Useful for reducing nanobenchmark variability (fewer context switches).
// Calls SetThreadAffinity.
jxl::Status PinThreadToCPU(int cpu);

// Random choice of CPU avoids overloading any one core. Calls PinThreadToCPU.
jxl::Status PinThreadToRandomCPU();

// Returns total physical memory size [MiB], or 0 if unknown. This function
// returns a cached value initialized on the first call.
size_t TotalMemoryMiB();

// Executes a command in a subprocess.
// Status RunCommand(const std::vector<std::string>& args);

}  // namespace cpu
}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_CPU_OS_SPECIFIC_H_
