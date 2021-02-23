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

#ifndef TOOLS_CPU_CPU_H_
#define TOOLS_CPU_CPU_H_

#include <stddef.h>
#include <stdint.h>

#include "lib/jxl/base/status.h"

namespace jpegxl {
namespace tools {
namespace cpu {

struct ProcessorTopology {
  size_t logical_per_core = 1;
  size_t cores_per_package = 1;
  size_t packages = 1;
};

// Relatively expensive, preferably only call once.
jxl::Status DetectProcessorTopology(ProcessorTopology* pt);

}  // namespace cpu
}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_CPU_CPU_H_
