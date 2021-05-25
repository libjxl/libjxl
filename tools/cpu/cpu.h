// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
