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

#include "tools/codec_config.h"

#include <hwy/targets.h>

#include "jxl/base/status.h"

namespace jpegxl {
namespace tools {

std::string CodecConfigString() {
  std::string config;

  std::string version = JPEGXL_VERSION;
  if (version != "(unknown)") {
    config += version + ' ';
  }

#if defined(ADDRESS_SANITIZER)
  config += " asan ";
#elif defined(MEMORY_SANITIZER)
  config += " msan ";
#elif defined(THREAD_SANITIZER)
  config += " tsan ";
#else
#endif

  bool saw_target = false;
  config += "Supported/generated: ";
  for (const uint32_t target : hwy::SupportedAndGeneratedTargets()) {
    config += hwy::TargetName(target);
    config += ',';
    saw_target = true;
  }
  JXL_ASSERT(saw_target);
  (void)saw_target;
  config.resize(config.size() - 1);  // remove trailing comma

  return config;
}

}  // namespace tools
}  // namespace jpegxl
