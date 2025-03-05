// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/codec_config.h"

#include <cstdint>
#include <cstdio>
#include <hwy/per_target.h>
#include <hwy/targets.h>
#include <string>

#include "tools/tool_version.h"

namespace jpegxl {
namespace tools {

std::string CodecConfigString(uint32_t lib_version) {
  std::string config;

  if (lib_version != 0) {
    char version_str[20];
    snprintf(version_str, sizeof(version_str), "v%d.%d.%d ",
             lib_version / 1000000, (lib_version / 1000) % 1000,
             lib_version % 1000);
    config += version_str;
  }

  std::string version = kJpegxlVersion;
  if (version != "(unknown)") {
    config += version + ' ';
  }

#if defined(ADDRESS_SANITIZER)
  config += " ASAN ";
#elif defined(MEMORY_SANITIZER)
  config += " MSAN ";
#elif defined(THREAD_SANITIZER)
  config += " TSAN ";
#else
#endif

  int64_t current = hwy::DispatchedTarget();
  bool seen_current = false;
  bool seen_target = false;
  config += "[";
  for (const int64_t target : hwy::SupportedAndGeneratedTargets()) {
    if (target == current) {
      config += '_';
      config += hwy::TargetName(target);
      config += '_';
      seen_current = true;
    } else {
      config += hwy::TargetName(target);
    }
    config += ',';
    seen_target = true;
  }
  if (!seen_target) {
    config += "no targets found,";
  } else if (!seen_current) {
    config += "unsupported but chosen: ";
    config += hwy::TargetName(current);
    config += ',';
  }
  config.resize(config.size() - 1);  // remove trailing comma
  config += "]";

  return config;
}

}  // namespace tools
}  // namespace jpegxl
