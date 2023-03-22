// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_ARGS_H_
#define TOOLS_ARGS_H_

// Helpers for parsing command line arguments. No include guard needed.

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <utility>

#include "lib/jxl/base/override.h"
#include "lib/jxl/base/status.h"

namespace jpegxl {
namespace tools {

static inline bool ParseOverride(const char* arg, jxl::Override* out) {
  const std::string s_arg(arg);
  if (s_arg == "1") {
    *out = jxl::Override::kOn;
    return true;
  }
  if (s_arg == "0") {
    *out = jxl::Override::kOff;
    return true;
  }
  fprintf(stderr, "Invalid flag, %s must be 0 or 1\n", arg);
  return JXL_FAILURE("Args");
}

static inline bool ParseFloatPair(const char* arg,
                                  std::pair<float, float>* out) {
  int parsed = sscanf(arg, "%f,%f", &out->first, &out->second);
  if (parsed == 1) {
    out->second = out->first;
  } else if (parsed != 2) {
    fprintf(stderr,
            "Unable to interpret as float pair separated by a comma: %s.\n",
            arg);
    return JXL_FAILURE("Args");
  }
  return true;
}

template <typename Map>
static inline bool ParseAndAppendKeyValue(const char* arg, Map* out) {
  const char* eq = strchr(arg, '=');
  if (!eq) {
    fprintf(stderr, "Expected argument as 'key=value' but received '%s'\n",
            arg);
    return false;
  }
  std::string key(arg, eq);
  out->Add(key, std::string(eq + 1));
  return true;
}

static inline bool ParseCString(const char* arg, const char** out) {
  *out = arg;
  return true;
}

static inline bool IncrementUnsigned(size_t* out) {
  (*out)++;
  return true;
}

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_ARGS_H_
