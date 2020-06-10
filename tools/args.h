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

#ifndef TOOLS_ARGS_H_
#define TOOLS_ARGS_H_

// Helpers for parsing command line arguments. No include guard needed.

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "jxl/base/override.h"
#include "jxl/base/status.h"
#include "jxl/codec_in_out.h"  // DecoderHints
#include "jxl/gaborish.h"

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

static inline bool ParseUnsigned(const char* arg, size_t* out) {
  char* end;
  *out = static_cast<size_t>(strtoull(arg, &end, 0));
  if (end[0] != '\0') {
    fprintf(stderr, "Unable to interpret as unsigned integer: %s.\n", arg);
    return JXL_FAILURE("Args");
  }
  return true;
}

static inline bool ParseUint32(const char* arg, uint32_t* out) {
  size_t value = 0;
  bool ret = ParseUnsigned(arg, &value);
  if (ret) *out = value;
  return ret;
}

static inline bool ParseSigned(const char* arg, int* out) {
  char* end;
  *out = static_cast<int>(strtol(arg, &end, 0));
  if (end[0] != '\0') {
    fprintf(stderr, "Unable to interpret as signed integer: %s.\n", arg);
    return JXL_FAILURE("Args");
  }
  return true;
}

static inline bool ParseFloat(const char* arg, float* out) {
  char* end;
  *out = static_cast<float>(strtod(arg, &end));
  if (end[0] != '\0') {
    fprintf(stderr, "Unable to interpret as float: %s.\n", arg);
    return JXL_FAILURE("Args");
  }
  return true;
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

static inline bool ParseDouble(const char* arg, double* out) {
  char* end;
  *out = static_cast<double>(strtod(arg, &end));
  if (end[0] != '\0') {
    fprintf(stderr, "Unable to interpret as double: %s.\n", arg);
    return JXL_FAILURE("Args");
  }
  return true;
}

static inline bool ParseAndAppendKeyValue(const char* arg,
                                          jxl::DecoderHints* out) {
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

static inline bool ParsePredictor(const char* arg, jxl::Predictor* out) {
  for (; *arg; arg++) {
    if (*arg >= '0' && *arg <= '9') {
      *out = static_cast<jxl::Predictor>(*arg - '0');
    } else {
      fprintf(stderr, "Invalid predictor value '%c', must be a digit.", *arg);
      return JXL_FAILURE("Args");
    }
  }
  return true;
}

static inline bool ParseString(const char* arg, std::string* out) {
  out->assign(arg);
  return true;
}

static inline bool ParseCString(const char* arg, const char** out) {
  *out = arg;
  return true;
}

static inline bool SetBooleanTrue(bool* out) {
  *out = true;
  return true;
}

static inline bool SetBooleanFalse(bool* out) {
  *out = false;
  return true;
}

static inline bool IncrementUnsigned(size_t* out) {
  (*out)++;
  return true;
}

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_ARGS_H_
