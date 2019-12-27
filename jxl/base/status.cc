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

#include "jxl/base/status.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <string>

#include "jxl/base/os_specific.h"

namespace jxl {

bool Debug(const char* format, ...) {
// Show the debug messages in debug or opt mode, not in release mode.
#if JXL_DEBUG_WARNING
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
#endif
  return false;
}

bool Abort(const char* file, int line, const char* format, ...) {
  char buf[2000];
  va_list args;
  va_start(args, format);
  vsnprintf(buf, sizeof(buf), format, args);
  va_end(args);

  const std::string call_stack;

  fprintf(stderr, "Abort at %s:%d: %s\n%s\n", file, line, buf,
          call_stack.c_str());
#if defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
  // If compiled with UBSAN, triggering an error gives us a call stack.
  printf("Deliberate ubsan div zero %zu\n", 10 / (TotalMemoryMiB() >> 30));
#endif
  abort();
}

}  // namespace jxl
