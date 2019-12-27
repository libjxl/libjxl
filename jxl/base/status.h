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

#ifndef JXL_BASE_STATUS_H_
#define JXL_BASE_STATUS_H_

// Error handling: Status return type + helper macros.

#include <stdio.h>
#include <stdlib.h>

#include "jxl/base/compiler_specific.h"

namespace jxl {

// Uncomment to abort when JXL_FAILURE is reached
// #define JXL_CRASH_ON_ERROR

#ifndef JXL_ENABLE_ASSERT
#define JXL_ENABLE_ASSERT 1
#endif

// Pass -DJXL_DEBUG_ON_ERROR at compile time to print debug messages when a
// function returns JXL_FAILURE or calls JXL_NOTIFY_ERROR. Note that this is
// irrelevant if you also pass -DJXL_CRASH_ON_ERROR.
#ifdef JXL_DEBUG_ON_ERROR
#undef JXL_DEBUG_ON_ERROR
#define JXL_DEBUG_ON_ERROR 1
#else  // JXL_DEBUG_ON_ERROR
#ifdef NDEBUG
#define JXL_DEBUG_ON_ERROR 0
#else  // JXL_DEBUG_ON_ERROR
#define JXL_DEBUG_ON_ERROR 1
#endif  // NDEBUG
#endif  // JXL_DEBUG_ON_ERROR

// The Verbose level for the library
#ifndef JXL_DEBUG_V_LEVEL
#define JXL_DEBUG_V_LEVEL 0
#endif  // JXL_DEBUG_V_LEVEL

// Print a debug message on standard error. You should use the JXL_DEBUG macro
// instead of calling Debug directly. This function returns false, so it can be
// used as a return value in JXL_FAILURE.
JXL_FORMAT(1, 2)
bool Debug(const char* format, ...);

// Print a debug message on standard error if "enabled" is true. "enabled" is
// normally a macro that evaluates to 0 or 1 at compile time, so the Debug
// function is never called and optimized out in release builds. Note that the
// arguments are compiled but not evaluated when enabled is false. The format
// string must be a explicit string in the call, for example:
//   JXL_DEBUG(JXL_DEBUG_MYMODULE, "my module message: %d", some_var);
// Add a header at the top of your module's .cc or .h file (depending on whether
// you have JXL_DEBUG calls from the .h as well) like this:
//   #ifndef JXL_DEBUG_MYMODULE
//   #define JXL_DEBUG_MYMODULE 0
//   #endif JXL_DEBUG_MYMODULE
#define JXL_DEBUG(enabled, format, ...)                         \
  do {                                                          \
    if (enabled) {                                              \
      ::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, \
                   ##__VA_ARGS__);                              \
    }                                                           \
  } while (0)

// JXL_DEBUG version that prints the debug message if the global verbose level
// defined at compile time by JXL_DEBUG_V_LEVEL is greater or equal than the
// passed level.
#define JXL_DEBUG_V(level, ...) \
  JXL_DEBUG(level <= JXL_DEBUG_V_LEVEL, __VA_ARGS__)

// Warnings (via JXL_WARNING) are enabled by default in debug builds (opt and
// debug).
#ifdef JXL_DEBUG_WARNING
#undef JXL_DEBUG_WARNING
#define JXL_DEBUG_WARNING 1
#else  // JXL_DEBUG_WARNING
#ifdef NDEBUG
#define JXL_DEBUG_WARNING 0
#else  // JXL_DEBUG_WARNING
#define JXL_DEBUG_WARNING 1
#endif  // NDEBUG
#endif  // JXL_DEBUG_WARNING
#define JXL_WARNING(...) JXL_DEBUG(JXL_DEBUG_WARNING, __VA_ARGS__)

// Exits the program after printing file/line plus a formatted string.
JXL_FORMAT(3, 4)
JXL_NORETURN bool Abort(const char* file, int line, const char* format, ...);

// Exits the program after printing file/line plus a formatted string.
#define JXL_ABORT(...) Abort(__FILE__, __LINE__, __VA_ARGS__)

// Does not guarantee running the code, use only for debug mode checks.
#if JXL_ENABLE_ASSERT
#define JXL_ASSERT(condition)                                    \
  do {                                                           \
    if (!(condition)) {                                          \
      ::jxl::Abort(__FILE__, __LINE__, "Assert %s", #condition); \
    }                                                            \
  } while (0)
#else
#define JXL_ASSERT(condition) \
  do {                        \
  } while (0)
#endif

// Same as above, but only runs in debug builds (builds where NDEBUG is not
// defined). This is useful for slower asserts that we want to run more rarely
// than usual. These will run on asan, msan and other debug builds, but not in
// opt or release.
#if !defined(NDEBUG) || defined(ADDRESS_SANITIZER) || \
    defined(MEMORY_SANITIZER) || defined(THREAD_SANITIZER)
#define JXL_DASSERT(condition)                                         \
  do {                                                                 \
    if (!(condition)) {                                                \
      ::jxl::Abort(__FILE__, __LINE__, "Debug Assert %s", #condition); \
    }                                                                  \
  } while (0)
#else
#define JXL_DASSERT(condition) \
  do {                         \
  } while (0)
#endif

// Always runs the condition, so can be used for non-debug calls.
#define JXL_CHECK(condition)                                    \
  do {                                                          \
    if (!(condition)) {                                         \
      ::jxl::Abort(__FILE__, __LINE__, "Check %s", #condition); \
    }                                                           \
  } while (0)

// Always runs the condition, so can be used for non-debug calls.
#define JXL_RETURN_IF_ERROR(condition) \
  do {                                 \
    if (!(condition)) return false;    \
  } while (0)

// Annotation for the location where an error condition is first noticed.
// Error codes are too unspecific to pinpoint the exact location, so we
// add a build flag that crashes and dumps stack at the actual error source.
#ifdef JXL_CRASH_ON_ERROR
#define JXL_NOTIFY_ERROR(...) \
  (void)::jxl::Abort(__FILE__, __LINE__, __VA_ARGS__)
#define JXL_FAILURE(...) ::jxl::Abort(__FILE__, __LINE__, __VA_ARGS__)
#else  // JXL_CRASH_ON_ERROR
#define JXL_NOTIFY_ERROR(...) JXL_DEBUG(JXL_DEBUG_ON_ERROR, __VA_ARGS__)
#define JXL_FAILURE(format, ...)                                               \
  ((JXL_DEBUG_ON_ERROR) &&                                                     \
   ::jxl::Debug(("%s:%d: " format "\n"), __FILE__, __LINE__, ##__VA_ARGS__) && \
   false)
#endif  // JXL_CRASH_ON_ERROR

// Drop-in replacement for bool that raises compiler warnings if not used
// after being returned from a function. Example:
// Status LoadFile(...) { return true; } is more compact than
// bool JXL_MUST_USE_RESULT LoadFile(...) { return true; }
class JXL_MUST_USE_RESULT Status {
 public:
  // We want implicit constructor from bool to allow returning "true" or "false"
  // on a function when using Status.
  Status(bool ok) : ok_(ok) {}  // NOLINT(google-explicit-constructor)

  // We also want implicit cast to bool to check for return values of functions.
  operator bool() const { return ok_; }  // NOLINT(google-explicit-constructor)

 private:
  bool ok_;
};

}  // namespace jxl

#endif  // JXL_BASE_STATUS_H_
