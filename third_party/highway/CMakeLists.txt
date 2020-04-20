# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.2)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

option(HWY_USE_GTEST "Links tests with GoogleTest")
if(HWY_USE_GTEST)
  find_package(GTest)
endif()

include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
   "int main() {
      #if !defined(__EMSCRIPTEN__)
      static_assert(false, \"__EMSCRIPTEN__ is not defined\");
      #endif
      return 0;
    }"
  HWY_EMSCRIPTEN
)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(HWY_SOURCES
    hwy/begin_target-inl.h
    hwy/end_target-inl.h
    hwy/foreach_target.h
    hwy/highway.h
    hwy/interface.h
    hwy/ops/arm_neon-inl.h
    hwy/ops/scalar-inl.h
    hwy/ops/wasm_128-inl.h
    hwy/ops/x86_128-inl.h
    hwy/ops/x86_256-inl.h
    hwy/ops/x86_512-inl.h
    hwy/supported_targets.cc
    hwy/tests/test_util-inl.h
    hwy/tests/test_util.h
)

set(HWY_FLAGS
  # Avoid changing binaries based on the current time and date.
  -Wno-builtin-macro-redefined
  -D__DATE__="redacted"
  -D__TIMESTAMP__="redacted"
  -D__TIME__="redacted"

  # Optimizations
  -fmerge-all-constants

  # Warnings
  -Wall
  -Wextra
  -Wformat-security
  -Wno-unused-function
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wvla
)

if(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  list(APPEND HWY_FLAGS
    -Wc++2a-extensions
    -Wfloat-overflow-conversion
    -Wfloat-zero-conversion
    -Wfor-loop-analysis
    -Wgnu-redeclared-enum
    -Winfinite-recursion
    -Wself-assign
    -Wstring-conversion
    -Wtautological-overlap-compare
    -Wthread-safety-analysis
    -Wundefined-func-template

    -fno-cxx-exceptions
    -fno-slp-vectorize
    -fno-vectorize

    # Use color in messages
    -fdiagnostics-show-option -fcolor-diagnostics
  )
endif()

if (WIN32)
  list(APPEND HWY_FLAGS
    -Wno-c++98-compat-pedantic
    -Wno-cast-align
    -Wno-double-promotion
    -Wno-float-equal
    -Wno-format-nonliteral
    -Wno-global-constructors
    -Wno-language-extension-token
    -Wno-missing-prototypes
    -Wno-shadow
    -Wno-shadow-field-in-constructor
    -Wno-sign-conversion
    -Wno-unused-member-function
    -Wno-unused-template
    -Wno-used-but-marked-unused
    -Wno-zero-as-null-pointer-constant
  )
else()
  list(APPEND HWY_FLAGS
    -fmath-errno
    -fno-exceptions
  )
endif()

if(HWY_USE_GTEST)
  list(APPEND HWY_FLAGS -DHWY_USE_GTEST)
endif()

add_library(hwy STATIC ${HWY_SOURCES})

target_compile_options(hwy PRIVATE ${HWY_FLAGS})
target_include_directories(hwy PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# -------------------------------------------------------- hwy_list_targets
# Generate a tool to pint the compiled-in targets as defined by the current
# flags. This tool will print to stderr at build time, right after building
# hwy.
add_executable(hwy_list_targets hwy/tests/list_targets.cc)
target_compile_options(hwy_list_targets PRIVATE ${HWY_FLAGS})
target_include_directories(hwy_list_targets PRIVATE
  $<TARGET_PROPERTY:hwy,INCLUDE_DIRECTORIES>)
add_custom_command(TARGET hwy POST_BUILD
    COMMAND $<TARGET_FILE:hwy_list_targets> || (exit 0))

# -------------------------------------------------------- Examples

add_executable(skeleton hwy/examples/skeleton_main.cc)
target_sources(skeleton PRIVATE
    hwy/examples/skeleton-inl.h
    hwy/examples/skeleton.cc
    hwy/examples/skeleton.h
    hwy/examples/skeleton_shared.h)
target_compile_options(skeleton PRIVATE ${HWY_FLAGS})
target_link_libraries(skeleton hwy)
set_target_properties(skeleton PROPERTIES PREFIX "tests/")

# -------------------------------------------------------- Tests

enable_testing()

set(HWY_TEST_FILES
  hwy/tests/arithmetic_test.cc
  hwy/tests/compare_test.cc
  hwy/tests/convert_test.cc
  hwy/tests/hwy_test.cc
  hwy/tests/logical_test.cc
  hwy/tests/memory_test.cc
  hwy/tests/swizzle_test.cc
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
foreach (TESTFILE IN LISTS HWY_TEST_FILES)
  # The TESTNAME is the name without the extension or directory.
  get_filename_component(TESTNAME ${TESTFILE} NAME_WE)
  add_executable(${TESTNAME} ${TESTFILE})
  target_compile_options(${TESTNAME} PRIVATE ${HWY_FLAGS})

  target_link_libraries(${TESTNAME} hwy)
  # Output test targets in the test directory.
  set_target_properties(${TESTNAME} PROPERTIES PREFIX "tests/")

  if(HWY_USE_GTEST)
    target_link_libraries(${TESTNAME} gtest gtest_main)
  endif()

  # gtest_discover_tests requires CMake 3.10 and compiling with gtest.
  # gtest_add_tests runs cmake every time a test file changes and requires
  # find_package(GTest) or include(GoogleTest). We know there is only a
  # single test case (due to runtime dispatch), so add manually.
  add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
endforeach ()
