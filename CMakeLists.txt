# Copyright (c) the JPEG XL Project
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

# Ubuntu bionic ships with cmake 3.10.
cmake_minimum_required(VERSION 3.10)

# Set compilers to default to Clang is not set.
if ("$ENV{CC}" STREQUAL "")
  set(ENV{CC} clang)
endif()
if ("$ENV{CXX}" STREQUAL "")
  set(ENV{CXX} clang++)
endif()

# Honor VISIBILITY_INLINES_HIDDEN on all types of targets.
if(POLICY CMP0063)
  cmake_policy(SET CMP0063 NEW)
endif()
# Pass CMAKE_EXE_LINKER_FLAGS to CC and CXX compilers when testing if they work.
if(POLICY CMP0065)
  cmake_policy(SET CMP0065 NEW)
endif()

project(JPEGXL LANGUAGES C CXX)

include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
   "int main() {
      #if !defined(__EMSCRIPTEN__)
      static_assert(false, \"__EMSCRIPTEN__ is not defined\");
      #endif
      return 0;
    }"
  JPEGXL_EMSCRIPTEN
)

message(STATUS "CMAKE_SYSTEM_PROCESSOR is ${CMAKE_SYSTEM_PROCESSOR}")
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsanitize=fuzzer-no-link" CXX_FUZZERS_SUPPORTED)
set(CXX_PIC_FLAGS "\
  -Xclang -mrelocation-model -Xclang pic \
  -Xclang -pic-level -Xclang 2")
check_cxx_compiler_flag("${CXX_PIC_FLAGS}" CXX_PIC_FLAGS_SUPPORTED)
check_cxx_compiler_flag("-Xclang -mconstructor-aliases" CXX_CONSTRUCTOR_ALIASES_SUPPORTED)

### Project build options:
if(${CXX_FUZZERS_SUPPORTED})
  # Enabled by default except on arm64, Windows and Apple builds.
  set(ENABLE_FUZZERS_DEFAULT true)
endif()
if(NOT APPLE AND NOT WIN32 AND NOT HAIKU AND ${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  # Enabled by default except on Windows and Apple builds.
  set(ENABLE_TCMALLOC_DEFAULT true)
endif()

set(JPEGXL_ENABLE_FUZZERS ${ENABLE_FUZZERS_DEFAULT} CACHE BOOL
    "Build JPEGXL fuzzer targets")
set(JPEGXL_ENABLE_BENCHMARK true CACHE BOOL
    "Build JPEGXL benchmark tools")
set(JPEGXL_ENABLE_SJPEG true CACHE BOOL
    "Build JPEGXL with support for encoding with sjpeg")
set(JPEGXL_ENABLE_SKCMS false CACHE BOOL
    "Where possible, use skcms instead of lcms2")
set(JPEGXL_ENABLE_VIEWERS true CACHE BOOL
    "Build JPEGXL viewer tools for evaluation")
set(JPEGXL_ENABLE_TCMALLOC ${ENABLE_TCMALLOC_DEFAULT} CACHE BOOL
    "Build JPEGXL using gperftools (tcmalloc) allocator.")
set(JPEGXL_ENABLE_PLUGINS true CACHE BOOL
    "Build third-party plugings to support JPEG XL in other applications.")
set(JPEGXL_ENABLE_COVERAGE false CACHE BOOL
    "Enable code coverage tracking for libjpegxl. This also enables debug and disables optimizations.")

# Require clang for building.
if(NOT ${CMAKE_C_COMPILER_ID} MATCHES "Clang" OR
   NOT ${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  message(FATAL_ERROR "${CMAKE_CXX_COMPILER_ID} compiler is not supported.\n"
    "Use clang instead:\n  CC=clang CXX=clang++ cmake ..")
endif()
if (${CMAKE_C_COMPILER_VERSION} VERSION_LESS 6 OR
    ${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 6)
  message(FATAL_ERROR
    "Minimum Clang version required is Clang 6, please update.")
endif()

# CMAKE_EXPORT_COMPILE_COMMANDS is used to generate the compilation database
# used by clang-tidy.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Global compiler flags for all targets here and in subdirectories.
add_definitions(
  # Avoid changing the binary based on the current time and date.
  -D__DATE__="redacted"
  -D__TIMESTAMP__="redacted"
  -D__TIME__="redacted"
)

# In CMake before 3.12 it is problematic to pass repeated flags like -Xclang.
# For this reason we place them in CMAKE_CXX_FLAGS instead.
# See https://gitlab.kitware.com/cmake/cmake/issues/15826

# Machine flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
  -Xclang -mrelax-all \
  -Xclang -mpie-copy-relocations \
  -funwind-tables")
if ("${CXX_PIC_FLAGS_SUPPORTED}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_PIC_FLAGS}")
endif()
if ("${CXX_CONSTRUCTOR_ALIASES_SUPPORTED}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -mconstructor-aliases")
endif()

if(WIN32)
# Not supported by clang-cl, but frame pointers are default on Windows
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-omit-frame-pointer")
endif()

# CPU flags - remove once we have NEON dynamic dispatch
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
# Older Clang requires at least armv7-a and GCC requires the latter two,
# otherwise __ARM_NEON is undefined. Keep -mcpu for now, might help with speed.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard")
endif()

# Force build with optimizations in release mode.
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")

add_compile_options(
  # Ignore this to allow redefining __DATE__ and others.
  -Wno-builtin-macro-redefined

  # Global warning settings.
  -Wall
)

if (NOT WIN32)
add_compile_options(-Werror)
endif ()

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(PkgConfig)
add_subdirectory(third_party)

set(THREADS_PREFER_PTHREAD_FLAG YES)
find_package(Threads REQUIRED)

# Enable tests regardless of where are they defined.
enable_testing()
include(CTest)

# The jpegxl library definition. This also defines the internal list of flags we
# use in our project (library, tests and tools): JPEGXL_INTERNAL_FLAGS.
include(jpegxl.cmake)

# Other libraries outside the core jpegxl library.
include(jpegxl_extras.cmake)
include(jpegxl_threads.cmake)

# Documentation generated by Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
set(DOXYGEN_GENERATE_HTML "YES")
set(DOXYGEN_GENERATE_XML "NO")
set(DOXYGEN_STRIP_FROM_PATH "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(DOXYGEN_USE_MDFILE_AS_MAINPAGE "README.md")
set(DOXYGEN_WARN_AS_ERROR "YES")
doxygen_add_docs(doc
  "${CMAKE_CURRENT_SOURCE_DIR}/include/jpegxl"
  "${CMAKE_CURRENT_SOURCE_DIR}/doc/api.txt"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Generating C API documentation")
else()
# Create a "doc" target for compatibility since "doc" is not otherwise added to
# the build when doxygen is not installed.
add_custom_target(doc false
  COMMENT "Error: Can't generate doc since Doxygen not installed.")
endif() # DOXYGEN_FOUND

# Plugins for third-party software
if (${JPEGXL_ENABLE_PLUGINS})
add_subdirectory(plugins)
endif ()

# Binary tools
add_subdirectory(tools)

# Unittests
cmake_policy(SET CMP0057 NEW)  # https://gitlab.kitware.com/cmake/cmake/issues/18198
include(GoogleTest)

# Tests for the jpegxl library.
include(jpegxltests.cmake)
