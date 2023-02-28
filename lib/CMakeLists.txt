# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set(JPEGXL_MAJOR_VERSION 0)
set(JPEGXL_MINOR_VERSION 9)
set(JPEGXL_PATCH_VERSION 0)
set(JPEGXL_LIBRARY_VERSION
    "${JPEGXL_MAJOR_VERSION}.${JPEGXL_MINOR_VERSION}.${JPEGXL_PATCH_VERSION}")

# This is the library API/ABI compatibility version. Changing this value makes
# the shared library incompatible with previous version. A program linked
# against this shared library SOVERSION will not run with an older SOVERSION.
# It is important to update this value when making incompatible API/ABI changes
# so that programs that depend on libjxl can update their dependencies. Semantic
# versioning allows 0.y.z to have incompatible changes in minor versions.
set(JPEGXL_SO_MINOR_VERSION 9)
if (JPEGXL_MAJOR_VERSION EQUAL 0)
  set(JPEGXL_LIBRARY_SOVERSION
      "${JPEGXL_MAJOR_VERSION}.${JPEGXL_SO_MINOR_VERSION}")
else()
  set(JPEGXL_LIBRARY_SOVERSION "${JPEGXL_MAJOR_VERSION}")
endif()


# List of warning and feature flags for our library and tests.
if (MSVC)
  set(JPEGXL_INTERNAL_FLAGS
    # TODO(janwas): add flags
  )
else ()
  set(JPEGXL_INTERNAL_FLAGS
    # F_FLAGS
    -fmerge-all-constants
    -fno-builtin-fwrite
    -fno-builtin-fread

    # WARN_FLAGS
    -Wall
    -Wextra
    -Wc++11-compat
    -Warray-bounds
    -Wformat-security
    -Wimplicit-fallthrough
    -Wno-register  # Needed by public headers in lcms
    -Wno-unused-function
    -Wno-unused-parameter
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wvla
  )

  # Warning flags supported by clang.
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(APPEND JPEGXL_INTERNAL_FLAGS
      -Wdeprecated-increment-bool
      # TODO(deymo): Add -Wextra-semi once we update third_party/highway.
      # -Wextra-semi
      -Wfloat-overflow-conversion
      -Wfloat-zero-conversion
      -Wfor-loop-analysis
      -Wgnu-redeclared-enum
      -Winfinite-recursion
      -Wliteral-conversion
      -Wno-c++98-compat
      -Wno-unused-command-line-argument
      -Wprivate-header
      -Wself-assign
      -Wstring-conversion
      -Wtautological-overlap-compare
      -Wthread-safety-analysis
      -Wundefined-func-template
      -Wunreachable-code
      -Wunused-comparison
    )
    if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
      list(APPEND HWY_FLAGS -Wc++2a-extensions)
    endif()
  endif()  # Clang

  if (WIN32)
    list(APPEND JPEGXL_INTERNAL_FLAGS
      -Wno-cast-align
      -Wno-double-promotion
      -Wno-float-equal
      -Wno-format-nonliteral
      -Wno-shadow
      -Wno-sign-conversion
      -Wno-zero-as-null-pointer-constant
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      list(APPEND JPEGXL_INTERNAL_FLAGS
        -Wno-used-but-marked-unused
        -Wno-unused-template
        -Wno-unused-member-function
        -Wno-shadow-field-in-constructor
        -Wno-language-extension-token
        -Wno-global-constructors
        -Wno-c++98-compat-pedantic
      )
    endif()  # Clang
  else()  # WIN32
    list(APPEND JPEGXL_INTERNAL_FLAGS
      -fsized-deallocation
      -fno-exceptions

      # Language flags
      -fmath-errno
    )

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      list(APPEND JPEGXL_INTERNAL_FLAGS
        -fnew-alignment=8
        -fno-cxx-exceptions
        -fno-slp-vectorize
        -fno-vectorize

        -disable-free
        -disable-llvm-verifier
      )
    endif()  # Clang
  endif()  # WIN32
endif()  #!MSVC

# strips the -static suffix from all the elements in LIST
function(strip_static OUTPUT_VAR LIB_LIST)
  foreach(lib IN LISTS ${LIB_LIST})
    string(REGEX REPLACE "-static$" "" lib "${lib}")
    list(APPEND out_list "${lib}")
  endforeach()
  set(${OUTPUT_VAR} ${out_list} PARENT_SCOPE)
endfunction()

# The jxl library definition.
include(jxl.cmake)

# Other libraries outside the core jxl library.
if(JPEGXL_ENABLE_TOOLS)
  include(jxl_extras.cmake)
endif()
include(jxl_threads.cmake)
find_package(JPEG)
if (JPEG_FOUND AND JPEGXL_ENABLE_JPEGLI)
  include(jpegli.cmake)
endif()

# Install all the library headers from the source and the generated ones. There
# is no distinction on which libraries use which header since it is expected
# that all developer libraries are available together at build time.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/jxl
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/jxl
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

# Destination of CMake files
if(NOT DEFINED JPEGXL_INSTALL_CMAKEDIR)
  set(JPEGXL_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/JPEGXL")
endif()

# Find modules also need to be installed for non-vendored libraries
if(JPEGXL_FORCE_SYSTEM_BROTLI)
  list(APPEND JPEGXL_FIND_MODULES "${PROJECT_SOURCE_DIR}/cmake/FindBrotli.cmake")
endif()
if(JPEGXL_FORCE_SYSTEM_HWY)
  list(APPEND JPEGXL_FIND_MODULES "${PROJECT_SOURCE_DIR}/cmake/FindHWY.cmake")
endif()
if(JPEGXL_FORCE_SYSTEM_LCMS2 AND NOT JPEGXL_ENABLE_SKCMS)
  list(APPEND JPEGXL_FIND_MODULES "${PROJECT_SOURCE_DIR}/cmake/FindLCMS2.cmake")
endif()

# Prepare and install the CMake config files
include(CMakePackageConfigHelpers)
configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/JPEGXLConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/JPEGXLConfig.cmake"
  INSTALL_DESTINATION "${JPEGXL_INSTALL_CMAKEDIR}")

write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/JPEGXLConfigVersion.cmake"
  VERSION "${JPEGXL_LIBRARY_VERSION}"
  COMPATIBILITY SameMinorVersion)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/JPEGXLConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/JPEGXLConfigVersion.cmake"
  ${JPEGXL_FIND_MODULES}
  DESTINATION "${JPEGXL_INSTALL_CMAKEDIR}")

# Export the library targets
install(EXPORT JxlTargets
  FILE "JPEGXLTargets.cmake"
  NAMESPACE JPEGXL::
  DESTINATION "${JPEGXL_INSTALL_CMAKEDIR}")

# If vendored dependencies were used, they also need to be exported
if(HAS_EXPORTED_VENDORED_DEPS)
  install(EXPORT JxlDeps 
    FILE "JPEGXLDeps.cmake"
    NAMESPACE JPEGXLDeps::
    DESTINATION "${JPEGXL_INSTALL_CMAKEDIR}")
endif()

if(BUILD_TESTING)
  cmake_policy(SET CMP0057 NEW)  # https://gitlab.kitware.com/cmake/cmake/issues/18198
  include(GoogleTest)
endif()

# Tests for the jxl library.
include(jxl_tests.cmake)

if(BUILD_TESTING)
  # Google benchmark for the jxl library
  include(jxl_benchmark.cmake)
endif()
