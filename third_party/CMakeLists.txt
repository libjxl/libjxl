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

# Enable tests in third_party/ as well.
enable_testing()
include(CTest)

# Add GTest from source and alias it to what the find_package(GTest) workflow
# defines. Omitting googletest/ directory would require it to be available in
# the base system instead, but it would work just fine. This makes packages
# using GTest and calling find_package(GTest) actually work.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/googletest/CMakeLists.txt")
  add_subdirectory(googletest EXCLUDE_FROM_ALL)

  set(GTEST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/googletest/googletest")
  set(GTEST_INCLUDE_DIR "$<TARGET_PROPERTY:INCLUDE_DIRECTORIES,gtest>"
      CACHE STRING "")
  set(GTEST_LIBRARY "$<TARGET_FILE:gtest>")
  set(GTEST_MAIN_LIBRARY "$<TARGET_FILE:gtest_main>")
  add_library(GTest::GTest ALIAS gtest)
  add_library(GTest::Main ALIAS gtest_main)

  set_target_properties(gtest PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  set_target_properties(gmock PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  set_target_properties(gtest_main PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  set_target_properties(gmock_main PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

  # googletest doesn't compile clean with clang-cl (-Wundef)
  if (WIN32 AND ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    set_target_properties(gtest PROPERTIES COMPILE_FLAGS "-Wno-error")
    set_target_properties(gmock PROPERTIES COMPILE_FLAGS "-Wno-error")
    set_target_properties(gtest_main PROPERTIES COMPILE_FLAGS "-Wno-error")
    set_target_properties(gmock_main PROPERTIES COMPILE_FLAGS "-Wno-error")
  endif ()
endif()
find_package(GTest REQUIRED)

# Highway
add_subdirectory(highway)

# lodepng
if( NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lodepng/.git" )
  message(SEND_ERROR "Please run git submodule update --init")
endif()
include(lodepng.cmake)

# brotli
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/brotli/c/include/brotli/decode.h")
  message(SEND_ERROR "Please run git submodule update --init")
endif ()
set(BROTLI_DISABLE_TESTS ON CACHE STRING "Disable Brotli tests")
add_subdirectory(brotli EXCLUDE_FROM_ALL)

# *cms
if (JPEGXL_ENABLE_SKCMS)
  if( NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/skcms/.git" )
    message(SEND_ERROR "Please run git submodule update --init")
  endif()
  include(skcms.cmake)
else ()
  if( NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lcms/.git" )
    message(SEND_ERROR "Please run git submodule update --init")
  endif()
  include(lcms2.cmake)
endif()

# brunsli
# depends both on brotli and googletest, so should be added after them.
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/brunsli/c/include/brunsli/brunsli_decode.h")
  message(SEND_ERROR "Please run git submodule update --init")
endif ()
add_subdirectory(brunsli EXCLUDE_FROM_ALL)

# sjpeg
if (JPEGXL_ENABLE_SJPEG)
  if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sjpeg/.git")
    message(SEND_ERROR "Please run git submodule update --init")
  endif()
  include(sjpeg.cmake)
endif ()

# std::thread support when cross-compiling with mingw.
# jpeglx lib uses <mutex> and therefore needs mingw-std-threads when compiling
# with mingw.
if(MINGW)
option(MINGW_STDTHREADS_GENERATE_STDHEADERS "" ON)
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/mingw-std-threads/mingw.thread.h")
  message(SEND_ERROR "Please run git submodule update --init")
endif ()
add_subdirectory(mingw-std-threads)
endif()
