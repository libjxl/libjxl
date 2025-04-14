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

set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")

if(BUILD_TESTING)
# Add GTest from source and alias it to what the find_package(GTest) workflow
# defines. Omitting googletest/ directory would require it to be available in
# the base system instead, but it would work just fine. This makes packages
# using GTest and calling find_package(GTest) actually work.
if (EXISTS "${SOURCE_DIR}/googletest/CMakeLists.txt" AND
    NOT JPEGXL_FORCE_SYSTEM_GTEST)
  set(BUILD_GMOCK OFF CACHE INTERNAL "")
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  include(GoogleTest)

  set(GTEST_ROOT "${SOURCE_DIR}/googletest/googletest")
  set(GTEST_INCLUDE_DIR "$<TARGET_PROPERTY:INCLUDE_DIRECTORIES,gtest>"
      CACHE STRING "")
  set(GTEST_LIBRARY "$<TARGET_FILE:gtest>")
  set(GTEST_MAIN_LIBRARY "$<TARGET_FILE:gtest_main>")

  set_target_properties(gtest PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
  set_target_properties(gtest_main PROPERTIES POSITION_INDEPENDENT_CODE TRUE)

  get_target_property(GOOGLETEST_VERSION gtest VERSION)
  message(STATUS "Using GTest from submodule: ${GOOGLETEST_VERSION}")

  # googletest doesn't compile clean with clang-cl (-Wundef)
  if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_target_properties(gtest PROPERTIES COMPILE_FLAGS "-Wno-error")
    set_target_properties(gtest_main PROPERTIES COMPILE_FLAGS "-Wno-error")
  endif ()
  configure_file("${SOURCE_DIR}/googletest/LICENSE"
                 ${PROJECT_BINARY_DIR}/LICENSE.googletest COPYONLY)
else()
  if(JPEGXL_DEP_LICENSE_DIR)
    configure_file("${JPEGXL_DEP_LICENSE_DIR}/googletest/copyright"
                   ${PROJECT_BINARY_DIR}/LICENSE.googletest COPYONLY)
  endif()  # JPEGXL_DEP_LICENSE_DIR
  find_package(GTest REQUIRED)
  include(GoogleTest)
  set_target_properties(GTest::GTest PROPERTIES IMPORTED_GLOBAL TRUE)
  set_target_properties(GTest::Main PROPERTIES IMPORTED_GLOBAL TRUE)
  add_library(gtest ALIAS GTest::GTest)
  add_library(gtest_main ALIAS GTest::Main)
endif()

endif()  # BUILD_TESTING
