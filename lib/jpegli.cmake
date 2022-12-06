# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set(JPEGLI_MAJOR_VERSION 62)
set(JPEGLI_MINOR_VERSION 3)
set(JPEGLI_PATCH_VERSION 0)
set(JPEGLI_LIBRARY_VERSION
    "${JPEGLI_MAJOR_VERSION}.${JPEGLI_MINOR_VERSION}.${JPEGLI_PATCH_VERSION}"
)

set(JPEGLI_LIBRARY_SOVERSION "${JPEGLI_MAJOR_VERSION}")

set(JPEGLI_INTERNAL_SOURCES
  jpegli/color_transform.h
  jpegli/color_transform.cc
  jpegli/decode_api.cc
  jpegli/decode_internal.h
  jpegli/decode_marker.h
  jpegli/decode_marker.cc
  jpegli/decode_scan.h
  jpegli/decode_scan.cc
  jpegli/error.h
  jpegli/error.cc
  jpegli/huffman.h
  jpegli/huffman.cc
  jpegli/idct.h
  jpegli/idct.cc
  jpegli/memory_manager.h
  jpegli/render.h
  jpegli/render.cc
  jpegli/source_manager.h
  jpegli/source_manager.cc
  jpegli/upsample.h
  jpegli/upsample.cc
)

set(JPEGLI_INTERNAL_LIBS
  hwy
  Threads::Threads
  ${ATOMICS_LIBRARIES}
)

set(OBJ_COMPILE_DEFINITIONS
  JPEGLI_MAJOR_VERSION=${JPEGLI_MAJOR_VERSION}
  JPEGLI_MINOR_VERSION=${JPEGLI_MINOR_VERSION}
  JPEGLI_PATCH_VERSION=${JPEGLI_PATCH_VERSION}
  # Used to determine if we are building the library when defined or just
  # including the library when not defined. This is public so libjpeg shared
  # library gets this define too.
  JPEGLI_INTERNAL_LIBRARY_BUILD
)

add_library(jpegli-obj OBJECT ${JPEGLI_INTERNAL_SOURCES})
target_compile_options(jpegli-obj PRIVATE ${JPEGXL_INTERNAL_FLAGS})
target_compile_options(jpegli-obj PUBLIC ${JPEGXL_COVERAGE_FLAGS})
set_property(TARGET jpegli-obj PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(jpegli-obj PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>"
  "$<BUILD_INTERFACE:$<TARGET_PROPERTY:hwy,INTERFACE_INCLUDE_DIRECTORIES>>"
)
target_compile_definitions(jpegli-obj PUBLIC
  ${OBJ_COMPILE_DEFINITIONS}
)

set(JPEGLI_INTERNAL_OBJECTS $<TARGET_OBJECTS:jpegli-obj>)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/jpegli)
add_library(jpeg SHARED ${JPEGLI_INTERNAL_OBJECTS})
strip_static(JPEGLI_INTERNAL_SHARED_LIBS JPEGLI_INTERNAL_LIBS)
target_link_libraries(jpeg PUBLIC ${JPEGXL_COVERAGE_FLAGS})
target_link_libraries(jpeg PRIVATE ${JPEGLI_INTERNAL_SHARED_LIBS})
set_target_properties(jpeg PROPERTIES
  VERSION ${JPEGLI_LIBRARY_VERSION}
  SOVERSION ${JPEGLI_LIBRARY_SOVERSION}
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/jpegli"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/jpegli")

# Add a jpeg.version file as a version script to tag symbols with the
# appropriate version number.
set_target_properties(jpeg PROPERTIES
  LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/jpegli/jpeg.version)
set_property(TARGET jpeg APPEND_STRING PROPERTY
  LINK_FLAGS " -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/jpegli/jpeg.version")

# This hides the default visibility symbols from static libraries bundled into
# the shared library. In particular this prevents exposing symbols from hwy
# in the shared library.
if(LINKER_SUPPORT_EXCLUDE_LIBS)
  set_property(TARGET jpeg APPEND_STRING PROPERTY
    LINK_FLAGS " ${LINKER_EXCLUDE_LIBS_FLAG}")
endif()

if(BUILD_TESTING)
set(TEST_FILES
  jpegli/decode_api_test.cc
)

# Individual test binaries:
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
foreach (TESTFILE IN LISTS TEST_FILES)
  # The TESTNAME is the name without the extension or directory.
  get_filename_component(TESTNAME ${TESTFILE} NAME_WE)
  add_executable(${TESTNAME} ${TESTFILE} jpegli/test_utils.h)
  target_compile_options(${TESTNAME} PRIVATE
    ${JPEGXL_INTERNAL_FLAGS}
    # Add coverage flags to the test binary so code in the private headers of
    # the library is also instrumented when running tests that execute it.
    ${JPEGXL_COVERAGE_FLAGS}
  )
  target_compile_definitions(${TESTNAME} PRIVATE
    -DTEST_DATA_PATH="${JPEGXL_TEST_DATA_PATH}")
  target_include_directories(${TESTNAME} PRIVATE "${PROJECT_SOURCE_DIR}")
  target_link_libraries(${TESTNAME}
    hwy
    jpeg
    gmock
    GTest::GTest
    GTest::Main
  )
  # Output test targets in the test directory.
  set_target_properties(${TESTNAME} PROPERTIES PREFIX "tests/")
  if (WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_target_properties(${TESTNAME} PROPERTIES COMPILE_FLAGS "-Wno-error")
  endif ()
  if(CMAKE_VERSION VERSION_LESS "3.10.3")
    gtest_discover_tests(${TESTNAME} TIMEOUT 240)
  else ()
    gtest_discover_tests(${TESTNAME} DISCOVERY_TIMEOUT 240)
  endif ()
endforeach ()
endif()
