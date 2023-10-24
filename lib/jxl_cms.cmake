# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include(compatibility.cmake)
include(jxl_lists.cmake)

add_library(jxl_cms-obj OBJECT ${JPEGXL_INTERNAL_CMS_SOURCES})
target_compile_options(jxl_cms-obj PRIVATE "${JPEGXL_INTERNAL_FLAGS}")
set_target_properties(jxl_cms-obj PROPERTIES POSITION_INDEPENDENT_CODE ON)
jxl_link_libraries(jxl_cms-obj jxl_includes)
target_include_directories(jxl_cms-obj PRIVATE
  ${JXL_HWY_INCLUDE_DIRS}
)

add_dependencies(jxl_cms-obj jxl_cms_export)

target_include_directories(jxl_cms-obj PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")

set(JXL_CMS_OBJECTS $<TARGET_OBJECTS:jxl_cms-obj>)

if (JPEGXL_ENABLE_SKCMS)
  target_include_directories(jxl_cms-obj PRIVATE
    $<TARGET_PROPERTY:skcms-obj,INCLUDE_DIRECTORIES>
  )
  list(APPEND JXL_CMS_OBJECTS $<TARGET_OBJECTS:skcms-obj>)
  target_compile_definitions(jxl_cms-obj PRIVATE JPEGXL_ENABLE_SKCMS=1)
  if (JPEGXL_BUNDLE_SKCMS)
    target_compile_definitions(jxl_cms-obj PRIVATE JPEGXL_BUNDLE_SKCMS=1)
  else ()
    target_link_libraries(jxl_cms-obj INTERFACE skcms)
  endif ()
else ()
  target_include_directories(jxl_cms-obj PRIVATE
    $<TARGET_PROPERTY:lcms2,INCLUDE_DIRECTORIES>
  )
  target_link_libraries(jxl_cms-obj INTERFACE lcms2)
endif ()

add_library(jxl_cms-static STATIC ${JXL_CMS_OBJECTS})
if (NOT WIN32 OR MINGW)
  set_target_properties(jxl_cms-static PROPERTIES OUTPUT_NAME "jxl_cms")
endif()

if (BUILD_SHARED_LIBS)
  add_library(jxl_cms SHARED ${JXL_CMS_OBJECTS})
  target_link_libraries(jxl_cms PRIVATE hwy)
  set_target_properties(jxl_cms PROPERTIES
    VERSION ${JPEGXL_LIBRARY_VERSION}
    SOVERSION ${JPEGXL_LIBRARY_SOVERSION}
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
else()  # BUILD_SHARED_LIBS
  add_library(jxl_cms ALIAS jxl_cms-static)
endif()  # BUILD_SHARED_LIBS
