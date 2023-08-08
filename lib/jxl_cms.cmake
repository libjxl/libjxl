# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include(compatibility.cmake)
include(jxl_lists.cmake)


add_library(jxl_cms STATIC
  ${JPEGXL_INTERNAL_CMS_SOURCES}
)

target_compile_options(jxl_cms PRIVATE "${JPEGXL_INTERNAL_FLAGS}")
target_link_libraries(jxl_cms PRIVATE jxl_includes hwy)

target_include_directories(jxl_cms PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")

if (JPEGXL_ENABLE_SKCMS)
  target_include_directories(jxl_cms PRIVATE
    $<TARGET_PROPERTY:skcms,INCLUDE_DIRECTORIES>
  )
  target_sources(jxl_cms PRIVATE jxl/jxl_skcms.h)
  target_compile_definitions(jxl_cms PRIVATE JPEGXL_ENABLE_SKCMS=1)
  if (JPEGXL_BUNDLE_SKCMS)
    target_compile_definitions(jxl_cms PRIVATE JPEGXL_BUNDLE_SKCMS=1)
    target_sources(jxl_cms PRIVATE $<TARGET_OBJECTS:skcms-obj>)
  else ()
    target_link_libraries(jxl_cms PRIVATE skcms)
  endif ()
else ()
  target_include_directories(jxl_cms PRIVATE
    $<TARGET_PROPERTY:lcms2,INCLUDE_DIRECTORIES>
  )
  target_link_libraries(jxl_cms PRIVATE lcms2)
endif ()
