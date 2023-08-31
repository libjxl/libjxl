# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include(compatibility.cmake)
include(jxl_lists.cmake)

# Headers for exporting/importing public headers
include(GenerateExportHeader)

# CMake does not allow generate_export_header for INTERFACE library, so we
# add this stub library just for file generation.
add_library(jxl_cms_export OBJECT "jxl/jxl_cms.h")
set_target_properties(jxl_cms_export PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  DEFINE_SYMBOL JXL_INTERNAL_LIBRARY_BUILD
  LINKER_LANGUAGE CXX
)
generate_export_header(jxl_cms_export
  BASE_NAME JXL_CMS
  EXPORT_FILE_NAME include/jxl/jxl_cms_export.h)


add_library(jxl_cms OBJECT
  ${JPEGXL_INTERNAL_CMS_SOURCES}
)

target_compile_options(jxl_cms PRIVATE "${JPEGXL_INTERNAL_FLAGS}")
set_target_properties(jxl_cms PROPERTIES POSITION_INDEPENDENT_CODE ON)
jxl_link_libraries(jxl_cms jxl_includes)
target_link_libraries(jxl_cms PRIVATE hwy)

add_dependencies(jxl_cms jxl_cms_export)

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
    target_sources(jxl_cms PUBLIC $<TARGET_OBJECTS:skcms-obj>)
  else ()
    target_link_libraries(jxl_cms INTERFACE skcms)
  endif ()
else ()
  target_include_directories(jxl_cms PRIVATE
    $<TARGET_PROPERTY:lcms2,INCLUDE_DIRECTORIES>
  )
  target_link_libraries(jxl_cms INTERFACE lcms2)
endif ()
