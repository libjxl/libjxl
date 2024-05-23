# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include(jxl_lists.cmake)

# Headers for exporting/importing public headers
include(GenerateExportHeader)

add_library(jxl_gain_map
  ${JPEGXL_INTERNAL_GAIN_MAP_SOURCES}
)

target_compile_options(jxl_gain_map PRIVATE "${JPEGXL_INTERNAL_FLAGS}")
set_target_properties(jxl_gain_map PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN 1)
target_link_libraries(jxl_gain_map PUBLIC jxl_base)
target_include_directories(jxl_gain_map PRIVATE
  # TODO: check if those are actually needed after implementing jxl_gain_map.cc
  ${JXL_HWY_INCLUDE_DIRS}
)
generate_export_header(jxl_gain_map
  BASE_NAME JXL_GAIN_MAP
  EXPORT_FILE_NAME include/jxl/jxl_gain_map_export.h)
target_include_directories(jxl_gain_map BEFORE PUBLIC
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>")

set(JPEGXL_GAIN_MAP_LIBRARY_REQUIRES "")

target_link_libraries(jxl_gain_map PRIVATE hwy)

set_target_properties(jxl_gain_map PROPERTIES
        VERSION ${JPEGXL_LIBRARY_VERSION}
        SOVERSION ${JPEGXL_LIBRARY_SOVERSION})

install(TARGETS jxl_gain_map
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# TODO: check if those are actually needed after implementing jxl_gain_map.cc
if (BUILD_SHARED_LIBS)
  set(JPEGXL_REQUIRES_TYPE "Requires.private")
  set(JPEGXL_GAIN_MAP_PRIVATE_LIBS "-lm ${PKGCONFIG_CXX_LIB}")
else()
  set(JPEGXL_REQUIRES_TYPE "Requires")
  set(JPEGXL_GAIN_MAP_PRIVATE_LIBS "-lm ${PKGCONFIG_CXX_LIB}")
endif()

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/jxl/libjxl_gain_map.pc.in"
               "libjxl_gain_map.pc" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libjxl_gain_map.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
