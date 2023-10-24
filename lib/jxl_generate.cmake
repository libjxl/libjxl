# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Install all the library headers from the source and the generated ones. There
# is no distinction on which libraries use which header since it is expected
# that all developer libraries are available together at build time.

# Headers for exporting/importing public headers
include(GenerateExportHeader)

# VERSION

configure_file("jxl/version.h.in" "include/jxl/version.h")

# EXPORTS

# CMake does not allow generate_export_header for INTERFACE library, so we
# add this stub OBJECT library just for file generation.
function(add_export_library NAME)
  add_library(${NAME} OBJECT "include/jxl/version.h")
  set_target_properties(${NAME} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN 1
    DEFINE_SYMBOL JXL_INTERNAL_LIBRARY_BUILD
    LINKER_LANGUAGE CXX
  )
endfunction()

add_export_library(jxl_export)
generate_export_header(jxl_export
  BASE_NAME JXL
  EXPORT_FILE_NAME "include/jxl/jxl_export.h")
  add_export_library(jxl_cms_export)
generate_export_header(jxl_cms_export
  BASE_NAME JXL_CMS
  EXPORT_FILE_NAME "include/jxl/jxl_cms_export.h")
add_export_library(jxl_threads_export)
generate_export_header(jxl_threads_export
  BASE_NAME JXL_THREADS
  EXPORT_FILE_NAME "include/jxl/jxl_threads_export.h")

# PkgConfig files
set(JPEGXL_LIBRARY_REQUIRES "libhwy libbrotlienc libbrotlidec libjxl_cms")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/jxl/libjxl.pc.in"
               "libjxl.pc" @ONLY)
set(JPEGXL_CMS_LIBRARY_REQUIRES "")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/jxl/libjxl_cms.pc.in"
               "libjxl_cms.pc" @ONLY)
set(JPEGXL_THREADS_LIBRARY_REQUIRES "")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/threads/libjxl_threads.pc.in"
               "libjxl_threads.pc" @ONLY)

# JPEGLI API

if (JPEGXL_ENABLE_JPEGLI)
  set(JPEGXL_LIBJPEG_SOURCE "../third_party/libjpeg-turbo")

  set(BITS_IN_JSAMPLE 8)
  set(MEM_SRCDST_SUPPORTED 1)

  if(JPEGLI_LIBJPEG_LIBRARY_SOVERSION STREQUAL "62")
    set(JPEG_LIB_VERSION 62)
  elseif(JPEGLI_LIBJPEG_LIBRARY_SOVERSION STREQUAL "7")
    set(JPEG_LIB_VERSION 70)
  elseif(JPEGLI_LIBJPEG_LIBRARY_SOVERSION STREQUAL "8")
    set(JPEG_LIB_VERSION 80)
  endif()

  configure_file(
    ${JPEGXL_LIBJPEG_SOURCE}/jconfig.h.in "include/jpegli/jconfig.h")
  configure_file(
    ${JPEGXL_LIBJPEG_SOURCE}/jpeglib.h "include/jpegli/jpeglib.h" COPYONLY)
  configure_file(
    ${JPEGXL_LIBJPEG_SOURCE}/jmorecfg.h "include/jpegli/jmorecfg.h" COPYONLY)
endif()
