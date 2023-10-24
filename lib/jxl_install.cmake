# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Install all the library headers from the source and the generated ones. There
# is no distinction on which libraries use which header since it is expected
# that all developer libraries are available together at build time.

# Includes

install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/jxl
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/jxl
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

if (JPEGXL_ENABLE_JPEGLI AND JPEGXL_INSTALL_JPEGLI_LIBJPEG)
  install(
    DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/jpegli/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
endif()  # JPEGXL_ENABLE_JPEGLI AND JPEGXL_INSTALL_JPEGLI_LIBJPEG

# Libraries

set(JPEGXL_STATIC_LIBS
  jxl_cms-static
  jxl_threads-static
  jxl-static
  jxl_dec-static
)

set(JPEGXL_SHARED_LIBS
  jxl_cms
  jxl_threads
  jxl_extras_codec
  # Only install libjxl shared library. The libjxl_dec is not installed since it
  # contains symbols also in libjxl which would conflict if programs try to use
  # both.
  jxl
)

install(TARGETS ${JPEGXL_STATIC_LIBS} DESTINATION ${CMAKE_INSTALL_LIBDIR})

if (BUILD_SHARED_LIBS)
  install(TARGETS ${JPEGXL_SHARED_LIBS}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()  # BUILD_SHARED_LIBS

if (JPEGXL_ENABLE_JPEGLI AND JPEGXL_INSTALL_JPEGLI_LIBJPEG)
  install(TARGETS jpeg
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()  # JPEGXL_ENABLE_JPEGLI AND JPEGXL_INSTALL_JPEGLI_LIBJPEG

# PkgConfig files

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libjxl_cms.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libjxl_threads.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libjxl.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

