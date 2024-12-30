# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

find_package(PkgConfig)
pkg_check_modules(Gimp2 IMPORTED_TARGET gimp-2.0>=2.10 gimpui-2.0>=2.10)
pkg_check_modules(Gimp3 IMPORTED_TARGET gimp-3.0 gimpui-3.0)

if (NOT Gimp2_FOUND AND NOT Gimp3_FOUND)
  message(WARNING "Gimp development libraries not found, the Gimp plugin will not be built")
  return ()
endif ()

add_executable(file-jxl WIN32
  common.h
  common.cc
  file-jxl-load.cc
  file-jxl-load.h
  file-jxl-save.cc
  file-jxl-save.h
  file-jxl.cc)

target_include_directories(file-jxl PUBLIC
    ${PROJECT_SOURCE_DIR})  # for plugins/gimp absolute paths.

if(Gimp2_FOUND)
  target_link_libraries(file-jxl jxl jxl_threads PkgConfig::Gimp2)
  pkg_get_variable(GIMP_LIB_DIR gimp-2.0 gimplibdir)
else()
  target_link_libraries(file-jxl jxl jxl_threads PkgConfig::Gimp3)
  pkg_get_variable(GIMP_LIB_DIR gimp-3.0 gimplibdir)
endif()

install(TARGETS file-jxl RUNTIME DESTINATION "${GIMP_LIB_DIR}/plug-ins/file-jxl/")
