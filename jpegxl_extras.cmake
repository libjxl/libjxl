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

set(JPEGXL_EXTRAS_SOURCES
  jxl/extras/codec.cc
  jxl/extras/codec.h
  jxl/extras/codec_apng.cc
  jxl/extras/codec_apng.h
  jxl/extras/codec_gif.cc
  jxl/extras/codec_gif.h
  jxl/extras/codec_jpg.cc
  jxl/extras/codec_jpg.h
  jxl/extras/codec_pgx.cc
  jxl/extras/codec_pgx.h
  jxl/extras/codec_png.cc
  jxl/extras/codec_png.h
  jxl/extras/codec_pnm.cc
  jxl/extras/codec_pnm.h
)

find_package(GIF REQUIRED 5)
find_package(JPEG REQUIRED)
find_package(ZLIB REQUIRED)  # dependency of PNG
find_package(PNG REQUIRED)
find_package(PkgConfig)
pkg_check_modules(OpenEXR IMPORTED_TARGET OpenEXR)

foreach (lib gif jpeg png zlib)
  string(TOUPPER "${lib}" LIB)
  find_library(${LIB}_STATIC_LIBRARY
    NAMES lib${lib}.a
    HINTS
      ENV ${LIB}_DIR
    PATH_SUFFIXES lib
  )
  if(NOT "${${LIB}_STATIC_LIBRARY}" STREQUAL "${LIB}_STATIC_LIBRARY-NOTFOUND")
    list(APPEND JPEGXL_STATIC_EXTRAS_LIBS "${${LIB}_STATIC_LIBRARY}")
  else()
    message("Warning: lib${lib}.a not found, using ${${LIB}_LIBRARIES}")
    list(APPEND JPEGXL_STATIC_EXTRAS_LIBS "${${LIB}_LIBRARIES}")
  endif()
endforeach()

# We only define a static library for jpegxl_extras since it uses internal parts
# of jpegxl library which are not accessible from outside the library in the
# shared library case.
add_library(jpegxl_extras-static STATIC "${JPEGXL_EXTRAS_SOURCES}")
target_compile_options(jpegxl_extras-static PRIVATE "${JPEGXL_INTERNAL_FLAGS}")
set_property(TARGET jpegxl_extras-static PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(jpegxl_extras-static PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${GIF_INCLUDE_DIRS}"
  "${JPEG_INCLUDE_DIRS}"
  "${PNG_INCLUDE_DIRS}"
)

target_link_libraries(jpegxl_extras-static PUBLIC
  jpegxl-static
  lodepng
  ${JPEGXL_STATIC_EXTRAS_LIBS}
)

if (JPEGXL_ENABLE_SJPEG)
  target_compile_definitions(jpegxl_extras-static PUBLIC -DJPEGXL_ENABLE_SJPEG=1)
  target_link_libraries(jpegxl_extras-static PUBLIC sjpeg)
endif ()

if (OpenEXR_FOUND)
  target_sources(jpegxl_extras-static PRIVATE
    jxl/extras/codec_exr.cc
    jxl/extras/codec_exr.h
  )
  target_compile_definitions(jpegxl_extras-static PRIVATE -DJPEGXL_ENABLE_EXR=1)
  target_link_libraries(jpegxl_extras-static PUBLIC PkgConfig::OpenEXR)
endif ()
