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
  jxl/extras/codec_pgx.cc
  jxl/extras/codec_pgx.h
  jxl/extras/codec_png.cc
  jxl/extras/codec_png.h
  jxl/extras/codec_pnm.cc
  jxl/extras/codec_pnm.h
)

find_package(GIF 5)
find_package(JPEG)
find_package(ZLIB)  # dependency of PNG
find_package(PNG)
pkg_check_modules(OpenEXR IMPORTED_TARGET OpenEXR)

if(PNG_FOUND AND ZLIB_FOUND)
  list(APPEND JPEGXL_EXTRAS_SOURCES
    jxl/extras/codec_apng.cc
    jxl/extras/codec_apng.h
  )
endif()

if(GIF_FOUND)
  list(APPEND JPEGXL_EXTRAS_SOURCES
    jxl/extras/codec_gif.cc
    jxl/extras/codec_gif.h
  )
endif()

if(JPEG_FOUND)
  list(APPEND JPEGXL_EXTRAS_SOURCES
    jxl/extras/codec_jpg.cc
    jxl/extras/codec_jpg.h
  )
endif()

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
  elseif(NOT "${${LIB}_LIBRARIES}" STREQUAL "${LIB}_LIBRARY-NOTFOUND")
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
)

if(GIF_FOUND)
  target_include_directories(jpegxl_extras-static PUBLIC "${GIF_INCLUDE_DIRS}")
endif()
if(JPEG_FOUND)
  target_include_directories(jpegxl_extras-static PUBLIC "${JPEG_INCLUDE_DIRS}")
endif()
if(PNG_FOUND)
  target_include_directories(jpegxl_extras-static PUBLIC "${PNG_INCLUDE_DIRS}")
endif()

target_link_libraries(jpegxl_extras-static PUBLIC
  jpegxl-static
  lodepng
  ${JPEGXL_STATIC_EXTRAS_LIBS}
)

if(GIF_FOUND)
  target_compile_definitions(jpegxl_extras-static PUBLIC -DJPEGXL_ENABLE_GIF=1)
endif()

if(PNG_FOUND AND ZLIB_FOUND)
  target_compile_definitions(jpegxl_extras-static PUBLIC -DJPEGXL_ENABLE_APNG=1)
endif()

if(JPEG_FOUND)
  target_compile_definitions(jpegxl_extras-static PUBLIC -DJPEGXL_ENABLE_JPEG=1)
endif()

if (JPEGXL_ENABLE_SJPEG)
  target_compile_definitions(jpegxl_extras-static PUBLIC -DJPEGXL_ENABLE_SJPEG=1)
  target_link_libraries(jpegxl_extras-static PUBLIC sjpeg)
endif ()

if (OpenEXR_FOUND AND NOT JPEGXL_EMSCRIPTEN)
  target_sources(jpegxl_extras-static PRIVATE
    jxl/extras/codec_exr.cc
    jxl/extras/codec_exr.h
  )
  target_compile_definitions(jpegxl_extras-static PUBLIC -DJPEGXL_ENABLE_EXR=1)
  target_link_libraries(jpegxl_extras-static PUBLIC PkgConfig::OpenEXR)
endif ()
