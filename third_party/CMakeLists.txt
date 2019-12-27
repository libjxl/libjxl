# Copyright (c) the JPEG XL Project
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Highway
set(HWY_USE_GTEST ON CACHE BOOL "HWY_USE_GTEST")
add_subdirectory(highway)

# lodepng
if( NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lodepng/.git" )
  message(SEND_ERROR "Please run git submodule update --init")
endif()
include(lodepng.cmake)

# brotli
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/brotli/c/include/brotli/decode.h")
  message(SEND_ERROR "Brotli is required to build, run \"git submodule update --init\" to get it, then run cmake again")
endif ()
add_subdirectory(brotli EXCLUDE_FROM_ALL)

# *cms
if (JPEGXL_ENABLE_SKCMS)
  if( NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/skcms/.git" )
    message(SEND_ERROR "Please run git submodule update --init")
  endif()
  include(skcms.cmake)
else ()
  if( NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/lcms/.git" )
    message(SEND_ERROR "Please run git submodule update --init")
  endif()
  include(lcms2.cmake)
endif()

# gperftools (TCMalloc)
if("${JPEGXL_ENABLE_TCMALLOC}")
  if( NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/gperftools/.git" )
    message(SEND_ERROR "Please run git submodule update --init")
  endif()
  include(gperftools.cmake)
endif()  # JPEGXL_ENABLE_TCMALLOC

add_subdirectory(googletest EXCLUDE_FROM_ALL)

# brunsli
# depends both on brotli and googletest, so should be added after them.
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/brunsli/c/include/brunsli/brunsli_decode.h")
  message(SEND_ERROR "Brunsli is required to build, run \"git submodule update --init\" to get it, then run cmake again")
endif ()
add_subdirectory(brunsli EXCLUDE_FROM_ALL)

# sjpeg
if (JPEGXL_ENABLE_SJPEG)
  if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sjpeg/.git")
    message(SEND_ERROR "Please run git submodule update --init")
  endif()
  include(sjpeg.cmake)
endif ()
