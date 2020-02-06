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
  message(SEND_ERROR "Please run git submodule update --init")
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

add_subdirectory(googletest EXCLUDE_FROM_ALL)

# brunsli
# depends both on brotli and googletest, so should be added after them.
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/brunsli/c/include/brunsli/brunsli_decode.h")
  message(SEND_ERROR "Please run git submodule update --init")
endif ()
add_subdirectory(brunsli EXCLUDE_FROM_ALL)

# sjpeg
if (JPEGXL_ENABLE_SJPEG)
  if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/sjpeg/.git")
    message(SEND_ERROR "Please run git submodule update --init")
  endif()
  include(sjpeg.cmake)
endif ()
