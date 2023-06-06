# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

include(jxl_lists.cmake)
include(compatibility.cmake)

add_library(jxl_cms-static STATIC EXCLUDE_FROM_ALL
  ${JPEGXL_INTERNAL_CMS_SOURCES})

target_compile_options(jxl_cms-static PUBLIC "${JPEGXL_INTERNAL_FLAGS}")
target_include_directories(jxl_cms-static PUBLIC
"${PROJECT_SOURCE_DIR}"
"${JXL_HWY_INCLUDE_DIRS}"
)


if (JPEGXL_ENABLE_SKCMS)
  target_include_directories(jxl_cms-static PRIVATE
    $<TARGET_PROPERTY:skcms,INCLUDE_DIRECTORIES>
  )
else ()
  target_include_directories(jxl_cms-static PRIVATE
    $<TARGET_PROPERTY:lcms2,INCLUDE_DIRECTORIES>
  )
endif ()
