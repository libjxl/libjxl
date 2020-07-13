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

cmake_minimum_required(VERSION 3.1)

# ICC detection library used by the comparison and viewer tools.
if(JPEGXL_ENABLE_VIEWERS)
if(WIN32)
  find_package(Qt5 QUIET COMPONENTS Widgets)
  if (NOT Qt5_FOUND)
    message(WARNING "Qt5 was not found.")
  else()
    add_library(icc_detect STATIC
      icc_detect/icc_detect_win32.cc
      icc_detect/icc_detect.h
    )
    target_link_libraries(icc_detect PUBLIC jpegxl-static Qt5::Widgets)
  endif()
else()
  find_package(Qt5 QUIET COMPONENTS Widgets X11Extras)
  find_package(ECM QUIET NO_MODULE)
  if (NOT Qt5_FOUND OR NOT ECM_FOUND)
    if (NOT Qt5_FOUND)
      message(WARNING "Qt5 was not found.")
    else()
      message(WARNING "extra-cmake-modules were not found.")
    endif()
  else()
    set(CMAKE_MODULE_PATH ${ECM_FIND_MODULE_DIR})
    find_package(XCB COMPONENTS XCB)
    if (XCB_FOUND)
      add_library(icc_detect STATIC
        icc_detect/icc_detect_x11.cc
        icc_detect/icc_detect.h
      )
      target_link_libraries(icc_detect PUBLIC jpegxl-static Qt5::Widgets Qt5::X11Extras XCB::XCB)
    endif ()
  endif()
endif()
endif()  # JPEGXL_ENABLE_VIEWERS

# The JPEGXL_VERSION is set from the builders.
if(NOT DEFINED JPEGXL_VERSION OR JPEGXL_VERSION STREQUAL "")
  set(JPEGXL_VERSION "(unknown)")
endif()

set(TOOL_BINARIES
  cbrunsli_tool
  cjpegxl
  djpegxl
  fuzzer_corpus

  butteraugli_main
  decode_and_encode
  epf_main
  ssimulacra_main

  xyb_range

  # Other tools added conditionally below:
  #  benchmark_xl
)

add_library(jpegxl_tool STATIC
  butteraugli_pnorm.cc
  cmdline.cc
)
target_compile_options(jpegxl_tool PUBLIC "${JPEGXL_INTERNAL_FLAGS}")
target_link_libraries(jpegxl_tool jpegxl-static)

target_include_directories(jpegxl_tool
  PUBLIC "${CMAKE_SOURCE_DIR}")

add_library(epf epf.cc epf.h)
target_link_libraries(epf jpegxl-static)

# Main compressors.
add_library(cjpegxltool STATIC
  cjpegxl.cc
  cjxl.cc
  cbrunsli.cc
  codec_config.cc
  speed_stats.cc
)
target_compile_definitions(cjpegxltool
  PRIVATE "-DJPEGXL_VERSION=\"${JPEGXL_VERSION}\"")
target_include_directories(cjpegxltool
  PUBLIC "${CMAKE_SOURCE_DIR}")
target_include_directories(cjpegxltool PRIVATE
  $<TARGET_PROPERTY:brunslicommon-static,INCLUDE_DIRECTORIES>
)
target_link_libraries(cjpegxltool
    box
    jpegxl-static
    jpegxl_extras-static
    jpegxl_threads
)

add_executable(cbrunsli_tool cbrunsli_main.cc)
add_executable(cjpegxl cjpegxl_main.cc)
add_executable(fuzzer_corpus fuzzer_corpus.cc)

# TODO: make brunsli own targets less "public"?
set_target_properties (cbrunsli_tool PROPERTIES OUTPUT_NAME "cbrunsli" )

target_link_libraries(cbrunsli_tool cjpegxltool)
target_link_libraries(cjpegxl cjpegxltool)

target_link_libraries(fuzzer_corpus cjpegxltool)

# Main decompressor.
add_executable(djpegxl
    codec_config.cc
    djxl.cc
    djpegxl.cc
    djpegxl_main.cc
    speed_stats.cc
)
target_compile_definitions(djpegxl
  PRIVATE "-DJPEGXL_VERSION=\"${JPEGXL_VERSION}\"")

# Other tools.
add_executable(ssimulacra_main ssimulacra_main.cc ssimulacra.cc)
add_executable(butteraugli_main butteraugli_main.cc)
add_executable(decode_and_encode decode_and_encode.cc)
add_executable(epf_main epf_main.cc)
add_executable(xyb_range xyb_range.cc)

target_link_libraries(epf_main epf)

# Benchmark tools.
if(${JPEGXL_ENABLE_BENCHMARK})
  list(APPEND TOOL_BINARIES
    benchmark_xl
  )

  add_executable(benchmark_xl
    benchmark/benchmark_xl.cc
    benchmark/benchmark_args.cc
    benchmark/benchmark_codec.cc
    benchmark/benchmark_file_io.cc
    benchmark/benchmark_stats.cc
    benchmark/benchmark_codec_custom.cc
    benchmark/benchmark_codec_custom.h
    benchmark/benchmark_codec_jxl.cc
    benchmark/benchmark_codec_jxl.h
    benchmark/benchmark_codec_png.cc
    benchmark/benchmark_codec_png.h
    codec_config.cc
    codec_config.h
    speed_stats.cc
    speed_stats.h
    ../third_party/dirent.cc
  )
  target_include_directories(benchmark_xl PRIVATE
    $<TARGET_PROPERTY:brunslicommon-static,INCLUDE_DIRECTORIES>
  )
  target_link_libraries(benchmark_xl Threads::Threads)
  target_compile_definitions(benchmark_xl
    PRIVATE "-DJPEGXL_VERSION=\"${JPEGXL_VERSION}\"")
  if(MINGW)
  # MINGW doesn't support glob.h.
  target_compile_definitions(benchmark_xl PRIVATE "-DHAS_GLOB=0")
  endif() # MINGW

  if(JPEG_FOUND)
    target_sources(benchmark_xl PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/benchmark/benchmark_codec_jpeg.cc"
      "${CMAKE_CURRENT_LIST_DIR}/benchmark/benchmark_codec_jpeg.h"
    )
    target_compile_definitions(benchmark_xl PRIVATE -DBENCHMARK_JPEG)
    target_include_directories(benchmark_xl PRIVATE ${JPEG_INCLUDE_DIR})
    target_link_libraries(benchmark_xl ${JPEG_LIBRARIES})
  endif ()

  find_package(PkgConfig)
  pkg_check_modules(WebP IMPORTED_TARGET libwebp)
  if(WebP_FOUND)
    target_sources(benchmark_xl PRIVATE
      "${CMAKE_CURRENT_LIST_DIR}/benchmark/benchmark_codec_webp.cc"
      "${CMAKE_CURRENT_LIST_DIR}/benchmark/benchmark_codec_webp.h"
    )
    target_compile_definitions(benchmark_xl PRIVATE -DBENCHMARK_WEBP)

    # Use the static version of webp if available.
    find_library(WebP_STATIC_LINK_LIBRARY NAMES libwebp.a
        PATHS "${WebP_LIBDIR}")
    if("${WebP_STATIC_LINK_LIBRARY}" STREQUAL "WebP_STATIC_LINK_LIBRARY-NOTFOUND")
      message(WARNING "Using dynamic libwebp")
      target_link_libraries(benchmark_xl PkgConfig::WebP)
    else()
      target_link_libraries(benchmark_xl "${WebP_STATIC_LINK_LIBRARY}")
      target_include_directories(benchmark_xl
          PRIVATE ${WebP_STATIC_INCLUDE_DIRS})
      target_compile_options(benchmark_xl PRIVATE ${WebP_STATIC_CFLAGS_OTHER})
    endif()
  endif()
endif()  # JPEGXL_ENABLE_BENCHMARK

# All tool binaries depend on "jpegxl" library and the tool helpers.
foreach(BINARY IN LISTS TOOL_BINARIES)
  target_include_directories("${BINARY}" PRIVATE "${CMAKE_SOURCE_DIR}")
  target_link_libraries("${BINARY}" box jpegxl-static jpegxl_extras-static jpegxl_threads jpegxl_tool)
endforeach()
install(TARGETS ${TOOL_BINARIES} RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

# Fuzzers.
if(${JPEGXL_ENABLE_FUZZERS})
  set(FUZZER_BINARIES
    color_encoding_fuzzer
    djxl_fuzzer
    icc_codec_fuzzer
    fields_fuzzer
    rans_fuzzer
    set_from_bytes_fuzzer
  )
  foreach(BINARY IN LISTS FUZZER_BINARIES)
    add_executable("${BINARY}" "${BINARY}.cc")
    target_include_directories("${BINARY}" PRIVATE "${CMAKE_SOURCE_DIR}")
    target_link_libraries("${BINARY}" jpegxl-static jpegxl_extras-static jpegxl_threads jpegxl_tool)
    target_compile_options("${BINARY}" PRIVATE "-fsanitize=fuzzer-no-link")
    target_link_libraries("${BINARY}" -fsanitize=fuzzer)
  endforeach()
endif()  # JPEGXL_ENABLE_FUZZERS

if(TARGET jpegxl)

# Library API test. This test is only to check that we can link against the
# shared library from C99 file and don't need to use internal symbols.
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/tests)
add_executable(libjpegxl_test libjpegxl_test.c)
set_property(TARGET libjpegxl_test PROPERTY C_STANDARD 99)
if(NOT ${SANITIZER} STREQUAL "none")
  # Linking a C test binary with the C++ JPEG XL implementation when using
  # address sanitizer is not well supported by clang 9, so force using clang++
  # for linking this test if a sanitizer is used.
  set_target_properties(libjpegxl_test PROPERTIES LINKER_LANGUAGE CXX)
endif()
set_target_properties(libjpegxl_test PROPERTIES PREFIX "tests/")
target_link_libraries(libjpegxl_test jpegxl)
target_compile_options(libjpegxl_test PRIVATE -Wall -Wextra -Werror)
if(NOT WIN32)
  target_compile_options(libjpegxl_test PRIVATE -pedantic)
endif()

add_test(NAME LibraryCLinkageTest COMMAND libjpegxl_test)

endif()  # TARGET jpegxl

# Tools defined in subdirectories.
if(${JPEGXL_ENABLE_VIEWERS})
add_subdirectory(comparison_viewer)
add_subdirectory(flicker_test)
add_subdirectory(viewer)
endif()

add_subdirectory(box)

if ("${JPEGXL_EMSCRIPTEN}")
# WASM API facade.
add_executable(jpegxl_emcc jpegxl_emcc.cc)
target_link_libraries(jpegxl_emcc jpegxl-static jpegxl_extras-static)
set_target_properties(jpegxl_emcc PROPERTIES LINK_FLAGS "\
  -O3\
  --closure 1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -flto \
  --llvm-lto 1 \
  -s DISABLE_EXCEPTION_CATCHING=1 \
  -s MODULARIZE=1 \
  -s FILESYSTEM=0 \
  -s EXPORT_NAME=\"JxlCodecModule\"\
  -s EXPORTED_FUNCTIONS='[\
    \"_compress\",\
    \"_decompress\",\
    \"_free\",\
    \"_malloc\"\
  ]'\
")
endif ()  # JPEGXL_EMSCRIPTEN
