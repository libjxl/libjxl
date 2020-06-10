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

set(JPEGXL_MAJOR_VERSION 0)
set(JPEGXL_MINOR_VERSION 0)
set(JPEGXL_PATCH_VERSION 1)
set(JPEGXL_LIBRARY_VERSION
    "${JPEGXL_MAJOR_VERSION}.${JPEGXL_MINOR_VERSION}.${JPEGXL_PATCH_VERSION}")

# This the library API compatibility version.
set(JPEGXL_LIBRARY_SOVERSION "${JPEGXL_MAJOR_VERSION}")

# TODO(deymo): Split this into encoder and decoder targets
set(JPEGXL_INTERNAL_SOURCES
  jxl/modular/encoding/context_predict.h
  jxl/modular/encoding/encoding.cc
  jxl/modular/encoding/encoding.h
  jxl/modular/encoding/ma.cc
  jxl/modular/encoding/ma.h
  jxl/modular/image/image.cc
  jxl/modular/image/image.h
  jxl/modular/options.h
  jxl/modular/transform/near-lossless.h
  jxl/modular/transform/palette.h
  jxl/modular/transform/quantize.h
  jxl/modular/transform/squeeze.h
  jxl/modular/transform/subtractgreen.h
  jxl/modular/transform/transform.cc
  jxl/modular/transform/transform.h
  jxl/ac_context.h
  jxl/ac_strategy.cc
  jxl/ac_strategy.h
  jxl/alpha.cc
  jxl/alpha.h
  jxl/ans_common.cc
  jxl/ans_common.h
  jxl/ans_params.h
  jxl/ar_control_field.cc
  jxl/ar_control_field.h
  jxl/aux_out.cc
  jxl/aux_out.h
  jxl/aux_out.h
  jxl/aux_out_fwd.h
  jxl/base/arch_specific.cc
  jxl/base/arch_specific.h
  jxl/base/bits.h
  jxl/base/byte_order.h
  jxl/base/cache_aligned.cc
  jxl/base/cache_aligned.h
  jxl/base/compiler_specific.h
  jxl/base/data_parallel.cc
  jxl/base/data_parallel.h
  jxl/base/descriptive_statistics.cc
  jxl/base/descriptive_statistics.h
  jxl/base/fast_log.h
  jxl/base/file_io.h
  jxl/base/iaca.h
  jxl/base/os_specific.cc
  jxl/base/os_specific.h
  jxl/base/override.h
  jxl/base/padded_bytes.cc
  jxl/base/padded_bytes.h
  jxl/base/profiler.cc
  jxl/base/profiler.h
  jxl/base/robust_statistics.h
  jxl/base/span.h
  jxl/base/status.cc
  jxl/base/status.h
  jxl/base/tsc_timer.h
  jxl/brotli.cc
  jxl/brotli.h
  jxl/brunsli.cc
  jxl/brunsli.h
  jxl/butteraugli/butteraugli.cc
  jxl/butteraugli/butteraugli.h
  jxl/chroma_from_luma.cc
  jxl/chroma_from_luma.h
  jxl/coeff_order.cc
  jxl/coeff_order.h
  jxl/coeff_order_fwd.h
  jxl/color_encoding.cc
  jxl/color_encoding.h
  jxl/color_management.cc
  jxl/color_management.h
  jxl/common.h
  jxl/compressed_dc.cc
  jxl/compressed_dc.h
  jxl/convolve.cc
  jxl/convolve.h
  jxl/convolve-inl.h
  jxl/dct_block-inl.h
  jxl/dct_scales.cc
  jxl/dct_scales.h
  jxl/dct_for_test.h
  jxl/dct_util.h
  jxl/dec_ans.cc
  jxl/dec_ans.h
  jxl/dec_bit_reader.h
  jxl/dec_cache.h
  jxl/dec_context_map.cc
  jxl/dec_context_map.h
  jxl/dec_dct-inl.h
  jxl/dec_dct.cc
  jxl/dec_dct.h
  jxl/dec_file.cc
  jxl/dec_file.h
  jxl/dec_frame.cc
  jxl/dec_frame.h
  jxl/dec_group.cc
  jxl/dec_group.h
  jxl/dec_modular.cc
  jxl/dec_modular.h
  jxl/dec_noise.cc
  jxl/dec_noise.h
  jxl/dec_params.h
  jxl/dec_reconstruct.cc
  jxl/dec_reconstruct.h
  jxl/dec_transforms-inl.h
  jxl/dec_transforms.cc
  jxl/dec_transforms.h
  jxl/dec_xyb.cc
  jxl/dec_xyb.h
  jxl/dec_xyb-inl.h
  jxl/detect_dots.cc
  jxl/detect_dots.h
  jxl/dot_dictionary.cc
  jxl/dot_dictionary.h
  jxl/enc_ac_strategy.cc
  jxl/enc_ac_strategy.h
  jxl/enc_adaptive_quantization.cc
  jxl/enc_adaptive_quantization.h
  jxl/enc_ans.cc
  jxl/enc_ans.h
  jxl/enc_bit_writer.cc
  jxl/enc_bit_writer.h
  jxl/enc_butteraugli_comparator.cc
  jxl/enc_butteraugli_comparator.h
  jxl/enc_cache.cc
  jxl/enc_cache.h
  jxl/enc_cluster.cc
  jxl/enc_cluster.h
  jxl/enc_comparator.cc
  jxl/enc_comparator.h
  jxl/enc_context_map.cc
  jxl/enc_context_map.h
  jxl/enc_dct-inl.h
  jxl/enc_dct.cc
  jxl/enc_dct.h
  jxl/enc_file.cc
  jxl/enc_file.h
  jxl/enc_frame.cc
  jxl/enc_frame.h
  jxl/enc_gamma_correct.h
  jxl/enc_group.cc
  jxl/enc_group.h
  jxl/enc_modular.cc
  jxl/enc_modular.h
  jxl/enc_noise.cc
  jxl/enc_noise.h
  jxl/enc_params.h
  jxl/enc_transforms-inl.h
  jxl/enc_transforms.cc
  jxl/enc_transforms.h
  jxl/enc_xyb.cc
  jxl/enc_xyb.h
  jxl/entropy_coder.cc
  jxl/entropy_coder.h
  jxl/epf.cc
  jxl/epf.h
  jxl/external_image.cc
  jxl/external_image.h
  jxl/fast_log-inl.h
  jxl/field_encodings.h
  jxl/fields.cc
  jxl/fields.h
  jxl/frame_header.cc
  jxl/frame_header.h
  jxl/gaborish.cc
  jxl/gaborish.h
  jxl/gauss_blur.cc
  jxl/gauss_blur.h
  jxl/headers.cc
  jxl/headers.h
  jxl/icc_codec.cc
  jxl/icc_codec.h
  jxl/image.cc
  jxl/image.h
  jxl/image_bundle.cc
  jxl/image_bundle.h
  jxl/image_ops.h
  jxl/jpegxl/decode.cc
  jxl/jxl_inspection.h
  jxl/lehmer_code.h
  jxl/linalg.cc
  jxl/linalg.h
  jxl/loop_filter.cc
  jxl/loop_filter.h
  jxl/luminance.cc
  jxl/luminance.h
  jxl/memory_manager_internal.cc
  jxl/memory_manager_internal.h
  jxl/multiframe.cc
  jxl/multiframe.h
  jxl/noise.h
  jxl/noise_distributions.h
  jxl/opsin_params.cc
  jxl/opsin_params.h
  jxl/optimize.cc
  jxl/optimize.h
  jxl/passes_state.cc
  jxl/passes_state.h
  jxl/patch_dictionary.cc
  jxl/patch_dictionary.h
  jxl/predictor-inl.h
  jxl/predictor.cc
  jxl/predictor_shared.h
  jxl/quant_weights.cc
  jxl/quant_weights.h
  jxl/quantizer.cc
  jxl/quantizer.h
  jxl/quantizer-inl.h
  jxl/rational_polynomial-inl.h
  jxl/splines.cc
  jxl/splines.h
  jxl/splines_fastmath.h
  jxl/toc.cc
  jxl/toc.h
  jxl/transpose-inl.h
  jxl/xorshift128plus-inl.h
)

set(JPEGXL_INTERNAL_FLAGS
  # F_FLAGS
  -fmerge-all-constants
  -fno-builtin-fwrite
  -fno-builtin-fread

  # WARN_FLAGS
  -Wall
  -Wextra
  -Wc++11-compat
  -Wc++2a-extensions
  -Wdeprecated-increment-bool
  -Wfloat-overflow-conversion
  -Wfloat-zero-conversion
  -Wfor-loop-analysis
  -Wformat-security
  -Wgnu-redeclared-enum
  -Wimplicit-fallthrough
  -Winfinite-recursion
  -Wliteral-conversion
  -Wno-c++98-compat
  -Wno-register  # Needed by public headers in lcms
  -Wno-unused-command-line-argument
  -Wno-sign-compare
  -Wno-unused-function
  -Wno-unused-parameter
  -Wnon-virtual-dtor
  -Woverloaded-virtual
  -Wprivate-header
  -Wself-assign
  -Wstring-conversion
  -Wtautological-overlap-compare
  -Wthread-safety-analysis
  -Wundefined-func-template
  -Wunused-comparison
  -Wvla
)

if (WIN32)
list(APPEND JPEGXL_INTERNAL_FLAGS
  -Wno-c++98-compat-pedantic
  -Wno-cast-align
  -Wno-double-promotion
  -Wno-float-equal
  -Wno-format-nonliteral
  -Wno-global-constructors
  -Wno-language-extension-token
  -Wno-missing-prototypes
  -Wno-shadow
  -Wno-shadow-field-in-constructor
  -Wno-sign-conversion
  -Wno-unused-member-function
  -Wno-unused-template
  -Wno-used-but-marked-unused
  -Wno-zero-as-null-pointer-constant
)
else()
list(APPEND JPEGXL_INTERNAL_FLAGS
  -fno-signed-char
  -fsized-deallocation
  -fnew-alignment=8
  -fno-cxx-exceptions
  -fno-exceptions
  -fno-slp-vectorize
  -fno-vectorize

  # Language flags
  -disable-free
  -disable-llvm-verifier
  -fmath-errno
)
endif ()

set(JPEGXL_INTERNAL_LIBS
  brotlicommon-static
  brotlienc-static
  brotlidec-static
  brunslicommon-static
  brunslidec-static
  brunslienc-static
  hwy
  Threads::Threads
  ${CMAKE_DL_LIBS}
)
set(JPEGXL_LIBRARY_REQUIRES
    "libbrotlicommon libbrotlienc libbrotlidec")

if (JPEGXL_ENABLE_SKCMS)
  list(APPEND JPEGXL_INTERNAL_FLAGS -DJPEGXL_ENABLE_SKCMS=1)
  list(APPEND JPEGXL_INTERNAL_LIBS skcms)
else ()
  list(APPEND JPEGXL_INTERNAL_LIBS lcms2)
endif ()

if(JPEGXL_ENABLE_COVERAGE)
set(JPEGXL_COVERAGE_FLAGS
    -g -O0 -fprofile-arcs -ftest-coverage -DJXL_DISABLE_SLOW_TESTS
    -DJXL_ENABLE_ASSERT=0 -DJXL_ENABLE_CHECK=0
)
endif()

# Object library. This is used to hold the set of objects and properties.
add_library(jpegxl-obj OBJECT ${JPEGXL_INTERNAL_SOURCES})
target_compile_options(jpegxl-obj PRIVATE ${JPEGXL_INTERNAL_FLAGS})
target_compile_options(jpegxl-obj PUBLIC ${JPEGXL_COVERAGE_FLAGS})
set_property(TARGET jpegxl-obj PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(jpegxl-obj PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/fastapprox
  $<TARGET_PROPERTY:hwy,INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:brotlicommon-static,INCLUDE_DIRECTORIES>
  $<TARGET_PROPERTY:brunslicommon-static,INCLUDE_DIRECTORIES>
)
target_compile_definitions(jpegxl-obj PUBLIC
  JPEGXL_MAJOR_VERSION=${JPEGXL_MAJOR_VERSION}
  JPEGXL_MINOR_VERSION=${JPEGXL_MINOR_VERSION}
  JPEGXL_PATCH_VERSION=${JPEGXL_PATCH_VERSION}
  # Used to determine if we are building the library when defined or just
  # including the library when not defined. This is public so libjpegxl shared
  # library gets this define too.
  JPEGXL_INTERNAL_LIBRARY_BUILD
)

if (JPEGXL_ENABLE_SKCMS)
  target_include_directories(jpegxl-obj PRIVATE
    $<TARGET_PROPERTY:skcms,INCLUDE_DIRECTORIES>
  )
else ()
  target_include_directories(jpegxl-obj PRIVATE
    $<TARGET_PROPERTY:lcms2,INCLUDE_DIRECTORIES>
  )
endif ()

# Headers for exporting/importing public headers
include(GenerateExportHeader)
# TODO(deymo): Add these visibility properties to the static dependencies of
# jpegxl-obj since those are currently compiled with the default visibility.
set_target_properties(jpegxl-obj PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN 1
  DEFINE_SYMBOL JPEGXL_INTERNAL_LIBRARY_BUILD
)
generate_export_header(jpegxl-obj
  BASE_NAME JPEGXL
  EXPORT_FILE_NAME include/jpegxl/jpegxl_export.h)
target_include_directories(jpegxl-obj PUBLIC
    ${CMAKE_BINARY_DIR}/include)

# Private static library. This exposes all the internal functions and is used
# for tests.
add_library(jpegxl-static STATIC $<TARGET_OBJECTS:jpegxl-obj>)
target_link_libraries(jpegxl-static
  PUBLIC ${JPEGXL_COVERAGE_FLAGS} ${JPEGXL_INTERNAL_LIBS} hwy)
target_include_directories(jpegxl-static PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include")

# JPEGXL_EXPORT is defined to "__declspec(dllimport)" automatically by CMake
# in Windows builds when including headers from the C API and compiling from
# outside the jpegxl library. This is required when using the shared library,
# however in windows this causes the function to not be found when linking
# against the static library. This define JPEGXL_EXPORT= here forces it to not
# use dllimport in tests and other tools that require the static library.
target_compile_definitions(jpegxl-static INTERFACE -DJPEGXL_EXPORT=)

# TODO(deymo): Move TCMalloc linkage to the tools/ directory since the library
# shouldn't do any allocs anyway.
if(${JPEGXL_ENABLE_TCMALLOC})
  pkg_check_modules(TCMalloc REQUIRED IMPORTED_TARGET libtcmalloc)
  target_link_libraries(jpegxl-static PUBLIC PkgConfig::TCMalloc)
endif()  # JPEGXL_ENABLE_TCMALLOC

if(MINGW)
# TODO(deymo): Remove threads from jpegxl-obj and jpegxl-static once we don't
# use mutex inside the jpegxl library.
target_include_directories(jpegxl-obj PUBLIC
  $<TARGET_PROPERTY:mingw_stdthreads,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(jpegxl-static PUBLIC mingw_stdthreads)
endif()

if ((NOT DEFINED "${TARGET_SUPPORTS_SHARED_LIBS}") OR "${TARGET_SUPPORTS_SHARED_LIBS}")

# Public shared library.
add_library(jpegxl SHARED $<TARGET_OBJECTS:jpegxl-obj>)
target_link_libraries(jpegxl PUBLIC ${JPEGXL_COVERAGE_FLAGS})
target_link_libraries(jpegxl PRIVATE ${JPEGXL_INTERNAL_LIBS})
target_include_directories(jpegxl PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_BINARY_DIR}/include")
set_target_properties(jpegxl PROPERTIES
  VERSION ${JPEGXL_LIBRARY_VERSION}
  SOVERSION ${JPEGXL_LIBRARY_SOVERSION})

# Add a jpegxl.version file as a version script to tag symbols with the
# appropriate version number. This script is also used to limit what's exposed
# in the shared library from the static dependencies bundled here.
set_target_properties(jpegxl PROPERTIES LINK_DEPENDS ${CMAKE_SOURCE_DIR}/jxl/jpegxl.version)
if(APPLE)
set_property(TARGET jpegxl APPEND_STRING PROPERTY
             LINK_FLAGS "-Wl,-exported_symbols_list,${CMAKE_SOURCE_DIR}/jxl/jpegxl_osx.syms")
elseif(WIN32)
  # Nothing needed here, we use __declspec(dllexport) (jpegxl_export.h)
else()
set_property(TARGET jpegxl APPEND_STRING PROPERTY
             LINK_FLAGS " -Wl,--version-script=${CMAKE_SOURCE_DIR}/jxl/jpegxl.version")
endif()  # APPLE
install(TARGETS jpegxl
  DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/jpegxl
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/jpegxl")
install(DIRECTORY ${CMAKE_BINARY_DIR}/include/jpegxl
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/jpegxl")

# Add a pkg-config file for libjpegxl.
configure_file("${CMAKE_SOURCE_DIR}/jxl/libjpegxl.pc.in" "libjpegxl.pc" @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libjpegxl.pc"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

endif()  # TARGET_SUPPORTS_SHARED_LIBS
