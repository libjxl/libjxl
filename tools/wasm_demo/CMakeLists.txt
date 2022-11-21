if (NOT JPEGXL_ENABLE_TOOLS OR NOT JPEGXL_EMSCRIPTEN)
  return()
endif()

# WASM API facade.
add_executable(jxl_decoder jxl_decoder.cc)
add_executable(jxl_decoder_for_test jxl_decoder.cc)
target_link_libraries(jxl_decoder jxl_extras-static)
target_link_libraries(jxl_decoder_for_test jxl_extras-static)

set(JXL_DECODER_SYMBOLS
  _malloc
  _free
  _jxlCreateInstance
  _jxlDestroyInstance
  _jxlFlush
  _jxlProcessInput
)
list(JOIN JXL_DECODER_SYMBOLS ", " JXL_DECODER_EXPORTS)

# TODO(eustas): How to set INITIAL_MEMORY on module startup?
set(JXL_DECODER_BASE_LINK_FLAGS "\
  -s USE_LIBPNG=1 \
  -s INITIAL_MEMORY=75mb \
  -s DISABLE_EXCEPTION_CATCHING=1 \
  -s MODULARIZE=1 \
  -s USE_PTHREADS=1 \
  -s PTHREAD_POOL_SIZE=4 \
  -s EXPORT_NAME=\"JxlCodecModule\" \
  -s \"EXPORTED_FUNCTIONS=[${JXL_DECODER_EXPORTS}]\" \
")

set_target_properties(jxl_decoder PROPERTIES LINK_FLAGS
  ${JXL_DECODER_BASE_LINK_FLAGS} + " -O3 -s FILESYSTEM=0 --closure 1 ")

set_target_properties(jxl_decoder_for_test PROPERTIES LINK_FLAGS
  ${JXL_DECODER_BASE_LINK_FLAGS} + " -O1 -s NODERAWFS=1 ")

if (BUILD_TESTING)
  add_test(
    NAME test_wasm_jxl_decoder
    COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR}
            --no-experimental-fetch
            ${CMAKE_CURRENT_SOURCE_DIR}/jxl_decoder_test.js
  )
  set_tests_properties(test_wasm_jxl_decoder PROPERTIES
    ENVIRONMENT NODE_PATH=$<TARGET_FILE_DIR:jxl_decoder_for_test>)
endif()  # BUILD_TESTING
