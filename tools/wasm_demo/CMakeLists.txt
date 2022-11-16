if (NOT JPEGXL_ENABLE_TOOLS OR NOT JPEGXL_EMSCRIPTEN)
  return()
endif()

# WASM API facade.
add_executable(jxl_decoder jxl_decoder.cc)
target_link_libraries(jxl_decoder
    jxl_extras-static
)

set(JXL_DECODER_SYMBOLS
  _malloc
  _free
  _jxlCreateInstance
  _jxlDestroyInstance
  _jxlFlush
  _jxlProcessInput
)
list(JOIN JXL_DECODER_SYMBOLS ", " JXL_DECODER_EXPORTS)

set_target_properties(jxl_decoder PROPERTIES LINK_FLAGS "\
  -O3 \
  --closure 1 \
  -s TOTAL_MEMORY=75mb \
  -s USE_LIBPNG=1 \
  -s DISABLE_EXCEPTION_CATCHING=1 \
  -s MODULARIZE=1 \
  -s FILESYSTEM=0 \
  -s USE_PTHREADS=1 \
  -s PTHREAD_POOL_SIZE=4 \
  -s EXPORT_NAME=\"JxlCodecModule\" \
  -s \"EXPORTED_FUNCTIONS=[${JXL_DECODER_EXPORTS}]\" \
")

if (BUILD_TESTING)
  add_test(
    NAME test_wasm_jxl_decoder
    COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR}
            ${CMAKE_CURRENT_SOURCE_DIR}/jxl_decoder_test.js
  )
  set_tests_properties(test_wasm_jxl_decoder PROPERTIES
    ENVIRONMENT NODE_PATH=$<TARGET_FILE_DIR:jxl_decoder>)
endif()  # BUILD_TESTING
