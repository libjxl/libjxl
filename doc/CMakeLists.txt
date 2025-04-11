# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Ubuntu focal ships with cmake 3.16.
cmake_minimum_required(VERSION 3.16...3.27)

project(LIBJXL_DOC LANGUAGES C CXX)

find_package(Doxygen OPTIONAL_COMPONENTS dot)

if(DOXYGEN_FOUND AND TARGET Doxygen::dot)
  set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib")
  set(DOXYGEN_GENERATE_HTML "YES")
  set(DOXYGEN_GENERATE_XML "YES")
  set(DOXYGEN_STRIP_FROM_PATH "${SOURCE_DIR}/include")
  if(JPEGXL_WARNINGS_AS_ERRORS)
    set(DOXYGEN_WARN_AS_ERROR "YES")
  endif()
  set(DOXYGEN_QUIET "YES")
  doxygen_add_docs(doc
    "${SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/api.txt"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Generating C API documentation")

  # Add sphinx doc build step for readthedocs.io (requires doxygen too).
  find_program(SPHINX_BUILD_PROGRAM sphinx-build)
  if(SPHINX_BUILD_PROGRAM)
    add_custom_command(
     OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/rtd/nonexistent"
     COMMENT "Generating readthedocs.io output on ${CMAKE_CURRENT_BINARY_DIR}/rtd"
     COMMAND ${SPHINX_BUILD_PROGRAM} -q -W -b html -j auto
       ${CMAKE_SOURCE_DIR}/sphinx
       ${CMAKE_CURRENT_BINARY_DIR}/rtd
     DEPENDS doc
    )
    # This command runs the documentation generation every time since the output
    # target file doesn't exist.
    add_custom_target(rtd-html
      DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/rtd/nonexistent
    )
  else() # SPHINX_BUILD_PROGRAM
    message(WARNING "sphinx-build not found, skipping rtd documentation")
  endif() # SPHINX_BUILD_PROGRAM

else()  # DOXYGEN_FOUND

  message(WARNING "Doxygen or Dot not installed; 'doc' target will FAIL")
  # Create a "doc" target for compatibility since "doc" is not otherwise added
  # to the build when doxygen is not installed.
  add_custom_target(doc false
    COMMENT "Error: Can't generate doc since Doxygen or Dot not installed.")

endif() # DOXYGEN_FOUND
