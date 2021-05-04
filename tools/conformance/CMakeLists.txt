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

add_executable(djxl_conformance djxl_conformance.cc)
target_link_libraries(djxl_conformance jxl_dec)

if(BUILD_TESTING AND CMAKE_EXECUTABLE_SUFFIX STREQUAL "")
# Script to validate the tooling.
find_program (BASH_PROGRAM bash)
if(BASH_PROGRAM)
  add_test(
    NAME conformance_tooling_test
    COMMAND
        ${BASH_PROGRAM} ${CMAKE_CURRENT_SOURCE_DIR}/tooling_test.sh
        ${CMAKE_BINARY_DIR})
  # Skip the test if dependencies are not available.
  set_tests_properties(conformance_tooling_test PROPERTIES SKIP_RETURN_CODE 254)
endif()
endif() # BUILD_TESTING
