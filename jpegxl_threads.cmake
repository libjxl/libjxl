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

set(JPEGXL_THREADS_SOURCES
  jxl/threads/thread_parallel_runner.cc
)

find_package(Threads REQUIRED)

# We only provide a static jpegxl_threads library since it is a non-stable C++
# API and very small. It is likely better to bundle it in the programs that
# need it.
add_library(jpegxl_threads STATIC "${JPEGXL_THREADS_SOURCES}")
target_compile_options(jpegxl_threads PRIVATE "${JPEGXL_INTERNAL_FLAGS}")
set_property(TARGET jpegxl_threads PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(jpegxl_threads PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(jpegxl_threads
  PUBLIC
    Threads::Threads
  PRIVATE
    #TODO(deymo): Change this to use the shared library. Right now it causes
    # problems when including both the shared and static library.
    jpegxl-static
)

if(MINGW)
target_link_libraries(jpegxl_threads PUBLIC mingw_stdthreads)
endif()
