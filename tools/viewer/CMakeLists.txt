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

find_package(Qt5 QUIET COMPONENTS Widgets)
if (NOT Qt5_FOUND)
  message(WARNING "Qt5 was not found. The directory viewer will not be built.")
  return()
endif ()

if (NOT TARGET icc_detect)
  message(WARNING "The directory viewer depends on the comparison tool and will also not be built.")
  return ()
endif ()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_executable(viewer WIN32
  load_jxl.cc
  load_jxl.h
  main.cc
  viewer_window.cc
  viewer_window.h
  viewer_window.ui
)
target_include_directories(viewer PRIVATE
  $<TARGET_PROPERTY:lcms2,INCLUDE_DIRECTORIES>
  "${PROJECT_SOURCE_DIR}"
)
target_link_libraries(viewer
  Qt5::Widgets
  icc_detect
  jxl
  jxl_threads
  lcms2
)
