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
  message(WARNING "Qt5 was not found. The flicker test tool will not be built.")
  return()
endif ()

if (NOT TARGET icc_detect OR NOT TARGET image_loading)
  message(WARNING "Comparison tool not built. The flicker test tool will not be built.")
  return()
endif ()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_executable(flicker_test WIN32
  main.cc
  parameters.cc
  parameters.h
  setup.cc
  setup.h
  setup.ui
  split_view.cc
  split_view.h
  test_window.cc
  test_window.h
  test_window.ui)

target_link_libraries(flicker_test PUBLIC
  Qt5::Widgets
  image_loading
  icc_detect
)
