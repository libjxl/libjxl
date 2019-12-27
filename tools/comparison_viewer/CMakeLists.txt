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

find_package(Qt5 COMPONENTS Concurrent Widgets X11Extras)
if (NOT Qt5_FOUND)
  message(WARNING "Qt5 was not found. The comparison tool will not be built.")
  return()
endif ()

if (NOT TARGET x11_icc)
  message(WARNING "x11_icc not built. The comparison tool will not be built.")
  return ()
endif ()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)

add_library(image_loading STATIC
  image_loading.cc
  image_loading.h
)
target_link_libraries(image_loading PUBLIC
  Qt5::Widgets
  jpegxl-static
  jpegxl_threads
  jpegxl_extras-static
)

add_executable(compare_codecs WIN32
  codec_comparison_window.cc
  codec_comparison_window.h
  codec_comparison_window.ui
  compare_codecs.cc
  settings.cc
  settings.h
  settings.ui
  split_image_renderer.cc
  split_image_renderer.h
  split_image_view.cc
  split_image_view.h
  split_image_view.ui
)
target_link_libraries(compare_codecs
  image_loading
  Qt5::Concurrent
  Qt5::Widgets
  Qt5::X11Extras
  x11_icc
)

add_executable(compare_images WIN32
  compare_images.cc
  settings.cc
  settings.h
  settings.ui
  split_image_renderer.cc
  split_image_renderer.h
  split_image_view.cc
  split_image_view.h
  split_image_view.ui
)
target_link_libraries(compare_images
  image_loading
  Qt5::Widgets
  Qt5::X11Extras
  x11_icc
)
