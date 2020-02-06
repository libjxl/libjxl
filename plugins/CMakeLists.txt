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

if(NOT JPEGXL_EMSCRIPTEN)
  # Problem: linking shared library
  add_subdirectory(gdk-pixbuf)
  # Problem: compilation error: ... declared as an array with a negative size
  add_subdirectory(gimp)
endif()

add_subdirectory(mime)
