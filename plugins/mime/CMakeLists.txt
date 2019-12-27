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

find_program(XDG_MIME_PROG xdg-mime)
if(NOT ${XDG_MIME_PROG} STREQUAL "XDG_MIME_PROG-NOTFOUND")
  install(CODE "execute_process(COMMAND ${XDG_MIME_PROG} install --novendor \
                              ${CMAKE_CURRENT_SOURCE_DIR}/image-x-jxl.xml)")
else()
  message(WARNING "xdg-mime not found")
endif()
# Can be uninstalled by `xdg-mine uninstall image-x-jxl.xml`.
