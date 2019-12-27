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

find_package(PkgConfig)
pkg_check_modules(Gimp IMPORTED_TARGET gimp-2.0>=2.10 gimpui-2.0>=2.10)

if (NOT Gimp_FOUND)
  message(WARNING "Gimp development libraries not found, the Gimp plugin will not be built")
  return ()
endif ()

option(JPEGXL_ENABLE_GIMP_SAVING "Enable saving to JPEG XL from GIMP" OFF)

add_executable(file-j
  common.h
  file-j-load.cc
  file-j-load.h
  file-j-save.cc
  file-j-save.h
  file-j.cc)
if (JPEGXL_ENABLE_GIMP_SAVING)
  target_compile_definitions(file-j PRIVATE -DJPEGXL_ENABLE_GIMP_SAVING=1)
endif ()
target_link_libraries(file-j jpegxl-static jpegxl_threads PkgConfig::Gimp)
install(TARGETS file-j RUNTIME DESTINATION lib/gimp/2.0/plug-ins/file-j)
