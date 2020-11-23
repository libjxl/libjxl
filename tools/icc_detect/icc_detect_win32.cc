// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tools/icc_detect/icc_detect.h"

#include <windows.h>

#include <memory>
#include <type_traits>

namespace jxl {

namespace {

struct HandleDeleter {
  void operator()(const HANDLE handle) const {
    if (handle != INVALID_HANDLE_VALUE) {
      CloseHandle(handle);
    }
  }
};
using HandleUniquePtr =
    std::unique_ptr<std::remove_pointer<HANDLE>::type, HandleDeleter>;

}  // namespace

QByteArray GetMonitorIccProfile(const QWidget* const widget) {
  const HWND window = reinterpret_cast<HWND>(widget->effectiveWinId());
  const HDC dc = GetDC(window);
  wchar_t profile_path[MAX_PATH];
  DWORD profile_path_size = MAX_PATH;
  if (!GetICMProfileW(dc, &profile_path_size, profile_path)) {
    ReleaseDC(window, dc);
    return QByteArray();
  }
  ReleaseDC(window, dc);
  HandleUniquePtr file(CreateFileW(profile_path, GENERIC_READ, FILE_SHARE_READ,
                                   nullptr, OPEN_EXISTING,
                                   FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
  if (file.get() == INVALID_HANDLE_VALUE) {
    return QByteArray();
  }
  LARGE_INTEGER profile_size;
  if (!GetFileSizeEx(file.get(), &profile_size)) {
    return QByteArray();
  }
  HandleUniquePtr mapping(
      CreateFileMappingW(file.get(), nullptr, PAGE_READONLY, 0, 0, nullptr));
  if (mapping == nullptr) {
    return QByteArray();
  }
  const char* const view = reinterpret_cast<const char*>(
      MapViewOfFile(mapping.get(), FILE_MAP_READ, 0, 0, 0));
  if (view == nullptr) {
    return QByteArray();
  }
  QByteArray profile(view, profile_size.QuadPart);
  UnmapViewOfFile(view);
  return profile;
}

}  // namespace jxl
