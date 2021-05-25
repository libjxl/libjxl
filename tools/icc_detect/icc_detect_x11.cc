// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/icc_detect/icc_detect.h"

#include <stdint.h>
#include <stdlib.h>
#include <xcb/xcb.h>

#include <QX11Info>
#include <algorithm>
#include <memory>

namespace jxl {

namespace {

constexpr char kIccProfileAtomName[] = "_ICC_PROFILE";
constexpr uint32_t kMaxIccProfileSize = 1 << 24;

struct FreeDeleter {
  void operator()(void* const p) const { std::free(p); }
};
template <typename T>
using XcbUniquePtr = std::unique_ptr<T, FreeDeleter>;

}  // namespace

QByteArray GetMonitorIccProfile(const QWidget* const widget) {
  Q_UNUSED(widget)
  xcb_connection_t* const connection = QX11Info::connection();
  if (connection == nullptr) {
    return QByteArray();
  }
  const int screen_number = QX11Info::appScreen();

  const xcb_intern_atom_cookie_t atomRequest =
      xcb_intern_atom(connection, /*only_if_exists=*/1,
                      sizeof kIccProfileAtomName - 1, kIccProfileAtomName);
  const XcbUniquePtr<xcb_intern_atom_reply_t> atomReply(
      xcb_intern_atom_reply(connection, atomRequest, nullptr));
  if (atomReply == nullptr) {
    return QByteArray();
  }
  const xcb_atom_t iccProfileAtom = atomReply->atom;

  const xcb_screen_t* screen = nullptr;
  int i = 0;
  for (xcb_screen_iterator_t it =
           xcb_setup_roots_iterator(xcb_get_setup(connection));
       it.rem; xcb_screen_next(&it)) {
    if (i == screen_number) {
      screen = it.data;
      break;
    }
    ++i;
  }
  if (screen == nullptr) {
    return QByteArray();
  }
  const xcb_get_property_cookie_t profileRequest = xcb_get_property(
      connection, /*_delete=*/0, screen->root, iccProfileAtom,
      XCB_GET_PROPERTY_TYPE_ANY, /*long_offset=*/0, kMaxIccProfileSize);
  const XcbUniquePtr<xcb_get_property_reply_t> profile(
      xcb_get_property_reply(connection, profileRequest, nullptr));
  if (profile == nullptr || profile->bytes_after > 0) {
    return QByteArray();
  }

  return QByteArray(
      reinterpret_cast<const char*>(xcb_get_property_value(profile.get())),
      xcb_get_property_value_length(profile.get()));
}

}  // namespace jxl
