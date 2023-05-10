// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_QT_COMPATIBILITY_H_
#define TOOLS_QT_COMPATIBILITY_H_

#include <xcb/xcb.h>

#include <QtGlobal>
#include <QPaintDevice>
#include <QPixmap>

namespace jpegxl {
namespace tools {

#if QT_VERSION < QT_VERSION_CHECK(5, 0, 0)
#error QT4 and earlier are not supported
#elif QT_VERSION < QT_VERSION_CHECK(5, 12, 0)
#error QT5 vesrions below 5.12.0 are not supported
#elif QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
void TransferPixelRatio(QPaintDevice* from, QPixmap* to) { /* no-op*/ }
qreal DevicePixelRatio(QPaintDevice* from) { return 1.0; }
QSizeF DeviceIndependentSize(QPixmap* obj) {
  return QSizeF(obj->size());
}
#include <QX11Info>
int DefaultScreenNumber(xcb_connection_t** connection) {
  *connection = QX11Info::connection();
  if (connection == nullptr) {
    return -1;
  }
  return QX11Info::appScreen();
}
#elif QT_VERSION < QT_VERSION_CHECK(6, 2, 0)
#error QT6 vesrions below 6.2.0 are not supported
#else
void TransferPixelRatio(QPaintDevice* from, QPixmap* to) {
  to->setDevicePixelRatio(from->devicePixelRatio());
}
qreal DevicePixelRatio(QPaintDevice* obj) { return obj->devicePixelRatio(); }
QSizeF DeviceIndependentSize(QPixmap* obj) {
  return obj->deviceIndependentSize();
}
#include <X11/Xlib.h>
int DefaultScreenNumber(xcb_connection_t** connection) {
  auto* const qX11App =
      qGuiApp->nativeInterface<QNativeInterface::QX11Application>();
  if (qX11App == nullptr) {
    return -1;
  }
  *connection = qX11App->connection();
  if (*connection == nullptr) {
    return -1;
  }

  return DefaultScreen(qX11App->display());
}
#endif

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_QT_COMPATIBILITY_H_
