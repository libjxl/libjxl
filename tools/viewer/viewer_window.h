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

#ifndef TOOLS_VIEWER_VIEWER_WINDOW_H_
#define TOOLS_VIEWER_VIEWER_WINDOW_H_

#include <QByteArray>
#include <QMainWindow>
#include <QStringList>

#include "tools/viewer/ui_viewer_window.h"

namespace jxl {

class ViewerWindow : public QMainWindow {
  Q_OBJECT
 public:
  explicit ViewerWindow(QWidget* parent = nullptr);

 public slots:
  void loadFilesAndDirectories(QStringList entries);

 private slots:
  void on_actionOpen_triggered();
  void on_actionPreviousImage_triggered();
  void on_actionNextImage_triggered();
  void refreshImage();

 private:
  const QByteArray monitorProfile_;
  Ui::ViewerWindow ui_;
  QStringList filenames_;
  int currentFileIndex_ = 0;
  bool hasWarnedAboutMonitorProfile_ = false;
};

}  // namespace jxl

#endif  // TOOLS_VIEWER_VIEWER_WINDOW_H_
