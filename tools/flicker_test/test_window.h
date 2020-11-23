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

#ifndef TOOLS_FLICKER_TEST_TEST_WINDOW_H_
#define TOOLS_FLICKER_TEST_TEST_WINDOW_H_

#include <QByteArray>
#include <QDir>
#include <QMainWindow>
#include <QStringList>
#include <QTextStream>

#include "tools/comparison_viewer/image_loading.h"
#include "tools/flicker_test/parameters.h"
#include "tools/flicker_test/ui_test_window.h"

namespace jxl {

class FlickerTestWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit FlickerTestWindow(FlickerTestParameters parameters,
                             QWidget* parent = nullptr);
  ~FlickerTestWindow() override = default;

  bool proceedWithTest() const { return proceed_; }

 private slots:
  void processTestResult(const QString& imageName, SplitView::Side originalSide,
                         SplitView::Side clickedSide, int clickDelayMSecs);

 private:
  void nextImage();

  Ui::FlickerTestWindow ui_;
  bool proceed_ = true;
  const QByteArray monitorProfile_;
  FlickerTestParameters parameters_;
  QDir originalFolder_, alteredFolder_;
  QFile outputFile_;
  QTextStream outputStream_;
  QStringList remainingImages_;
};

}  // namespace jxl

#endif  // TOOLS_FLICKER_TEST_TEST_WINDOW_H_
