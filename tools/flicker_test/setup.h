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

#ifndef TOOLS_FLICKER_TEST_SETUP_H_
#define TOOLS_FLICKER_TEST_SETUP_H_

#include <QWizard>

#include "tools/flicker_test/parameters.h"
#include "tools/flicker_test/ui_setup.h"

namespace jxl {

class FlickerTestWizard : public QWizard {
  Q_OBJECT

 public:
  explicit FlickerTestWizard(QWidget* parent = nullptr);
  ~FlickerTestWizard() override = default;

  FlickerTestParameters parameters() const;

 protected:
  bool validateCurrentPage() override;

 private slots:
  void on_originalFolderBrowseButton_clicked();
  void on_alteredFolderBrowseButton_clicked();
  void on_outputFileBrowseButton_clicked();

  void on_timingButtonBox_clicked(QAbstractButton* button);

  void updateTotalGrayTime();

 private:
  Ui::FlickerTestWizard ui_;
  QSettings settings_;
};

}  // namespace jxl

#endif  // TOOLS_FLICKER_TEST_SETUP_H_
