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

#ifndef TOOLS_COMPARISON_VIEWER_SPLIT_IMAGE_VIEW_H_
#define TOOLS_COMPARISON_VIEWER_SPLIT_IMAGE_VIEW_H_

#include <QWidget>

#include "tools/comparison_viewer/settings.h"
#include "tools/comparison_viewer/ui_split_image_view.h"

namespace jxl {

class SplitImageView : public QWidget {
  Q_OBJECT

 public:
  explicit SplitImageView(QWidget* parent = nullptr);
  ~SplitImageView() override = default;

  void setLeftImage(QImage image);
  void setRightImage(QImage image);
  void setMiddleImage(QImage image);

 signals:
  void renderingModeChanged(SplitImageRenderer::RenderingMode newMode);

 private slots:
  void on_settingsButton_clicked();

 private:
  Ui::SplitImageView ui_;
  SettingsDialog settings_;
};

}  // namespace jxl

#endif  // TOOLS_COMPARISON_VIEWER_SPLIT_IMAGE_VIEW_H_
