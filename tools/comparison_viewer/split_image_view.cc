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

#include "tools/comparison_viewer/split_image_view.h"

#include <utility>

#include <QLabel>

#include "tools/comparison_viewer/split_image_renderer.h"

namespace jxl {

SplitImageView::SplitImageView(QWidget* const parent) : QWidget(parent) {
  ui_.setupUi(this);

  ui_.splitImageRenderer->setRenderingSettings(settings_.renderingSettings());

  connect(ui_.middleWidthSlider, &QSlider::valueChanged,
          [this](const int value) {
            ui_.middleWidthDisplayLabel->setText(tr("%L1%").arg(value));
          });
  connect(ui_.middleWidthSlider, &QSlider::valueChanged, ui_.splitImageRenderer,
          &SplitImageRenderer::setMiddleWidthPercent);

  connect(ui_.zoomLevelSlider, &QSlider::valueChanged, [this](const int value) {
    if (value >= 0) {
      ui_.zoomLevelDisplayLabel->setText(tr("&times;%L1").arg(1 << value));
      ui_.splitImageRenderer->setZoomLevel(1 << value);
    } else {
      ui_.zoomLevelDisplayLabel->setText(tr("&times;1/%L1").arg(1 << -value));
      ui_.splitImageRenderer->setZoomLevel(1. / (1 << -value));
    }
  });

  connect(ui_.splitImageRenderer,
          &SplitImageRenderer::zoomLevelIncreaseRequested, [this]() {
            ui_.zoomLevelSlider->triggerAction(
                QAbstractSlider::SliderSingleStepAdd);
          });
  connect(ui_.splitImageRenderer,
          &SplitImageRenderer::zoomLevelDecreaseRequested, [this]() {
            ui_.zoomLevelSlider->triggerAction(
                QAbstractSlider::SliderSingleStepSub);
          });

  connect(ui_.splitImageRenderer, &SplitImageRenderer::renderingModeChanged,
          this, &SplitImageView::renderingModeChanged);
}

void SplitImageView::setLeftImage(QImage image) {
  ui_.splitImageRenderer->setLeftImage(std::move(image));
}

void SplitImageView::setRightImage(QImage image) {
  ui_.splitImageRenderer->setRightImage(std::move(image));
}

void SplitImageView::setMiddleImage(QImage image) {
  ui_.splitImageRenderer->setMiddleImage(std::move(image));
}

void SplitImageView::on_settingsButton_clicked() {
  if (settings_.exec()) {
    ui_.splitImageRenderer->setRenderingSettings(settings_.renderingSettings());
  }
}

}  // namespace jxl
