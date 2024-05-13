// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/comparison_viewer/settings.h"

namespace jpegxl {
namespace tools {

SettingsDialog::SettingsDialog(QWidget* const parent)
    : QDialog(parent), settings_("JPEG XL project", "Comparison tool") {
  ui_.setupUi(this);

  connect(ui_.zoomLevelSlider, &QSlider::valueChanged, [this](const int value) {
    if (value >= 0) {
      ui_.zoomLevelDisplayLabel->setText(tr("&times;%L1").arg(1 << value));
    } else {
      ui_.zoomLevelDisplayLabel->setText(tr("&times;1/%L1").arg(1 << -value));
    }
  });

  settings_.beginGroup("rendering");
  renderingSettings_.fadingMSecs = settings_.value("fadingMSecs", 300).toInt();
  settings_.beginGroup("gray");
  renderingSettings_.gray = settings_.value("enabled", false).toBool();
  renderingSettings_.grayMSecs = settings_.value("delayMSecs", 300).toInt();
  settings_.endGroup();
  settings_.beginGroup("zoomLevel");
  renderingSettings_.restoreLastZoomLevel =
      settings_.value("restoreLast", false).toBool();
  renderingSettings_.defaultLog2ZoomLevel =
      settings_.value("defaultLog2", 0).toInt();
  renderingSettings_.lastLog2ZoomLevel = settings_.value("lastLog2", 0).toInt();
  settings_.endGroup();
  settings_.endGroup();

  settingsToUi();
}

void SettingsDialog::setLastZoomLevel(int level) {
  settings_.beginGroup("rendering");
  settings_.beginGroup("zoomLevel");
  settings_.setValue("lastLog2", level);
  renderingSettings_.lastLog2ZoomLevel = level;
  settings_.endGroup();
  settings_.endGroup();
}

SplitImageRenderingSettings SettingsDialog::renderingSettings() const {
  return renderingSettings_;
}

void SettingsDialog::on_SettingsDialog_accepted() {
  renderingSettings_.fadingMSecs = ui_.fadingTime->value();
  renderingSettings_.gray = ui_.grayGroup->isChecked();
  renderingSettings_.grayMSecs = ui_.grayTime->value();
  renderingSettings_.restoreLastZoomLevel = ui_.lastZoomLevel->isChecked();
  renderingSettings_.defaultLog2ZoomLevel = ui_.zoomLevelSlider->value();

  settings_.beginGroup("rendering");
  settings_.setValue("fadingMSecs", renderingSettings_.fadingMSecs);
  settings_.beginGroup("gray");
  settings_.setValue("enabled", renderingSettings_.gray);
  settings_.setValue("delayMSecs", renderingSettings_.grayMSecs);
  settings_.endGroup();
  settings_.beginGroup("zoomLevel");
  settings_.setValue("restoreLast", renderingSettings_.restoreLastZoomLevel);
  settings_.setValue("defaultLog2", renderingSettings_.defaultLog2ZoomLevel);
  settings_.endGroup();
  settings_.endGroup();
}

void SettingsDialog::on_SettingsDialog_rejected() { settingsToUi(); }

void SettingsDialog::settingsToUi() {
  ui_.fadingTime->setValue(renderingSettings_.fadingMSecs);
  ui_.grayGroup->setChecked(renderingSettings_.gray);
  ui_.grayTime->setValue(renderingSettings_.grayMSecs);
  ui_.lastZoomLevel->setChecked(renderingSettings_.restoreLastZoomLevel);
  ui_.zoomLevelSlider->setValue(renderingSettings_.defaultLog2ZoomLevel);
}

}  // namespace tools
}  // namespace jpegxl
