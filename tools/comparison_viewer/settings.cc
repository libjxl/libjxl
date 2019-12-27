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

#include "tools/comparison_viewer/settings.h"

namespace jxl {

SettingsDialog::SettingsDialog(QWidget* const parent)
    : QDialog(parent), settings_("JPEG XL project", "Comparison tool") {
  ui_.setupUi(this);

  settings_.beginGroup("rendering");
  renderingSettings_.fadingMSecs = settings_.value("fadingMSecs", 300).toInt();
  settings_.beginGroup("gray");
  renderingSettings_.gray = settings_.value("enabled", false).toBool();
  renderingSettings_.grayMSecs = settings_.value("delayMSecs", 300).toInt();
  settings_.endGroup();
  settings_.endGroup();

  settingsToUi();
}

SplitImageRenderingSettings SettingsDialog::renderingSettings() const {
  return renderingSettings_;
}

void SettingsDialog::on_SettingsDialog_accepted() {
  renderingSettings_.fadingMSecs = ui_.fadingTime->value();
  renderingSettings_.gray = ui_.grayGroup->isChecked();
  renderingSettings_.grayMSecs = ui_.grayTime->value();

  settings_.beginGroup("rendering");
  settings_.setValue("fadingMSecs", renderingSettings_.fadingMSecs);
  settings_.beginGroup("gray");
  settings_.setValue("enabled", renderingSettings_.gray);
  settings_.setValue("delayMSecs", renderingSettings_.grayMSecs);
  settings_.endGroup();
  settings_.endGroup();
}

void SettingsDialog::on_SettingsDialog_rejected() { settingsToUi(); }

void SettingsDialog::settingsToUi() {
  ui_.fadingTime->setValue(renderingSettings_.fadingMSecs);
  ui_.grayGroup->setChecked(renderingSettings_.gray);
  ui_.grayTime->setValue(renderingSettings_.grayMSecs);
}

}  // namespace jxl
