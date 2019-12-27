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

#include "tools/flicker_test/parameters.h"

namespace jxl {

namespace {

constexpr char kPathsGroup[] = "paths";
constexpr char kOriginalFolderKey[] = "originalFolder";
constexpr char kAlteredFolderKey[] = "alteredFolder";
constexpr char kOutputFileKey[] = "outputFile";

constexpr char kTimingGroup[] = "timing";
constexpr char kAdvanceTimeKey[] = "advanceTimeMSecs";
constexpr char kViewingTimeKey[] = "viewingTimeSecs";
constexpr char kBlankingTimeKey[] = "blankingTimeMSecs";
constexpr char kGrayGroup[] = "gray";
constexpr char kGrayKey[] = "enabled";
constexpr char kGrayFadingTimeKey[] = "fadingTimeMSecs";
constexpr char kGrayTimeKey[] = "timeMSecs";

constexpr char kDisplayGroup[] = "display";
constexpr char kSpacingKey[] = "spacing";

}  // namespace

FlickerTestParameters FlickerTestParameters::loadFrom(
    QSettings* const settings) {
  FlickerTestParameters parameters;

  settings->beginGroup(kPathsGroup);
  parameters.originalFolder = settings->value(kOriginalFolderKey).toString();
  parameters.alteredFolder = settings->value(kAlteredFolderKey).toString();
  parameters.outputFile = settings->value(kOutputFileKey).toString();
  settings->endGroup();

  settings->beginGroup(kTimingGroup);
  parameters.advanceTimeMSecs = settings->value(kAdvanceTimeKey, 100).toInt();
  parameters.viewingTimeSecs = settings->value(kViewingTimeKey, 4).toInt();
  parameters.blankingTimeMSecs = settings->value(kBlankingTimeKey, 250).toInt();
  settings->beginGroup(kGrayGroup);
  parameters.gray = settings->value(kGrayKey, false).toBool();
  parameters.grayFadingTimeMSecs =
      settings->value(kGrayFadingTimeKey, 100).toInt();
  parameters.grayTimeMSecs = settings->value(kGrayTimeKey, 300).toInt();
  settings->endGroup();
  settings->endGroup();

  settings->beginGroup(kDisplayGroup);
  parameters.spacing = settings->value(kSpacingKey, 50).toInt();
  settings->endGroup();

  return parameters;
}

void FlickerTestParameters::saveTo(QSettings* const settings) const {
  settings->beginGroup(kPathsGroup);
  settings->setValue(kOriginalFolderKey, originalFolder);
  settings->setValue(kAlteredFolderKey, alteredFolder);
  settings->setValue(kOutputFileKey, outputFile);
  settings->endGroup();

  settings->beginGroup(kTimingGroup);
  settings->setValue(kAdvanceTimeKey, advanceTimeMSecs);
  settings->setValue(kViewingTimeKey, viewingTimeSecs);
  settings->setValue(kBlankingTimeKey, blankingTimeMSecs);
  settings->beginGroup(kGrayGroup);
  settings->setValue(kGrayKey, gray);
  settings->setValue(kGrayFadingTimeKey, grayFadingTimeMSecs);
  settings->setValue(kGrayTimeKey, grayTimeMSecs);
  settings->endGroup();
  settings->endGroup();

  settings->beginGroup(kDisplayGroup);
  settings->setValue(kSpacingKey, spacing);
  settings->endGroup();
}

}  // namespace jxl
