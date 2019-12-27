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

#ifndef TOOLS_FLICKER_TEST_PARAMETERS_H_
#define TOOLS_FLICKER_TEST_PARAMETERS_H_

#include <QSettings>

namespace jxl {

struct FlickerTestParameters {
  QString originalFolder;
  QString alteredFolder;
  QString outputFile;
  int advanceTimeMSecs;
  int viewingTimeSecs;
  int blankingTimeMSecs;
  bool gray;
  int grayFadingTimeMSecs;
  int grayTimeMSecs;
  int spacing;

  static FlickerTestParameters loadFrom(QSettings* settings);
  void saveTo(QSettings* settings) const;
};

}  // namespace jxl

#endif  // TOOLS_FLICKER_TEST_PARAMETERS_H_
