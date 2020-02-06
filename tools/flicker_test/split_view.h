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

#ifndef TOOLS_FLICKER_TEST_SPLIT_VIEW_H_
#define TOOLS_FLICKER_TEST_SPLIT_VIEW_H_

#include <QElapsedTimer>
#include <QImage>
#include <QPixmap>
#include <QTimer>
#include <QVariantAnimation>
#include <QWidget>
#include <random>

namespace jxl {

class SplitView : public QWidget {
  Q_OBJECT

 public:
  enum class Side {
    kLeft,
    kRight,
  };
  Q_ENUM(Side)

  explicit SplitView(QWidget* parent = nullptr);
  ~SplitView() override = default;

  void setOriginalImage(QImage image);
  void setAlteredImage(QImage image);

 signals:
  void testResult(const QString& imageName, Side flickeringSide,
                  Side clickedSide, int clickDelayMSecs);

 public slots:
  void setSpacing(int spacing);
  void startTest(QString imageName, int blankingTimeMSecs, int viewingTimeSecs,
                 int advanceTimeMSecs, bool gray, int grayFadingTimeMSecs,
                 int grayTimeMSecs);

 protected:
  void mousePressEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void paintEvent(QPaintEvent* event) override;

 private slots:
  void startDisplaying();

 private:
  enum class State {
    kBlanking,
    kDisplaying,
  };

  void updateMinimumSize();

  int spacing_ = 50;

  std::mt19937 g_;

  QString imageName_;
  QPixmap original_, altered_;
  Side originalSide_;
  bool clicking_ = false;
  Side clickedSide_;
  QRect leftRect_, rightRect_;
  State state_ = State::kDisplaying;
  bool gray_ = false;
  QTimer blankingTimer_;
  QTimer viewingTimer_;
  // Throughout each cycle, animates the opacity of the image being displayed
  // between 0 and 1 if fading to gray is enabled.
  QVariantAnimation flicker_;
  bool showingAltered_ = true;
  QElapsedTimer viewingStartTime_;
};

}  // namespace jxl

#endif  // TOOLS_FLICKER_TEST_SPLIT_VIEW_H_
