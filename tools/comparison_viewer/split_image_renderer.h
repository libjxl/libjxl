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

#ifndef TOOLS_COMPARISON_VIEWER_SPLIT_IMAGE_RENDERER_H_
#define TOOLS_COMPARISON_VIEWER_SPLIT_IMAGE_RENDERER_H_

#include <QImage>
#include <QKeyEvent>
#include <QMouseEvent>
#include <QPaintEvent>
#include <QPixmap>
#include <QVariantAnimation>
#include <QWheelEvent>
#include <QWidget>

namespace jxl {

struct SplitImageRenderingSettings {
  int fadingMSecs;
  bool gray;
  int grayMSecs;
};

class SplitImageRenderer : public QWidget {
  Q_OBJECT

 public:
  enum class RenderingMode {
    // The default mode when using the mouse: one (partial) image is shown on
    // each side of the cursor, with a vertical band of the middle image if
    // applicable.
    SPLIT,
    // Only show the left image (accessed by pressing the left arrow key when
    // the renderer has focus).
    LEFT,
    // Only show the right image (accessed by pressing the right arrow key).
    RIGHT,
    // Only show the middle image (accessed by pressing the up or down arrow
    // key).
    MIDDLE,
  };
  Q_ENUM(RenderingMode)

  explicit SplitImageRenderer(QWidget* parent = nullptr);
  ~SplitImageRenderer() override = default;

  QSize sizeHint() const override { return minimumSize(); }

  void setLeftImage(QImage image);
  void setRightImage(QImage image);
  void setMiddleImage(QImage image);

  void setRenderingSettings(const SplitImageRenderingSettings& settings);

 public slots:
  void setMiddleWidthPercent(int percent);
  void setZoomLevel(double scale);

 signals:
  void zoomLevelIncreaseRequested();
  void zoomLevelDecreaseRequested();

  void renderingModeChanged(RenderingMode newMode);

 protected:
  void keyPressEvent(QKeyEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void wheelEvent(QWheelEvent* event) override;
  void paintEvent(QPaintEvent* event) override;

 private:
  void updateMinimumSize();
  void setRenderingMode(RenderingMode newMode);

  QPixmap leftImage_, rightImage_, middleImage_;
  RenderingMode mode_ = RenderingMode::SPLIT;
  RenderingMode previousMode_ = RenderingMode::SPLIT;
  SplitImageRenderingSettings renderingSettings_;
  // Goes from 0 to the animation duration in milliseconds, as a float.
  QVariantAnimation fadingPoint_;
  int middleX_ = 0;
  int middleWidthPercent_ = 10;
  double scale_ = 1.;
};

}  // namespace jxl

#endif  // TOOLS_COMPARISON_VIEWER_SPLIT_IMAGE_RENDERER_H_
