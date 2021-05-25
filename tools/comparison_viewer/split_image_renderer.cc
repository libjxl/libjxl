// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/comparison_viewer/split_image_renderer.h"

#include <algorithm>
#include <cmath>
#include <utility>

#include <QEvent>
#include <QGuiApplication>
#include <QPainter>
#include <QPalette>
#include <QPen>
#include <QPoint>
#include <QRect>

namespace jxl {

SplitImageRenderer::SplitImageRenderer(QWidget* const parent)
    : QWidget(parent) {
  setAttribute(Qt::WA_OpaquePaintEvent);
  setMouseTracking(true);
  setFocusPolicy(Qt::WheelFocus);
  grabKeyboard();

  connect(&fadingPoint_, &QVariantAnimation::valueChanged,
          [this] { update(); });
}

void SplitImageRenderer::setLeftImage(QImage image) {
  leftImage_ = QPixmap::fromImage(std::move(image));
  updateMinimumSize();
  update();
}
void SplitImageRenderer::setRightImage(QImage image) {
  rightImage_ = QPixmap::fromImage(std::move(image));
  updateMinimumSize();
  update();
}
void SplitImageRenderer::setMiddleImage(QImage image) {
  middleImage_ = QPixmap::fromImage(std::move(image));
  updateMinimumSize();
  update();
}

void SplitImageRenderer::setRenderingSettings(
    const SplitImageRenderingSettings& settings) {
  renderingSettings_ = settings;
}

void SplitImageRenderer::setMiddleWidthPercent(const int percent) {
  middleWidthPercent_ = percent;
  update();
}

void SplitImageRenderer::setZoomLevel(double scale) {
  scale_ = scale;
  updateMinimumSize();
  update();
}

void SplitImageRenderer::keyPressEvent(QKeyEvent* const event) {
  switch (event->key()) {
    case Qt::Key_Left:
      setRenderingMode(RenderingMode::LEFT);
      break;

    case Qt::Key_Right:
      setRenderingMode(RenderingMode::RIGHT);
      break;

    case Qt::Key_Up:
    case Qt::Key_Down:
      setRenderingMode(RenderingMode::MIDDLE);
      break;

    case Qt::Key_Escape:
      QCoreApplication::quit();
      break;

    case Qt::Key_ZoomIn:
      emit zoomLevelIncreaseRequested();
      break;
    case Qt::Key_ZoomOut:
      emit zoomLevelDecreaseRequested();
      break;

    default:
      QWidget::keyPressEvent(event);
      break;
  }
  update();
}

void SplitImageRenderer::mouseMoveEvent(QMouseEvent* const event) {
  setRenderingMode(RenderingMode::SPLIT);
  middleX_ = event->pos().x();
  update();
}

void SplitImageRenderer::wheelEvent(QWheelEvent* event) {
  if (QGuiApplication::keyboardModifiers().testFlag(Qt::ControlModifier)) {
    if (event->angleDelta().y() > 0) {
      emit zoomLevelIncreaseRequested();
      return;
    } else if (event->angleDelta().y() < 0) {
      emit zoomLevelDecreaseRequested();
      return;
    }
  }

  event->ignore();
}

void SplitImageRenderer::paintEvent(QPaintEvent* const event) {
  QRectF drawingArea(0., 0., minimumWidth(), minimumHeight());

  QPainter painter(this);
  painter.fillRect(rect(), QColor(119, 119, 119));
  painter.translate(QRectF(rect()).center() - drawingArea.center());
  painter.scale(scale_, scale_);
  if (scale_ < 1.) {
    painter.setRenderHint(QPainter::SmoothPixmapTransform);
  }

  const auto drawSingleImage = [&](const RenderingMode mode) {
    const QPixmap* image = nullptr;
    switch (mode) {
      case RenderingMode::LEFT:
        image = &leftImage_;
        break;
      case RenderingMode::RIGHT:
        image = &rightImage_;
        break;
      case RenderingMode::MIDDLE:
        image = &middleImage_;
        break;

      default:
        return;
    }
    painter.drawPixmap(QPointF(0., 0.), *image);
  };

  if (mode_ != RenderingMode::SPLIT) {
    if (fadingPoint_.state() != QAbstractAnimation::Running) {
      drawSingleImage(mode_);
      return;
    }

    const float fadingPoint = fadingPoint_.currentValue().toFloat();
    if (renderingSettings_.gray) {
      if (fadingPoint < renderingSettings_.fadingMSecs) {
        painter.setOpacity((renderingSettings_.fadingMSecs - fadingPoint) /
                           renderingSettings_.fadingMSecs);
        drawSingleImage(previousMode_);
      } else if (fadingPoint > renderingSettings_.fadingMSecs +
                                   renderingSettings_.grayMSecs) {
        painter.setOpacity((fadingPoint - renderingSettings_.fadingMSecs -
                            renderingSettings_.grayMSecs) /
                           renderingSettings_.fadingMSecs);
        drawSingleImage(mode_);
      }
    } else {
      drawSingleImage(previousMode_);
      painter.setOpacity(fadingPoint / renderingSettings_.fadingMSecs);
      drawSingleImage(mode_);
    }

    return;
  }

  const qreal middleWidth =
      std::min<qreal>((minimumWidth() / scale_) * middleWidthPercent_ / 100.,
                      middleImage_.width());

  const double transformedMiddleX =
      painter.transform().inverted().map(QPointF(middleX_, 0.)).x();
  QRectF middleRect = middleImage_.rect();
  middleRect.setWidth(middleWidth);
  middleRect.moveCenter(QPointF(transformedMiddleX, middleRect.center().y()));
  middleRect.setLeft(std::round(middleRect.left()));
  middleRect.setRight(std::round(middleRect.right()));

  QRectF leftRect = leftImage_.rect();
  leftRect.setRight(middleRect.left());

  QRectF rightRect = rightImage_.rect();
  rightRect.setLeft(middleRect.right());

  painter.drawPixmap(leftRect, leftImage_, leftRect);
  painter.drawPixmap(rightRect, rightImage_, rightRect);
  painter.drawPixmap(middleRect, middleImage_, middleRect);

  QPen middlePen;
  middlePen.setStyle(Qt::DotLine);
  painter.setPen(middlePen);
  painter.drawLine(leftRect.topRight(), leftRect.bottomRight());
  painter.drawLine(rightRect.topLeft(), rightRect.bottomLeft());
}

void SplitImageRenderer::updateMinimumSize() {
  const int imagesWidth = std::max(
      std::max(leftImage_.width(), rightImage_.width()), middleImage_.width());
  const int imagesHeight =
      std::max(std::max(leftImage_.height(), rightImage_.height()),
               middleImage_.height());
  setMinimumSize(scale_ * QSize(imagesWidth, imagesHeight));
}

void SplitImageRenderer::setRenderingMode(const RenderingMode newMode) {
  if (newMode == mode_) return;
  previousMode_ = mode_;
  mode_ = newMode;
  if (previousMode_ == RenderingMode::SPLIT || mode_ == RenderingMode::SPLIT) {
    fadingPoint_.stop();
  } else {
    const int msecs =
        renderingSettings_.gray
            ? 2 * renderingSettings_.fadingMSecs + renderingSettings_.grayMSecs
            : renderingSettings_.fadingMSecs;
    const float startValue = fadingPoint_.state() == QAbstractAnimation::Running
                                 ? fadingPoint_.endValue().toFloat() -
                                       fadingPoint_.currentValue().toFloat()
                                 : 0.f;
    fadingPoint_.stop();
    fadingPoint_.setStartValue(startValue);
    fadingPoint_.setEndValue(static_cast<float>(msecs));
    fadingPoint_.setDuration(fadingPoint_.endValue().toFloat() -
                             fadingPoint_.startValue().toFloat());
    fadingPoint_.start();
  }
  emit renderingModeChanged(mode_);
}

}  // namespace jxl
