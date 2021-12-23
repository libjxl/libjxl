// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/comparison_viewer/image_loading.h"

#include <QRgb>
#include <QThread>

#include "lib/extras/codec.h"
#include "lib/extras/dec/color_hints.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/enc_color_management.h"
#include "tools/viewer/load_jxl.h"

namespace jxl {

namespace {

Status loadFromFile(const QString& filename,
                    const extras::ColorHints& color_hints,
                    CodecInOut* const decoded, ThreadPool* const pool) {
  PaddedBytes compressed;
  JXL_RETURN_IF_ERROR(ReadFile(filename.toStdString(), &compressed));
  const Span<const uint8_t> compressed_span(compressed);
  return SetFromBytes(compressed_span, color_hints, decoded, pool, nullptr);
}

}  // namespace

bool canLoadImageWithExtension(QString extension) {
  extension = extension.toLower();
  size_t bitsPerSampleUnused;
  return extension == "jxl" || extension == "j" || extension == "brn" ||
         extras::CodecFromExtension("." + extension.toStdString(),
                                    &bitsPerSampleUnused) !=
             jxl::extras::Codec::kUnknown;
}

QImage loadImage(const QString& filename, const QByteArray& targetIccProfile,
                 const float intensityTarget,
                 const QString& sourceColorSpaceHint) {
  qint64 elapsed;
  QImage img = loadJxlImage(filename, targetIccProfile, &elapsed);
  if (img.width() != 0 && img.height() != 0) {
    return img;
  }
  static ThreadPoolInternal pool(QThread::idealThreadCount());

  CodecInOut decoded;
  extras::ColorHints color_hints;
  if (!sourceColorSpaceHint.isEmpty()) {
    color_hints.Add("color_space", sourceColorSpaceHint.toStdString());
  }
  if (!loadFromFile(filename, color_hints, &decoded, &pool)) {
    return QImage();
  }
  decoded.metadata.m.SetIntensityTarget(intensityTarget);
  const ImageBundle& ib = decoded.Main();

  ColorEncoding targetColorSpace;
  PaddedBytes icc;
  icc.assign(reinterpret_cast<const uint8_t*>(targetIccProfile.data()),
             reinterpret_cast<const uint8_t*>(targetIccProfile.data() +
                                              targetIccProfile.size()));
  if (!targetColorSpace.SetICC(std::move(icc))) {
    targetColorSpace = ColorEncoding::SRGB(ib.IsGray());
  }
  Image3F converted;
  if (!ib.CopyTo(Rect(ib), targetColorSpace, GetJxlCms(), &converted, &pool)) {
    return QImage();
  }

  QImage image(converted.xsize(), converted.ysize(), QImage::Format_ARGB32);

  const auto ScaleAndClamp = [](const float x) {
    return Clamp1(x * 255 + .5f, 0.f, 255.f);
  };

  if (ib.HasAlpha()) {
    for (int y = 0; y < image.height(); ++y) {
      QRgb* const row = reinterpret_cast<QRgb*>(image.scanLine(y));
      const float* const alphaRow = ib.alpha().ConstRow(y);
      const float* const redRow = converted.ConstPlaneRow(0, y);
      const float* const greenRow = converted.ConstPlaneRow(1, y);
      const float* const blueRow = converted.ConstPlaneRow(2, y);
      for (int x = 0; x < image.width(); ++x) {
        row[x] = qRgba(ScaleAndClamp(redRow[x]), ScaleAndClamp(greenRow[x]),
                       ScaleAndClamp(blueRow[x]), ScaleAndClamp(alphaRow[x]));
      }
    }
  } else {
    for (int y = 0; y < image.height(); ++y) {
      QRgb* const row = reinterpret_cast<QRgb*>(image.scanLine(y));
      const float* const redRow = converted.ConstPlaneRow(0, y);
      const float* const greenRow = converted.ConstPlaneRow(1, y);
      const float* const blueRow = converted.ConstPlaneRow(2, y);
      for (int x = 0; x < image.width(); ++x) {
        row[x] = qRgb(ScaleAndClamp(redRow[x]), ScaleAndClamp(greenRow[x]),
                      ScaleAndClamp(blueRow[x]));
      }
    }
  }

  return image;
}

}  // namespace jxl
