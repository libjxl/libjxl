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

#include "tools/comparison_viewer/image_loading.h"

#include <QRgb>
#include <QThread>

#include "lib/extras/codec.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/luminance.h"

namespace jxl {

namespace {

Status loadFromFile(const QString& filename, CodecInOut* const decoded,
                    ThreadPool* const pool) {
  PaddedBytes compressed;
  JXL_RETURN_IF_ERROR(ReadFile(filename.toStdString(), &compressed));
  const Span<const uint8_t> compressed_span(compressed);
  jxl::DecompressParams dparams;
  return DecodeFile(dparams, compressed, decoded, /*aux_out=*/nullptr, pool) ||
         SetFromBytes(compressed_span, decoded, pool);
}

}  // namespace

bool canLoadImageWithExtension(QString extension) {
  extension = extension.toLower();
  size_t bitsPerSampleUnused;
  return extension == "jxl" || extension == "j" || extension == "brn" ||
         CodecFromExtension("." + extension.toStdString(),
                            &bitsPerSampleUnused) != jxl::Codec::kUnknown;
}

QImage loadImage(const QString& filename, PaddedBytes targetIccProfile,
                 const QString& sourceColorSpaceHint) {
  static ThreadPoolInternal pool(QThread::idealThreadCount());

  CodecInOut decoded;
  if (!sourceColorSpaceHint.isEmpty()) {
    decoded.dec_hints.Add("color_space", sourceColorSpaceHint.toStdString());
  }
  if (!loadFromFile(filename, &decoded, &pool)) {
    return QImage();
  }
  const ImageBundle& ib = decoded.Main();

  ColorEncoding targetColorSpace;
  if (!targetColorSpace.SetICC(std::move(targetIccProfile))) {
    targetColorSpace = ColorEncoding::SRGB(ib.IsGray());
  }
  Image3F converted;
  if (!ib.CopyTo(Rect(ib), targetColorSpace, &converted, &pool)) {
    return QImage();
  }

  QImage image(converted.xsize(), converted.ysize(), QImage::Format_ARGB32);

  if (ib.HasAlpha()) {
    const int alphaRightShiftAmount =
        static_cast<int>(decoded.metadata.m.GetAlphaBits()) - 8;
    for (int y = 0; y < image.height(); ++y) {
      QRgb* const row = reinterpret_cast<QRgb*>(image.scanLine(y));
      const uint16_t* const alphaRow = ib.alpha().ConstRow(y);
      const float* const redRow = converted.ConstPlaneRow(0, y);
      const float* const greenRow = converted.ConstPlaneRow(1, y);
      const float* const blueRow = converted.ConstPlaneRow(2, y);
      for (int x = 0; x < image.width(); ++x) {
        row[x] = qRgba(redRow[x], greenRow[x], blueRow[x],
                       alphaRow[x] >> alphaRightShiftAmount);
      }
    }
  } else {
    for (int y = 0; y < image.height(); ++y) {
      QRgb* const row = reinterpret_cast<QRgb*>(image.scanLine(y));
      const float* const redRow = converted.ConstPlaneRow(0, y);
      const float* const greenRow = converted.ConstPlaneRow(1, y);
      const float* const blueRow = converted.ConstPlaneRow(2, y);
      for (int x = 0; x < image.width(); ++x) {
        row[x] = qRgb(redRow[x], greenRow[x], blueRow[x]);
      }
    }
  }

  return image;
}

}  // namespace jxl
