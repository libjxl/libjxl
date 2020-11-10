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

#include "tools/viewer/load_jxl.h"

#include <stdint.h>

#include <QElapsedTimer>
#include <utility>

#include "jxl/decode.h"
#include "lib/jxl/aux_out.h"
#include "lib/jxl/base/arch_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/file_io.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/color_management.h"
#include "lib/jxl/dec_file.h"
#include "lib/jxl/dec_params.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"

namespace jxl {

QImage loadJxlImage(const QString& filename, PaddedBytes targetIccProfile,
                    qint64* elapsed_ns, bool* usedRequestedProfile) {
  static ProcessorTopology topology;
  static ThreadPoolInternal pool(topology.packages *
                                 topology.cores_per_package);

  PaddedBytes jpegXlData;
  if (!ReadFile(filename.toStdString(), &jpegXlData)) {
    return QImage();
  }
  if (jpegXlData.size() < 4) {
    return QImage();
  }
  CodecInOut io;
  QElapsedTimer timer;
  DecompressParams params;
  AuxOut localInfo;
  timer.start();
  if (!DecodeFile(params, jpegXlData, &io, &localInfo, &pool)) {
    return QImage();
  }
  if (elapsed_ns != nullptr) *elapsed_ns = timer.nsecsElapsed();

  const jxl::ImageBundle& ib = io.Main();
  ColorEncoding targetColorSpace;
  const bool profileSet = targetColorSpace.SetICC(std::move(targetIccProfile));
  if (usedRequestedProfile != nullptr) *usedRequestedProfile = profileSet;
  if (!profileSet) {
    targetColorSpace = ColorEncoding::SRGB(ib.IsGray());
  }
  Image3U decoded;
  if (!ib.CopyTo(Rect(ib), targetColorSpace, &decoded, &pool)) {
    return QImage();
  }

  QImage result(ib.xsize(), ib.ysize(),
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
                QImage::Format_RGBA64
#else
                QImage::Format_ARGB32
#endif
  );

  if (ib.HasAlpha()) {
    const int alphaLeftShiftAmount =
        16 - static_cast<int>(io.metadata.m.GetAlphaBits());
    for (int y = 0; y < result.height(); ++y) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
      QRgba64* const row = reinterpret_cast<QRgba64*>(result.scanLine(y));
#else
      QRgb* const row = reinterpret_cast<QRgb*>(result.scanLine(y));
#endif
      const uint16_t* const alphaRow = ib.alpha().ConstRow(y);
      const uint16_t* const redRow = decoded.ConstPlaneRow(0, y);
      const uint16_t* const greenRow = decoded.ConstPlaneRow(1, y);
      const uint16_t* const blueRow = decoded.ConstPlaneRow(2, y);
      for (int x = 0; x < result.width(); ++x) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
        row[x] = qRgba64(redRow[x], greenRow[x], blueRow[x],
                         alphaRow[x] << alphaLeftShiftAmount)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
                     .unpremultiplied()
#else
                     .toArgb32()
#endif
            ;
#else
        // Qt version older than 5.6 doesn't have a qRgba64.
        row[x] = qRgba(redRow[x] >> 8, greenRow[x] >> 8, blueRow[x] >> 8,
                       alphaRow[x] << alphaLeftShiftAmount);
#endif
      }
    }
  } else {
    for (int y = 0; y < result.height(); ++y) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
      QRgba64* const row = reinterpret_cast<QRgba64*>(result.scanLine(y));
#else
      QRgb* const row = reinterpret_cast<QRgb*>(result.scanLine(y));
#endif
      const uint16_t* const redRow = decoded.ConstPlaneRow(0, y);
      const uint16_t* const greenRow = decoded.ConstPlaneRow(1, y);
      const uint16_t* const blueRow = decoded.ConstPlaneRow(2, y);
      for (int x = 0; x < result.width(); ++x) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
        row[x] = qRgba64(redRow[x], greenRow[x], blueRow[x], 65535)
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
                     .unpremultiplied()
#else
                     .toArgb32()
#endif
            ;
#else
        // Qt version older than 5.6 doesn't have a qRgba64.
        row[x] = qRgb(redRow[x] >> 8, greenRow[x] >> 8, blueRow[x] >> 8);
#endif
      }
    }
  }

  return result;
}

}  // namespace jxl
