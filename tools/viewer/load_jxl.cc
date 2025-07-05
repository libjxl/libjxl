// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "tools/viewer/load_jxl.h"

#include <jxl/decode.h>
#include <jxl/decode_cxx.h>
#include <jxl/thread_parallel_runner_cxx.h>
#include <jxl/types.h>
#include <stdint.h>

#include <QColorSpace>
#include <QColorTransform>
#include <QElapsedTimer>
#include <QFile>

namespace jpegxl {
namespace tools {

QImage loadJxlImage(const QString& filename, const QByteArray& targetIccProfile,
                    qint64* elapsed_ns, bool* usedRequestedProfile) {
  auto runner = JxlThreadParallelRunnerMake(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());

  auto dec = JxlDecoderMake(nullptr);

#define EXPECT_TRUE(a)                                               \
  do {                                                               \
    if (!(a)) {                                                      \
      fprintf(stderr, "Assertion failure (%d): %s\n", __LINE__, #a); \
      return QImage();                                               \
    }                                                                \
  } while (false)
#define EXPECT_EQ(a, b)                                               \
  do {                                                                \
    int a_ = a;                                                       \
    int b_ = b;                                                       \
    if (a_ != b_) {                                                   \
      fprintf(stderr, "Assertion failure (%d): %s (%d) != %s (%d)\n", \
              __LINE__, #a, a_, #b, b_);                              \
      return QImage();                                                \
    }                                                                 \
  } while (false)

  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO |
                                                     JXL_DEC_COLOR_ENCODING |
                                                     JXL_DEC_FULL_IMAGE));
  QFile jpegXlFile(filename);
  if (!jpegXlFile.open(QIODevice::ReadOnly)) {
    return QImage();
  }
  const QByteArray jpegXlData = jpegXlFile.readAll();
  if (jpegXlData.size() < 4) {
    return QImage();
  }

  QElapsedTimer timer;
  timer.start();
  const uint8_t* jxl_data = reinterpret_cast<const uint8_t*>(jpegXlData.data());
  size_t jxl_size = jpegXlData.size();
  JxlDecoderSetInput(dec.get(), jxl_data, jxl_size);
  EXPECT_EQ(JXL_DEC_BASIC_INFO, JxlDecoderProcessInput(dec.get()));
  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec.get(), &info));
  size_t pixel_count = info.xsize * info.ysize;

  EXPECT_EQ(JXL_DEC_COLOR_ENCODING, JxlDecoderProcessInput(dec.get()));
  static const JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN,
                                        0};
  size_t icc_size;
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderGetICCProfileSize(
                dec.get(), JXL_COLOR_PROFILE_TARGET_DATA, &icc_size));
  QByteArray icc_profile(icc_size, Qt::Initialization::Uninitialized);
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetColorAsICCProfile(
                                 dec.get(), JXL_COLOR_PROFILE_TARGET_DATA,
                                 reinterpret_cast<uint8_t*>(icc_profile.data()),
                                 icc_profile.size()));

  auto float_pixels = std::make_unique<float[]>(pixel_count * 4);
  EXPECT_EQ(JXL_DEC_NEED_IMAGE_OUT_BUFFER, JxlDecoderProcessInput(dec.get()));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetImageOutBuffer(dec.get(), &format, float_pixels.get(),
                                        pixel_count * 4 * sizeof(float)));
  EXPECT_EQ(JXL_DEC_FULL_IMAGE, JxlDecoderProcessInput(dec.get()));

  float* qimage_data = float_pixels.release();
  QImage result(
      reinterpret_cast<const uchar*>(qimage_data), info.xsize, info.ysize,
      info.alpha_premultiplied ? QImage::Format_RGBA32FPx4_Premultiplied
                               : QImage::Format_RGBA32FPx4,
      [](void* info) { delete[] reinterpret_cast<float*>(info); }, qimage_data);

  QColorSpace source_colorspace = QColorSpace::fromIccProfile(icc_profile);
  QColorSpace target_colorspace = QColorSpace::fromIccProfile(targetIccProfile);
  if (usedRequestedProfile != nullptr) {
    *usedRequestedProfile = target_colorspace.isValidTarget();
  }
  if (!target_colorspace.isValidTarget()) {
    target_colorspace = QColorSpace::SRgb;
  }
  QColorTransform transform =
      source_colorspace.transformationToColorSpace(target_colorspace);
  result.applyColorTransform(transform, QImage::Format_RGBA64_Premultiplied);
  if (elapsed_ns != nullptr) *elapsed_ns = timer.nsecsElapsed();

  return result;
}

}  // namespace tools
}  // namespace jpegxl
