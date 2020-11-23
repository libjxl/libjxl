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
#include <QFile>

#include "jxl/decode.h"
#include "jxl/decode_cxx.h"
#include "jxl/thread_parallel_runner_cxx.h"
#include "jxl/types.h"
#include "skcms.h"

namespace jxl {

QImage loadJxlImage(const QString& filename, const QByteArray& targetIccProfile,
                    qint64* elapsed_ns, bool* usedRequestedProfile) {
  auto runner = JxlThreadParallelRunnerMake(
      nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());

  auto dec = JxlDecoderMake(nullptr);

#define EXPECT_TRUE(a)                                             \
  if (!(a)) {                                                      \
    fprintf(stderr, "Assertion failure (%d): %s\n", __LINE__, #a); \
    return QImage();                                               \
  }
#define EXPECT_EQ(a, b)                                               \
  {                                                                   \
    int a_ = a;                                                       \
    int b_ = b;                                                       \
    if (a_ != b_) {                                                   \
      fprintf(stderr, "Assertion failure (%d): %s (%d) != %s (%d)\n", \
              __LINE__, #a, a_, #b, b_);                              \
      return QImage();                                                \
    }                                                                 \
  }

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
  const uint8_t* next_in = reinterpret_cast<const uint8_t*>(jpegXlData.data());
  size_t avail_in = jpegXlData.size();
  EXPECT_EQ(JXL_DEC_BASIC_INFO,
            JxlDecoderProcessInput(dec.get(), &next_in, &avail_in));
  JxlBasicInfo info;
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetBasicInfo(dec.get(), &info));
  size_t pixel_count = info.xsize * info.ysize;

  EXPECT_EQ(JXL_DEC_COLOR_ENCODING,
            JxlDecoderProcessInput(dec.get(), &next_in, &avail_in));
  static const JxlPixelFormat format = {4, JXL_TYPE_FLOAT, JXL_NATIVE_ENDIAN,
                                        0};
  size_t icc_size;
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderGetICCProfileSize(
                dec.get(), &format, JXL_COLOR_PROFILE_TARGET_DATA, &icc_size));
  std::vector<uint8_t> icc_profile(icc_size);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderGetColorAsICCProfile(
                dec.get(), &format, JXL_COLOR_PROFILE_TARGET_DATA,
                icc_profile.data(), icc_profile.size()));

  std::vector<float> float_pixels(pixel_count * 4);
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetImageOutBuffer(dec.get(), &format, float_pixels.data(),
                                        pixel_count * 4 * sizeof(float)));
  EXPECT_EQ(JXL_DEC_FULL_IMAGE,
            JxlDecoderProcessInput(dec.get(), &next_in, &avail_in));

  std::vector<uint16_t> uint16_pixels(pixel_count * 4);
  skcms_ICCProfile jxl_profile;
  EXPECT_TRUE(
      skcms_Parse(icc_profile.data(), icc_profile.size(), &jxl_profile));
  skcms_ICCProfile target_profile;
  if (!skcms_Parse(targetIccProfile.data(), targetIccProfile.size(),
                   &target_profile)) {
    target_profile = *skcms_sRGB_profile();
    if (usedRequestedProfile) *usedRequestedProfile = false;
  } else {
    if (usedRequestedProfile) *usedRequestedProfile = true;
  }
  EXPECT_TRUE(skcms_Transform(
      float_pixels.data(), skcms_PixelFormat_RGBA_ffff,
      info.alpha_premultiplied ? skcms_AlphaFormat_PremulAsEncoded
                               : skcms_AlphaFormat_Unpremul,
      &jxl_profile, uint16_pixels.data(), skcms_PixelFormat_RGBA_16161616LE,
      skcms_AlphaFormat_Unpremul, &target_profile, pixel_count));
  if (elapsed_ns != nullptr) *elapsed_ns = timer.nsecsElapsed();

  QImage result(info.xsize, info.ysize,
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
                QImage::Format_RGBA64
#else
                QImage::Format_ARGB32
#endif
  );

  for (int y = 0; y < result.height(); ++y) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
    QRgba64* const row = reinterpret_cast<QRgba64*>(result.scanLine(y));
#else
    QRgb* const row = reinterpret_cast<QRgb*>(result.scanLine(y));
#endif
    const uint16_t* const data = uint16_pixels.data() + result.width() * y * 4;
    for (int x = 0; x < result.width(); ++x) {
#if QT_VERSION >= QT_VERSION_CHECK(5, 6, 0)
      row[x] = qRgba64(data[4 * x + 0], data[4 * x + 1], data[4 * x + 2],
                       data[4 * x + 3])
#if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
                   .unpremultiplied()
#else
                   .toArgb32()
#endif
          ;
#else
      // Qt version older than 5.6 doesn't have a qRgba64.
      row[x] = qRgba(data[4 * x + 0], data[4 * x + 1], data[4 * x + 2],
                     data[4 * x + 3]);
#endif
    }
  }
  return result;
}

}  // namespace jxl
