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

#include <QElapsedTimer>
#include <QFile>
#include <QtConcurrent>

#define CMS_NO_REGISTER_KEYWORD 1
#include "lcms2.h"
#undef CMS_NO_REGISTER_KEYWORD

namespace jpegxl {
namespace tools {

namespace {

struct CmsProfileCloser {
  void operator()(const cmsHPROFILE profile) const {
    if (profile != nullptr) {
      cmsCloseProfile(profile);
    }
  }
};
using CmsProfileUniquePtr =
    std::unique_ptr<std::remove_pointer<cmsHPROFILE>::type, CmsProfileCloser>;

struct CmsTransformDeleter {
  void operator()(const cmsHTRANSFORM transform) const {
    if (transform != nullptr) {
      cmsDeleteTransform(transform);
    }
  }
};
using CmsTransformUniquePtr =
    std::unique_ptr<std::remove_pointer<cmsHTRANSFORM>::type,
                    CmsTransformDeleter>;

}  // namespace

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
  std::vector<uint8_t> icc_profile(icc_size);
  EXPECT_EQ(JXL_DEC_SUCCESS, JxlDecoderGetColorAsICCProfile(
                                 dec.get(), JXL_COLOR_PROFILE_TARGET_DATA,
                                 icc_profile.data(), icc_profile.size()));

  std::vector<float> float_pixels(pixel_count * 4);
  EXPECT_EQ(JXL_DEC_NEED_IMAGE_OUT_BUFFER, JxlDecoderProcessInput(dec.get()));
  EXPECT_EQ(JXL_DEC_SUCCESS,
            JxlDecoderSetImageOutBuffer(dec.get(), &format, float_pixels.data(),
                                        pixel_count * 4 * sizeof(float)));
  EXPECT_EQ(JXL_DEC_FULL_IMAGE, JxlDecoderProcessInput(dec.get()));

  auto uint16_pixels = std::make_unique<uint16_t[]>(pixel_count * 4);
  const thread_local cmsContext context = cmsCreateContext(nullptr, nullptr);
  EXPECT_TRUE(context != nullptr);
  const CmsProfileUniquePtr jxl_profile(cmsOpenProfileFromMemTHR(
      context, icc_profile.data(), icc_profile.size()));
  EXPECT_TRUE(jxl_profile != nullptr);
  CmsProfileUniquePtr target_profile(cmsOpenProfileFromMemTHR(
      context, targetIccProfile.data(), targetIccProfile.size()));
  if (usedRequestedProfile != nullptr) {
    *usedRequestedProfile = (target_profile != nullptr);
  }
  if (target_profile == nullptr) {
    target_profile.reset(cmsCreate_sRGBProfileTHR(context));
  }
  EXPECT_TRUE(target_profile != nullptr);
  CmsTransformUniquePtr transform(cmsCreateTransformTHR(
      context, jxl_profile.get(), TYPE_RGBA_FLT, target_profile.get(),
      TYPE_RGBA_16, INTENT_RELATIVE_COLORIMETRIC, cmsFLAGS_COPY_ALPHA));
  EXPECT_TRUE(transform != nullptr);
  std::vector<size_t> row_indices(info.ysize);
  std::iota(row_indices.begin(), row_indices.end(), 0);
  QtConcurrent::blockingMap(row_indices, [&](size_t y) {
    cmsDoTransform(transform.get(), float_pixels.data() + 4 * y * info.xsize,
                   uint16_pixels.get() + 4 * y * info.xsize, info.xsize);
  });
  if (elapsed_ns != nullptr) *elapsed_ns = timer.nsecsElapsed();

  uint16_t* qimage_data = uint16_pixels.release();
  return QImage(
      reinterpret_cast<uchar*>(qimage_data), info.xsize, info.ysize,
      info.alpha_premultiplied ? QImage::Format_RGBA64_Premultiplied
                               : QImage::Format_RGBA64,
      [](void* info) { delete[] reinterpret_cast<uint16_t*>(info); },
      qimage_data);
}

}  // namespace tools
}  // namespace jpegxl
