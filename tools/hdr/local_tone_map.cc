// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/cms.h>

#include <cstdio>
#include <cstdlib>

#include "lib/jxl/base/rect.h"
#include "lib/jxl/base/status.h"
#include "tools/file_io.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "tools/hdr/local_tone_map.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/extras/codec.h"
#include "lib/extras/packed_image_convert.h"
#include "lib/extras/tone_mapping.h"
#include "lib/jxl/base/fast_math-inl.h"
#include "lib/jxl/convolve.h"
#include "lib/jxl/image_bundle.h"
#include "tools/cmdline.h"
#include "tools/no_memory_manager.h"
#include "tools/thread_pool_internal.h"

#if !defined(QUIT)
#define QUIT(M) JPEGXL_TOOLS_ABORT(M)
#endif

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
namespace {

using ::hwy::HWY_NAMESPACE::Add;
using ::hwy::HWY_NAMESPACE::Div;
using ::hwy::HWY_NAMESPACE::Lt;
using ::hwy::HWY_NAMESPACE::Max;
using ::hwy::HWY_NAMESPACE::Min;
using ::hwy::HWY_NAMESPACE::Mul;
using ::hwy::HWY_NAMESPACE::MulAdd;
using ::hwy::HWY_NAMESPACE::Sub;

constexpr size_t kDownsampling = 128;

// Color components must be in linear Rec. 2020.
template <typename V>
V ComputeLuminance(const float intensity_target, const V r, const V g,
                   const V b) {
  hwy::HWY_NAMESPACE::DFromV<V> df;
  const auto luminance =
      Mul(Set(df, intensity_target),
          MulAdd(Set(df, 0.2627f), r,
                 MulAdd(Set(df, 0.6780f), g, Mul(Set(df, 0.0593f), b))));
  return Max(Set(df, 1e-12f), luminance);
}

StatusOr<ImageF> DownsampledLuminances(const Image3F& image,
                                       const float intensity_target) {
  HWY_CAPPED(float, kDownsampling) d;
  JXL_ASSIGN_OR_RETURN(ImageF result,
                       ImageF::Create(jpegxl::tools::NoMemoryManager(),
                                      DivCeil(image.xsize(), kDownsampling),
                                      DivCeil(image.ysize(), kDownsampling)));
  FillImage(.5f * kDefaultIntensityTarget, &result);
  for (size_t y = 0; y < image.ysize(); ++y) {
    const float* const JXL_RESTRICT rows[3] = {image.ConstPlaneRow(0, y),
                                               image.ConstPlaneRow(1, y),
                                               image.ConstPlaneRow(2, y)};
    float* const JXL_RESTRICT result_row = result.Row(y / kDownsampling);

    for (size_t x = 0; x < image.xsize(); x += kDownsampling) {
      auto max = Set(d, result_row[x / kDownsampling]);
      for (size_t kx = 0; kx < kDownsampling && x + kx < image.xsize();
           kx += Lanes(d)) {
        max =
            Max(max, ComputeLuminance(
                         intensity_target, Load(d, rows[0] + x + kx),
                         Load(d, rows[1] + x + kx), Load(d, rows[2] + x + kx)));
      }
      result_row[x / kDownsampling] = GetLane(MaxOfLanes(d, max));
    }
  }
  HWY_FULL(float) df;
  for (size_t y = 0; y < result.ysize(); ++y) {
    float* const JXL_RESTRICT row = result.Row(y);
    for (size_t x = 0; x < result.xsize(); x += Lanes(df)) {
      Store(FastLog2f(df, Load(df, row + x)), df, row + x);
    }
  }
  return result;
}

StatusOr<ImageF> Upsample(const ImageF& image, ThreadPool* pool) {
  JXL_ASSIGN_OR_RETURN(ImageF upsampled_horizontally,
                       ImageF::Create(jpegxl::tools::NoMemoryManager(),
                                      2 * image.xsize(), image.ysize()));
  const auto BoundX = [&image](ssize_t x) {
    return Clamp1<ssize_t>(x, 0, image.xsize() - 1);
  };
  const auto process_row_h = [&](const int32_t y,
                                 const int32_t /*thread_id*/) -> Status {
    const float* const JXL_RESTRICT in_row = image.ConstRow(y);
    float* const JXL_RESTRICT out_row = upsampled_horizontally.Row(y);

    for (ssize_t x = 0; x < static_cast<ssize_t>(image.xsize()); ++x) {
      out_row[2 * x] = in_row[x];
      out_row[2 * x + 1] =
          0.5625f * (in_row[x] + in_row[BoundX(x + 1)]) -
          0.0625f * (in_row[BoundX(x - 1)] + in_row[BoundX(x + 2)]);
    }
    return true;
  };
  JPEGXL_TOOLS_CHECK(RunOnPool(pool, 0, image.ysize(), &ThreadPool::NoInit,
                               process_row_h, "UpsampleHorizontally"));

  HWY_FULL(float) df;
  JXL_ASSIGN_OR_RETURN(ImageF upsampled,
                       ImageF::Create(jpegxl::tools::NoMemoryManager(),
                                      2 * image.xsize(), 2 * image.ysize()));
  const auto BoundY = [&image](ssize_t y) {
    return Clamp1<ssize_t>(y, 0, image.ysize() - 1);
  };
  const auto process_row_v = [&](const int32_t y,
                                 const int32_t /*thread_id*/) -> Status {
    const float* const JXL_RESTRICT in_rows[4] = {
        upsampled_horizontally.ConstRow(BoundY(y - 1)),
        upsampled_horizontally.ConstRow(y),
        upsampled_horizontally.ConstRow(BoundY(y + 1)),
        upsampled_horizontally.ConstRow(BoundY(y + 2)),
    };
    float* const JXL_RESTRICT out_rows[2] = {
        upsampled.Row(2 * y),
        upsampled.Row(2 * y + 1),
    };

    for (ssize_t x = 0;
         x < static_cast<ssize_t>(upsampled_horizontally.xsize());
         x += Lanes(df)) {
      Store(Load(df, in_rows[1] + x), df, out_rows[0] + x);
      Store(MulAdd(Set(df, 0.5625f),
                   Add(Load(df, in_rows[1] + x), Load(df, in_rows[2] + x)),
                   Mul(Set(df, -0.0625f), Add(Load(df, in_rows[0] + x),
                                              Load(df, in_rows[3] + x)))),
            df, out_rows[1] + x);
    }
    return true;
  };
  JPEGXL_TOOLS_CHECK(RunOnPool(pool, 0, image.ysize(), &ThreadPool::NoInit,
                               process_row_v, "UpsampleVertically"));
  return upsampled;
}

Status ApplyLocalToneMapping(const ImageF& blurred_luminances,
                             const float intensity_target, Image3F* color,
                             ThreadPool* pool) {
  HWY_FULL(float) df;

  const auto log_default_intensity_target =
      Set(df, FastLog2f(kDefaultIntensityTarget));
  const auto log_10000 = Set(df, FastLog2f(10000.f));
  const auto process_row = [&](const int32_t y,
                               const int32_t /*thread_id*/) -> Status {
    float* const JXL_RESTRICT rows[3] = {
        color->PlaneRow(0, y), color->PlaneRow(1, y), color->PlaneRow(2, y)};
    const float* const JXL_RESTRICT blurred_lum_row =
        blurred_luminances.ConstRow(y);

    for (size_t x = 0; x < color->xsize(); x += Lanes(df)) {
      const auto log_local_max = Add(Load(df, blurred_lum_row + x), Set(df, 1));
      const auto luminance =
          ComputeLuminance(intensity_target, Load(df, rows[0] + x),
                           Load(df, rows[1] + x), Load(df, rows[2] + x));
      const auto log_luminance = Min(log_local_max, FastLog2f(df, luminance));
      const auto log_knee =
          Mul(log_default_intensity_target,
              MulAdd(Set(df, -0.85f),
                     Div(Sub(log_local_max, log_default_intensity_target),
                         Sub(log_10000, log_default_intensity_target)),
                     Set(df, 1.f)));
      const auto second_segment_position =
          Div(Sub(log_luminance, log_knee), Sub(log_local_max, log_knee));
      const auto log_new_luminance = IfThenElse(
          Lt(log_luminance, log_knee), log_luminance,
          MulAdd(second_segment_position,
                 MulAdd(Sub(log_default_intensity_target, log_knee),
                        second_segment_position, Sub(log_knee, log_luminance)),
                 log_luminance));
      const auto new_luminance = FastPow2f(df, log_new_luminance);
      const auto ratio = Div(Mul(Set(df, intensity_target), new_luminance),
                             Mul(luminance, Set(df, kDefaultIntensityTarget)));
      for (float* row : rows) {
        Store(Mul(ratio, Load(df, row + x)), df, row + x);
      }
    }
    return true;
  };
  JXL_RETURN_IF_ERROR(RunOnPool(pool, 0, color->ysize(), &ThreadPool::NoInit,
                                process_row, "ApplyLocalToneMapping"));

  return true;
}

}  // namespace
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace jxl {
namespace {

HWY_EXPORT(DownsampledLuminances);
HWY_EXPORT(Upsample);
HWY_EXPORT(ApplyLocalToneMapping);

Status Blur(ImageF* image) {
  static constexpr WeightsSeparable5 kBlurFilter = {
      {HWY_REP4(.375f), HWY_REP4(.25f), HWY_REP4(.0625f)},
      {HWY_REP4(.375f), HWY_REP4(.25f), HWY_REP4(.0625f)}};
  JXL_ASSIGN_OR_RETURN(ImageF blurred_once,
                       ImageF::Create(jpegxl::tools::NoMemoryManager(),
                                      image->xsize(), image->ysize()));
  JXL_RETURN_IF_ERROR(
      Separable5(*image, Rect(*image), kBlurFilter, nullptr, &blurred_once));
  JXL_RETURN_IF_ERROR(Separable5(blurred_once, Rect(blurred_once), kBlurFilter,
                                 nullptr, image));
  return true;
}

Status ProcessFrame(CodecInOut* image, float preserve_saturation,
                    ThreadPool* pool) {
  ColorEncoding linear_rec2020;
  JXL_RETURN_IF_ERROR(linear_rec2020.SetWhitePointType(WhitePoint::kD65));
  JXL_RETURN_IF_ERROR(linear_rec2020.SetPrimariesType(Primaries::k2100));
  linear_rec2020.Tf().SetTransferFunction(TransferFunction::kLinear);
  JXL_RETURN_IF_ERROR(linear_rec2020.CreateICC());
  JXL_RETURN_IF_ERROR(
      image->Main().TransformTo(linear_rec2020, *JxlGetDefaultCms(), pool));

  const float intensity_target = image->metadata.m.IntensityTarget();

  Image3F color = std::move(*image->Main().color());
  JXL_ASSIGN_OR_RETURN(
      ImageF subsampled_image,
      HWY_DYNAMIC_DISPATCH(DownsampledLuminances)(color, intensity_target));

  JXL_RETURN_IF_ERROR(Blur(&subsampled_image));
  ImageF blurred_luminances = std::move(subsampled_image);
  for (int downsampling = HWY_NAMESPACE::kDownsampling; downsampling > 1;
       downsampling >>= 1) {
    JXL_ASSIGN_OR_RETURN(
        blurred_luminances,
        HWY_DYNAMIC_DISPATCH(Upsample)(blurred_luminances,
                                       downsampling > 4 ? nullptr : pool));
  }

  JXL_RETURN_IF_ERROR(HWY_DYNAMIC_DISPATCH(ApplyLocalToneMapping)(
      blurred_luminances, intensity_target, &color, pool));

  JXL_RETURN_IF_ERROR(image->SetFromImage(std::move(color), linear_rec2020));
  image->metadata.m.color_encoding = linear_rec2020;
  image->metadata.m.SetIntensityTarget(kDefaultIntensityTarget);

  JXL_RETURN_IF_ERROR(GamutMap(image, preserve_saturation, pool));

  ColorEncoding rec2020_srgb = linear_rec2020;
  rec2020_srgb.Tf().SetTransferFunction(TransferFunction::kSRGB);
  JXL_RETURN_IF_ERROR(rec2020_srgb.CreateICC());
  JXL_RETURN_IF_ERROR(
      image->Main().TransformTo(rec2020_srgb, *JxlGetDefaultCms(), pool));
  image->metadata.m.color_encoding = rec2020_srgb;
  return true;
}

}  // namespace
}  // namespace jxl

int main(int argc, const char** argv) {
  jpegxl::tools::ThreadPoolInternal pool(8);

  jpegxl::tools::CommandLineParser parser;
  float preserve_saturation = .4f;
  parser.AddOptionValue(
      's', "preserve_saturation", "0..1",
      "to what extent to try and preserve saturation over luminance",
      &preserve_saturation, &jpegxl::tools::ParseFloat, 0);
  const char* input_filename = nullptr;
  auto input_filename_option = parser.AddPositionalOption(
      "input", true, "input image", &input_filename, 0);
  const char* output_filename = nullptr;
  auto output_filename_option = parser.AddPositionalOption(
      "output", true, "output image", &output_filename, 0);

  if (!parser.Parse(argc, argv)) {
    fprintf(stderr, "See -h for help.\n");
    return EXIT_FAILURE;
  }

  if (parser.HelpFlagPassed()) {
    parser.PrintHelp();
    return EXIT_SUCCESS;
  }

  if (!parser.GetOption(input_filename_option)->matched()) {
    fprintf(stderr, "Missing input filename.\nSee -h for help.\n");
    return EXIT_FAILURE;
  }
  if (!parser.GetOption(output_filename_option)->matched()) {
    fprintf(stderr, "Missing output filename.\nSee -h for help.\n");
    return EXIT_FAILURE;
  }

  jxl::CodecInOut image{jpegxl::tools::NoMemoryManager()};
  jxl::extras::ColorHints color_hints;
  color_hints.Add("color_space", "RGB_D65_202_Rel_PeQ");
  std::vector<uint8_t> encoded;
  JPEGXL_TOOLS_CHECK(jpegxl::tools::ReadFile(input_filename, &encoded));
  JPEGXL_TOOLS_CHECK(
      jxl::SetFromBytes(jxl::Bytes(encoded), color_hints, &image, pool.get()));

  JPEGXL_TOOLS_CHECK(
      jxl::ProcessFrame(&image, preserve_saturation, pool.get()));

  JxlPixelFormat format = {3, JXL_TYPE_UINT16, JXL_BIG_ENDIAN, 0};
  JXL_ASSIGN_OR_QUIT(jxl::extras::PackedPixelFile ppf,
                     jxl::extras::ConvertImage3FToPackedPixelFile(
                         *image.Main().color(), image.metadata.m.color_encoding,
                         format, pool.get()),
                     "ConvertImage3FToPackedPixelFile failed.");
  JPEGXL_TOOLS_CHECK(jxl::Encode(ppf, output_filename, &encoded, pool.get()));
  JPEGXL_TOOLS_CHECK(jpegxl::tools::WriteFile(output_filename, encoded));
  return EXIT_SUCCESS;
}

#endif
