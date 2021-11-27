// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "plugins/gimp/file-jxl-load.h"

#define _PROFILE_ORIG_ JXL_COLOR_PROFILE_TARGET_ORIGINAL
#define _PROFILE_DATA_ JXL_COLOR_PROFILE_TARGET_DATA
#define LOAD_PROC "file-jxl-load"

namespace jxl {

bool LoadJpegXlImage(const gchar *const filename, gint32 *const image_id) {
  std::vector<uint8_t> icc_orig;
  std::vector<uint8_t> icc_data;
  GimpColorProfile *profile_icc = nullptr;
  GimpColorProfile *profile_int = nullptr;
  bool is_linear = false;

  gint32 layer;

  gpointer pixels_buffer_1 = nullptr;
  gpointer pixels_buffer_2 = nullptr;
  size_t buffer_size = 0;

  cmsContext hContext = cmsCreateContext(nullptr, nullptr);
  cmsHPROFILE hInProfile, hOutProfile;
  cmsHTRANSFORM hTransform;
  uint32_t hInFormat = TYPE_RGB_FLT;
  uint32_t hOutFormat = 0;

  GimpImageBaseType image_type = GIMP_RGB;
  GimpImageType layer_type = GIMP_RGB_IMAGE;
  GimpPrecision precision_linear = GIMP_PRECISION_FLOAT_LINEAR;
  GimpPrecision precision_gamma = GIMP_PRECISION_FLOAT_GAMMA;
  JxlBasicInfo info = {};
  JxlPixelFormat format = {};

  format.num_channels = 4;
  format.data_type = JXL_TYPE_FLOAT;
  format.endianness = JXL_NATIVE_ENDIAN;
  format.align = 0;

  JpegXlGimpProgress gimp_load_progress(_("Opening JPEG XL file: %s"),
                                        filename);
  gimp_load_progress.update();

  // read file
  std::ifstream instream(filename, std::ios::in | std::ios::binary);
  std::vector<uint8_t> compressed((std::istreambuf_iterator<char>(instream)),
                                  std::istreambuf_iterator<char>());
  instream.close();

  gimp_load_progress.update();

  // multi-threaded parallel runner.
  auto runner = JxlResizableParallelRunnerMake(nullptr);

  auto dec = JxlDecoderMake(nullptr);
  if (JXL_DEC_SUCCESS !=
      JxlDecoderSubscribeEvents(dec.get(), JXL_DEC_BASIC_INFO |
                                               JXL_DEC_COLOR_ENCODING |
                                               JXL_DEC_FULL_IMAGE)) {
    g_printerr(LOAD_PROC " Error: JxlDecoderSubscribeEvents failed\n");
    return false;
  }

  if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(dec.get(),
                                                     JxlResizableParallelRunner,
                                                     runner.get())) {
    g_printerr(LOAD_PROC " Error: JxlDecoderSetParallelRunner failed\n");
    return false;
  }

  // grand decode loop...
  JxlDecoderSetInput(dec.get(), compressed.data(), compressed.size());

  while (true) {
    gimp_load_progress.update();

    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

    if (status == JXL_DEC_BASIC_INFO) {
      if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info)) {
        g_printerr(LOAD_PROC " Error: JxlDecoderGetBasicInfo failed\n");
        return false;
      }

      JxlResizableParallelRunnerSetThreads(
          runner.get(),
          JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      // check for ICC profile
      size_t icc_size = 0;
      JxlColorEncoding color_encoding;
      // Attempt to load original ICC profile
      if (JXL_DEC_SUCCESS !=
          JxlDecoderGetColorAsEncodedProfile(dec.get(), &format, _PROFILE_ORIG_,
                                             &color_encoding)) {
        if (JXL_DEC_SUCCESS != JxlDecoderGetICCProfileSize(dec.get(), &format,
                                                           _PROFILE_ORIG_,
                                                           &icc_size)) {
          g_printerr(LOAD_PROC
                     " Warning: JxlDecoderGetICCProfileSize failed "
                     "(_PROFILE_ORIG_)\n");
        }

        if (icc_size > 0) {
          icc_orig.resize(icc_size);
          if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(
                                     dec.get(), &format, _PROFILE_ORIG_,
                                     icc_orig.data(), icc_orig.size())) {
            g_printerr(LOAD_PROC
                       " Warning: JxlDecoderGetColorAsICCProfile failed "
                       "(_PROFILE_ORIG_)\n");
          }

          profile_icc = gimp_color_profile_new_from_icc_profile(
              icc_orig.data(), icc_orig.size(), nullptr);
        } else {
          g_printerr(LOAD_PROC " Warning: Empty ICC data (_PROFILE_ORIG_)\n");
        }
      }

      // Attempt to load data ICC profile... used for lcms2 transform
      if (JXL_DEC_SUCCESS !=
          JxlDecoderGetColorAsEncodedProfile(dec.get(), &format, _PROFILE_DATA_,
                                             &color_encoding)) {
        if (JXL_DEC_SUCCESS != JxlDecoderGetICCProfileSize(dec.get(), &format,
                                                           _PROFILE_DATA_,
                                                           &icc_size)) {
          g_printerr(LOAD_PROC
                     " Warning: JxlDecoderGetICCProfileSize failed "
                     "(_PROFILE_DATA_)\n");
        }

        if (icc_size > 0) {
          icc_data.resize(icc_size);
          if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(
                                     dec.get(), &format, _PROFILE_DATA_,
                                     icc_data.data(), icc_data.size())) {
            g_printerr(LOAD_PROC
                       " Warning: JxlDecoderGetColorAsICCProfile failed "
                       "(_PROFILE_DATA_)\n");
          }

          profile_int = gimp_color_profile_new_from_icc_profile(
              icc_data.data(), icc_data.size(), nullptr);
        } else {
          g_printerr(LOAD_PROC " Warning: Empty ICC data (_PROFILE_DATA_)\n");
        }
      }

      // Internal color profile detection...
      if (JXL_DEC_SUCCESS ==
          JxlDecoderGetColorAsEncodedProfile(dec.get(), &format, _PROFILE_DATA_,
                                             &color_encoding)) {
        g_printerr(LOAD_PROC " Info: Internal color encoding detected.\n");

        // figure out linearity of internal profile
        switch (color_encoding.transfer_function) {
          case JXL_TRANSFER_FUNCTION_LINEAR:
            is_linear = true;
            break;

          case JXL_TRANSFER_FUNCTION_709:
          case JXL_TRANSFER_FUNCTION_PQ:
          case JXL_TRANSFER_FUNCTION_HLG:
          case JXL_TRANSFER_FUNCTION_GAMMA:
          case JXL_TRANSFER_FUNCTION_DCI:
          case JXL_TRANSFER_FUNCTION_SRGB:
            is_linear = false;
            break;

          case JXL_TRANSFER_FUNCTION_UNKNOWN:
          default:
            is_linear = false;
            if (profile_icc) {
              g_printerr(LOAD_PROC
                         " Info: Unknown transfer function.  "
                         "ICC profile is present.");
            } else {
              g_printerr(LOAD_PROC
                         " Info: Unknown transfer function.  "
                         "No ICC profile present.");
            }
            break;
        }

        switch (color_encoding.color_space) {
          case JXL_COLOR_SPACE_RGB:
            if (color_encoding.white_point == JXL_WHITE_POINT_D65 &&
                color_encoding.primaries == JXL_PRIMARIES_SRGB) {
              if (is_linear) {
                profile_int = gimp_color_profile_new_rgb_srgb_linear();
              } else {
                profile_int = gimp_color_profile_new_rgb_srgb();
              }
            } else if (!is_linear &&
                       color_encoding.white_point == JXL_WHITE_POINT_D65 &&
                       (color_encoding.primaries_green_xy[0] == 0.2100 ||
                        color_encoding.primaries_green_xy[1] == 0.7100)) {
              // Probably Adobe RGB
              profile_int = gimp_color_profile_new_rgb_adobe();
            } else if (profile_icc) {
              g_printerr(LOAD_PROC
                         " Info: Unknown RGB colorspace.  "
                         "Using ICC profile.\n");
            } else {
              g_printerr(LOAD_PROC
                         " Info: Unknown RGB colorspace.  "
                         "Treating as sRGB.\n");
              if (is_linear) {
                profile_int = gimp_color_profile_new_rgb_srgb_linear();
              } else {
                profile_int = gimp_color_profile_new_rgb_srgb();
              }
            }
            break;

          case JXL_COLOR_SPACE_GRAY:
            if (!profile_icc ||
                color_encoding.white_point == JXL_WHITE_POINT_D65) {
              if (is_linear) {
                profile_int = gimp_color_profile_new_d65_gray_linear();
              } else {
                profile_int = gimp_color_profile_new_d65_gray_srgb_trc();
              }
            }
            break;
          case JXL_COLOR_SPACE_XYB:
          case JXL_COLOR_SPACE_UNKNOWN:
          default:
            if (profile_icc) {
              g_printerr(LOAD_PROC
                         " Info: Unknown colorspace.  "
                         "Using ICC profile.\n");
            } else {
              g_error(LOAD_PROC
                      " Warning: Unknown colorspace.  "
                      "Treating as sRGB profile.\n");

              if (is_linear) {
                profile_int = gimp_color_profile_new_rgb_srgb_linear();
              } else {
                profile_int = gimp_color_profile_new_rgb_srgb();
              }
            }
            break;
        }
      }

      // set pixel format
      if (info.num_color_channels > 1) {
        hOutFormat |= COLORSPACE_SH(PT_RGB) | CHANNELS_SH(3);
        if (info.alpha_bits == 0) {
          image_type = GIMP_RGB;
          layer_type = GIMP_RGB_IMAGE;
          format.num_channels = info.num_color_channels;
        } else {
          hInFormat |= EXTRA_SH(1);
          hOutFormat |= EXTRA_SH(1);
          image_type = GIMP_RGB;
          layer_type = GIMP_RGBA_IMAGE;
          format.num_channels = info.num_color_channels + 1;
        }
      } else if (info.num_color_channels == 1) {
        hOutFormat |= CHANNELS_SH(1);
        if (info.alpha_bits == 0) {
          image_type = GIMP_GRAY;
          layer_type = GIMP_GRAY_IMAGE;
          format.num_channels = info.num_color_channels;
        } else {
          hInFormat |= EXTRA_SH(1);
          hOutFormat |= EXTRA_SH(1);
          image_type = GIMP_GRAY;
          layer_type = GIMP_GRAYA_IMAGE;
          format.num_channels = info.num_color_channels + 1;
        }
      }

      // Set image bit depth and linearity
      if (info.bits_per_sample <= 8) {
        hOutFormat |= BYTES_SH(1);
        precision_linear = GIMP_PRECISION_U8_LINEAR;
        precision_gamma = GIMP_PRECISION_U8_GAMMA;
      } else if (info.bits_per_sample <= 16) {
        hOutFormat |= BYTES_SH(2);
        if (info.exponent_bits_per_sample > 0) {
          hOutFormat |= FLOAT_SH(1);
          precision_linear = GIMP_PRECISION_HALF_LINEAR;
          precision_gamma = GIMP_PRECISION_HALF_GAMMA;
        } else {
          precision_linear = GIMP_PRECISION_U16_LINEAR;
          precision_gamma = GIMP_PRECISION_U16_GAMMA;
        }
      } else {
        hOutFormat |= BYTES_SH(4);
        if (info.exponent_bits_per_sample > 0) {
          hOutFormat |= FLOAT_SH(1);
          precision_linear = GIMP_PRECISION_FLOAT_LINEAR;
          precision_gamma = GIMP_PRECISION_FLOAT_GAMMA;
        } else {
          precision_linear = GIMP_PRECISION_U32_LINEAR;
          precision_gamma = GIMP_PRECISION_U32_GAMMA;
        }
      }

      // setup lcms2 profile and transform
      hInProfile =
          cmsOpenProfileFromMemTHR(hContext, icc_data.data(), icc_data.size());
      hOutProfile =
          cmsOpenProfileFromMemTHR(hContext, icc_orig.data(), icc_orig.size());
      const uint32_t flags = cmsFLAGS_BLACKPOINTCOMPENSATION |
                             cmsFLAGS_HIGHRESPRECALC | cmsFLAGS_NULLTRANSFORM;

      hTransform = cmsCreateTransformTHR(hContext, hInProfile, hInFormat,
                                         hOutProfile, hOutFormat,
                                         INTENT_ABSOLUTE_COLORIMETRIC, flags);

      cmsCloseProfile(hInProfile);
      cmsCloseProfile(hOutProfile);

      if (!hTransform) {
        g_printerr(LOAD_PROC " Error: cmsCreateTransformTHR failed.\n");
        return false;
      }

      // create new image
      if (profile_icc) {
        if (is_linear) {
          *image_id = gimp_image_new_with_precision(
              info.xsize, info.ysize, image_type, precision_linear);
        } else {
          *image_id = gimp_image_new_with_precision(
              info.xsize, info.ysize, image_type, precision_gamma);
        }
      } else {
        *image_id = gimp_image_new_with_precision(info.xsize, info.ysize,
                                                  image_type, precision_gamma);
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      format.data_type = JXL_TYPE_FLOAT;
      if (JXL_DEC_SUCCESS !=
          JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size)) {
        g_printerr(LOAD_PROC " Error: JxlDecoderImageOutBufferSize failed\n");
        return false;
      }
      if (pixels_buffer_1 == nullptr) {
        pixels_buffer_1 = g_malloc(buffer_size);
        pixels_buffer_2 = g_malloc(buffer_size);
      }
      if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(dec.get(), &format,
                                                         pixels_buffer_1,
                                                         buffer_size)) {
        g_free(pixels_buffer_1);
        g_free(pixels_buffer_2);
        g_printerr(LOAD_PROC " Error: JxlDecoderSetImageOutBuffer failed\n");
        return false;
      }
    } else if (status == JXL_DEC_FULL_IMAGE || status == JXL_DEC_FRAME) {
      // create and insert layer
      layer = gimp_layer_new(*image_id, "Background", info.xsize, info.ysize,
                             layer_type, /*opacity=*/100,
                             gimp_image_get_default_new_layer_mode(*image_id));

      gimp_image_insert_layer(*image_id, layer, /*parent_id=*/-1,
                              /*position=*/0);

      GeglBuffer *buffer = gimp_drawable_get_buffer(layer);

      cmsDoTransform(hTransform, pixels_buffer_1, pixels_buffer_2,
                     info.xsize * info.ysize);

      gegl_buffer_set(buffer, GEGL_RECTANGLE(0, 0, info.xsize, info.ysize), 0,
                      nullptr, pixels_buffer_2, GEGL_AUTO_ROWSTRIDE);

      g_clear_object(&buffer);
    } else if (status == JXL_DEC_SUCCESS) {
      // All decoding successfully finished.
      // It's not required to call JxlDecoderReleaseInput(dec.get())
      // since the decoder will be destroyed.
      break;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      g_printerr(LOAD_PROC " Error: Already provided all input\n");
      return false;
    } else if (status == JXL_DEC_ERROR) {
      g_printerr(LOAD_PROC " Error: Decoder error\n");
      return false;
    } else {
      g_printerr(LOAD_PROC " Error: Unknown decoder status\n");
      return false;
    }
  }  // end grand decode loop

  g_free(pixels_buffer_1);
  g_free(pixels_buffer_2);

  if (profile_icc) {
    gimp_image_set_color_profile(*image_id, profile_icc);
  } else {
    gimp_image_set_color_profile(*image_id, profile_int);
  }

  gimp_load_progress.update();
  gimp_image_set_filename(*image_id, filename);

  gimp_load_progress.finished();
  return true;
}

}  // namespace jxl
