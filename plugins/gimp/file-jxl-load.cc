// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "plugins/gimp/file-jxl-load.h"

#define _PROFILE_TARGET_ JXL_COLOR_PROFILE_TARGET_DATA

namespace jxl {

bool LoadJpegXlImage(const gchar *const filename, gint32 *const image_id) {
  std::vector<uint8_t> icc_profile;
  GimpColorProfile *profile = nullptr;
  bool is_linear = false;

  gint32 layer;

  gpointer pixels_buffer;
  size_t buffer_size;

  GimpImageBaseType image_type;
  GimpImageType layer_type = GIMP_RGB_IMAGE;
  GimpPrecision precision = GIMP_PRECISION_U16_GAMMA;
  JxlBasicInfo info = {};
  JxlPixelFormat format = {};

  format.num_channels = 4;
  format.data_type = JXL_TYPE_UINT8;
  format.endianness = JXL_NATIVE_ENDIAN;
  format.align = 0;

  JpegXlGimpProgress gimp_load_progress(
      ("Opening:" + (std::string)filename).c_str());
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
    g_printerr("JXL Error: JxlDecoderSubscribeEvents failed\n");
    return false;
  }

  if (JXL_DEC_SUCCESS != JxlDecoderSetParallelRunner(dec.get(),
                                                     JxlResizableParallelRunner,
                                                     runner.get())) {
    g_printerr("JXL Error: JxlDecoderSetParallelRunner failed\n");
    return false;
  }

  // grand decode loop... Is there  a better way to organize this?
  JxlDecoderSetInput(dec.get(), compressed.data(), compressed.size());

  while (true) {
    gimp_load_progress.update();

    JxlDecoderStatus status = JxlDecoderProcessInput(dec.get());

    if (status == JXL_DEC_BASIC_INFO) {
      // g_message("JXL_DEC_BASIC_INFO");
      if (JXL_DEC_SUCCESS != JxlDecoderGetBasicInfo(dec.get(), &info)) {
        g_printerr("JXL Error: JxlDecoderGetBasicInfo failed\n");
        return false;
      }

      JxlResizableParallelRunnerSetThreads(
          runner.get(),
          JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
    } else if (status == JXL_DEC_COLOR_ENCODING) {
      // Load ICC profile
      size_t icc_size = 0;

      if (JXL_DEC_SUCCESS != JxlDecoderGetICCProfileSize(dec.get(), &format,
                                                         _PROFILE_TARGET_,
                                                         &icc_size)) {
        g_printerr("JXL Warning: JxlDecoderGetICCProfileSize failed\n");
      }

      if (icc_size > 0) {
        icc_profile.resize(icc_size);
        if (JXL_DEC_SUCCESS != JxlDecoderGetColorAsICCProfile(
                                   dec.get(), &format, _PROFILE_TARGET_,
                                   icc_profile.data(), icc_profile.size())) {
          g_printerr("JXL Warning: JxlDecoderGetColorAsICCProfile failed\n");
        }

        profile = gimp_color_profile_new_from_icc_profile(
            icc_profile.data(), icc_profile.size(), /*error=*/nullptr);

        if (profile) {
          is_linear = gimp_color_profile_is_linear(profile);
          g_printerr("JXL Info: Setting is_linear = %d\n", is_linear);
        } else {
          g_printerr("JXL Warning: Failed to read ICC profile.\n");
        }
      } else {
        g_printerr("JXL Warning: Empty ICC data.\n");
      }

      // Internal color profile detection...
      JxlColorEncoding color_encoding;
      if (JXL_DEC_SUCCESS ==
          JxlDecoderGetColorAsEncodedProfile(
              dec.get(), &format, _PROFILE_TARGET_, &color_encoding)) {
        g_printerr("JXL Info: Internal profile detected.\n");

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
            if (profile) {
              g_printerr(
                  "Info: Unknown transfer function.  "
                  "ICC profile is present.");
            } else {
              g_printerr(
                  "Info: Unknown transfer function.  "
                  "No ICC profile present.");
            }
            break;
        }

        switch (color_encoding.color_space) {
          case JXL_COLOR_SPACE_RGB:
            if (color_encoding.white_point == JXL_WHITE_POINT_D65 &&
                color_encoding.primaries == JXL_PRIMARIES_SRGB) {
              if (is_linear) {
                profile = gimp_color_profile_new_rgb_srgb_linear();
              } else {
                profile = gimp_color_profile_new_rgb_srgb();
              }
            } else if (!is_linear &&
                       color_encoding.white_point == JXL_WHITE_POINT_D65 &&
                       (color_encoding.primaries_green_xy[0] == 0.2100 ||
                        color_encoding.primaries_green_xy[1] == 0.7100)) {
              // Probably Adobe RGB
              profile = gimp_color_profile_new_rgb_adobe();
            } else if (profile) {
              g_printerr(
                  "JXL Info: Unknown RGB colorspace. Using ICC profile.\n");
            } else {
              g_printerr(
                  "JXL Info: Unknown RGB colorspace. Treating as sRGB.\n");
              if (is_linear) {
                profile = gimp_color_profile_new_rgb_srgb_linear();
              } else {
                profile = gimp_color_profile_new_rgb_srgb();
              }
            }
            break;

          case JXL_COLOR_SPACE_GRAY:
            if (!profile) {
              if (is_linear) {
                profile = gimp_color_profile_new_d65_gray_linear();
              } else {
                profile = gimp_color_profile_new_d65_gray_srgb_trc();
              }
            }
            break;
          case JXL_COLOR_SPACE_XYB:
          case JXL_COLOR_SPACE_UNKNOWN:
          default:
            if (profile) {
              g_printerr("JXL Info: Unknown colorspace. Using ICC profile.\n");
            } else {
              g_error(
                  "Warning: Unknown colorspace. Treating as sRGB profile.\n");

              if (is_linear) {
                profile = gimp_color_profile_new_rgb_srgb_linear();
              } else {
                profile = gimp_color_profile_new_rgb_srgb();
              }
            }
            break;
        }
      }

      // set pixel format
      if (info.num_color_channels > 1) {
        if (info.alpha_bits == 0) {
          image_type = GIMP_RGB;
          layer_type = GIMP_RGB_IMAGE;
          format.num_channels = info.num_color_channels;
        } else {
          image_type = GIMP_RGB;
          layer_type = GIMP_RGBA_IMAGE;
          format.num_channels = info.num_color_channels + 1;
        }
      } else if (info.num_color_channels == 1) {
        if (info.alpha_bits == 0) {
          image_type = GIMP_GRAY;
          layer_type = GIMP_GRAY_IMAGE;
          format.num_channels = info.num_color_channels;
        } else {
          image_type = GIMP_GRAY;
          layer_type = GIMP_GRAYA_IMAGE;
          format.num_channels = info.num_color_channels + 1;
        }
      }

      // Set bit depth and linearity
      if (info.bits_per_sample <= 8) {
        if (is_linear) {
          format.data_type = JXL_TYPE_UINT8;
          precision = GIMP_PRECISION_U8_LINEAR;
        } else {
          format.data_type = JXL_TYPE_UINT8;
          precision = GIMP_PRECISION_U8_GAMMA;
        }
      } else if (info.bits_per_sample <= 16) {
        if (info.exponent_bits_per_sample > 0) {
          if (is_linear) {
            format.data_type = JXL_TYPE_FLOAT16;
            precision = GIMP_PRECISION_HALF_LINEAR;
          } else {
            format.data_type = JXL_TYPE_FLOAT16;
            precision = GIMP_PRECISION_HALF_GAMMA;
          }
        } else if (is_linear) {
          format.data_type = JXL_TYPE_UINT16;
          precision = GIMP_PRECISION_U16_LINEAR;
        } else {
          format.data_type = JXL_TYPE_UINT16;
          precision = GIMP_PRECISION_U16_GAMMA;
        }
      } else {
        if (info.exponent_bits_per_sample > 0) {
          if (is_linear) {
            format.data_type = JXL_TYPE_FLOAT;
            precision = GIMP_PRECISION_FLOAT_LINEAR;
          } else {
            format.data_type = JXL_TYPE_FLOAT;
            precision = GIMP_PRECISION_FLOAT_GAMMA;
          }
        } else if (is_linear) {
          format.data_type = JXL_TYPE_UINT32;
          precision = GIMP_PRECISION_U32_LINEAR;
        } else {
          format.data_type = JXL_TYPE_UINT32;
          precision = GIMP_PRECISION_U32_GAMMA;
        }
      }

      // create new image with profile
      *image_id = gimp_image_new_with_precision(info.xsize, info.ysize,
                                                image_type, precision);

      if (profile) {
        gimp_image_set_color_profile(*image_id, profile);
      } else {
        g_printerr("JXL Error: No color profile.\n");
      }
    } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
      // g_message("JXL_DEC_NEED_IMAGE_OUT_BUFFER");
      if (JXL_DEC_SUCCESS !=
          JxlDecoderImageOutBufferSize(dec.get(), &format, &buffer_size)) {
        g_printerr("JXL Error: JxlDecoderImageOutBufferSize failed\n");
        return false;
      }

      pixels_buffer = g_malloc(buffer_size);

      if (JXL_DEC_SUCCESS != JxlDecoderSetImageOutBuffer(dec.get(), &format,
                                                         pixels_buffer,
                                                         buffer_size)) {
        g_printerr("JXL Error: JxlDecoderSetImageOutBuffer failed\n");
        return false;
      }
    } else if (status == JXL_DEC_FULL_IMAGE) {
      // g_message("JXL_DEC_FULL_IMAGE");
      // create and insert layer
      layer = gimp_layer_new(*image_id, "Background", info.xsize, info.ysize,
                             layer_type, /*opacity=*/100,
                             gimp_image_get_default_new_layer_mode(*image_id));

      gimp_image_insert_layer(*image_id, layer, /*parent_id=*/-1,
                              /*position=*/0);

      // move image to layer buffer; need to clear layer buffer to update layer
      GeglBuffer *buffer = gimp_drawable_get_buffer(layer);
      gegl_buffer_set(buffer, GEGL_RECTANGLE(0, 0, info.xsize, info.ysize), 0,
                      nullptr, pixels_buffer, GEGL_AUTO_ROWSTRIDE);

      g_clear_object(&buffer);
    } else if (status == JXL_DEC_SUCCESS) {
      // g_message("JXL_DEC_SUCCESS");
      // All decoding successfully finished.
      // It's not required to call JxlDecoderReleaseInput(dec.get())
      // since the decoder will be destroyed.
      break;
    } else if (status == JXL_DEC_NEED_MORE_INPUT) {
      // g_message("JXL_DEC_NEED_MORE_INPUT");
      g_printerr("JXL Error: Already provided all input\n");
      return false;
    } else if (status == JXL_DEC_ERROR) {
      // g_message("JXL_DEC_ERROR");
      g_printerr("JXL Error: Decoder error\n");
      return false;
    } else {
      g_printerr("JXL Error: Unknown decoder status\n");
      return false;
    }
  }  // end grand decode loop

  gimp_load_progress.update();
  gimp_image_set_filename(*image_id, filename);

  gimp_load_progress.finished();
  return true;
}

}  // namespace jxl
