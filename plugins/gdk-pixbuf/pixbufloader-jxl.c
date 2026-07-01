// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <jxl/codestream_header.h>
#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/resizable_parallel_runner.h>
#include <jxl/types.h>

#define GDK_PIXBUF_ENABLE_BACKEND
#include <gdk-pixbuf/gdk-pixbuf.h>
#undef GDK_PIXBUF_ENABLE_BACKEND

// Decoder state for loading a single static JPEG XL image.
typedef struct {
  // GDK interface implementation callbacks.
  GdkPixbufModuleSizeFunc image_size_callback;
  GdkPixbufModulePreparedFunc pixbuf_prepared_callback;
  GdkPixbufModuleUpdatedFunc area_updated_callback;
  gpointer user_data;

  // Output pixbuf for the decoded image.
  GdkPixbuf *pixbuf;

  // JPEG XL decoder and related structures.
  JxlParallelRunner *parallel_runner;
  JxlDecoder *decoder;
  JxlPixelFormat pixel_format;

  // Decoding is `done` when JXL_DEC_SUCCESS is received; calling
  // load_increment afterwards gives an error.
  gboolean done;

  // Image information.
  size_t xsize;
  size_t ysize;
  gboolean has_alpha;

  gchar *icc_base64;
} JxlDecoderState;

static void jxl_decoder_state_free(JxlDecoderState *state) {
  if (state == NULL) return;
  if (state->pixbuf != NULL) {
    g_object_unref(state->pixbuf);
  }
  JxlResizableParallelRunnerDestroy(state->parallel_runner);
  JxlDecoderDestroy(state->decoder);
  g_free(state->icc_base64);
  g_free(state);
}

static gpointer begin_load(GdkPixbufModuleSizeFunc size_func,
                            GdkPixbufModulePreparedFunc prepare_func,
                            GdkPixbufModuleUpdatedFunc update_func,
                            gpointer user_data, GError **error) {
  JxlDecoderState *decoder_state = g_new0(JxlDecoderState, 1);
  if (decoder_state == NULL) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "Creation of the decoder state failed");
    return NULL;
  }
  decoder_state->image_size_callback = size_func;
  decoder_state->pixbuf_prepared_callback = prepare_func;
  decoder_state->area_updated_callback = update_func;
  decoder_state->user_data = user_data;

  if (!(decoder_state->parallel_runner =
            JxlResizableParallelRunnerCreate(NULL))) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "Creation of the JXL parallel runner failed");
    goto cleanup;
  }

  if (!(decoder_state->decoder = JxlDecoderCreate(NULL))) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "Creation of the JXL decoder failed");
    goto cleanup;
  }

  JxlDecoderStatus status;

  if ((status = JxlDecoderSetParallelRunner(
           decoder_state->decoder, JxlResizableParallelRunner,
           decoder_state->parallel_runner)) != JXL_DEC_SUCCESS) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "JxlDecoderSetParallelRunner failed: %x", status);
    goto cleanup;
  }
  if ((status = JxlDecoderSubscribeEvents(
           decoder_state->decoder, JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING |
                                       JXL_DEC_FULL_IMAGE)) !=
      JXL_DEC_SUCCESS) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "JxlDecoderSubscribeEvents failed: %x", status);
    goto cleanup;
  }

  decoder_state->pixel_format.data_type = JXL_TYPE_FLOAT;
  decoder_state->pixel_format.endianness = JXL_NATIVE_ENDIAN;

  return decoder_state;
cleanup:
  jxl_decoder_state_free(decoder_state);
  return NULL;
}

static gboolean stop_load(gpointer context, GError **error) {
  JxlDecoderState *decoder_state = context;
  jxl_decoder_state_free(decoder_state);
  return TRUE;
}

static gboolean load_increment(gpointer context, const guchar *buf, guint size,
                                GError **error) {
  JxlDecoderState *decoder_state = context;
  if (decoder_state->done == TRUE) {
    g_warning_once("Trailing data found at end of JXL file");
    return TRUE;
  }

  JxlDecoderStatus status;

  if ((status = JxlDecoderSetInput(decoder_state->decoder, buf, size)) !=
      JXL_DEC_SUCCESS) {
    // Should never happen if things are done properly.
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "JXL decoder logic error: %x", status);
    return FALSE;
  }

  for (;;) {
    status = JxlDecoderProcessInput(decoder_state->decoder);
    switch (status) {
      case JXL_DEC_NEED_MORE_INPUT: {
        JxlDecoderReleaseInput(decoder_state->decoder);
        return TRUE;
      }

      case JXL_DEC_BASIC_INFO: {
        JxlBasicInfo info;
        if (JxlDecoderGetBasicInfo(decoder_state->decoder, &info) !=
            JXL_DEC_SUCCESS) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "JXLDecoderGetBasicInfo failed");
          return FALSE;
        }
        decoder_state->pixel_format.num_channels = info.alpha_bits > 0 ? 4 : 3;
        decoder_state->xsize = info.xsize;
        decoder_state->ysize = info.ysize;
        decoder_state->has_alpha = info.alpha_bits > 0;
        gint width = info.xsize;
        gint height = info.ysize;
        if (decoder_state->image_size_callback) {
          decoder_state->image_size_callback(&width, &height,
                                             decoder_state->user_data);
        }

        // GDK convention for signaling being interested only in the basic info.
        if (width == 0 || height == 0) {
          decoder_state->done = TRUE;
          return TRUE;
        }

        // Set an appropriate number of threads for the image size.
        JxlResizableParallelRunnerSetThreads(
            decoder_state->parallel_runner,
            JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));
        break;
      }

      case JXL_DEC_COLOR_ENCODING: {
        // Get the ICC color profile of the pixel data
        gpointer icc_buff;
        size_t icc_size;
        JxlColorEncoding color_encoding;
        if (JXL_DEC_SUCCESS == JxlDecoderGetColorAsEncodedProfile(
                                   decoder_state->decoder,
                                   JXL_COLOR_PROFILE_TARGET_ORIGINAL,
                                   &color_encoding)) {
          // we don't check the return status here because it's not a problem if
          // this fails
          JxlDecoderSetPreferredColorProfile(decoder_state->decoder,
                                             &color_encoding);
        }
        if (JXL_DEC_SUCCESS != JxlDecoderGetICCProfileSize(
                                   decoder_state->decoder,
                                   JXL_COLOR_PROFILE_TARGET_DATA, &icc_size)) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "JxlDecoderGetICCProfileSize failed");
          return FALSE;
        }
        if (!(icc_buff = g_malloc(icc_size))) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "Allocating ICC profile failed");
          return FALSE;
        }
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetColorAsICCProfile(decoder_state->decoder,
                                           JXL_COLOR_PROFILE_TARGET_DATA,
                                           icc_buff, icc_size)) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "JxlDecoderGetColorAsICCProfile failed");
          g_free(icc_buff);
          return FALSE;
        }
        decoder_state->icc_base64 = g_base64_encode(icc_buff, icc_size);
        g_free(icc_buff);
        if (!decoder_state->icc_base64) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "Allocating ICC profile base64 string failed");
          return FALSE;
        }

        // Allocate the output pixbuf after color info is known.
        decoder_state->pixbuf =
            gdk_pixbuf_new(GDK_COLORSPACE_RGB, decoder_state->has_alpha,
                           /*bits_per_sample=*/8, decoder_state->xsize,
                           decoder_state->ysize);
        if (decoder_state->pixbuf == NULL) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "Failed to allocate output pixel buffer");
          return FALSE;
        }
        gdk_pixbuf_set_option(decoder_state->pixbuf, "icc-profile",
                              decoder_state->icc_base64);
        decoder_state->pixel_format.align =
            gdk_pixbuf_get_rowstride(decoder_state->pixbuf);
        decoder_state->pixel_format.data_type = JXL_TYPE_UINT8;

        if (decoder_state->pixbuf_prepared_callback) {
          decoder_state->pixbuf_prepared_callback(
              decoder_state->pixbuf, NULL, decoder_state->user_data);
        }
        break;
      }

      case JXL_DEC_NEED_IMAGE_OUT_BUFFER: {
        decoder_state->pixel_format.align =
            gdk_pixbuf_get_rowstride(decoder_state->pixbuf);
        guint buf_size;
        guchar *dst =
            gdk_pixbuf_get_pixels_with_length(decoder_state->pixbuf, &buf_size);
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSetImageOutBuffer(decoder_state->decoder,
                                        &decoder_state->pixel_format, dst,
                                        buf_size)) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "JxlDecoderSetImageOutBuffer failed");
          return FALSE;
        }
        break;
      }

      case JXL_DEC_FULL_IMAGE: {
        if (decoder_state->area_updated_callback) {
          decoder_state->area_updated_callback(
              decoder_state->pixbuf, 0, 0,
              gdk_pixbuf_get_width(decoder_state->pixbuf),
              gdk_pixbuf_get_height(decoder_state->pixbuf),
              decoder_state->user_data);
        }
        break;
      }

      case JXL_DEC_SUCCESS: {
        decoder_state->done = TRUE;
        return TRUE;
      }

      default: {
        g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                    "Unexpected JxlDecoderProcessInput return code: %x",
                    status);
        return FALSE;
      }
    }
  }
  return TRUE;
}

static gboolean jxl_is_save_option_supported(const gchar *option_key) {
  if (g_strcmp0(option_key, "quality") == 0) {
    return TRUE;
  }

  return FALSE;
}

static gboolean jxl_image_saver(FILE *f, GdkPixbuf *pixbuf, gchar **keys,
                                gchar **values, GError **error) {
  long quality = 90; /* default; must be between 0 and 100 */
  double distance;
  gboolean save_alpha;
  JxlEncoder *encoder;
  void *parallel_runner;
  JxlEncoderFrameSettings *frame_settings;
  JxlBasicInfo output_info;
  JxlPixelFormat pixel_format;
  JxlColorEncoding color_profile;
  JxlEncoderStatus status;

  GByteArray *compressed;
  size_t offset = 0;
  uint8_t *next_out;
  size_t avail_out;

  if (f == NULL || pixbuf == NULL) {
    return FALSE;
  }

  if (keys && *keys) {
    gchar **kiter = keys;
    gchar **viter = values;

    while (*kiter) {
      if (strcmp(*kiter, "quality") == 0) {
        char *endptr = NULL;
        quality = strtol(*viter, &endptr, 10);

        if (endptr == *viter) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_BAD_OPTION,
                      "JXL quality must be a value between 0 and 100; value "
                      "\"%s\" could not be parsed.",
                      *viter);

          return FALSE;
        }

        if (quality < 0 || quality > 100) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_BAD_OPTION,
                      "JXL quality must be a value between 0 and 100; value "
                      "\"%ld\" is not allowed.",
                      quality);

          return FALSE;
        }
      } else {
        g_warning("Unrecognized parameter (%s) passed to JXL saver.", *kiter);
      }

      ++kiter;
      ++viter;
    }
  }

  if (gdk_pixbuf_get_bits_per_sample(pixbuf) != 8) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_UNKNOWN_TYPE,
                "Sorry, only 8bit images are supported by this JXL saver");
    return FALSE;
  }

  JxlEncoderInitBasicInfo(&output_info);
  output_info.have_container = JXL_FALSE;
  output_info.xsize = gdk_pixbuf_get_width(pixbuf);
  output_info.ysize = gdk_pixbuf_get_height(pixbuf);
  output_info.bits_per_sample = 8;
  output_info.orientation = JXL_ORIENT_IDENTITY;
  output_info.num_color_channels = 3;

  if (output_info.xsize == 0 || output_info.ysize == 0) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_CORRUPT_IMAGE,
                "Empty image, nothing to save");
    return FALSE;
  }

  save_alpha = gdk_pixbuf_get_has_alpha(pixbuf);

  pixel_format.data_type = JXL_TYPE_UINT8;
  pixel_format.endianness = JXL_NATIVE_ENDIAN;
  pixel_format.align = gdk_pixbuf_get_rowstride(pixbuf);

  if (save_alpha) {
    if (gdk_pixbuf_get_n_channels(pixbuf) != 4) {
      g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_UNKNOWN_TYPE,
                  "Unsupported number of channels");
      return FALSE;
    }

    output_info.num_extra_channels = 1;
    output_info.alpha_bits = 8;
    pixel_format.num_channels = 4;
  } else {
    if (gdk_pixbuf_get_n_channels(pixbuf) != 3) {
      g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_UNKNOWN_TYPE,
                  "Unsupported number of channels");
      return FALSE;
    }

    output_info.num_extra_channels = 0;
    output_info.alpha_bits = 0;
    pixel_format.num_channels = 3;
  }

  encoder = JxlEncoderCreate(NULL);
  if (!encoder) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "Creation of the JXL encoder failed");
    return FALSE;
  }

  parallel_runner = JxlResizableParallelRunnerCreate(NULL);
  if (!parallel_runner) {
    JxlEncoderDestroy(encoder);
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "Creation of the JXL decoder failed");
    return FALSE;
  }

  JxlResizableParallelRunnerSetThreads(
      parallel_runner, JxlResizableParallelRunnerSuggestThreads(
                           output_info.xsize, output_info.ysize));

  status = JxlEncoderSetParallelRunner(encoder, JxlResizableParallelRunner,
                                       parallel_runner);
  if (status != JXL_ENC_SUCCESS) {
    JxlResizableParallelRunnerDestroy(parallel_runner);
    JxlEncoderDestroy(encoder);
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "JxlDecoderSetParallelRunner failed: %x", status);
    return FALSE;
  }

  if (quality > 99) {
    output_info.uses_original_profile = JXL_TRUE;
    distance = 0;
  } else {
    output_info.uses_original_profile = JXL_FALSE;
    distance = JxlEncoderDistanceFromQuality((float)quality);
  }

  status = JxlEncoderSetBasicInfo(encoder, &output_info);
  if (status != JXL_ENC_SUCCESS) {
    JxlResizableParallelRunnerDestroy(parallel_runner);
    JxlEncoderDestroy(encoder);
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "JxlEncoderSetBasicInfo failed: %x", status);
    return FALSE;
  }

  JxlColorEncodingSetToSRGB(&color_profile, JXL_FALSE);
  status = JxlEncoderSetColorEncoding(encoder, &color_profile);
  if (status != JXL_ENC_SUCCESS) {
    JxlResizableParallelRunnerDestroy(parallel_runner);
    JxlEncoderDestroy(encoder);
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "JxlEncoderSetColorEncoding failed: %x", status);
    return FALSE;
  }

  frame_settings = JxlEncoderFrameSettingsCreate(encoder, NULL);
  if (frame_settings == NULL) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "JxlEncoderFrameSettingsCreate failed");
  }
  JxlEncoderSetFrameDistance(frame_settings, distance);
  JxlEncoderSetFrameLossless(frame_settings, output_info.uses_original_profile);

  status = JxlEncoderAddImageFrame(frame_settings, &pixel_format,
                                   gdk_pixbuf_read_pixels(pixbuf),
                                   gdk_pixbuf_get_byte_length(pixbuf));
  if (status != JXL_ENC_SUCCESS) {
    JxlResizableParallelRunnerDestroy(parallel_runner);
    JxlEncoderDestroy(encoder);
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "JxlEncoderAddImageFrame failed: %x", status);
    return FALSE;
  }

  JxlEncoderCloseInput(encoder);

  compressed = g_byte_array_sized_new(4096);
  g_byte_array_set_size(compressed, 4096);
  do {
    next_out = compressed->data + offset;
    avail_out = compressed->len - offset;
    status = JxlEncoderProcessOutput(encoder, &next_out, &avail_out);

    if (status == JXL_ENC_NEED_MORE_OUTPUT) {
      offset = next_out - compressed->data;
      g_byte_array_set_size(compressed, compressed->len * 2);
    } else if (status == JXL_ENC_ERROR) {
      JxlResizableParallelRunnerDestroy(parallel_runner);
      JxlEncoderDestroy(encoder);
      g_set_error(error, G_FILE_ERROR, 0, "JxlEncoderProcessOutput failed: %x",
                  status);
      return FALSE;
    }
  } while (status != JXL_ENC_SUCCESS);

  JxlResizableParallelRunnerDestroy(parallel_runner);
  JxlEncoderDestroy(encoder);

  g_byte_array_set_size(compressed, next_out - compressed->data);
  if (compressed->len > 0) {
    fwrite(compressed->data, 1, compressed->len, f);
    g_byte_array_free(compressed, TRUE);
    return TRUE;
  }

  return FALSE;
}

void fill_vtable(GdkPixbufModule *module) {
  module->begin_load = begin_load;
  module->stop_load = stop_load;
  module->load_increment = load_increment;
  module->is_save_option_supported = jxl_is_save_option_supported;
  module->save = jxl_image_saver;
}

void fill_info(GdkPixbufFormat *info) {
  static GdkPixbufModulePattern signature[] = {
      {"\xFF\x0A", "  ", 100},
      {"...\x0CJXL \x0D\x0A\x87\x0A", "zzz         ", 100},
      {NULL, NULL, 0},
  };

  static gchar *mime_types[] = {"image/jxl", NULL};

  static gchar *extensions[] = {"jxl", NULL};

  info->name = "jxl";
  info->signature = signature;
  info->description = "JPEG XL image";
  info->mime_types = mime_types;
  info->extensions = extensions;
  info->flags = GDK_PIXBUF_FORMAT_WRITABLE | GDK_PIXBUF_FORMAT_THREADSAFE;
  info->license = "BSD-3";
}
