// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define GDK_PIXBUF_ENABLE_BACKEND
#include <gdk-pixbuf/gdk-pixbuf.h>
#undef GDK_PIXBUF_ENABLE_BACKEND

#include "jxl/codestream_header.h"
#include "jxl/decode.h"
#include "jxl/resizable_parallel_runner.h"
#include "jxl/types.h"
#include "skcms.h"

typedef struct {
  GdkPixbufModuleSizeFunc image_size_callback;
  GdkPixbufModulePreparedFunc pixbuf_prepared_callback;
  GdkPixbufModuleUpdatedFunc area_updated_callback;
  gpointer user_data;

  GdkPixbuf *output;

  JxlParallelRunner *parallel_runner;
  JxlDecoder *decoder;
  JxlPixelFormat pixel_format;

  gboolean done;
  gboolean alpha_premultiplied;

  gpointer icc_buff;
  skcms_ICCProfile icc;

} GdkPixbufJXLDecoderState;

static gpointer begin_load(GdkPixbufModuleSizeFunc size_func,
                           GdkPixbufModulePreparedFunc prepare_func,
                           GdkPixbufModuleUpdatedFunc update_func,
                           gpointer user_data, GError **error) {
  GdkPixbufJXLDecoderState *decoder_state =
      g_malloc0(sizeof(GdkPixbufJXLDecoderState));
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
           decoder_state->decoder,
           JXL_DEC_BASIC_INFO | JXL_DEC_COLOR_ENCODING | JXL_DEC_FULL_IMAGE)) !=
      JXL_DEC_SUCCESS) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "JxlDecoderSubscribeEvents failed: %x", status);
    goto cleanup;
  }

  decoder_state->pixel_format.data_type = JXL_TYPE_FLOAT;
  decoder_state->pixel_format.endianness = JXL_NATIVE_ENDIAN;

  return decoder_state;
cleanup:
  JxlResizableParallelRunnerDestroy(decoder_state->parallel_runner);
  JxlDecoderDestroy(decoder_state->decoder);
  g_free(decoder_state);
  return NULL;
}

static gboolean stop_load(gpointer context, GError **error) {
  GdkPixbufJXLDecoderState *decoder_state = context;
  if (decoder_state->output) {
    g_object_unref(decoder_state->output);
  }
  JxlResizableParallelRunnerDestroy(decoder_state->parallel_runner);
  JxlDecoderDestroy(decoder_state->decoder);
  g_free(decoder_state->icc_buff);
  g_free(decoder_state);
  return TRUE;
}

static void draw_pixels(void *context, size_t x, size_t y, size_t num_pixels,
                        const void *pixels) {
  GdkPixbufJXLDecoderState *decoder_state = context;
  gboolean has_alpha = decoder_state->pixel_format.num_channels == 4;

  guchar *dst = gdk_pixbuf_get_pixels(decoder_state->output) +
                decoder_state->pixel_format.num_channels * x +
                gdk_pixbuf_get_rowstride(decoder_state->output) * y;

  skcms_Transform(
      pixels,
      has_alpha ? skcms_PixelFormat_RGBA_ffff : skcms_PixelFormat_RGB_fff,
      decoder_state->alpha_premultiplied ? skcms_AlphaFormat_PremulAsEncoded
                                         : skcms_AlphaFormat_Unpremul,
      &decoder_state->icc, dst,
      has_alpha ? skcms_PixelFormat_RGBA_8888 : skcms_PixelFormat_RGB_888,
      skcms_AlphaFormat_Unpremul, skcms_sRGB_profile(), num_pixels);
}

static gboolean load_increment(gpointer context, const guchar *buf, guint size,
                               GError **error) {
  GdkPixbufJXLDecoderState *decoder_state = context;
  if (decoder_state->done == TRUE) {
    g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                "JXL decoder load_increment called after end of file");
    return FALSE;
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
      case JXL_DEC_SUCCESS: {
        decoder_state->done = TRUE;
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
        decoder_state->alpha_premultiplied = info.alpha_premultiplied;
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
        // TODO(veluca): support rescaling.
        decoder_state->output =
            gdk_pixbuf_new(GDK_COLORSPACE_RGB, info.alpha_bits > 0,
                           /*bits_per_sample=*/8, info.xsize, info.ysize);
        if (decoder_state->output == NULL) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "Failed to allocate output pixel buffer");
          return FALSE;
        }
        // Set an appropriate number of threads for the image size.
        JxlResizableParallelRunnerSetThreads(
            decoder_state->parallel_runner,
            JxlResizableParallelRunnerSuggestThreads(info.xsize, info.ysize));

        if (decoder_state->pixbuf_prepared_callback) {
          decoder_state->pixbuf_prepared_callback(decoder_state->output, NULL,
                                                  decoder_state->user_data);
        }
        decoder_state->pixel_format.align =
            gdk_pixbuf_get_rowstride(decoder_state->output);
        break;
      }
      case JXL_DEC_COLOR_ENCODING: {
        // Get the ICC color profile of the pixel data
        size_t icc_size;
        if (JXL_DEC_SUCCESS != JxlDecoderGetICCProfileSize(
                                   decoder_state->decoder,
                                   &decoder_state->pixel_format,
                                   JXL_COLOR_PROFILE_TARGET_DATA, &icc_size)) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "JxlDecoderGetICCProfileSize failed");
          return FALSE;
        }
        if (!(decoder_state->icc_buff = g_malloc(icc_size))) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "Allocating ICC profile failed");
          return FALSE;
        }
        if (JXL_DEC_SUCCESS !=
            JxlDecoderGetColorAsICCProfile(decoder_state->decoder,
                                           &decoder_state->pixel_format,
                                           JXL_COLOR_PROFILE_TARGET_DATA,
                                           decoder_state->icc_buff, icc_size)) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "JxlDecoderGetColorAsICCProfile failed");
          return FALSE;
        }
        if (!skcms_Parse(decoder_state->icc_buff, icc_size,
                         &decoder_state->icc)) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "Invalid ICC profile from JXL image decoder");
          return FALSE;
        }
        break;
      }
      case JXL_DEC_NEED_IMAGE_OUT_BUFFER: {
        if (JXL_DEC_SUCCESS !=
            JxlDecoderSetImageOutCallback(decoder_state->decoder,
                                          &decoder_state->pixel_format,
                                          draw_pixels, decoder_state)) {
          g_set_error(error, GDK_PIXBUF_ERROR, GDK_PIXBUF_ERROR_FAILED,
                      "JxlDecoderSetImageOutCallback failed");
          return FALSE;
        }
        break;
      }
      case JXL_DEC_FULL_IMAGE: {
        // TODO(veluca): consider doing partial updates.
        if (decoder_state->area_updated_callback) {
          decoder_state->area_updated_callback(
              decoder_state->output, 0, 0,
              gdk_pixbuf_get_width(decoder_state->output),
              gdk_pixbuf_get_height(decoder_state->output),
              decoder_state->user_data);
        }
        // TODO(veluca): for animations, this will eventually only leave the
        // last frame.
        break;
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

void fill_vtable(GdkPixbufModule *module) {
  module->begin_load = begin_load;
  module->stop_load = stop_load;
  module->load_increment = load_increment;
  // TODO(veluca): implement animation and saving.
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
  // TODO(veluca): add writing support.
  info->flags = GDK_PIXBUF_FORMAT_THREADSAFE;
  info->license = "BSD-3";
}
