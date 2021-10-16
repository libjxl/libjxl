// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "plugins/gimp/file-jxl-save.h"

#include <cmath>

#define PLUG_IN_BINARY "file-jxl"
#define SAVE_PROC "file-jxl-save"

#define SCALE_WIDTH 300

namespace jxl {

namespace {

#ifndef g_clear_signal_handler
#include "gobject/gsignal.h"
// g_clear_signal_handler was added in glib 2.62
void g_clear_signal_handler(gulong* handler, gpointer instance) {
  if (handler != nullptr && *handler != 0) {
    g_signal_handler_disconnect(instance, *handler);
    *handler = 0;
  }
}
#endif  // g_clear_signal_handler

class JpegXlSaveOpts {
 public:
  float distance;
  float quality;

  bool is_linear = false;
  bool has_alpha = false;
  bool is_gray = false;

  bool use_container = true;
  int encoding_effort = 7;
  int faster_decoding = 0;

  JxlPixelFormat pixel_format;
  JxlBasicInfo basic_info;

  // functions
  JpegXlSaveOpts();

  bool SetDistance(float dist);
  bool SetQuality(float qual);
  bool SetDimensions(int x, int y);
  bool SetNumChannels(int channels);

  bool UpdateDistance();
  bool UpdateQuality();

  bool SetPrecision(int gimp_precision);

 private:
};  // class JpegXlSaveOpts

JpegXlSaveOpts jxl_save_opts;

class JpegXlSaveGui {
 public:
  bool SaveDialog();

 private:
  GtkAdjustment* entry_distance = nullptr;
  GtkAdjustment* entry_quality = nullptr;
  GtkAdjustment* entry_effort = nullptr;
  gulong handle_entry_quality = 0;
  gulong handle_entry_distance = 0;

  static bool GuiOnChangeQuality(GtkAdjustment* adj_qual, void* this_pointer);
  static bool GuiOnChangeDistance(GtkAdjustment* adj_dist, void* this_pointer);
  static bool GuiOnChangeEffort(GtkAdjustment* adj_effort);
};  // class JpegXlSaveGui

JpegXlSaveGui jxl_save_gui;

bool JpegXlSaveGui::GuiOnChangeQuality(GtkAdjustment* adj_qual,
                                       void* this_pointer) {
  JpegXlSaveGui* self = static_cast<JpegXlSaveGui*>(this_pointer);

  g_clear_signal_handler(&self->handle_entry_distance, self->entry_distance);
  g_clear_signal_handler(&self->handle_entry_quality, self->entry_quality);

  GtkAdjustment* adj_dist = self->entry_distance;
  jxl_save_opts.SetQuality(gtk_adjustment_get_value(adj_qual));
  gtk_adjustment_set_value(adj_dist, jxl_save_opts.distance);

  self->handle_entry_distance =
      g_signal_connect(self->entry_distance, "value-changed",
                       G_CALLBACK(GuiOnChangeDistance), self);
  self->handle_entry_quality =
      g_signal_connect(self->entry_quality, "value-changed",
                       G_CALLBACK(GuiOnChangeQuality), self);
  return true;
}

bool JpegXlSaveGui::GuiOnChangeDistance(GtkAdjustment* adj_dist,
                                        void* this_pointer) {
  JpegXlSaveGui* self = static_cast<JpegXlSaveGui*>(this_pointer);
  GtkAdjustment* adj_qual = self->entry_quality;

  g_clear_signal_handler(&self->handle_entry_distance, self->entry_distance);
  g_clear_signal_handler(&self->handle_entry_quality, self->entry_quality);

  jxl_save_opts.SetDistance(gtk_adjustment_get_value(adj_dist));
  gtk_adjustment_set_value(adj_qual, jxl_save_opts.quality);

  self->handle_entry_distance =
      g_signal_connect(self->entry_distance, "value-changed",
                       G_CALLBACK(GuiOnChangeDistance), self);
  self->handle_entry_quality =
      g_signal_connect(self->entry_quality, "value-changed",
                       G_CALLBACK(GuiOnChangeQuality), self);
  return true;
}

bool JpegXlSaveGui::GuiOnChangeEffort(GtkAdjustment* adj_effort) {
  float new_effort = 10 - gtk_adjustment_get_value(adj_effort);
  jxl_save_opts.encoding_effort = new_effort;
  return true;
}

bool JpegXlSaveGui::SaveDialog() {
  gboolean run;
  GtkWidget* dialog;
  GtkWidget* content_area;
  GtkWidget* main_vbox;
  GtkWidget* frame;
  GtkWidget* table;
  GtkWidget* vbox;
  GtkWidget* separator;

  // initialize export dialog
  gimp_ui_init(PLUG_IN_BINARY, true);
  dialog = gimp_export_dialog_new("JPEG XL", PLUG_IN_BINARY, SAVE_PROC);

  gtk_window_set_resizable(GTK_WINDOW(dialog), false);
  content_area = gimp_export_dialog_get_content_area(dialog);

  main_vbox = gtk_vbox_new(false, 6);
  gtk_container_set_border_width(GTK_CONTAINER(main_vbox), 6);
  gtk_box_pack_start(GTK_BOX(content_area), main_vbox, true, true, 0);
  gtk_widget_show(main_vbox);

  // Standard Settings Frame
  frame = gtk_frame_new(nullptr);
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_ETCHED_IN);
  gtk_box_pack_start(GTK_BOX(main_vbox), frame, false, false, 0);
  gtk_widget_show(frame);

  vbox = gtk_vbox_new(false, 6);
  gtk_container_set_border_width(GTK_CONTAINER(vbox), 6);
  gtk_container_add(GTK_CONTAINER(frame), vbox);
  gtk_widget_show(vbox);

  // Layout Table
  table = gtk_table_new(20, 3, false);
  gtk_table_set_col_spacings(GTK_TABLE(table), 6);
  gtk_box_pack_start(GTK_BOX(vbox), table, false, false, 0);
  gtk_widget_show(table);

  // Distance Slider
  gchar* distance_help = nullptr;
  distance_help =
      _("Butteraugli distance target.  Suggested values:"
        "\n\td\u00A0=\u00A00.3\tExcellent"
        "\n\td\u00A0=\u00A01\tVery Good"
        "\n\td\u00A0=\u00A02\tGood"
        "\n\td\u00A0=\u00A03\tFair"
        "\n\td\u00A0=\u00A06\tPoor");
  entry_distance = (GtkAdjustment*)gimp_scale_entry_new(
      GTK_TABLE(table), 0, 0, _("Distance"), SCALE_WIDTH, 0,
      jxl_save_opts.distance, 0.0, 15.0, 0.001, 1.0, 3, true, 0.0, 0.0,
      distance_help, SAVE_PROC);
  gimp_scale_entry_set_logarithmic((GtkObject*)entry_distance, true);

  // Quality Slider
  gchar* quality_help = nullptr;
  quality_help =
      _("JPEG-style Quality is remapped to distance.  "
        "Values roughly match libjpeg quality settings.");
  entry_quality = (GtkAdjustment*)gimp_scale_entry_new(
      GTK_TABLE(table), 0, 1, _("Quality"), SCALE_WIDTH, 0,
      jxl_save_opts.quality, 8.26, 100.0, 1.0, 10.0, 2, true, 0.0, 0.0,
      quality_help, SAVE_PROC);

  // Distance and Quality Signals
  handle_entry_distance = g_signal_connect(
      entry_distance, "value-changed", G_CALLBACK(GuiOnChangeDistance), this);
  handle_entry_quality = g_signal_connect(entry_quality, "value-changed",
                                          G_CALLBACK(GuiOnChangeQuality), this);

  // ----------
  separator = gtk_vseparator_new();
  gtk_table_attach(GTK_TABLE(table), separator, 0, 2, 2, 3, GTK_EXPAND,
                   GTK_EXPAND, 9, 9);
  gtk_widget_show(separator);

  // Encoding Effort / Speed
  gchar* effort_help = nullptr;
  effort_help =
      _("Adjust encoding speed.  Higher values are faster because "
        "the encoder uses less effort to hit distance targets.  "
        "As\u00A0a\u00A0result, image quality may be decreased.  "
        "Default\u00A0=\u00A03.");
  entry_effort = (GtkAdjustment*)gimp_scale_entry_new(
      GTK_TABLE(table), 0, 3, _("Speed"), SCALE_WIDTH, 0,
      10 - jxl_save_opts.encoding_effort, 1, 9, 1, 2, 0, true, 0.0, 0.0,
      effort_help, SAVE_PROC);

  // effort signal
  g_signal_connect(entry_effort, "value-changed", G_CALLBACK(GuiOnChangeEffort),
                   nullptr);

  // show dialog
  gtk_widget_show(dialog);

  GtkAllocation allocation;
  gtk_widget_get_allocation(dialog, &allocation);

  // int height = allocation.height;
  // int width = allocation.width;
  // gtk_widget_set_size_request(dialog, height * 1.5, height);

  run = (gimp_dialog_run(GIMP_DIALOG(dialog)) == GTK_RESPONSE_OK);
  gtk_widget_destroy(dialog);

  g_free(effort_help);
  g_free(quality_help);
  g_free(distance_help);

  return run;
}  // JpegXlSaveGui::SaveDialog

JpegXlSaveOpts::JpegXlSaveOpts() {
  SetDistance(1.0);

  pixel_format.num_channels = 4;
  pixel_format.data_type = JXL_TYPE_FLOAT;
  pixel_format.endianness = JXL_NATIVE_ENDIAN;
  pixel_format.align = 0;

  JxlEncoderInitBasicInfo(&basic_info);
  return;
}  // JpegXlSaveOpts constructor

bool JpegXlSaveOpts::SetDistance(float dist) {
  distance = dist;
  return UpdateQuality();
}

bool JpegXlSaveOpts::SetQuality(float qual) {
  quality = qual;
  return UpdateDistance();
}

bool JpegXlSaveOpts::UpdateQuality() {
  float qual;

  if (distance < 0.1) {
    qual = 100;
  } else if (distance > 6.56) {
    qual = 30 - 5 * log(abs(6.25 * distance - 40)) / log(2.5);
  } else {
    qual = 100 - (distance - 0.1) / 0.09;
  }

  if (qual < 0) {
    quality = 0.0;
  } else if (qual >= 100) {
    quality = 100.0;
  } else {
    quality = qual;
  }

  return true;
}

bool JpegXlSaveOpts::UpdateDistance() {
  float dist;
  if (quality >= 30) {
    dist = 0.1 + (100 - quality) * 0.09;
  } else {
    dist = 6.4 + pow(2.5, (30 - quality) / 5.0) / 6.25;
  }

  if (dist > 15) {
    distance = 15;
  } else {
    distance = dist;
  }
  return true;
}

bool JpegXlSaveOpts::SetDimensions(int x, int y) {
  basic_info.xsize = x;
  basic_info.ysize = y;
  return true;
}

bool JpegXlSaveOpts::SetNumChannels(int channels) {
  switch (channels) {
    case 1:
      pixel_format.num_channels = 1;
      basic_info.num_color_channels = 1;
      basic_info.num_extra_channels = 0;
      basic_info.alpha_bits = 0;
      basic_info.alpha_exponent_bits = 0;
      break;
    case 2:
      pixel_format.num_channels = 2;
      basic_info.num_color_channels = 1;
      basic_info.num_extra_channels = 1;
      basic_info.alpha_bits = int(std::fmin(16, basic_info.bits_per_sample));
      basic_info.alpha_exponent_bits = 0;
      break;
    case 3:
      pixel_format.num_channels = 3;
      basic_info.num_color_channels = 3;
      basic_info.num_extra_channels = 0;
      basic_info.alpha_bits = 0;
      basic_info.alpha_exponent_bits = 0;
      break;
    case 4:
      pixel_format.num_channels = 4;
      basic_info.num_color_channels = 3;
      basic_info.num_extra_channels = 1;
      basic_info.alpha_bits = int(std::fmin(16, basic_info.bits_per_sample));
      basic_info.alpha_exponent_bits = 0;
      break;
    default:
      SetNumChannels(3);
  }  // switch
  return true;
}  // JpegXlSaveOpts::SetNumChannels

bool JpegXlSaveOpts::SetPrecision(int gimp_precision) {
  // Note: GIMP pixel format cannot be used to set is_linear
  // It is not accurate.
  switch (gimp_precision) {
    case GIMP_PRECISION_HALF_GAMMA:
    case GIMP_PRECISION_HALF_LINEAR:
      basic_info.bits_per_sample = 16;
      basic_info.exponent_bits_per_sample = 5;
      break;

    // UINT32 not supported by encoder; using FLOAT instead
    case GIMP_PRECISION_U32_GAMMA:
    case GIMP_PRECISION_U32_LINEAR:
    case GIMP_PRECISION_FLOAT_GAMMA:
    case GIMP_PRECISION_FLOAT_LINEAR:
      basic_info.bits_per_sample = 32;
      basic_info.exponent_bits_per_sample = 8;
      break;

    case GIMP_PRECISION_U16_GAMMA:
    case GIMP_PRECISION_U16_LINEAR:
      basic_info.bits_per_sample = 16;
      basic_info.exponent_bits_per_sample = 0;
      break;

    case GIMP_PRECISION_U8_LINEAR:
    case GIMP_PRECISION_U8_GAMMA:
    default:
      basic_info.bits_per_sample = 8;
      basic_info.exponent_bits_per_sample = 0;
      break;
  }
  return true;
}  // JpegXlSaveOpts::SetPrecision

}  // namespace

bool SaveJpegXlImage(const gint32 image_id, const gint32 drawable_id,
                     const gint32 orig_image_id, const gchar* const filename) {
  if (!jxl_save_gui.SaveDialog()) {
    return true;
  }

  gint32 nlayers;
  gint32* layers;
  gint32 duplicate = gimp_image_duplicate(image_id);

  JpegXlGimpProgress gimp_save_progress(_("Saving JPEG XL file: %s"), filename);
  gimp_save_progress.update();

  GimpColorProfile* profile = gimp_image_get_effective_color_profile(image_id);
  jxl_save_opts.is_gray = gimp_color_profile_is_gray(profile);

  // TODO: Figure out a better way to determine whether an image is linear
  // gimp_color_profile_is_linear() and functions to get babl format often
  // report incorrect result.
  const char* profile_description = gimp_color_profile_get_label(profile);
  if (g_regex_match_simple("\\blinear\\b", profile_description,
                           G_REGEX_CASELESS, GRegexMatchFlags(0))) {
    jxl_save_opts.is_linear = true;
  } else {
    jxl_save_opts.is_linear = false;
  }

  gimp_save_progress.update();

  jxl_save_opts.SetDimensions(gimp_image_width(image_id),
                              gimp_image_height(image_id));

  jxl_save_opts.SetPrecision(gimp_image_get_precision(image_id));
  layers = gimp_image_get_layers(duplicate, &nlayers);

  for (int i = 0; i < nlayers; i++) {
    if (gimp_drawable_has_alpha(layers[i])) {
      jxl_save_opts.has_alpha = true;
      break;
    }
  }

  if (jxl_save_opts.basic_info.bits_per_sample < 32 ||
      jxl_save_opts.basic_info.exponent_bits_per_sample == 0) {
    gimp_image_convert_precision(duplicate, GIMP_PRECISION_FLOAT_LINEAR);
  } else {
    // cannot convert from float to float
    gimp_image_convert_precision(duplicate, GIMP_PRECISION_U32_LINEAR);
    gimp_image_convert_precision(duplicate, GIMP_PRECISION_FLOAT_LINEAR);
  }

  // get effective icc profile for lcms2 colorspace conversion
  gsize icc_size;
  const guint8* icc_bytes = nullptr;
  std::vector<uint8_t> icc_effective;
  profile = gimp_image_get_effective_color_profile(image_id);

  icc_bytes = gimp_color_profile_get_icc_profile(profile, &icc_size);

  icc_effective.assign(icc_bytes, icc_bytes + icc_size);

  // setup lcms2 profiles
  cmsContext hContext = cmsCreateContext(nullptr, nullptr);
  cmsHPROFILE hInProfile, hOutProfile;
  hInProfile = cmsOpenProfileFromMemTHR(hContext, icc_effective.data(),
                                        icc_effective.size());
  hOutProfile = cmsCreate_sRGBProfileTHR(hContext);

  const uint32_t flags =
      cmsFLAGS_BLACKPOINTCOMPENSATION | cmsFLAGS_HIGHRESPRECALC;
  cmsHTRANSFORM hTransform;

  if (jxl_save_opts.has_alpha) {
    hTransform = cmsCreateTransformTHR(hContext, hInProfile, TYPE_RGBA_FLT,
                                       hOutProfile, TYPE_RGBA_FLT,
                                       INTENT_ABSOLUTE_COLORIMETRIC, flags);
  } else {
    hTransform = cmsCreateTransformTHR(hContext, hInProfile, TYPE_RGB_FLT,
                                       hOutProfile, TYPE_RGB_FLT,
                                       INTENT_ABSOLUTE_COLORIMETRIC, flags);
  }

  cmsCloseProfile(hInProfile);
  cmsCloseProfile(hOutProfile);

  gimp_save_progress.update();

  // treat layers as animation frames, for now
  if (nlayers > 1) {
    jxl_save_opts.basic_info.have_animation = true;
    jxl_save_opts.basic_info.animation.tps_numerator = 100;
  }

  gimp_save_progress.update();

  // multi-threaded parallel runner.
  auto runner = JxlResizableParallelRunnerMake(nullptr);

  JxlResizableParallelRunnerSetThreads(
      runner.get(),
      JxlResizableParallelRunnerSuggestThreads(jxl_save_opts.basic_info.xsize,
                                               jxl_save_opts.basic_info.ysize));

  auto enc = JxlEncoderMake(/*memory_manager=*/nullptr);
  JxlEncoderUseContainer(enc.get(), jxl_save_opts.use_container);

  if (JXL_ENC_SUCCESS != JxlEncoderSetParallelRunner(enc.get(),
                                                     JxlResizableParallelRunner,
                                                     runner.get())) {
    g_printerr(SAVE_PROC " Error: JxlEncoderSetParallelRunner failed\n");
    return false;
  }

  // set up internal color profile
  JxlColorEncoding color_encoding = {};

  if (jxl_save_opts.is_linear) {
    JxlColorEncodingSetToLinearSRGB(&color_encoding, jxl_save_opts.is_gray);
  } else {
    JxlColorEncodingSetToSRGB(&color_encoding, jxl_save_opts.is_gray);
  }

  if (JXL_ENC_SUCCESS !=
      JxlEncoderSetColorEncoding(enc.get(), &color_encoding)) {
    g_printerr(SAVE_PROC " Warning: JxlEncoderSetColorEncoding failed\n");
  }

  // set encoder options
  JxlEncoderOptions* enc_opts;
  enc_opts = JxlEncoderOptionsCreate(enc.get(), nullptr);

  JxlEncoderOptionsSetEffort(enc_opts, jxl_save_opts.encoding_effort);
  JxlEncoderOptionsSetDecodingSpeed(enc_opts, jxl_save_opts.faster_decoding);

  // lossless mode
  if (jxl_save_opts.distance < 0.01) {
    JxlEncoderOptionsSetDistance(enc_opts, 0);
    JxlEncoderOptionsSetLossless(enc_opts, true);
  } else {
    JxlEncoderOptionsSetLossless(enc_opts, false);
    JxlEncoderOptionsSetDistance(enc_opts, jxl_save_opts.distance);
  }

  jxl_save_opts.SetNumChannels((jxl_save_opts.is_gray ? 1 : 3) +
                               (jxl_save_opts.has_alpha ? 1 : 0));

  if (JXL_ENC_SUCCESS !=
      JxlEncoderSetBasicInfo(enc.get(), &jxl_save_opts.basic_info)) {
    g_printerr(SAVE_PROC " Error: JxlEncoderSetBasicInfo failed\n");
    return false;
  }

  // process layers and compress into JXL
  size_t buffer_size =
      jxl_save_opts.basic_info.xsize * jxl_save_opts.basic_info.ysize *
      jxl_save_opts.pixel_format.num_channels * 4;  // bytes per sample

  for (int i = nlayers - 1; i >= 0; i--) {
    gimp_save_progress.update();

    // copy image into buffer...
    gpointer pixels_buffer_1 = g_malloc(buffer_size);
    gpointer pixels_buffer_2 = g_malloc(buffer_size);

    gimp_layer_resize_to_image_size(layers[i]);

    GeglBuffer* buffer = gimp_drawable_get_buffer(layers[i]);

    // using gegl_buffer_set_format to get the format because
    // gegl_buffer_get_format doesn't always get the original format
    const Babl* native_format = gegl_buffer_set_format(buffer, nullptr);

    gegl_buffer_get(buffer,
                    GEGL_RECTANGLE(0, 0, jxl_save_opts.basic_info.xsize,
                                   jxl_save_opts.basic_info.ysize),
                    1.0, native_format, pixels_buffer_1, GEGL_AUTO_ROWSTRIDE,
                    GEGL_ABYSS_NONE);
    g_clear_object(&buffer);

    if (!jxl_save_opts.is_gray) {
      cmsDoTransform(
          hTransform, pixels_buffer_1, pixels_buffer_2,
          jxl_save_opts.basic_info.xsize * jxl_save_opts.basic_info.ysize);
    }

    gimp_save_progress.update();

    // send layer to encoder
    if (JXL_ENC_SUCCESS !=
        JxlEncoderAddImageFrame(enc_opts, &jxl_save_opts.pixel_format,
                                pixels_buffer_2, buffer_size)) {
      g_printerr(SAVE_PROC " Error: JxlEncoderAddImageFrame failed\n");
      return false;
    }

    g_free(pixels_buffer_1);
    g_free(pixels_buffer_2);
  }

  JxlEncoderCloseInput(enc.get());

  // get data from encoder
  std::vector<uint8_t> compressed;
  compressed.resize(262144);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size();

  JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
  while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
    gimp_save_progress.update();

    process_result = JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
    if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
      size_t offset = next_out - compressed.data();
      compressed.resize(compressed.size() + 262144);
      next_out = compressed.data() + offset;
      avail_out = compressed.size() - offset;
    }
  }
  compressed.resize(next_out - compressed.data());

  if (JXL_ENC_SUCCESS != process_result) {
    g_printerr(SAVE_PROC " Error: JxlEncoderProcessOutput failed\n");
    return false;
  }

  // write file
  std::ofstream outstream(filename, std::ios::out | std::ios::binary);
  copy(compressed.begin(), compressed.end(),
       std::ostream_iterator<uint8_t>(outstream));

  gimp_save_progress.finished();
  return true;
}  // SaveJpegXlImage()

}  // namespace jxl
