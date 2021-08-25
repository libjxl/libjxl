// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "plugins/gimp/file-jxl-save.h"

#define PLUG_IN_BINARY "file-jxl"
#define SAVE_PROC "file-jxl-save"

#define SCALE_WIDTH 200

namespace jxl {

namespace {

class JpegXlSaveOpts {
 public:
  float distance;
  float quality;

  bool lossless = false;
  bool is_linear = false;
  bool has_alpha = false;
  bool is_gray = false;

  bool advanced_mode = false;
  bool use_container = true;
  bool save_exif = false;
  int encoding_effort = 7;
  int faster_decoding = 0;

  std::string babl_format_str = "RGB u16";
  std::string babl_type_str = "u16";
  std::string babl_model_str = "RGB";

  JxlPixelFormat pixel_format;
  JxlBasicInfo basic_info;

  // functions
  JpegXlSaveOpts();

  bool set_distance(float dist);
  bool set_quality(float qual);
  bool set_dimensions(int x, int y);
  bool set_num_channels(int channels);

  bool update_distance();
  bool update_quality();

  bool set_model(int gimp_model);

  bool update_babl_format();
  bool set_babl_model(std::string model);
  bool set_babl_type(std::string type);

  bool set_pixel_type(int type);
  bool set_precision(int gimp_precision);

 private:
};  // class JpegXlSaveOpts

JpegXlSaveOpts jxl_save_opts;

static bool gui_on_change_quality(GtkAdjustment* adj_qual,
                                  GtkAdjustment* adj_dist) {
  jxl_save_opts.quality = gtk_adjustment_get_value(adj_qual);
  jxl_save_opts.update_distance();
  gtk_adjustment_set_value(adj_dist, jxl_save_opts.distance);
  return true;
}

static bool gui_on_change_distance(GtkAdjustment* adj_dist,
                                   GtkAdjustment* adj_qual) {
  float new_distance = gtk_adjustment_get_value(adj_dist);
  jxl_save_opts.distance = new_distance;
  jxl_save_opts.update_quality();
  gtk_adjustment_set_value(adj_qual, jxl_save_opts.quality);

  // updating quality can change distance again
  // set it again to ensure user value is maintained
  if (jxl_save_opts.distance != new_distance) {
    gtk_adjustment_set_value(adj_dist, new_distance);
  }
  return true;
}

static bool gui_on_change_lossless(GtkWidget* toggle,
                                   GtkAdjustment* adjustments[]) {
  GtkAdjustment* adj_distance = adjustments[0];
  GtkAdjustment* adj_quality = adjustments[1];
  GtkAdjustment* adj_effort = adjustments[2];

  jxl_save_opts.lossless =
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle));

  g_message("lossless = %d", jxl_save_opts.lossless);

  if (jxl_save_opts.lossless) {
    gimp_scale_entry_set_sensitive((GtkObject*)adj_distance, false);
    gimp_scale_entry_set_sensitive((GtkObject*)adj_quality, false);
    gtk_adjustment_set_value(adj_quality, 100.0f);
    gtk_adjustment_set_value(adj_distance, 0.0f);

    jxl_save_opts.distance = 0;
    jxl_save_opts.update_quality();

    gtk_adjustment_set_value(adj_effort, 3);
    jxl_save_opts.encoding_effort = 3;
  } else {
    gimp_scale_entry_set_sensitive((GtkObject*)adj_distance, true);
    gimp_scale_entry_set_sensitive((GtkObject*)adj_quality, true);
    gtk_adjustment_set_value(adj_quality, 100.0f);
    gtk_adjustment_set_value(adj_distance, 0.1f);

    jxl_save_opts.distance = 0.1f;
    jxl_save_opts.update_quality();

    gtk_adjustment_set_value(adj_effort, 7);
    jxl_save_opts.encoding_effort = 7;
  }
  return true;
}

static bool gui_on_change_codestream(GtkWidget* toggle) {
  jxl_save_opts.use_container =
      !gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle));
  return true;
}

static bool gui_on_change_uses_original_profile(GtkWidget* toggle) {
  jxl_save_opts.basic_info.uses_original_profile =
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle));
  return true;
}

static bool gui_on_change_advanced_mode(GtkWidget* toggle,
                                        std::vector<GtkWidget*> advanced_opts) {
  jxl_save_opts.advanced_mode =
      gtk_toggle_button_get_active(GTK_TOGGLE_BUTTON(toggle));

  GtkWidget* frame = advanced_opts[0];

  gtk_widget_set_sensitive(frame, jxl_save_opts.advanced_mode);

  if (!jxl_save_opts.advanced_mode) {
    jxl_save_opts.lossless = false;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(advanced_opts[1]), false);

    jxl_save_opts.use_container = true;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(advanced_opts[2]), false);

    jxl_save_opts.basic_info.uses_original_profile = false;
    gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(advanced_opts[3]), false);

    // save metadata
    // gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(advanced_opts[4]), false);

    jxl_save_opts.encoding_effort = 7;
    gtk_adjustment_set_value((GtkAdjustment*)advanced_opts[5], 7);

    jxl_save_opts.faster_decoding = 0;
    gtk_adjustment_set_value((GtkAdjustment*)advanced_opts[6], 0);
  }
  return true;
}

bool SaveDialog() {
  gboolean run;
  GtkWidget* dialog;
  GtkWidget* content_area;
  GtkWidget* main_vbox;
  GtkWidget* frame;
  GtkWidget* toggle;
  GtkWidget* table;
  GtkWidget* vbox;
  GtkWidget* toggle_lossless;
  GtkWidget* frame_advanced;
  GtkAdjustment* entry_distance;
  GtkAdjustment* entry_quality;
  GtkAdjustment* entry_effort;
  GtkAdjustment* entry_faster;

  // initialize export dialog
  gimp_ui_init(PLUG_IN_BINARY, true);
  dialog = gimp_export_dialog_new("JPEG XL", PLUG_IN_BINARY, SAVE_PROC);

  gtk_window_set_resizable(GTK_WINDOW(dialog), true);
  content_area = gimp_export_dialog_get_content_area(dialog);

  main_vbox = gtk_vbox_new(false, 6);
  gtk_container_set_border_width(GTK_CONTAINER(main_vbox), 6);
  gtk_box_pack_start(GTK_BOX(content_area), main_vbox, true, true, 0);
  gtk_widget_show(main_vbox);

  // Standard Settings Frame
  frame = gtk_frame_new("Standard Settings");
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_ETCHED_IN);
  gtk_box_pack_start(GTK_BOX(main_vbox), frame, true, true, 0);
  gtk_widget_show(frame);

  vbox = gtk_vbox_new(false, 6);
  gtk_container_set_border_width(GTK_CONTAINER(vbox), 6);
  gtk_container_add(GTK_CONTAINER(frame), vbox);
  gtk_widget_show(vbox);

  // Butteraugli Distance
  static gchar distance_help[] =
      "Butteraugli distance.  Use lower values for higher quality.";
  frame = gtk_frame_new("Butteraugli Distance");
  gimp_help_set_help_data(frame, distance_help, nullptr);
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_NONE);
  gtk_box_pack_start(GTK_BOX(vbox), frame, false, false, 0);
  gtk_widget_show(frame);

  // Distance Scale
  table = gtk_table_new(1, 3, false);
  gtk_table_set_col_spacings(GTK_TABLE(table), 6);
  gtk_container_add(GTK_CONTAINER(frame), table);
  gtk_widget_show(table);

  entry_distance = (GtkAdjustment*)gimp_scale_entry_new(
      GTK_TABLE(table), 0, 0, "", SCALE_WIDTH, 0, jxl_save_opts.distance, 0.0,
      45.0, 0.001, 1.0, 3, true, 0.0, 0.0, distance_help, "file-jxl-save");

  gimp_scale_entry_set_logarithmic((GtkObject*)entry_distance, true);

  // JPEG-style Quality
  static gchar quality_help[] =
      "JPEG-style Quality setting is remapped to distance.  "
      "Values roughly match libjpeg quality.";
  frame = gtk_frame_new("JPEG-style Quality");
  gimp_help_set_help_data(frame, quality_help, nullptr);
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_NONE);
  gtk_box_pack_start(GTK_BOX(vbox), frame, false, false, 0);
  gtk_widget_show(frame);

  // Quality Scale
  table = gtk_table_new(1, 3, false);
  gtk_table_set_col_spacings(GTK_TABLE(table), 6);
  gtk_container_add(GTK_CONTAINER(frame), table);
  gtk_widget_show(table);

  entry_quality = (GtkAdjustment*)gimp_scale_entry_new(
      GTK_TABLE(table), 0, 0, "", SCALE_WIDTH, 0, jxl_save_opts.quality, 0.0,
      100.0, 1.0, 10.0, 2, true, 0.0, 0.0, quality_help, "file-jxl-save");

  // Distance and Quality Signals
  g_signal_connect(entry_distance, "value-changed",
                   G_CALLBACK(gui_on_change_distance), entry_quality);
  g_signal_connect(entry_quality, "value-changed",
                   G_CALLBACK(gui_on_change_quality), entry_distance);

  // Advanced Settings Frame
  std::vector<GtkWidget*> advanced_opts;

  frame_advanced = gtk_frame_new("Advanced Settings");
  gimp_help_set_help_data(frame_advanced,
                          "Advanced Settings that shouldn't be used.", nullptr);
  gtk_frame_set_shadow_type(GTK_FRAME(frame_advanced), GTK_SHADOW_ETCHED_IN);
  gtk_box_pack_start(GTK_BOX(main_vbox), frame_advanced, true, true, 0);
  gtk_widget_show(frame_advanced);

  gtk_widget_set_sensitive(frame_advanced, false);

  vbox = gtk_vbox_new(false, 6);
  gtk_container_set_border_width(GTK_CONTAINER(vbox), 6);
  gtk_container_add(GTK_CONTAINER(frame_advanced), vbox);
  gtk_widget_show(vbox);

  advanced_opts.push_back(frame_advanced);

  // lossless convenience checkbox
  static gchar lossless_help[] =
      "Compress using modular lossless mode.  "
      "Effort is set to 3 to improve performance.";
  toggle_lossless = gtk_check_button_new_with_label("Lossless Mode");
  gimp_help_set_help_data(toggle_lossless, lossless_help, nullptr);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle_lossless),
                               jxl_save_opts.lossless);
  gtk_box_pack_start(GTK_BOX(vbox), toggle_lossless, false, false, 0);
  gtk_widget_show(toggle_lossless);

  advanced_opts.push_back(toggle_lossless);

  // save raw codestream
  static gchar codestream_help[] =
      "Save the raw codestream, without a container.  "
      "Not recommended.  The container is required for "
      "metadata, and the overhead is miniscule.";
  toggle = gtk_check_button_new_with_label("Raw Codestream");
  gimp_help_set_help_data(toggle, codestream_help, nullptr);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
                               !jxl_save_opts.use_container);
  gtk_box_pack_start(GTK_BOX(vbox), toggle, false, false, 0);
  gtk_widget_show(toggle);

  g_signal_connect(toggle, "toggled", G_CALLBACK(gui_on_change_codestream),
                   nullptr);

  advanced_opts.push_back(toggle);

  // uses_original_profile
  static gchar uses_original_profile_help[] =
      "Prevents conversion to XYB colorspace.  "
      "File sizes are approximately doubled.  "
      "This option is not recommended.";
  toggle = gtk_check_button_new_with_label("Use Original Color Profile");
  gimp_help_set_help_data(toggle, uses_original_profile_help, nullptr);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
                               jxl_save_opts.basic_info.uses_original_profile);
  gtk_box_pack_start(GTK_BOX(vbox), toggle, false, false, 0);
  gtk_widget_show(toggle);

  g_signal_connect(toggle, "toggled",
                   G_CALLBACK(gui_on_change_uses_original_profile), nullptr);

  advanced_opts.push_back(toggle);

  // Save Exif Metadata
  toggle = gtk_check_button_new_with_label("Save Exif Metadata");
  gimp_help_set_help_data(
      toggle, "This feature is not yet available in the API.", nullptr);
  gtk_box_pack_start(GTK_BOX(vbox), toggle, false, false, 0);
  gtk_widget_set_sensitive(toggle, false);
  gtk_widget_show(toggle);

  advanced_opts.push_back(toggle);

  // Encoding Effort
  static gchar effort_help[] =
      "Encoding Effort: Higher number is more effort (slower).\n\tDefault = 7.";
  frame = gtk_frame_new("Encoding Effort");
  gimp_help_set_help_data(frame, effort_help, nullptr);
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_NONE);
  gtk_box_pack_start(GTK_BOX(vbox), frame, false, false, 0);
  gtk_widget_show(frame);

  // Effort Scale
  table = gtk_table_new(1, 3, false);
  gtk_table_set_col_spacings(GTK_TABLE(table), 6);
  gtk_container_add(GTK_CONTAINER(frame), table);
  gtk_widget_show(table);

  entry_effort = (GtkAdjustment*)gimp_scale_entry_new(
      GTK_TABLE(table), 0, 0, "", SCALE_WIDTH, 0, jxl_save_opts.encoding_effort,
      1, 9, 1, 2, 0, true, 0.0, 0.0, effort_help, "file-jxl-save");

  // Effort Signals
  g_signal_connect(entry_effort, "value-changed",
                   G_CALLBACK(gimp_int_adjustment_update),
                   &jxl_save_opts.encoding_effort);

  advanced_opts.push_back((GtkWidget*)entry_effort);

  // signal for lossless toggle
  // has to be put here to change effort setting
  GtkAdjustment* adjustments[] = {entry_distance, entry_quality, entry_effort};
  g_signal_connect(toggle_lossless, "toggled",
                   G_CALLBACK(gui_on_change_lossless), adjustments);

  // Faster Decoding
  static gchar faster_help[] =
      "Faster Decoding to improve decoding speed.  "
      "Higher values give higher speed at the expense of quality.\n"
      "\tDefault = 0.";
  frame = gtk_frame_new("Faster Decoding");
  gimp_help_set_help_data(frame, faster_help, nullptr);
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_NONE);
  gtk_box_pack_start(GTK_BOX(vbox), frame, false, false, 0);
  gtk_widget_show(frame);

  // Faster Decoding Scale
  table = gtk_table_new(1, 3, false);
  gtk_table_set_col_spacings(GTK_TABLE(table), 6);
  gtk_container_add(GTK_CONTAINER(frame), table);
  gtk_widget_show(table);

  entry_faster = (GtkAdjustment*)gimp_scale_entry_new(
      GTK_TABLE(table), 0, 0, "", SCALE_WIDTH, 0, jxl_save_opts.faster_decoding,
      0, 5, 1, 1, 0, true, 0.0, 0.0, faster_help, "file-jxl-save");

  advanced_opts.push_back((GtkWidget*)entry_faster);

  // Faster Decoding Signals
  g_signal_connect(entry_faster, "value-changed",
                   G_CALLBACK(gimp_int_adjustment_update),
                   &jxl_save_opts.faster_decoding);

  // Enable Advanced Settings
  frame = gtk_frame_new(0);
  gimp_help_set_help_data(frame, "Advanced Settings shouldn't be used.",
                          nullptr);
  gtk_frame_set_shadow_type(GTK_FRAME(frame), GTK_SHADOW_NONE);
  gtk_box_pack_start(GTK_BOX(main_vbox), frame, true, true, 0);
  gtk_widget_show(frame);

  vbox = gtk_vbox_new(false, 6);
  gtk_container_set_border_width(GTK_CONTAINER(vbox), 6);
  gtk_container_add(GTK_CONTAINER(frame), vbox);
  gtk_widget_show(vbox);

  static gchar advanced_help[] = "Use advanced settings with care.";
  toggle = gtk_check_button_new_with_label("Enable Advanced Settings");
  gimp_help_set_help_data(toggle, advanced_help, nullptr);
  gtk_toggle_button_set_active(GTK_TOGGLE_BUTTON(toggle),
                               jxl_save_opts.advanced_mode);
  gtk_box_pack_start(GTK_BOX(vbox), toggle, false, false, 0);
  gtk_widget_show(toggle);

  g_signal_connect(toggle, "toggled", G_CALLBACK(gui_on_change_advanced_mode),
                   &advanced_opts);

  // show dialog
  gtk_widget_show(dialog);

  GtkAllocation allocation;
  gtk_widget_get_allocation(dialog, &allocation);

  int height = allocation.height;
  gtk_widget_set_size_request(dialog, height * 1.4, height);

  run = (gimp_dialog_run(GIMP_DIALOG(dialog)) == GTK_RESPONSE_OK);
  gtk_widget_destroy(dialog);

  return run;
}  // SaveDialog

JpegXlSaveOpts::JpegXlSaveOpts() {
  set_distance(1.0f);

  pixel_format.num_channels = 4;
  pixel_format.data_type = JXL_TYPE_UINT8;
  pixel_format.endianness = JXL_NATIVE_ENDIAN;
  pixel_format.align = 0;

  basic_info.alpha_bits = 0;
  basic_info.alpha_exponent_bits = 0;
  basic_info.uses_original_profile = false;
  basic_info.intensity_target = 255.0f;
  basic_info.orientation = JXL_ORIENT_IDENTITY;

  return;
}  // JpegXlSaveOpts constructor

bool JpegXlSaveOpts::set_model(int gimp_model) {
  switch (gimp_model) {
    case GIMP_GRAY_IMAGE:
      if (is_linear) {
        set_babl_model("Y");
        set_num_channels(1);
      } else {
        set_babl_model("Y'");
        set_num_channels(1);
      }
      return true;

    case GIMP_GRAYA_IMAGE:
      if (is_linear) {
        set_babl_model("YA");
        set_num_channels(2);
      } else {
        set_babl_model("Y'A");
        set_num_channels(2);
      }
      return true;

    case GIMP_RGB_IMAGE:
      if (is_linear) {
        set_babl_model("RGB");
        set_num_channels(3);
      } else {
        set_babl_model("R'G'B'");
        set_num_channels(3);
      }
      return true;

    case GIMP_RGBA_IMAGE:
      if (is_linear) {
        set_babl_model("RGBA");
        set_num_channels(4);
      } else {
        set_babl_model("R'G'B'A");
        set_num_channels(4);
      }
      return true;

    default:
      g_printerr("JXL Error: Unsupported pixel format.\n");
      return false;
  }
}  // JpegXlSaveOpts::set_model

bool JpegXlSaveOpts::set_distance(float dist) {
  distance = dist;
  return update_quality();
}

bool JpegXlSaveOpts::set_quality(float qual) {
  quality = qual;
  return update_distance();
}

bool JpegXlSaveOpts::update_quality() {
  float qual;

  if (distance < 0.1f) {
    qual = 100;
  } else if (distance <= 6.4) {
    qual = 100 - (distance - 0.1) / 0.09f;
    lossless = false;
  } else {
    qual = 30 - 5 * (log(6.25 * distance - 40)) / log(2.5);
    lossless = false;
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

bool JpegXlSaveOpts::update_distance() {
  float dist;
  if (quality >= 30) {
    dist = 0.1 + (100 - quality) * 0.09;
  } else {
    dist = 6.4 + pow(2.5, (30 - quality) / 5.0f) / 6.25f;
  }

  distance = dist;
  return true;
}

bool JpegXlSaveOpts::set_dimensions(int x, int y) {
  basic_info.xsize = x;
  basic_info.ysize = y;
  return true;
}

bool JpegXlSaveOpts::set_num_channels(int channels) {
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
      basic_info.alpha_bits = basic_info.bits_per_sample;
      basic_info.alpha_exponent_bits = basic_info.exponent_bits_per_sample;
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
      basic_info.alpha_bits = basic_info.bits_per_sample;
      basic_info.alpha_exponent_bits = basic_info.exponent_bits_per_sample;
      break;
    default:
      set_num_channels(3);
  }  // switch
  return true;
}  // JpegXlSaveOpts::set_num_channels

bool JpegXlSaveOpts::update_babl_format() {
  babl_format_str = babl_model_str + " " + babl_type_str;
  return true;
}

bool JpegXlSaveOpts::set_babl_model(std::string model) {
  babl_model_str = model;
  return update_babl_format();
}

bool JpegXlSaveOpts::set_babl_type(std::string type) {
  babl_type_str = type;
  return update_babl_format();
}

bool JpegXlSaveOpts::set_pixel_type(int type) {
  switch (type) {
    case JXL_TYPE_FLOAT16:
      set_babl_type("half");
      pixel_format.data_type = JXL_TYPE_FLOAT16;
      basic_info.bits_per_sample = 16;
      basic_info.exponent_bits_per_sample = 5;
      break;

    case JXL_TYPE_FLOAT:
      set_babl_type("float");
      pixel_format.data_type = JXL_TYPE_FLOAT;
      basic_info.bits_per_sample = 32;
      basic_info.exponent_bits_per_sample = 8;
      break;

    // UINT32 is not yet supported.  Using UINT16 instead.
    // See documentation of JxlEncoderAddImageFrame().
    case JXL_TYPE_UINT32:
    case JXL_TYPE_UINT16:
      set_babl_type("u16");
      pixel_format.data_type = JXL_TYPE_UINT16;
      basic_info.bits_per_sample = 16;
      basic_info.exponent_bits_per_sample = 0;
      break;

    case JXL_TYPE_UINT8:
    default:
      set_babl_type("u8");
      pixel_format.data_type = JXL_TYPE_UINT8;
      basic_info.bits_per_sample = 8;
      basic_info.exponent_bits_per_sample = 0;
      break;
  }

  return true;
}  // JpegXlSaveOpts::set_pixel_type

bool JpegXlSaveOpts::set_precision(int gimp_precision) {
  switch (gimp_precision) {
    // Note: all floating point formats save as linear
    // to prevent gamma interpretation problems when viewing.
    // See documentation of JxlEncoderAddImageFrame().
    case GIMP_PRECISION_HALF_GAMMA:
    case GIMP_PRECISION_HALF_LINEAR:
      is_linear = true;
      set_pixel_type(JXL_TYPE_FLOAT16);
      break;
    case GIMP_PRECISION_FLOAT_GAMMA:
    case GIMP_PRECISION_FLOAT_LINEAR:
      is_linear = true;
      set_pixel_type(JXL_TYPE_FLOAT);
      break;

    // Note: all INT formats save as non-linear to prevent
    // gamma interpretation problems when viewing.
    // See documentation of JxlEncoderAddImageFrame().
    case GIMP_PRECISION_U32_GAMMA:
    case GIMP_PRECISION_U32_LINEAR:
      is_linear = false;
      set_pixel_type(JXL_TYPE_UINT32);
      break;
    case GIMP_PRECISION_U16_GAMMA:
    case GIMP_PRECISION_U16_LINEAR:
      is_linear = false;
      set_pixel_type(JXL_TYPE_UINT16);
      break;

    default:
    case GIMP_PRECISION_U8_LINEAR:
    case GIMP_PRECISION_U8_GAMMA:
      is_linear = false;
      set_pixel_type(JXL_TYPE_UINT8);
      break;
  }
  return true;
}  // JpegXlSaveOpts::set_precision

}  // namespace

bool SaveJpegXlImage(const gint32 image_id, const gint32 drawable_id,
                     const gint32 orig_image_id, const gchar* const filename) {
  if (!SaveDialog()) {
    return true;
  }

  gint32 nlayers;
  gint32* layers;

  JpegXlGimpProgress gimp_save_progress(
      ("Saving JPEG XL file:" + std::string(filename)).c_str());
  gimp_save_progress.update();

  // try to get ICC color profile...
  std::vector<uint8_t> icc;

  GimpColorProfile* profile = gimp_image_get_color_profile(image_id);

  if (profile) {
    jxl_save_opts.is_linear = gimp_color_profile_is_linear(profile);
    jxl_save_opts.is_gray = gimp_color_profile_is_gray(profile);

    g_printerr("JXL Info: Extracting ICC Profile...\n");
    gsize icc_size;
    const guint8* const icc_bytes =
        gimp_color_profile_get_icc_profile(profile, &icc_size);

    icc.assign(icc_bytes, icc_bytes + icc_size);
  } else {
    g_printerr("JXL Info: No ICC profile.  Exporting image anyway.\n");
  }

  gimp_save_progress.update();

  jxl_save_opts.set_dimensions(gimp_image_width(image_id),
                               gimp_image_height(image_id));

  layers = gimp_image_get_layers(image_id, &nlayers);

  for (gint32 i = 0; i < nlayers; i++) {
    if (gimp_drawable_has_alpha(layers[i])) {
      jxl_save_opts.has_alpha = true;
      break;
    }
  }

  gimp_save_progress.update();

  // JxlEncoderAddImageFrame() doesn't currently support
  // JXL_TYPE_UINT32.  Rebuilding babl_format to match
  // the JxlPixelFormat to allow  export to UINT16.
  //
  // When this is no longer necessary, will be able to
  // replace extraneous code with just:
  //    native_format = gegl_buffer_get_format(buffer);
  //
  const Babl* native_format;
  jxl_save_opts.set_precision(gimp_image_get_precision(image_id));
  jxl_save_opts.set_model(gimp_drawable_type(drawable_id));
  native_format = babl_format(jxl_save_opts.babl_format_str.c_str());

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
    g_printerr("JXL Error: JxlEncoderSetParallelRunner failed\n");
    return false;
  }

  if (JXL_ENC_SUCCESS !=
      JxlEncoderSetBasicInfo(enc.get(), &jxl_save_opts.basic_info)) {
    g_printerr("JXL Error: JxlEncoderSetBasicInfo failed\n");
    return false;
  }

  // try to use ICC profile
  if (icc.size() > 0 && !gimp_color_profile_is_gray(profile)) {
    if (JXL_ENC_SUCCESS !=
        JxlEncoderSetICCProfile(enc.get(), icc.data(), icc.size())) {
      g_printerr("JXL Warning: JxlEncoderSetICCProfile failed.\n");
      jxl_save_opts.basic_info.uses_original_profile = false;
    }
  } else {
    g_printerr("JXL Warning: Using internal profile.\n");
    jxl_save_opts.basic_info.uses_original_profile = false;
  }

  // detect internal color profile
  JxlColorEncoding color_encoding = {};

  if (jxl_save_opts.is_linear) {
    JxlColorEncodingSetToLinearSRGB(
        &color_encoding,
        /*is_gray=*/jxl_save_opts.pixel_format.num_channels < 3);
  } else {
    JxlColorEncodingSetToSRGB(
        &color_encoding,
        /*is_gray=*/jxl_save_opts.pixel_format.num_channels < 3);
  }

  if (JXL_ENC_SUCCESS !=
      JxlEncoderSetColorEncoding(enc.get(), &color_encoding)) {
    g_printerr("JXL Warning: JxlEncoderSetColorEncoding failed\n");
  }

  // set encoder options
  JxlEncoderOptions* enc_opts;
  enc_opts = JxlEncoderOptionsCreate(enc.get(), nullptr);

  JxlEncoderOptionsSetEffort(enc_opts, jxl_save_opts.encoding_effort);
  JxlEncoderOptionsSetDecodingSpeed(enc_opts, jxl_save_opts.faster_decoding);

  if (jxl_save_opts.lossless || jxl_save_opts.distance < 0.01f) {
    if (jxl_save_opts.basic_info.exponent_bits_per_sample > 0) {
      // lossless mode doesn't work with floating point
      jxl_save_opts.distance = 0.01;
      JxlEncoderOptionsSetLossless(enc_opts, false);
      JxlEncoderOptionsSetDistance(enc_opts, 0.01);
    } else {
      JxlEncoderOptionsSetDistance(enc_opts, 0);
      JxlEncoderOptionsSetLossless(enc_opts, true);
    }
  } else {
    JxlEncoderOptionsSetLossless(enc_opts, false);
    JxlEncoderOptionsSetDistance(enc_opts, jxl_save_opts.distance);
  }

  // process layers and compress into JXL
  std::vector<uint8_t> compressed;
  compressed.resize(262144);
  uint8_t* next_out = compressed.data();
  size_t avail_out = compressed.size();

  size_t buffer_size = jxl_save_opts.basic_info.xsize *
                       jxl_save_opts.basic_info.ysize *
                       jxl_save_opts.pixel_format.num_channels *
                       (jxl_save_opts.basic_info.bits_per_sample >> 3);

  nlayers = 1;  // just process one layer for now
  for (gint32 i = 0; i < nlayers; i++) {
    gimp_save_progress.update();

    // copy image into buffer...
    gpointer pixels_buffer;
    pixels_buffer = g_malloc(buffer_size);

    GeglBuffer* buffer = gimp_drawable_get_buffer(layers[i]);
    gegl_buffer_get(buffer,
                    GEGL_RECTANGLE(0, 0, jxl_save_opts.basic_info.xsize,
                                   jxl_save_opts.basic_info.ysize),
                    1.0, native_format, pixels_buffer, GEGL_AUTO_ROWSTRIDE,
                    GEGL_ABYSS_NONE);

    g_clear_object(&buffer);

    gimp_save_progress.update();

    // compress layer
    if (JXL_ENC_SUCCESS !=
        JxlEncoderAddImageFrame(enc_opts, &jxl_save_opts.pixel_format,
                                pixels_buffer, buffer_size)) {
      g_printerr("JXL Error: JxlEncoderAddImageFrame failed\n");
      return false;
    }

    // get data from encoder
    JxlEncoderStatus process_result = JXL_ENC_NEED_MORE_OUTPUT;
    while (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
      gimp_save_progress.update();

      process_result =
          JxlEncoderProcessOutput(enc.get(), &next_out, &avail_out);
      if (process_result == JXL_ENC_NEED_MORE_OUTPUT) {
        size_t offset = next_out - compressed.data();
        compressed.resize(compressed.size() + 262144);
        next_out = compressed.data() + offset;
        avail_out = compressed.size() - offset;
      }
    }

    if (JXL_ENC_SUCCESS != process_result) {
      g_printerr("JXL Error: JxlEncoderProcessOutput failed\n");
      return false;
    }
  }

  JxlEncoderCloseInput(enc.get());

  compressed.resize(next_out - compressed.data());

  // write file
  std::ofstream outstream(filename, std::ios::out | std::ios::binary);
  copy(compressed.begin(), compressed.end(),
       std::ostream_iterator<uint8_t>(outstream));

  gimp_save_progress.finished();
  return true;
}  // SaveJpegXlImage()

}  // namespace jxl
