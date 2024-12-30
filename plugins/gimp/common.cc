// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "plugins/gimp/common.h"

namespace jxl {

#if GIMP_MAJOR_VERSION >= 3
GimpDrawable* GimpLayerToDrawable(GimpLayer* ptr) {
  return reinterpret_cast<GimpDrawable*>(ptr);
}
GimpItem* GimpLayerToItem(GimpLayer* ptr) {
  return reinterpret_cast<GimpItem*>(ptr);
}
void GimpImageSetFileName(GimpImageOrId image_id,
                          const gchar* const file_name) {
  GFile* file = g_file_new_for_path(file_name);
  gimp_image_set_file(image_id, file);
  g_object_unref(file);
}
#else   // GIMP_MAJOR_VERSION == 2
gint32 GimpLayerToDrawable(gint32 id) { return id; }
gint32 GimpLayerToItem(gint32 id) { return id; }
void GimpImageSetFileName(GimpImageOrId image_id,
                          const gchar *const file_name) {
  gimp_image_set_filename(image_id, file_name);
}
#endif  // GIMP_MAJOR_VERSION

JpegXlGimpProgress::JpegXlGimpProgress(const char *message) {
  cur_progress = 0;
  max_progress = 100;

  gimp_progress_init_printf("%s\n", message);
}

void JpegXlGimpProgress::update() {
  gimp_progress_update(static_cast<float>(++cur_progress) /
                       static_cast<float>(max_progress));
}

void JpegXlGimpProgress::finished() { gimp_progress_update(1.0); }

}  // namespace jxl
