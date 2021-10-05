// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdarg.h>

#include "plugins/gimp/common.h"

namespace jxl {

JpegXlGimpProgress::JpegXlGimpProgress(const char* fmt, ...) {
  cur_progress = 0;
  max_progress = 100;

  va_list args;
  va_start(args, fmt);
  gchar* tmpstr = g_strdup_vprintf(fmt, args);
  gimp_progress_init_printf("%s", tmpstr);
  g_free(tmpstr);
  va_end(args);
}

void JpegXlGimpProgress::update() {
  gimp_progress_update((float)++cur_progress / (float)max_progress);
  return;
}

void JpegXlGimpProgress::finished() {
  gimp_progress_update(1.0);
  return;
}

}  // namespace jxl
