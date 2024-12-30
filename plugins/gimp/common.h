// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef PLUGINS_GIMP_COMMON_H_
#define PLUGINS_GIMP_COMMON_H_

#include <libgimp/gimp.h>
#include <libgimp/gimpui.h>
#include <math.h>

#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

#define PLUG_IN_BINARY "file-jxl"
#define SAVE_PROC "file-jxl-save"

// Defined by both FUIF and glib.
#undef MAX
#undef MIN
#undef CLAMP

#include <jxl/resizable_parallel_runner.h>
#include <jxl/resizable_parallel_runner_cxx.h>

namespace jxl {

#if GIMP_MAJOR_VERSION >= 3
using GimpImageOrId = GimpImage*;
using GimpLayerOrId = GimpLayer*;
const GimpLayerOrId kNoGimpLayerOrId = nullptr;
GimpDrawable* GimpLayerToDrawable(GimpLayer* ptr);
GimpItem* GimpLayerToItem(GimpLayer* ptr);
void GimpImageSetFileName(GimpImageOrId image_id, const gchar* const file_name);
#else  // GIMP_MAJOR_VERSION == 2
using GimpImageOrId = gint32;
using GimpLayerOrId = gint32;
const GimpLayerOrId kNoGimpLayerOrId = -1;
gint32 GimpLayerToDrawable(gint32 id);
gint32 GimpLayerToItem(gint32 id);
void GimpImageSetFileName(GimpImageOrId image_id, const gchar *const file_name);
#define GIMP_PRECISION_U8_NON_LINEAR GIMP_PRECISION_U8_GAMMA
#define GIMP_PRECISION_U16_NON_LINEAR GIMP_PRECISION_U16_GAMMA
#define GIMP_PRECISION_U32_NON_LINEAR GIMP_PRECISION_U32_GAMMA
#define GIMP_PRECISION_HALF_NON_LINEAR GIMP_PRECISION_HALF_GAMMA
#define GIMP_PRECISION_FLOAT_NON_LINEAR GIMP_PRECISION_FLOAT_GAMMA
#define GIMP_PRECISION_DOUBLE_NON_LINEAR GIMP_PRECISION_DOUBLE_GAMMA
#endif  // GIMP_MAJOR_VERSION

class JpegXlGimpProgress {
 public:
  explicit JpegXlGimpProgress(const char *message);
  void update();
  void finished();

 private:
  int cur_progress;
  int max_progress;

};  // class JpegXlGimpProgress

}  // namespace jxl

#endif  // PLUGINS_GIMP_COMMON_H_
