// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "lib/jxl/jxl_cms.h"
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/jxl_cms.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/printf_macros.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/field_encodings.h"
#include "lib/jxl/jxl_cms_internal.h"
#include "lib/jxl/matrix_ops.h"
#include "lib/jxl/transfer_functions-inl.h"
