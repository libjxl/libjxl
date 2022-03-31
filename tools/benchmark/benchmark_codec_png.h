// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_BENCHMARK_BENCHMARK_CODEC_PNG_H_
#define TOOLS_BENCHMARK_BENCHMARK_CODEC_PNG_H_

#if JPEGXL_ENABLE_APNG

#include <string>

#include "lib/jxl/base/status.h"
#include "tools/benchmark/benchmark_args.h"
#include "tools/benchmark/benchmark_codec.h"

namespace jxl {
ImageCodec* CreateNewPNGCodec(const BenchmarkArgs& args);

// Registers the png-specific command line options.
Status AddCommandLineOptionsPNGCodec(BenchmarkArgs* args);
}  // namespace jxl

#endif

#endif  // TOOLS_BENCHMARK_BENCHMARK_CODEC_PNG_H_
