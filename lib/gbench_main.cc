// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "benchmark/benchmark.h"

void RegisterDctBenchmarks();

int main(int argc, char** argv) {
  char arg0_default[] = "benchmark";
  char* args_default = arg0_default;
  if (!argv) {
    argc = 1;
    argv = &args_default;
  }
  RegisterDctBenchmarks();
  ::benchmark ::Initialize(&argc, argv);
  if (::benchmark ::ReportUnrecognizedArguments(argc, argv)) return 1;
  ::benchmark ::RunSpecifiedBenchmarks();
  ::benchmark ::Shutdown();
  return 0;
}
