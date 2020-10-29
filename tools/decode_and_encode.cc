// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>

#include <string>

#include "lib/extras/codec.h"
#include "lib/jxl/base/data_parallel.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/base/thread_pool_internal.h"
#include "lib/jxl/codec_in_out.h"

namespace jxl {
namespace {

// Reads an input file (typically PNM) with color_space hint and writes to an
// output file (typically PNG) which supports all required metadata.
int Convert(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    fprintf(stderr, "Args: in colorspace_description out [bits]\n");
    return 1;
  }
  const std::string& pathname_in = argv[1];
  const std::string& desc = argv[2];
  const std::string& pathname_out = argv[3];

  CodecInOut io;
  ThreadPoolInternal pool(4);
  io.dec_hints.Add("color_space", desc);
  if (!SetFromFile(pathname_in, &io, &pool)) {
    fprintf(stderr, "Failed to read %s\n", pathname_in.c_str());
    return 1;
  }

  if (!EncodeToFile(io, pathname_out, &pool)) {
    fprintf(stderr, "Failed to write %s\n", pathname_out.c_str());
    return 1;
  }

  return 0;
}

}  // namespace
}  // namespace jxl

int main(int argc, char** argv) { return jxl::Convert(argc, argv); }
