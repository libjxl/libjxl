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

#ifndef TOOLS_SPEED_STATS_H_
#define TOOLS_SPEED_STATS_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "lib/jxl/base/status.h"

namespace jpegxl {
namespace tools {

class SpeedStats {
 public:
  void NotifyElapsed(double elapsed_seconds);

  struct Summary {
    // How central_tendency was computed - depends on number of reps.
    const char* type;

    // Elapsed time
    double central_tendency;
    double min;
    double max;
    double variability;
  };

  // Non-const, may sort elapsed_.
  jxl::Status GetSummary(Summary* summary);

  // Sets the image size to allow computing MP/s values.
  void SetImageSize(size_t xsize, size_t ysize) {
    xsize_ = xsize;
    ysize_ = ysize;
  }

  // Sets the file size to allow computing MB/s values.
  void SetFileSize(size_t file_size) { file_size_ = file_size; }

  // Calls GetSummary and prints megapixels/sec. SetImageSize() must be called
  // once before this can be used.
  jxl::Status Print(size_t worker_threads);

 private:
  std::vector<double> elapsed_;
  size_t xsize_ = 0;
  size_t ysize_ = 0;

  // Size of the source binary file, meaningful when decoding a recompressed
  // JPEG.
  size_t file_size_ = 0;
};

}  // namespace tools
}  // namespace jpegxl

#endif  // TOOLS_SPEED_STATS_H_
