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

#ifndef JXL_ENC_HEURISTICS_H_
#define JXL_ENC_HEURISTICS_H_

// Hook for custom encoder heuristics (VarDCT only for now).

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "jxl/aux_out_fwd.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/image.h"

namespace jxl {

struct PassesEncoderState;
class ImageBundle;

class EncoderHeuristics {
 public:
  virtual ~EncoderHeuristics() = default;
  // Initializes encoder structures in `enc_state` using the original (linear
  // sRGB) image data in `linear`, and the XYB image data in `opsin`. Also
  // modifies the `opsin` image by applying Gaborish, and doing other
  // modifications if necessary.
  // `pool` is used for running the computations on multiple threads. `aux_out`
  // collects statistics and can be used to print debug images.
  virtual Status LossyFrameHeuristics(PassesEncoderState* enc_state,
                                      const ImageBundle* linear, Image3F* opsin,
                                      ThreadPool* pool, AuxOut* aux_out) = 0;
};

class DefaultEncoderHeuristics : public EncoderHeuristics {
 public:
  Status LossyFrameHeuristics(PassesEncoderState* enc_state,
                              const ImageBundle* linear, Image3F* opsin,
                              ThreadPool* pool, AuxOut* aux_out) override;
};

}  // namespace jxl

#endif
