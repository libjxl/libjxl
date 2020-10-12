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

#ifndef JXL_LOOP_FILTER_H_
#define JXL_LOOP_FILTER_H_

// Parameters for loop filter(s), stored in each frame.

#include <stddef.h>
#include <stdint.h>

#include "jxl/aux_out.h"
#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/status.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/field_encodings.h"

namespace jxl {

struct LoopFilter : public Fields {
  LoopFilter();
  const char* Name() const override { return "LoopFilter"; }

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  // TODO(deymo): Adjust padding based on the number of iterations.
  size_t PaddingRows() const { return (epf_iters ? 3 : 0) + (gab ? 1 : 0); }
  size_t PaddingCols() const {
    // Having less than one full block here breaks handling of sigma in EPF.
    // If no loop filter is used, no padding is necessary - indeed, adding
    // padding breaks the output as the padding area will not be processed
    // separately.
    return (epf_iters || gab) ? kBlockDim : 0;
  }

  mutable bool all_default;

  // --- Gaborish convolution
  bool gab;

  bool gab_custom;
  float gab_x_weight1;
  float gab_x_weight2;
  float gab_y_weight1;
  float gab_y_weight2;
  float gab_b_weight1;
  float gab_b_weight2;

  // --- Edge-preserving filter

  // Number of EPF stages to apply. 0 means EPF disabled. 1 applies only the
  // first stage, 2 applies both stages and 3 applies the first stage twice and
  // the second stage once.
  uint32_t epf_iters;

  bool epf_sharp_custom;
  enum { kEpfSharpEntries = 8 };
  float epf_sharp_lut[kEpfSharpEntries];

  bool epf_weight_custom;      // Custom weight params
  float epf_channel_scale[3];  // Relative weight of each channel
  float epf_pass1_zeroflush;   // Minimum weight for first pass
  float epf_pass2_zeroflush;   // Minimum weight for second pass

  bool epf_sigma_custom;        // Custom sigma parameters
  float epf_quant_mul;          // Sigma is ~ this * quant
  float epf_pass2_sigma_scale;  // Multiplier for sigma in the second pass
  float epf_border_sad_mul;     // (inverse) multiplier for sigma on borders

  uint64_t extensions;
};

Status ReadLoopFilter(BitReader* JXL_RESTRICT reader,
                      LoopFilter* JXL_RESTRICT loop_filter);

Status WriteLoopFilter(const LoopFilter& loop_filter,
                       BitWriter* JXL_RESTRICT writer, AuxOut* aux_out);

}  // namespace jxl

#endif  // JXL_LOOP_FILTER_H_
