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

struct LoopFilter {
  LoopFilter();
  static const char* Name() { return "LoopFilter"; }

  template <class Visitor>
  Status VisitFields(Visitor* JXL_RESTRICT visitor) {
    // Must come before AllDefault.

    if (visitor->AllDefault(*this, &all_default)) {
      // Overwrite all serialized fields, but not any nonserialized_*.
      visitor->SetDefault(this);
      return true;
    }

    visitor->Bool(true, &gab);
    if (visitor->Conditional(gab)) {
      visitor->Bool(false, &gab_custom);
      if (visitor->Conditional(gab_custom)) {
        visitor->F16(1.1 * 0.104699568f, &gab_x_weight1);
        visitor->F16(1.1 * 0.055680538f, &gab_x_weight2);
        visitor->F16(1.1 * 0.104699568f, &gab_y_weight1);
        visitor->F16(1.1 * 0.055680538f, &gab_y_weight2);
        visitor->F16(1.1 * 0.104699568f, &gab_b_weight1);
        visitor->F16(1.1 * 0.055680538f, &gab_b_weight2);
      }
    }

    visitor->Bool(true, &epf);
    if (visitor->Conditional(epf)) {
      visitor->Bool(false, &epf_sharp_custom);
      if (visitor->Conditional(epf_sharp_custom)) {
        for (size_t i = 0; i < kEpfSharpEntries; ++i) {
          visitor->F16(float(i) / float(kEpfSharpEntries - 1),
                       &epf_sharp_lut[i]);
        }
      }

      visitor->Bool(false, &epf_weight_custom);
      if (visitor->Conditional(epf_weight_custom)) {
        visitor->F16(40.0f, &epf_channel_scale[0]);
        visitor->F16(5.0f, &epf_channel_scale[1]);
        visitor->F16(3.5f, &epf_channel_scale[2]);
        visitor->F16(0.45f, &epf_pass1_zeroflush);
        visitor->F16(0.6f, &epf_pass2_zeroflush);
      }

      visitor->Bool(false, &epf_sigma_custom);
      if (visitor->Conditional(epf_sigma_custom)) {
        visitor->F16(0.315f, &epf_quant_mul);
        visitor->F16(4.0f, &epf_dc_range_mul);
        visitor->F16(6.5f, &epf_pass2_sigma_scale);
        visitor->F16(0.6666666666666666f, &epf_border_sad_mul);
      }
    }

    JXL_RETURN_IF_ERROR(visitor->BeginExtensions(&extensions));
    // Extensions: in chronological order of being added to the format.
    return visitor->EndExtensions();
  }

  void GaborishWeights(float* JXL_RESTRICT gab_weights) const {
    gab_weights[0] = 1;
    gab_weights[1] = gab_x_weight1;
    gab_weights[2] = gab_x_weight2;
    gab_weights[3] = 1;
    gab_weights[4] = gab_y_weight1;
    gab_weights[5] = gab_y_weight2;
    gab_weights[6] = 1;
    gab_weights[7] = gab_b_weight1;
    gab_weights[8] = gab_b_weight2;
    // Normalize
    for (size_t c = 0; c < 3; c++) {
      const float mul =
          1.0f / (gab_weights[3 * c] +
                  4 * (gab_weights[3 * c + 1] + gab_weights[3 * c + 2]));
      gab_weights[3 * c] *= mul;
      gab_weights[3 * c + 1] *= mul;
      gab_weights[3 * c + 2] *= mul;
    }
  }

  size_t FirstStageRows() const { return gab ? 1 : epf ? 2 : 0; }
  size_t PaddingRows() const { return (epf ? 3 : 0) + (gab ? 1 : 0); }
  size_t PaddingCols() const {
    // Having less than one full block here breaks handling of sigma in EPF.
    // If no loop filter is used, no padding is necessary - indeed, adding
    // padding breaks the output as the padding area will not be processed
    // separately.
    return (epf || gab) ? kBlockDim : 0;
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
  bool epf;

  bool epf_sharp_custom;
  enum { kEpfSharpEntries = 8 };
  float epf_sharp_lut[kEpfSharpEntries];

  bool epf_weight_custom;      // Custom weight params
  float epf_channel_scale[3];  // Relative weight of each channel
  float epf_pass1_zeroflush;   // Minimum weight for first pass
  float epf_pass2_zeroflush;   // Minimum weight for second pass

  bool epf_sigma_custom;        // Custom sigma parameters
  float epf_quant_mul;          // Sigma is ~ this * quant
  float epf_dc_range_mul;       // How much to increase sigma with DC range
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
