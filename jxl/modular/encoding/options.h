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

#ifndef JXL_MODULAR_ENCODING_OPTIONS_H_
#define JXL_MODULAR_ENCODING_OPTIONS_H_

#include "jxl/modular/image/image.h"

namespace jxl {

struct modular_options {
  // decoding options
  bool identify;  // don't decode image data, just decode header

  // used in both encode and decode
  int nb_channels;   // if full_header==false, need to specify how many channels
                     // to expect
  int skipchannels;  // the first <skipchannels> channels will not be
                     // encoded/decoded
  size_t max_chan_size;  // stop encoding/decoding when reaching a (non-meta)
                         // channel that has a dimension bigger than this

  // encoding options (some of which are needed during decoding too)
  int entropy_coder;  // 0 = MABEGABRAC, 1 = MABrotli, 2 = MARANS

  // MA options
  float nb_repeats;    // number of iterations to do to learn a MA tree (does
                       // not have to be an integer; if zero there is no MA
                       // context model)
  int max_properties;  // maximum number of (previous channel) properties to use
                       // in the MA trees
  float ctx_threshold;  // number of bits to be saved to justify adding
                        // another node to the MA tree (lower value = bigger
                        // context model)
  // Alternative heuristic tweaks.
  bool use_splitting_heuristics;
  size_t splitting_heuristics_max_properties;
  float splitting_heuristics_node_threshold;

  // Brotli options
  int brotli_effort;  // 0..11

  std::vector<int> predictor;  // predictor to use for each channel. last one
                               // gets repeated if needed

  int nb_wp_modes;

  // deprecated
  bool debug;      // produce debug images, including (for MABEGABRAC only) a
                   // compression heatmap
  Image *heatmap;  // produced if debug==true
};

void set_default_modular_options(struct modular_options &o);

}  // namespace jxl

#endif
