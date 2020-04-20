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

struct ModularOptions {
  // Decoding options:

  // When true, only decode header, not the image data.
  bool identify = false;

  /// Used in both encode and decode:

  // If full_header==false, need to specify how many channels to expect.
  int nb_channels = 1;

  // The first <skipchannels> channels will not be encoded/decoded.
  int skipchannels = 0;

  // Stop encoding/decoding when reaching a (non-meta) channel that has a
  // dimension bigger than max_chan_size.
  size_t max_chan_size = 0xFFFFFF;

  // Encoding options (some of which are needed during decoding too):
  enum EntropyCoder {
    kBrotli = 0,
    kMAANS,
  };
  EntropyCoder entropy_coder = kMAANS;

  // MA options:
  // Number of iterations to do to learn a MA tree (does not have to be an
  // integer; if zero there is no MA context model).
  float nb_repeats =
      .5f;  // learn MA tree by looking at 50% of the rows, in random order

  // Maximum number of (previous channel) properties to use in the MA trees
  int max_properties = 0;  // no previous channels

  // Alternative heuristic tweaks.
  size_t splitting_heuristics_max_properties;
  float splitting_heuristics_node_threshold;

  // Brotli options
  int brotli_effort = 11;  // 0..11

  // Predictor to use for each channel. If there are more channels than
  // predictors here the last one, or the default if empty, gets repeated.
  std::vector<int> predictor;

  int nb_wp_modes = 1;
};

}  // namespace jxl

#endif
