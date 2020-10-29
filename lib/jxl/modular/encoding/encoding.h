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

#ifndef LIB_JXL_MODULAR_ENCODING_ENCODING_H_
#define LIB_JXL_MODULAR_ENCODING_ENCODING_H_

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "lib/jxl/aux_out_fwd.h"
#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/padded_bytes.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/enc_ans.h"
#include "lib/jxl/enc_bit_writer.h"
#include "lib/jxl/image.h"
#include "lib/jxl/modular/encoding/context_predict.h"
#include "lib/jxl/modular/encoding/ma.h"
#include "lib/jxl/modular/image/image.h"
#include "lib/jxl/modular/options.h"
#include "lib/jxl/modular/transform/transform.h"

namespace jxl {

struct GroupHeader : public Fields {
  GroupHeader();

  const char *Name() const override { return "GroupHeader"; }

  Status VisitFields(Visitor *JXL_RESTRICT visitor) override {
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(false, &use_global_tree));
    JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&wp_header));
    uint32_t num_transforms = static_cast<uint32_t>(transforms.size());
    JXL_QUIET_RETURN_IF_ERROR(visitor->U32(Val(0), Val(1), BitsOffset(4, 2),
                                           BitsOffset(8, 18), 0,
                                           &num_transforms));
    if (visitor->IsReading()) transforms.resize(num_transforms);
    for (size_t i = 0; i < num_transforms; i++) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->VisitNested(&transforms[i]));
    }
    return true;
  }

  bool use_global_tree;
  weighted::Header wp_header;

  std::vector<Transform> transforms;
};

void PrintTree(const Tree &tree, const std::string &path);
Tree LearnTree(std::vector<Predictor> predictors,
               std::vector<std::vector<int32_t>> &&props,
               std::vector<std::vector<int32_t>> &&residuals,
               size_t total_pixels, const ModularOptions &options,
               const std::vector<ModularMultiplierInfo> &multiplier_info = {},
               StaticPropRange static_prop_range = {});

// TODO(veluca): make cleaner interfaces.

Status ModularGenericCompress(
    Image &image, const ModularOptions &opts, BitWriter *writer,
    AuxOut *aux_out = nullptr, size_t layer = 0, size_t group_id = 0,
    // For gathering data for producing a global tree.
    std::vector<std::vector<int32_t>> *props = nullptr,
    std::vector<std::vector<int32_t>> *residuals = nullptr,
    size_t *total_pixels = nullptr,
    // For encoding with global tree.
    const Tree *tree = nullptr, GroupHeader *header = nullptr,
    std::vector<Token> *tokens = nullptr, size_t *widths = nullptr,
    // Plot tree (if enabled) and predictor usage map.
    bool want_debug = false);

// undo_transforms == N > 0: undo all transforms except the first N
//                           (e.g. to represent YCbCr420 losslessly)
// undo_transforms == 0: undo all transforms
// undo_transforms == -1: undo all transforms but don't clamp to range
// undo_transforms == -2: don't undo any transform
Status ModularGenericDecompress(BitReader *br, Image &image,
                                GroupHeader *header, size_t group_id,
                                ModularOptions *options,
                                int undo_transforms = -1,
                                const Tree *tree = nullptr,
                                const ANSCode *code = nullptr,
                                const std::vector<uint8_t> *ctx_map = nullptr);
}  // namespace jxl

#endif  // LIB_JXL_MODULAR_ENCODING_ENCODING_H_
