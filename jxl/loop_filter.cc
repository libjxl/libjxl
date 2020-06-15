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

#include "jxl/loop_filter.h"

#include "jxl/aux_out.h"
#include "jxl/fields.h"

namespace jxl {

LoopFilter::LoopFilter() { Bundle::Init(this); }

Status ReadLoopFilter(BitReader* JXL_RESTRICT reader,
                      LoopFilter* JXL_RESTRICT loop_filter) {
  return Bundle::Read(reader, loop_filter);
}

Status WriteLoopFilter(const LoopFilter& loop_filter,
                       BitWriter* JXL_RESTRICT writer, AuxOut* aux_out) {
  return Bundle::Write(loop_filter, writer, kLayerHeader, aux_out);
}

}  // namespace jxl
