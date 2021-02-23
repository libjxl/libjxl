/* Copyright (c) the JPEG XL Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef JXL_BOX_H_
#define JXL_BOX_H_

#include <cstdint>
#include <vector>

#include "lib/jxl/base/span.h"

namespace jxl {

// A top-leves box in the box format.
typedef struct JxlBoxStruct {
  // The type of the box.
  // If "uuid", use extended_type instead
  char type[4];

  // The extended_type is only used when type == "uuid".
  // Extended types are not used in JXL. However, the box format itself
  // supports this so they are handled correctly.
  char extended_type[16];

  // Box data.
  Span<const uint8_t> data;

  // If the size is not given, the datasize extends to the end of the file.
  // If this field is false, the size field is not encoded when the box is
  // serialized.
  bool data_size_given;

  // Copies an encoded version of this box into `out`.
  void Encode(std::vector<uint8_t>* out);

  // If successfull, returns true and sets `in` to be the rest data (if any).
  // If unsuccessful, returns error and doesn't modify `in`.
  Status Decode(Span<uint8_t>* in);
} JxlBox;

typedef struct JxlContainerStruct {
  std::vector<JxlBox> boxes;

  void Encode(std::vector<uint8_t>* out);

  // If successful, returns true and sets `in` to be the rest data (if any).
  // If unsuccessful, returns error and doesn't modify `in`.
  Status Decode(Span<uint8_t>* in);
} JxlContainer;

}  // namespace jxl

#endif /* JXL_BOX_H_ */
