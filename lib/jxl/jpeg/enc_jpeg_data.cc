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

#include "lib/jxl/jpeg/enc_jpeg_data.h"

#include <brotli/encode.h>
#include <stdio.h>

namespace jxl {
namespace jpeg {

Status EncodeJPEGData(JPEGData& jpeg_data, PaddedBytes* bytes) {
  jpeg_data.app_marker_type.resize(jpeg_data.app_data.size());
  size_t num_icc = 0;
  size_t num_icc_jpeg = 0;
  for (size_t i = 0; i < jpeg_data.app_data.size(); i++) {
    const auto& app = jpeg_data.app_data[i];
    bool is_icc = app[0] == 0xE2;
    constexpr char kICCTag[] = "ICC_PROFILE";
    for (size_t i = 0; i < sizeof(kICCTag); i++) {
      is_icc &= app.size() > 3 + i && app[3 + i] == kICCTag[i];
    }
    if (is_icc) {
      is_icc &= app[15] == num_icc + 1;
      if (num_icc_jpeg == 0) {
        num_icc_jpeg = app[16];
      }
      is_icc &= num_icc_jpeg == app[16];
    }
    if (is_icc) {
      num_icc++;
      jpeg_data.app_marker_type[i] = AppMarkerType::kICC;
    }
  }
  if (num_icc != num_icc_jpeg) {
    return JXL_FAILURE("Invalid ICC chunks");
  }
  BitWriter writer;
  JXL_RETURN_IF_ERROR(Bundle::Write(jpeg_data, &writer, 0, nullptr));
  writer.ZeroPadToByte();
  *bytes = std::move(writer).TakeBytes();
  BrotliEncoderState* brotli_enc =
      BrotliEncoderCreateInstance(nullptr, nullptr, nullptr);
  BrotliEncoderSetParameter(brotli_enc, BROTLI_PARAM_QUALITY, 11);
  size_t total_data = 0;
  for (size_t i = 0; i < jpeg_data.app_data.size(); i++) {
    if (jpeg_data.app_marker_type[i] != AppMarkerType::kUnknown) {
      continue;
    }
    total_data += jpeg_data.app_data[i].size();
  }
  for (size_t i = 0; i < jpeg_data.com_data.size(); i++) {
    total_data += jpeg_data.com_data[i].size();
  }
  for (size_t i = 0; i < jpeg_data.inter_marker_data.size(); i++) {
    total_data += jpeg_data.inter_marker_data[i].size();
  }
  total_data += jpeg_data.tail_data.size();
  size_t initial_size = bytes->size();
  size_t brotli_capacity = BrotliEncoderMaxCompressedSize(total_data);
  BrotliEncoderSetParameter(brotli_enc, BROTLI_PARAM_SIZE_HINT, total_data);
  bytes->resize(bytes->size() + brotli_capacity);
  size_t enc_size = 0;
  auto br_append = [&](const std::vector<uint8_t>& data, bool last) {
    size_t available_in = data.size();
    const uint8_t* in = data.data();
    uint8_t* out = &(*bytes)[initial_size + enc_size];
    do {
      JXL_CHECK(BrotliEncoderCompressStream(
          brotli_enc, last ? BROTLI_OPERATION_FINISH : BROTLI_OPERATION_PROCESS,
          &available_in, &in, &brotli_capacity, &out, &enc_size));
    } while (BrotliEncoderHasMoreOutput(brotli_enc) || available_in > 0);
  };

  for (size_t i = 0; i < jpeg_data.app_data.size(); i++) {
    if (jpeg_data.app_marker_type[i] != AppMarkerType::kUnknown) {
      continue;
    }
    br_append(jpeg_data.app_data[i], /*last=*/false);
  }
  for (size_t i = 0; i < jpeg_data.com_data.size(); i++) {
    br_append(jpeg_data.com_data[i], /*last=*/false);
  }
  for (size_t i = 0; i < jpeg_data.inter_marker_data.size(); i++) {
    br_append(jpeg_data.inter_marker_data[i], /*last=*/false);
  }
  br_append(jpeg_data.tail_data, /*last=*/true);
  BrotliEncoderDestroyInstance(brotli_enc);
  bytes->resize(initial_size + enc_size);
  return true;
}
}  // namespace jpeg
}  // namespace jxl
