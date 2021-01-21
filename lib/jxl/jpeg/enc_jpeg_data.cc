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

namespace {

// TODO(eustas): move to jpeg_data, to use from codec_jpg as well.
// See if there is a canonically chunked ICC profile and mark corresponding
// app-tags with AppMarkerType::kICC.
Status DetectIccProfile(JPEGData& jpeg_data) {
  JXL_DASSERT(jpeg_data.app_data.size() == jpeg_data.app_marker_type.size());
  size_t num_icc = 0;
  size_t num_icc_jpeg = 0;
  for (size_t i = 0; i < jpeg_data.app_data.size(); i++) {
    const auto& app = jpeg_data.app_data[i];
    size_t pos = 0;
    if (app[pos++] != 0xE2) continue;
    // At least APPn + size; otherwise it should be intermarker-data.
    JXL_DASSERT(app.size() >= 3);
    size_t tag_length = (app[pos] << 8) + app[pos + 1];
    pos += 2;
    JXL_DASSERT(app.size() == tag_length + 1);
    constexpr char kICCTag[] = "ICC_PROFILE";  // Implicit \0 at the end
    constexpr size_t kICCTagSize = sizeof(kICCTag);
    // Empty payload is 2 bytes for tag length itself + signature
    if (tag_length < 2 + kICCTagSize) continue;

    bool is_icc = true;
    for (size_t j = 0; j < kICCTagSize; j++) {
      is_icc &= (app[pos++] == kICCTag[j]);
    }

    if (is_icc) {
      uint8_t chunk_id = app[pos++];
      uint8_t num_chunks = app[pos++];
      is_icc &= (chunk_id == num_icc + 1);
      if (num_icc_jpeg == 0) num_icc_jpeg = num_chunks;
      is_icc &= (num_icc_jpeg == num_chunks);
    }

    if (is_icc) {
      num_icc++;
      jpeg_data.app_marker_type[i] = AppMarkerType::kICC;
    }
  }
  if (num_icc != num_icc_jpeg) {
    return JXL_FAILURE("Invalid ICC chunks");
  }
  return true;
}

}  // namespace

Status EncodeJPEGData(JPEGData& jpeg_data, PaddedBytes* bytes) {
  jpeg_data.app_marker_type.resize(jpeg_data.app_data.size(),
                                   AppMarkerType::kUnknown);
  JXL_RETURN_IF_ERROR(DetectIccProfile(jpeg_data));
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
