// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/jpeg/enc_jpeg_data_reader.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "lib/jxl/jpeg/dec_jpeg_data_writer.h"
#include "lib/jxl/jpeg/jpeg_data.h"
#include "lib/jxl/testing.h"

namespace jxl {
namespace jpeg {
namespace {

// A one-pixel, single-component JPEG with a zero DC coefficient and an EOB.
std::vector<uint8_t> MakeJpeg(const std::vector<uint8_t>& markers) {
  std::vector<uint8_t> bytes = {0xff, 0xd8};
  bytes.insert(bytes.end(), markers.begin(), markers.end());
  const std::vector<uint8_t> dqt = {0xff, 0xdb, 0x00, 0x43, 0x00};
  bytes.insert(bytes.end(), dqt.begin(), dqt.end());
  bytes.insert(bytes.end(), 64, 1);
  const std::vector<uint8_t> sof = {0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x01,
                                    0x00, 0x01, 0x01, 0x01, 0x11, 0x00};
  bytes.insert(bytes.end(), sof.begin(), sof.end());
  const std::vector<uint8_t> dht = {0xff, 0xc4, 0x00, 0x26};
  bytes.insert(bytes.end(), dht.begin(), dht.end());
  for (uint8_t table : {0x00, 0x10}) {
    bytes.push_back(table);
    bytes.push_back(1);
    bytes.insert(bytes.end(), 15, 0);
    bytes.push_back(0);
  }
  const std::vector<uint8_t> tail = {0xff, 0xda, 0x00, 0x08, 0x01, 0x01, 0x00,
                                     0x00, 0x3f, 0x00, 0x3f, 0xff, 0xd9};
  bytes.insert(bytes.end(), tail.begin(), tail.end());
  return bytes;
}

TEST(JpegMarkerReaderTest, PreservesAppAndCommentBytes) {
  const std::vector<uint8_t> markers = {0xff, 0xe1, 0x00, 0x05, 0x00,
                                        0xff, 0x80, 0xff, 0xfe, 0x00,
                                        0x05, 'a',  0x00, 0xff};
  const std::vector<uint8_t> bytes = MakeJpeg(markers);
  JPEGData jpg;
  ASSERT_TRUE(
      ReadJpeg(bytes.data(), bytes.size(), JpegReadMode::kReadAll, &jpg));
  ASSERT_EQ(1u, jpg.app_data.size());
  ASSERT_EQ(1u, jpg.com_data.size());
  EXPECT_EQ((std::vector<uint8_t>{0xe1, 0x00, 0x05, 0x00, 0xff, 0x80}),
            jpg.app_data[0]);
  EXPECT_EQ((std::vector<uint8_t>{0xfe, 0x00, 0x05, 'a', 0x00, 0xff}),
            jpg.com_data[0]);
  EXPECT_EQ(1, jpg.width);
  EXPECT_EQ(1, jpg.height);
#if JPEGXL_ENABLE_TRANSCODE_JPEG
  std::vector<uint8_t> reconstructed;
  ASSERT_TRUE(
      WriteJpeg(jpg, [&reconstructed](const uint8_t* data, size_t size) {
        reconstructed.insert(reconstructed.end(), data, data + size);
        return size;
      }));
  EXPECT_EQ(bytes, reconstructed);
#endif
}

TEST(JpegMarkerReaderTest, AcceptsEmptyPayloadAndEveryAppMarker) {
  std::vector<uint8_t> markers;
  for (int marker = 0xe0; marker <= 0xef; ++marker) {
    markers.insert(markers.end(), {0xff, static_cast<uint8_t>(marker), 0, 2});
  }
  markers.insert(markers.end(), {0xff, 0xfe, 0, 2});
  const std::vector<uint8_t> bytes = MakeJpeg(markers);
  JPEGData jpg;
  ASSERT_TRUE(
      ReadJpeg(bytes.data(), bytes.size(), JpegReadMode::kReadAll, &jpg));
  ASSERT_EQ(16u, jpg.app_data.size());
  for (size_t i = 0; i < jpg.app_data.size(); ++i) {
    EXPECT_EQ((std::vector<uint8_t>{static_cast<uint8_t>(0xe0 + i), 0, 2}),
              jpg.app_data[i]);
  }
  ASSERT_EQ(1u, jpg.com_data.size());
  EXPECT_EQ((std::vector<uint8_t>{0xfe, 0, 2}), jpg.com_data[0]);
}

TEST(JpegMarkerReaderTest, AcceptsMaximumLengthPayload) {
  for (uint8_t marker : {0xe0, 0xfe}) {
    std::vector<uint8_t> markers = {0xff, marker, 0xff, 0xff};
    markers.insert(markers.end(), 65533, 0x81);
    const std::vector<uint8_t> bytes = MakeJpeg(markers);
    JPEGData jpg;
    ASSERT_TRUE(
        ReadJpeg(bytes.data(), bytes.size(), JpegReadMode::kReadAll, &jpg));
    const auto& data = marker == 0xfe ? jpg.com_data : jpg.app_data;
    ASSERT_EQ(1u, data.size());
    EXPECT_EQ(std::vector<uint8_t>(markers.begin() + 1, markers.end()),
              data[0]);
  }
}

TEST(JpegMarkerReaderTest, RejectsInvalidAndTruncatedSegmentsWithoutStoring) {
  const std::vector<std::vector<uint8_t>> segments = {
      {}, {0}, {0, 0}, {0, 1}, {0, 3}, {0, 5, 0x81, 0x82}, {0xff, 0xff}};
  for (uint8_t marker : {0xe0, 0xfe}) {
    for (const auto& segment : segments) {
      std::vector<uint8_t> bytes = {0xff, 0xd8, 0xff, marker};
      bytes.insert(bytes.end(), segment.begin(), segment.end());
      JPEGData jpg;
      EXPECT_FALSE(
          ReadJpeg(bytes.data(), bytes.size(), JpegReadMode::kReadAll, &jpg));
      EXPECT_TRUE(jpg.app_data.empty());
      EXPECT_TRUE(jpg.com_data.empty());
    }
  }
}

}  // namespace
}  // namespace jpeg
}  // namespace jxl
