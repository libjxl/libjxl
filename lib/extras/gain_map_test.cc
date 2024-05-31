// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "jxl/gain_map.h"

#include <jxl/encode.h>
#include <stdint.h>

#include <fstream>

#include "lib/jxl/base/span.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/test_memory_manager.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace {
bool ColorEncodingsEqual(const JxlColorEncoding& lhs,
                         const JxlColorEncoding& rhs) {
  return lhs.color_space == rhs.color_space &&
         lhs.white_point == rhs.white_point &&
         std::memcmp(lhs.white_point_xy, rhs.white_point_xy,
                     sizeof(lhs.white_point_xy)) == 0 &&
         lhs.primaries == rhs.primaries &&
         std::memcmp(lhs.primaries_red_xy, rhs.primaries_red_xy,
                     sizeof(lhs.primaries_red_xy)) == 0 &&
         std::memcmp(lhs.primaries_green_xy, rhs.primaries_green_xy,
                     sizeof(lhs.primaries_green_xy)) == 0 &&
         std::memcmp(lhs.primaries_blue_xy, rhs.primaries_blue_xy,
                     sizeof(lhs.primaries_blue_xy)) == 0 &&
         lhs.transfer_function == rhs.transfer_function &&
         lhs.gamma == rhs.gamma && lhs.rendering_intent == rhs.rendering_intent;
}

std::vector<uint8_t> GoldenTestGainMap(bool has_icc, bool has_color_encoding) {
  // Define the parts of the gain map
  uint8_t jhgm_version = 0x00;
  std::vector<uint8_t> gain_map_metadata_size = {0x00, 0x58};  // 88 in decimal
  // TODO(firsching): Replace with more realistic data
  std::string first_placeholder =
      "placeholder gain map metadata, fill with actual example after (ISO "
      "21496-1) is finalized";

  std::vector<uint8_t> color_encoding_size = {0x00, 0x00, 0x00, 0x00};
  if (has_color_encoding) {
    color_encoding_size[3] = 0x03;
  }
  std::vector<uint8_t> color_encoding = {0x50, 0xb4, 0x00};

  std::vector<uint8_t> icc_size = {0x00, 0x00, 0x00, 0x00};
  if (has_icc) {
    icc_size = {0x00, 0x00, 0x01, 0x7A};  // 378 in decimal
  }
  const uint8_t* raw_icc_data = reinterpret_cast<const uint8_t*>(
      "\x1f\x8b\x01\x33\x38\x18\x00\x30\x20\x8c"
      "\xe6\x81\x59\x00\x64\x69\x2c\x50\x80\xfc\xbc\x8e\xd6\xf7\x84\x66"
      "\x0c\x46\x68\x8e\xc9\x1e\x35\xb7\xe6\x79\x0a\x38\x0f\x2d\x0b\x15"
      "\x94\x56\x90\x28\x39\x09\x48\x27\x1f\xc3\x2a\x85\xb3\x82\x01\x46"
      "\x86\x28\x19\xe4\x64\x24\x3d\x69\x74\xa4\x9e\x24\x3e\x4a\x6d\x31"
      "\xa4\x54\x2a\x35\xc5\xf0\x9e\x34\xa0\x27\x8d\x8a\x04\xb0\xec\x8e"
      "\xdb\xee\xcc\x40\x5e\x71\x96\xcc\x99\x3e\x3a\x18\x42\x3f\xc0\x06"
      "\x5c\x04\xaf\x79\xdf\xa3\x7e\x47\x0f\x0f\xbd\x08\xd8\x3d\xa9\xd9"
      "\xf9\xdd\x3e\x57\x30\xa5\x36\x7e\xcc\x96\x57\xfa\x11\x41\x71\xdd"
      "\x1b\x8d\xa1\x79\xa5\x5c\xe4\x3e\xb4\xde\xde\xdf\x9c\xe4\xee\x4f"
      "\x28\xf8\x3e\x4c\xe2\xfa\x36\xfb\x3f\x13\x97\x1a\xc9\x34\xef\xc0"
      "\x17\x9a\x15\x92\x03\x4b\x83\xd5\x62\xf3\xc4\x20\xc7\xf3\x1c\x4c"
      "\x0d\xc2\xe1\x8c\x39\xc8\x64\xdc\xc8\xa5\x7b\x93\x18\xec\xec\xc5"
      "\xe0\x0a\x2f\xf0\x95\x12\x05\x0d\x60\x92\xa1\xf0\x0e\x65\x80\xa5"
      "\x52\xa1\xf3\x94\x3f\x6f\xc7\x0a\x45\x94\xc8\x1a\xc5\xf0\x34\xcd"
      "\xe3\x1d\x9b\xaf\x70\xfe\x8f\x19\x1d\x1f\x69\xba\x1d\xc2\xdf\xd9"
      "\x0b\x1f\xa6\x38\x02\x66\x78\x88\x72\x84\x76\xad\x04\x80\xd3\x69"
      "\x44\x71\x05\x71\xd2\xeb\xdf\xbf\xf3\x29\x70\x76\x02\xf6\x85\xf8"
      "\xf7\xef\xde\x90\x7f\xff\xf6\x15\x41\x96\x0b\x02\xd7\x15\xfb\xbe"
      "\x81\x18\x6c\x1d\xb2\x10\x18\xe2\x07\xea\x12\x40\x9b\x44\x58\xf1"
      "\x75\x85\x37\x0f\xd0\x68\x96\x7c\x39\x85\xf8\xea\xf7\x62\x47\xb0"
      "\x42\xeb\x43\x06\x70\xe4\x15\x96\x2a\x8b\x65\x3e\x4d\x98\x51\x03"
      "\x63\xf6\x14\xf5\xe5\xe0\x7a\x0e\xdf\x3e\x1b\x45\x9a\xef\x87\xe1"
      "\x3f\xcf\x69\x5c\x43\xda\x68\xde\x84\x26\x38\x6a\xf0\x35\xcb\x08");
  std::vector<uint8_t> icc_data;
  icc_data.assign(raw_icc_data, raw_icc_data + 378);
  std::string second_placeholder =
      "placeholder for an actual naked JPEG XL codestream";

  // Assemble the gain map
  std::vector<uint8_t> gain_map;
  gain_map.push_back(jhgm_version);
  gain_map.insert(gain_map.end(), gain_map_metadata_size.begin(),
                  gain_map_metadata_size.end());
  gain_map.insert(gain_map.end(), first_placeholder.begin(),
                  first_placeholder.end());
  gain_map.insert(gain_map.end(), color_encoding_size.begin(),
                  color_encoding_size.end());
  if (has_color_encoding) {
    gain_map.insert(gain_map.end(), color_encoding.begin(),
                    color_encoding.end());
  }
  gain_map.insert(gain_map.end(), icc_size.begin(), icc_size.end());
  if (has_icc) {
    gain_map.insert(gain_map.end(), icc_data.begin(), icc_data.end());
  }
  gain_map.insert(gain_map.end(), second_placeholder.begin(),
                  second_placeholder.end());

  return gain_map;
}

}  // namespace

namespace jxl {
namespace {

struct GainMapTestParams {
  bool has_color_encoding;
  std::vector<uint8_t> icc_data;
};

class GainMapTest : public ::testing::TestWithParam<GainMapTestParams> {
 protected:
  JxlMemoryManager* const memory_manager = jxl::test::MemoryManager();
  JxlGainMapBundle orig_bundle;
};

TEST_P(GainMapTest, GainMapRoundtrip) {
  size_t bundle_size;
  const GainMapTestParams& params = GetParam();
  std::vector<uint8_t> golden_gain_map =
      GoldenTestGainMap(!params.icc_data.empty(), params.has_color_encoding);
  // Initialize the bundle with some test data
  orig_bundle.jhgm_version = 0;
  const char* metadata_str =
      "placeholder gain map metadata, fill with actual example after (ISO "
      "21496-1) is finalized";
  std::vector<uint8_t> gain_map_metadata(metadata_str,
                                         metadata_str + strlen(metadata_str));
  orig_bundle.gain_map_metadata_size = gain_map_metadata.size();
  orig_bundle.gain_map_metadata = gain_map_metadata.data();

  // Use the ICC profile from the parameter
  orig_bundle.has_color_encoding = params.has_color_encoding;
  if (orig_bundle.has_color_encoding) {
    JxlColorEncoding color_encoding = {};
    JxlColorEncodingSetToLinearSRGB(&color_encoding, /*is_gray=*/JXL_FALSE);
    orig_bundle.color_encoding = color_encoding;
  }

  std::vector<uint8_t> alt_icc(params.icc_data.begin(), params.icc_data.end());
  orig_bundle.alt_icc = alt_icc.data();
  orig_bundle.alt_icc_size = alt_icc.size();

  const char* gain_map_str =
      "placeholder for an actual naked JPEG XL codestream";
  std::vector<uint8_t> gain_map(gain_map_str,
                                gain_map_str + strlen(gain_map_str));
  orig_bundle.gain_map_size = gain_map.size();
  orig_bundle.gain_map = gain_map.data();

  ASSERT_TRUE(
      JxlGainMapGetBundleSize(memory_manager, &orig_bundle, &bundle_size));
  EXPECT_EQ(bundle_size, golden_gain_map.size());

  std::vector<uint8_t> buffer(bundle_size);
  size_t bytes_written;
  ASSERT_TRUE(JxlGainMapWriteBundle(memory_manager, &orig_bundle, buffer.data(),
                                    buffer.size(), &bytes_written));
  EXPECT_EQ(bytes_written, bundle_size);
  std::ofstream dump("/tmp/gainmap.bin", std::ios::out);
  dump.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  dump.close();
  EXPECT_EQ(buffer[0], orig_bundle.jhgm_version);
  EXPECT_EQ(buffer.size(), golden_gain_map.size());
  EXPECT_TRUE(
      std::equal(buffer.begin(), buffer.end(), golden_gain_map.begin()));

  JxlGainMapBundle output_bundle;
  JxlGainMapGetBufferSizes(memory_manager, &output_bundle, buffer.data(),
                           buffer.size());
  EXPECT_EQ(output_bundle.gain_map_size, orig_bundle.gain_map_size);
  EXPECT_EQ(output_bundle.gain_map_metadata_size,
            orig_bundle.gain_map_metadata_size);
  EXPECT_EQ(output_bundle.alt_icc_size, orig_bundle.alt_icc_size);
}

JXL_GTEST_INSTANTIATE_TEST_SUITE_P(
    GainMapTestCases, GainMapTest,
    ::testing::Values(GainMapTestParams{true, std::vector<uint8_t>()},
                      GainMapTestParams{true, test::GetIccTestProfile()},
                      GainMapTestParams{false, test::GetIccTestProfile()},
                      GainMapTestParams{false, std::vector<uint8_t>()}),
    [](const testing::TestParamInfo<GainMapTest::ParamType>& info) {
      std::string name =
          "HasColorEncoding" + std::to_string(info.param.has_color_encoding);

      name += "ICCSize" + std::to_string(info.param.icc_data.size());

      return name;
    });

}  // namespace
}  // namespace jxl
