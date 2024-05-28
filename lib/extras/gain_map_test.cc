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
#include "lib/jxl/testing.h"

namespace {
// TODO: copied from decode_test.cc, move it somewhere so it can be re-used?

// Returns an ICC profile output by the JPEG XL decoder for RGB_D65_SRG_Rel_Lin,
// but with, on purpose, rXYZ, bXYZ and gXYZ (the RGB primaries) switched to a
// different order to ensure the profile does not match any known profile, so
// the encoder cannot encode it in a compact struct instead.
jxl::IccBytes GetIccTestProfile() {
  const uint8_t* profile = reinterpret_cast<const uint8_t*>(
      "\0\0\3\200lcms\0040\0\0mntrRGB XYZ "
      "\a\344\0\a\0\27\0\21\0$"
      "\0\37acspAPPL\0\0\0\1\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\1\0\0\366"
      "\326\0\1\0\0\0\0\323-lcms\372c\207\36\227\200{"
      "\2\232s\255\327\340\0\n\26\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
      "\0\0\0\0\0\0\0\0\rdesc\0\0\1 "
      "\0\0\0Bcprt\0\0\1d\0\0\1\0wtpt\0\0\2d\0\0\0\24chad\0\0\2x\0\0\0,"
      "bXYZ\0\0\2\244\0\0\0\24gXYZ\0\0\2\270\0\0\0\24rXYZ\0\0\2\314\0\0\0\24rTR"
      "C\0\0\2\340\0\0\0 gTRC\0\0\2\340\0\0\0 bTRC\0\0\2\340\0\0\0 "
      "chrm\0\0\3\0\0\0\0$dmnd\0\0\3$\0\0\0("
      "dmdd\0\0\3L\0\0\0002mluc\0\0\0\0\0\0\0\1\0\0\0\fenUS\0\0\0&"
      "\0\0\0\34\0R\0G\0B\0_\0D\0006\0005\0_\0S\0R\0G\0_\0R\0e\0l\0_"
      "\0L\0i\0n\0\0mluc\0\0\0\0\0\0\0\1\0\0\0\fenUS\0\0\0\344\0\0\0\34\0C\0o\0"
      "p\0y\0r\0i\0g\0h\0t\0 \0002\0000\0001\08\0 \0G\0o\0o\0g\0l\0e\0 "
      "\0L\0L\0C\0,\0 \0C\0C\0-\0B\0Y\0-\0S\0A\0 \0003\0.\0000\0 "
      "\0U\0n\0p\0o\0r\0t\0e\0d\0 "
      "\0l\0i\0c\0e\0n\0s\0e\0(\0h\0t\0t\0p\0s\0:\0/\0/"
      "\0c\0r\0e\0a\0t\0i\0v\0e\0c\0o\0m\0m\0o\0n\0s\0.\0o\0r\0g\0/"
      "\0l\0i\0c\0e\0n\0s\0e\0s\0/\0b\0y\0-\0s\0a\0/\0003\0.\0000\0/"
      "\0l\0e\0g\0a\0l\0c\0o\0d\0e\0)XYZ "
      "\0\0\0\0\0\0\366\326\0\1\0\0\0\0\323-"
      "sf32\0\0\0\0\0\1\fB\0\0\5\336\377\377\363%"
      "\0\0\a\223\0\0\375\220\377\377\373\241\377\377\375\242\0\0\3\334\0\0\300"
      "nXYZ \0\0\0\0\0\0o\240\0\08\365\0\0\3\220XYZ "
      "\0\0\0\0\0\0$\237\0\0\17\204\0\0\266\304XYZ "
      "\0\0\0\0\0\0b\227\0\0\267\207\0\0\30\331para\0\0\0\0\0\3\0\0\0\1\0\0\0\1"
      "\0\0\0\0\0\0\0\1\0\0\0\0\0\0chrm\0\0\0\0\0\3\0\0\0\0\243\327\0\0T|"
      "\0\0L\315\0\0\231\232\0\0&"
      "g\0\0\17\\mluc\0\0\0\0\0\0\0\1\0\0\0\fenUS\0\0\0\f\0\0\0\34\0G\0o\0o\0g"
      "\0l\0emluc\0\0\0\0\0\0\0\1\0\0\0\fenUS\0\0\0\26\0\0\0\34\0I\0m\0a\0g\0e"
      "\0 \0c\0o\0d\0e\0c\0\0");
  size_t profile_size = 896;
  jxl::IccBytes icc_profile;
  icc_profile.assign(profile, profile + profile_size);
  return icc_profile;
}

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
}  // namespace

namespace jxl {
namespace {

TEST(GainMapTest, TestGetBundleSize) {
  JxlGainMapBundle orig_bundle;

  // Initialize the bundle with some test data
  orig_bundle.jhgm_version = 0;

  // TODO(firsching): Replace with more realistic data
  const char* metadata_str =
      "placeholder gain map metadata, fill with actual example after (ISO "
      "21496-1) is finalized";
  std::vector<uint8_t> gain_map_metadata(metadata_str,
                                         metadata_str + strlen(metadata_str));
  orig_bundle.gain_map_metadata_size = gain_map_metadata.size();
  orig_bundle.gain_map_metadata = gain_map_metadata.data();

  // TODO: actually set a color encoding here
  JxlColorEncoding color_encoding = {};
  JxlColorEncodingSetToLinearSRGB(&color_encoding, false);

  orig_bundle.color_encoding = color_encoding;

  // Use the ICC profile from GetIccTestProfile
  jxl::IccBytes icc_profile = GetIccTestProfile();
  std::vector<uint8_t> alt_icc(icc_profile.begin(), icc_profile.end());
  orig_bundle.has_color_encoding = true;
  orig_bundle.alt_icc = alt_icc.data();
  orig_bundle.alt_icc_size = alt_icc.size();

  // TODO: use real image here
  const char* gain_map_str =
      "placeholder for an actual naked JPEG XL codestream";
  std::vector<uint8_t> gain_map(gain_map_str,
                                gain_map_str + strlen(gain_map_str));
  orig_bundle.gain_map_size = gain_map.size();
  orig_bundle.gain_map = gain_map.data();

  // Call the function and verify the result
  size_t bundle_size = JxlGainMapGetBundleSize(&orig_bundle);
  //
  EXPECT_GT(bundle_size, 140);

  std::vector<uint8_t> buffer(bundle_size);
  EXPECT_EQ(JxlGainMapWriteBundle(&orig_bundle, buffer.data(), buffer.size()),
            bundle_size);

  EXPECT_EQ(buffer[0], orig_bundle.jhgm_version);

  std::ofstream dump("/tmp/gainmap.bin", std::ios::out);
  dump.write(reinterpret_cast<const char*>(buffer.data()), buffer.size());
  dump.close();
  JxlGainMapBundle output_bundle;
  JxlGainMapGetBufferSizes(&output_bundle, buffer.data(), buffer.size());
  EXPECT_EQ(output_bundle.gain_map_size, gain_map.size());
  EXPECT_EQ(output_bundle.gain_map_metadata_size, gain_map_metadata.size());
  // TODO: decode compressed icc!
  EXPECT_EQ(output_bundle.alt_icc_size, icc_profile.size());
  std::vector<uint8_t> output_metadata(output_bundle.gain_map_metadata_size);
  std::vector<uint8_t> output_gain_map(output_bundle.gain_map_size);
  std::vector<uint8_t> output_alt_icc(output_bundle.alt_icc_size);
  output_bundle.gain_map_metadata = output_metadata.data();
  output_bundle.gain_map = output_gain_map.data();
  output_bundle.alt_icc = output_alt_icc.data();
  JxlGainMapReadBundle(&output_bundle, buffer.data(), buffer.size());
  EXPECT_EQ(orig_bundle.jhgm_version, output_bundle.jhgm_version);
  EXPECT_EQ(orig_bundle.has_color_encoding, orig_bundle.has_color_encoding);
  // check color_encoding was recovered (add "==" to ColorEncoding?)
  EXPECT_TRUE(ColorEncodingsEqual(orig_bundle.color_encoding,
                                  output_bundle.color_encoding));
  EXPECT_TRUE(std::equal(gain_map_metadata.begin(), gain_map_metadata.end(),
                         output_metadata.begin()));
  EXPECT_TRUE(
      std::equal(gain_map.begin(), gain_map.end(), output_gain_map.begin()));
  // EXPECT_TRUE(std::equal(alt_icc.begin(), alt_icc.end(),
  // output_alt_icc.begin()));
}

}  // namespace
}  // namespace jxl