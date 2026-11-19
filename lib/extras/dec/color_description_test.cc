// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/extras/dec/color_description.h"

#include <jxl/color_encoding.h>

#include <clocale>
#include <cstdio>
#include <string>

#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/test_utils.h"
#include "lib/jxl/testing.h"

namespace jxl {

// Verify ParseDescription(Description) yields the same ColorEncoding
TEST(ColorDescriptionTest, RoundTripAll) {
  for (const auto& cdesc : test::AllEncodings()) {
    const ColorEncoding c_original = test::ColorEncodingFromDescriptor(cdesc);
    const std::string description = Description(c_original);
    printf("%s\n", description.c_str());

    JxlColorEncoding c_external = {};
    EXPECT_TRUE(ParseDescription(description, &c_external));
    ColorEncoding c_internal;
    EXPECT_TRUE(c_internal.FromExternal(c_external));
    EXPECT_TRUE(c_original.SameColorEncoding(c_internal))
        << "Where c_original=" << c_original
        << " and c_internal=" << c_internal;
  }
}

TEST(ColorDescriptionTest, NanGamma) {
  const std::string description = "Gra_2_Per_gnan";
  JxlColorEncoding c;
  EXPECT_FALSE(ParseDescription(description, &c));
}

// Fractional values in a description (gamma, custom white point / primaries)
// must parse independently of the process LC_NUMERIC locale. Under a locale
// whose decimal separator is ',', the old strtod-based parser truncated
// "2.2" to 2 and "0.3127" to 0.
TEST(ColorDescriptionTest, FractionalValuesIgnoreLocale) {
  const char* current = std::setlocale(LC_NUMERIC, nullptr);
  const std::string saved_locale = current ? current : "C";
  bool have_comma_locale = false;
  for (const char* name : {"de_DE.UTF-8", "de_DE.utf8", "de_DE", "fr_FR.UTF-8",
                           "nl_NL.UTF-8", "de_DE.ISO8859-15"}) {
    if (std::setlocale(LC_NUMERIC, name) != nullptr &&
        std::string(std::localeconv()->decimal_point) == ",") {
      have_comma_locale = true;
      break;
    }
  }
  if (!have_comma_locale) {
    std::setlocale(LC_NUMERIC, saved_locale.c_str());
    GTEST_SKIP() << "No locale with ',' decimal separator available.";
  }

  JxlColorEncoding gamma = {};
  EXPECT_TRUE(ParseDescription("Gra_D65_Per_g2.2", &gamma));
  EXPECT_EQ(gamma.transfer_function, JXL_TRANSFER_FUNCTION_GAMMA);
  EXPECT_NEAR(gamma.gamma, 2.2, 1e-9);

  JxlColorEncoding wp = {};
  EXPECT_TRUE(ParseDescription("RGB_0.3127;0.3290_SRG_Rel_SRG", &wp));
  EXPECT_EQ(wp.white_point, JXL_WHITE_POINT_CUSTOM);
  EXPECT_NEAR(wp.white_point_xy[0], 0.3127, 1e-9);
  EXPECT_NEAR(wp.white_point_xy[1], 0.3290, 1e-9);

  std::setlocale(LC_NUMERIC, saved_locale.c_str());
}

}  // namespace jxl
