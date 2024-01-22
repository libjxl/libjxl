#!/bin/bash
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# Helper script to fix includes. Requires clang-tidy 18+.
# Might need jxl_{threads_}_exports.h copied to includes/jxl
# Also might require:
#   `export CPLUS_INCLUDE_PATH=/usr/lib/llvm-16/lib/clang/16/include/

SRC=$1
HERE=`pwd`
CLANG_TIDY_CONFIG="{\
  Checks: '-*,misc-include-cleaner',\
  CheckOptions: {\
    'misc-include-cleaner.IgnoreHeaders': 'gtest/.*;gmock/.*;testing.h'\
  }\
}"

`which clang-tidy` \
  -config="${CLANG_TIDY_CONFIG}" \
  -p build \
  -format-style=file \
  -quiet \
  -fix-errors \
  --extra-arg=-I${HERE}/lib/include \
  $SRC
sed -i -r 's/#include "jxl\/(.+)"/#include <jxl\/\1>/g' $SRC
clang-format -i $SRC
