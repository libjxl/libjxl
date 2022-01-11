#!/bin/bash -e
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.


DIR=$(realpath $(dirname $0))

mkdir -p /tmp/build-android
cd /tmp/build-android

CXX=$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang++

[ -f lodepng.cpp ] || wget https://raw.githubusercontent.com/lvandeve/lodepng/8c6a9e30576f07bf470ad6f09458a2dcd7a6a84a/lodepng.cpp
[ -f lodepng.h ] || wget https://raw.githubusercontent.com/lvandeve/lodepng/8c6a9e30576f07bf470ad6f09458a2dcd7a6a84a/lodepng.h
[ -f lodepng.o ] || $CXX lodepng.cpp -O3 -o lodepng.o -c

$CXX -O3 -g -DFASTLL_ENABLE_NEON_INTRINSICS -fopenmp \
  -I. lodepng.o \
  ${DIR}/fast_lossless.cc ${DIR}/fast_lossless_main.cc \
  -o fast_lossless
