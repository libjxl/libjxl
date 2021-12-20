#!/bin/bash -e
# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

mkdir -p build
cd build

CXX=clang++

[ -f lodepng.cpp ] || wget https://raw.githubusercontent.com/lvandeve/lodepng/8c6a9e30576f07bf470ad6f09458a2dcd7a6a84a/lodepng.cpp
[ -f lodepng.h ] || wget https://raw.githubusercontent.com/lvandeve/lodepng/8c6a9e30576f07bf470ad6f09458a2dcd7a6a84a/lodepng.h
[ -f lodepng.o ] || $CXX lodepng.cpp -O3 -mavx2 -o lodepng.o -c

$CXX -O3 -mavx2 -g -DFASTLL_ENABLE_AVX2_INTRINSICS -fopenmp \
  -I. lodepng.o \
  ../fast_lossless.cc ../fast_lossless_main.cc \
  -o fast_lossless
