# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

FROM emscripten/emsdk:3.1.20

RUN apt-get update && \
    apt install -y \
        doxygen \
        ninja-build \
        pkg-config

ENV CC=emcc
ENV CXX=em++
ENV BUILD_TARGET=wasm32
