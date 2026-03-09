# Copyright (c) the JPEG XL Project Authors. All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

FROM debian:trixie@sha256:6d87375016340817ac2391e670971725a9981cfc24e221c47734681ed0f6c0f5

RUN apt update
RUN apt install -y cmake clang doxygen g++ extra-cmake-modules \
  libgif-dev libjpeg-dev ninja-build libgoogle-perftools-dev \
  graphviz
RUN apt install -y qt6-base-dev libwebp-dev libgimp2.0-dev libopenexr-dev \
  libgtest-dev libgmock-dev libbenchmark-dev libbenchmark-tools
RUN apt install -y git

WORKDIR $OPT

# Get the emsdk repo.
RUN git clone https://github.com/emscripten-core/emsdk.git

# Enter that directory.
WORKDIR $OPT/emsdk

# Download and install the latest SDK tools.
RUN ./emsdk install latest

# Make the "latest" SDK "active" for the current user. (writes ~/.emscripten file)
RUN ./emsdk activate latest

# Install node
RUN apt install -y nodejs

RUN echo "NODE_JS='/usr/bin/node'" >> $OPT/emsdk/.emscripten

WORKDIR /workspaces/libjxl

CMD /bin/bash