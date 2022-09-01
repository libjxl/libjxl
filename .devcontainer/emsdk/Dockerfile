FROM emscripten/emsdk:3.1.20

RUN apt-get update && \
    apt install -y \
        doxygen \
        ninja-build \
        pkg-config

ENV CC=emcc
ENV CXX=em++
ENV BUILD_TARGET=wasm32
