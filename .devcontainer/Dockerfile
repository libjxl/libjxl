FROM mcr.microsoft.com/vscode/devcontainers/base:ubuntu-20.04

RUN apt-get update && \
    apt install -y \
        ccache \
        clang-7 \
        cmake \
        doxygen \
        libbenchmark-dev \
        libbenchmark-tools \
        libbrotli-dev \
        libgdk-pixbuf2.0-dev \
        libgif-dev \
        libgtest-dev \
        libgtk2.0-dev  \
        libjpeg-dev \
        libopenexr-dev \
        libpng-dev \
        libwebp-dev \
        ninja-build \
        pkg-config \
        xdg-utils \
        xvfb

ENV CC=clang-7
ENV CXX=clang++-7
