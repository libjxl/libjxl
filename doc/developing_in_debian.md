# Developing in Debian

These instructions assume an up-to-date Debian/Ubuntu system.
For other platforms, please instead use the following:

* [Cross Compiling for Windows with Crossroad](developing_with_crossroad.md).

## Minimum build dependencies

Apart from the dependencies in `third_party`, some of the tools use external
dependencies that need to be installed on your system first:

```bash
sudo apt install clang cmake doxygen graphviz ninja-build libpng-dev
```

Make sure your default `clang` compiler is at least version 6 by running

```bash
clang --version
```

If it still shows an old version despite having, for example, `clang-7`
installed, you need to update the default `clang` compiler.
On Debian-based systems run:

```bash
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-7 100
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-7 100
```

Optionally, to compile some of the extra tool support and tests you can install
the following packages:

```bash
sudo apt install extra-cmake-modules g++ libbenchmark-dev libbenchmark-tools \
  libgif-dev libgoogle-perftools-dev libgtest-dev libjpeg-dev libopenexr-dev \
  libwebp-dev qt6-base-dev xdg-utils
```

For the lint/coverage commands, you will also need additional packages:

```bash
sudo apt install clang-format clang-tidy curl parallel gcovr
```

## Building

The `libjxl` project uses CMake to build. We provide a script that simplifies
the invocation. To build and test the project, run

```bash
./ci.sh opt
```

This writes binaries to `build/tools` and runs unit tests. More information
on [build modes and testing](building_and_testing.md) is available.

## Debian Build with Docker

There is a Dockerfile contained in the `.devcontainer` which can be used
for building on windows to create a docker image. It is used as a devcontainer
in your editor, or from the command line via:

```bash
docker build . -f .devcontainer/Dockerfile -t libjxl
docker run --mount type=bind,src=.,dst=/workspaces/libjxl -i libjxl
./ci.sh release
```

See also the building WASM for how to use this image to build WASM builds.
