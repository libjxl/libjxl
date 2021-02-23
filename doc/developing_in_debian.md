# Developing in Debian

These instructions assume an up-to-date Debian/Ubuntu system.
For other platforms, please instead use the [Docker container](doc/developing_in_docker.md).

## Minimum build dependencies

Apart from the dependencies in third_party, some of the tools use external
dependencies that need to be installed in your system first:

```bash
sudo apt install cmake clang doxygen g++-8 extra-cmake-modules libgif-dev \
  libjpeg-dev ninja-build libgoogle-perftools-dev
```

Make sure your default "clang" compiler is at least version 6 running

```bash
clang --version
```

If it still shows an old version despite having for example a clang-7 installed, you need
to update the default clang compiler. In Debian-based systems run:

```bash
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-7 100
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-7 100
```

Optionally, to compile some of the extra tool support and tests you can install
the following packages:

```bash
sudo apt install qtbase5-dev libqt5x11extras5-dev libwebp-dev libgimp2.0-dev \
  libopenexr-dev libgtest-dev libgmock-dev libbenchmark-dev libbenchmark-tools
```

For the lint/coverage commands, you will also need additional packages:

```bash
sudo apt install clang-format clang-tidy curl parallel gcovr
```

## Building

The project uses CMake to build. We provide a script that simplifies the
invocation. To build and test the project, run

```bash
./ci.sh opt
```

This writes binaries to `build/tools` and runs unit tests. More information
on [build modes and testing](doc/building_and_testing.md) is available.
