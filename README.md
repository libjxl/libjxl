# JPEG XL reference implementation

This repository contains a reference implementation of JPEG XL (encoder and
decoder). As [previously announced](https://jpeg.org/items/20190803_press.html),
it is available under a royalty-free and open source license (Apache 2).

**NOTE**

JPEG XL is in the final stages of standardization, but minor changes to the
codestream are still likely. WARNING: until further notice, do not depend on
future decoders being able to decode the output of a current encoder.

## Checking out the code

This repository uses git submodules to handle some third party dependencies
under `third_party/`. To check out these dependencies as well clone the
repository with `--recursive`:

```shell
git clone git@gitlab.com:wg1/jpeg-xl.git --recursive
```

If you didn't check out recursively, and after any update run the following
command to check out the git submodules.

```shell
git submodule update --init --recursive
```

## Minimum build dependencies

Apart from the dependencies in third_party, some of the tools use external
dependencies that need to be installed in your system first. For a Debian/Ubuntu
based Linux distribution install:

```shell
sudo apt install cmake clang-6.0 g++-8 qtbase5-dev libqt5x11extras5-dev \
  extra-cmake-modules libgif-dev libjpeg-dev ninja-build
```

For developing changes in JPEG XL, take a look at the
[Building and Testing changes](doc/building_and_testing.md) guide.

Make sure your default "clang" compiler is at least version 6 running

```bash
clang --version
```

If it still shows an old version despite having a clang-6.0 installed, you need
to update the default clang compiler. In Debian-based systems run:

```bash
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-6.0 100
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-6.0 100
```

## Building

The project builds with cmake. We currently support Linux (tested with Debian).
To build the "release" version, you can use the following helper command:

```shell
./ci.sh release
```

This will build and test the project, and leave the binaries in the `build/`
directory. Check out the `tools` subdirectory for command-line tools that
interact with the library. You can set the environment variable SKIP_TEST=1 to
skip the test stage.

There are other build versions with more debug information useful when
developing. You can read more about it in the
[Building and Testing changes](doc/building_and_testing.md) guide.

## Documentation

*   [Developing in GitLab](doc/developing_in_gitlab.md)
*   [XL Overview](doc/xl_overview.md)
*   [Building and Testing changes](doc/building_and_testing.md)
*   [Software Contribution Guidelines](doc/guidelines.md)
*   [JPEG XL committee draft](https://arxiv.org/abs/1908.03565)
*   [Introductory paper](https://www.spiedigitallibrary.org/proceedings/Download?fullDOI=10.1117%2F12.2529237)
