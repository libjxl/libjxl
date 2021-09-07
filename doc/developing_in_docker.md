# Developing in Docker

Docker allows software to be run in a packaged container, isolated from the
host system. This allows code to be run in a standard environment instead
of dealing with different build environments during development.  It also
simplifies resolving external dependencies by including them in the automated
setup of the container environment.

## Set up the container

You can read installation instructions and download Docker for your
operating system at [Get Docker](https://docs.docker.com/get-docker/).

The image used by our builders is an Ubuntu Bionic image with all the
required dependencies and build tools installed. You can pull this image
from `gcr.io/jpegxl/jpegxl-builder` using the following command:

```bash
sudo docker pull gcr.io/jpegxl/jpegxl-builder
```

To use the Docker image you can run the following command:

```bash
sudo docker run -it --rm \
  --user $(id -u):$(id -g) \
  -v $HOME/jpeg-xl:/jpeg-xl -w /jpeg-xl \
  gcr.io/jpegxl/jpegxl-builder bash
```

This creates and runs a container that will be deleted after you exit the
terminal (`--rm` flag).

The `-v` flag is to map the directory containing your jpeg-xl checkout in your
host (assumed to be at `$HOME/jpeg-xl`) to a directory inside the container at
/jpeg-xl. Since the container is accessing the host folder directly,
changes made on the host will will be seen immediately in the container,
and vice versa.

On OSX, the path must be one of those shared and whitelisted with Docker. $HOME
(which is a subdirectory of /Users/) is known to work with the factory-default
settings of Docker.

On OSX, you may ignore the warning that Docker "cannot find name for group ID".
This warning may also appear on some Linux computers.

On Windows, you can run the following from the jpeg-xl directory obtained from
Gitlab:

```bash
docker run -u root:root -it --rm -v %cd%:/jpeg-xl -w /jpeg-xl \
  gcr.io/jpegxl/jpegxl-builder
```

## Basic building

Inside the Docker container, you can compile everything and run unit tests.
We need to specify `clang-7` because the default `clang` compiler is
not installed on the image.

```bash
CC=clang-7 CXX=clang++-7 ./ci.sh opt
```

This writes binaries to `/jpeg-xl/build/tools` and runs unit tests.
More information on [build modes and testing](building_and_testing.md) is
available.

If a `build` directory already exists and was configured for a different
compiler, `cmake` will complain. This can be avoided by renaming or removing
the existing `build` directory or setting the `BUILD_DIR` environment variable.

## Cross-compiling environments (optional)

We have installed the required cross-compiling tools in the main Docker image
`jpegxl-builder`. This allows compiling for other architectures, such as arm.
Tests will be emulated under `qemu`.

The Docker container has several `qemu-*-static` binaries (such as
`qemu-aarch64-static`) that emulate other architectures on x86_64. These
binaries are automatically used when running foreign architecture programs
in the container only if `binfmt` is installed and configured on the *host*
to use binaries from `/usr/bin` . This is the default location on Ubuntu/Debian.

You need to install both `binfmt-support` and `qemu-user-static` on the host,
since `binfmt-support` configures only `binfmt` signatures of architectures
that are installed.  If these are configured elsewhere on other distributions,
you can symlink them to `/usr/bin/qemu-*-static` inside the Docker container.

To install binfmt support in your Ubuntu host run *outside* the container:

```bash
sudo apt install binfmt-support qemu-user-static
```

Then to cross-compile and run unit tests execute the following commands:

```bash
export BUILD_TARGET=aarch64-linux-gnu CC=clang-7 CXX=clang++-7
./ci.sh release
```

The `BUILD_TARGET=aarch64-linux-gnu` environment variable tells the `ci.sh`
script to cross-compile for that target. This also changes the default
`BUILD_DIR` to `build-aarch64` since you never want to mix them with the `build`
of your host. You can also explicitly set a `BUILD_DIR` environment variable
that will be used instead. The list of supported `BUILD_TARGET` values for this
container is:

*    *the empty string* (for native x86_64 support)
*    aarch64-linux-gnu
*    arm-linux-gnueabihf
*    i686-linux-gnu
*    x86_64-w64-mingw32 (for Windows builds)
