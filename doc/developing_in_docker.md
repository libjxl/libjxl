# Developing in docker

Docker allows to run software in a packaged container, isolated from your host
system. This is useful to run the code in a standard environment instead of
dealing with problems caused by different building environments during
development, as well as to simplify external dependencies by including them
in the automated setup of this standard environment.

## Set up the container

Read over the [docker install instructions](https://docs.docker.com/install/) to
get docker installed on your computer. As of 2020-01, this requires creating a
free account with Docker.

The image used by our builders is an ubuntu-bionic image with all the required
dependencies and build tools installed. You can pull this image from
`gcr.io/jpegxl/jpegxl-builder` using the following command:

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

This creates and runs a container that will be deleted after you exit from this
terminal (`--rm` flag).

The `-v` flag is to map the directory containing your jpeg-xl checkout in your
host (assumed to be at `$HOME/jpeg-xl`) to a directory inside the container at
/jpeg-xl. This means that whenever you make changes to the code from your host
using your favorite editor they are also changed in the container, since the
directory is simply mounted into the container.

On OSX, the path must be one of those whitelisted/shared with Docker. $HOME
(which is a subdirectory of /Users/) is known to work with the factory-default
settings of Docker.

On OSX, "cannot find name for group ID" can be ignored.

On Windows, you can run the following from the jpeg-xl directory obtained from
Gitlab:

```bash
docker run -u root:root -it --rm -v %cd%:/jpeg-xl -w /jpeg-xl \
  gcr.io/jpegxl/jpegxl-builder
```

## Basic building

Inside the Docker container, you can compile everything and run unit tests
by running the following:

```bash
CC=clang-7 CXX=clang++-7 ./ci.sh opt
```

This writes binaries to `/jpeg-xl/build/tools` and runs unit tests.
More information on [build modes and testing](building_and_testing.md) is
available.

If there already was a build directory on the host this can give conflicts,
remove it first with `rm -rf build`.

Note that the default "clang" compiler is not installed on the image, hence we
specify clang-7. If a build/ directory already exists and was configured for
a different compiler, cmake will complain. This can be avoided by renaming any
existing build/ directory or setting the `BUILD_DIR` environment variable.

## Cross-compiling environments (optional)

We have installed the required cross-compiling tools in the main docker image
`jpegxl-builder`. This allows to compile for other architectures such as arm
and run the tests emulated under qemu.

The docker container already has several `qemu-*-static` binaries (such as
`qemu-aarch64-static`) that emulate other architectures on x86_64. These qemu
binaries are automatically used when running a foreign architecture program in
the container only if you have `binfmt` installed and configured to use the
binaries from `/usr/bin/qemu-*-static` in the *host*. This is the default
location in Ubuntu/Debian, however you need to install on the host both
`binfmt-support` and `qemu-user-static` even if the qemu-user-static binaries
are not used from the host, since binfmt-support in Debian/Ubuntu only
configures the `binfmt` signatures of the architectures you have a
qemu-user-static binary installed on the host. If you have these configured
somewhere else in other distribution, symlink that location to the
`/usr/bin/qemu-*-static` inside the docker before running. To install binfmt
support in your Ubuntu host run *outside* the container:

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
of your host. You can also explicitly set a `BUILD_DIR` environment variable and
that will be used instead. The list of supported BUILD_TARGET values for this
container is:

*    *the empty string* (for native x86_64 support)
*    aarch64-linux-gnu
*    arm-linux-gnueabihf
*    i686-linux-gnu
*    x86_64-w64-mingw32 (for Windows builds)
