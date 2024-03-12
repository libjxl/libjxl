### Reasoning 

Given the differences in compilers / environment it is not always clear why some
build / test fails in workflows. In this document we gather practices that
would help debugging workflows.

### Debugging workflows on GitHub

To connect to real workflow on GitHub one can use "tmate" plugin. To do that,
add the following snippet in workflow .yml:

```
 - name: Setup tmate session
   # Or other condition that pin-points a single strategy matrix item
   if: failure()
   uses: mxschmitt/action-tmate@a283f9441d2d96eb62436dc46d7014f5d357ac22 # v3.17
   timeout-minutes: 15
```

When the plugin is executed it dumps to log a command to "ssh" to that instance.

NB: since session is wrapped in tmux, scrolling might be very inconvenient.

### Debugging build_test_cross.yml locally

"cross" workflows are executed in container, so those are easy to reproduce
locally. Here is a snippet that reflects how setup / compilation are (currently)
done in the workflow:

```
docker run -it -v`pwd`:/libjxl debian:bookworm bash

cd /libjxl

export ARCH=i386 # arm64 armhf
export MAIN_LIST="amd64,${ARCH}"
export BUILD_DIR=build
export CC=clang-14
export CXX=clang++-14
export BUILD_TARGET=i686-linux-gnu # aarch64-linux-gnu arm-linux-gnueabihf

rm -f /var/lib/man-db/auto-update
apt-get update -y
apt-get install -y ca-certificates debian-ports-archive-keyring python3

dpkg --add-architecture ${ARCH}
python3 ./tools/scripts/transform_sources_list.py "${MAIN_LIST}"
apt update

apt-get install -y \
  clang-14 cmake doxygen g++-aarch64-linux-gnu graphviz libbrotli-dev:${ARCH} \
  libc6-dev-${ARCH}-cross libgdk-pixbuf2.0-dev:${ARCH} libgif-dev:${ARCH} \
  libgtk2.0-dev:${ARCH} libilmbase-dev:${ARCH} libjpeg-dev:${ARCH} \
  libopenexr-dev:${ARCH} libpng-dev:${ARCH} libstdc++-12-dev-${ARCH}-cross \
  libstdc++-12-dev:${ARCH} libwebp-dev:${ARCH} ninja-build pkg-config \
  qemu-user-static unzip xdg-utils xvfb

#apt-get install -y binutils-${BUILD_TARGET} gcc-${BUILD_TARGET}
#apt-get install -y \
#  libgoogle-perftools-dev:${ARCH} libgoogle-perftools4:${ARCH} \
#  libtcmalloc-minimal4:${ARCH} libunwind-dev:${ARCH}
#export CMAKE_FLAGS="-march=armv8-a+sve"

SKIP_TEST=1 ./ci.sh release \
  -DJPEGXL_FORCE_SYSTEM_BROTLI=ON \
  -DJPEGXL_ENABLE_JNI=OFF
#  -DCMAKE_CROSSCOMPILING_EMULATOR=/usr/bin/qemu-aarch64-static
#  -DJPEGXL_ENABLE_OPENEXR=off
#  -DJPEGXL_ENABLE_SIZELESS_VECTORS=on
#  -DCMAKE_CXX_FLAGS=-DJXL_HIGH_PRECISION=0
```
