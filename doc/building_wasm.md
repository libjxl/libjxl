# Building WASM artifacts

This file describes the building and testing of JPEG XL
[Web Assembly](https://webassembly.org/) bundles and wrappers.

These instructions assume an up-to-date Debian/Ubuntu system.
For other platforms, or if you encounter any difficulties,
please instead use the [Docker container](developing_in_docker.md).

For the sake of simplicity, it is considered, that the following environment
variables are set:

 * `OPT` - path to the directory containing additional software;
   the `emsdk` directory with the Emscripten SDK should reside there;
   in the Docker container (mentioned above) this should be `/opt`

## Requirements

[CMake](https://cmake.org/) is used as a build system. To install it, follow
[Debian build instructions](developing_in_debian.md).

[Emscripten SDK](https://emscripten.org/) is required for building
WebAssembly artifacts. To install it, follow the
[Download and Install](https://emscripten.org/docs/getting_started/downloads.html)
guide:

```bash
cd $OPT

# Get the emsdk repo.
git clone https://github.com/emscripten-core/emsdk.git

# Enter that directory.
cd emsdk

# Download and install the latest SDK tools.
./emsdk install latest

# Make the "latest" SDK "active" for the current user. (writes ~/.emscripten file)
./emsdk activate latest
```

[v8](https://v8.dev/) is a JavaScript engine used for running tests.
v8 has better WASM SIMD support than NodeJS 14.
To install it use [JSVU](https://github.com/GoogleChromeLabs/jsvu):

```bash
# Fix some v8 version know to work well.
export v8_version="8.5.133"

# Install JSVU
npm install jsvu -g

# Trick JSVU to install to specific location instead of user "home".
# Note: "os" flag should match the host OS.
HOME=$OPT jsvu --os=linux64 "v8@${v8_version}"

# Link v8 binary to version-indepentent path.
ln -s "$OPT/.jsvu/v8-${v8_version}" "$OPT/.jsvu/v8"
```

In [Docker container](developing_in_docker.md)
CMake, Emscripten SDK and V8 are pre-installed.

## Building and testing the project

```bash
# If your node version is <16.4.0, you might need to update to a newer version or override
# the node binary with a version which supports SIMD:
echo "NODE_JS='/path/to/node_binary'" >> $EMSDK/.emscripten

# Setup EMSDK and other environment variables. In practice EMSDK is set to be
# $OPT/emsdk.
source $OPT/emsdk/emsdk_env.sh

# Specify JS engine binary
export V8=$OPT/.jsvu/v8

# If building using the jpegxl-builder docker container prefix the following commands with:
# CMAKE_FLAGS=-I/usr/wasm32/include
# ex. CMAKE_FLAGS=-I/usr/wasm32/include BUILD_TARGET=wasm32 emconfigure ./ci.sh release

# Either build with regular WASM:
BUILD_TARGET=wasm32 emconfigure ./ci.sh release
# or with SIMD WASM:
BUILD_TARGET=wasm32 ENABLE_WASM_SIMD=1 emconfigure ./ci.sh release
```
