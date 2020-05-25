# Building WASM artifacts

This file describes the building and testing of JPEG XL
[Web Assembly](https://webassembly.org/) bundles and wrappers.

These instructions assume an up-to-date Debian/Ubuntu system.
For other platforms, or if you encounter any difficulties,
please instead use the [Docker container](doc/developing_in_docker.md).

For the sake of simplicity, it is considered, that the following environment
variables are set:

 * `OPT` - path to the directory containing additional software;
   the `emsdk` directory with the Emscripten SDK should reside there;
   in the Docker container (mentioned above) this should be `/opt`
 * `NODE` - path to capcble NodeJS binary; if variable is not set, the
   default binary will be used (returned by `which node`); please make sure,
   that either `NODE` variable, or `which node` points to the capable NodeJS
   binary

## Requirements

[CMake](https://cmake.org/) is used as a build system. To install it, follow
[Debian build instructions](doc/building_in_debian.md).

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

NodeJS is used to run tests. Emscripten SDK is shipped with NodeJS binary.
Unfortunately, currently supplied version does not fully support SIMD features.
`emsdk_env.sh` modifies `PATH` variable so, that bundled NodeJS binary is
prioritized in path resolution. To overcome that, please set `NODE` variable
to point to the capable NodeJS binary
(`$NODE -v` should report at least `v13.11.0`).

In [Docker container](doc/developing_in_docker.md)
CMake and Emscripten SDK are pre-installed.

## Building and testing the project

```bash
# Setup EMSDK and other environment variables. In practice EMSDK is set to be
# $OPT/emsdk.
source $OPT/emsdk/emsdk_env.sh

# Tune CMake for WASM-cross-compilation.
export CMAKE_TOOLCHAIN_FILE="$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake"

# Either build with regular WASM:
emconfigure ./ci.sh release
# or with SIMD WASM:
ENABLE_WASM_SIMD=1 emconfigure ./ci.sh release
```
