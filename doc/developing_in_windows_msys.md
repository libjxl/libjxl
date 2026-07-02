# Developing for Windows with MSYS2

[MSYS2](https://www.msys2.org/) ("minimal system 2") is a software distribution and a development platform based on MinGW and Cygwin.  It provides a  Unix-like environment to build code on Windows.  These instructions were written with a 64-bit instance of Windows 10 running on a VM.  They may also work on native instances of Windows and other versions of Windows.

## Build Environments

MSYS2 provides multiple development [environments](https://www.msys2.org/docs/environments/).  By convention, they are referred to in uppercase.  They target slightly different platforms, runtime libraries, and compiler toolchains.  For interoperability with Visual Studio projects, use the UCRT64 environment.

Since all of the build environments are built on top of the MSYS environment, **all updates and package installation must be done from within the MSYS environment**.  After making any package changes, `exit` all MSYS2 terminals and restart the desired build-environment.  This reminder is repeated multiple times throughout this guide.

* **MINGW32 (Deprecated):**  To compile for 32-bit Windows (on 64-bit Windows), use packages from the `mingw32` group.  Package names are prefixed with `mingw-w64-i686`.  *Note: 32-bit MSYS2 is deprecated and some newer packages (like `clang`) are no longer available.*

* **MINGW64 (Phasing Out):**  This uses the older MSVCRT runtime, which is widely available across Windows systems. Package names are prefixed with `mingw-w64-x86_64`. *Note: As of March 2026, the MINGW64 environment is being phased out in favor of UCRT environments.*

* **UCRT64 (Recommended):**  The Universal C Runtime (UCRT) is used by recent versions of Microsoft Visual Studio. It is the modern standard on Windows 10/11. Package names are prefixed with `mingw-w64-ucrt-x86_64`. MSYS2 strongly recommends using UCRT64 over MINGW64.

* **CLANG64 (Recommended):**  Packages are prefixed with `mingw-w64-clang-x86_64`. MSYS2 recommends CLANG64 as a modern alternative alongside UCRT64.

## Install and Upgrade MSYS2

Download MSYS2 from the homepage.  Install at a location without any spaces on a drive with ample free space.  After installing the packages used in this guide, MSYS2 used about 15GB of space.

Toward the end of installation, select the option to run MSYS2 now.  A command-line window will open.  Run the following command, and answer the prompts to update the repository and close the terminal.

```bash
pacman -Syu
```

Now restart the MSYS environment and run the following command to complete updates:

```bash
pacman -Su
```

## Package Management

Packages are organized in groups, which share the build environment name, but in lower case.  Then they have name prefixes that indicate which group they belong to.  Consider this package search: `pacman -Ss cmake`

```
mingw64/mingw-w64-x86_64-cmake
ucrt64/mingw-w64-ucrt-x86_64-cmake
clang64/mingw-w64-clang-x86_64-cmake
msys/cmake
```

We can see the organization `group/prefix-name`.  When installing packages, the group name is optional.

```bash
pacman -S mingw-w64-x86_64-cmake
```
 
For tools that need to be aware of the compiler to function, install the package that corresponds with the specific build-environment you plan to use.  For `cmake`, install the `mingw64` version.  The generic `msys/cmake` will not function correctly because it will not find the compiler.  For other tools, the generic `msys` version is adequate, like `msys/git`.

To remove packages, use:

```bash
pacman -Rsc [package-name]
```

## Worst-Case Scenario...

If packages management is done within a build environment other than MSYS, the environment structure will be disrupted and compilation will likely fail.  If this happens, it may be necessary to reinstall MSYS2.

1. Rename the `msys64` folder to `msys64.bak`.

2. Use the installer to reinstall MSYS2 to `msys64`.

3. Copy packages from `msys64.bak/var/cache/pacman/pkg/` to the new installation to save download time and bandwidth.

4. Use `pacman` from within the MSYS environment to install and update packages.

5. After successfully building a project, it is safe to delete `msys64.bak`

## The UCRT64 Environment

Next set up the UCRT64 environment.  The following commands should be run within the MSYS environment.  `pacman -S` is used to install packages.  The `--needed` argument prevents packages from being reinstalled.

> [!NOTE]
> If you are unsure what dependencies `libjxl` requires in general, refer to [BUILDING.md](../BUILDING.md) as a starting point. The command below installs the MSYS2 equivalents for those packages.

```bash
pacman -S --needed base-devel mingw-w64-ucrt-x86_64-toolchain
pacman -S git mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja \
    mingw-w64-ucrt-x86_64-gtest mingw-w64-ucrt-x86_64-giflib \
    mingw-w64-ucrt-x86_64-libpng mingw-w64-ucrt-x86_64-libjpeg-turbo 
```

## Build `libjxl`

Download the source from the libjxl [releases](https://github.com/libjxl/libjxl/releases) page.  Alternatively, you may obtain the latest development version with `git`.  Run `./deps.sh` to ensure additional third-party dependencies are downloaded.

Start the UCRT64 environment, create a build directory within the source directory, and configure with `cmake`.

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release \
   -DBUILD_TESTING=OFF \
   -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_PLUGINS=ON \
   -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_STATIC=ON ..
```

You could optionally add `-DCMAKE_EXE_LINKER_FLAGS=-s` to strip symbols to reduce the size of the binaries. Check the output to see if any dependencies were missed and need to be installed.  Adding `-G Ninja` may be helpful, but on my computer, Ninja was selected by default.  Remember that package changes must be done from the MSYS environment.  Then exit all MSYS2 terminals and restart the build environment.

If all went well, you may now build `libjxl` using either `cmake` or `ninja` directly:

```bash
# Build using cmake
cmake --build . --parallel

# Or, build directly with ninja (if using the Ninja generator)
ninja
```

Do not be alarmed by the compiler warnings.  They are a caused by differences between gcc/g++ and clang.  The build should complete successfully.  Then `cjxl`, `djxl`, `jxlinfo`, and others can be run from within the build environment.  Because we used `-DJPEGXL_STATIC=ON`, these executables are statically linked and can be easily moved to a native Windows environment without resolving `dll` issues.

> [!WARNING]
> If you enable OpenEXR support using `-DJPEGXL_ENABLE_OPENEXR=ON` alongside the OpenEXR package provided by `pacman`, the resulting binaries will **not** compile fully statically. If you require standalone static builds with EXR support, you must compile OpenEXR separately from source. Otherwise, leave it disabled.

## The `clang` Compiler

To use the `clang` compiler, install the packages that correspond with the environment you wish to use.  Remember to make package changes from within the MSYS environment.

```
mingw-w64-x86_64-clang
mingw-w64-x86_64-clang-tools-extra
mingw-w64-x86_64-compiler-rt

mingw-w64-ucrt-x86_64-clang
mingw-w64-ucrt-x86_64-clang-tools-extra
mingw-w64-ucrt-x86_64-compiler-rt
```

After the `clang` compiler is installed, 'libjxl' can be built with the `./ci.sh` script.

> [!IMPORTANT]
> The `./ci.sh` script relies on GNU Parallel for some of its commands (like `tidy` and testing). Ensure you install it from your MSYS terminal by running `pacman -S parallel` before using the script.

```bash
./ci.sh release -DBUILD_TESTING=OFF \
    -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_MANPAGES=OFF \
    -DJPEGXL_STATIC=ON
```

If the script doesn't work, navigate to the build directory to reconfigure and build with `cmake`.

```bash
export CC=clang && export CXX=clang++
cmake -DCMAKE_BUILD_TYPE=Release \
   -DBUILD_TESTING=OFF \
   -DJPEGXL_ENABLE_BENCHMARK=OFF -DJPEGXL_ENABLE_PLUGINS=ON \
   -DJPEGXL_ENABLE_MANPAGES=OFF -DJPEGXL_STATIC=ON ..
```

### Link Time Optimization (LTO)

If you want to build `libjxl` with Link Time Optimization (LTO) using Clang, be aware that it will **fail to link** in the `MINGW64` environment. This is because LLVM currently cannot handle `emutls` during LTO code generation, which is used by GCC/libstdc++ in the MINGW64 environment.

To successfully build with LTO, you **must** use an environment with native TLS support, such as `CLANG64` or `UCRT64`. Keep in mind that `-DJPEGXL_LTO=ON` enables full LTO, which can take a long time to link. Thin LTO is recommended for faster link times by passing `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`.
