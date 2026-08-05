# Building and Testing

This file describes the building and testing facilities provided by the `ci.sh` script. `ci.sh` is a convenient wrapper around `cmake` and `ctest` that simplifies building and testing the project with different configurations. It assumes you already have the build environment set up.

## Basic building

To build the JPEG XL software and run its unit tests, run:

```bash
./ci.sh release
```

Once the build completes successfully, you will find the generated executables (like `cjxl` and `djxl`) in the `build/tools/` directory.

### Cleaning the Build

If you ever need to start fresh (for example, if you change a major dependency or your build gets into a weird state), simply delete the build directory:

```bash
rm -rf build/
```

## Testing

By default, `./ci.sh` build commands like `release` and `opt` will automatically run all unit tests after compiling. If you want to skip testing (e.g., for a faster build), you can set the environment variable `SKIP_TEST=1`:

```bash
SKIP_TEST=1 ./ci.sh release
```

It is possible to manually run all the tests in parallel in all your CPUs with
the command:

```bash
./ci.sh test
```

It is also possible for faster iteration to run a specific test binary directly.
Tests are run with the `ctest` command and arguments passed to `ci.sh test` are
forwarded to `ctest` with the appropriate environment variables set. For
example, to list all the available tests you can run:

```bash
./ci.sh test -N
```

To run a specific test from the list or actually a set of tests matching a
regular expression you can use `ctest`'s parameter `-R`:

```bash
./ci.sh test -R ^MyPrefixTe
```

That command would run any test whose name that starts with `MyPrefixTe`. For
more options run `ctest --help`, for example, you can pass `-j1` if you want
to run only one test at a time instead of our default of multiple tests in
parallel.

## Build Configurations

The `./ci.sh` script provides three primary build configurations:

* **`release`**: Builds an optimized binary and strips out all debugging symbols. Produces the smallest, fastest executable, and is ideal for end-users.
* **`opt`**: Equivalent to CMake's `RelWithDebInfo`. Builds an optimized binary but keeps debugging symbols. Ideal for when you want speed (like a release build) but still need to profile performance or debug crashes.
* **`debug`**: Disables optimizations and includes full debugging information. Best for stepping through code line-by-line during development.

Running `./ci.sh` with no parameters will print out a full list of available commands.

### Cross-compiling

To compile the code for an architecture different than the one you are running
you can pass a
[toolchain file](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html)
to cmake if you have one for your target, or you can use the `BUILD_TARGET`
environment variable in `./ci.sh`. For some targets such the Windows targets
`ci.sh` sets up extra environment variables that are needed for testing.

This assumes that you already have a cross-compiling environment set up and the
library dependencies are already installed for the target architecture as well.

For example, to compile for the `aarch64-linux-gnu` target triplet you can run:

```bash
BUILD_TARGET=aarch64-linux-gnu ./ci.sh release
```

Whenever using a `BUILD_TARGET` or even a custom `BUILD_DIR` these variables
must be set for **every call** to `ci.sh` even calls to `ci.sh test`, for which
we recommend exporting them in your shell session, for example:

```bash
export BUILD_TARGET=x86_64-w64-mingw32 BUILD_DIR=build-foobar
```

### Format checks (lint)

```bash
./ci.sh lint
```

Linter checks will verify that the format of your patch conforms to the project
style. For this, we run clang-format only on the lines that were changed by
your commits.

If your local git branch is tracking `origin/main` and you landed a few
commits in your branch, running this lint command will check all the changes
made from the common ancestor with `origin/main` to the latest changes,
including uncommitted changes. The output of the program will show the patch
that should be applied to fix your commits. You can apply these changes with the
following command from the base directory of the git checkout:

```bash
./ci.sh lint | patch -p1
```

### Programming errors (tidy)

```bash
./ci.sh tidy
```

clang-tidy is a tool to check common programming errors in C++, and other valid
C++ constructions that are discouraged by the style guide or otherwise dangerous
and may constitute a bug.

To run clang-tidy on the files changed by your changes you can run `./ci.sh
tidy`. Note that this will report all the problems encountered in any file that
was modified by one of your commits, not just on the lines that your commits
modified.


### Address Sanitizer (asan)

```bash
./ci.sh asan
```

ASan builds allow to check for invalid address usages, such as use-after-free.
To perform these checks, as well as other undefined behavior checks we only need
to build and run the unittests with ASan enabled which can be easily achieved
with the command above. If you want to have the ASan build files separated from
your regular `build/` directory to quickly switch between asan and regular
builds, you can pass the build directory target as follows:

```bash
BUILD_DIR=build-asan ./ci.sh asan
```

### Memory Sanitizer (msan)

MSan allows to check for invalid memory accesses at runtime, such as using an
uninitialized value which likely means that there is a bug. To run these checks,
a specially compiled version of the project and tests is needed.

For building with MSan, you need to build a version of libc++ with
`-fsanitize=memory` so we can link against it from the MSan build. Also, having
an `llvm-symbolizer` installed is very helpful to obtain stack traces that
include the symbols (functions and line numbers). To install `llvm-symbolizer`
on a Debian-based system run:

```bash
sudo apt install llvm # or llvm-7, etc for a specific version.
```

To install a version of libc++ compiled with `-fsanitize=memory` you can use the
`./ci.sh msan_install` command helper. This will download, compile and install
libc++ and libc++abi in the `${HOME}/.msan` directory to be used later.

After this is set up, you can build the project using the following command:

```bash
./ci.sh msan
```

This command by default uses the `build` directory to store the cmake and object
files. If you want to have a separate build directory configured with msan you
can for example call:

```bash
BUILD_DIR=build-msan ./ci.sh msan
```
