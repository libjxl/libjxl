# Building and Testing

## Basic building

If you only want to build the project but not modify it you can run:

```bash
./ci.sh release
```

Running `./ci.sh` with no parameters shows a list of available commands. For
example, you can run `opt` for optimized developer builds with symbols or
`debug` for debug builds which do not have NDEBUG defined and therefore include
more runtime debug information.

## Sending patches

Before sending a patch, make sure your patch conforms to the
[project guidelines](guidelines.md) and test it following these steps.

You will need to install a few other dependencies for verifying these patches,
which are used by the `./ci.sh` helper command.

```bash
sudo apt install clang-format-6.0 clang-tidy-6.0 curl parallel
```

### Build and unittest checks

Your patch must build in the "release" and "debug" configurations at least, and
pass all the tests. The "opt" configuration will be tested in the Merge Request
pipeline which is equivalent to a "release" build with debug information.

```bash
./ci.sh opt
```

All of the tests must pass. If you added new functions, you should also add
tests to these functions in the same commit.

### Cross-compiling

To compile the code for an architecture different than the one you are running
you can pass a
[toolchain file](https://cmake.org/cmake/help/latest/manual/cmake-toolchains.7.html)
to cmake if you have one for your target, or you can use the `BUILD_TARGET`
environment variable in `./ci.sh`. This assumes that you already have a
cross-compiling environment set up and the library dependencies are already
installed for the target architecture as well. Alternatively, you can use one of
the [jpegxl docker containers](developing_in_docker.md) already configured to
cross-compile and run for other architectures.

For example, to compile for the `aarch64-linux-gnu` target triplet you can run:

```bash
BUILD_TARGET=aarch64-linux-gnu ./ci.sh release
```

### Format checks (lint)

```bash
./ci.sh lint
```

Linter checks will verify that the format of your patch conforms to the project
style (see [guidelines](guidelines.md)). For this, we run clang-format only on
the lines that were changed by your commits.

If your local git branch is tracking `origin/master` and you landed a few
commits in your branch, running this lint command will check all the changes
made from the common ancestor with `origin/master` to the latest changes,
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

## Testing

All of the `./ci.sh` workflows like `release`, `opt`, etc will run the test
after building. It is possible to manually run the tests, for example:

```bash
cd build/
ctest -j32 --output-on-failure
```

It is also possible for faster iteration to run a specific test binary directly.
googletest binaries support a glob filter to match the name of the test, for
example `TransposeTest*` will execute all the test names that start with
`TransposeTest`.

```bash
build/dct_test --gtest_filter=TransposeTest.Transpose8

```

If you want to run multiple tests from different binaries, you can pass a
regular expression filter to ctest. Note that this is a regular expression
instead of a glob match like in the `gtest_filter`.

```bash
cd build/
ctest -j32 --output-on-failure -R '.*Transpose.*'
```

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
sudo apt install llvm # or llvm-6.0, llvm-7, etc for a specific version.
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

### Benchmark

To run the benchmark:

```shell
build/tools/benchmark_xl --input "/path/*.png" --codec jxl:d1,jxl:d2,jxl:d4
```
