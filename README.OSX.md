## Disclaimer

OSX builds have "best effort" support, i.e. build might not work at all, some
tests may fail and some sub-projects are excluded from build.

This manual outlines OSX specific setup. For general building and testing
instructions see "[README](README.md)" and
"[Building and Testing changes](doc/building_and_testing.md)".

## Dependencies

[Homebrew](https://brew.sh/) is a popular package manager that can be used for
installing dependencies.

Make sure that `brew doctor` does not report serious problems and up-to-date
version of XCode is installed.

Installing (actually, building) `clang` might take a couple hours.

```bash
brew install llvm
```

```bash
brew install clang-format coreutils cmake giflib libjpeg ninja parallel
```

If `git-clang-format` command is not accessible, run

```bash
brew link --overwrite clang-format
```

Before building the project check that `which clang` is
`/usr/local/opt/llvm/bin/clang`, not the one provided by XCode. If not, update
`PATH` environment variable.

Also, `export CMAKE_PREFIX_PATH=/usr/local/opt/giflib` might be necessary for
correct include paths resolving.
