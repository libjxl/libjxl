# JPEG XL reference implementation

<img src="doc/jxl.svg" width="100" align="right" alt="JXL logo">

This repository contains a reference implementation of JPEG XL (encoder and
decoder), called `libjxl`.

JPEG XL is in the final stages of standardization and its codestream format is
frozen.

The libraries API, command line options and tools in this repository are subject
to change, however files encoded with `cjxl` conform to the JPEG XL format
specification and can be decoded with current and future `djxl` decoders or
`libjxl` decoding library.

## Quick start guide

For more details and other workflows see the "Advanced guide" below.

### Checking out the code

```bash
git clone https://github.com/libjxl/libjxl.git --recursive
```

This repository uses git submodules to handle some third party dependencies
under `third_party/`, that's why is important to pass `--recursive`. If you
didn't check out with `--recursive`, or any submodule has changed, run:
`git submodule update --init --recursive`.

Important: If you downloaded a zip file or tarball from the web interface you
won't get the needed submodules and the code will not compile. You can download
these external dependencies from source running `./deps.sh`. The git workflow
described above is recommended instead.

### Installing dependencies

Required dependencies for compiling the code, in a Debian/Ubuntu based
distribution run:

```bash
sudo apt install cmake pkg-config libbrotli-dev
```

Optional dependencies for supporting other formats in the `cjxl`/`djxl` tools,
in a Debian/Ubuntu based distribution run:

```bash
sudo apt install libgif-dev libjpeg-dev libopenexr-dev libpng-dev libwebp-dev
```

We recommend using a recent Clang compiler (version 7 or newer), for that
install clang and set `CC` and `CXX` variables. For example, with clang-7:

```bash
sudo apt install clang-7
export CC=clang-7 CXX=clang++-7
```

### Building

```bash
cd libjxl
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF ..
cmake --build . -- -j$(nproc)
```

The encoder/decoder tools will be available in the `build/tools` directory.

### <a name="installing"></a> Installing

```bash
sudo cmake --install .
```

### Basic encoder/decoder

To encode a source image to JPEG XL with default settings:

```bash
build/tools/cjxl input.png output.jxl
```

For more settings run `build/tools/cjxl --help` or for a full list of options
run `build/tools/cjxl -v -v --help`.

To decode a JPEG XL file run:

```bash
build/tools/djxl input.jxl output.png
```

When possible `cjxl`/`djxl` are able to read/write the following
image formats: .exr, .gif, .jpeg/.jpg, .pfm, .pgm/.ppm, .pgx, .png.

### Benchmarking

For speed benchmarks on single images in single or multi-threaded decoding
`djxl` can print decoding speed information. See `djxl --help` for details
on the decoding options and note that the output image is optional for
benchmarking purposes.

For more comprehensive benchmarking options, see the
[benchmarking guide](doc/benchmarking.md).

## Advanced guide

### Building with Docker

We build a common environment based on Debian/Ubuntu using Docker. Other
systems may have different combinations of versions and dependencies that
have not been tested and may not work. For those cases we recommend using the
Docker environment as explained in the
[step by step guide](doc/developing_in_docker.md).

### Building JPEG XL for developers

For experienced developers, we also provide build instructions for an [up to
date Debian-based Linux](doc/developing_in_debian.md) and [64-bit
Windows](doc/developing_in_windows.md). If you encounter any difficulties,
please use Docker instead.

## License

This software is available under a 3-clause BSD license which can be found in
the [LICENSE](LICENSE) file, with an "Additional IP Rights Grant" as outlined in
the [PATENTS](PATENTS) file.

Please note that the PATENTS file only mentions Google since Google is the legal
entity receiving the Contributor License Agreements (CLA) from all contributors
to the JPEG XL Project, including the initial main contributors to the JPEG XL
format: Cloudinary and Google.

## Additional documentation

### Codec description

*   [Introductory paper](https://www.spiedigitallibrary.org/proceedings/Download?fullDOI=10.1117%2F12.2529237) (open-access)
*   [XL Overview](doc/xl_overview.md) - a brief introduction to the source code modules
*   [JPEG XL white paper](http://ds.jpeg.org/whitepapers/jpeg-xl-whitepaper.pdf)
*   [JPEG XL official website](https://jpeg.org/jpegxl)
*   [JPEG XL community website](https://jpegxl.info)

### Development process
*   [Docker setup - **start here**](doc/developing_in_docker.md)
*   [Building on Debian](doc/developing_in_debian.md) - for experts only
*   [Building on Windows](doc/developing_in_windows.md) - for experts only
*   [More information on testing/build options](doc/building_and_testing.md)
*   [Git guide for JPEG XL](doc/developing_in_github.md) - for developers only
*   [Building Web Assembly artifacts](doc/building_wasm.md)

### Contact

If you encounter a bug or other issue with the software, please open an Issue here.

There is a [subreddit about JPEG XL](https://www.reddit.com/r/jpegxl/), and
informal chatting with developers and early adopters of `libjxl` can be done on the
[JPEG XL Discord server](https://discord.gg/DqkQgDRTFu).
