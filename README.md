# JPEG XL reference implementation

This repository contains a reference implementation of JPEG XL (encoder and
decoder). As [previously announced](https://jpeg.org/items/20190803_press.html),
it is available under a royalty-free and open source license (Apache 2).

**NOTE**

JPEG XL is in the final stages of standardization, but minor changes to the
codestream are still likely. WARNING: until further notice, do not depend on
future decoders being able to decode the output of a current encoder.

## Checking out the code

This repository uses git submodules to handle some third party dependencies
under `third_party/`. To also check out these dependencies, clone the
repository with `--recursive`:

```bash
git clone https://gitlab.com/wg1/jpeg-xl.git  --recursive
```

If you didn't check out with `--recursive`, or any of the third party
dependencies have changed, run the following command:

```bash
git submodule update --init --recursive
```

## Building

To avoid system incompatibilities, we **strongly recommend** using Docker to
build and test the software, as explained in the
[step by step guide](doc/developing_in_docker.md).

For experienced developers, we also provide build instructions for an [up to
date Debian-based Linux](doc/developing_in_debian.md) and [64-bit
Windows](doc/developing_in_windows.md). If you encounter any difficulties,
please use Docker instead.

The resulting binaries are in the `build` directory and its subdirectories.

## CPU requirements

JPEG XL no longer requires a particular CPU. The software chooses and uses the
best available instruction set for the current CPU.

## Basic encoder/decoder

`build/tools/cjpegxl input.png output.jxl` encodes to JPEG XL with default
settings. For a list of common options, run ``build/tools/cjpegxl`;
`build/tools/cjpegxl -v` shows more.

Here and in general, the JPEG XL tools are able to read/write the following
image formats: .exr, .gif, .jpeg/.jpg, .pfm, .pgm/.ppm, .pgx, .png.

`build/tools/djpegxl output.jxl output.png` decodes JPEG XL to other formats.

## Benchmarking

We recommend `build/tools/benchmark_xl` as a convenient method for reading
images or image sequences, encoding them using various codecs (jpeg jxl png
webp), decoding the result, and computing objective quality metrics. An example
invocation is:

```bash
build/tools/benchmark_xl --input "/path/*.png" --codec jxl:wombat:d1,jxl:cheetah:d2
```

Multiple comma-separated codecs are allowed. The characters after : are
parameters for the codec, separated by colons, in this case specifying maximum
target psychovisual distances of 1 and 2 (higher implies lower quality) and
the encoder effort (see below). Other common parameters are `r0.5` (target
bitrate 0.5 bits per pixel) and `q92` (quality 92, on a scale of 0-100, where
higher is better). The `jxl` codec supports the following additional parameters:

Speed: `falcon`, `cheetah`, `hare`, `wombat`, `squirrel`, `kitten`, `tortoise`
control the encoder effort in ascending order.

*   `falcon` disables all of the following tools.
*   `cheetah` enables coefficient reordering, context clustering, and heuristics
    for selecting DCT sizes and quantization steps.
*   `hare` enables Gaborish filtering, chroma from luma, and an initial estimate
    of quantization steps.
*   `wombat` enables error diffusion quantization and full DCT size selection
    heuristics.
*   `squirrel` (default) enables dots, patches, and spline detection, and full
    context clustering.
*   `kitten` optimizes the adaptive quantization for a psychovisual metric.
*   `tortoise` enables a more thorough adaptive quantization search.

Mode: JPEG XL has several modes for various types of content. The default mode
is suitable for photographic material. One of the following alternatives may be
selected:

*   `mg` activates modular mode (useful for non-photographic images such as
    screen content).
*   `bg` activates lossless JPEG reconstruction with parallel decoding (the
    input must have been a JPEG file).
*   `b:file` activates lossless JPEG reconstruction with more compact encodings,
    but without the option of parallel decoding.

Other arguments to benchmark_xl include:

*   `save_compressed`: save codestreams to `output_dir`.
*   `save_decompressed`: save decompressed outputs to `output_dir`.
*   `output_extension`: selects the format used to output decoded images.
*   `num_threads`: number of codec instances that will independently
    encode/decode images, or 0.
*   `inner_threads`: how many threads each instance should use for parallel
    encoding/decoding, or 0.
*   `encode_reps`/`decode_reps`: how many times to repeat encoding/decoding
    each image, for more consistent measurements (we recommend 10).

The benchmark output begins with a header:

```
Compr              Input    Compr            Compr       Compr  Decomp  Butteraugli
Method            Pixels     Size              BPP   #    MP/s    MP/s     Distance    Error p norm           BPP*pnorm   Errors
```

`ComprMethod` lists each each comma-separated codec. `InputPixels` is the number
of pixels in the input image. `ComprSize` is the codestream size in bytes and
`ComprBPP` the bitrate. `Compr MP/s` and `Decomp MP/s` are the
compress/decompress throughput, in units of Megapixels/second.
`Butteraugli Distance` indicates the maximum psychovisual error in the decoded
image (larger is worse). `Error p norm` is a similar summary of the psychovisual
error, but closer to an average, giving less weight to small low-quality
regions. `BPP*pnorm` is the product of `ComprBPP` and `Error p norm`, which is a
figure of merit for the codec (lower is better). `Errors` is nonzero if errors
occurred while loading or encoding/decoding the image.

## Additional documentation

### Codec description

*   [Introductory paper](https://www.spiedigitallibrary.org/proceedings/Download?fullDOI=10.1117%2F12.2529237) (open-access)
*   [XL Overview](doc/xl_overview.md) - a brief introduction to the
    source code modules
*   [JPEG XL committee draft](https://arxiv.org/abs/1908.03565)
*   JPEG XL white paper with overview of applications and coding tools:
    WG1 output document number wg1n86059

### Development process
*   [Docker setup - **start here**](doc/developing_in_docker.md)
*   [Building on Debian](doc/developing_in_debian.md) - for experts only
*   [Building on Windows](doc/developing_in_windows.md) - for experts only
*   [More information on testing/build options](doc/building_and_testing.md)
*   [Git guide for JPEG XL](doc/developing_in_gitlab.md) - for developers only
*   [Building Web Assembly artifacts](doc/building_wasm.md)
