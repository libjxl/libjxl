# Benchmarking

For speed benchmarks on single images in single or multi-threaded decoding
`djxl` can print decoding speed information. See `djxl --help` for details
on the decoding options and note that the output image is optional for
benchmarking purposes.

For a more comprehensive comparison of compression density between multiple
options, the tool `benchmark_xl` can be used (see below).

## Benchmarking with benchmark_xl

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
the encoder effort (see below). Another common parameter is `q92` (quality 92, on a scale of 0-100, where
higher is better). Quality is directly mapped to distance (quality 90 equals a distance of 1). The `jxl` codec supports the following additional parameters:

Speeds: `lightning`, `thunder`, `falcon`, `cheetah`, `hare`, `wombat`, `squirrel`,
`kitten`, `tortoise`, `glacier`, and `tectonic_plate` control the encoder effort in ascending order. This also
affects memory usage: using lower effort will typically reduce memory consumption
during encoding.

[Encode_effort.md](https://github.com/libjxl/libjxl/blob/main/doc/encode_effort.md) describes what the various effort settings do.

Mode: JPEG XL has two modes. The default is Var-DCT mode, which is suitable for
lossy compression. The other mode is Modular mode, which is suitable for lossless
compression. Modular mode can also do lossy compression (e.g. `jxl:m:q50`).

*   `m` activates modular mode.

Other arguments to benchmark_xl include:

*   `--save_compressed`: save codestreams to `output_dir`.
*   `--save_decompressed`: save decompressed outputs to `output_dir`.
*   `--output_extension`: selects the format used to output decoded images.
*   `--num_threads`: number of codec instances that will independently
    encode/decode images, or 0.
*   `--inner_threads`: how many threads each instance should use for parallel
    encoding/decoding, or 0.
*   `--encode_reps`/`--decode_reps`: how many times to repeat encoding/decoding
    each image, for more consistent measurements (we recommend 10).

The benchmark output begins with a header:

```
Encoding    kPixels   Bytes  BPP    E MP/s    D MP/s    Max norm    SSIMULACRA2 PSNR    pnorm   BPP*pnorm   QABPP   Bugs
```

`Encoding` lists each each comma-separated codec. `kPixels` is the number
of pixels in the input image. `Bytes` is the codestream size in bytes and
`BPP` stands for Bits Per Pixel. `E MP/s` and `D MP/s` are the
compress/decompress throughput, in units of Megapixels/second.
`Max norm` indicates the maximum psychovisual error in the decoded
image (larger is worse). `pnorm` is a similar summary of the psychovisual
error, but closer to an average, giving less weight to small low-quality
regions. `SSIMULACRA2` is a modern psychovisal metric, the range is 100
(lossless) to -∞. `PSNR` is a signal-to-noise ratio meausred in dB.
`BPP*pnorm` is the product of `BPP` and `pnorm`, which is a figure of merit
for the codec (lower is better). `QABPP` is quality adjusted bits per pixel,
which is represented as `BPP`*`Max norm`. `Bugs` is nonzero if errors occurred
while loading or encoding/decoding the image.
