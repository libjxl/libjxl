# High bit depth JPEG encoder
`cjpeg_hdr` is an (experimental) JPEG encoder that can preserve a higher bit
depth than a traditional JPEG encoder. In particular, it may be used to produce
HDR JPEGs that do not show obvious signs of banding.

Note that at this point in time `cjpeg_hdr` does not attempt to actually
*compress* the image - it behaves in the same way as a "quality 100" JPEG
encoder would normally do, i.e. no quantization, to achieve the maximum
possible visual quality.  Moreover, no Huffman optimization is performed.

## Generating HBD JPEGs
Note: this and the following sections assume that `libjxl` has been built in
the `build/` directory, either by using CMake or by running `./ci.sh opt`.

It should be sufficient to run `build/tools/cjpeg_hdr input_image output.jpg`.
Various input formats are supported, including NetBPM and (8- or 16-bit) PNG.

If the PNG image includes a colour profile, it will be copied in the resulting
JPEG image. If this colour profile approximates the PQ or HLG transfer curves,
some applications will consider the resulting image to be HDR.

To attach a PQ profile to an image without a colour profile (or with a
different colour profile), the following command can be used:

```
 build/tools/decode_and_encode input RGB_D65_202_Rel_PeQ output_with_pq.png 16
```

Similarly, to attach an HLG profile, the following command can be used

```
 build/tools/decode_and_encode input RGB_D65_202_Rel_HLG output_with_pq.png 16
```

## Decoding HBD JPEGs
HBD JPEGs are fully retrocompatible with libjpeg, and any JPEG viewer ought to
be able to visualize them. Nonetheless, to achieve the best visual quality, a
high bit depth decoder should be used.

Such a decoder does not exist today. As a workaround, it is possible to do a
lossless conversion to JPEG XL and then view the resulting image:

```
  build/tools/cjxl --jpeg_transcode_disable_cfl hbd.jpeg hbd.jxl
```

The resulting JPEG XL file can be visualized, for example, in a browser,
assuming that the corresponding flag is enabled in the settings.

In particular, if the HBD JPEG has a PQ or HLG profile attached and the current
display is an HDR display, Chrome ought to visualize the image as HDR content.

It is also possible to convert the JPEG XL file back to a 16-bit PNG:

```
  build/tools/djxl hbd.jxl --bits_per_sample=16 output.png
```

Note however that as of today (2 Nov 2021) Chrome does not interpret such a PNG
as an HDR image, even if a PQ or HLG profile is attached. Thus, to display the
HDR image correctly it is recommended to either display the JPEG XL image
directly or to convert the PNG to a format that Chrome interprets as HDR, such
as AVIF. This can be done with the following command for a PQ image:

```
  avifenc -l -y 444 --depth 10 --cicp 9/16/9 image.png output.avif
```

and the following one for an HLG image:

```
  avifenc -l -y 444 --depth 10 --cicp 9/18/9 image.png output.avif
```
