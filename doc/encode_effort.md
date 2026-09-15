# Encode effort settings

Various trade-offs between encode speed and compression performance can be selected in libjxl. In `cjxl`, this is done via the `--effort` (`-e`) option.
Higher effort means slower encoding; generally the higher the effort, the more coding tools are used, computationally more expensive heuristics are used,
and more exhaustive search is performed. 
Generally, efforts range between `1` and `10`, but there is also `e11` if you pass the flag `--allow_expert_options` (in combination with "lossless", i.e. `-d 0`). It is considered an expert option because it can be extremely slow.


For lossy compression, higher effort results in better visual quality at a given filesize, and also better
encoder consistency, i.e. less image-dependent variation in the actual visual quality that is achieved. This means that for lossy compression,
higher effort does not necessarily mean smaller filesizes for every image — some images may be somewhat lower quality than desired when using
lower effort heuristics, and to improve consistency, higher effort heuristics may decide to use more bytes for them.

For lossless compression, higher effort should result in smaller filesizes, although this is not guaranteed;
in particular, e2 can be better than e3 for non-photographic images, and e3 can be better than e4 for photographic images.

The following table describes what the various effort settings do:

|Effort | Modular (lossless) | VarDCT (lossy) |
|-------|--------------------|----------------|
| e1 | fast-lossless, fixed YCoCg RCT, fixed ClampedGradient predictor, simple palette detection, no MA tree (one context for everything), Huffman, simple rle-only lz77 | N/A (gets bumped to e2) |
| e2 | global channel palette, fixed MA tree (context based on Gradient-error), ANS, otherwise same as e1 | only 8x8, basically XYB jpeg with ANS |
| e3 | same as e2 but fixed Weighted predictor and fixed MA tree with context based on WP-error | e2 + better ANS |
| e4 | try both ClampedGradient and Weighted predictor, learned MA tree, global palette | coefficient reordering |
| e5 | e4 + patches, local palette / local channel palette, different local RCTs | e4 + simple variable blocks heuristics, adaptive quantization, gabor-like transform, chroma from luma |
| e6 | e5 + more RCTs and MA tree properties | e5 + full variable blocks heuristics |
| e7 | e6 + more RCTs and MA tree properties | e6 + error diffusion, patches (see chunked encoding) + better chroma from luma |
| e8 | e7 + more RCTs, MA tree properties, and Weighted predictor parameters | e7 + better adaptive quantization with butteraugli iterations |
| e9 | e8 + more RCTs, MA tree properties, and Weighted predictor parameters | e8 + more butteraugli iterations |
| e10 | e9 + global MA tree, try all predictors, and disables chunked encoding | e9 + exhaustive block heuristics, iterative downsampling (only used when `--resampling=2`), and disables chunked encoding |
| e11 | e10 + previous-channel MA tree properties, different group dimensions, and try multiple e10 configurations | N/A (bumped down to e10) |

For the entropy coding (context clustering, lz77 search, hybriduint configuration): slower/more exhaustive search as effort goes up.

<u>Chunked encoding (streaming) is also disabled under these circumstances:</u>
* When using default buffering (`--buffering=-1`):
  * Effort 7 VarDCT at distances ≥3.0. (patches get enabled)
  * Efforts 8 & 9 VarDCT at distances >0.5. (hidden check in code)
* When using `--buffering=0` (buffer entire image).
* When using `--buffering=1` and the image is 2048x2048 or smaller.
* When using any buffering mode, and the image has 8 or fewer total groups. The default group size is 256x256
  (meaning this applies to images smaller than ~768x768), but this threshold dynamically scales if the group
  size is changed via the `-g` flag in Modular mode.
* Lossless Jpeg transcoding.
* VarDCT at distances ≥10.
* Lossy Modular.
* When using any of these flags:
  * `--patches=1`
  * `--progressive_dc >0`
  * `--progressive_ac`
  * `--qprogressive_ac`
  * `-p`
  * `-d 0` and `-R 1`
  * `--noise=1`
  * `--resampling >1`
  * `--disable_perceptual_optimizations`

> [!NOTE]
> In version 0.12, the new flags `--buffering` and `--output_mode` were introduced to explicitly control streaming behavior. `--buffering` controls input buffering (`0` for full buffering, `1` to stream large images, `2` to stream with a lower threshold). `--output_mode` controls output codestream buffering and streaming order. By default, output is buffered (`--output_mode 0`) to allow basic progressive loading.
