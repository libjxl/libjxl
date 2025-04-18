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
| e1 | fast-lossless, fixed YCoCg RCT, fixed ClampedGradient predictor, simple palette detection, no MA tree (one context for everything), Huffman, simple rle-only lz77 | only 8x8, basically XYB jpeg with ANS |
| e2 | global channel palette, fixed MA tree (context based on Gradient-error), ANS, otherwise same as e1 | same as e1 |
| e3 | same as e2 but fixed Weighted predictor and fixed MA tree with context based on WP-error | e2 + better ANS |
| e4 | try both ClampedGradient and Weighted predictor, learned MA tree, global palette | coefficient reordering |
| e5 | e4 + patches, local palette / local channel palette, different local RCTs | e4 + simple variable blocks heuristics, adaptive quantization, gabor-like transform, chroma from luma |
| e6 | e5 + more RCTs and MA tree properties | e5 + error diffusion, full variable blocks heuristics |
| e7 | e6 + more RCTs and MA tree properties | e6 + patches (including dots) |
| e8 | e7 + more RCTs, MA tree properties, and Weighted predictor parameters | e7 + Butteraugli iterations for adaptive quantization |
| e9 | e8 + more RCTs, MA tree properties, and Weighted predictor parameters | e8 + more Butteraugli iterations |
| e10 | e9 + global MA tree, try all predictors, and disables chunked encoding | e9 + more thorough adaptive quantization, disables chunked encoding and uses iterative downsampling |
| e11 | e10 + previous-channel MA tree properties, different group dimensions, and try multiple e10 configurations | N/A |

For the entropy coding (context clustering, lz77 search, hybriduint configuration): slower/more exhaustive search as effort goes up.

<u>Chunked encoding is also disabled under these circumstances:</u>
* When the image is smaller than 2048x2048.
* Lossless Jpeg transcoding.
* VarDCT at distances ≥10.
* Effort 7 VarDCT at distances ≥3.0.
* Efforts 8 & 9 VarDCT at distances >0.5.
* Lossy Modular.
* When using any of these flags:
  * `--patches=1`
  * `--progressive_dc >0`
  * `-p`
  * `-d 0` and `-R 1`
  * `--noise=1`
  * `--resampling >1`
  * `--disable_perceptual_optimizations`
