# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5] - 2021-08-02
### Added
 - API: New function to decode the image using a callback outputting a part of a
   row per call.
 - API: 16-bit float output support.
 - API: `JxlDecoderRewind` and `JxlDecoderSkipFrames` functions to skip more
   efficiently to earlier animation frames.
 - API: `JxlDecoderSetPreferredColorProfile` function to choose color profile in
   certain circumstances.
 - encoder: Adding `center_x` and `center_y` flags for more control of the tile
   order.
 - New encoder speeds `lightning` (1) and `thunder` (2).

### Changed
 - Re-licensed the project under a BSD 3-Clause license. See the
   [LICENSE](LICENSE) and [PATENTS](PATENTS) files for details.
 - Full JPEG XL part 1 specification support: Implemented all the spec required
   to decode files to pixels, including cases that are not used by the encoder
   yet. Part 2 of the spec (container format) is final but not fully implemented
   here.
 - Butteraugli metric improvements. Exact numbers are different from previous
   versions.
 - Memory reductions during decoding.
 - Reduce the size of the jxl_dec library by removing dependencies.
 - A few encoding speedups.
 - Clarify the security policy.
 - Significant encoding improvements (~5 %) and less ringing.
 - Butteraugli metric to have some less masking.
 - `cjxl` flag `--speed` is deprecated and replaced by the `--effort` synonym.

### Removed
- API for returning a downsampled DC was deprecated
  (`JxlDecoderDCOutBufferSize` and `JxlDecoderSetDCOutBuffer`) and will be
  removed in the next release.

## [0.3.7] - 2021-03-29
### Changed
 - Fix a rounding issue in 8-bit decoding.

## [0.3.6] - 2021-03-25
### Changed
 - Fix a bug that could result in the generation of invalid codestreams as
   well as failure to decode valid streams.

## [0.3.5] - 2021-03-23
### Added
 - New encode-time options for faster decoding at the cost of quality.
 - Man pages for cjxl and djxl.

### Changed
 - Memory usage improvements.
 - Faster decoding to 8-bit output with the C API.
 - GIMP plugin: avoid the sRGB conversion dialog for sRGB images, do not show
   a console window on Windows.
 - Various bug fixes.

## [0.3.4] - 2021-03-16
### Changed
 - Improved box parsing.
 - Improved metadata handling.
 - Performance and memory usage improvements.

## [0.3.3] - 2021-03-05
### Changed
 - Performance improvements for small images.
 - Add a (flag-protected) non-high-precision mode with better speed.
 - Significantly speed up the PQ EOTF.
 - Allow optional HDR tone mapping in djxl (--tone_map, --display_nits).
 - Change the behavior of djxl -j to make it consistent with cjxl (#153).
 - Improve image quality.
 - Improve EXIF handling.

## [0.3.2] - 2021-02-12
### Changed
 - Fix embedded ICC encoding regression
   [#149](https://gitlab.com/wg1/jpeg-xl/-/issues/149).

## [0.3.1] - 2021-02-10
### Changed
 - New experimental Butteraugli API (`jxl/butteraugli.h`).
 - Encoder improvements to low quality settings.
 - Bug fixes, including fuzzer-found potential security bug fixes.
 - Fixed `-q 100` and `-d 0` not triggering lossless modes.

## [0.3] - 2021-01-29
### Changed
 - Minor change to the Decoder C API to accommodate future work for other ways
   to provide input.
 - Future decoder C API changes will be backwards compatible.
 - Lots of bug fixes since the previous version.

## [0.2] - 2020-12-24
### Added
 - JPEG XL bitstream format is frozen. Files encoded with 0.2 will be supported
   by future versions.

### Changed
 - Files encoded with previous versions are not supported.

## [0.1.1] - 2020-12-01

## [0.1] - 2020-11-14
### Added
 - Initial release of an encoder (`cjxl`) and decoder (`djxl`) that work
   together as well as a benchmark tool for comparison with other codecs
   (`benchmark_xl`).
 - Note: JPEG XL format is in the final stages of standardization, minor changes
   to the codestream format are still possible but we are not expecting any
   changes beyond what is required by bug fixing.
 - API: new decoder API in C, check the `examples/` directory for its example
   usage. The C API is a work in progress and likely to change both in API and
   ABI in future releases.
