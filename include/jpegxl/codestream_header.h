/* Copyright (c) the JPEG XL Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/** @file jpegxl/codestream_header.h
 * @brief Definitions of structs and enums for the metadata from the JPEG XL
 * codestream headers (signature, metadata, preview dimensions, ...), excluding
 * color encoding which is in color_encoding.h.
 */

#ifndef JPEGXL_CODESTREAM_HEADER_H_
#define JPEGXL_CODESTREAM_HEADER_H_

#include <stddef.h>
#include <stdint.h>

#include "jpegxl/color_encoding.h"
#include "jpegxl/types.h"

#if defined(__cplusplus) || defined(c_plusplus)
extern "C" {
#endif

/** Signature type of the codestream.
 */
typedef enum {
  /** JPEG XL codestream.
   */
  JPEGXL_SIG_TYPE_JPEGXL = 0,

  /** Transcoded JPEG image signature was found. The decoder will be
   * able to transcode back to the JPEG codestream passed to the encoder.
   */
  JPEGXL_SIG_TYPE_TRANSCODED_JPEG = 1,

  /** JPEG codestream, which would preferably also be decoded using this
   * decoder in case the codestream contains JPEG XL extensions (marker
   * segments).
   */
  JPEGXL_SIG_TYPE_JPEG = 2,
} JpegxlSignatureType;

/** Image orientation metadata.
 * Values 1..8 match the EXIF definitions.
 * The name indicates the operation to perform to transform from the encoded
 * image to the display image.
 */
typedef enum {
  JPEGXL_ORIENT_IDENTITY = 1,
  JPEGXL_ORIENT_FLIP_HORIZONTAL = 2,
  JPEGXL_ORIENT_ROTATE_180 = 3,
  JPEGXL_ORIENT_FLIP_VERTICAL = 4,
  JPEGXL_ORIENT_TRANSPOSE = 5,
  JPEGXL_ORIENT_ROTATE_90_CW = 6,
  JPEGXL_ORIENT_ANTI_TRANSPOSE = 7,
  JPEGXL_ORIENT_ROTATE_90_CCW = 8,
} JpegxlOrientation;

/** Given type of an extra channel.
 */
typedef enum {
  JPEGXL_CHANNEL_ALPHA,
  JPEGXL_CHANNEL_DEPTH,
  JPEGXL_CHANNEL_SPOT_COLOR,
  JPEGXL_CHANNEL_SELECTION_MASK,
  JPEGXL_CHANNEL_BLACK,
  JPEGXL_CHANNEL_CFA,
  JPEGXL_CHANNEL_THERMAL,
  JPEGXL_CHANNEL_RESERVED0,
  JPEGXL_CHANNEL_RESERVED1,
  JPEGXL_CHANNEL_RESERVED2,
  JPEGXL_CHANNEL_RESERVED3,
  JPEGXL_CHANNEL_RESERVED4,
  JPEGXL_CHANNEL_RESERVED5,
  JPEGXL_CHANNEL_RESERVED6,
  JPEGXL_CHANNEL_RESERVED7,
  JPEGXL_CHANNEL_UNKNOWN,
  JPEGXL_CHANNEL_OPTIONAL
} JpegxlExtraChannelType;

/** Indicates what the next frame will be "based" on.
 * A full frame (have_crop = false) can be based on a frame if and only if the
 * frame and the base are lossy. The rendered frame will then be the sum of
 * the two. A cropped frame can be based on any kind of frame. The rendered
 * frame will be obtained by blitting. Stored in FrameHeader and
 * ExtraChannelInfo to allow independent control for main and extra channels.
 */
typedef enum {
  /** The next frame will be based on the same frame as the current one.
   */
  JPEGXL_FRAME_BASE_EXISTING,
  /** The next frame will be based on the current one.
   */
  JPEGXL_FRAME_BASE_CURRENT_FRAME,
  /** The next frame will be a full frame (have_crop = false) and will not be
   * based on any frame, but start from a value of 0 in main and extra channels.
   */
  JPEGXL_FRAME_BASE_NONE,
} JpegxlFrameBase;

/** Indicates how to combine the current frame with the previous "base". Stored
 * in FrameHeader and ExtraChannelInfo to allow independent control for main and
 * extra channels.
 */
typedef enum {
  /** The new values (in the crop) replace the old ones
   */
  JPEGXL_BLEND_MODE_REPLACE,
  /** The new values (in the crop) get added to the old ones
   */
  JPEGXL_BLEND_MODE_ADD,
  /** The new values (in the crop) replace the old ones if alpha>0.
   * Not allowed for the first alpha channel.
   */
  JPEGXL_BLEND_MODE_BLEND,
} JpegxlBlendMode;

/** Basic image information. This information is available from the file
 * signature and first part of the codestream header.
 */
typedef struct JpegxlBasicInfo {
  // TODO(lode): need additional fields for (transcoded) JPEG? For reusable
  // fields orientation must be read from Exif APP1. For has_icc_profile: must
  // look up where ICC profile is guaranteed to be in a JPEG file to be able to
  // indicate this.

  // TODO(lode): make struct packed, and/or make this opaque struct with getter
  // functions (still separate struct from opaque decoder)

  /** Whether the codestream is embedded in the container format. If true,
   * metadata information and extensions may be available in addition to the
   * codestream.
   */
  JPEGXL_BOOL have_container;

  /** Signature of the codestream.
   */
  JpegxlSignatureType signature_type;

  /** Width of the image in pixels, before applying orientation.
   */
  uint32_t xsize;

  /** Height of the image in pixels, before applying orientation.
   */
  uint32_t ysize;

  /** Original image color channel bit depth.
   */
  uint32_t bits_per_sample;

  /** Original image color channel floating point exponent bits, or 0 if they
   * are unsigned integer. For example, if the original data is half-precision
   * (binary16) floating point, bits_per_sample is 16 and
   * exponent_bits_per_sample is 5, and so on for other floating point
   * precisions.
   */
  uint32_t exponent_bits_per_sample;

  /** Upper bound on the intensity level present in the image in nits. For
   * unsigned integer pixel encodings, this is the brightness of the largest
   * representable value. The image does not necessarily contain a pixel
   * actually this bright. An encoder is allowed to set 255 for SDR images
   * without computing a histogram.
   */
  float intensity_target;

  /** Lower bound on the intensity level present in the image. This may be
   * loose, i.e. lower than the actual darkest pixel. When tone mapping, a
   * decoder will map [min_nits, intensity_target] to the display range.
   */
  float min_nits;

  /** See the description of relative_to_max_display.
   */
  JPEGXL_BOOL relative_to_max_display;

  /** The tone mapping will leave unchanged (linear mapping) any pixels whose
   * brightness is strictly below this. The interpretation depends on
   * relative_to_max_display. If true, this is a ratio [0, 1] of the maximum
   * display brightness [nits], otherwise an absolute brightness [nits].
   */
  float linear_below;

  /** Indicates a preview image exists near the beginning of the codestream.
   * The preview itself or its dimensions are not included in the basic info.
   */
  JPEGXL_BOOL have_preview;

  /** Indicates animation frames exist in the codestream. The animation
   * information is not included in the basic info.
   */
  JPEGXL_BOOL have_animation;

  /** Image orientation, value 1-8 matching the values used by JEITA CP-3451C
   * (Exif version 2.3).
   */
  JpegxlOrientation orientation;

  /** Number of additional image channels. Information of all the individual
   * extra channels is not included in the basic info struct, except for the
   * first alpha channel in the fields below. Information for other extra
   * channels can be queried from the decoder at this point, however.
   * TODO(lode): implement that feature
   */
  uint32_t num_extra_channels;

  /** Bit depth of the encoded alpha channel, or 0 if there is no alpha channel.
   */
  uint32_t alpha_bits;

  /** Alpha channel floating point exponent bits, or 0 if they are unsigned
   * integer.
   */
  uint32_t alpha_exponent_bits;

  /** Whether the alpha channel is premultiplied
   */
  JPEGXL_BOOL alpha_premultiplied;
} JpegxlBasicInfo;

/** Information for a single extra channel.
 */
typedef struct {
  /** Given type of an extra channel.
   */
  JpegxlExtraChannelType type;

  /** Base for next frame
   */
  JpegxlFrameBase next_frame_base;

  /** Blend mode for next frame
   */
  JpegxlBlendMode blend_mode;

  /** Total bits per sample for this channel.
   */
  uint32_t bits_per_sample;

  /** Floating point exponent bits per channel, or 0 if they are unsigned
   * integer.
   */
  uint32_t exponent_bits_per_sample;

  /** The exponent the channel is downsampled by on each axis.
   * TODO(lode): expand this comment to match the JPEG XL specification,
   * specify how to upscale, how to round the size computation, and to which
   * extra channels this field applies.
   */
  uint32_t dim_shift;

  /** Length of the extra channel name in bytes, or 0 if no name.
   * Excludes null termination character.
   */
  uint32_t name_length;

  /** Whether alpha channel uses premultiplied alpha. Only applicable if
   * type is JPEGXL_CHANNEL_ALPHA.
   */
  JPEGXL_BOOL alpha_associated;

  /** Spot color of the current spot channel in linear RGBA. Only applicable if
   * type is JPEGXL_CHANNEL_SPOT_COLOR.
   */
  float spot_color[4];

  /** Only applicable if type is JPEGXL_CHANNEL_CFA.
   * TODO(lode): add comment about the meaning of this field.
   */
  uint32_t cfa_channel;
} JpegxlExtraChannelInfo;

// TODO(lode): add API to get the codestream header extensions.
/** Extensions in the codestream header. */
typedef struct {
  /** Extension bits. */
  uint64_t extensions;
} JpegxlHeaderExtensions;

/** The codestream preview header */
typedef struct {
  /** Preview width in pixels */
  uint32_t xsize;

  /** Preview height in pixels */
  uint32_t ysize;
} JpegxlPreviewHeader;

/** The codestream animation header */
typedef struct {
  /** Indicates there is a single frame followed by zero or more frames with
   * animation duration of 0. That means, there are animation frames, but they
   * are laid on top of each other to form a still image, rather than an
   * animation.
   */
  JPEGXL_BOOL composite_still;

  /** Numerator of ticks per second of a single animation frame time unit */
  uint32_t tps_numerator;

  /** Denominator of ticks per second of a single animation frame time unit */
  uint32_t tps_denominator;

  /** Amount of animation loops, or 0 to repeat infinitely */
  uint32_t num_loops;

  /** Whether animation time codes are present at animation frames in the
   * codestream */
  JPEGXL_BOOL have_timecodes;
} JpegxlAnimationHeader;

#if defined(__cplusplus) || defined(c_plusplus)
}
#endif

#endif /* JPEGXL_CODESTREAM_HEADER_H_ */
