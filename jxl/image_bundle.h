// Copyright (c) the JPEG XL Project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef JXL_IMAGE_BUNDLE_H_
#define JXL_IMAGE_BUNDLE_H_

// The main image or frame consists of a bundle of associated images.

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/data_parallel.h"
#include "jxl/base/status.h"
#include "jxl/color_encoding.h"
#include "jxl/common.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/dec_xyb.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/field_encodings.h"
#include "jxl/frame_header.h"
#include "jxl/image.h"
#include "jxl/opsin_params.h"
#include "jxl/quantizer.h"

namespace jxl {

// EXIF orientation of the image. This field overrides any field present in
// actual EXIF metadata. The value tells which transformation the decoder must
// apply after decoding to display the image with the correct orientation.
enum class Orientation : uint32_t {
  // Values 1..8 match the EXIF definitions.
  kIdentity = 1,
  kFlipHorizontal,
  kRotate180,
  kFlipVertical,
  kTranspose,
  kRotate90,
  kAntiTranspose,
  kRotate270,
};
// Don't need an EnumBits because Orientation is not read via Enum().

struct ExtraChannelInfo {
  uint32_t rendered;  // 0 = no, 1 = as spot color, other values undefined
  float color[4];     // spot color in linear RGBA
};

struct OpsinInverseMatrix {
  OpsinInverseMatrix();
  static const char* Name() { return "OpsinInverseMatrix"; }

  template <class Visitor>
  Status VisitFields(Visitor* JXL_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;
    for (int i = 0; i < 9; ++i) {
      visitor->F16(DefaultInverseOpsinAbsorbanceMatrix()[i],
                   &inverse_matrix[i]);
    }
    for (int i = 0; i < 3; ++i) {
      visitor->F16(kNegOpsinAbsorbanceBiasRGB[i], &opsin_biases[i]);
    }
    for (int i = 0; i < 4; ++i) {
      visitor->F16(kDefaultQuantBias[i], &quant_biases[i]);
    }
    return true;
  }

  OpsinParams ToOpsinParams() const {
    OpsinParams opsin_params;
    InitSIMDInverseMatrix(inverse_matrix, opsin_params.inverse_opsin_matrix);
    std::copy(std::begin(opsin_biases), std::end(opsin_biases),
              opsin_params.opsin_biases);
    for (int i = 0; i < 3; ++i) {
      opsin_params.opsin_biases_cbrt[i] =
          std::cbrt(opsin_params.opsin_biases[i]);
    }
    opsin_params.opsin_biases_cbrt[3] = opsin_params.opsin_biases[3] = 255;
    std::copy(std::begin(quant_biases), std::end(quant_biases),
              opsin_params.quant_biases);
    return opsin_params;
  }

  mutable bool all_default;

  float inverse_matrix[9];
  float opsin_biases[3];
  float quant_biases[4];
};

// Less frequently changed fields, grouped into a separate bundle so they do not
// need to be signaled when some ImageMetadata fields are non-default.
struct ImageMetadata2 {
  ImageMetadata2();
  static const char* Name() { return "ImageMetadata2"; }

  template <class Visitor>
  Status VisitFields(Visitor* JXL_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;

    JXL_RETURN_IF_ERROR(visitor->VisitNested(&opsin_inverse_matrix));

    visitor->Bool(false, &have_preview);
    visitor->Bool(false, &have_animation);

    visitor->Bits(3, 0, &orientation_minus_1);
    // (No need for bounds checking because we read exactly 3 bits)

    visitor->U32(Val(0), Val(8), Val(16), Bits(4), 0, &depth_bits);
    visitor->U32(Val(0), Val(3), Val(4), BitsOffset(3, 1), 0, &depth_shift);
    if ((1U << depth_shift) > kGroupDim) {
      return JXL_FAILURE("depth_shift too large");
    }

    visitor->U32(Val(0), Bits(4), BitsOffset(8, 16), BitsOffset(12, 1), 0,
                 &num_extra_channels);
    if (visitor->Conditional(num_extra_channels != 0)) {
      visitor->U32(Val(0), Val(8), Val(16), Bits(4), 0, &extra_channel_bits);
      extra_channel_info.resize(num_extra_channels);
      for (size_t i = 0; i < num_extra_channels; i++) {
        visitor->U32(Val(0), Val(1), Val(2), Bits(6), 0,
                     &extra_channel_info[i].rendered);
        if (visitor->Conditional(extra_channel_info[i].rendered == 1)) {
          for (float& c : extra_channel_info[i].color) {
            visitor->F16(0, &c);
          }
        }
      }
    }

    visitor->BeginExtensions(&extensions);
    // Extensions: in chronological order of being added to the format.
    return visitor->EndExtensions();
  }

  bool HasDepth() const { return depth_bits != 0; }

  mutable bool all_default;

  OpsinInverseMatrix opsin_inverse_matrix;

  bool have_preview;
  bool have_animation;

  uint32_t orientation_minus_1;
  uint32_t depth_bits;  // 0 if no depth channel present
  uint32_t depth_shift;

  uint32_t num_extra_channels;
  uint32_t extra_channel_bits;  // 0 if num_extra_channels == 0
  std::vector<ExtraChannelInfo> extra_channel_info;

  uint64_t extensions;
};

// Properties of the original image bundle. This enables Encode(Decode()) to
// re-create an equivalent image without user input.
struct ImageMetadata {
  ImageMetadata();
  static const char* Name() { return "ImageMetadata"; }

  template <class Visitor>
  Status VisitFields(Visitor* JXL_RESTRICT visitor) {
    if (visitor->AllDefault(*this, &all_default)) return true;

    visitor->Bool(false, &have_icc);

    visitor->U32(Val(8), Val(16), Val(32), Bits(5), 8, &bits_per_sample);
    JXL_RETURN_IF_ERROR(visitor->VisitNested(&color_encoding));
    visitor->U32(Val(0), Val(8), Val(16), Bits(4), 0, &alpha_bits);

    // 250, 1000, 4000 are common; don't anticipate more than 10,000.
    visitor->U32(Val(5), Val(20), Val(80), BitsOffset(10, 1),
                 kDefaultIntensityTarget / 50, &target_nits_div50_);

    JXL_RETURN_IF_ERROR(visitor->VisitNested(&m2));

    return true;
  }

  bool HasAlpha() const { return alpha_bits != 0; }

  void SetIntensityTarget(float intensity_target) {
    target_nits_div50_ = static_cast<uint32_t>(intensity_target * 0.02f);
  }
  float IntensityTarget() const { return target_nits_div50_ * 50.f; }

  mutable bool all_default;
  bool have_icc;

  ImageMetadata2 m2;  // often default

  uint32_t bits_per_sample;
  ColorEncoding color_encoding;
  uint32_t alpha_bits;  // 0 if no alpha channel present

 private:
  uint32_t target_nits_div50_;
};

// Chooses a default intensity target based on the transfer function of the
// image, if known. For SDR images or images not known to be HDR, returns
// kDefaultIntensityTarget, for images known to have PQ or HLG transfer function
// returns a higher value.
float ChooseDefaultIntensityTarget(const ImageMetadata& metadata);

Status ReadImageMetadata(BitReader* JXL_RESTRICT reader,
                         ImageMetadata* JXL_RESTRICT metadata);

Status WriteImageMetadata(const ImageMetadata& metadata,
                          BitWriter* JXL_RESTRICT writer, size_t layer,
                          AuxOut* aux_out);

// A bundle of color/alpha/depth/plane images.
class ImageBundle {
 public:
  // Caller is responsible for setting metadata before calling Set*.
  explicit ImageBundle(ImageMetadata* metadata) : metadata_(metadata) {}

  // Move-only (allows storing in std::vector).
  ImageBundle(ImageBundle&&) = default;
  ImageBundle& operator=(ImageBundle&&) = default;

  ImageBundle Copy() const {
    ImageBundle copy(metadata_);
    copy.color_ = CopyImage(color_);
    copy.c_current_ = c_current_;
    copy.alpha_ = CopyImage(alpha_);
    copy.depth_ = CopyImage(depth_);
    copy.extra_channels_.reserve(extra_channels_.size());
    for (const ImageU& plane : extra_channels_) {
      copy.extra_channels_.emplace_back(CopyImage(plane));
    }
    return copy;
  }

  // -- SIZE

  size_t xsize() const {
    if (jpeg_xsize != 0) return jpeg_xsize;
    if (color_.xsize() != 0) return color_.xsize();
    return extra_channels_.empty() ? 0 : extra_channels_[0].xsize();
  }
  size_t ysize() const {
    if (jpeg_ysize != 0) return jpeg_ysize;
    if (color_.ysize() != 0) return color_.ysize();
    return extra_channels_.empty() ? 0 : extra_channels_[0].ysize();
  }
  void ShrinkTo(size_t xsize, size_t ysize);

  // -- COLOR

  // Whether color() is valid/usable. Returns true in most cases. Even images
  // with spot colors (one example of when !planes().empty()) typically have a
  // part that can be converted to RGB.
  bool HasColor() const { return color_.xsize() != 0; }

  // For resetting the size when switching from a reference to main frame.
  void RemoveColor() { color_ = Image3F(); }

  // Do not use if !HasColor().
  const Image3F& color() const {
    // If this fails, Set* was not called - perhaps because decoding failed?
    JXL_DASSERT(HasColor());
    return color_;
  }

  // Do not use if !HasColor().
  Image3F* MutableColor() {
    JXL_DASSERT(HasColor());
    return &color_;
  }

  // If c_current.IsGray(), all planes must be identical. NOTE: c_current is
  // independent of metadata()->color_encoding, which is the original, whereas
  // a decoder might return pixels in a different c_current.
  void SetFromImage(Image3F&& color, const ColorEncoding& c_current);

  // Sets image data from 8-bit sRGB pixel array in bytes.
  // Amount of input bytes per pixel must be:
  // (is_gray ? 1 : 3) + (has_alpha ? 1 : 0)
  Status SetFromSRGB(size_t xsize, size_t ysize, bool is_gray, bool has_alpha,
                     const uint8_t* pixels, const uint8_t* end,
                     ThreadPool* pool = nullptr);

  // Sets image data from 16-bit sRGB data.
  // Amount of input uint16_t's per pixel must be:
  // (is_gray ? 1 : 3) + (has_alpha ? 1 : 0)
  Status SetFromSRGB(size_t xsize, size_t ysize, bool is_gray, bool has_alpha,
                     const uint16_t* pixels, const uint16_t* end,
                     ThreadPool* pool = nullptr);

  // Sets image data from sRGB pixel array in bytes.
  // This low-level function supports both 8-bit and 16-bit data in bytes to
  // provide efficient access to arbitrary byte order.
  // Amount of input bytes per pixel must be:
  // ((is_gray ? 1 : 3) + (has_alpha ? 1 : 0)) * (is_16bit ? 2 : 1)
  // The ordering of the channels is interleaved RGBA or gray+alpha in that
  // order.
  // The 16-bit byte order is given by big_endian, and this has no effect when
  // is_16bit is false.
  Status SetFromSRGB(size_t xsize, size_t ysize, bool is_gray, bool has_alpha,
                     bool is_16bit, bool big_endian, const uint8_t* pixels,
                     const uint8_t* end, ThreadPool* pool = nullptr);

  // -- COLOR ENCODING

  const ColorEncoding& c_current() const { return c_current_; }

  // Returns whether the color image has identical planes. Once established by
  // Set*, remains unchanged until a subsequent Set* or TransformTo.
  bool IsGray() const { return c_current_.IsGray(); }

  bool IsSRGB() const { return c_current_.IsSRGB(); }
  bool IsLinearSRGB() const {
    return c_current_.white_point == WhitePoint::kD65 &&
           c_current_.primaries == Primaries::kSRGB && c_current_.tf.IsLinear();
  }

  // Transforms color to c_desired and sets c_current to c_desired. Alpha and
  // metadata remains unchanged.
  Status TransformTo(const ColorEncoding& c_desired,
                     ThreadPool* pool = nullptr);

  // Copies this:rect, converts to c_desired, and allocates+fills out.
  Status CopyTo(const Rect& rect, const ColorEncoding& c_desired, Image3B* out,
                ThreadPool* pool = nullptr) const;
  Status CopyTo(const Rect& rect, const ColorEncoding& c_desired, Image3U* out,
                ThreadPool* pool = nullptr) const;
  Status CopyTo(const Rect& rect, const ColorEncoding& c_desired, Image3F* out,
                ThreadPool* pool = nullptr) const;
  Status CopyToSRGB(const Rect& rect, Image3B* out,
                    ThreadPool* pool = nullptr) const;

  // Detect 'real' bit depth, which can be lower than nominal bit depth
  // (this is common in PNG), returns 'real' bit depth
  size_t DetectRealBitdepth() const;

  // -- ALPHA

  void SetAlpha(ImageU&& alpha);
  bool HasAlpha() const { return alpha_.xsize() != 0; }
  const ImageU& alpha() const {
    JXL_ASSERT(HasAlpha());
    return alpha_;
  }

  // Reverts SetAlpha AND sets metadata->alpha_bits to 0. Called after noticing
  // that all alpha values are opaque.
  void RemoveAlpha();

  // -- DEPTH

  void SetDepth(ImageU&& depth);
  bool HasDepth() const { return depth_.xsize() != 0; }
  const ImageU& depth() const {
    JXL_ASSERT(HasDepth());
    return depth_;
  }
  // Returns the dimensions of the depth image
  size_t DepthSize(size_t size) const {
    const size_t mask = (1u << metadata_->m2.depth_shift) - 1;
    return (size + mask) >> metadata_->m2.depth_shift;
  }

  // -- EXTRA CHANNELS

  // Extra channels of unknown interpretation (e.g. spot colors).
  void SetExtraChannels(std::vector<ImageU>&& extra_channels);
  bool HasExtraChannels() const { return !extra_channels_.empty(); }
  const std::vector<ImageU>& extra_channels() const {
    JXL_ASSERT(HasExtraChannels());
    return extra_channels_;
  }

  const ImageMetadata* metadata() const { return metadata_; }

  void VerifyMetadata() const;

  void SetDecodedBytes(size_t decoded_bytes) { decoded_bytes_ = decoded_bytes; }
  size_t decoded_bytes() const { return decoded_bytes_; }

  // if not empty: image data represents quantized DCT-8 coefficients
  // (used for JPEG transcoding)
  std::vector<int> jpeg_quant_table;
  size_t jpeg_xsize = 0;  // image dimensions of input JPEG
  size_t jpeg_ysize = 0;  // (can be up to 7 smaller than color_ dimensions)
  // these fields are used to signal the input JPEG color space
  ColorTransform color_transform = ColorTransform::kNone;
  YCbCrChromaSubsampling chroma_subsampling = YCbCrChromaSubsampling::k444;

 private:
  // Called after any Set* to ensure their sizes are compatible.
  void VerifySizes() const;

  // Required for TransformTo so that an ImageBundle is self-sufficient. Always
  // points to the same thing, but cannot be const-pointer because that prevents
  // the compiler from generating a move ctor. Mostly pointer-to-const except as
  // required by RemoveAlpha.
  ImageMetadata* metadata_;

  // Initialized by Set*:
  Image3F color_;  // If empty, planes_ is not; all planes equal if IsGray().
  ColorEncoding c_current_;  // of color_

  // Initialized by SetAlpha.
  ImageU alpha_;  // Empty or same size as color_.

  // Initialized by SetDepth.
  ImageU depth_;  // Empty or size >> depth_shift_.

  // Initialized by SetPlanes; size = ImageMetadata.num_extra_channels
  std::vector<ImageU> extra_channels_;

  // How many bytes of the input were actually read.
  size_t decoded_bytes_ = 0;
};

// Does color transformation from in.c_current() to c_desired if the color
// encodings are different, or nothing if they are already the same.
// If color transformation is done, stores the transformed values into store and
// sets the out pointer to store, else leaves store untouched and sets the out
// pointer to &in.
// Returns false if color transform fails.
Status TransformIfNeeded(const ImageBundle& in, const ColorEncoding& c_desired,
                         ThreadPool* pool, ImageBundle* store,
                         const ImageBundle** out);

}  // namespace jxl

#endif  // JXL_IMAGE_BUNDLE_H_
