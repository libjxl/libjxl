// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_COLOR_ENCODING_EXTERNAL_H_
#define LIB_JXL_COLOR_ENCODING_EXTERNAL_H_

#include <jxl/color_encoding.h>
#include <stdint.h>

#include <cmath>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"

namespace jxl {

using IccBytes = std::vector<uint8_t>;
struct ColorEncoding;
struct CIExy;

// NOTE: for XYB colorspace, the created profile can be used to transform a
// *scaled* XYB image (created by ScaleXYB()) to another colorspace.
Status MaybeCreateProfile(const ColorEncoding& c, IccBytes* JXL_RESTRICT icc);

Status CIEXYZFromWhiteCIExy(const CIExy& xy, float XYZ[3]);

Status ConvertExternalToInternalColorEncoding(const JxlColorEncoding& external,
                                              jxl::ColorEncoding* internal);

// Returns whether the two inputs are approximately equal.
static inline bool ApproxEq(const double a, const double b,
                            double max_l1 = 1E-3) {
  // Threshold should be sufficient for ICC's 15-bit fixed-point numbers.
  // We have seen differences of 7.1E-5 with lcms2 and 1E-3 with skcms.
  return std::abs(a - b) <= max_l1;
}

// (All CIE units are for the standard 1931 2 degree observer)

// Color space the color pixel data is encoded in. The color pixel data is
// 3-channel in all cases except in case of kGray, where it uses only 1 channel.
// This also determines the amount of channels used in modular encoding.
enum class ColorSpace : uint32_t {
  // Trichromatic color data. This also includes CMYK if a kBlack
  // ExtraChannelInfo is present. This implies, if there is an ICC profile, that
  // the ICC profile uses a 3-channel color space if no kBlack extra channel is
  // present, or uses color space 'CMYK' if a kBlack extra channel is present.
  kRGB,
  // Single-channel data. This implies, if there is an ICC profile, that the ICC
  // profile also represents single-channel data and has the appropriate color
  // space ('GRAY').
  kGray,
  // Like kRGB, but implies fixed values for primaries etc.
  kXYB,
  // For non-RGB/gray data, e.g. from non-electro-optical sensors. Otherwise
  // the same conditions as kRGB apply.
  kUnknown
  // NB: don't forget to update EnumBits!
};

// Values from CICP ColourPrimaries.
enum class WhitePoint : uint32_t {
  kD65 = 1,     // sRGB/BT.709/Display P3/BT.2020
  kCustom = 2,  // Actual values encoded in separate fields
  kE = 10,      // XYZ
  kDCI = 11,    // DCI-P3
  // NB: don't forget to update EnumBits!
};

// Values from CICP ColourPrimaries
enum class Primaries : uint32_t {
  kSRGB = 1,    // Same as BT.709
  kCustom = 2,  // Actual values encoded in separate fields
  k2100 = 9,    // Same as BT.2020
  kP3 = 11,
  // NB: don't forget to update EnumBits!
};

// Values from CICP TransferCharacteristics
enum class TransferFunction : uint32_t {
  k709 = 1,
  kUnknown = 2,
  kLinear = 8,
  kSRGB = 13,
  kPQ = 16,   // from BT.2100
  kDCI = 17,  // from SMPTE RP 431-2 reference projector
  kHLG = 18,  // from BT.2100
  // NB: don't forget to update EnumBits!
};

enum class RenderingIntent : uint32_t {
  // Values match ICC sRGB encodings.
  kPerceptual = 0,  // good for photos, requires a profile with LUT.
  kRelative,        // good for logos.
  kSaturation,      // perhaps useful for CG with fully saturated colors.
  kAbsolute,        // leaves white point unchanged; good for proofing.
  // NB: don't forget to update EnumBits!
};

// Chromaticity (Y is omitted because it is 1 for primaries/white points)
struct CIExy {
  double x = 0.0;
  double y = 0.0;
};

struct PrimariesCIExy {
  CIExy r;
  CIExy g;
  CIExy b;
};

static double F64FromCustomxyI32(const int32_t i) { return i * 1E-6; }
static Status F64ToCustomxyI32(const double f, int32_t* JXL_RESTRICT i) {
  if (!(-4 <= f && f <= 4)) {
    return JXL_FAILURE("F64 out of bounds for CustomxyI32");
  }
  *i = static_cast<int32_t>(roundf(f * 1E6));
  return true;
}

// Serializable form of CIExy.
struct Customxy {
  CIExy Get() const {
    CIExy xy;
    xy.x = F64FromCustomxyI32(x);
    xy.y = F64FromCustomxyI32(y);
    return xy;
  }

  constexpr static int32_t kMinValue = -(1 << 21);
  constexpr static int32_t kMaxValue = (1 << 21) - 1;

  // Returns false if x or y do not fit in the encoding.
  Status Set(const CIExy& xy) {
    JXL_RETURN_IF_ERROR(F64ToCustomxyI32(xy.x, &x));
    JXL_RETURN_IF_ERROR(F64ToCustomxyI32(xy.y, &y));
    if (x < kMinValue || x > kMaxValue || y < kMinValue || y > kMaxValue) {
      return JXL_FAILURE("Unable to encode XY %f %f", xy.x, xy.y);
    }
    return true;
  }

  int32_t x;
  int32_t y;
};

struct CustomTransferFunction {
  // Highest reasonable value for the gamma of a transfer curve.
  static constexpr uint32_t kMaxGamma = 8192;
  static constexpr uint32_t kGammaMul = 10000000;

  // Sets fields and returns true if nonserialized_color_space has an implicit
  // transfer function, otherwise leaves fields unchanged and returns false.
  bool SetImplicit() {
    if (nonserialized_color_space == ColorSpace::kXYB) {
      if (!SetGamma(1.0 / 3)) JXL_ASSERT(false);
      return true;
    }
    return false;
  }

  // Gamma: only used for PNG inputs
  bool IsGamma() const { return have_gamma_; }

  double GetGamma() const {
    JXL_ASSERT(IsGamma());
    return gamma_ * 1E-7;  // (0, 1)
  }

  Status SetGamma(double gamma) {
    if (gamma < (1.0f / kMaxGamma) || gamma > 1.0) {
      return JXL_FAILURE("Invalid gamma %f", gamma);
    }

    have_gamma_ = false;
    if (ApproxEq(gamma, 1.0)) {
      transfer_function_ = TransferFunction::kLinear;
      return true;
    }
    if (ApproxEq(gamma, 1.0 / 2.6)) {
      transfer_function_ = TransferFunction::kDCI;
      return true;
    }
    // Don't translate 0.45.. to kSRGB nor k709 - that might change pixel
    // values because those curves also have a linear part.

    have_gamma_ = true;
    gamma_ = roundf(gamma * kGammaMul);
    transfer_function_ = TransferFunction::kUnknown;
    return true;
  }

  TransferFunction GetTransferFunction() const {
    JXL_ASSERT(!IsGamma());
    return transfer_function_;
  }

  void SetTransferFunction(const TransferFunction tf) {
    have_gamma_ = false;
    transfer_function_ = tf;
  }

  bool IsUnknown() const {
    return !have_gamma_ && (transfer_function_ == TransferFunction::kUnknown);
  }

  bool IsSRGB() const {
    return !have_gamma_ && (transfer_function_ == TransferFunction::kSRGB);
  }

  bool IsLinear() const {
    return !have_gamma_ && (transfer_function_ == TransferFunction::kLinear);
  }

  bool IsPQ() const {
    return !have_gamma_ && (transfer_function_ == TransferFunction::kPQ);
  }

  bool IsHLG() const {
    return !have_gamma_ && (transfer_function_ == TransferFunction::kHLG);
  }

  bool Is709() const {
    return !have_gamma_ && (transfer_function_ == TransferFunction::k709);
  }

  bool IsDCI() const {
    return !have_gamma_ && (transfer_function_ == TransferFunction::kDCI);
  }

  bool IsSame(const CustomTransferFunction& other) const {
    if (have_gamma_ != other.have_gamma_) return false;
    if (have_gamma_) {
      if (gamma_ != other.gamma_) return false;
    } else {
      if (transfer_function_ != other.transfer_function_) return false;
    }
    return true;
  }

  ColorSpace nonserialized_color_space = ColorSpace::kRGB;

 private:
  friend struct CustomTransferFunctionProxy;

  bool have_gamma_;

  // OETF exponent to go from linear to gamma-compressed.
  uint32_t gamma_;  // Only used if have_gamma_.

  // Can be kUnknown.
  TransferFunction transfer_function_;  // Only used if !have_gamma_.
};

// Compact encoding of data required to interpret and translate pixels to a
// known color space. Stored in Metadata. Thread-compatible.
struct ColorEncoding {
  // Returns true if an ICC profile was successfully created from fields.
  // Must be called after modifying fields. Defined in color_management.cc.
  Status CreateICC() {
    InternalRemoveICC();
    return MaybeCreateProfile(*this, &icc_);
  }

  // Returns non-empty and valid ICC profile, unless:
  // - between calling InternalRemoveICC() and CreateICC() in tests;
  // - WantICC() == true and SetICC() was not yet called;
  // - after a failed call to SetSRGB(), SetICC(), or CreateICC().
  const IccBytes& ICC() const { return icc_; }

  // Internal only, do not call except from tests.
  void InternalRemoveICC() { icc_.clear(); }

  // Returns true if `icc` is assigned and decoded successfully. If so,
  // subsequent WantICC() will return true until DecideIfWantICC() changes it.
  // Returning false indicates data has been lost.
  Status SetICC(IccBytes&& icc, const JxlCmsInterface* cms) {
    if (icc.empty()) return false;
    icc_ = std::move(icc);

    if (cms == nullptr) {
      want_icc_ = true;
      have_fields_ = false;
      return true;
    }

    if (!SetFieldsFromICC(*cms)) {
      InternalRemoveICC();
      return false;
    }

    want_icc_ = true;
    return true;
  }

  // Sets the raw ICC profile bytes, without parsing the ICC, and without
  // updating the direct fields such as whitepoint, primaries and color
  // space. Functions to get and set fields, such as SetWhitePoint, cannot be
  // used anymore after this and functions such as IsSRGB return false no matter
  // what the contents of the icc profile.
  Status SetICCRaw(IccBytes&& icc) {
    if (icc.empty()) return false;
    icc_ = std::move(icc);

    want_icc_ = true;
    have_fields_ = false;
    return true;
  }

  // Returns whether to send the ICC profile in the codestream.
  bool WantICC() const { return want_icc_; }

  // Return whether the direct fields are set, if false but ICC is set, only
  // raw ICC bytes are known.
  bool HaveFields() const { return have_fields_; }

  // Causes WantICC() to return false if ICC() can be reconstructed from fields.
  void DecideIfWantICC(const JxlCmsInterface& cms) {
    if (icc_.empty()) return;

    JxlColorEncoding c;
    JXL_BOOL cmyk;
    if (!cms.set_fields_from_icc(cms.set_fields_data, icc_.data(), icc_.size(),
                                 &c, &cmyk)) {
      return;
    }
    if (cmyk) return;

    IccBytes new_icc;
    if (!MaybeCreateProfile(*this, &new_icc)) return;

    want_icc_ = false;
  }

  bool IsGray() const { return color_space_ == ColorSpace::kGray; }
  bool IsCMYK() const { return cmyk_; }
  size_t Channels() const { return IsGray() ? 1 : 3; }

  // Returns false if the field is invalid and unusable.
  bool HasPrimaries() const {
    return !IsGray() && color_space_ != ColorSpace::kXYB;
  }

  // Returns true after setting the field to a value defined by color_space,
  // otherwise false and leaves the field unchanged.
  bool ImplicitWhitePoint() {
    if (color_space_ == ColorSpace::kXYB) {
      white_point = WhitePoint::kD65;
      return true;
    }
    return false;
  }

  // Returns whether the color space is known to be sRGB. If a raw unparsed ICC
  // profile is set without the fields being set, this returns false, even if
  // the content of the ICC profile would match sRGB.
  bool IsSRGB() const {
    if (!have_fields_) return false;
    if (!IsGray() && color_space_ != ColorSpace::kRGB) return false;
    if (white_point != WhitePoint::kD65) return false;
    if (primaries != Primaries::kSRGB) return false;
    if (!tf.IsSRGB()) return false;
    return true;
  }

  // Returns whether the color space is known to be linear sRGB. If a raw
  // unparsed ICC profile is set without the fields being set, this returns
  // false, even if the content of the ICC profile would match linear sRGB.
  bool IsLinearSRGB() const {
    if (!have_fields_) return false;
    if (!IsGray() && color_space_ != ColorSpace::kRGB) return false;
    if (white_point != WhitePoint::kD65) return false;
    if (primaries != Primaries::kSRGB) return false;
    if (!tf.IsLinear()) return false;
    return true;
  }

  Status SetSRGB(const ColorSpace cs,
                 const RenderingIntent ri = RenderingIntent::kRelative) {
    InternalRemoveICC();
    JXL_ASSERT(cs == ColorSpace::kGray || cs == ColorSpace::kRGB);
    color_space_ = cs;
    white_point = WhitePoint::kD65;
    primaries = Primaries::kSRGB;
    tf.SetTransferFunction(TransferFunction::kSRGB);
    rendering_intent = ri;
    return CreateICC();
  }

  // Accessors ensure tf.nonserialized_color_space is updated at the same time.
  ColorSpace GetColorSpace() const { return color_space_; }
  void SetColorSpace(const ColorSpace cs) {
    color_space_ = cs;
    tf.nonserialized_color_space = cs;
  }

  CIExy GetWhitePoint() const;
  Status SetWhitePoint(const CIExy& xy);

  PrimariesCIExy GetPrimaries() const;
  Status SetPrimaries(const PrimariesCIExy& xy);

  // Checks if the color spaces (including white point / primaries) are the
  // same, but ignores the transfer function, rendering intent and ICC bytes.
  bool SameColorSpace(const ColorEncoding& other) const {
    if (color_space_ != other.color_space_) return false;

    if (white_point != other.white_point) return false;
    if (white_point == WhitePoint::kCustom) {
      if (white_.x != other.white_.x || white_.y != other.white_.y)
        return false;
    }

    if (HasPrimaries() != other.HasPrimaries()) return false;
    if (HasPrimaries()) {
      if (primaries != other.primaries) return false;
      if (primaries == Primaries::kCustom) {
        if (red_.x != other.red_.x || red_.y != other.red_.y) return false;
        if (green_.x != other.green_.x || green_.y != other.green_.y)
          return false;
        if (blue_.x != other.blue_.x || blue_.y != other.blue_.y) return false;
      }
    }
    return true;
  }

  // Checks if the color space and transfer function are the same, ignoring
  // rendering intent and ICC bytes
  bool SameColorEncoding(const ColorEncoding& other) const {
    return SameColorSpace(other) && tf.IsSame(other.tf);
  }

  // Only valid if HaveFields()
  WhitePoint white_point = WhitePoint::kD65;
  Primaries primaries = Primaries::kSRGB;  // Only valid if HasPrimaries()
  CustomTransferFunction tf;
  RenderingIntent rendering_intent = RenderingIntent::kRelative;

 private:
  friend struct ColorEncodingProxy;

  // Returns true if all fields have been initialized (possibly to kUnknown).
  // Returns false if the ICC profile is invalid or decoding it fails.
  Status SetFieldsFromICC(const JxlCmsInterface& cms) {
    // In case parsing fails, mark the ColorEncoding as invalid.
    SetColorSpace(ColorSpace::kUnknown);
    tf.SetTransferFunction(TransferFunction::kUnknown);

    if (icc_.empty()) return JXL_FAILURE("Empty ICC profile");

    JxlColorEncoding external;
    JXL_BOOL cmyk;
    JXL_RETURN_IF_ERROR(cms.set_fields_from_icc(
        cms.set_fields_data, icc_.data(), icc_.size(), &external, &cmyk));
    if (cmyk) {
      cmyk_ = true;
      return true;
    }
    IccBytes icc = std::move(icc_);
    JXL_RETURN_IF_ERROR(ConvertExternalToInternalColorEncoding(external, this));
    icc_ = std::move(icc);
    return true;
  }

  // If true, the codestream contains an ICC profile and we do not serialize
  // fields. Otherwise, fields are serialized and we create an ICC profile.
  bool want_icc_ = false;

  // When false, fields such as white_point and tf are invalid and must not be
  // used. This occurs after setting a raw bytes-only ICC profile, only the
  // ICC bytes may be used. The color_space_ field is still valid.
  bool have_fields_ = true;

  IccBytes icc_;  // Valid ICC profile

  ColorSpace color_space_ = ColorSpace::kRGB;  // Can be kUnknown
  bool cmyk_ = false;

  // Only used if white_point == kCustom.
  Customxy white_;

  // Only used if primaries == kCustom.
  Customxy red_;
  Customxy green_;
  Customxy blue_;
};

enum class ExtraTF {
  kNone,
  kPQ,
  kHLG,
  kSRGB,
};

}  // namespace jxl

#endif  // LIB_JXL_COLOR_ENCODING_EXTERNAL_H_
