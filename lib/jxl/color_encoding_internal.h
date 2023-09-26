// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_COLOR_ENCODING_INTERNAL_H_
#define LIB_JXL_COLOR_ENCODING_INTERNAL_H_

// Metadata for color space conversions.

#include <jxl/cms_interface.h>
#include <jxl/color_encoding.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#include <cmath>  // std::abs
#include <ostream>
#include <string>
#include <vector>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/color_encoding_external.h"
#include "lib/jxl/field_encodings.h"

namespace jxl {

static inline const char* EnumName(ColorSpace /*unused*/) {
  return "ColorSpace";
}
static inline constexpr uint64_t EnumBits(ColorSpace /*unused*/) {
  using CS = ColorSpace;
  return MakeBit(CS::kRGB) | MakeBit(CS::kGray) | MakeBit(CS::kXYB) |
         MakeBit(CS::kUnknown);
}

static inline const char* EnumName(WhitePoint /*unused*/) {
  return "WhitePoint";
}
static inline constexpr uint64_t EnumBits(WhitePoint /*unused*/) {
  return MakeBit(WhitePoint::kD65) | MakeBit(WhitePoint::kCustom) |
         MakeBit(WhitePoint::kE) | MakeBit(WhitePoint::kDCI);
}

static inline const char* EnumName(Primaries /*unused*/) { return "Primaries"; }
static inline constexpr uint64_t EnumBits(Primaries /*unused*/) {
  using Pr = Primaries;
  return MakeBit(Pr::kSRGB) | MakeBit(Pr::kCustom) | MakeBit(Pr::k2100) |
         MakeBit(Pr::kP3);
}

static inline const char* EnumName(TransferFunction /*unused*/) {
  return "TransferFunction";
}
static inline constexpr uint64_t EnumBits(TransferFunction /*unused*/) {
  using TF = TransferFunction;
  return MakeBit(TF::k709) | MakeBit(TF::kLinear) | MakeBit(TF::kSRGB) |
         MakeBit(TF::kPQ) | MakeBit(TF::kDCI) | MakeBit(TF::kHLG) |
         MakeBit(TF::kUnknown);
}

static inline const char* EnumName(RenderingIntent /*unused*/) {
  return "RenderingIntent";
}
static inline constexpr uint64_t EnumBits(RenderingIntent /*unused*/) {
  using RI = RenderingIntent;
  return MakeBit(RI::kPerceptual) | MakeBit(RI::kRelative) |
         MakeBit(RI::kSaturation) | MakeBit(RI::kAbsolute);
}

struct CustomxyProxy : public Fields {
  CustomxyProxy(Customxy* ref, bool init, bool* all_default = nullptr);
  JXL_FIELDS_NAME(Customxy)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

  Customxy* ref;
};

struct CustomTransferFunctionProxy : public Fields {
  CustomTransferFunctionProxy(CustomTransferFunction* ref, bool init,
                              bool* all_default = nullptr);
  JXL_FIELDS_NAME(CustomTransferFunction)

  CustomTransferFunction* ref;

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;
};

// Compact encoding of data required to interpret and translate pixels to a
// known color space. Stored in Metadata. Thread-compatible.
struct ColorEncodingProxy : public Fields {
  ColorEncodingProxy(ColorEncoding* ref, bool init, bool* all_default);
  JXL_FIELDS_NAME(ColorEncoding)

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override;

 private:
  ColorEncoding* ref;
  bool* all_default;
};

// Returns ready-to-use color encodings (initialized on-demand).
const ColorEncoding& ColorEncodingSRGB(bool is_gray = false);
const ColorEncoding& ColorEncodingLinearSRGB(bool is_gray = false);

// Returns a representation of the ColorEncoding fields (not icc).
// Example description: "RGB_D65_SRG_Rel_Lin"
std::string Description(const ColorEncoding& c);
static inline std::ostream& operator<<(std::ostream& os,
                                       const ColorEncoding& c) {
  return os << Description(c);
}

void ConvertInternalToExternalColorEncoding(const jxl::ColorEncoding& internal,
                                            JxlColorEncoding* external);

Status PrimariesToXYZ(float rx, float ry, float gx, float gy, float bx,
                      float by, float wx, float wy, float matrix[9]);
Status PrimariesToXYZD50(float rx, float ry, float gx, float gy, float bx,
                         float by, float wx, float wy, float matrix[9]);
Status AdaptToXYZD50(float wx, float wy, float matrix[9]);

class ColorSpaceTransform {
 public:
  explicit ColorSpaceTransform(const JxlCmsInterface& cms) : cms_(cms) {}
  ~ColorSpaceTransform() {
    if (cms_data_ != nullptr) {
      cms_.destroy(cms_data_);
    }
  }

  // Cannot copy.
  ColorSpaceTransform(const ColorSpaceTransform&) = delete;
  ColorSpaceTransform& operator=(const ColorSpaceTransform&) = delete;

  Status Init(const ColorEncoding& c_src, const ColorEncoding& c_dst,
              float intensity_target, size_t xsize, size_t num_threads) {
    xsize_ = xsize;
    JxlColorProfile input_profile;
    icc_src_ = c_src.ICC();
    input_profile.icc.data = icc_src_.data();
    input_profile.icc.size = icc_src_.size();
    ConvertInternalToExternalColorEncoding(c_src,
                                           &input_profile.color_encoding);
    input_profile.num_channels = c_src.IsCMYK() ? 4 : c_src.Channels();
    JxlColorProfile output_profile;
    icc_dst_ = c_dst.ICC();
    output_profile.icc.data = icc_dst_.data();
    output_profile.icc.size = icc_dst_.size();
    ConvertInternalToExternalColorEncoding(c_dst,
                                           &output_profile.color_encoding);
    if (c_dst.IsCMYK())
      return JXL_FAILURE("Conversion to CMYK is not supported");
    output_profile.num_channels = c_dst.Channels();
    cms_data_ = cms_.init(cms_.init_data, num_threads, xsize, &input_profile,
                          &output_profile, intensity_target);
    JXL_RETURN_IF_ERROR(cms_data_ != nullptr);
    return true;
  }

  float* BufSrc(const size_t thread) const {
    return cms_.get_src_buf(cms_data_, thread);
  }

  float* BufDst(const size_t thread) const {
    return cms_.get_dst_buf(cms_data_, thread);
  }

  Status Run(const size_t thread, const float* buf_src, float* buf_dst) {
    return cms_.run(cms_data_, thread, buf_src, buf_dst, xsize_);
  }

 private:
  JxlCmsInterface cms_;
  void* cms_data_ = nullptr;
  // The interface may retain pointers into these.
  IccBytes icc_src_;
  IccBytes icc_dst_;
  size_t xsize_;
};

}  // namespace jxl

#endif  // LIB_JXL_COLOR_ENCODING_INTERNAL_H_
