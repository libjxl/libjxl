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

#ifndef JXL_FRAME_HEADER_H_
#define JXL_FRAME_HEADER_H_

// Frame header with backward and forward-compatible extension capability and
// compressed integer fields.

#include <stddef.h>
#include <stdint.h>

#include <string>

#include "jxl/aux_out_fwd.h"
#include "jxl/base/compiler_specific.h"
#include "jxl/base/override.h"
#include "jxl/base/padded_bytes.h"
#include "jxl/base/status.h"
#include "jxl/coeff_order_fwd.h"
#include "jxl/common.h"
#include "jxl/dec_bit_reader.h"
#include "jxl/enc_bit_writer.h"
#include "jxl/fields.h"
#include "jxl/gaborish.h"

namespace jxl {

enum class FrameEncoding : uint32_t {
  kVarDCT,
  kModular,
};

enum class ColorTransform : uint32_t {
  kXYB,    // Values are encoded with XYB. May only be used if
           // ImageBundle::xyb_encoded.
  kNone,   // Values are encoded according to the attached color profile. May
           // only be used if !ImageBundle::xyb_encoded.
  kYCbCr,  // Values are encoded according to the attached color profile, but
           // transformed to YCbCr. May only be used if
           // !ImageBundle::xyb_encoded.
  kSRGB    // Values are encoded with sRGB, which may differ from the attached
         // color profile. May only be used if ImageBundle::xyb_encoded. If you
         // have !ImageBundle::xyb_encoded and sRGB is desired, use an sRGB
         // color profile and set this to kNone instead.
         // TODO(lode): support kSRGB in encoder and decoder.
};

struct YCbCrChromaSubsampling : public Fields {
  YCbCrChromaSubsampling();
  const char* Name() const override { return "YCbCrChromaSubsampling"; }
  size_t HShift(size_t c) const { return maxhs_ - kHShift[channel_mode_[c]]; }
  size_t VShift(size_t c) const { return maxvs_ - kVShift[channel_mode_[c]]; }

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override {
    // TODO(veluca): consider allowing 4x downsamples
    for (size_t i = 0; i < 3; i++) {
      visitor->Bits(2, 0, &channel_mode_[i]);
    }
    Recompute();
    return true;
  }

  uint8_t MaxHShift() const { return maxhs_; }
  uint8_t MaxVShift() const { return maxvs_; }

  // Uses JPEG channel order (Y, Cb, Cr).
  Status Set(const uint8_t* hsample, const uint8_t* vsample) {
    for (size_t c = 0; c < 3; c++) {
      size_t cjpeg = c < 2 ? c ^ 1 : c;
      size_t i = 0;
      for (; i < 4; i++) {
        if (1 << kHShift[i] == hsample[cjpeg] &&
            1 << kVShift[i] == vsample[cjpeg]) {
          channel_mode_[c] = i;
          break;
        }
      }
      if (i == 4) {
        return JXL_FAILURE("Invalid subsample mode");
      }
    }
    Recompute();
    return true;
  }

  bool Is444() const {
    for (size_t c : {0, 2}) {
      if (channel_mode_[c] != channel_mode_[1]) {
        return false;
      }
    }
    return true;
  }

  bool Is420() const {
    return channel_mode_[0] == 1 && channel_mode_[1] == 0 &&
           channel_mode_[2] == 1;
  }

  bool Is422() const {
    for (size_t c : {0, 2}) {
      if (kHShift[channel_mode_[c]] == kHShift[channel_mode_[1]] + 1 &&
          kVShift[channel_mode_[c]] == kVShift[channel_mode_[1]]) {
        return false;
      }
    }
    return true;
  }

  bool Is440() const {
    for (size_t c : {0, 2}) {
      if (kHShift[channel_mode_[c]] == kHShift[channel_mode_[1]] &&
          kVShift[channel_mode_[c]] == kVShift[channel_mode_[1]] + 1) {
        return false;
      }
    }
    return true;
  }

 private:
  void Recompute() {
    maxhs_ = 0;
    maxvs_ = 0;
    for (size_t i = 0; i < 3; i++) {
      maxhs_ = std::max(maxhs_, kHShift[channel_mode_[i]]);
      maxvs_ = std::max(maxvs_, kVShift[channel_mode_[i]]);
    }
  }
  static constexpr uint8_t kHShift[4] = {0, 1, 1, 0};
  static constexpr uint8_t kVShift[4] = {0, 1, 0, 1};
  uint32_t channel_mode_[3];
  uint8_t maxhs_;
  uint8_t maxvs_;
};

static inline const char* EnumName(ColorTransform /*unused*/) {
  return "ColorTransform";
}
static inline constexpr uint64_t EnumBits(ColorTransform /*unused*/) {
  return MakeBit(ColorTransform::kXYB) | MakeBit(ColorTransform::kNone) |
         MakeBit(ColorTransform::kYCbCr) | MakeBit(ColorTransform::kSRGB);
}
static inline const char* EnumName(YCbCrChromaSubsampling /*unused*/) {
  return "YCbCrChromaSubsampling";
}

static inline void VisitNameString(Visitor* JXL_RESTRICT visitor,
                                   std::string* name) {
  uint32_t name_length = name->length();
  // Allows layer name lengths up to 1071 bytes
  visitor->U32(Val(0), Bits(4), BitsOffset(5, 16), BitsOffset(10, 48), 0,
               &name_length);
  if (visitor->IsReading()) {
    name->resize(name_length);
  }
  for (size_t i = 0; i < name_length; i++) {
    uint32_t c = (*name)[i];
    visitor->Bits(8, 0, &c);
    (*name)[i] = c;
  }
}

// Indicates what the next frame will be "based" on.
// A full frame (have_crop = false) can be based on a frame if and only if the
// frame and the base are lossy. The rendered frame will then be the sum of
// the two. A cropped frame can be based on any kind of frame. The rendered
// frame will be obtained by blitting. Stored in FrameHeader and
// ExtraChannelInfo to allow independent control for main and extra channels.
enum class NewBase {
  // The next frame will be based on the same frame as the current one.
  kExisting,
  // The next frame will be based on the current one.
  kCurrentFrame,
  // The next frame will be a full frame (have_crop = false) and will not be
  // based on any frame, but start from a value of 0 in main and extra channels.
  kNone,
};

static inline Status VisitNewBase(Visitor* JXL_RESTRICT visitor,
                                  NewBase* new_base) {
  uint32_t encoded = static_cast<uint32_t>(*new_base);
  visitor->U32(Val(static_cast<uint32_t>(NewBase::kExisting)),
               Val(static_cast<uint32_t>(NewBase::kCurrentFrame)),
               Val(static_cast<uint32_t>(NewBase::kNone)), Val(3),
               static_cast<uint32_t>(NewBase::kCurrentFrame), &encoded);
  if (encoded == 3) {
    return JXL_FAILURE("Invalid new_base");
  }
  *new_base = static_cast<NewBase>(encoded);
  return true;
}

// Indicates how to combine the current frame with the previous "base". Stored
// in FrameHeader and ExtraChannelInfo to allow independent control for main and
// extra channels. Update tools/djxl.cc if blend modes change.
enum class BlendMode {
  // The new values (in the crop) replace the old ones
  kReplace,
  // The new values (in the crop) get added to the old ones
  kAdd,
  // The new values (in the crop) replace the old ones if alpha>0.
  // Not allowed for the first alpha channel.
  kBlend,
};

static inline Status VisitBlendMode(Visitor* JXL_RESTRICT visitor,
                                    BlendMode* blend_mode) {
  uint32_t encoded = static_cast<uint32_t>(*blend_mode);
  visitor->U32(Val(static_cast<uint32_t>(BlendMode::kReplace)),
               Val(static_cast<uint32_t>(BlendMode::kAdd)),
               Val(static_cast<uint32_t>(BlendMode::kBlend)), Val(3),
               static_cast<uint32_t>(BlendMode::kReplace), &encoded);
  if (encoded == 3) {
    return JXL_FAILURE("Invalid blend_mode");
  }
  *blend_mode = static_cast<BlendMode>(encoded);
  return true;
}

struct AnimationFrame : public Fields {
  AnimationFrame();
  const char* Name() const override { return "AnimationFrame"; }

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override {
    if (visitor->Conditional(!nonserialized_composite_still)) {
      visitor->U32(Val(0), Val(1), Bits(8), Bits(32), 0, &duration);
    }

    VisitNameString(visitor, &name);

    if (visitor->Conditional(duration > 0)) {
      JXL_RETURN_IF_ERROR(VisitNewBase(visitor, &new_base));
    }
    JXL_RETURN_IF_ERROR(VisitBlendMode(visitor, &blend_mode));

    if (visitor->Conditional(nonserialized_have_timecode)) {
      visitor->Bits(32, 0, &timecode);
    }

    visitor->Bool(false, &have_crop);
    if (visitor->Conditional(have_crop)) {
      const U32Enc enc(Bits(8), BitsOffset(11, 256), BitsOffset(14, 2304),
                       BitsOffset(30, 18688));
      visitor->U32(enc, 0, &x0);
      visitor->U32(enc, 0, &y0);
      visitor->U32(enc, 0, &xsize);
      visitor->U32(enc, 0, &ysize);
    }

    visitor->Bool(true, &is_last);

    return true;
  }

  // How long to wait [in ticks, see Animation{}] after rendering.
  // May be 0 if the current frame serves as a foundation for a frame with crop.
  uint32_t duration;

  // Optional layer name (UTF-8)
  std::string name;

  NewBase new_base;
  BlendMode blend_mode;

  bool nonserialized_have_timecode = false;
  bool nonserialized_composite_still = false;
  uint32_t timecode;  // 0xHHMMSSFF

  bool have_crop;  // If false, origin/size are zero == "full frame"
  uint32_t x0;
  uint32_t y0;
  uint32_t xsize;
  uint32_t ysize;

  bool is_last;
};

// For decoding to lower resolutions. Cannot mix with animation.
struct Passes : public Fields {
  Passes();
  const char* Name() const override { return "Passes"; }

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override {
    visitor->U32(Val(1), Val(2), Val(3), BitsOffset(3, 4), 1, &num_passes);
    JXL_ASSERT(num_passes <= kMaxNumPasses);  // Cannot happen when reading

    if (visitor->Conditional(num_passes != 1)) {
      visitor->U32(Val(0), Val(1), Val(2), BitsOffset(1, 3), 0,
                   &num_downsample);
      JXL_ASSERT(num_downsample <= 4);  // 1,2,4,8
      if (num_downsample > num_passes) {
        return JXL_FAILURE("num_downsample %u > num_passes %u", num_downsample,
                           num_passes);
      }

      for (uint32_t i = 0; i < num_passes - 1; i++) {
        visitor->Bits(2, 0, &shift[i]);
      }
      shift[num_passes - 1] = 0;

      for (uint32_t i = 0; i < num_downsample; ++i) {
        visitor->U32(Val(1), Val(2), Val(4), Val(8), 1, &downsample[i]);
      }
      for (uint32_t i = 0; i < num_downsample; ++i) {
        visitor->U32(Val(0), Val(1), Val(2), Bits(3), 0, &last_pass[i]);
        if (last_pass[i] >= num_passes) {
          return JXL_FAILURE("last_pass %u >= num_passes %u", last_pass[i],
                             num_passes);
        }
      }
    }

    return true;
  }

  uint32_t num_passes;      // <= kMaxNumPasses
  uint32_t num_downsample;  // <= num_passes

  // Array of num_downsample pairs. downsample=1/last_pass=num_passes-1 and
  // downsample=8/last_pass=0 need not be specified; they are implicit.
  uint32_t downsample[kMaxNumPasses];
  uint32_t last_pass[kMaxNumPasses];
  // Array of shift values for each pass. It is implicitly assumed to be 0 for
  // the last pass.
  uint32_t shift[kMaxNumPasses];
};

// Image/frame := one of more of these, where the last has is_last = true.
// Starts at a byte-aligned address "a"; the next pass starts at "a + size".
struct FrameHeader : public Fields {
  // Optional postprocessing steps. These flags are the source of truth;
  // Override must set/clear them rather than change their meaning. Values
  // chosen such that typical flags == 0 (encoded in only two bits).
  enum Flags {
    // Often but not always off => low bit value:

    // Inject noise into decoded output.
    kNoise = 1,

    // Overlay patches.
    kPatches = 2,

    // 4, 8 = reserved for future sometimes-off

    // Overlay splines.
    kSplines = 16,

    kUseDcFrame = 32,  // Implies kSkipAdaptiveDCSmoothing.

    // 64 = reserved for future often-off

    // Almost always on => negated:

    kSkipAdaptiveDCSmoothing = 128,
  };

  FrameHeader();
  const char* Name() const override { return "FrameHeader"; }

  Status VisitFields(Visitor* JXL_RESTRICT visitor) override {
    if (visitor->AllDefault(*this, &all_default)) {
      // Overwrite all serialized fields, but not any nonserialized_*.
      visitor->SetDefault(this);
      return true;
    }

    // Up to 3 pyramid levels - for up to 4096x downsampling.
    visitor->U32(Val(0), Val(1), Val(2), Val(3), 0, &dc_level);

    if (visitor->Conditional(dc_level == 0)) {
      visitor->Bool(false, &has_animation);
      if (visitor->Conditional(has_animation))
        JXL_RETURN_IF_ERROR(visitor->VisitNested(&animation_frame));
    }

    bool is_modular = (encoding == FrameEncoding::kModular);
    visitor->Bool(false, &is_modular);
    encoding = (is_modular ? FrameEncoding::kModular : FrameEncoding::kVarDCT);

    // In case of xyb_encoded, the options are kXYB or kSRGB. In case of
    // !xyb_encoded, the options are kNone or kYCbCr. The alternate options are
    // respectively kSRGB and kYCbCr.
    bool alternate =
        (nonserialized_xyb_image ? (color_transform == ColorTransform::kSRGB)
                                 : (color_transform == ColorTransform::kYCbCr));
    visitor->Bool(false, &alternate);
    if (nonserialized_xyb_image) {
      color_transform =
          (alternate ? ColorTransform::kSRGB : ColorTransform::kXYB);
    } else {
      color_transform =
          (alternate ? ColorTransform::kYCbCr : ColorTransform::kNone);
    }
    if (visitor->Conditional(color_transform == ColorTransform::kYCbCr)) {
      JXL_RETURN_IF_ERROR(visitor->VisitNested(&chroma_subsampling));
    }
    if (visitor->Conditional(IsLossy() ||
                             encoding == FrameEncoding::kModular)) {
      JXL_RETURN_IF_ERROR(visitor->VisitNested(&passes));
      visitor->U64(0, &flags);
    }
    if (visitor->Conditional(encoding == FrameEncoding::kModular)) {
      visitor->Bits(2, 1, &group_size_shift);
    }
    // This field only makes sense for kVarDCT:
    if (visitor->Conditional(IsLossy())) {
      visitor->U32(Val(0), Val(1), Val(2), Val(3), 1, &x_qm_scale);
    }

    // Save frame as a reference frame.
    if (visitor->Conditional(!animation_frame.is_last && dc_level == 0)) {
      visitor->U32(Val(0), Val(1), Val(2), Val(3), 0, &save_as_reference);
    }
    if (visitor->Conditional(save_as_reference == 0 && dc_level == 0)) {
      visitor->Bool(false, &is_displayed);
    }

    visitor->BeginExtensions(&extensions);
    // Extensions: in chronological order of being added to the format.
    return visitor->EndExtensions();
  }

  bool IsLossy() const { return encoding == FrameEncoding::kVarDCT; }

  // Sets/clears `flag` based upon `condition`.
  void UpdateFlag(const bool condition, const uint64_t flag) {
    if (condition) {
      flags |= flag;
    } else {
      flags &= ~flag;
    }
  }

  // Returns whether this frame is a 'normal' frame that will be displayed on
  // its own (as opposed to a reference/patch or DC level).
  bool IsDisplayed() const {
    return (save_as_reference == 0 || is_displayed) && dc_level == 0;
  }

  mutable bool all_default;

  uint32_t dc_level;

  // Per-frame animation flag - may differ between reference and displayed
  // frames. Default false enables all_default for non-animation frames.
  bool has_animation;
  AnimationFrame animation_frame;

  FrameEncoding encoding;

  ColorTransform color_transform;
  YCbCrChromaSubsampling chroma_subsampling;

  Passes passes;  // only if IsLossy()

  uint64_t flags;

  uint32_t group_size_shift;  // only if encoding == kModular;

  uint32_t x_qm_scale;  // only if IsLossy()

  // 0 = don't save. Otherwise, save as reference frame of ID
  // `save_as_reference`-1.
  uint32_t save_as_reference;  // if !animation_frame.is_last && dc_level == 0.
  bool is_displayed;           // if dc_level == 0 && save_as_reference == 0

  // Must match the xyb_encoded in image metadata
  bool nonserialized_xyb_image = true;

  uint64_t extensions;
};

Status ReadFrameHeader(BitReader* JXL_RESTRICT reader,
                       FrameHeader* JXL_RESTRICT frame);

Status WriteFrameHeader(const FrameHeader& frame,
                        BitWriter* JXL_RESTRICT writer, AuxOut* aux_out);

// Shared by enc/dec. 5F and 13 are by far the most common for d1/2/4/8, 0
// ensures low overhead for small images.
static constexpr U32Enc kOrderEnc =
    U32Enc(Val(0x5F), Val(0x13), Val(0), Bits(kNumOrders));

}  // namespace jxl

#endif  // JXL_FRAME_HEADER_H_
