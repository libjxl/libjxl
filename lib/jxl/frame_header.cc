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

#include "lib/jxl/frame_header.h"

#include "lib/jxl/aux_out.h"
#include "lib/jxl/fields.h"

namespace jxl {

constexpr uint8_t YCbCrChromaSubsampling::kHShift[];
constexpr uint8_t YCbCrChromaSubsampling::kVShift[];

AnimationFrame::AnimationFrame(const ImageMetadata* metadata)
    : nonserialized_image_metadata(metadata) {
  Bundle::Init(this);
}
YCbCrChromaSubsampling::YCbCrChromaSubsampling() { Bundle::Init(this); }
Passes::Passes() { Bundle::Init(this); }
FrameHeader::FrameHeader(const ImageMetadata* metadata)
    : animation_frame(metadata), nonserialized_image_metadata(metadata) {
  Bundle::Init(this);
}

Status ReadFrameHeader(BitReader* JXL_RESTRICT reader,
                       FrameHeader* JXL_RESTRICT frame) {
  return Bundle::Read(reader, frame);
}

Status WriteFrameHeader(const FrameHeader& frame,
                        BitWriter* JXL_RESTRICT writer, AuxOut* aux_out) {
  return Bundle::Write(frame, writer, kLayerHeader, aux_out);
}

UpsamplingMode::UpsamplingMode() { Bundle::Init(this); }

Status UpsamplingMode::VisitFields(Visitor* JXL_RESTRICT visitor) {
  JXL_QUIET_RETURN_IF_ERROR(
      visitor->U32(Val(1), Val(2), Val(4), Val(8), 1, &upsampling_factor));
  if (visitor->Conditional(upsampling_factor != 1)) {
    if (visitor->IsReading() &&
        !(upsampling_factor == 1 || upsampling_factor == 2 ||
          upsampling_factor == 4 || upsampling_factor == 8)) {
      return JXL_FAILURE("Upsampling only possible for factor in {1,2,4,8}");
    }
    JXL_QUIET_RETURN_IF_ERROR(visitor->Bool(true, &default_upsampling_weights));
    if (visitor->Conditional(!default_upsampling_weights)) {
      if (visitor->Conditional(upsampling_factor == 2)) {
        // 4 5x5 kernels, but all of them can be obtained by symmetry from one,
        // which is symmetric along its main diagonal. The top-left kernel is
        // defined by
        //
        // 0  1  2  3  4
        // 1  5  6  7  8
        // 2  6  9 10 11
        // 3  7 10 12 13
        // 4  8 11 13 14
        float constexpr kWeights2[15] = {
            -0.01716200f, -0.03452303f, -0.04022174f, -0.02921014f,
            -0.00624645f, 0.14111091f,  0.28896755f,  0.00278718f,
            -0.01610267f, 0.56661550f,  0.03777607f,  -0.01986694f,
            -0.03144731f, -0.01185068f, -0.00213539f};
        for (size_t i = 0; i < 15; i++) {
          JXL_QUIET_RETURN_IF_ERROR(
              visitor->F16(kWeights2[i], &upsampling2_weights[i]));
        }
      }
      if (visitor->Conditional(upsampling_factor == 4)) {
        // 16 5x5 kernels, but all of them can be obtained by symmetry from
        // three, two of which are symmetric along their main diagonals. The top
        // left 4 kernels are defined by
        //
        // 0  1  2  3  4   5  6  7  8  9
        // 1 10 11 12 13  14 15 16 17 18
        // 2 11 19 20 21  22 23 24 25 26
        // 3 12 20 27 28  29 30 31 32 33
        // 4 13 21 28 34  35 36 37 38 39
        //
        // 5 14 22 29 35  40 41 42 43 44
        // 6 15 23 30 36  41 45 46 47 48
        // 7 16 24 31 37  42 46 49 50 51
        // 8 17 25 32 38  43 47 50 52 53
        // 9 18 26 33 39  44 48 51 53 54
        constexpr float kWeights4[55] = {
            -0.02419067f, -0.03491987f, -0.03693351f, -0.03094285f,
            -0.00529785f, -0.01663432f, -0.03556863f, -0.03888905f,
            -0.03516850f, -0.00989469f, 0.23651958f,  0.33392945f,
            -0.01073543f, -0.01313181f, -0.03556694f, 0.13048175f,
            0.40103025f,  0.03951150f,  -0.02077584f, 0.46914198f,
            -0.00209270f, -0.01484589f, -0.04064806f, 0.18942530f,
            0.56279892f,  0.06674400f,  -0.02335494f, -0.03551682f,
            -0.00754830f, -0.02267919f, -0.02363578f, 0.00315804f,
            -0.03399098f, -0.01359519f, -0.00091653f, -0.00335467f,
            -0.01163294f, -0.01610294f, -0.00974088f, -0.00191622f,
            -0.01095446f, -0.03198464f, -0.04455121f, -0.02799790f,
            -0.00645912f, 0.06390599f,  0.22963888f,  0.00630981f,
            -0.01897349f, 0.67537268f,  0.08483369f,  -0.02534994f,
            -0.02205197f, -0.01667999f, -0.00384443f};
        for (size_t i = 0; i < 55; i++) {
          JXL_QUIET_RETURN_IF_ERROR(
              visitor->F16(kWeights4[i], &upsampling4_weights[i]));
        }
      }
      if (visitor->Conditional(upsampling_factor == 8)) {
        // 64 5x5 kernels, all of them can be obtained by symmetry from
        // 10, 4 of which are symmetric along their main diagonals. The top
        // left 16 kernels are defined by
        //  0  1  2  3  4   5  6  7  8  9   a  b  c  d  e   f 10 11 12 13
        //  1 14 15 16 17  18 19 1a 1b 1c  1d 1e 1f 20 21  22 23 24 25 26
        //  2 15 27 28 29  2a 2b 2c 2d 2e  2f 30 31 32 33  34 35 36 37 38
        //  3 16 28 39 3a  3b 3c 3d 3e 3f  40 41 42 43 44  45 46 47 48 49
        //  4 17 29 3a 4a  4b 4c 4d 4e 4f  50 51 52 53 54  55 56 57 58 59

        //  5 18 2a 3b 4b  5a 5b 5c 5d 5e  5f 60 61 62 63  64 65 66 67 68
        //  6 19 2b 3c 4c  5b 69 6a 6b 6c  6d 6e 6f 70 71  72 73 74 75 76
        //  7 1a 2c 3d 4d  5c 6a 77 78 79  7a 7b 7c 7d 7e  7f 80 81 82 83
        //  8 1b 2d 3e 4e  5d 6b 78 84 85  86 87 88 89 8a  8b 8c 8d 8e 8f
        //  9 1c 2e 3f 4f  5e 6c 79 85 90  91 92 93 94 95  96 97 98 99 9a

        //  a 1d 2f 40 50  5f 6d 7a 86 91  9b 9c 9d 9e 9f  a0 a1 a2 a3 a4
        //  b 1e 30 41 51  60 6e 7b 87 92  9c a5 a6 a7 a8  a9 aa ab ac ad
        //  c 1f 31 42 52  61 6f 7c 88 93  9d a6 ae af b0  b1 b2 b3 b4 b5
        //  d 20 32 43 53  62 70 7d 89 94  9e a7 af b6 b7  b8 b9 ba bb bc
        //  e 21 33 44 54  63 71 7e 8a 95  9f a8 b0 b7 bd  be bf c0 c1 c2

        //  f 22 34 45 55  64 72 7f 8b 96  a0 a9 b1 b8 be  c3 c4 c5 c6 c7
        // 10 23 35 46 56  65 73 80 8c 97  a1 aa b2 b9 bf  c4 c8 c9 ca cb
        // 11 24 36 47 57  66 74 81 8d 98  a2 ab b3 ba c0  c5 c9 cc cd ce
        // 12 25 37 48 58  67 75 82 8e 99  a3 ac b4 bb c1  c6 ca cd cf d0
        // 13 26 38 49 59  68 76 83 8f 9a  a4 ad b5 bc c2  c7 cb ce d0 d1
        constexpr float kWeights8[210] = {
            -0.02928613f, -0.03706353f, -0.03783812f, -0.03324558f,
            -0.00447632f, -0.02519406f, -0.03752601f, -0.03901508f,
            -0.03663285f, -0.00646649f, -0.02066407f, -0.03838633f,
            -0.04002101f, -0.03900035f, -0.00901973f, -0.01626393f,
            -0.03954148f, -0.04046620f, -0.03979621f, -0.01224485f,
            0.29895328f,  0.35757708f,  -0.02447552f, -0.01081748f,
            -0.04314594f, 0.23903219f,  0.41119301f,  -0.00573046f,
            -0.01450239f, -0.04246845f, 0.17567618f,  0.45220643f,
            0.02287757f,  -0.01936783f, -0.03583255f, 0.11572472f,
            0.47416733f,  0.06284440f,  -0.02685066f, 0.42720050f,
            -0.02248939f, -0.01155273f, -0.04562755f, 0.28689496f,
            0.49093869f,  -0.00007891f, -0.01545926f, -0.04562659f,
            0.21238920f,  0.53980934f,  0.03369474f,  -0.02070211f,
            -0.03866988f, 0.14229550f,  0.56593398f,  0.08045181f,
            -0.02888298f, -0.03680918f, -0.00542229f, -0.02920477f,
            -0.02788574f, -0.02118180f, -0.03942402f, -0.00775547f,
            -0.02433614f, -0.03193943f, -0.02030828f, -0.04044014f,
            -0.01074016f, -0.01930822f, -0.03620399f, -0.01974125f,
            -0.03919545f, -0.01456093f, -0.00045072f, -0.00360110f,
            -0.01020207f, -0.01231907f, -0.00638988f, -0.00071592f,
            -0.00279122f, -0.00957115f, -0.01288327f, -0.00730937f,
            -0.00107783f, -0.00210156f, -0.00890705f, -0.01317668f,
            -0.00813895f, -0.00153491f, -0.02128481f, -0.04173044f,
            -0.04831487f, -0.03293190f, -0.00525260f, -0.01720322f,
            -0.04052736f, -0.05045706f, -0.03607317f, -0.00738030f,
            -0.01341764f, -0.03965629f, -0.05151616f, -0.03814886f,
            -0.01005819f, 0.18968273f,  0.33063684f,  -0.01300105f,
            -0.01372950f, -0.04017465f, 0.13727832f,  0.36402234f,
            0.01027890f,  -0.01832107f, -0.03365072f, 0.08734506f,
            0.38194295f,  0.04338228f,  -0.02525993f, 0.56408126f,
            0.00458352f,  -0.01648227f, -0.04887868f, 0.24585519f,
            0.62026135f,  0.04314807f,  -0.02213737f, -0.04158014f,
            0.16637289f,  0.65027023f,  0.09621636f,  -0.03101388f,
            -0.04082742f, -0.00904519f, -0.02790922f, -0.02117818f,
            0.00798662f,  -0.03995711f, -0.01243427f, -0.02231705f,
            -0.02946266f, 0.00992055f,  -0.03600283f, -0.01684920f,
            -0.00111684f, -0.00411204f, -0.01297130f, -0.01723725f,
            -0.01022545f, -0.00165306f, -0.00313110f, -0.01218016f,
            -0.01763266f, -0.01125620f, -0.00231663f, -0.01374149f,
            -0.03797620f, -0.05142937f, -0.03117307f, -0.00581914f,
            -0.01064003f, -0.03608089f, -0.05272168f, -0.03375670f,
            -0.00795586f, 0.09628104f,  0.27129991f,  -0.00353779f,
            -0.01734151f, -0.03153981f, 0.05686230f,  0.28500998f,
            0.02230594f,  -0.02374955f, 0.68214326f,  0.05018048f,
            -0.02320852f, -0.04383616f, 0.18459474f,  0.71517975f,
            0.10805613f,  -0.03263677f, -0.03637639f, -0.01394373f,
            -0.02511203f, -0.01728636f, 0.05407331f,  -0.02867568f,
            -0.01893131f, -0.00240854f, -0.00446511f, -0.01636187f,
            -0.02377053f, -0.01522848f, -0.00333334f, -0.00819975f,
            -0.02964169f, -0.04499287f, -0.02745350f, -0.00612408f,
            0.02727416f,  0.19446600f,  0.00159832f,  -0.02232473f,
            0.74982506f,  0.11452620f,  -0.03348048f, -0.01605681f,
            -0.02070339f, -0.00458223f};
        for (size_t i = 0; i < 210; i++) {
          JXL_QUIET_RETURN_IF_ERROR(
              visitor->F16(kWeights8[i], &upsampling8_weights[i]));
        }
      }
    }
  }
  return true;
}

Status ExtraChannelAnimation::VisitFields(Visitor* JXL_RESTRICT visitor) {
  // This function assumes that nonserialized_image_metadata and that there
  // are 1 or more extra channels, if not this Field should not be serialized
  // all (not even its all_default field) so this function is not called then.
  const std::vector<ExtraChannelInfo>& extra_channels =
      nonserialized_animation_frame->nonserialized_image_metadata->m2
          .extra_channel_info;

  if (visitor->AllDefault(*this, &all_default)) {
    // All default here means that the blend mode values of all extra channels
    // match those of the color channels.
    visitor->SetDefault(this);
    return true;
  }

  if (new_base.size() != extra_channels.size()) {
    new_base.resize(extra_channels.size(),
                    nonserialized_animation_frame->new_base);
  }
  if (blend_mode.size() != extra_channels.size()) {
    blend_mode.resize(extra_channels.size(),
                      nonserialized_animation_frame->blend_mode);
  }

  for (size_t i = 0; i < new_base.size(); i++) {
    JXL_QUIET_RETURN_IF_ERROR(VisitNewBase(
        visitor, nonserialized_animation_frame->new_base, &new_base[i]));
  }
  for (size_t i = 0; i < blend_mode.size(); i++) {
    JXL_QUIET_RETURN_IF_ERROR(VisitBlendMode(
        visitor, nonserialized_animation_frame->blend_mode, &blend_mode[i]));
  }

  for (size_t i = 0; i < blend_channel.size(); i++) {
    bool blend_uses_alpha = blend_mode[i] == BlendMode::kBlend ||
                            blend_mode[i] == BlendMode::kAlphaWeightedAdd;
    if (visitor->Conditional(extra_channels.size() > 1 && blend_uses_alpha)) {
      JXL_QUIET_RETURN_IF_ERROR(visitor->U32(
          Val(0), Val(1), BitsOffset(2, 2), BitsOffset(8, 6),
          nonserialized_animation_frame->blend_channel, &blend_channel[i]));
    }
  }
  return true;
}

}  // namespace jxl
