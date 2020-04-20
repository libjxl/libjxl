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

#ifndef JXL_DCT_SCALES_H_
#define JXL_DCT_SCALES_H_

// Scaling factors.

#include <stddef.h>

namespace jxl {

// Final scaling factors of outputs/inputs in the Arai, Agui, and Nakajima
// algorithm computing the DCT/IDCT (described in the book JPEG: Still Image
// Data Compression Standard, section 4.3.5) and the "A low multiplicative
// complexity fast recursive DCT-2 algorithm" (Maxim Vashkevich, Alexander
// Pertrovsky) algorithm. Note that the DCT and the IDCT scales of these two
// algorithms are flipped. We use the first algorithm for DCT8, and the second
// one for all other DCTs.
/* Python snippet to produce these tables for the Arai, Agui, Nakajima
 * algorithm:
 *
from mpmath import *
N = 8
def iscale(u):
  eps = sqrt(mpf(0.5)) if u == 0 else mpf(1.0)
  return sqrt(mpf(2) / mpf(N)) * eps * cos(mpf(u) * pi / mpf(2 * N))
def scale(u):
  return mpf(1) / (mpf(N) * iscale(i))
mp.dps = 18
print(", ".join([str(scale(i)) + 'f' for i in range(N)]))
print(", ".join([str(iscale(i)) + 'f' for i in range(N)]))
 */
static constexpr float kDCTScales1[1] = {1.0f};
static constexpr float kIDCTScales1[1] = {1.0f};
static constexpr float kDCTScales2[2] = {0.707106781186547524f,
                                         0.707106781186547524f};
static constexpr float kIDCTScales2[2] = {0.707106781186547524f,
                                          0.707106781186547524f};
static constexpr float kDCTScales4[4] = {0.5f, 0.653281482438188264f, 0.5f,
                                         0.270598050073098492f};
static constexpr float kIDCTScales4[4] = {0.5f, 0.382683432365089772f, 0.5f,
                                          0.923879532511286756f};
static constexpr float kDCTScales8[8] = {
    0.353553390593273762f, 0.254897789552079584f, 0.270598050073098492f,
    0.30067244346752264f,  0.353553390593273762f, 0.449988111568207852f,
    0.653281482438188264f, 1.28145772387075309f};

static constexpr float kIDCTScales8[8] = {
    0.353553390593273762f, 0.490392640201615225f, 0.461939766255643378f,
    0.415734806151272619f, 0.353553390593273762f, 0.277785116509801112f,
    0.191341716182544886f, 0.0975451610080641339f};

static constexpr float kIDCTScales16[16] = {0.25f,
                                            0.177632042131274808f,
                                            0.180239955501736978f,
                                            0.184731156892216368f,
                                            0.191341716182544886f,
                                            0.200444985785954314f,
                                            0.212607523691814112f,
                                            0.228686034616512494f,
                                            0.25f,
                                            0.278654739432954475f,
                                            0.318189645143208485f,
                                            0.375006192208515097f,
                                            0.461939766255643378f,
                                            0.608977011699708658f,
                                            0.906127446352887843f,
                                            1.80352839005774887f};

static constexpr float kDCTScales16[16] = {0.25f,
                                           0.351850934381595615f,
                                           0.346759961330536865f,
                                           0.33832950029358817f,
                                           0.326640741219094132f,
                                           0.311806253246667808f,
                                           0.293968900604839679f,
                                           0.273300466750439372f,
                                           0.25f,
                                           0.224291896585659071f,
                                           0.196423739596775545f,
                                           0.166663914619436624f,
                                           0.135299025036549246f,
                                           0.102631131880589345f,
                                           0.0689748448207357531f,
                                           0.0346542922997728657f};

static constexpr float kIDCTScales32[32] = {
    0.176776695296636881f, 0.125150749558799075f, 0.125604821547038926f,
    0.126367739974385915f, 0.127448894776039792f, 0.128861827480656137f,
    0.13062465373492222f,  0.132760647772446044f, 0.135299025036549246f,
    0.138275974008611132f, 0.141736008704089426f, 0.145733742051533468f,
    0.15033622173376132f,  0.155626030758916204f, 0.161705445839997532f,
    0.168702085363751436f, 0.176776695296636881f, 0.186134067750574612f,
    0.197038655862812556f, 0.20983741135388176f,  0.224994055784103926f,
    0.243142059465490173f, 0.265169421497586868f, 0.292359983358221239f,
    0.326640741219094132f, 0.371041154078541569f, 0.430611774559583482f,
    0.514445252488352888f, 0.640728861935376545f, 0.851902104617179697f,
    1.27528715467229096f,  2.5475020308870142f};

static constexpr float kDCTScales32[32] = {
    0.176776695296636881f,  0.249698864051293098f,  0.248796181668049222f,
    0.247294127491195243f,  0.245196320100807612f,  0.242507813298635998f,
    0.239235083933052216f,  0.235386016295755195f,  0.230969883127821689f,
    0.225997323280860833f,  0.220480316087088757f,  0.214432152500068017f,
    0.207867403075636309f,  0.200801882870161227f,  0.19325261334068424f,
    0.185237781338739773f,  0.176776695296636881f,  0.1678897387117546f,
    0.158598321040911375f,  0.148924826123108336f,  0.138892558254900556f,
    0.128525686048305432f,  0.117849184206499412f,  0.106888773357570524f,
    0.0956708580912724429f, 0.0842224633480550127f, 0.0725711693136155919f,
    0.0607450449758159725f, 0.048772580504032067f,  0.0366826186138404379f,
    0.0245042850823901505f, 0.0122669185818545036f};

// TODO(veluca): switch to struct template
template <size_t N>
constexpr const float* DCTScales() {
  return N == 1 ? kDCTScales1
                : (N == 2 ? kDCTScales2
                          : (N == 4 ? kDCTScales4
                                    : (N == 8 ? kDCTScales8
                                              : (N == 16 ? kDCTScales16
                                                         : kDCTScales32))));
}

template <size_t N>
constexpr const float* IDCTScales() {
  return N == 1 ? kIDCTScales1
                : (N == 2 ? kIDCTScales2
                          : (N == 4 ? kIDCTScales4
                                    : (N == 8 ? kIDCTScales8
                                              : (N == 16 ? kIDCTScales16
                                                         : kIDCTScales32))));
}

// For n != 0, the n-th basis function of a N-DCT, evaluated in pixel k, has a
// value of cos((k+1/2) n/(2N) pi). When downsampling by 2x, we average
// the values for pixel k and k+1 to get the value for pixel (k/2), thus we get
//
// [cos((k+1/2) n/N pi) + cos((k+3/2) n/N pi)]/2 =
// cos(n/(2N) pi) cos((k+1) n/N pi) =
// cos(n/(2N) pi) cos(((k/2)+1/2) n/(N/2) pi)
//
// which is exactly the same as the value of pixel k/2 of a N/2-sized DCT,
// except for the cos(n/(2N) pi) scaling factor (which does *not*
// depend on the pixel). Thus, when using the lower-frequency coefficients of a
// DCT-N to compute a DCT-(N/2), they should be scaled by this constant. Scaling
// factors for a DCT-(N/4) etc can then be obtained by successive
// multiplications. The structs below contain the above-mentioned scaling
// factors.
template <size_t FROM, size_t TO>
struct DCTResampleScales;

template <>
struct DCTResampleScales<8, 1> {
  static constexpr float kScales[1] = {
      1.000000000000000000,
  };
};

template <>
struct DCTResampleScales<8, 2> {
  static constexpr float kScales[2] = {
      1.000000000000000000,
      0.906127446352887778,
  };
};

template <>
struct DCTResampleScales<16, 2> {
  static constexpr float kScales[2] = {
      1.000000000000000000,
      0.901764195028874394,
  };
};

template <>
struct DCTResampleScales<16, 4> {
  static constexpr float kScales[4] = {
      1.000000000000000000,
      0.976062531202202877,
      0.906127446352887778,
      0.795666809947927156,
  };
};

template <>
struct DCTResampleScales<32, 4> {
  static constexpr float kScales[4] = {
      1.000000000000000000,
      0.974886821136879522,
      0.901764195028874394,
      0.787054918159101335,
  };
};

template <>
struct DCTResampleScales<32, 8> {
  static constexpr float kScales[8] = {
      1.000000000000000000, 0.993985983084976765, 0.976062531202202877,
      0.946582901544112176, 0.906127446352887778, 0.855491189274751540,
      0.795666809947927156, 0.727823404688121345,
  };
};

// Inverses of the above.
template <>
struct DCTResampleScales<1, 8> {
  static constexpr float kScales[1] = {
      1.000000000000000000,
  };
};

template <>
struct DCTResampleScales<2, 8> {
  static constexpr float kScales[2] = {
      1.000000000000000000,
      1.103597517131772232,
  };
};

template <>
struct DCTResampleScales<2, 16> {
  static constexpr float kScales[2] = {
      1.000000000000000000,
      1.108937353592731823,
  };
};

template <>
struct DCTResampleScales<4, 16> {
  static constexpr float kScales[4] = {
      1.000000000000000000,
      1.024524523821556565,
      1.103597517131772232,
      1.256807482098500017,
  };
};

template <>
struct DCTResampleScales<4, 32> {
  static constexpr float kScales[4] = {
      1.000000000000000000,
      1.025760096781116015,
      1.108937353592731823,
      1.270559368765487251,
  };
};

template <>
struct DCTResampleScales<8, 32> {
  static constexpr float kScales[8] = {
      1.000000000000000000, 1.006050404147911470, 1.024524523821556565,
      1.056431505754806377, 1.103597517131772232, 1.168919110491081437,
      1.256807482098500017, 1.373959663235216677,
  };
};

template <size_t V>
struct square_root {
  static constexpr float value =
      square_root<V / 2>::value * 1.4142135623730951f;
};

template <>
struct square_root<1> {
  static constexpr float value = 1.0f;
};

// Apply the DCT algorithm-intrinsic constants to DCTResampleScale.
// Note that the DCTScales constants give results that are scaled with a factor
// proportional to 1/sqrt(N), so we counteract that here.
// We also use the fact that 1/(sqrt(N) DCTScales(N)) == sqrt(N) IDCTScales(N)
// to avoid a division.
template <size_t FROM, size_t TO>
constexpr float DCTTotalResampleScale(size_t x) {
  return square_root<FROM>::value * DCTScales<FROM>()[x] *
         square_root<TO>::value * IDCTScales<TO>()[x] *
         DCTResampleScales<FROM, TO>::kScales[x];
}

}  // namespace jxl

#endif  // JXL_DCT_SCALES_H_
