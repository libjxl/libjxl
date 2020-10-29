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

#ifndef LIB_JXL_SPLINES_FASTMATH_H_
#define LIB_JXL_SPLINES_FASTMATH_H_

#include <cmath>

#include "lib/jxl/common.h"  // kPi

namespace jxl {
namespace splines_internal {

template <int N>
static constexpr float InvSqrt();
template <>
constexpr float InvSqrt<2>() {
  return 0.7071067811865475f;
}
template <>
constexpr float InvSqrt<4>() {
  return 0.5f;
}
template <>
constexpr float InvSqrt<8>() {
  return 0.35355339059327373f;
}
template <>
constexpr float InvSqrt<16>() {
  return 0.25f;
}
template <>
constexpr float InvSqrt<32>() {
  return 0.17677669529663687f;
}

template <int Size>
static float InterpolateLUT(const float LUT[Size], const float i) {
  if (i >= Size - 1) {
    return LUT[Size - 1];
  }
  const int i_int = static_cast<int>(i);
  const float i_frac = i - i_int;
  return (1 - i_frac) * LUT[i_int] + i_frac * LUT[i_int + 1];
}

// kCosLUT[i] = cos(i * (kPi / 2) / 63)
static constexpr float kCosLUT[64] = {
    1.f,
    0.999689182000816f,
    0.998756921218922f,
    0.99720379718118f,
    0.995030775365401f,
    0.992239206600172f,
    0.988830826225129f,
    0.984807753012208f,
    0.980172487848544f,
    0.974927912181824f,
    0.969077286229078f,
    0.962624246950012f,
    0.955572805786141f,
    0.947927346167132f,
    0.939692620785908f,
    0.930873748644204f,
    0.921476211870408f,
    0.911505852311673f,
    0.900968867902419f,
    0.889871808811469f,
    0.878221573370229f,
    0.866025403784439f,
    0.853290881632156f,
    0.840025923150772f,
    0.826238774315995f,
    0.811938005715857f,
    0.797132507222923f,
    0.78183148246803f,
    0.766044443118978f,
    0.749781202967734f,
    0.733051871829826f,
    0.715866849259718f,
    0.698236818086073f,
    0.680172737770919f,
    0.661685837596859f,
    0.642787609686539f,
    0.623489801858734f,
    0.603804410325477f,
    0.58374367223479f,
    0.563320058063622f,
    0.542546263865759f,
    0.521435203379498f,
    0.5f,
    0.478253978621318f,
    0.456210657353163f,
    0.433883739117558f,
    0.411287103130612f,
    0.388434796274695f,
    0.365341024366395f,
    0.342020143325669f,
    0.318486650251684f,
    0.294755174410904f,
    0.270840468143005f,
    0.246757397690294f,
    0.222520933956314f,
    0.198146143199398f,
    0.17364817766693f,
    0.149042266176175f,
    0.124343704647485f,
    0.0995678465958167f,
    0.0747300935864244f,
    0.0498458856606972f,
    0.024930691738073f,
    0.f,
};

static constexpr float InaccurateFMod(const float x, const float y) {
  return x - static_cast<int>(x / y) * y;
}

static inline float Cos(float x) {
  if (x < 0) x = -x;
  if (x >= 2 * kPi) x = InaccurateFMod(x, 2 * kPi);
  if (x > kPi) x = 2 * kPi - x;
  bool opposite = false;
  if (x > kPi / 2) {
    x = kPi - x;
    opposite = true;
  }
  float result =
      InterpolateLUT<64>(kCosLUT, x * static_cast<float>(63 / (kPi / 2)));
  if (opposite) result = -result;
  return result;
}

// kErfLUT[i] = erf(2 * i / 63)
static constexpr float kErfLUT[64] = {
    0.f,
    0.0358095307155359f,
    0.0715469677785195f,
    0.107140652672516f,
    0.142519792779007f,
    0.177614883452066f,
    0.212358117215062f,
    0.246683776068649f,
    0.280528603134456f,
    0.313832150144761f,
    0.346537097619992f,
    0.378589544947352f,
    0.409939267978589f,
    0.440539942196093f,
    0.470349329946658f,
    0.499329430703871f,
    0.52744659378564f,
    0.554671593415279f,
    0.580979666465684f,
    0.606350513659535f,
    0.630768265407845f,
    0.65422141384884f,
    0.676702712994044f,
    0.698209049194451f,
    0.718741284403263f,
    0.738304074930581f,
    0.756905668557882f,
    0.774557683005487f,
    0.791274868824678f,
    0.807074859818627f,
    0.821977914084574f,
    0.836006648716235f,
    0.849185771113207f,
    0.861541809716806f,
    0.873102846833217f,
    0.883898256019451f,
    0.893958446299829f,
    0.903314615255194f,
    0.911998512788506f,
    0.92004221712334f,
    0.927477924340631f,
    0.934337752507907f,
    0.94065356120808f,
    0.946456787035268f,
    0.951778295396072f,
    0.956648248739137f,
    0.961095991135758f,
    0.965149948951748f,
    0.968837547187f,
    0.972185140915123f,
    0.975217961131759f,
    0.977960074216708f,
    0.980434354131662f,
    0.982662466411504f,
    0.984664862961992f,
    0.986460786649043f,
    0.988068284653553f,
    0.989504229569219f,
    0.990784347237614f,
    0.991923250343289f,
    0.992934476830092f,
    0.993830532246756f,
    0.994622935183356f,
    0.995322265018953f,
};

static inline float Erf(float x) {
  bool opposite = false;
  if (x < 0) {
    x = -x;
    opposite = true;
  }
  float result;
  if (x > 2) {
    result = 1.f;
  } else {
    result = InterpolateLUT<64>(kErfLUT, x * (.5f * 63));
  }
  if (opposite) result = -result;
  return result;
}

}  // namespace splines_internal
}  // namespace jxl

#endif  // LIB_JXL_SPLINES_FASTMATH_H_
