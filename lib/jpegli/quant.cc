// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jpegli/quant.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "lib/jpegli/adaptive_quantization.h"
#include "lib/jpegli/common.h"
#include "lib/jpegli/encode_internal.h"
#include "lib/jpegli/error.h"
#include "lib/jpegli/memory_manager.h"
#include "lib/jxl/base/byte_order.h"
#include "lib/jxl/base/status.h"

namespace jpegli {

namespace {
static constexpr float kBaseQuantMatrixXYB[] = {
    // c = 0
    7.5629935265f,
    19.8247814178f,
    22.5724945068f,
    20.6706695557f,
    22.6864585876f,
    23.5696277618f,
    25.8129081726f,
    36.3307571411f,
    19.8247814178f,
    21.5503177643f,
    19.9372234344f,
    20.5424213409f,
    21.8645496368f,
    23.9041385651f,
    28.2844066620f,
    32.6609764099f,
    22.5724945068f,
    19.9372234344f,
    21.9017257690f,
    19.1223449707f,
    21.7515811920f,
    24.6724700928f,
    25.4249649048f,
    32.6653823853f,
    20.6706695557f,
    20.5424213409f,
    19.1223449707f,
    20.1610221863f,
    25.3719692230f,
    25.9668903351f,
    30.9804954529f,
    31.3406009674f,
    22.6864585876f,
    21.8645496368f,
    21.7515811920f,
    25.3719692230f,
    26.2431850433f,
    40.5992202759f,
    43.2624626160f,
    63.3010940552f,
    23.5696277618f,
    23.9041385651f,
    24.6724700928f,
    25.9668903351f,
    40.5992202759f,
    48.3026771545f,
    34.0964355469f,
    61.9852142334f,
    25.8129081726f,
    28.2844066620f,
    25.4249649048f,
    30.9804954529f,
    43.2624626160f,
    34.0964355469f,
    34.4937438965f,
    66.9702758789f,
    36.3307571411f,
    32.6609764099f,
    32.6653823853f,
    31.3406009674f,
    63.3010940552f,
    61.9852142334f,
    66.9702758789f,
    39.9652709961f,
    // c = 1
    1.6262000799f,
    3.2199242115f,
    3.4903779030f,
    3.9148359299f,
    4.8337211609f,
    4.9108843803f,
    5.3137121201f,
    6.1676793098f,
    3.2199242115f,
    3.4547898769f,
    3.6036829948f,
    4.2652835846f,
    4.8368387222f,
    4.8226222992f,
    5.6120514870f,
    6.3431472778f,
    3.4903779030f,
    3.6036829948f,
    3.9044559002f,
    4.3374395370f,
    4.8435096741f,
    5.4057979584f,
    5.6066360474f,
    6.1075134277f,
    3.9148359299f,
    4.2652835846f,
    4.3374395370f,
    4.6064834595f,
    5.1751475334f,
    5.4013924599f,
    6.0399808884f,
    6.7825231552f,
    4.8337211609f,
    4.8368387222f,
    4.8435096741f,
    5.1751475334f,
    5.3748049736f,
    6.1410837173f,
    7.6529307365f,
    7.5235214233f,
    4.9108843803f,
    4.8226222992f,
    5.4057979584f,
    5.4013924599f,
    6.1410837173f,
    6.3431472778f,
    7.1083049774f,
    7.6008300781f,
    5.3137121201f,
    5.6120514870f,
    5.6066360474f,
    6.0399808884f,
    7.6529307365f,
    7.1083049774f,
    7.0943155289f,
    7.0478363037f,
    6.1676793098f,
    6.3431472778f,
    6.1075134277f,
    6.7825231552f,
    7.5235214233f,
    7.6008300781f,
    7.0478363037f,
    6.9186143875f,
    // c = 2
    3.3038473129f,
    10.0689258575f,
    12.2785224915f,
    14.6041173935f,
    16.2107315063f,
    19.2314529419f,
    28.0129547119f,
    55.6682891846f,
    10.0689258575f,
    11.4085016251f,
    11.3871345520f,
    15.4934167862f,
    16.5364933014f,
    14.9153423309f,
    26.3748722076f,
    40.8614425659f,
    12.2785224915f,
    11.3871345520f,
    17.0886878967f,
    13.9500350952f,
    16.0003223419f,
    28.5660629272f,
    26.2124195099f,
    30.1260128021f,
    14.6041173935f,
    15.4934167862f,
    13.9500350952f,
    21.1235027313f,
    26.1579780579f,
    25.5579223633f,
    40.6859359741f,
    33.8056335449f,
    16.2107315063f,
    16.5364933014f,
    16.0003223419f,
    26.1579780579f,
    26.8042831421f,
    26.1587715149f,
    35.7343978882f,
    43.6857032776f,
    19.2314529419f,
    14.9153423309f,
    28.5660629272f,
    25.5579223633f,
    26.1587715149f,
    34.5418128967f,
    41.3197937012f,
    48.7867660522f,
    28.0129547119f,
    26.3748722076f,
    26.2124195099f,
    40.6859359741f,
    35.7343978882f,
    41.3197937012f,
    47.6329460144f,
    55.3498458862f,
    55.6682891846f,
    40.8614425659f,
    30.1260128021f,
    33.8056335449f,
    43.6857032776f,
    48.7867660522f,
    55.3498458862f,
    63.6065597534f,
};

static const float kBaseQuantMatrixYCbCr[] = {
    // c = 0
    1.4076321125f,
    2.6927082539f,
    2.6927735806f,
    2.9220938683f,
    3.0870633125f,
    3.4968640804f,
    3.5730612278f,
    3.5978596210f,
    2.6927082539f,
    2.6926636696f,
    2.7195601463f,
    2.9238407612f,
    3.1882488728f,
    3.0607142448f,
    3.1882314682f,
    3.8304426670f,
    2.6927735806f,
    2.7195601463f,
    2.9532215595f,
    3.5562388897f,
    3.7088179588f,
    3.0576279163f,
    3.7443304062f,
    4.2484717369f,
    2.9220938683f,
    2.9238407612f,
    3.5562388897f,
    3.0594384670f,
    4.1780085564f,
    4.9221563339f,
    4.7842588425f,
    4.6059336662f,
    3.0870633125f,
    3.1882488728f,
    3.7088179588f,
    4.1780085564f,
    4.3475294113f,
    5.5422372818f,
    5.5741071701f,
    5.4531836510f,
    3.4968640804f,
    3.0607142448f,
    3.0576279163f,
    4.9221563339f,
    5.5422372818f,
    5.4393601418f,
    5.1039180756f,
    6.0990614891f,
    3.5730612278f,
    3.1882314682f,
    3.7443304062f,
    4.7842588425f,
    5.5741071701f,
    5.1039180756f,
    5.4144043922f,
    5.4524297714f,
    3.5978596210f,
    3.8304426670f,
    4.2484717369f,
    4.6059336662f,
    5.4531836510f,
    6.0990614891f,
    5.4524297714f,
    4.3595433235f,
    // c = 1
    2.8152642250f,
    10.4298934937f,
    16.1451492310f,
    15.3725156784f,
    17.6543502808f,
    19.1104965210f,
    17.5021877289f,
    29.5177459717f,
    10.4298934937f,
    15.7448558807f,
    16.8441677094f,
    15.3214502335f,
    17.5918464661f,
    16.8787574768f,
    27.0867996216f,
    21.3443832397f,
    16.1451492310f,
    16.8441677094f,
    14.7525558472f,
    18.0765247345f,
    18.2206096649f,
    23.2126445770f,
    98.1291885376f,
    23.6039886475f,
    15.3725156784f,
    15.3214502335f,
    18.0765247345f,
    17.2925109863f,
    16.1435356140f,
    24.0464611053f,
    27.1577339172f,
    35.3269882202f,
    17.6543502808f,
    17.5918464661f,
    18.2206096649f,
    16.1435356140f,
    19.2819595337f,
    16.2939300537f,
    19.6862888336f,
    51.0941123962f,
    19.1104965210f,
    16.8787574768f,
    23.2126445770f,
    24.0464611053f,
    16.2939300537f,
    32.3153648376f,
    45.7272338867f,
    64.6245880127f,
    17.5021877289f,
    27.0867996216f,
    98.1291885376f,
    27.1577339172f,
    19.6862888336f,
    45.7272338867f,
    61.8331909180f,
    85.0626754761f,
    29.5177459717f,
    21.3443832397f,
    23.6039886475f,
    35.3269882202f,
    51.0941123962f,
    64.6245880127f,
    85.0626754761f,
    112.7605514526f,
    // c = 2
    2.8152642250f,
    5.4735932350f,
    7.3637795448f,
    6.5195322037f,
    8.1501169205f,
    8.7243938446f,
    8.7219915390f,
    9.3618907928f,
    5.4735932350f,
    7.1514792442f,
    7.2054982185f,
    8.1126995087f,
    8.1497650146f,
    7.1335659027f,
    7.8453893661f,
    8.3512821198f,
    7.3637795448f,
    7.2054982185f,
    6.9224662781f,
    8.0766754150f,
    9.1168527603f,
    7.3714752197f,
    7.3646650314f,
    8.6790895462f,
    6.5195322037f,
    8.1126995087f,
    8.0766754150f,
    7.8294739723f,
    7.7385902405f,
    7.8628563881f,
    7.4404106140f,
    8.4759435654f,
    8.1501169205f,
    8.1497650146f,
    9.1168527603f,
    7.7385902405f,
    7.0960793495f,
    8.9185447693f,
    8.2047510147f,
    7.8465061188f,
    8.7243938446f,
    7.1335659027f,
    7.3714752197f,
    7.8628563881f,
    8.9185447693f,
    8.6063842773f,
    9.7156696320f,
    64.6700744629f,
    8.7219915390f,
    7.8453893661f,
    7.3646650314f,
    7.4404106140f,
    8.2047510147f,
    9.7156696320f,
    61.9934043884f,
    83.2930450439f,
    9.3618907928f,
    8.3512821198f,
    8.6790895462f,
    8.4759435654f,
    7.8465061188f,
    64.6700744629f,
    83.2930450439f,
    113.0502548218f,
};

static const float kBaseQuantMatrixStd[] = {
    // c = 0
    16.0f, 11.0f, 10.0f, 16.0f, 24.0f, 40.0f, 51.0f, 61.0f,      //
    12.0f, 12.0f, 14.0f, 19.0f, 26.0f, 58.0f, 60.0f, 55.0f,      //
    14.0f, 13.0f, 16.0f, 24.0f, 40.0f, 57.0f, 69.0f, 56.0f,      //
    14.0f, 17.0f, 22.0f, 29.0f, 51.0f, 87.0f, 80.0f, 62.0f,      //
    18.0f, 22.0f, 37.0f, 56.0f, 68.0f, 109.0f, 103.0f, 77.0f,    //
    24.0f, 35.0f, 55.0f, 64.0f, 81.0f, 104.0f, 113.0f, 92.0f,    //
    49.0f, 64.0f, 78.0f, 87.0f, 103.0f, 121.0f, 120.0f, 101.0f,  //
    72.0f, 92.0f, 95.0f, 98.0f, 112.0f, 100.0f, 103.0f, 99.0f,   //
    // c = 1
    17.0f, 18.0f, 24.0f, 47.0f, 99.0f, 99.0f, 99.0f, 99.0f,  //
    18.0f, 21.0f, 26.0f, 66.0f, 99.0f, 99.0f, 99.0f, 99.0f,  //
    24.0f, 26.0f, 56.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f,  //
    47.0f, 66.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f,  //
    99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f,  //
    99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f,  //
    99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f,  //
    99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f, 99.0f,  //
};

constexpr float kZeroBiasMulXYB[] = {0.5f, 0.5f, 0.5f};
constexpr float kZeroBiasMulYCbCr[] = {0.7f, 1.0f, 0.8f};

constexpr uint8_t kTransferFunctionPQ = 16;
constexpr uint8_t kTransferFunctionHLG = 18;

float DistanceToLinearQuality(float distance) {
  if (distance <= 0.1f) {
    return 1.0f;
  } else if (distance <= 4.6f) {
    return (200.0f / 9.0f) * (distance - 0.1f);
  } else if (distance <= 6.4f) {
    return 5000.0f / (100.0f - (distance - 0.1f) / 0.09f);
  } else if (distance < 25.0f) {
    return 530000.0f /
           (3450.0f -
            300.0f * std::sqrt((848.0f * distance - 5330.0f) / 120.0f));
  } else {
    return 5000.0f;
  }
}

float DistanceToScale(float distance, int k) {
  constexpr float kExponent[DCTSIZE2] = {
      1.00f, 0.51f, 0.67f, 0.74f, 1.00f, 1.00f, 1.00f, 1.00f,  //
      0.51f, 0.66f, 0.69f, 0.87f, 1.00f, 1.00f, 1.00f, 1.00f,  //
      0.67f, 0.69f, 0.84f, 0.83f, 0.96f, 1.00f, 1.00f, 1.00f,  //
      0.74f, 0.87f, 0.83f, 1.00f, 1.00f, 0.91f, 0.91f, 1.00f,  //
      1.00f, 1.00f, 0.96f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f,  //
      1.00f, 1.00f, 1.00f, 0.91f, 1.00f, 1.00f, 1.00f, 1.00f,  //
      1.00f, 1.00f, 1.00f, 0.91f, 1.00f, 1.00f, 1.00f, 1.00f,  //
      1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f,  //
  };
  constexpr float kDist0 = 1.5f;  // distance where non-linearity kicks in.
  if (distance < kDist0) {
    return distance;
  }
  const float exp = kExponent[k];
  const float mul = std::pow(kDist0, 1.0 - exp);
  return std::max<float>(0.5f * distance, mul * std::pow(distance, exp));
}

}  // namespace

void SetQuantMatrices(j_compress_ptr cinfo, float distance,
                      bool add_two_chroma_tables) {
  jpeg_comp_master* m = cinfo->master;
  const bool xyb = m->xyb_mode && cinfo->jpeg_color_space == JCS_RGB;

  // Global scale is chosen in a way that butteraugli 3-norm matches libjpeg
  // with the same quality setting. Fitted for quality 90 on jyrki31 corpus.
  constexpr float kGlobalScaleXYB = 1.44563150f;
  constexpr float kGlobalScaleYCbCr = 1.73480749f;

  float global_scale;
  bool non_linear_scaling = true;
  const float* base_quant_matrix[NUM_QUANT_TBLS];
  int num_base_tables;

  if (xyb) {
    global_scale = kGlobalScaleXYB;
    num_base_tables = 3;
    base_quant_matrix[0] = kBaseQuantMatrixXYB;
    base_quant_matrix[1] = kBaseQuantMatrixXYB + DCTSIZE2;
    base_quant_matrix[2] = kBaseQuantMatrixXYB + 2 * DCTSIZE2;
  } else if (cinfo->jpeg_color_space == JCS_YCbCr && !m->use_std_tables) {
    global_scale = kGlobalScaleYCbCr;
    if (m->cicp_transfer_function == kTransferFunctionPQ) {
      global_scale *= .4f;
    } else if (m->cicp_transfer_function == kTransferFunctionHLG) {
      global_scale *= .5f;
    }
    if (add_two_chroma_tables) {
      cinfo->comp_info[2].quant_tbl_no = 2;
      num_base_tables = 3;
      base_quant_matrix[0] = kBaseQuantMatrixYCbCr;
      base_quant_matrix[1] = kBaseQuantMatrixYCbCr + DCTSIZE2;
      base_quant_matrix[2] = kBaseQuantMatrixYCbCr + 2 * DCTSIZE2;
    } else {
      num_base_tables = 2;
      base_quant_matrix[0] = kBaseQuantMatrixYCbCr;
      // Use the Cr table for both Cb and Cr.
      base_quant_matrix[1] = kBaseQuantMatrixYCbCr + 2 * DCTSIZE2;
    }
  } else {
    global_scale = 0.01f * DistanceToLinearQuality(distance);
    non_linear_scaling = false;
    num_base_tables = 2;
    base_quant_matrix[0] = kBaseQuantMatrixStd;
    base_quant_matrix[1] = kBaseQuantMatrixStd + DCTSIZE2;
  }

  int quant_max = m->force_baseline ? 255 : 32767U;
  for (int quant_idx = 0; quant_idx < num_base_tables; ++quant_idx) {
    const float* base_qm = base_quant_matrix[quant_idx];
    JQUANT_TBL** qtable = &cinfo->quant_tbl_ptrs[quant_idx];
    if (*qtable == nullptr) {
      *qtable = jpegli_alloc_quant_table(reinterpret_cast<j_common_ptr>(cinfo));
    }
    for (int k = 0; k < DCTSIZE2; ++k) {
      float scale = global_scale;
      if (non_linear_scaling) {
        scale *= DistanceToScale(distance, k);
      }
      int qval = std::round(scale * base_qm[k]);
      (*qtable)->quantval[k] = std::max(1, std::min(qval, quant_max));
    }
    (*qtable)->sent_table = FALSE;
  }
}

void InitQuantizer(j_compress_ptr cinfo) {
  jpeg_comp_master* m = cinfo->master;
  const bool xyb = m->xyb_mode && cinfo->jpeg_color_space == JCS_RGB;
  // Compute quantization multupliers from the quant table values.
  for (int c = 0; c < cinfo->num_components; ++c) {
    int quant_idx = cinfo->comp_info[c].quant_tbl_no;
    JQUANT_TBL* quant_table = cinfo->quant_tbl_ptrs[quant_idx];
    if (!quant_table) {
      JPEGLI_ERROR("Missing quantization table %d for component %d", quant_idx,
                   c);
    }
    for (size_t k = 0; k < DCTSIZE2; k++) {
      int val = quant_table->quantval[k];
      if (val == 0) {
        JPEGLI_ERROR("Invalid quantval 0.");
      }
      m->quant_mul[c][k] = 8.0f / val;
    }
  }
  int y_channel = cinfo->jpeg_color_space == JCS_RGB ? 1 : 0;
  jpeg_component_info* y_comp = &cinfo->comp_info[y_channel];
  int y_quant_dc = cinfo->quant_tbl_ptrs[y_comp->quant_tbl_no]->quantval[0];
  for (int c = 0; c < cinfo->num_components; ++c) {
    if (c < 3 && y_quant_dc <= 2) {
      m->zero_bias_mul[c] = xyb ? kZeroBiasMulXYB[c] : kZeroBiasMulYCbCr[c];
    } else {
      m->zero_bias_mul[c] = 0.5f;
    }
  }
}

}  // namespace jpegli
