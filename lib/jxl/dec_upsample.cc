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

#include "lib/jxl/dec_upsample.h"

#include "lib/jxl/image_ops.h"

namespace jxl {
namespace {
void Upsample2x(const Image3F& src, Image3F* out, const float* weights) {
  // 0  1  2  3  4
  // 1  5  6  7  8
  // 2  6  9 10 11
  // 3  7 10 12 13
  // 4  8 11 13 14
  const float kernel[5][5] = {
      {weights[0], weights[1], weights[2], weights[3], weights[4]},
      {weights[1], weights[5], weights[6], weights[7], weights[8]},
      {weights[2], weights[6], weights[9], weights[10], weights[11]},
      {weights[3], weights[7], weights[10], weights[12], weights[13]},
      {weights[4], weights[8], weights[11], weights[13], weights[14]},
  };
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < out->ysize(); y++) {
      float* dst_row = out->PlaneRow(c, y);
      const float* src_rows[5];
      for (int iy = -2; iy <= 2; iy++) {
        src_rows[iy + 2] =
            src.PlaneRow(c, Mirror(static_cast<int>(y) / 2 + iy, src.ysize()));
      }
      for (size_t x = 0; x < out->xsize(); x++) {
        int src_x[5];
        for (int ix = -2; ix <= 2; ix++) {
          src_x[ix + 2] = Mirror(static_cast<int>(x) / 2 + ix, src.xsize());
        }
        float result = 0;
        float min = src_rows[0][src_x[0]];
        float max = src_rows[0][src_x[0]];
        for (size_t iy = 0; iy < 5; iy++) {
          for (size_t ix = 0; ix < 5; ix++) {
            float v = src_rows[iy][src_x[ix]];
            result += kernel[y % 2 ? 4 - iy : iy][x % 2 ? 4 - ix : ix] * v;
            min = std::min(v, min);
            max = std::max(v, max);
          }
        }
        // Avoid overshooting.
        dst_row[x] = std::min(std::max(result, min), max);
      }
    }
  }
}
void Upsample4x(const Image3F& src, Image3F* out, const float* weights) {
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
  const float kernel[2][2][5][5] = {
      {{{weights[0], weights[1], weights[2], weights[3], weights[4]},
        {weights[1], weights[10], weights[11], weights[12], weights[13]},
        {weights[2], weights[11], weights[19], weights[20], weights[21]},
        {weights[3], weights[12], weights[20], weights[27], weights[28]},
        {weights[4], weights[13], weights[21], weights[28], weights[34]}},

       {{weights[5], weights[6], weights[7], weights[8], weights[9]},
        {weights[14], weights[15], weights[16], weights[17], weights[18]},
        {weights[22], weights[23], weights[24], weights[25], weights[26]},
        {weights[29], weights[30], weights[31], weights[32], weights[33]},
        {weights[35], weights[36], weights[37], weights[38], weights[39]}}},

      {{{weights[5], weights[14], weights[22], weights[29], weights[35]},
        {weights[6], weights[15], weights[23], weights[30], weights[36]},
        {weights[7], weights[16], weights[24], weights[31], weights[37]},
        {weights[8], weights[17], weights[25], weights[32], weights[38]},
        {weights[9], weights[18], weights[26], weights[33], weights[39]}},

       {{weights[40], weights[41], weights[42], weights[43], weights[44]},
        {weights[41], weights[45], weights[46], weights[47], weights[48]},
        {weights[42], weights[46], weights[49], weights[50], weights[51]},
        {weights[43], weights[47], weights[50], weights[52], weights[53]},
        {weights[44], weights[48], weights[51], weights[53], weights[54]}}}};
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < out->ysize(); y++) {
      float* dst_row = out->PlaneRow(c, y);
      const float* src_rows[5];
      for (int iy = -2; iy <= 2; iy++) {
        src_rows[iy + 2] =
            src.PlaneRow(c, Mirror(static_cast<int>(y) / 4 + iy, src.ysize()));
      }
      for (size_t x = 0; x < out->xsize(); x++) {
        int src_x[5];
        for (int ix = -2; ix <= 2; ix++) {
          src_x[ix + 2] = Mirror(static_cast<int>(x) / 4 + ix, src.xsize());
        }
        float result = 0;
        float min = src_rows[0][src_x[0]];
        float max = src_rows[0][src_x[0]];
        for (size_t iy = 0; iy < 5; iy++) {
          for (size_t ix = 0; ix < 5; ix++) {
            float v = src_rows[iy][src_x[ix]];
            result += kernel[y % 4 < 2 ? y % 2 : 1 - y % 2]
                            [x % 4 < 2 ? x % 2 : 1 - x % 2]
                            [y % 4 < 2 ? iy : 4 - iy][x % 4 < 2 ? ix : 4 - ix] *
                      v;
            min = std::min(v, min);
            max = std::max(v, max);
          }
        }
        // Avoid overshooting.
        dst_row[x] = std::min(std::max(result, min), max);
      }
    }
  }
}
void Upsample8x(const Image3F& src, Image3F* out, const float* weights) {
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
  const float kernel[4][4][5][5] = {
      {{{weights[0], weights[1], weights[2], weights[3], weights[4]},
        {weights[1], weights[20], weights[21], weights[22], weights[23]},
        {weights[2], weights[21], weights[39], weights[40], weights[41]},
        {weights[3], weights[22], weights[40], weights[57], weights[58]},
        {weights[4], weights[23], weights[41], weights[58], weights[74]}},

       {{weights[5], weights[6], weights[7], weights[8], weights[9]},
        {weights[24], weights[25], weights[26], weights[27], weights[28]},
        {weights[42], weights[43], weights[44], weights[45], weights[46]},
        {weights[59], weights[60], weights[61], weights[62], weights[63]},
        {weights[75], weights[76], weights[77], weights[78], weights[79]}},

       {{weights[10], weights[11], weights[12], weights[13], weights[14]},
        {weights[29], weights[30], weights[31], weights[32], weights[33]},
        {weights[47], weights[48], weights[49], weights[50], weights[51]},
        {weights[64], weights[65], weights[66], weights[67], weights[68]},
        {weights[80], weights[81], weights[82], weights[83], weights[84]}},

       {{weights[15], weights[16], weights[17], weights[18], weights[19]},
        {weights[34], weights[35], weights[36], weights[37], weights[38]},
        {weights[52], weights[53], weights[54], weights[55], weights[56]},
        {weights[69], weights[70], weights[71], weights[72], weights[73]},
        {weights[85], weights[86], weights[87], weights[88], weights[89]}}},

      {{{weights[5], weights[24], weights[42], weights[59], weights[75]},
        {weights[6], weights[25], weights[43], weights[60], weights[76]},
        {weights[7], weights[26], weights[44], weights[61], weights[77]},
        {weights[8], weights[27], weights[45], weights[62], weights[78]},
        {weights[9], weights[28], weights[46], weights[63], weights[79]}},

       {{weights[90], weights[91], weights[92], weights[93], weights[94]},
        {weights[91], weights[105], weights[106], weights[107], weights[108]},
        {weights[92], weights[106], weights[119], weights[120], weights[121]},
        {weights[93], weights[107], weights[120], weights[132], weights[133]},
        {weights[94], weights[108], weights[121], weights[133], weights[144]}},

       {{weights[95], weights[96], weights[97], weights[98], weights[99]},
        {weights[109], weights[110], weights[111], weights[112], weights[113]},
        {weights[122], weights[123], weights[124], weights[125], weights[126]},
        {weights[134], weights[135], weights[136], weights[137], weights[138]},
        {weights[145], weights[146], weights[147], weights[148], weights[149]}},

       {{weights[100], weights[101], weights[102], weights[103], weights[104]},
        {weights[114], weights[115], weights[116], weights[117], weights[118]},
        {weights[127], weights[128], weights[129], weights[130], weights[131]},
        {weights[139], weights[140], weights[141], weights[142], weights[143]},
        {weights[150], weights[151], weights[152], weights[153],
         weights[154]}}},

      {{{weights[10], weights[29], weights[47], weights[64], weights[80]},
        {weights[11], weights[30], weights[48], weights[65], weights[81]},
        {weights[12], weights[31], weights[49], weights[66], weights[82]},
        {weights[13], weights[32], weights[50], weights[67], weights[83]},
        {weights[14], weights[33], weights[51], weights[68], weights[84]}},

       {{weights[95], weights[109], weights[122], weights[134], weights[145]},
        {weights[96], weights[110], weights[123], weights[135], weights[146]},
        {weights[97], weights[111], weights[124], weights[136], weights[147]},
        {weights[98], weights[112], weights[125], weights[137], weights[148]},
        {weights[99], weights[113], weights[126], weights[138], weights[149]}},

       {{weights[155], weights[156], weights[157], weights[158], weights[159]},
        {weights[156], weights[165], weights[166], weights[167], weights[168]},
        {weights[157], weights[166], weights[174], weights[175], weights[176]},
        {weights[158], weights[167], weights[175], weights[182], weights[183]},
        {weights[159], weights[168], weights[176], weights[183], weights[189]}},

       {{weights[160], weights[161], weights[162], weights[163], weights[164]},
        {weights[169], weights[170], weights[171], weights[172], weights[173]},
        {weights[177], weights[178], weights[179], weights[180], weights[181]},
        {weights[184], weights[185], weights[186], weights[187], weights[188]},
        {weights[190], weights[191], weights[192], weights[193],
         weights[194]}}},

      {{{weights[15], weights[34], weights[52], weights[69], weights[85]},
        {weights[16], weights[35], weights[53], weights[70], weights[86]},
        {weights[17], weights[36], weights[54], weights[71], weights[87]},
        {weights[18], weights[37], weights[55], weights[72], weights[88]},
        {weights[19], weights[38], weights[56], weights[73], weights[89]}},

       {{weights[100], weights[114], weights[127], weights[139], weights[150]},
        {weights[101], weights[115], weights[128], weights[140], weights[151]},
        {weights[102], weights[116], weights[129], weights[141], weights[152]},
        {weights[103], weights[117], weights[130], weights[142], weights[153]},
        {weights[104], weights[118], weights[131], weights[143], weights[154]}},

       {{weights[160], weights[169], weights[177], weights[184], weights[190]},
        {weights[161], weights[170], weights[178], weights[185], weights[191]},
        {weights[162], weights[171], weights[179], weights[186], weights[192]},
        {weights[163], weights[172], weights[180], weights[187], weights[193]},
        {weights[164], weights[173], weights[181], weights[188], weights[194]}},

       {{weights[195], weights[196], weights[197], weights[198], weights[199]},
        {weights[196], weights[200], weights[201], weights[202], weights[203]},
        {weights[197], weights[201], weights[204], weights[205], weights[206]},
        {weights[198], weights[202], weights[205], weights[207], weights[208]},
        {weights[199], weights[203], weights[206], weights[208],
         weights[209]}}}};
  for (size_t c = 0; c < 3; c++) {
    for (size_t y = 0; y < out->ysize(); y++) {
      float* dst_row = out->PlaneRow(c, y);
      const float* src_rows[5];
      for (int iy = -2; iy <= 2; iy++) {
        src_rows[iy + 2] =
            src.PlaneRow(c, Mirror(static_cast<int>(y) / 8 + iy, src.ysize()));
      }
      for (size_t x = 0; x < out->xsize(); x++) {
        int src_x[5];
        for (int ix = -2; ix <= 2; ix++) {
          src_x[ix + 2] = Mirror(static_cast<int>(x) / 8 + ix, src.xsize());
        }
        float result = 0;
        float min = src_rows[0][src_x[0]];
        float max = src_rows[0][src_x[0]];
        for (size_t iy = 0; iy < 5; iy++) {
          for (size_t ix = 0; ix < 5; ix++) {
            float v = src_rows[iy][src_x[ix]];
            result += kernel[y % 8 < 4 ? y % 4 : 3 - y % 4]
                            [x % 8 < 4 ? x % 4 : 3 - x % 4]
                            [y % 8 < 4 ? iy : 4 - iy][x % 8 < 4 ? ix : 4 - ix] *
                      v;
            min = std::min(v, min);
            max = std::max(v, max);
          }
        }
        // Avoid overshooting.
        dst_row[x] = std::min(std::max(result, min), max);
      }
    }
  }
}
}  // namespace

void Upsample(Image3F* src, size_t upsampling,
              const CustomTransformData& data) {
  if (upsampling == 1) return;
  Image3F out(src->xsize() * upsampling, src->ysize() * upsampling);
  if (upsampling == 2) {
    Upsample2x(*src, &out, data.upsampling2_weights);
  } else if (upsampling == 4) {
    Upsample4x(*src, &out, data.upsampling4_weights);
  } else if (upsampling == 8) {
    Upsample8x(*src, &out, data.upsampling8_weights);
  } else {
    JXL_ABORT("Not implemented");
  }
  *src = std::move(out);
}

}  // namespace jxl
