// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "lib/jxl/dec_16bit_reconstruct.h"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "lib/jxl/dec_16bit_reconstruct.cc"
#include <hwy/foreach_target.h>
#include <hwy/highway.h>

#include "lib/jxl/dec_xyb-inl.h"

HWY_BEFORE_NAMESPACE();
namespace jxl {
namespace HWY_NAMESPACE {
void FinalizeImageRect16(Image3F* input_image, const Rect& input_rect,
                         ImageF* alpha, const Rect& alpha_rect,
                         PassesDecoderState* dec_state, size_t thread,
                         const Rect& frame_rect) {
  // This function is very NEON-specific. As such, it uses intrinsics directly.
#if HWY_TARGET == HWY_NEON
  // WARNING: doing fixed point arithmetic correctly is very complicated.
  // Changes to this function should be thoroughly tested.
  JXL_ASSERT(SameSize(input_rect, frame_rect));
  if (alpha) JXL_ASSERT(SameSize(alpha_rect, input_rect));

  const FrameDimensions& frame_dim = dec_state->shared->frame_dim;

  ImageS& storage = dec_state->fixpoint_buffer[thread];

  size_t x0 = frame_rect.x0();
  size_t x1 = std::min(x0 + frame_rect.xsize(), frame_dim.xsize_upsampled);
  size_t xs = x1 - x0;

  int16_t* xyba[4] = {};
  xyba[0] = storage.Row(9);
  xyba[1] = storage.Row(10);
  xyba[2] = storage.Row(11);
  if (alpha) xyba[3] = storage.Row(12);

  for (size_t y = 0; y < frame_rect.ysize() + 2 &&
                     y + frame_rect.y0() < frame_dim.ysize_upsampled + 2;
       y++) {
    // Mirroring
    size_t yfrom =
        y + frame_rect.y0() == 0
            ? 0
            : (y + frame_rect.y0() == frame_dim.ysize_upsampled + 1 ? y - 2
                                                                    : y - 1);

    ConvertToFixpoint<18>(input_rect.ConstPlaneRow(*input_image, 0, yfrom) - 1,
                          storage.Row(0 + (y % 3)), xs + 2);
    ConvertToFixpoint<15>(input_rect.ConstPlaneRow(*input_image, 1, yfrom) - 1,
                          storage.Row(3 + (y % 3)), xs + 2);
    ConvertToFixpoint<15>(input_rect.ConstPlaneRow(*input_image, 2, yfrom) - 1,
                          storage.Row(6 + (y % 3)), xs + 2);

    // x mirroring
    if (x0 == 0) {
      for (size_t c : {0, 1, 2}) {
        storage.Row(c * 3 + (y % 3))[0] = storage.Row(c * 3 + (y % 3))[1];
      }
    }
    if (x1 == frame_dim.xsize_upsampled && frame_dim.xsize_upsampled % 8 == 0) {
      for (size_t c : {0, 1, 2}) {
        storage.Row(c * 3 + (y % 3))[xs + 1] = storage.Row(c * 3 + (y % 3))[xs];
      }
    }

    if (y < 2) continue;

    // Apply Gaborish
    for (size_t c : {0, 1, 2}) {
      int16_t* row_top = storage.Row(c * 3 + ((y - 2) % 3)) + 1;
      int16_t* row_mid = storage.Row(c * 3 + ((y - 1) % 3)) + 1;
      int16_t* row_bot = storage.Row(c * 3 + ((y - 0) % 3)) + 1;
      for (size_t x = 0; x < xs; x += 8) {
        int16x8_t p01 = vld1q_s16(row_top + x);
        int16x8_t p00 = vld1q_s16(row_top + x - 1);
        int16x8_t p02 = vld1q_s16(row_top + x + 1);
        int16x8_t p11 = vld1q_s16(row_mid + x);
        int16x8_t p10 = vld1q_s16(row_mid + x - 1);
        int16x8_t p12 = vld1q_s16(row_mid + x + 1);
        int16x8_t p21 = vld1q_s16(row_bot + x);
        int16x8_t p20 = vld1q_s16(row_bot + x - 1);
        int16x8_t p22 = vld1q_s16(row_bot + x + 1);

        int16x8_t sum0 = vqrdmulhq_n_s16(p11, 19211);  // 0.586279
        int16x8_t sum1pre =
            vhaddq_s16(vhaddq_s16(p01, p21), vhaddq_s16(p10, p12));
        int16x8_t sum1 = vqrdmulhq_n_s16(sum1pre, 8850);  // 0.067521 * 4
        int16x8_t sum2pre =
            vhaddq_s16(vhaddq_s16(p00, p22), vhaddq_s16(p02, p20));
        int16x8_t sum2 = vqrdmulhq_n_s16(sum2pre, 4707);  // 0.035909 * 4

        int16x8_t out = vaddq_s16(sum0, vaddq_s16(sum1, sum2));
        vst1q_s16(xyba[c] + x, out);
      }
    }

    if (alpha)
      ConvertToFixpoint<8>(alpha_rect.ConstRow(*alpha, y), xyba[3], xs);
    uint8_t* out = dec_state->rgb_output +
                   dec_state->rgb_stride * (frame_rect.y0() + y - 2) +
                   x0 * (dec_state->rgb_output_is_rgba ? 4 : 3);
    FastXYBTosRGB8Impl(const_cast<const int16_t**>(xyba), out, xs, !!alpha);
  }
#else
  JXL_ABORT("Unreachable");
#endif
}
}  // namespace HWY_NAMESPACE
}  // namespace jxl
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace jxl {
HWY_EXPORT(FinalizeImageRect16);
void FinalizeImageRect16(Image3F* input_image, const Rect& input_rect,
                         ImageF* alpha, const Rect& alpha_rect,
                         PassesDecoderState* dec_state, size_t thread,
                         const Rect& frame_rect) {
  HWY_DYNAMIC_DISPATCH(FinalizeImageRect16)
  (input_image, input_rect, alpha, alpha_rect, dec_state, thread, frame_rect);
}
}  // namespace jxl
#endif
